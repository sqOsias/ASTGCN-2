import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Union

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.lstm_model import LSTMPredictor
from lib.config import load_config
from model.astgcn import ASTGCN
from model.model_config import get_backbones_from_config
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


def _resolve_device(device_arg: str, config_ctx: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'config':
        if config_ctx.startswith('gpu'):
            gpu_index = int(config_ctx[config_ctx.index('-') + 1:])
            if torch.cuda.is_available():
                return torch.device(f'cuda:{gpu_index}')
            return torch.device('cpu')
        if config_ctx.startswith('cpu'):
            return torch.device('cpu')
        return torch.device(config_ctx)
    return torch.device(device_arg)


def _infer_num_features(data_path: str, add_time_features: bool = True) -> int:
    data = np.load(data_path, mmap_mode='r')['data']
    base_features = int(data.shape[2])
    return base_features + 2 if add_time_features else base_features


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_forward(model: torch.nn.Module, inputs: TensorOrList):
    if isinstance(inputs, list):
        return model(inputs)
    return model(inputs)


def _profile_flops(model: torch.nn.Module, inputs: TensorOrList, device: torch.device) -> float:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                profile_memory=False,
                record_shapes=False,
            ) as prof:
                _run_forward(model, inputs)
        total_flops = 0.0
        for item in prof.key_averages():
            if getattr(item, 'flops', None) is not None:
                total_flops += float(item.flops)
        return total_flops
    except Exception:
        return float('nan')


def _estimate_lstm_flops(model: LSTMPredictor, inputs: torch.Tensor) -> float:
    batch_size, seq_len, input_size = inputs.shape
    hidden_size = int(model.lstm.hidden_size)
    num_layers = int(model.lstm.num_layers)
    output_size = int(model.fc.out_features)

    total_flops = 0.0
    current_input = input_size
    for _ in range(num_layers):
        # 每层每步 FLOPs 近似：
        # 4个门，每个门包含 input->hidden 与 hidden->hidden 的乘加（*2）
        per_step = 8.0 * (current_input * hidden_size + hidden_size * hidden_size)
        total_flops += batch_size * seq_len * per_step
        current_input = hidden_size

    # 最后全连接层 FLOPs（乘加）
    total_flops += batch_size * 2.0 * hidden_size * output_size
    return total_flops


def _measure_latency_ms(model: torch.nn.Module, inputs: TensorOrList, device: torch.device,
                        warmup: int, runs: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _run_forward(model, inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(runs):
            _run_forward(model, inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end = time.perf_counter()

    return (end - start) * 1000.0 / float(runs)


def _prepare_inputs(cfg, batch_size: int, num_features: int, device: torch.device):
    num_nodes = cfg.data.num_of_vertices
    pph = cfg.data.points_per_hour
    horizon = cfg.data.num_for_predict

    def _time_len(multiplier: int) -> int:
        return max(1, multiplier * pph)

    week_input = torch.randn(batch_size, num_nodes, num_features, _time_len(cfg.training.num_of_weeks), device=device)
    day_input = torch.randn(batch_size, num_nodes, num_features, _time_len(cfg.training.num_of_days), device=device)
    recent_input = torch.randn(batch_size, num_nodes, num_features, _time_len(cfg.training.num_of_hours), device=device)

    lstm_seq_len = _time_len(cfg.training.num_of_hours)
    lstm_input = torch.randn(batch_size, lstm_seq_len, 1, device=device)

    return [week_input, day_input, recent_input], lstm_input, horizon


def _human_flops(flops: float) -> str:
    if not np.isfinite(flops):
        return 'N/A'
    if flops >= 1e9:
        return f'{flops / 1e9:.3f} G'
    return f'{flops / 1e6:.3f} M'


def _profile_one(name: str, model: torch.nn.Module, inputs: TensorOrList, device: torch.device,
                 warmup: int, runs: int) -> Dict[str, Union[str, float]]:
    model.eval()

    with torch.no_grad():
        _run_forward(model, inputs)

    params = _count_params(model)
    flops = _profile_flops(model, inputs, device)
    if isinstance(model, LSTMPredictor) and isinstance(inputs, torch.Tensor):
        flops = max(flops, _estimate_lstm_flops(model, inputs)) if np.isfinite(flops) else _estimate_lstm_flops(model, inputs)
    latency_ms = _measure_latency_ms(model, inputs, device, warmup=warmup, runs=runs)

    return {
        'Model': name,
        'Params': params,
        'Params_M': params / 1e6,
        'FLOPs': flops,
        'FLOPs_G': flops / 1e9 if np.isfinite(flops) else float('nan'),
        'Latency_ms': latency_ms,
    }


def _profile_repeated(name: str, model: torch.nn.Module, make_inputs, device: torch.device,
                      warmup: int, runs: int, repeats: int, base_seed: int) -> Dict[str, Union[str, float]]:
    records = []
    for repeat_idx in range(repeats):
        seed = base_seed + repeat_idx
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = make_inputs()
        rec = _profile_one(name, model, inputs, device, warmup=warmup, runs=runs)
        records.append(rec)

    params = int(records[0]['Params'])
    params_m = float(records[0]['Params_M'])
    flops_values = np.array([float(r['FLOPs']) for r in records], dtype=np.float64)
    latency_values = np.array([float(r['Latency_ms']) for r in records], dtype=np.float64)

    finite_flops = flops_values[np.isfinite(flops_values)]
    flops_mean = float(np.mean(finite_flops)) if finite_flops.size > 0 else float('nan')
    flops_std = float(np.std(finite_flops)) if finite_flops.size > 0 else float('nan')

    latency_mean = float(np.mean(latency_values))
    latency_std = float(np.std(latency_values))

    return {
        'Model': name,
        'Params': params,
        'Params_M': params_m,
        'FLOPs': flops_mean,
        'FLOPs_G': flops_mean / 1e9 if np.isfinite(flops_mean) else float('nan'),
        'FLOPs_std': flops_std,
        'FLOPs_G_std': flops_std / 1e9 if np.isfinite(flops_std) else float('nan'),
        'Latency_ms': latency_mean,
        'Latency_ms_std': latency_std,
        'Repeats': repeats,
    }


def _save_csv(rows: List[Dict[str, Union[str, float]]], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    fields = [
        'Model', 'Params', 'Params_M',
        'FLOPs', 'FLOPs_G', 'FLOPs_std', 'FLOPs_G_std',
        'Latency_ms', 'Latency_ms_std', 'Repeats'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_table(rows: List[Dict[str, Union[str, float]]], device: torch.device, batch_size: int):
    print('\n=== Model Complexity Comparison ===')
    print(f'Device: {device} | Batch Size: {batch_size}')
    print(f"{'Model':<21} {'Params (M)':>12} {'FLOPs (mean±std)':>22} {'Latency ms (mean±std)':>24}")
    print('-' * 88)
    for row in rows:
        flops_mean = float(row['FLOPs'])
        flops_std = float(row.get('FLOPs_std', 0.0))
        latency_mean = float(row['Latency_ms'])
        latency_std = float(row.get('Latency_ms_std', 0.0))

        if np.isfinite(flops_mean) and np.isfinite(flops_std):
            if abs(flops_mean) >= 1e9:
                flops_text = f"{flops_mean / 1e9:.3f}±{flops_std / 1e9:.3f} G"
            else:
                flops_text = f"{flops_mean / 1e6:.3f}±{flops_std / 1e6:.3f} M"
        else:
            flops_text = 'N/A'

        print(
            f"{row['Model']:<21} "
            f"{row['Params_M']:>12.3f} "
            f"{flops_text:>22} "
            f"{latency_mean:>9.3f}±{latency_std:<9.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description='Compare Params/FLOPs/Latency for LSTM, ASTGCN, AST-Informer')
    parser.add_argument('--config', type=str, default='configurations/PEMS04.conf')
    parser.add_argument('--device', type=str, default='auto',
                        help='auto | config | cpu | cuda:0 ...')
    parser.add_argument('--batch_size', type=int, default=1, help='Profiling batch size')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup runs for latency')
    parser.add_argument('--runs', type=int, default=100, help='Measured runs for latency')
    parser.add_argument('--repeats', type=int, default=5, help='Repeat profiling N times and report mean/std')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for repeated profiling')
    parser.add_argument('--lstm_hidden', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--ours_spatial_mode', type=int, default=1)
    parser.add_argument('--ours_temporal_mode', type=int, default=1)
    parser.add_argument('--output_csv', type=str, default='results/model_complexity_comparison.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(args.device, cfg.training.ctx)

    num_features = _infer_num_features(cfg.data.graph_signal_matrix_filename, add_time_features=True)
    ast_inputs, lstm_input, horizon = _prepare_inputs(cfg, args.batch_size, num_features, device)

    all_backbones = get_backbones_from_config(cfg.data, cfg.training, device)
    ag_dict = asdict(cfg.upgrade.adaptive_graph)
    tf_dict = asdict(cfg.upgrade.transformer)

    lstm = LSTMPredictor(
        input_size=1,
        hidden_size=args.lstm_hidden,
        output_size=horizon,
        num_layers=args.lstm_layers,
        dropout=0.0,
    ).to(device)

    astgcn_base = ASTGCN(cfg.data.num_for_predict, all_backbones).to(device)

    ast_informer = UpgradeASTGCN(
        num_of_features=num_features,
        num_for_prediction=cfg.data.num_for_predict,
        all_backbones=all_backbones,
        num_of_vertices=cfg.data.num_of_vertices,
        spatial_mode=args.ours_spatial_mode,
        temporal_mode=args.ours_temporal_mode,
        adaptive_graph_cfg=ag_dict,
        transformer_cfg=tf_dict,
    ).to(device)

    def _make_ast_inputs():
        ast_inputs_local, _, _ = _prepare_inputs(cfg, args.batch_size, num_features, device)
        return ast_inputs_local

    def _make_lstm_input():
        _, lstm_input_local, _ = _prepare_inputs(cfg, args.batch_size, num_features, device)
        return lstm_input_local

    rows = [
        _profile_repeated('LSTM', lstm, _make_lstm_input, device, args.warmup, args.runs, args.repeats, args.seed),
        _profile_repeated('ASTGCN (Base-ST)', astgcn_base, _make_ast_inputs, device, args.warmup, args.runs, args.repeats, args.seed),
        _profile_repeated('AST-Informer (Ours)', ast_informer, _make_ast_inputs, device, args.warmup, args.runs, args.repeats, args.seed),
    ]

    _save_csv(rows, args.output_csv)
    _print_table(rows, device, args.batch_size)
    print(f'\nSaved CSV -> {args.output_csv}')


if __name__ == '__main__':
    main()
