import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import load_config
from lib.data_preparation import read_and_generate_dataset
from lib.metrics import mean_absolute_error, mean_squared_error, masked_mape_np
from model.astgcn import ASTGCN
from model.model_config import get_backbones_from_config
from model.mstgcn import MSTGCN
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


def _resolve_device(device_arg: str, config_ctx: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device_arg == 'config':
        if config_ctx.startswith('gpu'):
            gpu_index = int(config_ctx[config_ctx.index('-') + 1:])
            return torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
        if config_ctx.startswith('cpu'):
            return torch.device('cpu')
        return torch.device(config_ctx)
    return torch.device(device_arg)


def _find_runs_with_best_checkpoint(results_dir: str) -> List[str]:
    run_dirs = []
    for root, _, files in os.walk(results_dir):
        if 'best_model.pth' in files and os.path.basename(root) == 'checkpoints':
            run_dirs.append(os.path.dirname(root))
    run_dirs.sort()
    return run_dirs


def _load_run_modes(run_dir: str) -> Tuple[int, int]:
    resolved_path = os.path.join(run_dir, 'configs', 'resolved_config.json')
    if os.path.exists(resolved_path):
        with open(resolved_path, 'r', encoding='utf-8') as f:
            resolved = json.load(f)
        model_upgrade = resolved.get('ModelUpgrade', {})
        spatial_mode = int(model_upgrade.get('spatial_mode', 0))
        temporal_mode = int(model_upgrade.get('temporal_mode', 0))
        return spatial_mode, temporal_mode

    run_id = os.path.basename(run_dir)
    parts = run_id.split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return 0, 0


def _resolve_run_model_name(run_dir: str, training_model_name: str) -> str:
    runtime_path = os.path.join(run_dir, 'artifacts', 'runtime.json')
    if os.path.exists(runtime_path):
        with open(runtime_path, 'r', encoding='utf-8') as f:
            rt = json.load(f)
        runtime_model = str(rt.get('model_name', '')).strip()
        if runtime_model:
            return runtime_model

    if ',' in training_model_name:
        return training_model_name.split(',')[0].strip()
    return training_model_name.strip()


def _prepare_test_data(cfg, batch_size: int):
    all_data = read_and_generate_dataset(
        cfg.data.graph_signal_matrix_filename,
        cfg.training.num_of_weeks,
        cfg.training.num_of_days,
        cfg.training.num_of_hours,
        cfg.data.num_for_predict,
        cfg.data.points_per_hour,
        cfg.training.merge,
    )

    test_split = all_data['test']
    test_week = torch.from_numpy(test_split['week']).float()
    test_day = torch.from_numpy(test_split['day']).float()
    test_recent = torch.from_numpy(test_split['recent']).float()
    test_target = torch.from_numpy(test_split['target']).float()

    test_loader = DataLoader(
        TensorDataset(test_week, test_day, test_recent, test_target),
        batch_size=batch_size,
        shuffle=False,
    )

    target_np = test_split['target']
    true_value = target_np.transpose((0, 2, 1)).reshape(target_np.shape[0], -1)
    num_features = int(test_recent.shape[-2])

    return test_loader, true_value, num_features


def _build_model(cfg, model_name: str, all_backbones, num_features: int,
                 spatial_mode: int, temporal_mode: int, device: torch.device):
    if model_name == 'MSTGCN' and (spatial_mode != 0 or temporal_mode != 0):
        raise ValueError('Upgrade modes only supported for ASTGCN')

    if spatial_mode == 0 and temporal_mode == 0:
        if model_name == 'MSTGCN':
            net = MSTGCN(cfg.data.num_for_predict, all_backbones).to(device)
        elif model_name == 'ASTGCN':
            net = ASTGCN(cfg.data.num_for_predict, all_backbones).to(device)
        else:
            raise ValueError(f'Unsupported model_name: {model_name}')
    else:
        net = UpgradeASTGCN(
            num_of_features=num_features,
            num_for_prediction=cfg.data.num_for_predict,
            all_backbones=all_backbones,
            num_of_vertices=cfg.data.num_of_vertices,
            spatial_mode=spatial_mode,
            temporal_mode=temporal_mode,
            adaptive_graph_cfg=asdict(cfg.upgrade.adaptive_graph),
            transformer_cfg=asdict(cfg.upgrade.transformer),
        ).to(device)

    return net


def _predict(net: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> np.ndarray:
    net.eval()
    preds = []
    with torch.no_grad():
        for week, day, recent, _ in test_loader:
            week = week.to(device)
            day = day.to(device)
            recent = recent.to(device)
            out = net([week, day, recent])
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _init_lazy_params_with_one_batch(net: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    net.eval()
    with torch.no_grad():
        for week, day, recent, _ in test_loader:
            week = week.to(device)
            day = day.to(device)
            recent = recent.to(device)
            net([week, day, recent])
            break


def _compute_metrics(prediction: np.ndarray, true_value: np.ndarray,
                     num_of_vertices: int, horizons: List[int]) -> Dict[int, Dict[str, float]]:
    prediction_flat = prediction.transpose((0, 2, 1)).reshape(prediction.shape[0], -1)

    metrics = {}
    for h in horizons:
        upper = h * num_of_vertices
        mae = mean_absolute_error(true_value[:, :upper], prediction_flat[:, :upper])
        rmse = mean_squared_error(true_value[:, :upper], prediction_flat[:, :upper]) ** 0.5
        mape = masked_mape_np(true_value[:, :upper], prediction_flat[:, :upper], 0)
        metrics[h] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
        }
    return metrics


def _write_metrics_csv(run_dir: str, rows: List[Dict]):
    output_path = os.path.join(run_dir, 'metrics', 'test_metrics_recomputed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'horizon', 'MAE', 'RMSE', 'MAPE'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(rows: List[Dict], output_csv: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_csv), exist_ok=True) if os.path.dirname(output_csv) else None
    fields = [
        'run_group', 'run_id', 'model_name', 'spatial_mode', 'temporal_mode',
        'best_epoch', 'best_val_loss', 'checkpoint_path', 'recomputed_metrics_path',
        'horizon', 'MAE', 'RMSE', 'MAPE', 'status', 'error'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_all(results_dir: str, device_arg: str, batch_size: int,
                 output_csv: str, max_runs: int = 0):
    run_dirs = _find_runs_with_best_checkpoint(results_dir)
    if not run_dirs:
        print(f'[WARN] No best_model.pth found under: {results_dir}')
        return

    if max_runs > 0:
        run_dirs = run_dirs[:max_runs]

    print(f'[INFO] Found {len(run_dirs)} runs with best checkpoint.')

    data_cache = {}
    summary_rows = []

    for idx, run_dir in enumerate(run_dirs, 1):
        run_id = os.path.basename(run_dir)
        run_group = os.path.basename(os.path.dirname(run_dir))
        config_path = os.path.join(run_dir, 'configs', 'train.conf')
        ckpt_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')

        print(f'\n[{idx}/{len(run_dirs)}] Evaluating {run_group}/{run_id}')

        base_info = {
            'run_group': run_group,
            'run_id': run_id,
            'checkpoint_path': ckpt_path,
            'recomputed_metrics_path': os.path.join(run_dir, 'metrics', 'test_metrics_recomputed.csv'),
        }

        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f'Missing config: {config_path}')

            cfg = load_config(config_path)
            spatial_mode, temporal_mode = _load_run_modes(run_dir)
            model_name = _resolve_run_model_name(run_dir, cfg.training.model_name)
            device = _resolve_device(device_arg, cfg.training.ctx)

            cache_key = (
                cfg.data.graph_signal_matrix_filename,
                cfg.training.num_of_weeks,
                cfg.training.num_of_days,
                cfg.training.num_of_hours,
                cfg.data.num_for_predict,
                cfg.data.points_per_hour,
                cfg.training.merge,
                batch_size,
            )
            if cache_key not in data_cache:
                data_cache[cache_key] = _prepare_test_data(cfg, batch_size)
            test_loader, true_value, num_features = data_cache[cache_key]

            all_backbones = get_backbones_from_config(cfg.data, cfg.training, device)
            net = _build_model(
                cfg, model_name, all_backbones, num_features,
                spatial_mode, temporal_mode, device,
            )

            _init_lazy_params_with_one_batch(net, test_loader, device)
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(state_dict)

            pph = cfg.data.points_per_hour
            horizons = [h for h in [3, pph, pph * 3] if h <= cfg.data.num_for_predict]
            prediction = _predict(net, test_loader, device)
            metrics = _compute_metrics(prediction, true_value, cfg.data.num_of_vertices, horizons)

            val_path = os.path.join(run_dir, 'metrics', 'val_metrics.csv')
            best_epoch, best_val_loss = -1, float('nan')
            if os.path.exists(val_path):
                with open(val_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = [r for r in reader if int(r['epoch']) > 0]
                if rows:
                    best_row = min(rows, key=lambda r: float(r['validation_loss']))
                    best_epoch = int(best_row['epoch'])
                    best_val_loss = float(best_row['validation_loss'])

            metric_rows = []
            for h in horizons:
                metric_rows.append({
                    'epoch': best_epoch,
                    'horizon': h,
                    'MAE': metrics[h]['MAE'],
                    'RMSE': metrics[h]['RMSE'],
                    'MAPE': metrics[h]['MAPE'],
                })
            _write_metrics_csv(run_dir, metric_rows)

            for h in horizons:
                summary_rows.append({
                    **base_info,
                    'model_name': model_name,
                    'spatial_mode': spatial_mode,
                    'temporal_mode': temporal_mode,
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss,
                    'horizon': h,
                    'MAE': metrics[h]['MAE'],
                    'RMSE': metrics[h]['RMSE'],
                    'MAPE': metrics[h]['MAPE'],
                    'status': 'ok',
                    'error': '',
                })

            print(f'[OK] horizons={horizons}, metrics saved.')

        except Exception as exc:
            summary_rows.append({
                **base_info,
                'model_name': '',
                'spatial_mode': '',
                'temporal_mode': '',
                'best_epoch': '',
                'best_val_loss': '',
                'horizon': '',
                'MAE': '',
                'RMSE': '',
                'MAPE': '',
                'status': 'failed',
                'error': str(exc),
            })
            print(f'[FAILED] {exc}')

    _write_summary_csv(summary_rows, output_csv)
    print(f'\n[INFO] Summary saved to: {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all best_model.pth checkpoints under results directory')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto', help='auto | config | cpu | cuda:0 ...')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_runs', type=int, default=0, help='0 means evaluate all runs')
    parser.add_argument('--output_csv', type=str, default='results/recomputed_best_checkpoint_metrics.csv')
    args = parser.parse_args()

    evaluate_all(
        results_dir=args.results_dir,
        device_arg=args.device,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
        max_runs=args.max_runs,
    )
