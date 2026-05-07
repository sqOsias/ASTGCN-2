"""Benchmark one-epoch training time for ASTGCN vs UpgradeASTGCN on PEMS04/PEMS08.

用法：
    python scripts/benchmark_epoch_time.py \
        --configs configurations/PEMS04.conf configurations/PEMS08.conf \
        --variants base upgrade \
        --warmup_epochs 1 \
        --measure_epochs 2 \
        --output benchmark_epoch_time.md
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import load_config
from lib.data_preparation import read_and_generate_dataset
from model.astgcn import ASTGCN
from model.model_config import get_backbones_from_config
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


def _resolve_device(ctx: str) -> torch.device:
    if ctx.startswith('gpu'):
        idx = int(ctx[ctx.index('-') + 1:])
        return torch.device(f'cuda:{idx}' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')


def _prepare_loader(cfg, batch_size: int):
    all_data = read_and_generate_dataset(
        cfg.data.graph_signal_matrix_filename,
        cfg.training.num_of_weeks,
        cfg.training.num_of_days,
        cfg.training.num_of_hours,
        cfg.data.num_for_predict,
        cfg.data.points_per_hour,
        cfg.training.merge,
    )
    tr = all_data['train']
    num_features = int(tr['recent'].shape[-2])
    dataset = TensorDataset(
        torch.from_numpy(tr['week']).float(),
        torch.from_numpy(tr['day']).float(),
        torch.from_numpy(tr['recent']).float(),
        torch.from_numpy(tr['target']).float(),
    )
    del all_data
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, num_features, len(dataset)


def _build_net(variant: str, cfg, num_features: int, device: torch.device):
    backbones = get_backbones_from_config(cfg.data, cfg.training, device)
    if variant == 'base':
        net = ASTGCN(cfg.data.num_for_predict, backbones).to(device)
    elif variant == 'upgrade':
        net = UpgradeASTGCN(
            num_of_features=num_features,
            num_for_prediction=cfg.data.num_for_predict,
            all_backbones=backbones,
            num_of_vertices=cfg.data.num_of_vertices,
            spatial_mode=1,
            temporal_mode=1,
            adaptive_graph_cfg=asdict(cfg.upgrade.adaptive_graph),
            transformer_cfg=asdict(cfg.upgrade.transformer),
        ).to(device)
    else:
        raise ValueError(variant)
    return net


def _run_one_epoch(net, loader, optimizer, loss_fn, device):
    net.train()
    t0 = time.time()
    n_batches = 0
    for w, d, r, t in loader:
        w = w.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        out = net([w, d, r])
        loss = loss_fn(out, t).mean() * 0.5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_batches += 1
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.time() - t0, n_batches


def benchmark(config_path: str, variant: str, warmup_epochs: int,
              measure_epochs: int, batch_size_override: int = 0):
    cfg = load_config(config_path)
    device = _resolve_device(cfg.training.ctx)
    bs = batch_size_override or cfg.training.batch_size

    loader, num_features, n_samples = _prepare_loader(cfg, bs)

    net = _build_net(variant, cfg, num_features, device)
    # 触发 lazy 参数
    for w, d, r, _ in loader:
        w = w.to(device); d = d.to(device); r = r.to(device)
        with torch.no_grad():
            net([w, d, r])
        break

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none')

    # warmup
    for _ in range(warmup_epochs):
        _run_one_epoch(net, loader, optimizer, loss_fn, device)

    # measure
    times = []
    for _ in range(measure_epochs):
        dt, n_batches = _run_one_epoch(net, loader, optimizer, loss_fn, device)
        times.append(dt)

    times = np.array(times)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    del net, loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'config': config_path,
        'dataset': os.path.basename(config_path).replace('.conf', ''),
        'variant': variant,
        'batch_size': bs,
        'n_samples': n_samples,
        'n_batches': n_batches,
        'n_params': n_params,
        'mean_sec': float(times.mean()),
        'std_sec': float(times.std()),
        'min_sec': float(times.min()),
        'max_sec': float(times.max()),
        'device': str(device),
    }


def _fmt_params(n: int) -> str:
    if n >= 1e6:
        return f'{n/1e6:.2f} M'
    if n >= 1e3:
        return f'{n/1e3:.2f} K'
    return str(n)


def write_markdown(rows: List[dict], output: str, env_info: dict):
    lines = []
    lines.append('# ASTGCN vs UpgradeASTGCN 单 epoch 训练时间基准')
    lines.append('')
    lines.append(f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'- 设备: `{env_info["device_name"]}`')
    lines.append(f'- PyTorch: `{env_info["torch_version"]}` | CUDA: `{env_info["cuda_version"]}`')
    lines.append('')
    lines.append('## 测试说明')
    lines.append('')
    lines.append('- `variant=base` 对应 `ASTGCN`（spatial_mode=0, temporal_mode=0）')
    lines.append('- `variant=upgrade` 对应 `UpgradeASTGCN`（spatial_mode=1, temporal_mode=1）')
    lines.append('- `batch_size` 取自各数据集配置文件')
    lines.append('- 先做 warmup epoch，再测量后续若干 epoch，报告均值/标准差/最小/最大')
    lines.append('')
    lines.append('## 结果')
    lines.append('')
    lines.append('| 数据集 | 模型 | Params | Batch Size | Samples | Batches/epoch | 平均 (s) | 标准差 (s) | 最小 (s) | 最大 (s) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        lines.append(
            f'| {r["dataset"]} | {"ASTGCN" if r["variant"]=="base" else "UpgradeASTGCN"} '
            f'| {_fmt_params(r["n_params"])} | {r["batch_size"]} | {r["n_samples"]} '
            f'| {r["n_batches"]} | {r["mean_sec"]:.2f} | {r["std_sec"]:.2f} '
            f'| {r["min_sec"]:.2f} | {r["max_sec"]:.2f} |'
        )
    lines.append('')
    lines.append('## 相对加速/放缓')
    lines.append('')
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r['dataset'], {})[r['variant']] = r['mean_sec']
    lines.append('| 数据集 | ASTGCN (s) | UpgradeASTGCN (s) | Upgrade / Base |')
    lines.append('|---|---|---|---|')
    for ds, mp in by_ds.items():
        base = mp.get('base')
        up = mp.get('upgrade')
        if base and up:
            lines.append(f'| {ds} | {base:.2f} | {up:.2f} | {up/base:.2f}x |')
    lines.append('')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[OK] Markdown saved to: {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+',
                        default=['configurations/PEMS04.conf',
                                 'configurations/PEMS08.conf'])
    parser.add_argument('--variants', nargs='+', default=['base', 'upgrade'])
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--measure_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=0,
                        help='0 表示使用配置文件里的 batch_size')
    parser.add_argument('--output', type=str,
                        default='benchmark_epoch_time.md')
    args = parser.parse_args()

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    rows = []
    for cfg_path in args.configs:
        for variant in args.variants:
            print(f'\n=== {cfg_path} | {variant} ===')
            try:
                res = benchmark(cfg_path, variant,
                                args.warmup_epochs, args.measure_epochs,
                                args.batch_size)
                print(f'  mean={res["mean_sec"]:.2f}s std={res["std_sec"]:.2f}s '
                      f'(bs={res["batch_size"]}, batches={res["n_batches"]})')
                rows.append(res)
            except Exception as e:
                print(f'  [FAILED] {e}')
                rows.append({
                    'config': cfg_path,
                    'dataset': os.path.basename(cfg_path).replace('.conf', ''),
                    'variant': variant,
                    'batch_size': 0, 'n_samples': 0, 'n_batches': 0,
                    'n_params': 0,
                    'mean_sec': float('nan'), 'std_sec': float('nan'),
                    'min_sec': float('nan'), 'max_sec': float('nan'),
                    'device': 'n/a',
                })

    env_info = {
        'device_name': (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else 'CPU'),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda or 'cpu',
    }
    write_markdown(rows, args.output, env_info)


if __name__ == '__main__':
    main()
