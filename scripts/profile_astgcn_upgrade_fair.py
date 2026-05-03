#!/usr/bin/env python
import argparse
import csv
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from thop import profile as thop_profile

ROOT = "/root/ASTGCN-2"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.config import load_config
from model.astgcn import ASTGCN
from model.model_config import get_backbones
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(config_path: str, device: torch.device):
    cfg = load_config(config_path)
    all_backbones = get_backbones(
        config_filename=config_path,
        adj_filename=cfg.data.adj_filename,
        ctx=device,
    )
    ag_dict = asdict(cfg.upgrade.adaptive_graph)
    tf_dict = asdict(cfg.upgrade.transformer)

    astgcn = ASTGCN(cfg.data.num_for_predict, all_backbones).to(device).eval()
    upgrade_astgcn = UpgradeASTGCN(
        num_of_features=3,
        num_for_prediction=cfg.data.num_for_predict,
        all_backbones=all_backbones,
        num_of_vertices=cfg.data.num_of_vertices,
        spatial_mode=1,
        temporal_mode=1,
        adaptive_graph_cfg=ag_dict,
        transformer_cfg=tf_dict,
    ).to(device).eval()
    return cfg, astgcn, upgrade_astgcn


def make_inputs(batch_size: int, num_nodes: int, in_steps: int, num_features: int, device: torch.device):
    week_input = torch.randn(batch_size, num_nodes, num_features, in_steps, device=device)
    day_input = torch.randn(batch_size, num_nodes, num_features, in_steps, device=device)
    recent_input = torch.randn(batch_size, num_nodes, num_features, in_steps, device=device)
    return [week_input, day_input, recent_input]


def profile_flops_params(model: torch.nn.Module, x_list: List[torch.Tensor]) -> Tuple[float, float]:
    with torch.no_grad():
        macs, params = thop_profile(model, inputs=(x_list,), verbose=False)
    return float(params), float(macs) * 2.0


def measure_latency_ms(model: torch.nn.Module, x_list: List[torch.Tensor], warmup: int, runs: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(x_list)
        torch.cuda.synchronize()

        values = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x_list)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def profile_one(
    name: str,
    model: torch.nn.Module,
    batch_size: int,
    num_nodes: int,
    in_steps: int,
    num_features: int,
    device: torch.device,
    warmup: int,
    runs: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    flops_vals = []
    latency_vals = []
    params_val = None

    for i in range(repeats):
        set_seed(seed + i)
        x_list = make_inputs(batch_size, num_nodes, in_steps, num_features, device)

        with torch.no_grad():
            model(x_list)
            torch.cuda.synchronize()

        params, flops = profile_flops_params(model, x_list)
        lat = measure_latency_ms(model, x_list, warmup, runs)

        if params_val is None:
            params_val = params
        flops_vals.append(flops)
        latency_vals.append(lat)

    flops_arr = np.asarray(flops_vals, dtype=np.float64)
    lat_arr = np.asarray(latency_vals, dtype=np.float64)
    latency_ms = float(np.mean(lat_arr))

    return {
        "model": name,
        "params": int(params_val),
        "params_m": float(params_val / 1e6),
        "flops": float(np.mean(flops_arr)),
        "flops_g": float(np.mean(flops_arr) / 1e9),
        "flops_std": float(np.std(flops_arr)),
        "latency_ms": latency_ms,
        "latency_ms_std": float(np.std(lat_arr)),
        "throughput_sps": float(batch_size * 1000.0 / latency_ms),
    }


def merge_into_csv(target_csv: str, new_rows: List[Dict[str, float]]) -> None:
    fieldnames = [
        "model",
        "params",
        "params_m",
        "flops",
        "flops_g",
        "flops_std",
        "latency_ms",
        "latency_ms_std",
        "throughput_sps",
    ]

    old_rows = []
    if os.path.exists(target_csv):
        with open(target_csv, "r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))

    replace_set = {"ASTGCN", "upgradeASTGCN"}
    kept_rows = [r for r in old_rows if r.get("model") not in replace_set]
    merged_rows = kept_rows + new_rows

    os.makedirs(os.path.dirname(target_csv), exist_ok=True)
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair profile for ASTGCN and upgradeASTGCN")
    parser.add_argument("--config", type=str, default="/root/ASTGCN-2/configurations/PEMS04.conf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=307)
    parser.add_argument("--in_steps", type=int, default=12)
    parser.add_argument("--num_features", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target_csv",
        type=str,
        default="/root/BasicTS/results/complexity/fair_five_models_pems04.csv",
    )
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("Fair protocol requires CUDA for latency via torch.cuda.Event")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    cfg, astgcn, upgrade_astgcn = build_models(args.config, device)
    if cfg.data.num_of_vertices != args.num_nodes:
        raise ValueError(f"Config num_of_vertices={cfg.data.num_of_vertices} does not match --num_nodes={args.num_nodes}")

    rows = [
        profile_one(
            "ASTGCN",
            astgcn,
            args.batch_size,
            args.num_nodes,
            args.in_steps,
            args.num_features,
            device,
            args.warmup,
            args.runs,
            args.repeats,
            args.seed,
        ),
        profile_one(
            "upgradeASTGCN",
            upgrade_astgcn,
            args.batch_size,
            args.num_nodes,
            args.in_steps,
            args.num_features,
            device,
            args.warmup,
            args.runs,
            args.repeats,
            args.seed,
        ),
    ]

    merge_into_csv(args.target_csv, rows)
    for row in rows:
        print(row)
    print(f"Merged into: {args.target_csv}")


if __name__ == "__main__":
    main()
