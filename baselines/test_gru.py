"""
GRU Baseline — Test Script
Loads checkpoint from training, runs inference on test set,
outputs: test_metrics.csv, predictions .npz
Usage: python -m baselines.test_gru [--batch_size 64]
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.gru_model import GRUPredictor
from baselines.utils.data_utils import load_data
from lib.metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def test(args):
    results_dir = os.path.join('baselines', 'results')

    # ---- load meta from training ----
    h = args.horizon
    meta_path = os.path.join(results_dir, f'gru_h{h}_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Please run training first:\n"
            f"  python -m baselines.run_gru --horizon {h}"
        )
    with open(meta_path) as f:
        meta = json.load(f)

    print("=" * 60)
    print("GRU Test — loading checkpoint")
    print(f"  Model config : hidden={meta['hidden_size']}, horizon={meta['horizon']}")
    print(f"  Trained on   : sensor={meta['sensor_id']}, seq_len={meta['seq_len']}")
    print(f"  Best val loss: {meta['best_val_loss']:.6f}")
    print(f"  Data path    : {meta['data_path']}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data (only need test split) ----
    _, _, (test_X, test_Y), stats = load_data(
        meta['data_path'],
        sensor_id=meta['sensor_id'],
        seq_len=meta['seq_len'],
        horizon=meta['horizon'],
    )

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_Y)),
        batch_size=args.batch_size,
    )
    print(f"  Test samples : {len(test_X)}")

    # ---- load model ----
    ckpt_path = os.path.join(results_dir, f'best_gru_h{h}.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = GRUPredictor(
        input_size=1,
        hidden_size=meta['hidden_size'],
        output_size=meta['horizon'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ---- inference ----
    preds_list, trues_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds_list.append(out.cpu().numpy())
            trues_list.append(y.numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # ---- inverse normalization ----
    mean, std = meta['mean'], meta['std']
    preds = preds * std + mean
    trues = trues * std + mean

    # ---- metrics ----
    mae = mean_absolute_error(preds, trues)
    mse = mean_squared_error(preds, trues)
    rmse = mse ** 0.5
    mape = masked_mape_np(preds, trues, 0)

    print("-" * 60)
    print(f"Test Results (Sensor {meta['sensor_id']}):")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.4f}%")

    # ---- save per-model test metrics CSV ----
    metrics_csv = os.path.join(results_dir, f'gru_h{h}_test_metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'sensor_id', 'seq_len', 'horizon', 'MAE', 'RMSE', 'MAPE(%)'])
        w.writerow(['GRU', meta['sensor_id'], meta['seq_len'], meta['horizon'], mae, rmse, mape])

    # ---- append to unified metrics CSV ----
    unified_csv = os.path.join(results_dir, 'all_test_metrics.csv')
    write_header = not os.path.exists(unified_csv) or os.path.getsize(unified_csv) == 0
    with open(unified_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['model', 'sensor_id', 'seq_len', 'horizon', 'MAE', 'RMSE', 'MAPE(%)'])
        w.writerow(['GRU', meta['sensor_id'], meta['seq_len'], meta['horizon'], mae, rmse, mape])

    # ---- save predictions ----
    npz_path = os.path.join(results_dir, f'gru_h{h}_results.npz')
    np.savez_compressed(
        npz_path,
        prediction=preds,
        ground_truth=trues,
        mae=np.float64(mae),
        mse=np.float64(mse),
        rmse=np.float64(rmse),
        mape=np.float64(mape),
        sensor_id=np.int64(meta['sensor_id']),
        seq_len=np.int64(meta['seq_len']),
        horizon=np.int64(meta['horizon']),
        mean=np.float64(mean),
        std=np.float64(std),
    )

    print("-" * 60)
    print("Test Finished. Output files:")
    print(f"  Metrics CSV  : {metrics_csv}")
    print(f"  Predictions  : {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test GRU baseline (uses saved checkpoint)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=12, help='Prediction horizon (must match training)')
    args = parser.parse_args()

    test(args)
