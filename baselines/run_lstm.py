"""
LSTM Baseline — Training Script
Saves: checkpoint (.pth), training log (.csv), normalization stats (.json)
Usage: python -m baselines.run_lstm [--epochs 100 ...]
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.lstm_model import LSTMPredictor
from baselines.utils.data_utils import load_data


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = os.path.join('baselines', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ---- data ----
    (train_X, train_Y), (val_X, val_Y), _, stats = load_data(
        args.data_path, sensor_id=args.sensor_id,
        seq_len=args.seq_len, horizon=args.horizon,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)),
        batch_size=args.batch_size)

    # ---- model ----
    model = LSTMPredictor(
        input_size=1, hidden_size=args.hidden_size,
        output_size=args.horizon,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    training_log = []

    print(f"Using device: {device}")
    print(f"Start Training LSTM ...")
    print(f"  Hidden: {args.hidden_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"  Sensor: {args.sensor_id} | Seq_len: {args.seq_len} | Horizon: {args.horizon}")
    print(f"  Train/Val samples: {stats['train_samples']}/{stats['val_samples']}")
    print("-" * 60)

    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss.append(loss.item())

        avg_train = np.mean(train_loss)
        avg_val = np.mean(val_loss)

        saved = ''
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(results_dir, f'best_lstm_h{args.horizon}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_size': args.hidden_size,
                'horizon': args.horizon,
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            saved = ' *'

        training_log.append([epoch + 1, avg_train, avg_val, best_val_loss])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  "
                  f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}{saved}")

    # ---- save training log ----
    log_csv = os.path.join(results_dir, f'lstm_h{args.horizon}_training_log.csv')
    with open(log_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss'])
        w.writerows(training_log)

    # ---- save normalization stats (test script needs them) ----
    meta_path = os.path.join(results_dir, f'lstm_h{args.horizon}_meta.json')
    meta = {
        'model': 'LSTM',
        'hidden_size': args.hidden_size,
        'sensor_id': args.sensor_id,
        'seq_len': args.seq_len,
        'horizon': args.horizon,
        'data_path': args.data_path,
        'mean': stats['mean'],
        'std': stats['std'],
        'best_val_loss': best_val_loss,
        'total_epochs': args.epochs,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("-" * 60)
    print("Training Finished.")
    print(f"  Checkpoint : {os.path.join(results_dir, f'best_lstm_h{args.horizon}.pth')}")
    print(f"  Training log: {log_csv}")
    print(f"  Meta info   : {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM baseline')
    parser.add_argument('--data_path', type=str, default='data/PEMS04/pems04.npz')
    parser.add_argument('--sensor_id', type=int, default=0, help='Sensor ID to predict')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    args = parser.parse_args()

    train(args)
