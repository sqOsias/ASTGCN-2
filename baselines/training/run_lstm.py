import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMPredictor
from lib.metrics import mean_absolute_error, mean_squared_error, masked_mape_np


def load_data(data_path, sensor_id=0, train_ratio=0.6, val_ratio=0.2, seq_len=12, horizon=12):
    data = np.load(data_path)['data']
    speed_data = data[:, sensor_id, 2]
    
    mean = speed_data.mean()
    std = speed_data.std()
    speed_norm = (speed_data - mean) / std
    
    X, Y = [], []
    for i in range(len(speed_norm) - seq_len - horizon + 1):
        X.append(speed_norm[i : i+seq_len])
        Y.append(speed_norm[i+seq_len : i+seq_len+horizon])
        
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    
    X = np.expand_dims(X, axis=-1)
    
    num_samples = len(X)
    train_split = int(num_samples * train_ratio)
    val_split = int(num_samples * (train_ratio + val_ratio))
    
    train_X, train_Y = X[:train_split], Y[:train_split]
    val_X, val_Y = X[train_split:val_split], Y[train_split:val_split]
    test_X, test_Y = X[val_split:], Y[val_split:]
    
    return (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), (mean, std)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), (mean, std) = load_data(
        args.data_path, sensor_id=args.sensor_id, seq_len=args.seq_len, horizon=args.horizon
    )
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y)), 
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)), 
                            batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_Y)), 
                             batch_size=args.batch_size)
    
    model = LSTMPredictor(input_size=1, hidden_size=args.hidden_size, output_size=args.horizon).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print("Start Training...")
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
        
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'baselines/results/best_lstm.pth')
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training Finished. Testing...")
    
    model.load_state_dict(torch.load('baselines/results/best_lstm.pth'))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    preds = preds * std + mean
    trues = trues * std + mean
    
    mae = mean_absolute_error(preds, trues)
    mse = mean_squared_error(preds, trues)
    rmse = mse ** 0.5
    mape = masked_mape_np(preds, trues, 0)
    
    print(f"Test Results (Sensor {args.sensor_id}):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    results_dir = os.path.join('baselines', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'lstm_results.npz')
    np.savez_compressed(
        results_path,
        prediction=preds,
        ground_truth=trues,
        mae=np.array(mae, dtype=np.float64),
        mse=np.array(mse, dtype=np.float64),
        rmse=np.array(rmse, dtype=np.float64),
        mape=np.array(mape, dtype=np.float64),
        sensor_id=np.array(args.sensor_id, dtype=np.int64),
        seq_len=np.array(args.seq_len, dtype=np.int64),
        horizon=np.array(args.horizon, dtype=np.int64),
        mean=np.array(mean, dtype=np.float64),
        std=np.array(std, dtype=np.float64)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
