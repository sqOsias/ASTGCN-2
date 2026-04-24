"""
Shared data loading utilities for LSTM / GRU baselines.
"""
import numpy as np


def load_data(data_path, sensor_id=0, train_ratio=0.6, val_ratio=0.2, seq_len=12, horizon=12):
    """
    Load PEMS data for a single sensor and split into train / val / test.

    Returns
    -------
    (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), stats
        stats = dict with keys 'mean', 'std', 'sensor_id', 'seq_len', 'horizon'
    """
    data = np.load(data_path)['data']
    speed_data = data[:, sensor_id, 2]

    mean = float(speed_data.mean())
    std = float(speed_data.std())
    speed_norm = (speed_data - mean) / std

    X, Y = [], []
    for i in range(len(speed_norm) - seq_len - horizon + 1):
        X.append(speed_norm[i: i + seq_len])
        Y.append(speed_norm[i + seq_len: i + seq_len + horizon])

    X = np.expand_dims(np.array(X, dtype=np.float32), axis=-1)
    Y = np.array(Y, dtype=np.float32)

    num_samples = len(X)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    stats = {
        'mean': mean,
        'std': std,
        'sensor_id': sensor_id,
        'seq_len': seq_len,
        'horizon': horizon,
        'train_samples': train_end,
        'val_samples': val_end - train_end,
        'test_samples': num_samples - val_end,
    }

    return (
        (X[:train_end], Y[:train_end]),
        (X[train_end:val_end], Y[train_end:val_end]),
        (X[val_end:], Y[val_end:]),
        stats,
    )
