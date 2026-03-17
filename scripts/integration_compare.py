import argparse
import csv
import os


def load_metrics(path):
    metrics = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            horizon = int(row['horizon'])
            metrics[horizon] = {
                'MAE': float(row['MAE']),
                'RMSE': float(row['RMSE']),
                'MAPE': float(row['MAPE'])
            }
    return metrics


def compare_metrics(base_metrics, new_metrics, threshold):
    for horizon, base_row in base_metrics.items():
        if horizon not in new_metrics:
            raise SystemExit('Missing horizon %s in new metrics' % horizon)
        new_row = new_metrics[horizon]
        for key in ['MAE', 'RMSE', 'MAPE']:
            base_value = base_row[key]
            new_value = new_row[key]
            if base_value == 0:
                continue
            diff_ratio = abs(new_value - base_value) / base_value
            if diff_ratio > threshold:
                raise SystemExit('Metric %s horizon %s diff %.4f exceeds threshold %.4f' % (
                    key, horizon, diff_ratio, threshold
                ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--new_dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.03)
    args = parser.parse_args()

    base_metrics_path = os.path.join(args.base_dir, 'metrics', 'test_metrics.csv')
    new_metrics_path = os.path.join(args.new_dir, 'metrics', 'test_metrics.csv')
    base_metrics = load_metrics(base_metrics_path)
    new_metrics = load_metrics(new_metrics_path)
    compare_metrics(base_metrics, new_metrics, args.threshold)
    print('Integration check passed')


if __name__ == '__main__':
    main()
