# -*- coding:utf-8 -*-

"""
实验管理模块。
负责实验目录结构创建、配置快照保存、指标 CSV 写入、
权重文件清理、预测结果保存等产物管理工作。
"""

import os
import csv
import json
import shutil
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

from .config import ExperimentConfig


# ---------------------------------------------------------------------------
# YAML 简易序列化（无需引入第三方库）
# ---------------------------------------------------------------------------

def _write_yaml(path, data, indent=0):
    lines = []
    pad = '  ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append('%s%s:' % (pad, key))
                lines.extend(_write_yaml(None, value, indent + 1))
            elif isinstance(value, list):
                lines.append('%s%s:' % (pad, key))
                for item in value:
                    lines.append('%s- %s' % ('  ' * (indent + 1), item))
            else:
                lines.append('%s%s: %s' % (pad, key, value))
    if path:
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    return lines


# ---------------------------------------------------------------------------
# 安全保存 npz
# ---------------------------------------------------------------------------

def save_npz(file_path, **arrays):
    temp_path = file_path + '.tmp.npz'
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, file_path)


# ---------------------------------------------------------------------------
# 实验运行管理器
# ---------------------------------------------------------------------------

class ExperimentManager:
    """管理单次训练运行的所有实验产物。"""

    def __init__(self, params_path: str, config: ExperimentConfig,
                 run_id: str, timestamp: str,
                 spatial_mode: int, temporal_mode: int,
                 device: torch.device, force: bool = False):
        self.params_path = params_path
        self.config = config
        self.run_id = run_id
        self.timestamp = timestamp
        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode
        self.device = device
        self.force = force

        # 子目录路径
        self.checkpoints_dir = os.path.join(params_path, 'checkpoints')
        self.logs_dir = os.path.join(params_path, 'logs')
        self.metrics_dir = os.path.join(params_path, 'metrics')
        self.predictions_dir = os.path.join(params_path, 'predictions')
        self.artifacts_dir = os.path.join(params_path, 'artifacts')
        self.configs_dir = os.path.join(params_path, 'configs')

        # 关键文件路径
        self.best_model_path = os.path.join(self.checkpoints_dir, 'best_model.pth')
        self.train_loss_csv = os.path.join(self.metrics_dir, 'train_loss.csv')
        self.val_metrics_csv = os.path.join(self.metrics_dir, 'val_metrics.csv')
        self.test_metrics_csv = os.path.join(self.metrics_dir, 'test_metrics.csv')

    # ----- 目录与初始化 -----

    def setup(self):
        """创建实验目录结构并保存配置快照。"""
        for path in [self.checkpoints_dir, self.logs_dir, self.metrics_dir,
                     self.predictions_dir, self.artifacts_dir, self.configs_dir]:
            os.makedirs(path, exist_ok=True)

        self._save_config_snapshot()
        self._init_metric_csvs()
        self._cleanup_legacy_files()

    def _save_config_snapshot(self):
        """保存原始 conf 文件副本、resolved JSON 和 YAML。"""
        # 原始配置副本
        config_snapshot_path = os.path.join(self.configs_dir, 'train.conf')
        if os.path.exists(self.config.config_file):
            shutil.copyfile(self.config.config_file, config_snapshot_path)

        # resolved JSON
        resolved = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'config_file': self.config.config_file,
            'force': self.force,
            'Data': asdict(self.config.data),
            'Training': asdict(self.config.training),
            'ModelUpgrade': {
                'spatial_mode': self.spatial_mode,
                'temporal_mode': self.temporal_mode,
                'adaptive_graph': asdict(self.config.upgrade.adaptive_graph),
                'transformer': asdict(self.config.upgrade.transformer),
            },
            'runtime': {
                'device': str(self.device),
                'torch_version': torch.__version__,
                'seed': self.config.training.seed,
            }
        }
        resolved_path = os.path.join(self.configs_dir, 'resolved_config.json')
        with open(resolved_path, 'w') as f:
            json.dump(resolved, f, indent=2, ensure_ascii=False)

        # YAML
        yaml_data = {
            'Data': asdict(self.config.data),
            'Training': asdict(self.config.training),
            'ModelUpgrade': {
                'spatial_mode': self.spatial_mode,
                'temporal_mode': self.temporal_mode,
                'adaptive_graph': asdict(self.config.upgrade.adaptive_graph),
                'transformer': asdict(self.config.upgrade.transformer),
            }
        }
        _write_yaml(os.path.join(self.configs_dir, 'config.yaml'), yaml_data)

        # 环境信息
        env_path = os.path.join(self.artifacts_dir, 'environment.txt')
        with open(env_path, 'w') as f:
            f.write('torch=%s\n' % torch.__version__)
            f.write('cuda_available=%s\n' % torch.cuda.is_available())
            f.write('device=%s\n' % str(self.device))
            f.write('seed=%s\n' % self.config.training.seed)

    def _init_metric_csvs(self):
        """初始化指标 CSV 文件的表头。"""
        with open(self.train_loss_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'global_step', 'training_loss', 'batch_time_sec'])
        with open(self.val_metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'validation_loss'])
        with open(self.test_metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'horizon', 'MAE', 'RMSE', 'MAPE'])

    def _cleanup_legacy_files(self):
        """清理旧版产物文件。"""
        cleanup_intermediate_weight_files(self.params_path, self.checkpoints_dir,
                                          self.best_model_path)
        legacy_stats_files = [
            os.path.join(self.params_path, 'stats_data.csv'),
            os.path.join(self.artifacts_dir, 'stats_data.csv'),
            os.path.join(self.params_path, 'stats_data.json'),
            os.path.join(self.artifacts_dir, 'stats_data.json'),
            os.path.join(self.params_path, 'stats_meta.json'),
            os.path.join(self.artifacts_dir, 'stats_meta.json'),
        ]
        for path in legacy_stats_files:
            if os.path.exists(path):
                os.remove(path)

    # ----- 指标记录 -----

    def log_train_loss(self, epoch: int, global_step: int,
                       training_loss: float, batch_time: float):
        with open(self.train_loss_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, global_step, training_loss, batch_time])

    def log_val_loss(self, epoch: int, val_loss: float):
        with open(self.val_metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, val_loss])

    def log_test_metrics(self, epoch: int, metrics: Dict, horizons: list):
        with open(self.test_metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for horizon in horizons:
                row = metrics[horizon]
                writer.writerow([epoch, horizon, row['MAE'], row['RMSE'], row['MAPE']])

    # ----- 模型与预测保存 -----

    def save_best_model(self, net, epoch: int, val_loss: float,
                        prediction: np.ndarray, test_target: np.ndarray):
        torch.save(net.state_dict(), self.best_model_path)
        cleanup_intermediate_weight_files(self.params_path, self.checkpoints_dir,
                                          self.best_model_path)
        test_results_path = os.path.join(self.predictions_dir, 'test_results.npz')
        save_npz(
            test_results_path,
            prediction=prediction,
            ground_truth=test_target,
            epoch=np.array(epoch, dtype=np.int64),
            validation_loss=np.array(val_loss, dtype=np.float64),
        )

    def save_stats(self, stats_data: dict):
        stats_npz_path = os.path.join(self.params_path, 'stats_data.npz')
        stats_artifact_path = os.path.join(self.artifacts_dir, 'stats_data.npz')
        save_npz(stats_npz_path, **stats_data)
        save_npz(stats_artifact_path, **stats_data)

    def save_runtime(self, train_elapsed: float, gpu_peak: int,
                     model_name: str, learning_rate: float):
        runtime_path = os.path.join(self.artifacts_dir, 'runtime.json')
        with open(runtime_path, 'w') as f:
            json.dump({
                'train_seconds': float(train_elapsed),
                'gpu_peak_bytes': gpu_peak,
                'spatial_mode': self.spatial_mode,
                'temporal_mode': self.temporal_mode,
                'model_name': model_name,
                'learning_rate': float(learning_rate),
            }, f, indent=2, ensure_ascii=False)

    def copy_prediction_file(self, prediction_filename: str):
        if not prediction_filename:
            return
        prediction_path = os.path.normpath(prediction_filename)
        if not prediction_path.endswith('.npz'):
            prediction_path = prediction_path + '.npz'
        src = os.path.join(self.predictions_dir, 'test_results.npz')
        if os.path.exists(src):
            shutil.copyfile(src, prediction_path)

    def final_cleanup(self):
        cleanup_intermediate_weight_files(self.params_path, self.checkpoints_dir,
                                          self.best_model_path)


# ---------------------------------------------------------------------------
# 权重文件清理（独立函数，供内外调用）
# ---------------------------------------------------------------------------

def cleanup_intermediate_weight_files(params_root, checkpoints_root,
                                      keep_file_path: Optional[str] = None):
    keep_file_path = os.path.normpath(keep_file_path) if keep_file_path else None
    for root in [params_root, checkpoints_root]:
        if not os.path.exists(root):
            continue
        for filename in os.listdir(root):
            current_path = os.path.normpath(os.path.join(root, filename))
            if keep_file_path and current_path == keep_file_path:
                continue
            if os.path.isfile(current_path) and filename.endswith(('.pt', '.pth', '.ckpt')):
                os.remove(current_path)
    legacy_best_model_json = os.path.join(checkpoints_root, 'best_model_weights.json')
    if os.path.exists(legacy_best_model_json):
        os.remove(legacy_best_model_json)
    legacy_best_model_dir = os.path.join(checkpoints_root, 'best_model')
    if os.path.exists(legacy_best_model_dir):
        shutil.rmtree(legacy_best_model_dir)


# ---------------------------------------------------------------------------
# 实验目录构建辅助
# ---------------------------------------------------------------------------

def build_params_path(training_cfg, model_name: str, lr: float,
                      spatial_mode: int, temporal_mode: int,
                      timestamp: str) -> str:
    from .config import _format_lr_tag
    run_tag = '%s_%s_%s' % (spatial_mode, temporal_mode, timestamp)
    group_tag = '%s_lr%s' % (model_name, _format_lr_tag(lr))
    if training_cfg.params_dir and training_cfg.params_dir != "None":
        return os.path.join(training_cfg.params_dir, group_tag, run_tag)
    return os.path.join('results', group_tag, run_tag)


def ensure_params_dir(params_path: str, force: bool):
    if os.path.exists(params_path) and not force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
