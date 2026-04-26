# -*- coding:utf-8 -*-
"""
Configuration constants for the real-time traffic prediction system.
"""

import json
import os


_BACKEND_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, '../..'))


def _to_abs_path(path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(_PROJECT_ROOT, path_value))


def _resolve_run_config_path(model_path):
    checkpoints_dir = os.path.dirname(model_path)
    run_dir = os.path.dirname(checkpoints_dir)
    return os.path.join(run_dir, 'configs', 'resolved_config.json')


CONFIG = {
    'num_of_vertices': 307,
    'num_for_predict': 12,
    'points_per_hour': 12,
    'num_of_weeks': 1,
    'num_of_days': 1,
    'num_of_hours': 3,
    'K': 3,
    'num_of_chev_filters': 64,
    'num_of_time_filters': 64,
    'time_conv_strides': 1,
    'time_interval_minutes': 5,
    'simulation_speed': 1.0,
    'num_input_features': 5,
    'spatial_mode': 0,
    'temporal_mode': 0,
    'adaptive_graph_cfg': {
        'embedding_dim': 10,
        'sparse_ratio': 0.2,
        'directed': True,
    },
    'transformer_cfg': {
        'd_model': 32,
        'n_heads': 2,
        'e_layers': 2,
        'dropout': 0.1,
        'max_len': 36,
        'factor': 5,
    },
    'data_path': os.path.join(_PROJECT_ROOT, 'data/PEMS04/pems04.npz'),
    'distance_path': os.path.join(_PROJECT_ROOT, 'data/PEMS04/distance.csv'),
    'model_path': os.path.join(
        _PROJECT_ROOT,
        'results/ASTGCN_lr0p001/1_1_20260414153308/checkpoints/best_model.pth',
    ),
}


def _derive_runtime_values():
    week_len = CONFIG['num_of_weeks'] * CONFIG['points_per_hour']
    day_len = CONFIG['num_of_days'] * CONFIG['points_per_hour']
    hour_len = CONFIG['num_of_hours'] * CONFIG['points_per_hour']
    CONFIG['week_steps'] = week_len
    CONFIG['day_steps'] = day_len
    CONFIG['hour_steps'] = hour_len
    CONFIG['history_steps'] = max(week_len, day_len, hour_len)


def load_model_runtime_config():
    """Load dynamic runtime settings from model run configs if present."""
    model_path = _to_abs_path(CONFIG['model_path'])
    CONFIG['model_path'] = model_path

    run_cfg_path = _resolve_run_config_path(model_path)
    CONFIG['run_config_path'] = run_cfg_path

    if not os.path.exists(run_cfg_path):
        print(f"[config] run config not found, using defaults: {run_cfg_path}")
        _derive_runtime_values()
        return CONFIG

    with open(run_cfg_path, 'r', encoding='utf-8') as f:
        run_cfg = json.load(f)

    data_cfg = run_cfg.get('Data', {})
    train_cfg = run_cfg.get('Training', {})
    upgrade_cfg = run_cfg.get('ModelUpgrade', {})

    if data_cfg:
        CONFIG['num_of_vertices'] = int(data_cfg.get('num_of_vertices', CONFIG['num_of_vertices']))
        CONFIG['num_for_predict'] = int(data_cfg.get('num_for_predict', CONFIG['num_for_predict']))
        CONFIG['points_per_hour'] = int(data_cfg.get('points_per_hour', CONFIG['points_per_hour']))
        CONFIG['data_path'] = _to_abs_path(data_cfg.get('graph_signal_matrix_filename', CONFIG['data_path']))
        CONFIG['distance_path'] = _to_abs_path(data_cfg.get('adj_filename', CONFIG['distance_path']))

    if train_cfg:
        CONFIG['num_of_weeks'] = int(train_cfg.get('num_of_weeks', CONFIG['num_of_weeks']))
        CONFIG['num_of_days'] = int(train_cfg.get('num_of_days', CONFIG['num_of_days']))
        CONFIG['num_of_hours'] = int(train_cfg.get('num_of_hours', CONFIG['num_of_hours']))
        CONFIG['K'] = int(train_cfg.get('K', CONFIG['K']))

    CONFIG['spatial_mode'] = int(upgrade_cfg.get('spatial_mode', 0))
    CONFIG['temporal_mode'] = int(upgrade_cfg.get('temporal_mode', 0))
    CONFIG['adaptive_graph_cfg'] = upgrade_cfg.get('adaptive_graph', CONFIG['adaptive_graph_cfg'])

    transformer_cfg = upgrade_cfg.get('transformer', {})
    if transformer_cfg:
        merged = dict(CONFIG['transformer_cfg'])
        merged.update(transformer_cfg)
        CONFIG['transformer_cfg'] = merged

    _derive_runtime_values()
    print(
        "[config] loaded model runtime config:",
        f"spatial_mode={CONFIG['spatial_mode']}, temporal_mode={CONFIG['temporal_mode']},",
        f"vertices={CONFIG['num_of_vertices']}, history_steps={CONFIG['history_steps']}"
    )
    return CONFIG


_derive_runtime_values()
