# -*- coding:utf-8 -*-

import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse
import csv
import json

import numpy as np
import torch  # MXNet:import mxnet as mx → PyTorch:import torch
from torch import nn  # MXNet:from mxnet import gluon → PyTorch:from torch import nn
from torch.utils.data import TensorDataset, DataLoader  # MXNet:gluon.data.ArrayDataset/DataLoader → PyTorch:TensorDataset/DataLoader
from torch.utils.tensorboard import SummaryWriter  # MXNet:from mxboard import SummaryWriter → PyTorch:torch.utils.tensorboard.SummaryWriter


from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from model.model_config import get_backbones

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,default="configurations/PEMS04.conf",
                    help="configuration file path", required=False)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()

# mxboard log dir
if os.path.exists('logs'):
    shutil.rmtree('logs')
    print('Remove log dir')

# read configuration
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

def _parse_list(value):
    value = value.strip()
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    items = [item.strip() for item in value.split(',')]
    return [item for item in items if item]


def _parse_int_list(value):
    return [int(v) for v in _parse_list(value)]


def _parse_bool(value):
    value = str(value).strip().lower()
    return value in ['1', 'true', 'yes', 'y', 'on']


def _format_lr_tag(value):
    return ('%s' % value).replace('.', 'p')


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


class MyInit(object):  # MXNet:mx.init.Initializer → PyTorch:custom init wrapper
    @staticmethod
    def _init_weight(name, data):
        if len(data.shape) < 2:
            nn.init.uniform_(data)  # MXNet:Uniform init → PyTorch:nn.init.uniform_
            print('Init', name, tuple(data.shape), 'with Uniform')
        else:
            nn.init.xavier_uniform_(data)  # MXNet:Xavier init → PyTorch:nn.init.xavier_uniform_
            print('Init', name, tuple(data.shape), 'with Xavier')


def cleanup_intermediate_weight_files(params_root, checkpoints_root, keep_file_path):
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


def save_npz(file_path, **arrays):
    temp_path = file_path + '.tmp.npz'
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, file_path)


def run_training_block(params_path, model_name, model, ctx, optimizer, learning_rate,
                       epochs, batch_size, num_of_weeks, num_of_days, num_of_hours,
                       merge, seed, timestamp, spatial_mode, temporal_mode,
                       adaptive_graph_cfg, transformer_cfg, horizons):
    run_id = os.path.basename(os.path.normpath(params_path))
    checkpoints_dir = os.path.join(params_path, 'checkpoints')
    logs_dir = os.path.join(params_path, 'logs')
    metrics_dir = os.path.join(params_path, 'metrics')
    predictions_dir = os.path.join(params_path, 'predictions')
    artifacts_dir = os.path.join(params_path, 'artifacts')
    configs_dir = os.path.join(params_path, 'configs')

    for path in [checkpoints_dir, logs_dir, metrics_dir, predictions_dir,
                 artifacts_dir, configs_dir]:
        os.makedirs(path, exist_ok=True)

    config_snapshot_path = os.path.join(configs_dir, 'train.conf')
    shutil.copyfile(args.config, config_snapshot_path)

    resolved_config_path = os.path.join(configs_dir, 'resolved_config.json')
    resolved_config = {
        'run_id': run_id,
        'timestamp': timestamp,
        'config_file': args.config,
        'force': args.force,
        'Data': dict(data_config),
        'Training': dict(training_config),
        'ModelUpgrade': {
            'spatial_mode': int(spatial_mode),
            'temporal_mode': int(temporal_mode),
            'adaptive_graph': adaptive_graph_cfg,
            'transformer': transformer_cfg
        },
        'runtime': {
            'device': str(ctx),
            'torch_version': torch.__version__,
            'seed': seed
        }
    }
    with open(resolved_config_path, 'w') as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False)

    config_yaml_path = os.path.join(configs_dir, 'config.yaml')
    _write_yaml(config_yaml_path, {
        'Data': dict(data_config),
        'Training': dict(training_config),
        'ModelUpgrade': {
            'spatial_mode': int(spatial_mode),
            'temporal_mode': int(temporal_mode),
            'adaptive_graph': adaptive_graph_cfg,
            'transformer': transformer_cfg
        }
    })

    environment_path = os.path.join(artifacts_dir, 'environment.txt')
    with open(environment_path, 'w') as f:
        f.write('torch=%s\n' % torch.__version__)
        f.write('cuda_available=%s\n' % torch.cuda.is_available())
        f.write('device=%s\n' % str(ctx))
        f.write('seed=%s\n' % seed)

    train_loss_csv = os.path.join(metrics_dir, 'train_loss.csv')
    val_metrics_csv = os.path.join(metrics_dir, 'val_metrics.csv')
    test_metrics_csv = os.path.join(metrics_dir, 'test_metrics.csv')
    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    cleanup_intermediate_weight_files(params_path, checkpoints_dir, best_model_path)

    with open(train_loss_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'global_step', 'training_loss', 'batch_time_sec'])
    with open(val_metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'validation_loss'])
    with open(test_metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'horizon', 'MAE', 'RMSE', 'MAPE'])

    print("Reading data...")
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # true_value 将在删除 all_data 之前创建
    # true_value = (all_data['test']['target'].transpose((0, 2, 1))
    #               .reshape(all_data['test']['target'].shape[0], -1))

    # 提取统计信息
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean'].copy()  # 创建副本以避免引用问题
        stats_data[type_ + '_std'] = stats['std'].copy()

    # 提取特征数量用于模型初始化
    num_of_features = all_data['train']['recent'].shape[-2]  # 获取输入数据的特征维度 (倒数第二维)

    # 为减少内存使用，仅将小批量数据加载到设备
    # 首先将数据转为tensor但暂不移动到设备
    train_week_tensor = torch.from_numpy(all_data['train']['week']).float()
    train_day_tensor = torch.from_numpy(all_data['train']['day']).float()
    train_recent_tensor = torch.from_numpy(all_data['train']['recent']).float()
    train_target_tensor = torch.from_numpy(all_data['train']['target']).float()

    val_week_tensor = torch.from_numpy(all_data['val']['week']).float()
    val_day_tensor = torch.from_numpy(all_data['val']['day']).float()
    val_recent_tensor = torch.from_numpy(all_data['val']['recent']).float()
    val_target_tensor = torch.from_numpy(all_data['val']['target']).float()

    test_week_tensor = torch.from_numpy(all_data['test']['week']).float()
    test_day_tensor = torch.from_numpy(all_data['test']['day']).float()
    test_recent_tensor = torch.from_numpy(all_data['test']['recent']).float()
    test_target_tensor = torch.from_numpy(all_data['test']['target']).float()

    # 保存测试目标数据用于后续评估
    test_target_np = all_data['test']['target']
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # 删除原始数据以释放内存
    del all_data

    # 创建数据加载器，但不在这里将数据移到设备
    train_loader = DataLoader(
        TensorDataset(train_week_tensor, train_day_tensor, train_recent_tensor, train_target_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_week_tensor, val_day_tensor, val_recent_tensor, val_target_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test_week_tensor, test_day_tensor, test_recent_tensor, test_target_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    # 保存统计信息
    stats_npz_path = os.path.join(params_path, 'stats_data.npz')
    stats_npz_artifact_path = os.path.join(artifacts_dir, 'stats_data.npz')
    save_npz(stats_npz_path, **stats_data)
    save_npz(stats_npz_artifact_path, **stats_data)

    legacy_stats_files = [
        os.path.join(params_path, 'stats_data.csv'),
        os.path.join(artifacts_dir, 'stats_data.csv'),
        os.path.join(params_path, 'stats_data.json'),
        os.path.join(artifacts_dir, 'stats_data.json'),
        os.path.join(params_path, 'stats_meta.json'),
        os.path.join(artifacts_dir, 'stats_meta.json'),
    ]
    for legacy_path in legacy_stats_files:
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    # 训练时再将批次数据移动到设备（如果使用GPU）
    if str(ctx).startswith('cuda'):
        print("注意: 使用GPU训练，将在训练循环中动态移动数据到GPU以节省内存")

    loss_function = nn.MSELoss(reduction='none')

    all_backbones = get_backbones(args.config, adj_filename, ctx)

    if model_name == 'MSTGCN' and (spatial_mode != 0 or temporal_mode != 0):
        raise SystemExit('Upgrade modes only supported for ASTGCN')
    if spatial_mode == 0 and temporal_mode == 0:
        net = model(num_for_predict, all_backbones).to(ctx)
    else:
        from model.upgrade.astgcn_upgrade import UpgradeASTGCN
        net = UpgradeASTGCN(
            num_of_features,      # 输入特征数量
            num_for_predict,      # 预测时间步数
            all_backbones,        # 骨干网络配置
            num_of_vertices,      # 节点数量
            spatial_mode,         # 空间模式
            temporal_mode,        # 时间模式
            adaptive_graph_cfg,   # 自适应图配置
            transformer_cfg       # Transformer配置
        ).to(ctx)
    for val_w, val_d, val_r, val_t in val_loader:
        # 将数据移动到正确的设备
        val_w, val_d, val_r = val_w.to(ctx), val_d.to(ctx), val_r.to(ctx)
        net([val_w, val_d, val_r])
        break
    for name, param in net.named_parameters():
        MyInit._init_weight(name, param.data)

    if optimizer.lower() == 'adam':
        trainer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer.lower() == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        trainer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    else:
        raise SystemExit('Unsupported optimizer for PyTorch: %s' % optimizer)

    sw = SummaryWriter(log_dir=logs_dir, flush_secs=5)

    val_loss = compute_val_loss(net, val_loader, loss_function, sw, epoch=0)
    with open(val_metrics_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, val_loss])

    test_metrics, _ = evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch=0, horizons=horizons)
    with open(test_metrics_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for horizon in horizons:
            row = test_metrics[horizon]
            writer.writerow([0, horizon, row['MAE'], row['RMSE'], row['MAPE']])

    best_val_loss = float('inf')

    if ctx.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(ctx)
    train_start_time = time()
    global_step = 1
    for epoch in range(1, epochs + 1):
        net.train()

        for train_w, train_d, train_r, train_t in train_loader:
            start_time = time()

            # 将数据移动到正确的设备
            train_w, train_d, train_r, train_t = train_w.to(ctx), train_d.to(ctx), train_r.to(ctx), train_t.to(ctx)

            with torch.set_grad_enabled(True):
                output = net([train_w, train_d, train_r])
                l = 0.5 * loss_function(output, train_t)
            trainer.zero_grad()
            loss_scalar = l.mean()
            loss_scalar.backward()
            trainer.step()
            training_loss = loss_scalar.item()

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 100 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs'
                      % (global_step, training_loss, time() - start_time))
            with open(train_loss_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, training_loss, time() - start_time])
            global_step += 1

        for name, param in net.named_parameters():
            try:
                if param.grad is not None:
                    sw.add_histogram(tag=name + "_grad",
                                     values=param.grad.detach().cpu().numpy(),
                                     global_step=global_step,
                                     bins=1000)
            except:
                print("can't plot histogram of {}_grad".format(name))

        val_loss = compute_val_loss(net, val_loader, loss_function, sw, epoch)
        with open(val_metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_loss])

        test_metrics, _ = evaluate(net, test_loader, true_value, num_of_vertices, sw, epoch, horizons=horizons)
        with open(test_metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for horizon in horizons:
                row = test_metrics[horizon]
                writer.writerow([epoch, horizon, row['MAE'], row['RMSE'], row['MAPE']])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_model_path)
            cleanup_intermediate_weight_files(params_path, checkpoints_dir, best_model_path)
            prediction = predict(net, test_loader)
            test_results_path = os.path.join(predictions_dir, 'test_results.npz')
            save_npz(
                test_results_path,
                prediction=prediction,
                ground_truth=test_target_np,
                epoch=np.array(epoch, dtype=np.int64),
                validation_loss=np.array(best_val_loss, dtype=np.float64)
            )
            print('save best model to file: %s' % (best_model_path))

    train_elapsed = time() - train_start_time
    gpu_peak = 0
    if ctx.type == 'cuda':
        gpu_peak = int(torch.cuda.max_memory_allocated(ctx))
    runtime_path = os.path.join(artifacts_dir, 'runtime.json')
    with open(runtime_path, 'w') as f:
        json.dump({
            'train_seconds': float(train_elapsed),
            'gpu_peak_bytes': gpu_peak,
            'spatial_mode': int(spatial_mode),
            'temporal_mode': int(temporal_mode),
            'model_name': model_name,
            'learning_rate': float(learning_rate)
        }, f, indent=2, ensure_ascii=False)

    sw.close()
    cleanup_intermediate_weight_files(params_path, checkpoints_dir, best_model_path)

    if 'prediction_filename' in training_config:
        prediction_path = os.path.normpath(training_config['prediction_filename'])
        if not prediction_path.endswith('.npz'):
            prediction_path = prediction_path + '.npz'
        if os.path.exists(os.path.join(predictions_dir, 'test_results.npz')):
            shutil.copyfile(os.path.join(predictions_dir, 'test_results.npz'), prediction_path)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_names = _parse_list(training_config['model_name'])
    learning_rates = [float(v) for v in _parse_list(training_config['learning_rate'])]
    spatial_modes = [0]
    temporal_modes = [0]
    adaptive_graph_cfg = {
        'embedding_dim': 10,
        'sparse_ratio': 0.0,
        'directed': True
    }
    transformer_cfg = {
        'd_model': 64,
        'n_heads': 4,
        'e_layers': 2,
        'dropout': 0.1,
        'max_len': points_per_hour * 3,
        'topk_ratio': 0.5
    }

    if config.has_section('ModelUpgrade'):
        upgrade_cfg = config['ModelUpgrade']
        if 'spatial_mode' in upgrade_cfg:
            spatial_modes = _parse_int_list(upgrade_cfg['spatial_mode'])
        if 'temporal_mode' in upgrade_cfg:
            temporal_modes = _parse_int_list(upgrade_cfg['temporal_mode'])

    if config.has_section('AdaptiveGraph'):
        ag_cfg = config['AdaptiveGraph']
        if 'embedding_dim' in ag_cfg:
            adaptive_graph_cfg['embedding_dim'] = int(ag_cfg['embedding_dim'])
        if 'sparse_ratio' in ag_cfg:
            adaptive_graph_cfg['sparse_ratio'] = float(ag_cfg['sparse_ratio'])
        if 'directed' in ag_cfg:
            adaptive_graph_cfg['directed'] = _parse_bool(ag_cfg['directed'])

    if config.has_section('Transformer'):
        tf_cfg = config['Transformer']
        if 'd_model' in tf_cfg:
            transformer_cfg['d_model'] = int(tf_cfg['d_model'])
        if 'n_heads' in tf_cfg:
            transformer_cfg['n_heads'] = int(tf_cfg['n_heads'])
        if 'e_layers' in tf_cfg:
            transformer_cfg['e_layers'] = int(tf_cfg['e_layers'])
        if 'dropout' in tf_cfg:
            transformer_cfg['dropout'] = float(tf_cfg['dropout'])
        if 'max_len' in tf_cfg:
            transformer_cfg['max_len'] = int(tf_cfg['max_len'])
        if 'topk_ratio' in tf_cfg:
            transformer_cfg['topk_ratio'] = float(tf_cfg['topk_ratio'])

    horizons = [3, points_per_hour, points_per_hour * 3]
    horizons = [h for h in horizons if h <= num_for_predict]
    base_prediction_name = training_config['prediction_filename'] if 'prediction_filename' in training_config else None

    for model_name in model_names:
        for learning_rate in learning_rates:
            for spatial_mode in spatial_modes:
                for temporal_mode in temporal_modes:
                    training_config['model_name'] = model_name
                    training_config['learning_rate'] = str(learning_rate)

                    ctx = training_config['ctx']
                    optimizer = training_config['optimizer']
                    epochs = int(training_config['epochs'])
                    batch_size = int(training_config['batch_size'])
                    num_of_weeks = int(training_config['num_of_weeks'])
                    num_of_days = int(training_config['num_of_days'])
                    num_of_hours = int(training_config['num_of_hours'])
                    merge = bool(int(training_config['merge']))
                    seed = int(training_config['seed']) if 'seed' in training_config else 1

                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                    if ctx.startswith('cpu'):
                        ctx = torch.device('cpu')
                    elif ctx.startswith('gpu'):
                        gpu_index = int(ctx[ctx.index('-') + 1:])
                        ctx = torch.device('cuda:%s' % gpu_index if torch.cuda.is_available() else 'cpu')

                    print('Model is %s' % (model_name))
                    if model_name == 'MSTGCN':
                        from model.mstgcn import MSTGCN as model
                    elif model_name == 'ASTGCN':
                        from model.astgcn import ASTGCN as model
                    else:
                        raise SystemExit('Wrong type of model!')

                    run_tag = '%s_%s_%s' % (spatial_mode, temporal_mode, timestamp)
                    group_tag = '%s_lr%s' % (model_name, _format_lr_tag(learning_rate))
                    if 'params_dir' in training_config and training_config['params_dir'] != "None" and training_config['params_dir'] != "":
                        params_path = os.path.join(training_config['params_dir'], group_tag, run_tag)
                    else:
                        params_path = os.path.join('results', group_tag, run_tag)

                    if os.path.exists(params_path) and not args.force:
                        raise SystemExit("Params folder exists! Select a new params path please!")
                    else:
                        if os.path.exists(params_path):
                            shutil.rmtree(params_path)
                        os.makedirs(params_path)
                        print('Create params directory %s' % (params_path))

                    if base_prediction_name:
                        training_config['prediction_filename'] = '%s_%s_%s' % (base_prediction_name, group_tag, run_tag)

                    run_training_block(params_path, model_name, model, ctx, optimizer, learning_rate,
                                       epochs, batch_size, num_of_weeks, num_of_days, num_of_hours,
                                       merge, seed, timestamp, spatial_mode, temporal_mode,
                                       adaptive_graph_cfg, transformer_cfg, horizons)