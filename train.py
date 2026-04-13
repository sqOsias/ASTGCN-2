# -*- coding:utf-8 -*-

import os
import shutil
from time import time
from datetime import datetime
import argparse
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from lib.config import (
    load_config, get_model_names, format_lr_tag,
    ExperimentConfig, _parse_list,
)
from lib.experiment import ExperimentManager, build_params_path, ensure_params_dir
from lib.logger import get_logger, TBWriter
from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from model.model_config import get_backbones_from_config


# ---------------------------------------------------------------------------
# 权重初始化
# ---------------------------------------------------------------------------

class WeightInitializer:
    @staticmethod
    def init_weight(name, data, logger=None):
        if len(data.shape) < 2:
            nn.init.uniform_(data)
            if logger:
                logger.info('Init %s %s with Uniform', name, tuple(data.shape))
        else:
            nn.init.xavier_uniform_(data)
            if logger:
                logger.info('Init %s %s with Xavier', name, tuple(data.shape))


# ---------------------------------------------------------------------------
# 解析设备字符串
# ---------------------------------------------------------------------------

def _resolve_device(ctx_str: str) -> torch.device:
    if ctx_str.startswith('cpu'):
        return torch.device('cpu')
    elif ctx_str.startswith('gpu'):
        gpu_index = int(ctx_str[ctx_str.index('-') + 1:])
        return torch.device('cuda:%s' % gpu_index if torch.cuda.is_available() else 'cpu')
    return torch.device(ctx_str)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _prepare_data(cfg: ExperimentConfig, logger):
    """读取数据并返回 DataLoader、统计信息、测试真值等。"""
    data_cfg = cfg.data
    train_cfg = cfg.training

    logger.info("Reading data...")
    all_data = read_and_generate_dataset(
        data_cfg.graph_signal_matrix_filename,
        train_cfg.num_of_weeks, train_cfg.num_of_days, train_cfg.num_of_hours,
        data_cfg.num_for_predict, data_cfg.points_per_hour, train_cfg.merge,
    )

    # 提取统计信息
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean'].copy()
        stats_data[type_ + '_std'] = stats['std'].copy()

    num_of_features = all_data['train']['recent'].shape[-2]

    # 构建 tensor
    def _to_tensors(split):
        return (
            torch.from_numpy(split['week']).float(),
            torch.from_numpy(split['day']).float(),
            torch.from_numpy(split['recent']).float(),
            torch.from_numpy(split['target']).float(),
        )

    train_tensors = _to_tensors(all_data['train'])
    val_tensors = _to_tensors(all_data['val'])
    test_tensors = _to_tensors(all_data['test'])

    test_target_np = all_data['test']['target']
    true_value = (test_target_np.transpose((0, 2, 1))
                  .reshape(test_target_np.shape[0], -1))

    del all_data

    bs = train_cfg.batch_size
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(*test_tensors), batch_size=bs, shuffle=False)

    return (train_loader, val_loader, test_loader,
            stats_data, num_of_features, test_target_np, true_value)


# ---------------------------------------------------------------------------
# 构建模型
# ---------------------------------------------------------------------------

def _build_model(model_name, cfg, all_backbones, num_of_features,
                 spatial_mode, temporal_mode, ctx, logger):
    data_cfg = cfg.data
    ag_dict = asdict(cfg.upgrade.adaptive_graph)
    tf_dict = asdict(cfg.upgrade.transformer)

    if model_name == 'MSTGCN' and (spatial_mode != 0 or temporal_mode != 0):
        raise SystemExit('Upgrade modes only supported for ASTGCN')

    if spatial_mode == 0 and temporal_mode == 0:
        if model_name == 'MSTGCN':
            from model.mstgcn import MSTGCN as ModelClass
        elif model_name == 'ASTGCN':
            from model.astgcn import ASTGCN as ModelClass
        else:
            raise SystemExit('Wrong type of model!')
        net = ModelClass(data_cfg.num_for_predict, all_backbones).to(ctx)
    else:
        from model.upgrade.astgcn_upgrade import UpgradeASTGCN
        net = UpgradeASTGCN(
            num_of_features, data_cfg.num_for_predict, all_backbones,
            data_cfg.num_of_vertices, spatial_mode, temporal_mode,
            ag_dict, tf_dict,
        ).to(ctx)

    logger.info('Model is %s (spatial=%d, temporal=%d)', model_name, spatial_mode, temporal_mode)
    return net


# ---------------------------------------------------------------------------
# 构建优化器
# ---------------------------------------------------------------------------

def _build_optimizer(net, optimizer_name: str, lr: float):
    name = optimizer_name.lower()
    if name == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(net.parameters(), lr=lr)
    raise SystemExit('Unsupported optimizer: %s' % optimizer_name)


# ---------------------------------------------------------------------------
# 核心训练循环
# ---------------------------------------------------------------------------

def run_training(cfg: ExperimentConfig, params_path: str,
                 model_name: str, learning_rate: float,
                 spatial_mode: int, temporal_mode: int,
                 timestamp: str, force: bool):
    """单次训练运行。"""
    train_cfg = cfg.training
    data_cfg = cfg.data

    # 随机种子
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ctx = _resolve_device(train_cfg.ctx)
    run_id = os.path.basename(os.path.normpath(params_path))

    # 实验管理器
    exp = ExperimentManager(
        params_path=params_path, config=cfg, run_id=run_id,
        timestamp=timestamp, spatial_mode=spatial_mode,
        temporal_mode=temporal_mode, device=ctx, force=force,
    )
    exp.setup()

    # 日志
    logger = get_logger('ASTGCN', log_dir=exp.logs_dir)
    tb = TBWriter(log_dir=exp.logs_dir)
    logger.info('Create params directory %s', params_path)

    # 数据
    (train_loader, val_loader, test_loader,
     stats_data, num_of_features, test_target_np, true_value) = _prepare_data(cfg, logger)
    exp.save_stats(stats_data)

    if str(ctx).startswith('cuda'):
        logger.info("Using GPU training...")

    # 模型
    all_backbones = get_backbones_from_config(data_cfg, train_cfg, ctx)
    net = _build_model(model_name, cfg, all_backbones, num_of_features,
                       spatial_mode, temporal_mode, ctx, logger)

    # 前向一次以触发 lazy init
    for val_w, val_d, val_r, val_t in val_loader:
        val_w, val_d, val_r = val_w.to(ctx), val_d.to(ctx), val_r.to(ctx)
        net([val_w, val_d, val_r])
        break
    for name, param in net.named_parameters():
        WeightInitializer.init_weight(name, param.data, logger)

    trainer = _build_optimizer(net, train_cfg.optimizer, learning_rate)
    loss_function = nn.MSELoss(reduction='none')

    # 计算评估 horizon
    pph = data_cfg.points_per_hour
    horizons = [h for h in [3, pph, pph * 3] if h <= data_cfg.num_for_predict]

    # epoch 0 基线评估
    val_loss = compute_val_loss(net, val_loader, loss_function, tb.sw, epoch=0)
    exp.log_val_loss(0, val_loss)
    test_metrics, _ = evaluate(net, test_loader, true_value,
                               data_cfg.num_of_vertices, tb.sw, epoch=0, horizons=horizons)
    exp.log_test_metrics(0, test_metrics, horizons)

    # 训练循环
    best_val_loss = float('inf')
    if ctx.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(ctx)
    train_start_time = time()
    global_step = 1

    for epoch in range(1, train_cfg.epochs + 1):
        net.train()
        for train_w, train_d, train_r, train_t in train_loader:
            print("近期输入数据 train_r 的形状是：", train_r.shape)
            print("真实标签 train_t 的形状是：", train_t.shape)
            start_time = time()
            train_w = train_w.to(ctx)
            train_d = train_d.to(ctx)
            train_r = train_r.to(ctx)
            train_t = train_t.to(ctx)

            with torch.set_grad_enabled(True):
                output = net([train_w, train_d, train_r])
                l = 0.5 * loss_function(output, train_t)
            trainer.zero_grad()
            loss_scalar = l.mean()
            loss_scalar.backward()
            trainer.step()
            training_loss = loss_scalar.item()

            tb.add_scalar('training_loss', training_loss, global_step)

            if global_step % 100 == 0:
                logger.info('global step: %s, training loss: %.2f, time: %.2fs',
                            global_step, training_loss, time() - start_time)
            exp.log_train_loss(epoch, global_step, training_loss, time() - start_time)
            global_step += 1

        # 梯度直方图
        tb.log_gradients(net, global_step, logger)

        # 验证
        val_loss = compute_val_loss(net, val_loader, loss_function, tb.sw, epoch)
        exp.log_val_loss(epoch, val_loss)


        # todo 是在每个epoch里面验证和测试吗
        # 测试
        test_metrics, _ = evaluate(net, test_loader, true_value,
                                   data_cfg.num_of_vertices, tb.sw, epoch, horizons=horizons)
        exp.log_test_metrics(epoch, test_metrics, horizons)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            prediction = predict(net, test_loader)
            exp.save_best_model(net, epoch, best_val_loss, prediction, test_target_np)
            logger.info('save best model to file: %s', exp.best_model_path)

    # 运行统计
    train_elapsed = time() - train_start_time
    gpu_peak = int(torch.cuda.max_memory_allocated(ctx)) if ctx.type == 'cuda' else 0
    exp.save_runtime(train_elapsed, gpu_peak, model_name, learning_rate)

    tb.close()
    exp.final_cleanup()

    # 复制预测文件
    if train_cfg.prediction_filename:
        pred_name = '%s_%s_lr%s_%s_%s_%s' % (
            train_cfg.prediction_filename, model_name,
            format_lr_tag(learning_rate), spatial_mode, temporal_mode, timestamp,
        )
        exp.copy_prediction_file(pred_name)

    logger.info('Training finished. Elapsed: %.1fs, GPU peak: %d bytes',
                train_elapsed, gpu_peak)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configurations/PEMS04.conf",
                        help="configuration file path", required=False)
    parser.add_argument("--force", type=str, default=False,
                        help="remove params dir", required=False)
    args = parser.parse_args()

    # 清理全局日志目录
    if os.path.exists('logs'):
        shutil.rmtree('logs')

    # 加载配置
    cfg = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    model_names = get_model_names(cfg.training)
    learning_rates = [float(v) for v in _parse_list(str(cfg.training.learning_rate))]

    for model_name in model_names:
        for lr in learning_rates:
            for spatial_mode in cfg.upgrade.spatial_modes:
                for temporal_mode in cfg.upgrade.temporal_modes:
                    params_path = build_params_path(
                        cfg.training, model_name, lr,
                        spatial_mode, temporal_mode, timestamp,
                    )
                    ensure_params_dir(params_path, args.force)

                    run_training(
                        cfg=cfg,
                        params_path=params_path,
                        model_name=model_name,
                        learning_rate=lr,
                        spatial_mode=spatial_mode,
                        temporal_mode=temporal_mode,
                        timestamp=timestamp,
                        force=args.force,
                    )


if __name__ == "__main__":
    main()