# -*- coding:utf-8 -*-

import configparser
import warnings
import torch  # MXNet:from mxnet import nd → PyTorch:import torch

from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix


def _build_backbone_pair(K, cheb_polynomials, time_conv_strides):
    """构建一对 backbone 字典（stride=N 和 stride=1）。"""
    return [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": time_conv_strides,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]


def get_backbones_from_config(data_cfg, training_cfg, ctx):
    """
    从已解析的 DataConfig / TrainingConfig 构建 backbones。

    Parameters
    ----------
    data_cfg : lib.config.DataConfig
    training_cfg : lib.config.TrainingConfig
    ctx : torch.device

    Returns
    -------
    list[list[dict]]
    """
    adj_mx = get_adjacency_matrix(data_cfg.adj_filename, data_cfg.num_of_vertices)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polys = [
        torch.tensor(i, dtype=torch.float32, device=ctx)
        for i in cheb_polynomial(L_tilde, training_cfg.K)
    ]
    return [
        _build_backbone_pair(training_cfg.K, cheb_polys, training_cfg.num_of_weeks),
        _build_backbone_pair(training_cfg.K, cheb_polys, training_cfg.num_of_days),
        _build_backbone_pair(training_cfg.K, cheb_polys, training_cfg.num_of_hours),
    ]


def get_backbones(config_filename, adj_filename, ctx):
    """向后兼容接口，推荐使用 get_backbones_from_config。"""
    warnings.warn(
        "get_backbones() is deprecated, use get_backbones_from_config() instead.",
        DeprecationWarning, stacklevel=2
    )
    config = configparser.ConfigParser()
    config.read(config_filename)

    K = int(config['Training']['K'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_hours = int(config['Training']['num_of_hours'])
    num_of_vertices = int(config['Data']['num_of_vertices'])

    adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polys = [
        torch.tensor(i, dtype=torch.float32, device=ctx)
        for i in cheb_polynomial(L_tilde, K)
    ]
    return [
        _build_backbone_pair(K, cheb_polys, num_of_weeks),
        _build_backbone_pair(K, cheb_polys, num_of_days),
        _build_backbone_pair(K, cheb_polys, num_of_hours),
    ]
