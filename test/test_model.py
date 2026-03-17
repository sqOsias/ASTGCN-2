# -*- coding:utf-8 -*-

import sys
import pytest
import numpy as np
import torch  # MXNet:from mxnet import nd → PyTorch:import torch

sys.path.append('.')


def test_ASTGCN_submodule():
    from model.astgcn import ASTGCN_submodule
    x = torch.rand(size=(32, 307, 3, 24), dtype=torch.float32)  # MXNet:nd.random_uniform → PyTorch:torch.rand
    K = 3
    cheb_polynomials = [torch.rand(size=(307, 307), dtype=torch.float32) for i in range(K)]  # MXNet:nd.random_uniform → PyTorch:torch.rand
    backbone = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 2,
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
    net = ASTGCN_submodule(12, backbone)
    output = net(x)
    assert output.shape == (32, 307, 12)
    assert output.mean().dtype == torch.float32  # MXNet:asscalar() → PyTorch:tensor dtype


def test_predict1():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    ctx = torch.device('cpu')  # MXNet:mx.cpu() → PyTorch:torch.device('cpu')
    all_backbones = get_backbones('configurations/PEMS04.conf',
                                  'data/PEMS04/distance.csv', ctx)

    net = ASTGCN(12, all_backbones)
    test_w = torch.rand(size=(16, 307, 3, 12), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    test_d = torch.rand(size=(16, 307, 3, 12), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    test_r = torch.rand(size=(16, 307, 3, 36), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    output = net([test_w, test_d, test_r])
    assert output.shape == (16, 307, 12)
    assert output.mean().dtype == torch.float32  # MXNet:asscalar() → PyTorch:tensor dtype


def test_predict2():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    ctx = torch.device('cpu')  # MXNet:mx.cpu() → PyTorch:torch.device('cpu')
    all_backbones = get_backbones('configurations/PEMS08.conf',
                                  'data/PEMS08/distance.csv', ctx)

    net = ASTGCN(12, all_backbones)
    test_w = torch.rand(size=(8, 170, 3, 12), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    test_d = torch.rand(size=(8, 170, 3, 12), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    test_r = torch.rand(size=(8, 170, 3, 36), dtype=torch.float32, device=ctx)  # MXNet:nd.random_uniform(ctx=ctx) → PyTorch:torch.rand(device=ctx)
    output = net([test_w, test_d, test_r])
    assert output.shape == (8, 170, 12)
    assert output.mean().dtype == torch.float32  # MXNet:asscalar() → PyTorch:tensor dtype
