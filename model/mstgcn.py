# -*- coding:utf-8 -*-
# pylint: disable=no-member

import torch  # MXNet:from mxnet import nd → PyTorch:import torch
from torch import nn  # MXNet:from mxnet.gluon import nn → PyTorch:from torch import nn
import torch.nn.functional as F  # MXNet:nd.relu → PyTorch:F.relu


class cheb_conv(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, num_of_filters, K, cheb_polynomials, **kwargs):
        '''
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def _lazy_theta(self, shape, x):
        if self.Theta is None:
            theta = torch.empty(*shape, device=x.device, dtype=x.dtype)
            nn.init.xavier_uniform_(theta)
            self.Theta = nn.Parameter(theta)

    def forward(self, x):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix,
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        self._lazy_theta((self.K, num_of_features, self.num_of_filters), x)  # MXNet:deferred_init → PyTorch:lazy init

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]

            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device, dtype=x.dtype)  # MXNet:nd.zeros(ctx=...) → PyTorch:torch.zeros(device=...)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                theta_k = self.Theta[k]  # MXNet:param.data()[k] → PyTorch:Parameter slicing
                rhs = torch.matmul(graph_signal.permute(0, 2, 1), T_k).permute(0, 2, 1)  # MXNet:nd.dot+transpose → PyTorch:matmul+permute
                output = output + torch.matmul(rhs, theta_k)  # MXNet:nd.dot → PyTorch:torch.matmul
            outputs.append(output.unsqueeze(-1))  # MXNet:expand_dims → PyTorch:unsqueeze

        return F.relu(torch.cat(outputs, dim=-1))  # MXNet:nd.concat/nd.relu → PyTorch:torch.cat/F.relu


class MSTGCN_block(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    def __init__(self, backbone, **kwargs):
        '''
        Parameters
        ----------
        backbone: dict, should have 5 keys
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_strides",
                        "cheb_polynomials"
        '''
        super(MSTGCN_block, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.cheb_conv = cheb_conv(num_of_filters=num_of_chev_filters,
                                   K=K,
                                   cheb_polynomials=cheb_polynomials)
        self.time_conv = nn.Conv2d(  # MXNet:nn.Conv2D → PyTorch:nn.Conv2d
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, time_conv_strides)
        )
        self.residual_conv = nn.LazyConv2d(  # MXNet:Conv2D deferred infer channels → PyTorch:LazyConv2d
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides)
        )
        self.ln = nn.LayerNorm(num_of_time_filters)  # MXNet:nn.LayerNorm(axis=2) → PyTorch:nn.LayerNorm(C) with permute

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        '''

        # cheb gcn
        spatial_gcn = self.cheb_conv(x)

        # convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # MXNet:transpose/Conv2D → PyTorch:permute/Conv2d

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # MXNet:transpose/Conv2D → PyTorch:permute/Conv2d

        output = F.relu(x_residual + time_conv_output)  # MXNet:nd.relu → PyTorch:F.relu
        return self.ln(output.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # MXNet:LayerNorm(axis=2) → PyTorch:permute+LayerNorm


class MSTGCN_submodule(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    a module in MSTGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(MSTGCN_submodule, self).__init__()

        self.blocks = nn.ModuleList()
        for backbone in backbones:
            self.blocks.append(MSTGCN_block(backbone))
        self.final_conv = nn.LazyConv2d(  # MXNet:Conv2D deferred infer channels → PyTorch:LazyConv2d
            out_channels=num_for_prediction,
            kernel_size=(1, backbones[-1]['num_of_time_filters'])
        )
        self.W = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, num_for_prediction)

        '''

        for block in self.blocks:
            x = block(x)
        module_output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)  # MXNet:transpose/Conv2D → PyTorch:permute/Conv2d
        _, num_of_vertices, num_for_prediction = module_output.shape
        if self.W is None:
            weight = torch.empty(num_of_vertices, num_for_prediction, device=module_output.device, dtype=module_output.dtype)
            nn.init.xavier_uniform_(weight)
            self.W = nn.Parameter(weight)  # MXNet:deferred_init param W → PyTorch:lazy nn.Parameter
        return module_output * self.W


class MSTGCN(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    MSTGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules
        '''
        super(MSTGCN, self).__init__()
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.num_for_prediction = num_for_prediction

        self.submodules = nn.ModuleList()
        for backbones in all_backbones:
            self.submodules.append(MSTGCN_submodule(num_for_prediction, backbones))  # MXNet:register_child → PyTorch:ModuleList

    def forward(self, x_list):
        '''
        Parameters
        ----------
        x_list: list[mx.ndarray],
                shape is (batch_size, num_of_vertices,
                          num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: mx.ndarray,
               shape is (batch_size, num_of_vertices, num_for_prediction)

        '''
        if len(x_list) != len(self.submodules):
            raise ValueError("num of submodule not equals to "
                             "length of the input list")

        num_of_vertices_set = {i.shape[1] for i in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! "
                             "Check if your input data have same "
                             "size on axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return torch.stack(submodule_outputs, dim=0).sum(dim=0)  # MXNet:nd.add_n → PyTorch:stack+sum
