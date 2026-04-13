# -*- coding:utf-8 -*-
# pylint: disable=no-member

import torch  # MXNet:from mxnet import nd → PyTorch:import torch
from torch import nn  # MXNet:from mxnet.gluon import nn → PyTorch:from torch import nn
import torch.nn.functional as F  # MXNet:nd.relu/nd.sigmoid → PyTorch:F.relu/F.sigmoid


class Spatial_Attention_layer(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    compute spatial attention scores
    '''
    def __init__(self, **kwargs):
        super(Spatial_Attention_layer, self).__init__()
        self.W_1 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.W_2 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.W_3 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.b_s = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.V_s = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def _lazy_parameter(self, name, shape, x):
        param = getattr(self, name)
        if param is None:
            tensor = torch.empty(*shape, device=x.device, dtype=x.dtype)
            if len(shape) < 2:
                nn.init.uniform_(tensor)
            else:
                nn.init.xavier_uniform_(tensor)
            setattr(self, name, nn.Parameter(tensor))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)

        '''
        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer the shape of params
        self._lazy_parameter('W_1', (num_of_timesteps,), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('W_2', (num_of_features, num_of_timesteps), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('W_3', (num_of_features,), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('b_s', (1, num_of_vertices, num_of_vertices), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('V_s', (num_of_vertices, num_of_vertices), x)  # MXNet:deferred_init → PyTorch:lazy init

        # compute spatial attention scores
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)  # MXNet:nd.dot → PyTorch:torch.matmul

        # shape of rhs is (batch_size, T, V)
        rhs = torch.tensordot(self.W_3, x.permute(2, 0, 3, 1), dims=([0], [0]))  # MXNet:nd.dot(vector,4D) → PyTorch:torch.tensordot

        # shape of product is (batch_size, V, V)
        product = torch.bmm(lhs, rhs)  # MXNet:nd.batch_dot → PyTorch:torch.bmm

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s).permute(1, 2, 0)).permute(2, 0, 1)  # MXNet:nd.dot/sigmoid/transpose → PyTorch:matmul/sigmoid/permute

        # normalization
        S = S - torch.max(S, dim=1, keepdim=True).values  # MXNet:nd.max(axis,keepdims) → PyTorch:torch.max(dim,keepdim)
        exp = torch.exp(S)  # MXNet:nd.exp → PyTorch:torch.exp
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)  # MXNet:nd.sum(axis,keepdims) → PyTorch:torch.sum(dim,keepdim)
        return S_normalized


class cheb_conv_with_SAt(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    K-order chebyshev graph convolution with Spatial Attention scores
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
        super(cheb_conv_with_SAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters  # 输出的特征通道数
        self.cheb_polynomials = cheb_polynomials # 预先计算好的切比雪夫多项式列表，长度为 K ,这些矩阵定义了图的拓扑结构（如邻接关系）
        # 图卷积层的可学习权重矩阵
        self.Theta = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def _lazy_theta(self, shape, x):
        if self.Theta is None:
            theta = torch.empty(*shape, device=x.device, dtype=x.dtype)
            nn.init.xavier_uniform_(theta)
            self.Theta = nn.Parameter(theta)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: mx.ndarray, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: mx.ndarray, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        '''
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape

        self._lazy_theta((self.K, num_of_features, self.num_of_filters), x)  # MXNet:deferred_init → PyTorch:lazy init

        outputs = []
        for time_step in range(num_of_timesteps):
            # shape is (B N C T) → (B N C) 取出每个时间步的图信号
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device, dtype=x.dtype)  # MXNet:nd.zeros(ctx=...) → PyTorch:torch.zeros(device=...)
            for k in range(self.K):

                # shape of T_k is (N, N)
                T_k = self.cheb_polynomials[k]

                # shape of T_k_with_at is (batch_size, N, N)
                # T_k表示k阶切比雪夫多项式，定义了图中节点的静态 k 阶邻居关系
                # spatial_attention表示空间注意力分数，反映了此时时刻节点间的实时关联。
                # 用注意力分数（动态权重）对切比雪夫矩阵（静态邻居）进行了缩放
                # 表示在物理上的 k 阶邻居中，哪些节点对当前路段的影响力在此时此刻是最大的
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (N, num_of_filters)
                # # theta_k 是第 k 阶卷积核的可学习权重参数矩阵（形状 [F, num_filters]）。
                # 它负责将聚合后的原始特征映射到更高维的特征空间。      
                theta_k = self.Theta[k]  

                # shape is (batch_size, N, F)
                # T_k_with_at表示动态权重矩阵
                # graph_signal 表示输入的图信号矩阵，形状为 (batch_size, N, F)
                # 对于每个路口，根据动态邻居权重，把周围路口在此时刻的交通特征（如车速、流量）吸收到自己身上
                rhs = torch.bmm(T_k_with_at.permute(0, 2, 1), graph_signal) 

                # 将吸收了空间信息的聚合特征 [B, N, F] 通过权重矩阵进行线性变换，映射到输出通道数
                # 输入：（B, N, F）× （F, num_filters）
                # 输出：（B, N, num_filters）
                output = output + torch.matmul(rhs, theta_k)  

            outputs.append(output.unsqueeze(-1))  #
        return F.relu(torch.cat(outputs, dim=-1))  # 


class Temporal_Attention_layer(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    compute temporal attention scores
    '''
    def __init__(self, **kwargs):
        super(Temporal_Attention_layer, self).__init__()
        self.U_1 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.U_2 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.U_3 = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.b_e = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter
        self.V_e = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def _lazy_parameter(self, name, shape, x):
        param = getattr(self, name)
        if param is None:
            tensor = torch.empty(*shape, device=x.device, dtype=x.dtype)
            if len(shape) < 2:
                nn.init.uniform_(tensor)
            else:
                nn.init.xavier_uniform_(tensor)
            setattr(self, name, nn.Parameter(tensor))

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})

        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # defer shape
        self._lazy_parameter('U_1', (num_of_vertices,), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('U_2', (num_of_features, num_of_vertices), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('U_3', (num_of_features,), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('b_e', (1, num_of_timesteps, num_of_timesteps), x)  # MXNet:deferred_init → PyTorch:lazy init
        self._lazy_parameter('V_e', (num_of_timesteps, num_of_timesteps), x)  # MXNet:deferred_init → PyTorch:lazy init

        # compute temporal attention scores
        # shape is (N, T, V)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1), self.U_2)  # MXNet:nd.dot+transpose → PyTorch:matmul+permute

        # shape is (N, V, T)
        rhs = torch.tensordot(self.U_3, x.permute(2, 0, 1, 3), dims=([0], [0]))  # MXNet:nd.dot(vector,4D) → PyTorch:torch.tensordot

        product = torch.bmm(lhs, rhs)  # MXNet:nd.batch_dot → PyTorch:torch.bmm

        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e).permute(1, 2, 0)).permute(2, 0, 1)  # MXNet:nd.dot/sigmoid/transpose → PyTorch:matmul/sigmoid/permute

        # normailzation
        E = E - torch.max(E, dim=1, keepdim=True).values  # MXNet:nd.max(axis,keepdims) → PyTorch:torch.max(dim,keepdim)
        exp = torch.exp(E)  # MXNet:nd.exp → PyTorch:torch.exp
        E_normalized = exp / torch.sum(exp, dim=1, keepdim=True)  # MXNet:nd.sum(axis,keepdims) → PyTorch:torch.sum(dim,keepdim)
        return E_normalized


class ASTGCN_block(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    def __init__(self, backbone, **kwargs):
        '''
        Parameters
        ----------
        backbone: dict, should have 6 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",
                        "time_conv_strides",
                        "cheb_polynomials"
        '''
        super(ASTGCN_block, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.SAt = Spatial_Attention_layer()
        self.cheb_conv_SAt = cheb_conv_with_SAt(
            num_of_filters=num_of_chev_filters,
            K=K,
            cheb_polynomials=cheb_polynomials
        )
        self.TAt = Temporal_Attention_layer()
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
        (batch_size, num_of_vertices,
         num_of_features, num_of_timesteps) = x.shape
        # shape is (batch_size, T, T)
        temporal_At = self.TAt(x)

        x_TAt = torch.bmm(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)  # MXNet:nd.batch_dot/reshape → PyTorch:bmm/reshape

        # cheb gcn with spatial attention
        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # MXNet:transpose/Conv2D → PyTorch:permute/Conv2d

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # MXNet:transpose/Conv2D → PyTorch:permute/Conv2d

        output = F.relu(x_residual + time_conv_output)  # MXNet:nd.relu → PyTorch:F.relu
        return self.ln(output.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # MXNet:LayerNorm(axis=2) → PyTorch:permute+LayerNorm


class ASTGCN_submodule(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    a module in ASTGCN
    '''
    def __init__(self, num_for_prediction, backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones

        '''
        super(ASTGCN_submodule, self).__init__()

        self.blocks = nn.ModuleList()  # MXNet:nn.Sequential.add → PyTorch:nn.ModuleList.append
        for backbone in backbones:
            self.blocks.append(ASTGCN_block(backbone))  # MXNet:Sequential.add → PyTorch:ModuleList.append

        self.final_conv = nn.LazyConv2d(  # MXNet:Conv2D deferred infer channels → PyTorch:LazyConv2d
            out_channels=num_for_prediction,
            kernel_size=(1, backbones[-1]['num_of_time_filters'])
        )
        self.W = None  # MXNet:self.params.get(...allow_deferred_init=True) → PyTorch:lazy nn.Parameter

    def forward(self, x):
        '''
        Parameters
        ----------
        x: mx.ndarray,
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


class ASTGCN(nn.Module):  # MXNet:nn.Block → PyTorch:nn.Module
    '''
    ASTGCN, 3 sub-modules, for hour, day, week respectively
    '''
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        '''
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list],
                       3 backbones for "hour", "day", "week" submodules
        '''
        super(ASTGCN, self).__init__()
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones "
                             "must be greater than 0")

        self.submodules = nn.ModuleList()
        for backbones in all_backbones:
            self.submodules.append(ASTGCN_submodule(num_for_prediction, backbones))  # MXNet:register_child → PyTorch:ModuleList

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
                             "Check if your input data have same size"
                             "at axis 1.")

        batch_size_set = {i.shape[0] for i in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx])
                             for idx in range(len(x_list))]

        return torch.stack(submodule_outputs, dim=0).sum(dim=0)  # MXNet:nd.add_n → PyTorch:stack+sum
