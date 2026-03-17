import torch
from torch import nn
import torch.nn.functional as F

from model.astgcn import Spatial_Attention_layer, Temporal_Attention_layer
from model.upgrade.adaptive_graph import AdaptiveGraph
from model.upgrade.temporal_transformer import TemporalTransformer


class AdaptiveChebConv(nn.Module):
    def __init__(self, num_of_filters, K, num_of_vertices, adaptive_graph):
        super(AdaptiveChebConv, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.num_of_vertices = num_of_vertices
        self.adaptive_graph = adaptive_graph
        self.Theta = None

    def _lazy_theta(self, shape, x):
        if self.Theta is None:
            theta = torch.empty(*shape, device=x.device, dtype=x.dtype)
            nn.init.xavier_uniform_(theta)
            self.Theta = nn.Parameter(theta)

    def forward(self, x, spatial_attention):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        self._lazy_theta((self.K, num_of_features, self.num_of_filters), x)
        adj = self.adaptive_graph()
        if spatial_attention is not None:
            adj = adj.unsqueeze(0) * spatial_attention
        else:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        eye = torch.eye(num_of_vertices, device=x.device, dtype=x.dtype).unsqueeze(0)
        supports = [eye, adj]
        for _ in range(2, self.K):
            supports.append(torch.matmul(supports[-1], adj))

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device, dtype=x.dtype)
            for k in range(self.K):
                theta_k = self.Theta[k]
                rhs = torch.bmm(supports[k].permute(0, 2, 1), graph_signal)
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class UpgradeASTGCNBlock(nn.Module):
    def __init__(self, backbone, num_of_vertices, spatial_mode, temporal_mode,
                 adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCNBlock, self).__init__()
        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode

        self.SAt = Spatial_Attention_layer()
        if spatial_mode == 0:
            from model.astgcn import cheb_conv_with_SAt
            self.spatial_conv = cheb_conv_with_SAt(
                num_of_filters=num_of_chev_filters,
                K=K,
                cheb_polynomials=cheb_polynomials
            )
        else:
            self.adaptive_graph = AdaptiveGraph(
                num_nodes=num_of_vertices,
                embedding_dim=adaptive_graph_cfg['embedding_dim'],
                sparse_ratio=adaptive_graph_cfg['sparse_ratio'],
                directed=adaptive_graph_cfg['directed']
            )
            self.spatial_conv = AdaptiveChebConv(
                num_of_filters=num_of_chev_filters,
                K=K,
                num_of_vertices=num_of_vertices,
                adaptive_graph=self.adaptive_graph
            )

        if temporal_mode == 0:
            self.TAt = Temporal_Attention_layer()
            self.time_conv = nn.Conv2d(
                in_channels=num_of_chev_filters,
                out_channels=num_of_time_filters,
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=(1, time_conv_strides)
            )
        else:
            self.transformer = TemporalTransformer(
                input_dim=num_of_chev_filters,
                d_model=transformer_cfg['d_model'],
                n_heads=transformer_cfg['n_heads'],
                e_layers=transformer_cfg['e_layers'],
                dropout=transformer_cfg['dropout'],
                max_len=transformer_cfg['max_len'],
                topk_ratio=transformer_cfg['topk_ratio']
            )
            self.transformer_out = nn.Linear(transformer_cfg['d_model'], num_of_time_filters)

        self.residual_conv = nn.LazyConv2d(
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides)
        )
        self.ln = nn.LayerNorm(num_of_time_filters)

    def forward(self, x):
        spatial_at = self.SAt(x)
        spatial_gcn = self.spatial_conv(x, spatial_at)

        if self.temporal_mode == 0:
            temporal_at = self.TAt(x)
            spatial_gcn = torch.matmul(
                spatial_gcn.reshape(spatial_gcn.shape[0], spatial_gcn.shape[1], -1),
                temporal_at
            ).reshape(spatial_gcn.shape[0], spatial_gcn.shape[1], spatial_gcn.shape[2], -1)
            time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        else:
            b, n, c, t = spatial_gcn.shape
            time_features = None
            if x.shape[2] >= 2:
                time_features = x[:, :, -2:, :].permute(0, 1, 3, 2).reshape(b * n, t, 2)
            transformer_in = spatial_gcn.permute(0, 1, 3, 2).reshape(b * n, t, c)
            transformer_out = self.transformer(transformer_in, time_features=time_features)
            transformer_out = self.transformer_out(transformer_out)
            time_conv_output = transformer_out.reshape(b, n, t, -1).permute(0, 1, 3, 2)

        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        output = F.relu(x_residual + time_conv_output)
        output = output.permute(0, 3, 1, 2)
        output = self.ln(output)
        output = output.permute(0, 2, 3, 1)
        return output


class UpgradeASTGCNSubmodule(nn.Module):
    def __init__(self, num_for_prediction, backbones, num_of_vertices,
                 spatial_mode, temporal_mode, adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCNSubmodule, self).__init__()
        self.blocks = nn.ModuleList()
        for backbone in backbones:
            self.blocks.append(UpgradeASTGCNBlock(
                backbone=backbone,
                num_of_vertices=num_of_vertices,
                spatial_mode=spatial_mode,
                temporal_mode=temporal_mode,
                adaptive_graph_cfg=adaptive_graph_cfg,
                transformer_cfg=transformer_cfg
            ))

        self.final_conv = nn.LazyConv2d(
            out_channels=num_for_prediction,
            kernel_size=(1, backbones[-1]['num_of_time_filters'])
        )
        self.W = None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        module_output = self.final_conv(x.permute(0, 2, 1, 3))[:, :, :, -1].permute(0, 2, 3, 1)
        if self.W is None:
            self.W = nn.Parameter(torch.empty(module_output.shape[2], device=x.device, dtype=x.dtype))
            nn.init.xavier_uniform_(self.W.unsqueeze(0))
        return module_output * self.W


class UpgradeASTGCN(nn.Module):
    def __init__(self, num_for_prediction, all_backbones, num_of_vertices,
                 spatial_mode, temporal_mode, adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCN, self).__init__()
        self.submodules = nn.ModuleList()
        for backbones in all_backbones:
            self.submodules.append(UpgradeASTGCNSubmodule(
                num_for_prediction, backbones, num_of_vertices,
                spatial_mode, temporal_mode, adaptive_graph_cfg, transformer_cfg
            ))

    def forward(self, x_list):
        submodule_outputs = []
        for idx in range(len(x_list)):
            submodule_outputs.append(self.submodules[idx](x_list[idx]))
        return torch.stack(submodule_outputs, dim=0).sum(dim=0)
