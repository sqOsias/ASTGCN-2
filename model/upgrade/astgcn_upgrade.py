import torch
from torch import nn
import torch.nn.functional as F

from model.astgcn import Spatial_Attention_layer, Temporal_Attention_layer
from model.upgrade.adaptive_graph import AdaptiveGraph
from model.upgrade.temporal_transformer import TemporalTransformer

class AdaptiveDiffusionConv(nn.Module):
    # 修复更名：自适应矩阵不严格正交，摒弃切比雪夫采用随机游走扩散卷积更数学严谨
    def __init__(self, in_channels, num_of_filters, K, num_of_vertices, adaptive_graph):
        super(AdaptiveDiffusionConv, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.num_of_vertices = num_of_vertices
        self.adaptive_graph = adaptive_graph
        
        # 修复致命错误：移除了 Lazy 的初始化，严格在 init 中注册 nn.Parameter
        self.Theta = nn.Parameter(torch.empty(self.K, in_channels, self.num_of_filters))
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x, spatial_attention):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        adj = self.adaptive_graph()
        if spatial_attention is not None:
            # 确保 spatial_attention 与 adj 形状兼容
            if spatial_attention.dim() == 3:
                # 如果 spatial_attention 已经是 (batch_size, num_of_vertices, num_of_vertices)
                adj = adj.unsqueeze(0) * spatial_attention
            else:
                # 如果 spatial_attention 是 (num_of_vertices, num_of_vertices)，则扩展到批次
                adj = adj.unsqueeze(0) * spatial_attention.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # 修复：确保 eye 矩阵也扩展到相同批次大小
        eye = torch.eye(num_of_vertices, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        supports = [eye, adj]
        for _ in range(2, self.K):
            supports.append(torch.matmul(supports[-1], adj))

        outputs =[]
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
    def __init__(self, in_channels, backbone, num_of_vertices, spatial_mode, temporal_mode,
                 adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCNBlock, self).__init__()
        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']

        self.spatial_mode = spatial_mode
        self.temporal_mode = temporal_mode

        self.SAt = Spatial_Attention_layer()
        if spatial_mode == 0:
            from model.astgcn import cheb_conv_with_SAt
            self.spatial_conv = cheb_conv_with_SAt(
                num_of_filters=num_of_chev_filters,
                K=K,
                cheb_polynomials=backbone.get("cheb_polynomials")
            )
        else:
            self.adaptive_graph = AdaptiveGraph(
                num_nodes=num_of_vertices,
                embedding_dim=adaptive_graph_cfg['embedding_dim'],
                sparse_ratio=adaptive_graph_cfg['sparse_ratio'],
                directed=adaptive_graph_cfg['directed']
            )
            # 修复：传递了明确的 in_channels 输入特征
            self.spatial_conv = AdaptiveDiffusionConv(
                in_channels=in_channels,
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
                factor=transformer_cfg.get('factor', 5)
            )
            self.transformer_out = nn.Linear(transformer_cfg['d_model'], num_of_time_filters)

        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides) if temporal_mode == 0 else 1
        )
        self.ln = nn.LayerNorm(num_of_time_filters)

    def forward(self, x):
        spatial_at = self.SAt(x)
        spatial_gcn = self.spatial_conv(x, spatial_at)

        if self.temporal_mode == 0:
            temporal_at = self.TAt(x)
            b, n, c, t = spatial_gcn.shape
            # [修复]：正确合并 Node 和 Channel 维度，保留 Time 维度在末尾进行矩阵乘法
            spatial_gcn = torch.matmul(
                spatial_gcn.reshape(b, -1, t),
                temporal_at
            ).reshape(b, n, c, t)
            time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        else:
            b, n, c, t = spatial_gcn.shape
            time_features = None
            if x.shape[2] >= 2: # 剥离时间的辅助特征维度的输入
                time_features = x[:, :, -2:, :].permute(0, 1, 3, 2).reshape(b * n, t, 2)
            
            transformer_in = spatial_gcn.permute(0, 1, 3, 2).reshape(b * n, t, c)
            transformer_out = self.transformer(transformer_in, time_features=time_features)
            transformer_out = self.transformer_out(transformer_out)
            
            t_new = transformer_out.shape[1]
            time_conv_output = transformer_out.reshape(b, n, t_new, -1).permute(0, 1, 3, 2)

        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        
        # 确保 x_residual 和 time_conv_output 形状完全匹配
        # 兼容蒸馏操作改变的时间轴维度长度
        if self.temporal_mode != 0:
            t_new = time_conv_output.shape[-1]
            if x_residual.shape[-1] != t_new:
                x_residual = F.interpolate(x_residual, size=t_new, mode='nearest')
        
        # 额外确保通道维度也匹配（防止任何可能的维度不匹配）
        if x_residual.shape[2] != time_conv_output.shape[2]:
            # 如果通道数不匹配，调整x_residual的通道数以匹配time_conv_output
            target_channels = time_conv_output.shape[2]
            current_channels = x_residual.shape[2]
            if target_channels < current_channels:
                # 如果目标通道数较小，截取前面的通道
                x_residual = x_residual[:, :, :target_channels, :]
            elif target_channels > current_channels:
                # 如果目标通道数较大，用零填充
                pad_size = target_channels - current_channels
                zeros_pad = torch.zeros(x_residual.shape[0], x_residual.shape[1], 
                                        pad_size, x_residual.shape[3], 
                                        device=x_residual.device, dtype=x_residual.dtype)
                x_residual = torch.cat([x_residual, zeros_pad], dim=2)
        
        output = F.relu(x_residual + time_conv_output)
        output = output.permute(0, 3, 1, 2)
        output = self.ln(output)
        output = output.permute(0, 2, 3, 1)
        return output


class UpgradeASTGCNSubmodule(nn.Module):
    def __init__(self, in_channels, num_for_prediction, backbones, num_of_vertices,
                 spatial_mode, temporal_mode, adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCNSubmodule, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        
        for backbone in backbones:
            self.blocks.append(UpgradeASTGCNBlock(
                in_channels=current_channels,
                backbone=backbone,
                num_of_vertices=num_of_vertices,
                spatial_mode=spatial_mode,
                temporal_mode=temporal_mode,
                adaptive_graph_cfg=adaptive_graph_cfg,
                transformer_cfg=transformer_cfg
            ))
            current_channels = backbone['num_of_time_filters']

        # 修复致命错误 RuntimeError：抛弃了僵化的 Conv2d
        # 通过 LazyLinear 在 forward 时自动推断 (C * T_剩余) 的大小，映射到目标长序列预测(如 36)，从根源解决输入维度报错
        self.final_linear = nn.LazyLinear(num_for_prediction)
        
        # 修复：提前进行 W 初始化，消灭 forward() 里声明 Parameter
        self.W = nn.Parameter(torch.empty(num_of_vertices, num_for_prediction))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        b, n, c, t = x.shape
        # 将被压缩好的时间与通道全部展平 -> (Batch, N, C_out * T_out)
        x_flat = x.reshape(b, n, c * t)
        
        # 映射到未来生成式预测
        module_output = self.final_linear(x_flat)
        
        return module_output * self.W


class UpgradeASTGCN(nn.Module):
    def __init__(self, num_of_features, num_for_prediction, all_backbones, num_of_vertices,
                 spatial_mode, temporal_mode, adaptive_graph_cfg, transformer_cfg):
        super(UpgradeASTGCN, self).__init__()
        self.submodules = nn.ModuleList()
        
        for backbones in all_backbones:
            self.submodules.append(UpgradeASTGCNSubmodule(
                in_channels=num_of_features, # 传入初始数据的特征数 (例如速度1+时间特征2 = 3维特征)
                num_for_prediction=num_for_prediction,
                backbones=backbones,
                num_of_vertices=num_of_vertices,
                spatial_mode=spatial_mode,
                temporal_mode=temporal_mode,
                adaptive_graph_cfg=adaptive_graph_cfg,
                transformer_cfg=transformer_cfg
            ))

    def forward(self, x_list):
        submodule_outputs =[]
        for idx in range(len(x_list)):
            submodule_outputs.append(self.submodules[idx](x_list[idx]))
        return torch.stack(submodule_outputs, dim=0).sum(dim=0)