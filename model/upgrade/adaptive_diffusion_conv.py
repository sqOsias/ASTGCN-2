import torch
from torch import nn
import torch.nn.functional as F

class AdaptiveDiffusionConv(nn.Module):
    # 自适应扩散卷积层，支持动态调整图结构和不同阶数的邻居信息融合
    def __init__(self, in_channels, num_of_filters, K, num_of_vertices, adaptive_graph):
        super(AdaptiveDiffusionConv, self).__init__()
        self.K = K # K阶
        self.num_of_filters = num_of_filters # 输出特征维度
        self.num_of_vertices = num_of_vertices # 图中节点数量
        self.adaptive_graph = adaptive_graph # 自适应图结构学习模块

        # 修复致命错误：移除了 Lazy 的初始化，严格在 init 中注册 nn.Parameter
        self.Theta = nn.Parameter(torch.empty(self.K, in_channels, self.num_of_filters))
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x, spatial_attention):
        # print("AdaptiveDiffusionConv输入数据 x 的形状是：", x.shape)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        adj = (
            self.adaptive_graph()
        )  # # 自适应学习得到的邻接矩阵，形状为 (N, N)，表示每个节点与其他节点的关系强度
        if spatial_attention is not None:
            if spatial_attention.dim() == 3:
                # 如果 spatial_attention 已经是 (B，N，N)
                # 让底图乘上“空间注意力评分”
                # 物理意义：在底图的基础上，根据此时此刻的交通拥堵情况，动态修正管道的粗细
                adj = adj.unsqueeze(0) * spatial_attention
            else:
                # 如果spatial_attention 是 (num_of_vertices, num_of_vertices)，则扩展到批次
                adj = adj.unsqueeze(0) * spatial_attention.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
        else:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # 在空间维度上，为模型划定“感受野”
        # 扩散支撑矩阵列表 supports 包含了从 0 阶到 K-1 阶的扩散矩阵，分别对应不同阶数的邻居信息
        eye = (
            torch.eye(num_of_vertices, device=x.device, dtype=x.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        supports = [eye, adj]
        for _ in range(2, self.K):
            supports.append(torch.matmul(supports[-1], adj))

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(
                (batch_size, num_of_vertices, self.num_of_filters),
                device=x.device,
                dtype=x.dtype,
            )
            for k in range(self.K):
                theta_k = self.Theta[k]
                rhs = torch.bmm(supports[k].permute(0, 2, 1), graph_signal)
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))
