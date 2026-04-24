import torch
from torch import nn
import torch.nn.functional as F
import math

def _causal_mask(length, device):
    """生成下三角掩码矩阵，防止利用未来信息预测"""
    mask = torch.triu(
        torch.ones(length, length, device=device), diagonal=1
    )  # 1. 生成上三角全1矩阵（主对角线以上为1，对角线及以下为0）
    mask = mask.masked_fill(
        mask == 1, float("-inf")
    )  # 2. 把上三角的1替换为负无穷，softmax后会变成0，屏蔽未来位置
    return mask 

class ProbSparseAttention(nn.Module):
    """真实的 Informer 概率稀疏注意力核心"""
    def __init__(self, d_model, n_heads, factor=5, dropout=0.0):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model # 输入特征维度
        self.n_heads = n_heads # 注意力头数
        self.head_dim = d_model // n_heads # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子 1/√d
        self.factor = factor # 控制稀疏程度

        # 三个线性层，生成 Q K V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        Q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        K = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # ProbSparse核心: 动态计算采样数量与 Top-u 的限制
        # 计算采样数量：U_part = factor * log(L)，但不超过 L 本身
        # U_part = self.factor * int(math.ceil(math.log(L))) 
        # u = min(U_part, L)
        # sample_k = min(U_part, L)

        # [修复] 增加对极短序列 (L=1) 的兜底保护，防止 log(1)=0 导致崩溃
        L_safe = max(L, 2)  # 强制 L 至少按 2 计算对数
        U_part = self.factor * int(math.ceil(math.log(L_safe))) 
        
        # 确保采样数量和 top-u 至少为 1，且不超过序列总长 L
        u = max(1, min(U_part, L))
        sample_k = max(1, min(U_part, L))
        

        # 稀疏性评估：从 Key 中随机选取少量子集
        K_sample = K[:, :, torch.randperm(L)[:sample_k], :]

        # 计算 Q 和 采样K 的内积 【B，H，L，D】×【B，H，sample_k，D】 -> 【B，H，L，sample_k】
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) * self.scale

        # 稀疏性测量指标：M(q_i) = max(q_i * K) - mean(q_i * K)
        # 【B，H，L，sample_k】 -> max 和 mean 都在最后一个维度上计算，得到【B，H，L】
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)

        # 选出具有显著特征突变（拥堵等）的 Top-u 个活跃 Query
        # M_top: (B, H, u) 包含value 值和被选中的 Query 索引
        M_top = M.topk(u, dim=-1)
        # Q:  (B, H, L, D) -> Q_reduce: (B, H, u, D) 仅保留被选中的 Query
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], M_top.indices, :] 

        # 仅对激活的 Query 计算完整的 Attention
        # Q_reduce: (B, H, u, D) -> Q_K: (B, H, u, L) 仅保留被选中的 Query 与 全部 Key 的内积
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) * self.scale

        if attn_mask is not None: 
            # Mask 进行广播并筛选，保持对应 
            # M_top.indices 的维度是 [B, H, u]
            reduced_mask = attn_mask[M_top.indices, :]  # (B, H, u, L)
            # [修复维度] Q_K = Q_K + reduced_mask.unsqueeze(0).unsqueeze(0) 
            Q_K = Q_K + reduced_mask

        # Q_K: (B, H, u, L) -> attn: (B, H, u, L) 仅保留被选中的 Query 与 全部 Key 的注意力权重
        attn = torch.softmax(Q_K, dim=-1)
        attn = self.dropout(attn)
        out_reduce = torch.matmul(attn, V) # (B, H, u, D)

        # 对于其余平稳时间段的未被选中的 Query，直接使用 V 的 mean 进行冗余填充
        # V.mean → (B, H, 1, D)
        # expand → (B, H, L, D)
        out = V.mean(dim=-2, keepdim=True).expand(B, self.n_heads, L, self.head_dim).clone()
        # 填回算好的真实值
        out[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], M_top.indices, :] = out_reduce
        # 【B，H，L，D】 -> 【B，L，d_model】
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class DistillingLayer(nn.Module):
    """时间蒸馏层：利用池化成倍压缩长序列时间长度，提炼宏观大尺度特征
    每经过一次蒸馏，序列长度减半，但特征维度不变，模型能更快捕捉长时序的全局规律"""
    def __init__(self, d_model):
        super(DistillingLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # 最大池化：步长为2，时间长度减半
        self.activation = nn.ELU()

    def forward(self, x):
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2) # Conv1d 需要 (B, 通道，长度) 输入格式
        # 卷积+—激活+池化
        x = self.pool(self.activation(self.conv(x)))
        # -> 返回 (B, L/2, D)
        return x.transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, e_layers, dropout, max_len=512, factor=5):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model) # 输入特征维度映射到 Transformer 的 d_model 维度
        self.pos_emb = nn.Parameter(torch.empty(1, max_len, d_model)) # 可学习位置编码：时序数据必须加入位置信息
        nn.init.xavier_uniform_(self.pos_emb) 
        self.temporal_proj = nn.Linear(2, d_model) # 时间标签映射到 d_model 维度
        # 可学习的缩放参数：调整位置/时间编码的权重
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.temporal_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        
        # 堆叠编码器层+蒸馏层
        self.layers = nn.ModuleList()
        self.distils = nn.ModuleList()
        
        for i in range(e_layers):
            # 单层编码器：因果卷积 + 稀疏注意力 + FFN + 三层归一化
            self.layers.append(nn.ModuleDict({
                'causal_conv': nn.Conv1d(d_model, d_model, kernel_size=3, padding=0),
                'attn': ProbSparseAttention(d_model, n_heads, factor=factor, dropout=dropout),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
            'norm1': nn.LayerNorm(d_model), # 卷积残差归一化
            'norm2': nn.LayerNorm(d_model), # 注意力残差归一化
            'norm3': nn.LayerNorm(d_model), # FFN残差归一化
            }))
            
            # 在每层之后均插入蒸馏模块 (最后一层除外)
            if i < e_layers - 1:
                self.distils.append(DistillingLayer(d_model))
            else:
                self.distils.append(None)

    def forward(self, x, time_features=None):
        B, L, input_dim = x.shape
        pos = self.pos_emb[:, :L, :] # 从最大长度的可学习位置编码中，动态截取当前输入序列长度L的部分
        x = self.input_proj(x) # (B, L, input_dim) -> (B, L, d_model)
        x = x + self.pos_scale * pos # (B, L, d_model) 位置编码*可学习的缩放系数，加到投影后的输入特征上
        
        # 补充时间特征
        if time_features is not None:
            time_embed = self.temporal_proj(time_features)
            x = x + self.temporal_scale * time_embed
            
        x = self.dropout(x)
        
        for i, layer in enumerate(self.layers):
            # x: (B, L, D)
            current_L = x.shape[1]
            # 动态更新 Mask，因为随后的蒸馏操作会使得时间步 L 缩小一半
            attn_mask = _causal_mask(current_L, x.device)
            
            # 1. 因果一维卷积 (严格防止未来信息泄露)
            conv_in = x.transpose(1, 2) # (B, D, L)
            conv_in = F.pad(conv_in, (2, 0)) 
            conv_out = layer['causal_conv'](conv_in).transpose(1, 2)
            x = layer['norm1'](x + conv_out)
            
            # 2. 核心概率自注意力模块
            attn_out = layer['attn'](x, attn_mask)
            x = layer['norm2'](x + attn_out)
            
            # 3. 前馈神经网络 FFN (带 Norm3 修复)
            ffn_out = layer['ffn'](x)
            x = layer['norm3'](x + ffn_out)
            
            # 4. 时间步成倍蒸馏降维
            if self.distils[i] is not None:
                x = self.distils[i](x)
                
        return x
