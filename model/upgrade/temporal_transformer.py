import torch
from torch import nn
import torch.nn.functional as F
import math

def _causal_mask(length, device):
    """生成下三角掩码矩阵，防止利用未来信息预测"""
    mask = torch.triu(torch.ones(length, length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class ProbSparseAttention(nn.Module):
    """真实的 Informer 概率稀疏注意力核心"""
    def __init__(self, d_model, n_heads, factor=5, dropout=0.0):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.factor = factor
        
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
        U_part = self.factor * int(math.ceil(math.log(L))) 
        u = min(U_part, L)
        sample_k = min(U_part, L)
        
        # 稀疏性评估：从 Key 中随机选取少量子集
        K_sample = K[:, :, torch.randperm(L)[:sample_k], :]
        
        # 计算 Q 和 采样K 的内积
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) * self.scale
        
        # 稀疏性测量指标：M(q_i) = max(q_i * K) - mean(q_i * K)
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)
        
        # 选出具有显著特征突变（拥堵等）的 Top-u 个活跃 Query
        M_top = M.topk(u, dim=-1)
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], M_top.indices, :] 
        
        # 仅对激活的 Query 计算完整的 Attention
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            # Mask 进行广播并筛选，保持对应
            reduced_mask = attn_mask[M_top.indices, :]
            Q_K = Q_K + reduced_mask.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(Q_K, dim=-1)
        attn = self.dropout(attn)
        out_reduce = torch.matmul(attn, V) # (B, H, u, D)
        
        # 对于其余平稳时间段的未被选中的 Query，直接使用 V 的 mean 进行冗余填充
        out = V.mean(dim=-2, keepdim=True).expand(B, self.n_heads, L, self.head_dim).clone()
        # 填回算好的真实值
        out[torch.arange(B)[:, None, None], torch.arange(self.n_heads)[None, :, None], M_top.indices, :] = out_reduce
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class DistillingLayer(nn.Module):
    """时间蒸馏层：利用池化成倍压缩长序列时间长度，提炼宏观大尺度特征"""
    def __init__(self, d_model):
        super(DistillingLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.activation = nn.ELU()

    def forward(self, x):
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = self.pool(self.activation(self.conv(x)))
        # -> 返回 (B, L/2, D)
        return x.transpose(1, 2)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, e_layers, dropout, max_len=512, factor=5):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pos_emb) # 修复初始化
        self.temporal_proj = nn.Linear(2, d_model)
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.temporal_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList()
        self.distils = nn.ModuleList()
        
        for i in range(e_layers):
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
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model) # 修复：为FFN补充必要的归一化，防止网络越界
            }))
            
            # 在每层之后均插入蒸馏模块 (最后一层除外)
            if i < e_layers - 1:
                self.distils.append(DistillingLayer(d_model))
            else:
                self.distils.append(None)

    def forward(self, x, time_features=None):
        B, L, _ = x.shape
        pos = self.pos_emb[:, :L, :]
        x = self.input_proj(x)
        x = x + self.pos_scale * pos
        
        if time_features is not None:
            time_embed = self.temporal_proj(time_features)
            x = x + self.temporal_scale * time_embed
            
        x = self.dropout(x)
        
        for i, layer in enumerate(self.layers):
            current_L = x.shape[1]
            # 动态更新 Mask，因为随后的蒸馏操作会使得时间步 L 缩小一半
            attn_mask = _causal_mask(current_L, x.device)
            
            # 1. 因果一维卷积 (严格防止未来信息泄露)
            conv_in = x.transpose(1, 2)
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