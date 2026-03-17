import torch
from torch import nn
import torch.nn.functional as F


def _causal_mask(length, device):
    mask = torch.triu(torch.ones(length, length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, topk_ratio=0.5):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.topk_ratio = float(topk_ratio)

    def forward(self, x, attn_mask):
        batch_size, length, _ = x.shape
        q = self.q_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask

        k_num = max(1, int(self.topk_ratio * length))
        topk = torch.topk(scores, k=k_num, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk.indices, topk.values)
        attn = torch.softmax(mask, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        return self.out_proj(out)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, e_layers, dropout, max_len=512, topk_ratio=0.5):
        super(TemporalTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.temporal_proj = nn.Linear(2, d_model)
        self.pos_scale = nn.Parameter(torch.ones(1))
        self.temporal_scale = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for _ in range(e_layers):
            self.layers.append(nn.ModuleDict({
                'causal_conv': nn.Conv1d(d_model, d_model, kernel_size=3, padding=0),
                'attn': ProbSparseAttention(d_model, n_heads, dropout=dropout, topk_ratio=topk_ratio),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }))

    def forward(self, x, time_features=None):
        batch_size, length, _ = x.shape
        pos = self.pos_emb[:, :length, :]
        x = self.input_proj(x)
        x = x + self.pos_scale * pos
        if time_features is not None:
            time_embed = self.temporal_proj(time_features)
            x = x + self.temporal_scale * time_embed
        x = self.dropout(x)
        attn_mask = _causal_mask(length, x.device)
        for layer in self.layers:
            conv_in = x.transpose(1, 2)
            conv_in = F.pad(conv_in, (2, 0))
            conv_out = layer['causal_conv'](conv_in).transpose(1, 2)
            x = layer['norm1'](x + conv_out)
            attn_out = layer['attn'](x, attn_mask)
            x = layer['norm2'](x + attn_out)
            x = x + layer['ffn'](x)
        return x
