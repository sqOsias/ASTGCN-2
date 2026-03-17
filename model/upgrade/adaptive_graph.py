import torch
from torch import nn
import torch.nn.functional as F

class AdaptiveGraph(nn.Module):
    def __init__(self, num_nodes, embedding_dim, sparse_ratio=0.0, directed=True):
        super(AdaptiveGraph, self).__init__()
        self.num_nodes = num_nodes
        self.sparse_ratio = float(sparse_ratio)
        self.directed = bool(directed)
        
        # 修复：摒弃 randn 避免产生极端的 logits 造成 Softmax 梯度消失
        self.node_emb_src = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.node_emb_dst = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.node_emb_src)
        nn.init.xavier_uniform_(self.node_emb_dst)

    def forward(self):
        logits = torch.matmul(self.node_emb_src, self.node_emb_dst.t())
        if not self.directed:
            logits = (logits + logits.t()) * 0.5
        logits = F.relu(logits)
        adj = torch.softmax(logits, dim=1)
        if not self.directed:
            adj = (adj + adj.t()) * 0.5
            
        # 稀疏化控制
        if self.sparse_ratio > 0:
            k = max(1, int((1.0 - self.sparse_ratio) * self.num_nodes))
            topk = torch.topk(adj, k=k, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, topk.indices, 1.0)
            adj = adj * mask
            row_sum = adj.sum(dim=1, keepdim=True)
            adj = torch.where(row_sum > 0, adj / row_sum, adj)
            
        return adj