import sys
import torch

sys.path.append('.')


def test_adaptive_graph_sparsity():
    from model.upgrade.adaptive_graph import AdaptiveGraph
    num_nodes = 20
    sparse_ratio = 0.5
    graph = AdaptiveGraph(num_nodes=num_nodes, embedding_dim=8, sparse_ratio=sparse_ratio, directed=True)
    adj = graph()
    nonzero_ratio = (adj > 0).float().mean().item()
    assert nonzero_ratio <= 1.0 - sparse_ratio + 0.1


def test_adaptive_graph_undirected():
    from model.upgrade.adaptive_graph import AdaptiveGraph
    num_nodes = 10
    graph = AdaptiveGraph(num_nodes=num_nodes, embedding_dim=4, sparse_ratio=0.0, directed=False)
    adj = graph()
    diff = torch.abs(adj - adj.t()).max().item()
    assert diff < 1e-5
