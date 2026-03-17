import sys
import torch

sys.path.append('.')


def test_temporal_transformer_shape():
    from model.upgrade.temporal_transformer import TemporalTransformer
    model = TemporalTransformer(input_dim=4, d_model=8, n_heads=2, e_layers=1, dropout=0.0, max_len=64, topk_ratio=1.0)
    x = torch.randn(2, 36, 4)
    out = model(x, time_features=None)
    assert out.shape == (2, 36, 8)


def test_temporal_transformer_causal():
    from model.upgrade.temporal_transformer import TemporalTransformer
    torch.manual_seed(1)
    model = TemporalTransformer(input_dim=3, d_model=8, n_heads=2, e_layers=1, dropout=0.0, max_len=64, topk_ratio=1.0)
    x1 = torch.randn(1, 12, 3)
    x2 = x1.clone()
    x2[:, 6:, :] = torch.randn(1, 6, 3)
    out1 = model(x1, time_features=None)
    out2 = model(x2, time_features=None)
    diff = torch.abs(out1[:, :6, :] - out2[:, :6, :]).max().item()
    assert diff < 1e-5
