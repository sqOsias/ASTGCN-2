# ASTGCN vs UpgradeASTGCN 单 epoch 训练时间基准

- 生成时间: 2026-05-04 17:03:54
- 设备: `NVIDIA GeForce RTX 4090`
- PyTorch: `2.9.1+cu130` | CUDA: `13.0`

## 测试说明

- `variant=base` 对应 `ASTGCN`（spatial_mode=0, temporal_mode=0）
- `variant=upgrade` 对应 `UpgradeASTGCN`（spatial_mode=1, temporal_mode=1）
- `batch_size` 取自各数据集配置文件
- 先做 warmup epoch，再测量后续若干 epoch，报告均值/标准差/最小/最大

## 结果

| 数据集 | 模型 | Params | Batch Size | Samples | Batches/epoch | 平均 (s) | 标准差 (s) | 最小 (s) | 最大 (s) |
|---|---|---|---|---|---|---|---|---|---|
| PEMS04 | ASTGCN | 1.37 M | 64 | 8979 | 141 | 40.02 | 0.28 | 39.74 | 40.30 |
| PEMS04 | UpgradeASTGCN | 1.59 M | 64 | 8979 | 141 | 60.17 | 0.29 | 59.88 | 60.46 |
| PEMS08 | ASTGCN | 552.22 K | 32 | 9497 | 297 | 37.66 | 0.96 | 36.71 | 38.62 |
| PEMS08 | UpgradeASTGCN | 1.77 M | 32 | 9497 | 297 | 75.85 | 0.43 | 75.42 | 76.28 |

## 相对加速/放缓

| 数据集 | ASTGCN (s) | UpgradeASTGCN (s) | Upgrade / Base |
|---|---|---|---|
| PEMS04 | 40.02 | 60.17 | 1.50x |
| PEMS08 | 37.66 | 75.85 | 2.01x |
