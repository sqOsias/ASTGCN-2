# 运行手册

## 配置启停组件

在配置文件中加入以下段落：

```
[ModelUpgrade]
spatial_mode = 0,1
temporal_mode = 0,1

[AdaptiveGraph]
embedding_dim = 10
sparse_ratio = 0.2
directed = true

[Transformer]
d_model = 64
n_heads = 4
e_layers = 2
dropout = 0.1
max_len = 36
topk_ratio = 0.5
```

- spatial_mode: 0 使用固定距离矩阵，1 使用自适应图
- temporal_mode: 0 使用原 Attention+Conv，1 使用 Transformer 分支

## 运行训练

```
python3 train.py --config configurations/PEMS04.conf --force True
```

## 输出目录

每个实验组合会输出到：

```
{params_dir 或 params}/MODEL_lrLR/{spatial_mode}_{temporal_mode}_{timestamp}
```

目录内包含：

- checkpoints/best_model.pth
- metrics/train_loss.csv
- metrics/val_metrics.csv
- metrics/test_metrics.csv
- artifacts/runtime.json
- configs/train.conf
- configs/config.yaml

## 对比实验结果读取

各组合的指标在 metrics/test_metrics.csv 内。runtime.json 记录训练耗时与 GPU 峰值显存。

## 集成对比检查

```
python3 scripts/integration_compare.py --base_dir <[0,0]目录> --new_dir <[1,1]目录> --threshold 0.03
```
