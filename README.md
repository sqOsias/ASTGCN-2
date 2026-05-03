# ASTGCN-2

一个基于 PyTorch 的交通时空预测项目，包含：

- `ASTGCN` / `MSTGCN` 主干训练流程
- 升级版 `UpgradeASTGCN`（可配置空间/时间模块）
- `LSTM` / `GRU` 基线模型
- 模型复杂度对比（Params / FLOPs / Latency）
- 批量加载 `results` 下最佳 checkpoint 进行测试重评估

---

## 1. 项目说明

本项目使用配置驱动训练（`.conf`），训练产物会按实验目录自动归档到 `results/` 下，包括：

- 训练日志
- 验证/测试指标
- 最佳模型权重
- 配置快照与运行时信息

> 关键点：`best_model.pth` 是**验证集最优**时保存的权重，不是最后一个 epoch 的权重。

---

## 2. 环境准备

## 2.1 安装依赖

```bash
pip install -r requirements.txt
```

如果你使用 conda：

```bash
conda activate ast
```

## 2.2 数据准备

请确认配置文件中的数据路径可用（如 `configurations/PEMS04.conf`、`configurations/PEMS08.conf` 中的 `graph_signal_matrix_filename` / `adj_filename`）。

---

## 3. 快速开始

## 3.1 训练主模型（ASTGCN/MSTGCN/升级版）

```bash
python train.py --config configurations/PEMS04.conf
```

或使用脚本：

```bash
bash scripts/train.sh
```

训练完成后，结果会写入 `results/<group>/<run_id>/`。

---

## 3.2 批量重测 `results` 下所有最佳模型（你当前的 test 脚本）

你已将批量评估脚本放在项目根目录 `test.py`，用法如下：

```bash
python test.py \
  --results_dir results \
  --device auto \
  --batch_size 64 \
  --max_runs 0 \
  --output_csv results/recomputed_best_checkpoint_metrics.csv
```

参数说明：

- `--results_dir`：实验结果根目录
- `--device`：`auto | config | cpu | cuda:0`
- `--batch_size`：测试 batch size
- `--max_runs`：限制重测实验数量；`0` 表示全部
- `--output_csv`：汇总输出路径

输出内容：

- 每个 run 会生成：`metrics/test_metrics_recomputed.csv`
- 全局汇总：`results/recomputed_best_checkpoint_metrics.csv`

---

## 3.3 模型复杂度对比（Params / FLOPs / Latency）

运行：

```bash
python scripts/compare_model_complexity.py \
  --config configurations/PEMS04.conf \
  --device auto \
  --batch_size 1 \
  --warmup 20 \
  --runs 100 \
  --repeats 10 \
  --seed 42
```

输出：

- 控制台表格（均值±方差）
- `results/model_complexity_comparison.csv`

你也可直接执行：

```bash
bash scripts/compare_model_complexity.sh
```

---

## 3.4 训练/测试基线模型（LSTM/GRU）

```bash
bash baselines/train_baselines.sh train
bash baselines/train_baselines.sh test
```

或单模型：

```bash
bash baselines/train_baselines.sh train lstm
bash baselines/train_baselines.sh test gru
```

---

## 4. 目录结构（核心）

```text
ASTGCN-2/
├── train.py                                # 主训练入口
├── test.py                                 # 批量重测 best checkpoint（你放置的脚本）
├── summary_results.py                      # 汇总实验最佳epoch指标
├── scripts/
│   ├── train.sh                            # 训练快捷脚本
│   ├── compare_model_complexity.py         # 复杂度对比
│   └── evaluate_best_checkpoints.py        # 与 test.py 等价的脚本版本
├── baselines/
│   ├── run_lstm.py / run_gru.py            # 基线训练
│   ├── test_lstm.py / test_gru.py          # 基线测试
│   └── train_baselines.sh                  # 基线一键脚本
├── lib/                                    # 配置、数据处理、指标、日志
├── model/                                  # ASTGCN/MSTGCN/UpgradeASTGCN
└── results/                                # 训练与评估输出目录
```

---

## 5. 结果解释建议

- 模型选择应基于 `val_metrics.csv` 的最优 `validation_loss`。
- 测试集用于最终评估，不建议用于选 epoch（避免数据泄露）。
- 批量重测脚本用于统一复核历史实验，确保 `best_model.pth` 的可复现性。

---

## 6. 常见问题

### Q1：`best_model.pth` 是最后一轮吗？
不是。仅在验证集 loss 改善时覆盖保存。

### Q2：为什么某些 run 加载 checkpoint 报错？
通常是 lazy 参数尚未初始化。`test.py` 已通过“先前向一次再加载权重”处理该问题。

### Q3：FLOPs 方差为什么常常是 0？
在相同模型结构与输入形状下，理论 FLOPs 基本固定；延迟方差更能反映硬件波动。

---

## 7. 建议工作流

1. 用 `train.py` 跑实验
2. 用 `summary_results.py` 汇总最佳 epoch 结果
3. 用 `test.py` 对历史 best checkpoint 做批量重测复核
4. 用 `compare_model_complexity.py` 生成复杂度对比表

---

