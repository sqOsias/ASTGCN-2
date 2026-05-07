# AST-Informer 实验参数与数据集统计

> 本文档汇总 AST-Informer（即 `UpgradeASTGCN` 在 `spatial_mode=1, temporal_mode=1` 模式下的实例化版本）的核心超参数与 PEMS04 数据集的切片/划分统计。所有数值均直接对应仓库中的可复现配置：`configurations/PEMS04.conf`、`lib/data_preparation.py`、`model/upgrade/`、`model/model_config.py`。

---

## 一、AST-Informer 模型核心超参数

模型由 **空间分支（自适应图扩散卷积）** + **时序分支（Informer-style 概率稀疏 Transformer）** 组成，3 路并联（小时 / 日 / 周）后求和得到最终预测。

### 1.1 顶层结构与控制开关

| 参数 | 取值 | 说明 | 来源 |
|------|------|------|------|
| `model_name` | `ASTGCN`（实例化为 `UpgradeASTGCN`） | 顶层模型类 | `PEMS04.conf [Training]` |
| `spatial_mode` | **1** | 0 = 原始 Chebyshev 卷积；**1 = 自适应图扩散卷积** | `PEMS04.conf [ModelUpgrade]` |
| `temporal_mode` | **1** | 0 = 原始时间注意力 + Conv2d；**1 = Informer 时序 Transformer** | `PEMS04.conf [ModelUpgrade]` |
| `num_of_weeks` | 1 | 周分支历史窗口（小时数） | `PEMS04.conf [Training]` |
| `num_of_days` | 1 | 日分支历史窗口（小时数） | 同上 |
| `num_of_hours` | 1 | 小时分支历史窗口（小时数） | 同上 |
| `num_for_predict` | 12 | 预测未来步数（=60 min） | `PEMS04.conf [Data]` |

### 1.2 空间分支：自适应图扩散卷积

| 参数 | 取值 | 含义 |
|------|------|------|
| `K`（Chebyshev 阶数） | **3** | 扩散卷积的传播阶数，等价于聚合 K 阶邻居 |
| `num_of_chev_filters` | 64 | 图卷积输出通道数 | 
| `embedding_dim`（节点嵌入维度） | **10** | 每个传感器节点的可学习向量维度 |
| `sparse_ratio` | **0.8** | 自适应邻接矩阵稀疏化阈值（保留 top-20% 边） |
| `directed` | true | 是否使用有向图（前向/后向两个嵌入矩阵） |

> 自适应图：`A_adapt = softmax(ReLU(E1 · E2^T))`，再经 `sparse_ratio` 截断仅保留显著连接，可在训练中动态学习路网拓扑而不依赖静态距离图。

### 1.3 时序分支：Informer 概率稀疏 Transformer

| 参数 | 取值 | 含义 |
|------|------|------|
| `d_model` | **32** | Transformer 隐藏维度 |
| `n_heads` | **2** | 多头注意力头数（每头维度 = 32/2 = 16） |
| `e_layers` | **3** | 编码器堆叠层数（每层 = 因果卷积 + ProbSparse Attn + FFN） |
| `dropout` | 0.1 | 注意力与 FFN 的 dropout 比例 |
| `max_len` | 36 | 最大序列长度上限（= `points_per_hour × 3`） |
| `topk_ratio` | 0.5 | ProbSparse 中 Top-u 选择比例 |
| `factor`（采样因子，硬编码） | 5 | `U_part = factor · ⌈log L⌉`，控制稀疏采样查询数 |
| 因果卷积 kernel | 3 | 每层编码器内部时间因果卷积 |
| FFN 内部维度 | `4 · d_model` = 128 | 前馈层中间维度 |
| Distilling Layer | Conv1d(k=3) + MaxPool1d(k=3, s=2) | 每层之间将时间维度减半（最后一层除外） |
| 位置编码 | 可学习 + 时间标签（time_of_day, day_of_week）的线性映射 | `pos_scale`/`temporal_scale` 为可学习缩放系数 |

> ProbSparse Attention 仅对 Top-u 个"信息量大"的 query 计算完整注意力，其余使用 V 的均值填充，将注意力复杂度从 O(L²) 降至 O(L log L)。

### 1.4 ASTGCN 三分支结构（每分支 2 个块）

每条分支均由 2 个 `UpgradeASTGCNBlock` 串联，核心通道与时序参数：

| 块内参数 | 取值 | 备注 |
|------|------|------|
| `num_of_chev_filters` | 64 | 图卷积输出 |
| `num_of_time_filters` | 64 | 时间卷积输出 |
| 时间卷积 kernel | (1, 3) | 沿时间维度的 1D 卷积 |
| `time_conv_strides`（第 1 块） | =`num_of_{weeks/days/hours}`=1 | 第 1 块时间步长 |
| `time_conv_strides`（第 2 块） | 1 | 第 2 块时间步长 |
| 残差连接 | LazyConv2d (k=1, s=`time_conv_strides`) | 通道数自适应 |
| 归一化 | `LayerNorm(num_of_time_filters)` | 每块输出后 |

### 1.5 训练超参数

| 参数 | 取值 | 来源 |
|------|------|------|
| 优化器 | **Muon** | `PEMS04.conf [Training]` |
| 学习率 | 1e-3 (0.001) | 同上 |
| Batch size | **64** | 同上 |
| Epochs | **50** | 同上 |
| 损失函数 | MSELoss（reduction='none'） | `train.py` |
| 随机种子 | 1 | `PEMS04.conf [Training]` |
| `merge` | 0（不合并训练/验证集） | 同上 |
| 设备 | GPU-0（NVIDIA RTX 4090） | `PEMS04.conf [Training] ctx=gpu-0` |
| 权重初始化 | `WeightInitializer.init_weight`（通常为 Xavier） | `train.py` |

---

## 二、PEMS04 数据集切片与样本划分统计

### 2.1 原始数据规模

| 字段 | 数值 | 含义 |
|------|------|------|
| 数据文件 | `data/PEMS04/pems04.npz` | 原始时空张量 |
| 邻接关系文件 | `data/PEMS04/distance.csv` | 路网边表（带距离） |
| 节点数 `N` | **307** | 加州 SF Bay Area 主干道传感器 |
| 时间步数 `T` | **16,992** | 每 5 分钟采样一次 |
| 采样间隔 | 5 min（`points_per_hour = 12`） | 一天 288 步 |
| 覆盖时长 | **59 天**（≈ 8.43 周） | 16992 / 288 |
| 原始特征数 | 3 | 流量 (flow) / 占有率 (occupancy) / 速度 (speed) |
| 时间特征 | 2（自动追加） | 一天中的时刻 + 一周中的星期（均归一化到 [0,1)） |
| 模型输入特征 `F` | **5** | 3 + 2 |
| 预测目标 | 速度（特征 idx=2，单位 km/h） | `data_preparation.py` |
| 边数（distance.csv） | **340** 条有向边 | 文件 341 行 - 1 行表头 |

### 2.2 滑窗采样规则

对每个潜在的预测起点 `label_start_idx`，从原始时间序列中切出三段历史输入与一段未来目标：

| 分支 | 时间偏移参考点 | 输入长度 | 含义 |
|------|----------------|----------|------|
| **小时分支 (recent)** | 紧邻 `label_start_idx` 之前 1 小时 | `1 × 12 = 12` 步 | 短期连续历史 |
| **日分支 (day)** | 前 1 天的同一时段 | `1 × 12 = 12` 步 | 日周期性 |
| **周分支 (week)** | 前 1 周的同一时段 | `1 × 12 = 12` 步 | 周周期性 |
| **预测目标 (target)** | `label_start_idx` 起 12 步 | `12` 步（60 min） | 未来一小时速度 |

每个有效样本张量形状（按 `(N, F, T_window)` 排列）：

```
week  : (307, 5, 12)
day   : (307, 5, 12)
recent: (307, 5, 12)
target: (307, 12)        # 仅速度通道
```

### 2.3 有效样本数与划分（6 : 2 : 2）

> 由于周分支需要往前回溯 7 天 = `7 × 24 × 12 = 2016` 步，前 2016 个时刻无法作为预测起点；后端再扣除 12 步预测窗口。

| 项目 | 数量 | 计算 |
|------|------|------|
| 总有效样本 | **14,964** | `T - num_for_predict - 7·24·12 = 16992 - 12 - 2016` |
| 训练集（前 60%） | **8,978** | `int(14964 × 0.6)` |
| 验证集（中 20%） | **2,993** | `int(14964 × 0.8) − int(14964 × 0.6)` |
| 测试集（末 20%） | **2,993** | `14964 − int(14964 × 0.8)` |
| 划分方式 | 时间顺序切分（无 shuffle） | `lib/data_preparation.py:124-125` |

### 2.4 张量形状汇总（送入 DataLoader 之前）

每个划分集合返回 4 个张量（week / day / recent / target），布局 `(B, N, F, T)`：

| 集合 | week / day / recent | target |
|------|---------------------|--------|
| 训练 | `(8978, 307, 5, 12)` | `(8978, 307, 12)` |
| 验证 | `(2993, 307, 5, 12)` | `(2993, 307, 12)` |
| 测试 | `(2993, 307, 5, 12)` | `(2993, 307, 12)` |

### 2.5 归一化策略

- **方法**：Z-score（按 `(B, N, F, T)` 张量逐 axis=0 求均值/方差）。
- **统计量来源**：仅基于训练集计算 `mean / std`，再应用到 train / val / test。
- **target 不参与归一化**，保持原始速度值（km/h），便于直接评估 MAE / RMSE / MAPE。
- 三个分支（week / day / recent）**各自独立**计算归一化统计量。

### 2.6 DataLoader 配置

| 参数 | 取值 |
|------|------|
| Batch size | 64 |
| 训练集 shuffle | True |
| 验证集 / 测试集 shuffle | False |
| 训练 batch 数 | `⌈8978 / 64⌉ = 141` |
| 验证 batch 数 | `⌈2993 / 64⌉ = 47` |
| 测试 batch 数 | `⌈2993 / 64⌉ = 47` |

---

## 三、关键超参数速查表（论文嵌入版）

| 模块 | 超参数 | 取值 |
|------|--------|------|
| **数据** | 节点数 N / 总时长 T / 采样 | 307 / 16,992 步 / 5 min |
| **数据** | 训练 / 验证 / 测试 | 8,978 / 2,993 / 2,993（6:2:2） |
| **数据** | 输入窗口 / 预测窗口 | 12 步 / 12 步（各 60 min） |
| **数据** | 输入特征 F | 5（flow, occupancy, speed, time-of-day, day-of-week） |
| **空间** | Chebyshev 阶 K | 3 |
| **空间** | 自适应嵌入维度 | 10 |
| **空间** | 稀疏率 sparse_ratio | 0.8（保留 top 20% 边） |
| **空间** | 卷积通道 num_of_chev_filters | 64 |
| **时序** | d_model | 32 |
| **时序** | n_heads | 2 |
| **时序** | e_layers | 3 |
| **时序** | ProbSparse factor | 5 |
| **时序** | Distilling Layer | 每层后池化 1/2 |
| **时序** | dropout | 0.1 |
| **训练** | Optimizer / LR | Muon / 1e-3 |
| **训练** | Epochs / Batch | 50 / 64 |
| **训练** | Loss / Seed | MSE / 1 |
| **硬件** | GPU | NVIDIA RTX 4090 (24 GB) |

---

*本文档对应代码版本：`configurations/PEMS04.conf` (commit at run `1_1_20260503100704`)。如需复现实验，直接执行 `python train.py --config configurations/PEMS04.conf`。*
