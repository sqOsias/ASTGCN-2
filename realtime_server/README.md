# 实时交通态势感知系统

基于 ASTGCN 深度学习模型的实时车辆速度预测与可视化系统。

---

## 1. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      前端  Vue3 + ECharts + Element Plus             │
│                      (Vite dev server, port 3000)                    │
│  ┌───────────────┬─────────────────┬────────────────┬────────────┐  │
│  │  TopologyView │ TimeSeriesView  │ AttentionView  │ModelCompare│  │
│  │  宏观路网拓扑  │  时序推演        │ 注意力热力图   │ 模型对比   │  │
│  └───────────────┴─────────────────┴────────────────┴────────────┘  │
│  App.vue  ─── WebSocket /ws ──── Vite Proxy ──────────────────────  │
└──────────────────────────────────────────────────────────────────────┘
                               │ HTTP & WebSocket (via Vite proxy)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   后端  FastAPI + PyTorch  (port 8000)               │
│                                                                      │
│  main.py ─── routes.py ─── simulation.py ─── inference.py           │
│   入口         API端点       仿真循环          ASTGCN推理             │
│              model_loader.py  data_loader.py   config.py  state.py  │
│               模型加载         数据加载          配置       全局状态   │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           数据层                                     │
│  data/PEMS04/pems04.npz        16992×307×3  交通流(flow,occupy,speed)│
│  data/PEMS04/distance.csv      340 条边      传感器间道路距离         │
│  results/.../best_model.pth    ASTGCN 预训练权重                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 快速启动

### 方式一：一键启动（推荐）

```bash
cd /root/ASTGCN-2
chmod +x start_server.sh
./start_server.sh
```

同时启动后端 (port 8000) + 前端 (port 3000)，`Ctrl+C` 一键停止两个服务。

### 方式二：分离启动

```bash
# 终端 1 - 后端
cd /root/ASTGCN-2/realtime_server
./start_backend.sh          # 默认 8000，可传参: ./start_backend.sh 8080

# 终端 2 - 前端
cd /root/ASTGCN-2/realtime_server
./start_frontend.sh         # 默认 3000，可传参: ./start_frontend.sh 3001
```

### 访问大屏

打开浏览器访问 `http://localhost:3000`

---

## 3. 启动脚本说明

| 脚本 | 位置 | 作用 |
|------|------|------|
| `start_server.sh` | 项目根目录 | 一键启动前后端，Ctrl+C 统一停止 |
| `start_backend.sh` | `realtime_server/` | 单独启动后端，支持端口参数 |
| `start_frontend.sh` | `realtime_server/` | 单独启动前端，支持端口参数 |

所有脚本启动前会**自动清理**目标端口上的残留进程。

---

## 4. 后端代码结构

```
realtime_server/backend/          共 884 行
├── main.py           (48行)    入口：创建 FastAPI app，注册 CORS 和路由
├── config.py         (24行)    CONFIG 字典：路径、模型参数、仿真参数
├── state.py          (70行)    SystemState 全局单例 + Pydantic 数据模型
├── model_loader.py  (113行)    ASTGCN 模型构建、LazyConv warmup、权重加载
├── data_loader.py   (210行)    PEMS04 数据加载、时间特征、边列表、节点布局、注意力矩阵
├── inference.py      (98行)    模型推理（含 fallback）、MAE/RMSE 指标计算
├── simulation.py     (89行)    仿真主循环（每秒一帧）、WebSocket 广播
├── routes.py        (232行)    REST API + WebSocket 端点 + 生命周期事件
└── requirements.txt            Python 依赖
```

### 模块依赖关系

```
main.py
  └── routes.py  (register_routes)
        ├── model_loader.py  (startup 时调用 load_model)
        │     ├── config.py
        │     ├── state.py
        │     └── data_loader.py  (load_edge_list)
        ├── data_loader.py   (startup 时调用 load_data)
        │     ├── config.py
        │     └── state.py
        ├── simulation.py    (startup 时启动 simulation_loop)
        │     ├── inference.py
        │     │     ├── config.py
        │     │     └── state.py
        │     ├── config.py
        │     └── state.py
        └── state.py
```

### 核心数据流

```
pems04.npz ──load_data()──▶ state.all_data (16992×307×5，含2个时间特征)
                            state.speed_data (16992×307，km/h)
                            state.mean / state.std (归一化参数)

每秒 simulation_loop():
  state.all_data[index] ──▶ sliding_window (最近36帧)
                       ──▶ run_inference()
                            ├── 归一化 + 切分 week/day/hour 输入
                            ├── ASTGCN 模型前向传播
                            └── 输出 307×12 预测速度 (km/h)
                       ──▶ WebSocket broadcast ──▶ 前端
```

---

## 5. 前端代码结构

```
realtime_server/frontend/src/      共 2199 行
├── main.js            (31行)    Vue 入口：注册 Element Plus、全局图标
├── App.vue           (260行)    根组件：Tab 导航、WebSocket 连接、数据分发
├── style.css         (156行)    全局样式：Tailwind + cyber 主题
└── components/
    ├── TopologyView.vue      (571行)  Tab1: 路网拓扑图 (ECharts Graph)
    ├── TimeSeriesView.vue    (501行)  Tab2: 时序折线图 + 误差分析
    ├── AttentionView.vue     (479行)  Tab3: 注意力热力图 + 节点关联
    └── ModelComparisonView.vue(357行) Tab4: 模型性能对比
```

### 组件数据流

```
App.vue
  │
  ├── WebSocket /ws ◀─── 后端每秒推送
  │     ├── networkData[]      307个节点的实时速度 + 预测速度
  │     ├── systemMetrics      MAE / RMSE
  │     └── historyBuffer[]    最近48帧历史（用于回放）
  │
  ├── GET /api/topology ◀─── 一次性加载
  │     ├── topology.nodes[]   307个节点坐标
  │     └── topology.edges[]   340条边（来自 distance.csv）
  │
  └── Props 分发到子组件:
        TopologyView     ◀── networkData, topology, metrics, historyBuffer
        TimeSeriesView   ◀── networkData, metrics
        AttentionView    ◀── (自行请求 /api/attention)
        ModelComparisonView ◀── (静态数据)
```

---

## 6. 功能说明

### Tab 1: 宏观路网拓扑
- 307 节点的交通网络拓扑图，颜色根据速度动态变化
- **实时路况** / **预测态势** 切换：显示当前真实速度 vs 模型预测30分钟后速度
- **历史回放**：滑块回放最近4小时数据，观察拥堵传播
- **震中聚焦**：高亮 TOP5 拥堵节点及其邻居

### Tab 2: 微观时序推演
- 单节点时序图：真实车速（青色）vs 预测车速（橙色虚线）
- 预测误差柱状图 + 多步预测对比
- 节点状态面板：当前速度、拥堵等级、趋势

### Tab 3: 注意力可解释性
- 307×307 空间注意力矩阵热力图
- 点击查看节点对间的关联权重
- TOP 10 最强空间依赖排行

### Tab 4: 模型性能对比
- ASTGCN vs STGCN / GRU / LSTM / HA 指标对比
- 多步预测衰减曲线

---

## 7. API 接口

### WebSocket
- `ws://localhost:8000/ws` — 实时数据推流（每秒307个节点速度+预测）

### REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/api/topology` | GET | 网络拓扑（节点坐标+边列表） |
| `/api/node/{id}/history` | GET | 节点历史速度+预测记录 |
| `/api/attention` | GET | 307×307 注意力矩阵 |
| `/api/attention/{id}?top_k=10` | GET | 节点 TOP-K 空间关联 |
| `/api/congestion/top?limit=5` | GET | 拥堵排行榜 |
| `/api/stats` | GET | 系统运行状态 |
| `/api/simulation/speed` | POST | 设置仿真速度 (0.1-10x) |
| `/api/simulation/pause` | POST | 暂停仿真 |
| `/api/simulation/resume` | POST | 恢复仿真 |

---

## 8. 配置说明

编辑 `backend/config.py`：

```python
CONFIG = {
    'num_of_vertices': 307,        # 传感器节点数
    'num_for_predict': 12,         # 预测步数（12步=1小时）
    'points_per_hour': 12,         # 每小时采样点数
    'num_of_weeks': 1,             # 周期输入：1周
    'num_of_days': 1,              # 周期输入：1天
    'num_of_hours': 3,             # 近期输入：3小时（36步）
    'K': 3,                        # Chebyshev 多项式阶数
    'num_of_chev_filters': 64,     # 图卷积滤波器数
    'num_of_time_filters': 64,     # 时间卷积滤波器数
    'time_interval_minutes': 5,     # 每步时间间隔（分钟）
    'simulation_speed': 1.0,        # 仿真速度倍率
    'data_path': '...',             # PEMS04 数据路径
    'distance_path': '...',         # 邻接矩阵路径
    'model_path': '...',            # 预训练权重路径
}
```

---

## 9. 技术栈

| 层 | 技术 |
|----|------|
| **前端框架** | Vue 3 (Composition API) + Vite |
| **可视化** | ECharts 5 (vue-echarts) |
| **UI 组件** | Element Plus |
| **样式** | TailwindCSS |
| **后端框架** | FastAPI (异步) |
| **深度学习** | PyTorch (ASTGCN) |
| **实时通信** | WebSocket |
| **数据处理** | NumPy, NetworkX |
