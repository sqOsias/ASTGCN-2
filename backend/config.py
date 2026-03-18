"""
配置文件 - 系统配置参数
"""
import os
from typing import Dict, Any

# 基础配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型配置
MODEL_CONFIG = {
    'num_of_vertices': 307,  # PEMS04数据集的节点数
    'len_input': 36,         # 输入序列长度
    'num_for_prediction': 12, # 预测序列长度
    'spatial_mode': 1,       # 空间模式
    'temporal_mode': 1,      # 时间模式
    'device': 'cuda' if os.environ.get('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
}

# 数据配置
DATA_CONFIG = {
    'dataset_path': os.path.join(BASE_DIR, 'data/PEMS04/pems04.npz'),
    'distance_path': os.path.join(BASE_DIR, 'data/PEMS04/distance.csv'),
    'processed_data_dir': os.path.join(BASE_DIR, 'data/processed'),
    'node_count': 307,
    'features': ['speed', 'flow', 'occupancy'],  # 支持的特征类型
    'normalization_method': 'min_max',  # 标准化方法
    'missing_value_method': 'interpolation'  # 缺失值处理方法
}

# WebSocket配置
WEBSOCKET_CONFIG = {
    'ping_interval': 30,      # 心跳间隔（秒）
    'max_connections': 100,   # 最大连接数
    'buffer_size': 1024,      # 缓冲区大小
    'timeout': 60             # 超时时间（秒）
}

# 推理配置
INFERENCE_CONFIG = {
    'batch_size': 1,          # 推理批次大小
    'max_workers': 2,         # 最大推理工作线程数
    'cache_predictions': True, # 是否缓存预测结果
    'enable_async': True,     # 是否启用异步推理
    'queue_timeout': 10       # 队列超时时间
}

# 滑动窗口配置
SLIDING_WINDOW_CONFIG = {
    'maxlen': 12,             # 滑动窗口最大长度（代表过去3小时的36个5分钟时间步）
    'time_step_minutes': 5,   # 时间步长度（分钟）
    'update_interval': 1,     # 更新间隔（秒）- 现实中的5分钟压缩为1秒
    'history_horizon': 36     # 历史时间范围
}

# 前端配置
FRONTEND_CONFIG = {
    'title': '实时车辆速度预测系统',
    'dashboard_theme': 'dark',  # 仪表板主题
    'refresh_interval': 1000,   # 刷新间隔（毫秒）
    'animation_duration': 500,  # 动画持续时间（毫秒）
    'max_display_points': 1000  # 最大显示点数
}

# 性能指标配置
METRICS_CONFIG = {
    'mae_window': 100,        # MAE计算窗口大小
    'rmse_window': 100,       # RMSE计算窗口大小
    'mape_window': 100,       # MAPE计算窗口大小
    'enable_logging': True,   # 是否记录指标日志
    'log_interval': 60        # 日志记录间隔（秒）
}

# 速度分类阈值（用于拥堵状态判断）
SPEED_THRESHOLDS = {
    'free_flow_min': 60,      # 畅通最小速度(km/h)
    'congested_max': 30,      # 缓行最大速度(km/h) 
    'jam_max': 10             # 拥堵最大速度(km/h)
}

# 服务器配置
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': False,          # 开发模式下启用热重载
    'workers': 1,             # 工作进程数
    'log_level': 'info'       # 日志级别
}


def get_config(key: str) -> Any:
    """获取配置值"""
    configs = [
        MODEL_CONFIG,
        DATA_CONFIG, 
        WEBSOCKET_CONFIG,
        INFERENCE_CONFIG,
        SLIDING_WINDOW_CONFIG,
        FRONTEND_CONFIG,
        METRICS_CONFIG,
        SPEED_THRESHOLDS,
        SERVER_CONFIG
    ]
    
    for config in configs:
        if key in config:
            return config[key]
    
    raise KeyError(f"配置键 '{key}' 不存在")


def update_config(key: str, value: Any) -> None:
    """更新配置值（注意：这只会修改运行时配置，不会持久化）"""
    configs = [
        MODEL_CONFIG,
        DATA_CONFIG, 
        WEBSOCKET_CONFIG,
        INFERENCE_CONFIG,
        SLIDING_WINDOW_CONFIG,
        FRONTEND_CONFIG,
        METRICS_CONFIG,
        SPEED_THRESHOLDS,
        SERVER_CONFIG
    ]
    
    for config in configs:
        if key in config:
            config[key] = value
            return
    
    raise KeyError(f"配置键 '{key}' 不存在")


# 速度颜色映射（用于可视化）
COLOR_MAPPING = {
    'free_flow': '#00FF00',    # 绿色 - 畅通 (>60 km/h)
    'moderate': '#FFFF00',     # 黄色 - 缓行 (30-60 km/h) 
    'congested': '#FF8C00',    # 橙色 - 拥堵 (10-30 km/h)
    'jam': '#FF0000',          # 红色 - 严重拥堵 (<10 km/h)
    'unknown': '#808080'       # 灰色 - 未知/无数据
}


# 路径配置
PATH_CONFIG = {
    'model_weights': os.path.join(BASE_DIR, 'params'),
    'logs': os.path.join(BASE_DIR, 'logs'),
    'temp': os.path.join(BASE_DIR, 'temp'),
    'static_files': os.path.join(BASE_DIR, 'frontend/static'),
    'templates': os.path.join(BASE_DIR, 'frontend/templates')
}


# 确保必要的目录存在
for path_key, path_value in PATH_CONFIG.items():
    os.makedirs(path_value, exist_ok=True)


def get_all_configs() -> Dict[str, Any]:
    """获取所有配置"""
    all_configs = {}
    all_configs.update(MODEL_CONFIG)
    all_configs.update(DATA_CONFIG)
    all_configs.update(WEBSOCKET_CONFIG)
    all_configs.update(INFERENCE_CONFIG)
    all_configs.update(SLIDING_WINDOW_CONFIG)
    all_configs.update(FRONTEND_CONFIG)
    all_configs.update(METRICS_CONFIG)
    all_configs.update(SPEED_THRESHOLDS)
    all_configs.update(SERVER_CONFIG)
    all_configs.update(PATH_CONFIG)
    
    return all_configs