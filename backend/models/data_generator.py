import numpy as np
from datetime import datetime, timedelta
import random


def generate_realistic_traffic_data(nodes=307, base_timestamp=None):
    """
    生成更符合真实情况的交通数据
    """
    if base_timestamp is None:
        base_timestamp = datetime.now()
    
    # 生成基础速度数据（模拟真实交通模式）
    base_speeds = []
    for node in range(nodes):
        # 每个节点有不同的基础速度范围
        base_speed = random.uniform(30, 70)  # km/h
        base_speeds.append(base_speed)
    
    speeds = []
    for i in range(nodes):
        # 添加时间和空间相关的变化
        hour_factor = 1 + 0.3 * np.sin((base_timestamp.hour + base_timestamp.minute/60) * np.pi / 12 - np.pi/2)  # 早晚高峰
        noise = random.uniform(-5, 5)  # 随机噪声
        speed = max(5, min(120, base_speeds[i] * hour_factor + noise))  # 限制在合理范围内
        speeds.append(speed)
    
    return {
        'timestamp': base_timestamp.isoformat(),
        'speeds': speeds,
        'congestion_level': calculate_congestion_level(speeds),
        'average_speed': sum(speeds) / len(speeds),
        'min_speed': min(speeds),
        'max_speed': max(speeds)
    }


def calculate_congestion_level(speeds):
    """计算拥堵水平"""
    avg_speed = sum(speeds) / len(speeds)
    if avg_speed > 60:
        return 'low'
    elif avg_speed > 40:
        return 'medium'
    else:
        return 'high'


def load_real_dataset(dataset_path="/root/ASTGCN-2/data/PEMS04/pems04.npz"):
    """
    加载真实数据集
    """
    try:
        data = np.load(dataset_path)
        # 假设数据格式为 (time_steps, nodes, features)
        traffic_data = data['data']  # shape: (16992, 307, 3)
        print(f"成功加载真实数据集: shape={traffic_data.shape}")
        return traffic_data
    except Exception as e:
        print(f"加载真实数据集失败: {e}")
        return None


def get_real_data_point(traffic_data, time_index):
    """
    从真实数据集中获取特定时间点的数据
    """
    if traffic_data is not None and time_index < len(traffic_data):
        # 取全部3个特征
        raw_data = traffic_data[time_index]  # (307, 3)
        speed_data = raw_data[:, 0]  # 取第一个特征作为速度
        return {
            'timestamp': f"2023-01-{(time_index % 28 + 1):02d} {((time_index // 2) % 24):02d}:{((time_index * 2) % 60):02d}:00",
            'speeds': speed_data.tolist(),
            'congestion_level': calculate_congestion_level(speed_data.tolist()),
            'average_speed': float(np.mean(speed_data)),
            'min_speed': float(np.min(speed_data)),
            'max_speed': float(np.max(speed_data)),
            'raw_data': raw_data  # 包含全部3个特征
        }
    else:
        # 如果超出范围，生成模拟数据
        return generate_realistic_traffic_data()


def get_historical_data(traffic_data, start_index, length=36):
    """
    获取历史数据片段用于滑动窗口
    """
    data_points = []
    for i in range(length):
        idx = start_index - length + 1 + i
        if 0 <= idx < len(traffic_data):
            data_points.append(get_real_data_point(traffic_data, idx))
        else:
            # 如果索引超出范围，使用模拟数据
            data_points.append(generate_realistic_traffic_data())
    return data_points