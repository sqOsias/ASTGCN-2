"""
滑动窗口数据结构 - 用于维护时间序列数据
"""
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
import json


class SlidingWindow:
    """固定长度的滑动窗口数据结构"""
    
    def __init__(self, maxlen: int = 12):
        """
        初始化滑动窗口
        
        Args:
            maxlen: 窗口最大长度，默认为12（代表过去3小时的36个5分钟时间步）
        """
        self.maxlen = maxlen
        self.window = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
    
    def append(self, data: Dict[str, Any]):
        """添加新数据到窗口"""
        self.window.append(data)
        self.timestamps.append(data.get('timestamp'))
    
    def extend(self, data_list: List[Dict[str, Any]]):
        """批量添加数据"""
        for data in data_list:
            self.append(data)
    
    def get_recent(self, n: int = None) -> List[Dict[str, Any]]:
        """获取最近的n个数据项"""
        if n is None:
            n = len(self.window)
        return list(self.window)[-n:]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """获取窗口中的所有数据"""
        return list(self.window)
    
    def get_size(self) -> int:
        """获取当前窗口大小"""
        return len(self.window)
    
    def is_full(self) -> bool:
        """检查窗口是否已满"""
        return len(self.window) == self.maxlen
    
    def clear(self):
        """清空窗口"""
        self.window.clear()
        self.timestamps.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取窗口数据的统计信息"""
        if not self.window:
            return {}
        
        # 收集速度数据
        speeds = []
        for item in self.window:
            if 'speeds' in item and item['speeds']:
                speeds.extend(item['speeds'])
        
        if not speeds:
            return {}
        
        speeds_array = np.array(speeds)
        return {
            'count': len(speeds),
            'mean': float(np.mean(speeds_array)),
            'std': float(np.std(speeds_array)),
            'min': float(np.min(speeds_array)),
            'max': float(np.max(speeds_array)),
            'median': float(np.median(speeds_array))
        }
    
    def get_speed_distribution(self) -> Dict[str, int]:
        """获取速度分布统计（用于拥堵分析）"""
        if not self.window:
            return {'free_flow': 0, 'congested': 0, 'jam': 0}
        
        # 统计最近的数据（当前状态）
        current_speeds = self.window[-1]['speeds'] if self.window else []
        
        free_flow = sum(1 for s in current_speeds if s > 30)  # >30km/h: 畅通
        congested = sum(1 for s in current_speeds if 10 <= s <= 30)  # 10-30km/h: 缓行
        jam = sum(1 for s in current_speeds if s < 10)  # <10km/h: 拥堵
        
        return {
            'free_flow': free_flow,
            'congested': congested,
            'jam': jam,
            'total': len(current_speeds)
        }
    
    def get_congestion_index(self) -> float:
        """计算拥堵指数（0-1范围，1表示全部拥堵）"""
        distribution = self.get_speed_distribution()
        total = distribution['total']
        if total == 0:
            return 0.0
        
        # 加权计算拥堵指数：拥堵权重1.0，缓行权重0.5，畅通权重0.0
        congestion_score = (
            distribution['jam'] * 1.0 +
            distribution['congested'] * 0.5 +
            distribution['free_flow'] * 0.0
        )
        
        return congestion_score / total if total > 0 else 0.0


class TimeSeriesBuffer:
    """时间序列数据缓冲区"""
    
    def __init__(self, node_count: int, feature_count: int = 1):
        """
        初始化时间序列缓冲区
        
        Args:
            node_count: 节点数量（例如307个传感器）
            feature_count: 特征数量（例如速度、流量等）
        """
        self.node_count = node_count
        self.feature_count = feature_count
        self.data_buffer = {}  # 存储各个时间序列
        
    def add_series(self, series_name: str, data: np.ndarray):
        """添加时间序列数据"""
        if data.shape[0] != self.node_count:
            raise ValueError(f"数据节点数量({data.shape[0]})与预期({self.node_count})不符")
        
        self.data_buffer[series_name] = data
    
    def get_series(self, series_name: str) -> Optional[np.ndarray]:
        """获取指定名称的时间序列"""
        return self.data_buffer.get(series_name)
    
    def get_all_series_names(self) -> List[str]:
        """获取所有时间序列名称"""
        return list(self.data_buffer.keys())
    
    def update_series(self, series_name: str, new_data: np.ndarray):
        """更新时间序列数据"""
        if series_name in self.data_buffer:
            old_data = self.data_buffer[series_name]
            # 这里可以实现滑动窗口更新逻辑
            updated_data = np.concatenate([old_data, new_data], axis=-1)  # 沿时间轴拼接
            # 限制最大长度
            max_length = 100  # 可配置的最大时间步数
            if updated_data.shape[-1] > max_length:
                updated_data = updated_data[..., -max_length:]
            
            self.data_buffer[series_name] = updated_data
        else:
            self.add_series(series_name, new_data)


# 测试代码
if __name__ == "__main__":
    # 创建滑动窗口实例
    sw = SlidingWindow(maxlen=5)
    
    # 添加测试数据
    for i in range(7):
        test_data = {
            'timestamp': f'2023-01-01T12:{i:02d}:00',
            'speeds': [float(j + i) for j in range(10)],  # 10个虚拟节点的速度
            'node_count': 10
        }
        sw.append(test_data)
        print(f"添加数据后窗口大小: {sw.get_size()}")
    
    print(f"当前窗口数据: {sw.get_recent()}")
    print(f"统计信息: {sw.get_statistics()}")
    print(f"速度分布: {sw.get_speed_distribution()}")
    print(f"拥堵指数: {sw.get_congestion_index():.2f}")