"""
数据预处理工具模块
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import torch


class DataProcessor:
    """数据预处理类"""
    
    def __init__(self):
        self.scalers = {}
        self.stats = {}
    
    def normalize_data(self, data: np.ndarray, method: str = 'min_max') -> Tuple[np.ndarray, Dict]:
        """
        标准化数据
        
        Args:
            data: 输入数据
            method: 标准化方法 ('min_max', 'z_score', 'robust')
        
        Returns:
            标准化后的数据和统计信息
        """
        if method == 'min_max':
            min_val = np.min(data, axis=0, keepdims=True)
            max_val = np.max(data, axis=0, keepdims=True)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            stats = {'min': min_val, 'max': max_val, 'method': 'min_max'}
        elif method == 'z_score':
            mean_val = np.mean(data, axis=0, keepdims=True)
            std_val = np.std(data, axis=0, keepdims=True)
            normalized = (data - mean_val) / (std_val + 1e-8)
            stats = {'mean': mean_val, 'std': std_val, 'method': 'z_score'}
        elif method == 'robust':
            median_val = np.median(data, axis=0, keepdims=True)
            mad = np.median(np.abs(data - median_val), axis=0, keepdims=True)
            normalized = (data - median_val) / (mad + 1e-8)
            stats = {'median': median_val, 'mad': mad, 'method': 'robust'}
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        return normalized, stats
    
    def denormalize_data(self, normalized_data: np.ndarray, stats: Dict) -> np.ndarray:
        """反标准化数据"""
        method = stats['method']
        
        if method == 'min_max':
            return normalized_data * (stats['max'] - stats['min'] + 1e-8) + stats['min']
        elif method == 'z_score':
            return normalized_data * (stats['std'] + 1e-8) + stats['mean']
        elif method == 'robust':
            return normalized_data * (stats['mad'] + 1e-8) + stats['median']
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def fill_missing_values(self, data: np.ndarray, method: str = 'interpolation') -> np.ndarray:
        """
        填充缺失值
        
        Args:
            data: 输入数据
            method: 填充方法 ('interpolation', 'mean', 'forward_fill', 'backward_fill')
        """
        if method == 'interpolation':
            # 对于时间序列数据，使用线性插值
            df = pd.DataFrame(data)
            filled_df = df.interpolate(method='linear', axis=0)
            return filled_df.values
        elif method == 'mean':
            # 使用均值填充
            mask = np.isnan(data)
            mean_vals = np.nanmean(data, axis=0, keepdims=True)
            filled_data = np.where(mask, mean_vals, data)
            return filled_data
        elif method == 'forward_fill':
            # 前向填充
            df = pd.DataFrame(data)
            filled_df = df.fillna(method='ffill', axis=0)
            return filled_df.values
        elif method == 'backward_fill':
            # 后向填充
            df = pd.DataFrame(data)
            filled_df = df.fillna(method='bfill', axis=0)
            return filled_df.values
        else:
            raise ValueError(f"不支持的填充方法: {method}")
    
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """
        检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法 ('iqr', 'z_score')
            threshold: 阈值
        
        Returns:
            布尔数组，True表示异常值位置
        """
        if method == 'iqr':
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == 'z_score':
            z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
            outliers = z_scores > threshold
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        return outliers
    
    def remove_outliers(self, data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """移除异常值（替换为NaN）"""
        outliers = self.detect_outliers(data, method, threshold)
        cleaned_data = data.copy()
        cleaned_data[outliers] = np.nan
        return cleaned_data


class TrafficDataProcessor(DataProcessor):
    """交通数据专用预处理器"""
    
    def __init__(self):
        super().__init__()
        self.speed_limits = {}  # 存储各路段的速度限制
    
    def process_speed_data(self, speed_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        处理速度数据
        
        Args:
            speed_data: 速度数据 (nodes, timesteps)
        
        Returns:
            处理后的数据和统计信息
        """
        # 1. 填充缺失值
        filled_data = self.fill_missing_values(speed_data, method='interpolation')
        
        # 2. 检测并处理异常值（速度不能为负或过高）
        outliers = self.detect_outliers(filled_data, method='iqr', threshold=2.0)
        
        # 将异常值设为NaN以便后续插值
        processed_data = filled_data.copy()
        processed_data[outliers] = np.nan
        
        # 再次插值处理异常值
        processed_data = self.fill_missing_values(processed_data, method='interpolation')
        
        # 3. 标准化
        normalized_data, stats = self.normalize_data(processed_data, method='min_max')
        
        # 4. 确保数据合理性（速度非负）
        normalized_data = np.clip(normalized_data, 0, 1)
        
        return normalized_data, stats
    
    def calculate_traffic_metrics(self, speed_data: np.ndarray) -> Dict[str, float]:
        """
        计算交通指标
        
        Args:
            speed_data: 速度数据 (nodes, timesteps)
        
        Returns:
            交通指标字典
        """
        if speed_data.size == 0:
            return {}
        
        # 计算基本统计
        avg_speed = np.mean(speed_data, axis=0)  # 每个时间步的平均速度
        node_avg_speed = np.mean(speed_data, axis=1)  # 每个节点的平均速度
        overall_avg = np.mean(speed_data)
        
        # 计算拥堵指标
        congestion_ratio = np.mean(speed_data < 20)  # 低于20km/h认为拥堵
        heavy_congestion_ratio = np.mean(speed_data < 10)  # 严重拥堵比例
        
        # 计算波动性
        speed_variance = np.var(speed_data, axis=0)
        avg_variance = np.mean(speed_variance)
        
        return {
            'overall_average_speed': float(overall_avg),
            'avg_speed_by_time': avg_speed.tolist(),
            'avg_speed_by_node': node_avg_speed.tolist(),
            'congestion_ratio': float(congestion_ratio),
            'heavy_congestion_ratio': float(heavy_congestion_ratio),
            'average_variance': float(avg_variance),
            'total_nodes': int(speed_data.shape[0]),
            'total_timesteps': int(speed_data.shape[1])
        }
    
    def create_feature_matrix(self, raw_data: Dict[str, np.ndarray], 
                            feature_types: List[str] = None) -> np.ndarray:
        """
        创建特征矩阵
        
        Args:
            raw_data: 原始数据字典
            feature_types: 要使用的特征类型
        
        Returns:
            特征矩阵 (nodes, features, timesteps)
        """
        if feature_types is None:
            feature_types = ['speed', 'flow', 'occupancy']  # 默认特征类型
        
        # 假设我们有速度、流量、占有率等特征
        features = []
        
        for feat_type in feature_types:
            if feat_type in raw_data:
                feat_data = raw_data[feat_type]
                # 确保数据形状正确
                if len(feat_data.shape) == 2:  # (nodes, timesteps)
                    features.append(feat_data)
                else:
                    features.append(feat_data.squeeze())
        
        if features:
            # 堆叠特征
            feature_matrix = np.stack(features, axis=1)  # (nodes, num_features, timesteps)
        else:
            # 如果没有指定特征，使用速度数据
            speed_data = raw_data.get('speed', raw_data.get('data', np.zeros((307, 1))))
            if len(speed_data.shape) == 2:
                feature_matrix = np.expand_dims(speed_data, axis=1)  # (nodes, 1, timesteps)
            else:
                feature_matrix = speed_data
        
        return feature_matrix


class ModelDataConverter:
    """模型数据转换器"""
    
    @staticmethod
    def convert_to_model_format(data: np.ndarray, sequence_length: int = 36) -> Dict[str, torch.Tensor]:
        """
        将数据转换为模型输入格式
        
        Args:
            data: 输入数据 (nodes, features, timesteps)
            sequence_length: 序列长度
        
        Returns:
            模型输入字典
        """
        if len(data.shape) == 2:  # (nodes, timesteps)
            data = np.expand_dims(data, axis=1)  # (nodes, 1, timesteps)
        
        nodes, features, timesteps = data.shape
        
        # 确保有足够的历史数据
        if timesteps < sequence_length:
            # 如果数据不足，重复最后一个时间步
            pad_length = sequence_length - timesteps
            pad_data = np.tile(data[:, :, -1:], (1, 1, pad_length))
            data = np.concatenate([data, pad_data], axis=2)
            timesteps = sequence_length
        
        # 提取最近sequence_length个时间步
        recent_data = data[:, :, -sequence_length:]
        
        # 为简化，我们将数据复制为week、day、recent三个部分
        # 在实际应用中，这三个部分应该代表不同时间尺度的历史数据
        week_data = recent_data.copy()
        day_data = recent_data.copy()
        recent_input = recent_data.copy()
        
        # 转换为PyTorch张量并添加batch维度
        week_tensor = torch.FloatTensor(week_data).unsqueeze(0)  # (1, nodes, features, sequence_length)
        day_tensor = torch.FloatTensor(day_data).unsqueeze(0)
        recent_tensor = torch.FloatTensor(recent_input).unsqueeze(0)
        
        return {
            'week': week_tensor,
            'day': day_tensor, 
            'recent': recent_tensor
        }
    
    @staticmethod
    def extract_predictions(prediction: torch.Tensor, target_nodes: int = 307) -> np.ndarray:
        """
        提取预测结果
        
        Args:
            prediction: 模型预测输出
            target_nodes: 目标节点数
        
        Returns:
            预测结果数组
        """
        pred_np = prediction.detach().cpu().numpy()
        
        # 确保形状正确
        if pred_np.ndim == 3:  # (batch, nodes, timesteps)
            result = pred_np[0]  # 移除batch维度
        elif pred_np.ndim == 4:  # (batch, nodes, features, timesteps)
            result = pred_np[0, :, 0, :]  # 移除batch和特征维度，保留(nodes, timesteps)
        else:
            result = pred_np
        
        return result