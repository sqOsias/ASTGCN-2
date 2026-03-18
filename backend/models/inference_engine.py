"""
推理引擎模块 - 处理模型推理逻辑
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


class InferenceEngine:
    """推理引擎类 - 负责模型推理"""
    
    def __init__(self, model: UpgradeASTGCN, device: torch.device):
        self.model = model
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=2)  # 线程池用于非阻塞推理
        
    def preprocess_input(self, raw_data: np.ndarray) -> torch.Tensor:
        """预处理输入数据"""
        # 确保数据形状正确
        if len(raw_data.shape) == 2:  # (nodes, features)
            raw_data = np.expand_dims(raw_data, axis=0)  # 添加时间维度
        elif len(raw_data.shape) == 1:  # (nodes,)
            raw_data = np.expand_dims(raw_data, axis=1)  # 添加特征维度
            raw_data = np.expand_dims(raw_data, axis=0)  # 添加时间维度
        
        # 转换为tensor并移动到设备
        tensor = torch.FloatTensor(raw_data).to(self.device)
        
        # 确保形状为 (batch, nodes, features, timesteps)
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)  # 添加batch维度
        
        return tensor
    
    def run_inference_sync(self, input_tensors: List[torch.Tensor]) -> torch.Tensor:
        """同步执行推理"""
        self.model.eval()
        with torch.no_grad():
            # 确保所有输入都在正确的设备上
            input_tensors = [t.to(self.device) for t in input_tensors]
            
            # 执行模型推理
            prediction = self.model(input_tensors)
        
        return prediction
    
    async def run_inference_async(self, input_tensors: List[torch.Tensor]) -> torch.Tensor:
        """异步执行推理（使用线程池避免阻塞事件循环）"""
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            self.executor, 
            self.run_inference_sync, 
            input_tensors
        )
        return prediction
    
    def postprocess_output(self, prediction: torch.Tensor) -> np.ndarray:
        """后处理输出数据"""
        # 移动到CPU并转换为numpy
        result = prediction.cpu().numpy()
        
        # 确保返回合适的形状
        if result.ndim == 3:  # (batch, nodes, timesteps)
            result = result[0]  # 移除batch维度
        elif result.ndim == 4:  # (batch, nodes, features, timesteps)
            result = result[0]  # 移除batch维度，保留(nodes, features, timesteps)
        
        return result


class StreamingInferenceWorker:
    """流式推理工作器 - 管理推理流程"""
    
    def __init__(self, model: UpgradeASTGCN, device: torch.device):
        self.inference_engine = InferenceEngine(model, device)
        self.is_running = False
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)  # 添加executor属性
    
    def prepare_model_inputs(self, sliding_window_data: List[Dict]) -> List[torch.Tensor]:
        """准备模型输入 - 将滑动窗口数据转换为模型所需的格式"""
        if not sliding_window_data:
            raise ValueError("滑动窗口数据为空")
        
        # ASTGCN需要3种时间尺度的数据：week, day, recent
        # 每个数据的形状应该是 (batch, nodes, features, timesteps)
        
        # 获取节点数量（从第一个数据项获取）
        if not sliding_window_data:
            raise ValueError("滑动窗口数据为空")
        
        num_nodes = len(sliding_window_data[0]['speeds'])
        
        # 提取最近36个时间步的数据作为recent输入
        # 从滑动窗口数据中提取速度数据
        recent_speeds = []
        for item in sliding_window_data[-36:]:  # 最近36个时间步
            # 使用原始数据中的特征维度
            # 由于数据集是 (16992, 307, 3)，每个item的raw_data应该是3个特征
            raw_data = np.array(item.get('raw_data', []))
            
            if raw_data.size > 0 and len(raw_data.shape) > 1:
                # 如果有完整的原始数据 (nodes, features)，直接使用
                recent_speeds.append(raw_data.T)  # 转置为 (features, nodes)，稍后调整
            else:
                # 否则只使用速度数据，但扩展到3个特征（用相同数据填充）
                speeds = np.array(item['speeds'])  # (nodes,)
                # 扩展到3个特征 (nodes, 3) - 模拟3个特征的情况
                expanded_data = np.tile(speeds.reshape(-1, 1), (1, 3))  # (nodes, 3)
                recent_speeds.append(expanded_data.T)  # (3, nodes)，稍后调整
        
        # 将 (timesteps, features, nodes) 转换为 (nodes, features, timesteps)
        recent_data_3d = np.stack(recent_speeds, axis=0)  # (timesteps, features, nodes)
        recent_data_3d = np.transpose(recent_data_3d, (2, 1, 0))  # (nodes, features, timesteps)
        
        # 为week和day创建类似的数据（在实际应用中，这些应该是不同时间尺度的数据）
        # 为了演示目的，我们使用最近的数据，但在实际应用中应使用相应时间尺度的数据
        week_array = recent_data_3d.copy()  # (nodes, features, timesteps)
        day_array = recent_data_3d.copy()   # (nodes, features, timesteps)
        
        # 确保形状为 (batch, nodes, features, timesteps)
        week_batch = np.expand_dims(week_array, axis=0)    # (1, nodes, features, timesteps)
        day_batch = np.expand_dims(day_array, axis=0)      # (1, nodes, features, timesteps) 
        recent_batch = np.expand_dims(recent_data_3d, axis=0) # (1, nodes, features, timesteps)
        
        # 转换为tensor并移动到设备
        week_tensor = torch.FloatTensor(week_batch).to(self.inference_engine.device)
        day_tensor = torch.FloatTensor(day_batch).to(self.inference_engine.device)
        recent_tensor = torch.FloatTensor(recent_batch).to(self.inference_engine.device)
        
        return [week_tensor, day_tensor, recent_tensor]
    
    async def process_inference_request(self, sliding_window_data: List[Dict]) -> Optional[np.ndarray]:
        """处理推理请求"""
        if not sliding_window_data or len(sliding_window_data) < 12:
            return None
        
        try:
            # 准备输入
            input_tensors = self.prepare_model_inputs(sliding_window_data)
            
            # 异步执行推理
            prediction = await self.inference_engine.run_inference_async(input_tensors)
            
            # 后处理输出
            result = self.inference_engine.postprocess_output(prediction)
            
            return result
            
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
            return None
    
    def start_worker(self):
        """启动工作器"""
        self.is_running = True
    
    def stop_worker(self):
        """停止工作器"""
        self.is_running = False
        self.executor.shutdown(wait=True)