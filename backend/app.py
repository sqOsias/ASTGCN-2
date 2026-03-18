"""
FastAPI后端应用 - 实时车辆速度预测系统
包含WebSocket服务和流式推理引擎
"""
import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model.upgrade.astgcn_upgrade import UpgradeASTGCN


app = FastAPI(title="实时车辆速度预测系统", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型加载
model: Optional[UpgradeASTGCN] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量
clients = []
sliding_window = deque(maxlen=12)  # 固定长度12的滑动窗口
prediction_buffer = {}

class PredictionResult(BaseModel):
    timestamp: str
    predicted_speeds: List[float]
    actual_speeds: List[float]
    mae: float
    rmse: float

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.disconnect(connection)

manager = WebSocketManager()

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    global model
    
    # 这里应该加载训练好的模型
    try:
        # 临时创建模型实例，实际使用时应该加载预训练权重
        # 加载配置参数
        import configparser
        config = configparser.ConfigParser()
        config.read('configurations/PEMS04.conf')
        
        num_of_vertices = int(config['Data']['num_of_vertices'])  # 307
        num_for_prediction = int(config['Data']['num_for_predict'])  # 12
        
        # 创建模型实例
        model = UpgradeASTGCN(
            num_of_vertices=num_of_vertices,
            len_input=36,
            num_for_prediction=num_for_prediction,
            spatial_mode=1,
            temporal_mode=1
        )
        
        # 尝试加载预训练权重
        try:
            checkpoint_path = "params/ASTGCN_lr0p001/1_1_20260317110708/net_1.params"  # 示例路径
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"模型权重加载成功: {checkpoint_path}")
        except Exception as e:
            print(f"无法加载预训练权重，使用随机初始化: {e}")
        
        model.to(device)
        model.eval()
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点 - 实时数据流处理"""
    await manager.connect(websocket)
    try:
        while True:
            # 等待客户端消息（可选，主要用于心跳或其他控制命令）
            data = await websocket.receive_text()
            # 解析控制命令
            command = json.loads(data)
            
            if command.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class DataProducer:
    """数据生产者 - 模拟实时数据流"""
    
    def __init__(self):
        self.current_time_idx = 0
        self.dataset = None
        self.is_running = False
        
        # 加载测试数据
        try:
            self.dataset = np.load('data/PEMS04/pems04.npz')
            print(f"数据集加载成功，形状: {self.dataset['data'].shape}")
        except Exception as e:
            print(f"数据集加载失败: {e}")
    
    async def start_streaming(self):
        """启动数据流"""
        if self.dataset is None:
            print("没有可用的数据集，无法启动数据流")
            return
            
        self.is_running = True
        while self.is_running:
            if self.dataset is not None:
                # 从数据集中获取当前时间步的数据
                total_timesteps = self.dataset['data'].shape[0]
                current_data = self.dataset['data'][self.current_time_idx % total_timesteps]
                
                # 更新滑动窗口
                if len(current_data.shape) == 2:  # (nodes, features)
                    # 取速度特征（假设第一个特征是速度）
                    speed_data = current_data[:, 0] if current_data.shape[1] > 0 else current_data[:, 0]
                    
                    # 添加时间戳信息
                    timestamp = datetime.now().isoformat()
                    window_item = {
                        "timestamp": timestamp,
                        "speeds": speed_data.tolist(),
                        "node_count": len(speed_data)
                    }
                    
                    sliding_window.append(window_item)
                    
                    # 触发推理
                    await self.run_inference()
                
                # 更新索引
                self.current_time_idx += 1
                
            # 模拟时间流逝（现实中是5分钟，这里压缩为1秒）
            await asyncio.sleep(1)
    
    async def run_inference(self):
        """运行模型推理"""
        if len(sliding_window) >= 12 and model is not None:
            try:
                # 准备输入数据
                input_data = []
                for item in list(sliding_window)[-12:]:  # 最近12个时间步
                    speeds = np.array(item['speeds']).reshape(-1, 1)  # 添加特征维度
                    input_data.append(speeds)
                
                # 转换为模型输入格式
                input_tensor = np.stack(input_data, axis=-1)  # (nodes, features, timesteps)
                input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0)  # 添加batch维度
                input_tensor = input_tensor.to(device)
                
                # 运行推理
                with torch.no_grad():
                    # 为简化，构造完整的输入格式 [week, day, recent]
                    # 在实际应用中，需要根据模型的实际输入要求调整
                    dummy_week = input_tensor.clone()
                    dummy_day = input_tensor.clone()
                    recent_input = input_tensor  # 使用最近的数据作为输入
                    
                    prediction = model([dummy_week, dummy_day, recent_input])
                    
                    # 处理预测结果
                    pred_np = prediction.cpu().numpy()[0]  # 移除batch维度
                    
                    # 发送预测结果给所有客户端
                    result = {
                        "type": "prediction_update",
                        "timestamp": datetime.now().isoformat(),
                        "predictions": pred_np.tolist(),  # 预测的速度数据
                        "current_data": list(sliding_window)[-1]["speeds"] if sliding_window else [],
                        "node_count": len(pred_np) if isinstance(pred_np, np.ndarray) else len(pred_np) if isinstance(pred_np, list) else 0
                    }
                    
                    await manager.broadcast(result)
                    
            except Exception as e:
                print(f"推理过程中出现错误: {e}")
    
    def stop_streaming(self):
        """停止数据流"""
        self.is_running = False

# 创建数据生产者实例
data_producer = DataProducer()

@app.post("/start_streaming")
async def start_streaming():
    """启动数据流的API端点"""
    asyncio.create_task(data_producer.start_streaming())
    return {"message": "数据流已启动"}

@app.get("/status")
async def get_status():
    """获取系统状态"""
    return {
        "model_loaded": model is not None,
        "window_size": len(sliding_window),
        "client_count": len(manager.active_connections),
        "data_producer_running": data_producer.is_running
    }

# 挂载静态文件目录
try:
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
except:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)