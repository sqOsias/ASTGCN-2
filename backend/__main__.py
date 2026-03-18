"""
主应用文件 - 整合所有组件
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.websocket import manager, websocket_endpoint
from backend.config import (
    DATA_CONFIG,
    INFERENCE_CONFIG,
    MODEL_CONFIG,
    SLIDING_WINDOW_CONFIG,
    get_config
)
from backend.models.inference_engine import StreamingInferenceWorker
from backend.models.sliding_window import SlidingWindow
from backend.utils.data_processor import TrafficDataProcessor, ModelDataConverter
from model.upgrade.astgcn_upgrade import UpgradeASTGCN


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="实时车辆速度预测系统", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model: Optional[UpgradeASTGCN] = None
inference_worker: Optional[StreamingInferenceWorker] = None
sliding_window: Optional[SlidingWindow] = None
data_processor: Optional[TrafficDataProcessor] = None
model_converter: Optional[ModelDataConverter] = None
device = torch.device(MODEL_CONFIG['device'])

# 数据生产者
class RealTimeDataProducer:
    """实时数据生产者"""
    
    def __init__(self):
        self.current_time_idx = 0
        self.dataset = None
        self.is_running = False
        self.loop = None
        
        # 加载数据集
        try:
            if os.path.exists(DATA_CONFIG['dataset_path']):
                self.dataset = np.load(DATA_CONFIG['dataset_path'])
                logger.info(f"数据集加载成功: {DATA_CONFIG['dataset_path']}")
                logger.info(f"数据形状: {self.dataset['data'].shape}")
            else:
                logger.warning(f"数据集不存在: {DATA_CONFIG['dataset_path']}")
                # 创建模拟数据
                self.dataset = {
                    'data': np.random.rand(1000, 307, 5).astype(np.float32) * 60  # 模拟速度数据 (0-60 km/h)
                }
                logger.info("使用模拟数据进行演示")
        except Exception as e:
            logger.error(f"数据集加载失败: {e}")
            # 创建模拟数据
            self.dataset = {
                'data': np.random.rand(1000, 307, 5).astype(np.float32) * 60
            }
    
    async def start_streaming(self):
        """启动数据流"""
        self.is_running = True
        logger.info("开始数据流...")
        
        while self.is_running:
            try:
                if self.dataset is not None:
                    # 获取当前时间步的数据
                    total_timesteps = self.dataset['data'].shape[0]
                    current_data = self.dataset['data'][self.current_time_idx % total_timesteps]
                    
                    # 预处理数据
                    if data_processor:
                        processed_data, _ = data_processor.process_speed_data(current_data[:, 0:1])  # 取第一个特征（速度）
                        # 重新塑形为 (nodes, features, timesteps=1)
                        if len(processed_data.shape) == 2:  # (nodes, timesteps)
                            processed_data = np.expand_dims(processed_data, axis=1)  # (nodes, features=1, timesteps)
                    
                    # 创建窗口项目 - 包含完整的3个特征
                    timestamp = datetime.now().isoformat()
                    window_item = {
                        "timestamp": timestamp,
                        "speeds": current_data[:, 0].tolist(),  # 取速度特征
                        "raw_data": current_data.tolist(),  # 包含全部3个特征
                        "node_count": current_data.shape[0],
                        "feature_count": current_data.shape[1],  # 特征数
                        "congestion_level": self.calculate_congestion_level(current_data[:, 0].tolist())
                    }
                    
                    # 更新滑动窗口
                    if sliding_window:
                        sliding_window.append(window_item)
                        
                        # 如果窗口满了，触发推理
                        if sliding_window.is_full():
                            await self.trigger_inference()
                
                # 更新索引
                self.current_time_idx += 1
                
                # 等待指定时间间隔
                await asyncio.sleep(SLIDING_WINDOW_CONFIG['update_interval'])
                
            except Exception as e:
                logger.error(f"数据流处理错误: {e}")
                await asyncio.sleep(1)  # 出错时等待1秒再继续
    
    def calculate_congestion_level(self, speeds):
        """计算拥堵等级"""
        avg_speed = sum(speeds) / len(speeds)
        if avg_speed > 50:
            return 'low'
        elif avg_speed > 30:
            return 'medium'
        else:
            return 'high'
    
    async def trigger_inference(self):
        """触发推理"""
        if inference_worker and sliding_window and sliding_window.get_size() >= 12:
            try:
                # 获取滑动窗口数据
                window_data = sliding_window.get_recent()
                
                # 执行推理
                predictions = await inference_worker.process_inference_request(window_data)
                
                if predictions is not None:
                    # 发送预测结果
                    prediction_msg = {
                        "type": "prediction_update",
                        "timestamp": datetime.now().isoformat(),
                        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                        "current_data": sliding_window.get_recent()[-1]["speeds"] if sliding_window.get_recent() else [],
                        "node_count": MODEL_CONFIG['num_of_vertices']
                    }
                    
                    await manager.broadcast(prediction_msg)
                    
                    # 计算并发送拥堵指数
                    congestion_index = sliding_window.get_congestion_index()
                    congestion_msg = {
                        "type": "congestion_update",
                        "timestamp": datetime.now().isoformat(),
                        "congestion_data": {
                            "index": congestion_index,
                            "distribution": sliding_window.get_speed_distribution()
                        }
                    }
                    await manager.broadcast(congestion_msg)
                    
            except Exception as e:
                logger.error(f"推理过程中出现错误: {e}")
    
    def stop_streaming(self):
        """停止数据流"""
        self.is_running = False
        logger.info("数据流已停止")

# 创建数据生产者实例
data_producer = RealTimeDataProducer()


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global model, inference_worker, sliding_window, data_processor, model_converter
    
    logger.info("正在初始化系统组件...")
    
    # 初始化数据处理器
    data_processor = TrafficDataProcessor()
    
    # 初始化模型数据转换器
    model_converter = ModelDataConverter()
    
    # 初始化滑动窗口
    sliding_window = SlidingWindow(maxlen=SLIDING_WINDOW_CONFIG['maxlen'])
    
    # 初始化模型
    try:
        # 需要从 model_config 获取 backbones 配置
        from model.model_config import get_backbones
        all_backbones = get_backbones('configurations/PEMS04.conf', DATA_CONFIG['distance_path'], device)
        
        model = UpgradeASTGCN(
            num_of_features=3,  # 根据数据集特征数设定 (PEMS04数据集有3个特征)
            num_for_prediction=MODEL_CONFIG['num_for_prediction'],
            all_backbones=all_backbones,
            num_of_vertices=MODEL_CONFIG['num_of_vertices'],
            spatial_mode=MODEL_CONFIG['spatial_mode'],
            temporal_mode=MODEL_CONFIG['temporal_mode'],
            adaptive_graph_cfg={
                'embedding_dim': 10,
                'sparse_ratio': 0.2,
                'directed': True
            },
            transformer_cfg={
                'd_model': 32,
                'n_heads': 2,
                'e_layers': 2,
                'dropout': 0.1,
                'max_len': 36,
                'factor': 5
            }
        )
        
        # 尝试加载预训练权重
        param_dir = Path("params")
        if param_dir.exists():
            checkpoint_files = list(param_dir.rglob("*net_*.params"))
            if checkpoint_files:
                checkpoint_path = str(checkpoint_files[0])
                try:
                    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
                    logger.info(f"模型权重加载成功: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"无法加载预训练权重: {e}，使用随机初始化")
            else:
                logger.info("未找到预训练权重，使用随机初始化")
        
        model.to(device)
        model.eval()
        logger.info("模型加载完成")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        # 创建一个简化的模型用于演示
        model = UpgradeASTGCN(
            num_of_vertices=MODEL_CONFIG['num_of_vertices'],
            len_input=MODEL_CONFIG['len_input'],
            num_for_prediction=MODEL_CONFIG['num_for_prediction'],
            spatial_mode=0,  # 使用基础模式
            temporal_mode=0
        )
        model.to(device)
        model.eval()
        logger.info("使用简化模型进行演示")
    
    # 初始化推理工作器
    inference_worker = StreamingInferenceWorker(model, device)
    inference_worker.start_worker()
    
    logger.info("系统初始化完成")
    
    # 启动数据流
    asyncio.create_task(data_producer.start_streaming())


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("正在关闭系统...")
    data_producer.stop_streaming()
    if inference_worker:
        inference_worker.stop_worker()
    logger.info("系统已关闭")


@app.websocket("/ws")
async def websocket_endpoint_wrapper(websocket: WebSocket):
    """WebSocket端点包装器"""
    await websocket_endpoint(websocket)


# 移除根路径的API路由，让静态文件路由生效


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model_loaded": model is not None,
            "inference_worker_running": inference_worker is not None,
            "sliding_window_ready": sliding_window is not None,
            "data_producer_running": data_producer.is_running,
            "active_connections": len(manager.active_connections)
        }
    }


@app.get("/status")
async def get_system_status():
    """获取系统状态"""
    status = {
        "model_loaded": model is not None,
        "model_on_device": str(device) if model else "N/A",
        "window_size": sliding_window.get_size() if sliding_window else 0,
        "window_capacity": SLIDING_WINDOW_CONFIG['maxlen'] if sliding_window else 0,
        "active_connections": len(manager.active_connections),
        "data_producer_running": data_producer.is_running,
        "total_processed_samples": data_producer.current_time_idx if hasattr(data_producer, 'current_time_idx') else 0
    }
    
    if sliding_window:
        status.update({
            "window_statistics": sliding_window.get_statistics(),
            "speed_distribution": sliding_window.get_speed_distribution(),
            "congestion_index": sliding_window.get_congestion_index()
        })
    
    return status


@app.post("/control/start_streaming")
async def start_streaming():
    """启动数据流"""
    if not data_producer.is_running:
        asyncio.create_task(data_producer.start_streaming())
        return {"message": "数据流已启动"}
    else:
        return {"message": "数据流已在运行中"}


@app.post("/control/stop_streaming")
async def stop_streaming():
    """停止数据流"""
    data_producer.stop_streaming()
    return {"message": "数据流已停止"}


# 挂载静态文件
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
    logger.info(f"静态文件已挂载: {frontend_path.absolute()}")
else:
    logger.warning(f"前端目录不存在: {frontend_path.absolute()}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=get_config('host'),
        port=get_config('port'),
        reload=get_config('reload'),
        log_level=get_config('log_level')
    )