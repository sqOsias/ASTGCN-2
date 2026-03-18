"""
WebSocket路由模块 - 处理实时通信
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List

from fastapi import WebSocket, WebSocketDisconnect

from backend.models.sliding_window import SlidingWindow
from backend.models.inference_engine import StreamingInferenceWorker


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_info: Dict[str, Dict] = {}  # 存储客户端信息
    
    async def connect(self, websocket: WebSocket):
        """建立连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        client_id = f"{id(websocket)}_{datetime.now().timestamp()}"
        self.client_info[client_id] = {
            'websocket': websocket,
            'connected_at': datetime.now(),
            'last_ping': datetime.now()
        }
        
        print(f"新客户端连接: {client_id}, 当前连接数: {len(self.active_connections)}")
        
        # 发送欢迎消息
        welcome_msg = {
            "type": "welcome",
            "client_id": client_id,
            "server_time": datetime.now().isoformat(),
            "message": "连接成功，开始接收实时交通数据"
        }
        await self.send_personal_message(welcome_msg, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """断开连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # 清理客户端信息
            for client_id, info in list(self.client_info.items()):
                if info['websocket'] == websocket:
                    print(f"客户端断开连接: {client_id}")
                    del self.client_info[client_id]
                    break
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """发送个人消息给指定客户端"""
        try:
            await websocket.send_text(json.dumps(message))
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            print(f"发送消息给客户端时出错: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接的客户端"""
        disconnected_clients = []
        
        for websocket in self.active_connections[:]:  # 使用切片副本避免修改列表时出错
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected_clients.append(websocket)
            except Exception as e:
                print(f"广播消息时出错: {e}")
                disconnected_clients.append(websocket)
        
        # 清理断开的连接
        for websocket in disconnected_clients:
            self.disconnect(websocket)
    
    def get_client_count(self) -> int:
        """获取活跃客户端数量"""
        return len(self.active_connections)
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            
            if msg_type == "ping":
                # 心跳响应
                response = {
                    "type": "pong",
                    "server_time": datetime.now().isoformat(),
                    "client_count": self.get_client_count()
                }
                await self.send_personal_message(response, websocket)
                
            elif msg_type == "request_data":
                # 请求当前数据
                # 这里应该从全局状态获取当前数据
                response = {
                    "type": "current_data",
                    "timestamp": datetime.now().isoformat(),
                    "data": data.get("data", {})
                }
                await self.send_personal_message(response, websocket)
                
            elif msg_type == "subscribe":
                # 订阅特定类型的数据
                subscription_type = data.get("subscription_type", "all")
                response = {
                    "type": "subscription_ack",
                    "subscription_type": subscription_type,
                    "message": f"已订阅 {subscription_type} 类型数据"
                }
                await self.send_personal_message(response, websocket)
                
            else:
                print(f"未知消息类型: {msg_type}")
                
        except json.JSONDecodeError:
            print("接收到无效的JSON消息")
        except Exception as e:
            print(f"处理客户端消息时出错: {e}")


# 全局连接管理器实例
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点函数"""
    await manager.connect(websocket)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            await manager.handle_client_message(websocket, data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket连接出现异常: {e}")
        manager.disconnect(websocket)


# 辅助函数：发送预测更新
async def send_prediction_update(predictions: dict):
    """发送预测更新给所有客户端"""
    message = {
        "type": "prediction_update",
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions,
        "update_type": "real_time"
    }
    await manager.broadcast(message)


# 辅助函数：发送状态更新
async def send_status_update(status: dict):
    """发送系统状态更新"""
    message = {
        "type": "system_status",
        "timestamp": datetime.now().isoformat(),
        "status": status
    }
    await manager.broadcast(message)


# 辅助函数：发送拥堵指数更新
async def send_congestion_update(congestion_data: dict):
    """发送拥堵指数更新"""
    message = {
        "type": "congestion_update",
        "timestamp": datetime.now().isoformat(),
        "congestion_data": congestion_data
    }
    await manager.broadcast(message)


# 辅助函数：发送模型性能指标
async def send_performance_metrics(metrics: dict):
    """发送模型性能指标"""
    message = {
        "type": "performance_metrics",
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    await manager.broadcast(message)