# -*- coding:utf-8 -*-
"""
Global application state and Pydantic models.
"""

from datetime import datetime
from collections import deque
from typing import List, Dict

from fastapi import WebSocket
from pydantic import BaseModel


class SystemState:
    def __init__(self):
        self.model = None
        self.device = None
        self.cheb_polynomials = None
        self.adjacency_matrix = None
        self.edge_list = []
        self.node_positions = {}  # Spectral layout positions
        
        # Data buffers
        self.all_data = None  # Full dataset
        self.speed_data = None  # Speed-only data for display
        self.current_index = 0  # Current position in dataset
        self.history_steps = 36
        self.prediction_steps = 12
        self.sliding_window = deque(maxlen=self.history_steps)
        
        # History for metrics calculation
        self.prediction_history = deque(maxlen=self.prediction_steps)
        self.real_history = deque(maxlen=48)  # Store real values for comparison
        
        # Statistics for normalization
        self.mean = None
        self.std = None
        
        # WebSocket connections
        self.connections: List[WebSocket] = []
        
        # Simulation state
        self.is_running = False
        self.simulation_task = None
        self.virtual_time = datetime(2026, 3, 28, 0, 0, 0)
        
        # Attention matrix cache
        self.attention_matrix = None

    def configure_runtime(self, history_steps: int, prediction_steps: int):
        """Reconfigure dynamic buffers based on loaded model settings."""
        self.history_steps = int(history_steps)
        self.prediction_steps = int(prediction_steps)
        self.sliding_window = deque(maxlen=self.history_steps)
        self.prediction_history = deque(maxlen=self.prediction_steps)
        self.real_history = deque(maxlen=max(self.prediction_steps * 4, self.history_steps + self.prediction_steps))


# Singleton instance
state = SystemState()


# ============== Pydantic Models ==============
class NodeStatus(BaseModel):
    node_id: int
    current_real_speed: float
    future_pred_speeds: List[float]

class SystemMetrics(BaseModel):
    current_mae: float
    current_rmse: float

class StreamPayload(BaseModel):
    timestamp: str
    system_metrics: SystemMetrics
    network_status: List[NodeStatus]

class TopologyResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
