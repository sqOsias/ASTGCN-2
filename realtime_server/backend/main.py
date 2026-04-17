# -*- coding:utf-8 -*-
"""
Real-time Vehicle Speed Prediction System - Backend Server
FastAPI + WebSocket + PyTorch Inference Engine

Architecture:
    main.py          - App entry point (this file)
    config.py        - Configuration constants
    state.py         - Global state & Pydantic models
    model_loader.py  - ASTGCN model loading & warmup
    data_loader.py   - Dataset, edge list, layout, attention matrix
    inference.py     - Model inference & metrics
    simulation.py    - Simulation loop & WebSocket broadcast
    routes.py        - REST API & WebSocket endpoints
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from routes import register_routes

# ============== FastAPI App ==============
app = FastAPI(
    title="Real-time Traffic Prediction API",
    description="ASTGCN-based real-time vehicle speed prediction system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app)

# ============== Main Entry ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
