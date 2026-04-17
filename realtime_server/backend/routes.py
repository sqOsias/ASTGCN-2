# -*- coding:utf-8 -*-
"""
API endpoints (REST + WebSocket).
"""

import asyncio
import json

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, HTTPException

from config import CONFIG
from state import state
from simulation import simulation_loop
from route_planner import plan_routes


def register_routes(app):
    """Register all API routes on the FastAPI app instance."""

    # ============== Lifecycle ==============
    @app.on_event("startup")
    async def startup_event():
        """Initialize system on startup"""
        from model_loader import load_model
        from data_loader import load_data

        print("=" * 50)
        print("Real-time Traffic Prediction System Starting...")
        print("=" * 50)
        
        load_model()
        load_data()
        
        # Start simulation
        state.is_running = True
        state.simulation_task = asyncio.create_task(simulation_loop())
        
        print("System ready!")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        state.is_running = False
        if state.simulation_task:
            state.simulation_task.cancel()

    # ============== REST API ==============
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {"status": "running", "timestamp": state.virtual_time.strftime("%Y-%m-%d %H:%M:%S")}

    @app.get("/api/topology")
    async def get_topology():
        """Get network topology with spectral layout positions"""
        nodes = [
            {
                "id": i,
                "name": f"Node_{i}",
                "category": i % 5,
                "x": state.node_positions.get(i, {}).get('x', 400),
                "y": state.node_positions.get(i, {}).get('y', 300)
            }
            for i in range(CONFIG['num_of_vertices'])
        ]
        
        return {
            "nodes": nodes,
            "edges": state.edge_list
        }

    @app.get("/api/node/{node_id}/history")
    async def get_node_history(node_id: int):
        """Get historical data for a specific node"""
        if node_id < 0 or node_id >= CONFIG['num_of_vertices']:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Get last 36 time steps from sliding window
        history = [
            {
                "time_step": i,
                "real_speed": round(float(frame[node_id]), 2)
            }
            for i, frame in enumerate(state.sliding_window)
        ]
        
        # Get prediction history for this node
        pred_history = []
        for pred_time, predictions in state.prediction_history:
            pred_history.append({
                "timestamp": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_speeds": [round(float(p), 2) for p in predictions[node_id]]
            })
        
        return {
            "node_id": node_id,
            "history": history,
            "predictions": pred_history
        }

    @app.get("/api/attention")
    async def get_attention_matrix():
        """Get attention matrix for heatmap visualization"""
        if state.attention_matrix is None:
            raise HTTPException(status_code=503, detail="Attention matrix not ready")
        
        return {
            "matrix": state.attention_matrix.tolist(),
            "size": CONFIG['num_of_vertices']
        }

    @app.get("/api/attention/{node_id}")
    async def get_node_attention(node_id: int, top_k: int = 10):
        """Get top-k attention connections for a specific node"""
        if node_id < 0 or node_id >= CONFIG['num_of_vertices']:
            raise HTTPException(status_code=404, detail="Node not found")
        
        if state.attention_matrix is None:
            raise HTTPException(status_code=503, detail="Attention matrix not ready")
        
        # Get attention weights for this node
        weights = state.attention_matrix[node_id]
        
        # Get top-k connections
        top_indices = np.argsort(weights)[-top_k:][::-1]
        
        connections = [
            {
                "target_node": int(idx),
                "attention_weight": round(float(weights[idx]), 4)
            }
            for idx in top_indices
        ]
        
        return {
            "source_node": node_id,
            "top_connections": connections
        }

    @app.get("/api/congestion/top")
    async def get_top_congestion(limit: int = 5):
        """Get top congested nodes based on predicted speeds"""
        if len(state.prediction_history) == 0:
            return {"nodes": []}
        
        # Get latest predictions
        _, latest_pred = state.prediction_history[-1]
        
        # Calculate average predicted speed for next hour
        avg_speeds = np.mean(latest_pred, axis=1)
        
        # Get nodes with lowest speeds (most congested)
        congested_indices = np.argsort(avg_speeds)[:limit]
        
        result = [
            {
                "node_id": int(idx),
                "avg_predicted_speed": round(float(avg_speeds[idx]), 2),
                "current_speed": round(float(list(state.sliding_window)[-1][idx]), 2)
            }
            for idx in congested_indices
        ]
        
        return {"nodes": result}

    @app.get("/api/stats")
    async def get_system_stats():
        """Get system statistics"""
        return {
            "total_nodes": CONFIG['num_of_vertices'],
            "total_edges": len(state.edge_list),
            "current_index": state.current_index,
            "total_samples": len(state.all_data) if state.all_data is not None else 0,
            "virtual_time": state.virtual_time.strftime("%Y-%m-%d %H:%M:%S"),
            "connected_clients": len(state.connections),
            "simulation_speed": CONFIG['simulation_speed']
        }

    @app.post("/api/simulation/speed")
    async def set_simulation_speed(speed: float):
        """Set simulation speed multiplier"""
        if speed < 0.1 or speed > 10:
            raise HTTPException(status_code=400, detail="Speed must be between 0.1 and 10")
        CONFIG['simulation_speed'] = speed
        return {"simulation_speed": speed}

    @app.post("/api/simulation/pause")
    async def pause_simulation():
        """Pause the simulation"""
        state.is_running = False
        return {"status": "paused"}

    @app.post("/api/simulation/resume")
    async def resume_simulation():
        """Resume the simulation"""
        if not state.is_running:
            state.is_running = True
            state.simulation_task = asyncio.create_task(simulation_loop())
        return {"status": "running"}

    # ============== Route Planning ==============
    @app.get("/api/route/plan")
    async def route_plan(source: int, target: int, k: int = 3):
        """Plan K time-dependent routes from source to target.

        Uses ASTGCN predicted future speeds to compute ETAs that
        account for upcoming congestion, not just current conditions.
        """
        result = plan_routes(source, target, k)
        if 'error' in result and 'routes' not in result:
            raise HTTPException(status_code=400, detail=result['error'])
        return result

    @app.get("/api/route/components")
    async def route_components():
        """Return connected component info so frontend can validate
        source/target selection before calling plan."""
        from route_planner import build_adjacency
        from collections import deque

        adj = build_adjacency()
        visited = set()
        components = []
        for n in range(CONFIG['num_of_vertices']):
            if n in visited:
                continue
            comp = []
            queue = deque([n])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nbr, _ in adj.get(node, []):
                    if nbr not in visited:
                        queue.append(nbr)
            components.append(comp)

        components.sort(key=len, reverse=True)
        return {
            'num_components': len(components),
            'components': [
                {'id': i, 'size': len(c), 'nodes': c}
                for i, c in enumerate(components)
            ]
        }

    # ============== WebSocket ==============
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time data streaming"""
        await websocket.accept()
        state.connections.append(websocket)
        print(f"Client connected. Total connections: {len(state.connections)}")
        
        try:
            while True:
                # Keep connection alive, handle any client messages
                data = await websocket.receive_text()
                
                # Handle client commands
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif msg.get("type") == "request_topology":
                        topology = await get_topology()
                        await websocket.send_text(json.dumps({
                            "type": "topology",
                            "data": topology
                        }))
                except json.JSONDecodeError:
                    pass
                    
        except WebSocketDisconnect:
            print("Client disconnected")
        finally:
            if websocket in state.connections:
                state.connections.remove(websocket)
