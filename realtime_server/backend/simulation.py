# -*- coding:utf-8 -*-
"""
Simulation engine: main loop and WebSocket broadcast.
"""

import asyncio
import json
from datetime import timedelta

from config import CONFIG
from state import state
from inference import run_inference, calculate_metrics


async def simulation_loop():
    """Main simulation loop - runs every second"""
    print("Starting simulation loop...")
    
    while state.is_running:
        try:
            # Check if we have data left
            if state.current_index >= len(state.all_data):
                state.current_index = 36  # Loop back
            
            # Get current data (all features for model, speed for display)
            current_frame = state.all_data[state.current_index]  # All features
            current_speed = state.speed_data[state.current_index]  # Speed only for display
            
            # Update sliding window with all features
            state.sliding_window.append(current_frame)
            state.real_history.append(current_speed.copy())
            
            # Run inference
            predictions = run_inference()
            
            # Store prediction for metrics
            state.prediction_history.append((state.virtual_time, predictions.copy()))
            
            # Calculate metrics
            mae, rmse = calculate_metrics()
            
            # Build payload
            payload = {
                "timestamp": state.virtual_time.strftime("%Y-%m-%d %H:%M:%S"),
                "current_index": state.current_index,
                "system_metrics": {
                    "current_mae": round(mae, 2),
                    "current_rmse": round(rmse, 2)
                },
                "network_status": [
                    {
                        "node_id": i,
                        "current_real_speed": round(float(current_speed[i]), 2),
                        "future_pred_speeds": [round(float(p), 2) for p in predictions[i]]
                    }
                    for i in range(CONFIG['num_of_vertices'])
                ]
            }
            
            # Broadcast to all connected clients
            await broadcast(json.dumps(payload))
            
            # Advance time
            state.current_index += 1
            state.virtual_time += timedelta(minutes=CONFIG['time_interval_minutes'])
            
            # Wait for next tick
            await asyncio.sleep(1.0 / CONFIG['simulation_speed'])
            
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1.0)


async def broadcast(message: str):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for connection in state.connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in state.connections:
            state.connections.remove(conn)
