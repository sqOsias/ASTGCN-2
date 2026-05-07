# -*- coding:utf-8 -*-
"""
Simulation engine: main loop and WebSocket broadcast.
"""

import asyncio
import json
import time
from datetime import timedelta

from config import CONFIG
from state import state
from inference import run_inference, calculate_metrics


# Rolling stage-timing buffer (most recent tick) for benchmarking / observability
_LAST_STAGE_TIMINGS_MS = {
    'slice_ms': 0.0,
    'inference_ms': 0.0,
    'metrics_ms': 0.0,
    'serialize_ms': 0.0,
    'broadcast_ms': 0.0,
    'tick_total_ms': 0.0,
}


def get_last_stage_timings():
    """Expose the last-tick server-side stage timings (for /api/benchmark)."""
    return dict(_LAST_STAGE_TIMINGS_MS)


async def simulation_loop():
    """Main simulation loop - runs every second"""
    print("Starting simulation loop...")
    
    while state.is_running:
        try:
            tick_t0 = time.perf_counter()

            # Check if we have data left
            if state.current_index >= len(state.all_data):
                state.current_index = state.history_steps  # Loop back
            
            # ---- Stage 1: data slice (virtual gear trigger) ----
            t_slice0 = time.perf_counter()
            current_frame = state.all_data[state.current_index]  # All features
            current_speed = state.speed_data[state.current_index]  # Speed only for display
            state.sliding_window.append(current_frame)
            state.real_history.append(current_speed.copy())
            t_slice1 = time.perf_counter()

            # ---- Stage 2: GPU forward inference ----
            predictions = run_inference()
            t_infer1 = time.perf_counter()

            # Store prediction for metrics
            state.prediction_history.append((state.virtual_time, predictions.copy()))

            # ---- Stage 3: metrics ----
            mae, rmse = calculate_metrics()
            t_metrics1 = time.perf_counter()

            # ---- Stage 4: build + serialize payload ----
            slice_ms = (t_slice1 - t_slice0) * 1000.0
            inference_ms = (t_infer1 - t_slice1) * 1000.0
            metrics_ms = (t_metrics1 - t_infer1) * 1000.0

            server_send_ts_ms = time.time() * 1000.0
            payload = {
                "timestamp": state.virtual_time.strftime("%Y-%m-%d %H:%M:%S"),
                "current_index": state.current_index,
                "server_send_ts_ms": server_send_ts_ms,
                "stage_timings_ms": {
                    "slice": round(slice_ms, 3),
                    "inference": round(inference_ms, 3),
                    "metrics": round(metrics_ms, 3),
                },
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
            message = json.dumps(payload)
            t_serialize1 = time.perf_counter()

            # ---- Stage 5: WebSocket broadcast ----
            await broadcast(message)
            t_broadcast1 = time.perf_counter()

            serialize_ms = (t_serialize1 - t_metrics1) * 1000.0
            broadcast_ms = (t_broadcast1 - t_serialize1) * 1000.0
            tick_total_ms = (t_broadcast1 - tick_t0) * 1000.0

            _LAST_STAGE_TIMINGS_MS.update({
                'slice_ms': round(slice_ms, 3),
                'inference_ms': round(inference_ms, 3),
                'metrics_ms': round(metrics_ms, 3),
                'serialize_ms': round(serialize_ms, 3),
                'broadcast_ms': round(broadcast_ms, 3),
                'tick_total_ms': round(tick_total_ms, 3),
                'num_clients': len(state.connections),
                'server_send_ts_ms': server_send_ts_ms,
            })

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
