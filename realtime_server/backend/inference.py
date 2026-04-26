# -*- coding:utf-8 -*-
"""
ASTGCN inference engine and metrics calculation.
"""

import numpy as np
import torch

from config import CONFIG
from state import state


def run_inference():
    """Run ASTGCN model inference for real-time prediction"""
    num_pred_steps = CONFIG['num_for_predict']
    num_nodes = CONFIG['num_of_vertices']
    points_per_hour = CONFIG['points_per_hour']
    
    if state.model is None or state.all_data is None:
        return np.zeros((num_nodes, num_pred_steps))
    
    try:
        window_data = np.array(list(state.sliding_window))
        
        # Prepare input: normalize
        normalized = (window_data - state.mean) / state.std  # (36, N, 5)
        
        # Build week/day/hour inputs matching training data format
        week_len = CONFIG['num_of_weeks'] * points_per_hour
        day_len = CONFIG['num_of_days'] * points_per_hour
        hour_len = CONFIG['num_of_hours'] * points_per_hour

        required_len = max(week_len, day_len, hour_len)
        if window_data.shape[0] < required_len:
            return _fallback_prediction(num_nodes, num_pred_steps)
        
        hour_data = normalized[-hour_len:]
        day_data = normalized[-day_len:]
        week_data = normalized[-week_len:]
        
        # Reshape to (1, N, F, T) matching model input format
        week_input = torch.from_numpy(week_data).float().permute(1, 2, 0).unsqueeze(0).to(state.device)
        day_input = torch.from_numpy(day_data).float().permute(1, 2, 0).unsqueeze(0).to(state.device)
        hour_input = torch.from_numpy(hour_data).float().permute(1, 2, 0).unsqueeze(0).to(state.device)
        
        with torch.no_grad():
            output = state.model([week_input, day_input, hour_input])
        
        # Output is predicted speed (already in raw scale after model)
        predictions = output.squeeze(0).cpu().numpy()
        
        # Convert to km/h for display
        predictions = predictions * 1.609
        predictions = np.clip(predictions, 0, 180)
        return predictions
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: persistence with trend based on recent history
        return _fallback_prediction(num_nodes, num_pred_steps)


def _fallback_prediction(num_nodes, num_pred_steps):
    """Fallback prediction using persistence with trend"""
    window_data = np.array(list(state.sliding_window))
    current_speed = window_data[-1, :, 2] * 1.609
    trend = (window_data[-1, :, 2] - window_data[-6, :, 2]) * 1.609 / 5
    
    predictions = np.zeros((num_nodes, num_pred_steps))
    for i in range(num_pred_steps):
        predictions[:, i] = current_speed + trend * (i + 1) + np.random.randn(num_nodes) * 2.0
    
    return np.clip(predictions, 0, 180)


def calculate_metrics():
    """Calculate real-time MAE and RMSE from prediction history"""
    required_real = max(CONFIG['num_for_predict'] + 1, 2)
    if len(state.prediction_history) < 1 or len(state.real_history) < required_real:
        return 0.0, 0.0
    
    # Get predictions made 12 steps ago and compare with actual values
    mae_list = []
    rmse_list = []
    
    for i, (pred_time, predictions) in enumerate(state.prediction_history):
        # Find corresponding real values
        steps_ago = len(state.prediction_history) - i
        if steps_ago <= len(state.real_history):
            real_values = list(state.real_history)[-steps_ago]
            
            # Compare first prediction step
            pred_first = predictions[:, 0]
            error = np.abs(pred_first - real_values)
            mae_list.append(np.mean(error))
            rmse_list.append(np.sqrt(np.mean(error ** 2)))
    
    if mae_list:
        return float(np.mean(mae_list)), float(np.mean(rmse_list))
    return 0.0, 0.0
