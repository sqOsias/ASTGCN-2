# -*- coding:utf-8 -*-
"""
Configuration constants for the real-time traffic prediction system.
"""

import os

CONFIG = {
    'num_of_vertices': 307,
    'num_for_predict': 12,  # Predict 12 time steps (1 hour)
    'points_per_hour': 12,
    'num_of_weeks': 1,
    'num_of_days': 1,
    'num_of_hours': 3,  # 3 hours history = 36 time steps
    'K': 3,  # Chebyshev polynomial order
    'num_of_chev_filters': 64,
    'num_of_time_filters': 64,
    'time_conv_strides': 1,
    'time_interval_minutes': 5,  # Each time step = 5 minutes
    'simulation_speed': 1.0,  # Real 1 second = 1 traffic time step (5 min)
    'data_path': os.path.join(os.path.dirname(__file__), '../../data/PEMS04/pems04.npz'),
    'distance_path': os.path.join(os.path.dirname(__file__), '../../data/PEMS04/distance.csv'),
    'model_path': os.path.join(os.path.dirname(__file__), '../../results/ASTGCN_lr0p001/0_0_20260318033244/checkpoints/best_model.pth'),
}
