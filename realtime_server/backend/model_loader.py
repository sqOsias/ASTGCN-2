# -*- coding:utf-8 -*-
"""
ASTGCN model loading, warmup, and weight initialization.
"""

import os

import torch

from model.astgcn import ASTGCN
from lib.utils import get_adjacency_matrix, scaled_Laplacian, cheb_polynomial

from config import CONFIG
from state import state
from data_loader import load_edge_list


def load_model():
    """Load ASTGCN model and prepare for inference"""
    print("Loading model...")
    
    # Determine device
    state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {state.device}")
    
    # Load adjacency matrix
    print(f"Loading adjacency matrix from: {CONFIG['distance_path']}")
    state.adjacency_matrix = get_adjacency_matrix(
        CONFIG['distance_path'], 
        CONFIG['num_of_vertices']
    )
    
    # Load edge list for frontend
    load_edge_list()
    
    # Compute Chebyshev polynomials
    L_tilde = scaled_Laplacian(state.adjacency_matrix)
    state.cheb_polynomials = [
        torch.from_numpy(i).float().to(state.device) 
        for i in cheb_polynomial(L_tilde, CONFIG['K'])
    ]
    
    # Create model backbone configuration (2 blocks per submodule to match saved weights)
    # Training uses _build_backbone_pair(K, polys, stride) with stride=num_of_{weeks,days,hours}
    # Each pair: [block0 with given stride, block1 with stride=1]
    def make_backbone_pair(time_conv_strides):
        return [
            {
                'K': CONFIG['K'],
                'num_of_chev_filters': CONFIG['num_of_chev_filters'],
                'num_of_time_filters': CONFIG['num_of_time_filters'],
                'time_conv_kernel_size': 3,
                'time_conv_strides': time_conv_strides,
                'cheb_polynomials': state.cheb_polynomials
            },
            {
                'K': CONFIG['K'],
                'num_of_chev_filters': CONFIG['num_of_chev_filters'],
                'num_of_time_filters': CONFIG['num_of_time_filters'],
                'time_conv_kernel_size': 3,
                'time_conv_strides': 1,
                'cheb_polynomials': state.cheb_polynomials
            }
        ]
    
    # Create model (3 submodules for week, day, hour, each with 2 blocks)
    all_backbones = [
        make_backbone_pair(CONFIG['num_of_weeks']),   # week: stride=1
        make_backbone_pair(CONFIG['num_of_days']),    # day: stride=1
        make_backbone_pair(CONFIG['num_of_hours']),   # hour: stride=3
    ]
    state.model = ASTGCN(
        num_for_prediction=CONFIG['num_for_predict'],
        all_backbones=all_backbones
    ).to(state.device)
    
    # Warm up model with dummy input to initialize lazy parameters BEFORE loading weights
    _warmup_model()
    
    # Load pretrained weights
    model_path = CONFIG['model_path']
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=state.device)
            state.model.load_state_dict(ckpt, strict=True)
            print(f"Loaded pretrained weights from: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")
            print("Using random initialization for demo")
    else:
        print(f"Warning: Model file not found: {model_path}")
        print("Using random initialization for demo")
    
    state.model.eval()
    print("Model loaded successfully!")


def _warmup_model():
    """Warm up model with dummy input to initialize lazy parameters"""
    print("Warming up model...")
    with torch.no_grad():
        # Create dummy inputs matching training data shape
        # Training adds 2 time features (time_of_day, day_of_week) to 3 raw features = 5 total
        batch_size = 1
        num_vertices = CONFIG['num_of_vertices']
        num_features = 5  # 3 raw (flow, occupy, speed) + 2 time features
        
        dummy_week = torch.randn(batch_size, num_vertices, num_features, CONFIG['num_of_weeks'] * CONFIG['points_per_hour']).to(state.device)
        dummy_day = torch.randn(batch_size, num_vertices, num_features, CONFIG['num_of_days'] * CONFIG['points_per_hour']).to(state.device)
        dummy_hour = torch.randn(batch_size, num_vertices, num_features, CONFIG['num_of_hours'] * CONFIG['points_per_hour']).to(state.device)
        
        _ = state.model([dummy_week, dummy_day, dummy_hour])
    print("Model warmed up!")
