# -*- coding:utf-8 -*-
"""
Data loading: PEMS04 dataset, edge list, node layout, attention matrix.
"""

import csv

import numpy as np
import networkx as nx
from collections import defaultdict

from config import CONFIG
from state import state


# ============== Edge List & Layout ==============
def load_edge_list():
    """Load edge list from distance.csv"""
    state.edge_list = []
    with open(CONFIG['distance_path'], 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            state.edge_list.append({
                'source': int(row[0]),
                'target': int(row[1]),
                'distance': float(row[2]) if len(row) > 2 else 1.0
            })
    print(f"Loaded {len(state.edge_list)} edges")
    
    # Compute node positions using spectral layout
    compute_node_positions()


def compute_node_positions():
    """Compute node positions using force-directed spring layout"""
    num_nodes = CONFIG['num_of_vertices']
    print("Computing force-directed layout...")
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for edge in state.edge_list:
        G.add_edge(edge['source'], edge['target'], weight=edge['distance'])
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    # Canvas dimensions
    canvas_width, canvas_height = 750, 620
    margin = 50
    
    # Compute spring layout with good parameters
    # k controls optimal distance between nodes, higher = more spread
    # iterations ensures convergence
    pos = nx.spring_layout(
        G, 
        k=2.5,           # Optimal distance between nodes
        iterations=150,  # More iterations for better convergence
        seed=42,         # Fixed seed for reproducible layout
        scale=1.0
    )
    
    # Get bounds of computed positions
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Scale to canvas with margin
    scale_x = (canvas_width - 2 * margin) / max(max_x - min_x, 0.001)
    scale_y = (canvas_height - 2 * margin) / max(max_y - min_y, 0.001)
    scale = min(scale_x, scale_y)  # Uniform scaling to preserve aspect ratio
    
    # Center offset
    center_x = (canvas_width - (max_x - min_x) * scale) / 2
    center_y = (canvas_height - (max_y - min_y) * scale) / 2
    
    # Convert to canvas coordinates
    positions = {}
    for node, (x, y) in pos.items():
        positions[node] = {
            'x': (x - min_x) * scale + center_x,
            'y': (y - min_y) * scale + center_y
        }
    
    state.node_positions = positions
    print(f"Force-directed layout completed for {num_nodes} nodes")


# ============== Traffic Data ==============
def _build_time_features(sequence_length, num_of_vertices, points_per_hour):
    """Build time features (time_of_day, day_of_week) matching training data prep"""
    points_per_day = 24 * points_per_hour
    time_index = np.arange(sequence_length)
    time_of_day = (time_index % points_per_day) / float(points_per_day)
    day_of_week = ((time_index // points_per_day) % 7) / 7.0
    time_of_day = np.repeat(time_of_day[:, np.newaxis], num_of_vertices, axis=1)[:, :, np.newaxis]
    day_of_week = np.repeat(day_of_week[:, np.newaxis], num_of_vertices, axis=1)[:, :, np.newaxis]
    return np.concatenate([time_of_day, day_of_week], axis=2).astype(np.float32)


def load_data():
    """Load traffic data from npz file"""
    print(f"Loading data from: {CONFIG['data_path']}")
    data = np.load(CONFIG['data_path'])
    raw_data = data['data']  # Shape: (T, N, 3) - flow, occupy, speed
    
    # Add time features to match training data (3 raw + 2 time = 5 features)
    time_features = _build_time_features(
        raw_data.shape[0], raw_data.shape[1], CONFIG['points_per_hour'])
    state.all_data = np.concatenate([raw_data, time_features], axis=2)  # Shape: (T, N, 5)
    
    # Speed is in mph, convert to km/h for display (1 mph = 1.609 km/h)
    state.speed_data = raw_data[:, :, 2] * 1.609  # Speed feature (index 2), converted to km/h
    
    # Calculate statistics for normalization (per feature, matching training)
    state.mean = np.mean(state.all_data, axis=(0, 1))  # Shape: (5,)
    state.std = np.std(state.all_data, axis=(0, 1))  # Shape: (5,)
    state.std[state.std == 0] = 1.0  # Avoid division by zero
    
    print(f"Data loaded: shape={state.all_data.shape}, num_features={state.all_data.shape[2]}")
    
    # Initialize sliding window with first 36 frames
    for i in range(36):
        state.sliding_window.append(state.all_data[i])
    state.current_index = 36
    
    # Generate synthetic attention matrix
    generate_attention_matrix()


# ============== Attention Matrix ==============
def generate_attention_matrix():
    """Generate synthetic attention matrix based on adjacency"""
    # Create base attention from adjacency
    adj = state.adjacency_matrix.copy()
    
    # Add self-attention
    np.fill_diagonal(adj, 1.0)
    
    # Normalize
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    attention = adj / row_sum
    
    # Add some randomness to simulate learned attention
    noise = np.random.randn(*attention.shape) * 0.1
    attention = np.clip(attention + noise, 0, 1)
    
    # Re-normalize
    row_sum = attention.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    state.attention_matrix = attention / row_sum
