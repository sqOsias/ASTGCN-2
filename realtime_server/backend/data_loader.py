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
    """Compute node positions using metro-map style layout for highway corridors"""
    num_nodes = CONFIG['num_of_vertices']
    print("Computing metro-map style layout...")
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for edge in state.edge_list:
        G.add_edge(edge['source'], edge['target'], weight=edge['distance'])
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Found {len(components)} connected components")
    
    # For each component, find the longest path (main corridor)
    def find_longest_path(subgraph):
        """Find longest path in subgraph using BFS from endpoints"""
        nodes = list(subgraph.nodes())
        if len(nodes) <= 1:
            return nodes
        
        # Find leaf nodes (degree 1) as potential endpoints
        leaves = [n for n in nodes if subgraph.degree(n) == 1]
        if not leaves:
            leaves = nodes[:2]
        
        # BFS to find farthest node from first leaf
        start = leaves[0]
        distances = nx.single_source_shortest_path_length(subgraph, start)
        end = max(distances.keys(), key=lambda x: distances[x])
        
        # Get shortest path as main corridor
        try:
            path = nx.shortest_path(subgraph, start, end)
            return path
        except:
            return nodes
    
    # Layout parameters
    canvas_width, canvas_height = 750, 650
    margin = 40
    
    positions = {}
    y_offset = margin
    corridor_height = 35  # Height between corridors
    
    # Sort components by size (largest first)
    sorted_components = sorted(components, key=len, reverse=True)
    
    for comp_idx, component in enumerate(sorted_components):
        subgraph = G.subgraph(component)
        main_path = find_longest_path(subgraph)
        
        if len(main_path) == 0:
            continue
        
        # Calculate x positions along the corridor
        path_len = len(main_path)
        x_step = (canvas_width - 2 * margin) / max(path_len - 1, 1)
        
        # Place main corridor nodes
        placed = set()
        for i, node in enumerate(main_path):
            positions[node] = {
                'x': margin + i * x_step,
                'y': y_offset
            }
            placed.add(node)
        
        # Place branch nodes (nodes connected to main path but not on it)
        branch_offset = 20
        for node in component:
            if node in placed:
                continue
            
            # Find connected node on main path
            neighbors = list(subgraph.neighbors(node))
            parent = None
            for n in neighbors:
                if n in placed:
                    parent = n
                    break
            
            if parent is not None:
                px, py = positions[parent]['x'], positions[parent]['y']
                # Alternate above/below the main path
                direction = 1 if (node % 2 == 0) else -1
                positions[node] = {
                    'x': px + np.random.uniform(-10, 10),
                    'y': py + direction * branch_offset
                }
                placed.add(node)
        
        # Handle remaining unplaced nodes
        for node in component:
            if node not in placed:
                positions[node] = {
                    'x': margin + np.random.uniform(0, canvas_width - 2*margin),
                    'y': y_offset + np.random.uniform(-15, 15)
                }
        
        y_offset += corridor_height + 25
        
        # Wrap to next "column" if too many corridors
        if y_offset > canvas_height - margin:
            y_offset = margin
    
    state.node_positions = positions
    print(f"Metro-map layout completed for {num_nodes} nodes")


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
