# -*- coding:utf-8 -*-
"""
Time-dependent dynamic route planning engine.

Core idea: use ASTGCN predicted future speeds (not just current speeds)
to compute routes that avoid upcoming congestion.

Algorithm:
  - Time-dependent Dijkstra with a "virtual clock" that advances as
    the vehicle traverses edges.
  - Edge speed = average of predicted speeds at the two endpoint nodes
    at the virtual clock's current time.
  - When virtual time exceeds the prediction horizon (60 min / 12 steps),
    falls back to the last available prediction step.
  - Top-K diverse routes via edge-penalty method: after finding the
    optimal route, penalize its edges and re-run to find alternatives.

Units:
  - Distance: meters  (from distance.csv)
  - Speed: km/h       (converted from mph in data_loader)
  - Time: minutes      (for ETA display)
"""

import heapq
from collections import defaultdict

import numpy as np

from config import CONFIG
from state import state

# Minimum speed floor to avoid division-by-zero (km/h)
MIN_SPEED_KMH = 5.0

# Penalty multiplier for edges already used in previous routes
EDGE_PENALTY = 3.0


# ============== Graph Construction ==============

def build_adjacency():
    """Build bidirectional adjacency dict from state.edge_list.

    Returns:
        dict[int, list[tuple[int, float]]]:  node -> [(neighbor, distance_m), ...]
    """
    adj = defaultdict(list)
    for edge in state.edge_list:
        src, dst, dist = edge['source'], edge['target'], edge['distance']
        adj[src].append((dst, dist))
        adj[dst].append((src, dist))
    return dict(adj)


def _build_edge_dist_lookup(adj):
    """Build a fast (src, dst) -> distance lookup from adjacency dict."""
    lookup = {}
    for node, neighbors in adj.items():
        for nbr, dist in neighbors:
            lookup[(node, nbr)] = dist
    return lookup


# ============== Speed Lookup ==============

def get_speed_at_time(node_id, time_offset_min, current_speeds, pred_speeds):
    """Get the predicted speed of a node at a future time offset.

    Args:
        node_id:          Sensor node index (0-306).
        time_offset_min:  Minutes elapsed since departure.
        current_speeds:   (N,) array of current real speeds in km/h.
        pred_speeds:      (N, 12) array of predicted speeds in km/h.

    Returns:
        float: speed in km/h, floored at MIN_SPEED_KMH.
    """
    step_duration = CONFIG['time_interval_minutes']  # 5 min

    if time_offset_min < step_duration:
        # Still within the first 5-minute window -> use current real speed
        return max(float(current_speeds[node_id]), MIN_SPEED_KMH)

    # Map to 0-indexed prediction step
    step = int(time_offset_min / step_duration) - 1

    # Clamp: if beyond prediction horizon, use last available step
    step = min(step, CONFIG['num_for_predict'] - 1)  # max index = 11

    return max(float(pred_speeds[node_id, step]), MIN_SPEED_KMH)


def _edge_speed(src, dst, time_offset_min, current_speeds, pred_speeds):
    """Edge traversal speed = arithmetic mean of two endpoint speeds."""
    s1 = get_speed_at_time(src, time_offset_min, current_speeds, pred_speeds)
    s2 = get_speed_at_time(dst, time_offset_min, current_speeds, pred_speeds)
    return (s1 + s2) / 2.0


# ============== Time-Dependent Dijkstra ==============

def _td_dijkstra(adj, source, target, current_speeds, pred_speeds):
    """Time-dependent Dijkstra that advances a virtual clock along edges.

    Returns:
        dict with keys: found, path, total_distance_m, eta_minutes
    """
    # Priority queue entries: (cumulative_minutes, node_id, path_list, total_dist_m)
    pq = [(0.0, source, [source], 0.0)]
    best = {}  # node -> best cumulative time seen

    while pq:
        t_min, node, path, dist_m = heapq.heappop(pq)

        if node in best:
            continue
        best[node] = t_min

        if node == target:
            return {
                'found': True,
                'path': path,
                'total_distance_m': dist_m,
                'total_distance_km': round(dist_m / 1000.0, 2),
                'eta_minutes': round(t_min, 1),
            }

        for nbr, edge_dist in adj.get(node, []):
            if nbr in best:
                continue
            speed = _edge_speed(node, nbr, t_min, current_speeds, pred_speeds)
            edge_time = (edge_dist / 1000.0) / speed * 60.0  # minutes
            heapq.heappush(pq, (t_min + edge_time, nbr, path + [nbr], dist_m + edge_dist))

    return {'found': False, 'path': [], 'total_distance_km': 0, 'eta_minutes': 0, 'total_distance_m': 0}


# ============== Route ETA Re-computation ==============

def _compute_route_eta(path, edge_lookup, current_speeds, pred_speeds):
    """Re-compute accurate ETA/distance for an already-known path."""
    total_time = 0.0
    total_dist = 0.0
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        dist = edge_lookup.get((src, dst), 500.0)
        speed = _edge_speed(src, dst, total_time, current_speeds, pred_speeds)
        total_time += (dist / 1000.0) / speed * 60.0
        total_dist += dist
    return {
        'total_distance_m': total_dist,
        'total_distance_km': round(total_dist / 1000.0, 2),
        'eta_minutes': round(total_time, 1),
    }


# ============== Speed Profile ==============

def _build_speed_profile(path, edge_lookup, current_speeds, pred_speeds):
    """Build per-node speed profile along the route for frontend charts."""
    profile = []
    cum_time = 0.0
    cum_dist = 0.0

    for i, node in enumerate(path):
        speed = get_speed_at_time(node, cum_time, current_speeds, pred_speeds)
        profile.append({
            'node_id': node,
            'time_min': round(cum_time, 1),
            'speed_kmh': round(speed, 1),
            'distance_km': round(cum_dist / 1000.0, 2),
        })
        if i < len(path) - 1:
            nxt = path[i + 1]
            dist = edge_lookup.get((node, nxt), 500.0)
            spd = _edge_speed(node, nxt, cum_time, current_speeds, pred_speeds)
            cum_time += (dist / 1000.0) / spd * 60.0
            cum_dist += dist

    return profile


# ============== K Diverse Routes ==============

def _prediction_coverage(eta_minutes):
    """Compute what fraction of the trip is covered by model predictions.

    Returns a float in [0, 1].  A trip of 3 min mostly uses current speeds
    (coverage ≈ 0); a trip of 30 min uses many prediction steps (coverage ≈ 0.5).
    """
    step_min = CONFIG['time_interval_minutes']  # 5
    horizon = CONFIG['num_for_predict'] * step_min  # 60
    if eta_minutes <= step_min:
        return 0.0  # entire trip within current-speed window
    covered = min(eta_minutes, horizon) - step_min
    return round(covered / max(eta_minutes, 0.1), 2)


def _find_k_routes(adj, source, target, current_speeds, pred_speeds, k=3):
    """Find K diverse routes using edge-penalty method.

    Searches up to 3*k attempts to find k *distinct* paths.
    """
    edge_lookup = _build_edge_dist_lookup(adj)
    routes = []
    seen_paths = set()  # deduplicate
    penalty_edges = set()
    max_attempts = k * 3

    for attempt in range(max_attempts):
        if len(routes) >= k:
            break

        # Build penalised adjacency for diversity
        if attempt == 0:
            search_adj = adj
        else:
            search_adj = {}
            penalty_mult = EDGE_PENALTY * (1 + attempt * 0.5)  # escalating penalty
            for node, neighbors in adj.items():
                search_adj[node] = []
                for nbr, dist in neighbors:
                    key = (min(node, nbr), max(node, nbr))
                    pen = penalty_mult if key in penalty_edges else 1.0
                    search_adj[node].append((nbr, dist * pen))

        result = _td_dijkstra(search_adj, source, target, current_speeds, pred_speeds)
        if not result['found']:
            break

        # Deduplicate: skip if path is identical to one already found
        path_key = tuple(result['path'])
        if path_key in seen_paths:
            # Still mark edges so next attempt penalises harder
            for j in range(len(result['path']) - 1):
                a, b = result['path'][j], result['path'][j + 1]
                penalty_edges.add((min(a, b), max(a, b)))
            continue
        seen_paths.add(path_key)

        # Re-compute with original distances for accurate ETA (penalised dists are inflated)
        if attempt > 0:
            accurate = _compute_route_eta(result['path'], edge_lookup, current_speeds, pred_speeds)
            result.update(accurate)

        result['speed_profile'] = _build_speed_profile(
            result['path'], edge_lookup, current_speeds, pred_speeds)
        result['route_index'] = len(routes)
        result['prediction_coverage'] = _prediction_coverage(result['eta_minutes'])
        routes.append(result)

        # Mark edges of this route for future penalty
        for j in range(len(result['path']) - 1):
            a, b = result['path'][j], result['path'][j + 1]
            penalty_edges.add((min(a, b), max(a, b)))

    return routes


# ============== Static-Speed Baseline ==============

def _static_eta(adj, source, target, current_speeds):
    """Compute ETA using only current (static) speeds for comparison."""
    static_pred = np.tile(current_speeds[:, np.newaxis], (1, CONFIG['num_for_predict']))
    result = _td_dijkstra(adj, source, target, current_speeds, static_pred)
    return round(result['eta_minutes'], 1) if result['found'] else None


# ============== Public API ==============

def plan_routes(source: int, target: int, k: int = 3) -> dict:
    """Main entry point — called from the route planning API endpoint.

    Returns a JSON-serialisable dict with route results.
    """
    num_v = CONFIG['num_of_vertices']
    if source < 0 or source >= num_v or target < 0 or target >= num_v:
        return {'error': f'Node IDs must be in [0, {num_v - 1}]'}
    if source == target:
        return {'error': 'Source and target must differ'}

    adj = build_adjacency()

    # Retrieve latest speed data
    idx = max(state.current_index - 1, 0)
    current_speeds = state.speed_data[idx] if state.speed_data is not None else np.full(num_v, 60.0)

    if len(state.prediction_history) > 0:
        _, pred_speeds = state.prediction_history[-1]
    else:
        # No predictions yet — fall back to current speed everywhere
        pred_speeds = np.tile(current_speeds[:, np.newaxis], (1, CONFIG['num_for_predict']))

    # Find K routes using predicted speeds
    routes = _find_k_routes(adj, source, target, current_speeds, pred_speeds, k)

    if not routes:
        return {
            'error': 'No path exists between these two nodes (they are in different network components)',
            'source': source,
            'target': target,
            'routes': [],
        }

    # Static baseline for comparison
    static_eta = _static_eta(adj, source, target, current_speeds)

    # Summary analytics
    best_eta = routes[0]['eta_minutes']
    step_min = CONFIG['time_interval_minutes']
    horizon_min = CONFIG['num_for_predict'] * step_min
    coverage = _prediction_coverage(best_eta)

    return {
        'source': source,
        'target': target,
        'routes': routes,
        'static_eta_minutes': static_eta,
        'prediction_horizon_min': horizon_min,
        'prediction_coverage': coverage,
        'timestamp': state.virtual_time.strftime("%Y-%m-%d %H:%M:%S"),
        'note': (
            '行程较短，全程在当前速度窗口内，预测未参与计算'
            if best_eta < step_min else
            f'预测覆盖 {int(coverage * 100)}% 行程'
            if coverage < 1.0 else
            '行程完全被预测覆盖'
        ),
    }
