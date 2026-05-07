# -*- coding:utf-8 -*-
"""
High-Concurrency WebSocket Load Benchmark.

Spawns N concurrent WebSocket clients against the FastAPI ASGI backend,
measures throughput, end-to-end latency, dropped/disconnected clients,
and (optionally) samples backend CPU/RSS via psutil.

Why this exists
---------------
Section 6 of the thesis claims that an ASGI-based backend handles
concurrent streaming pushes without the head-of-line blocking that
WSGI/synchronous frameworks suffer from. This benchmark provides
empirical evidence for that claim by:

  1. Opening N concurrent WebSocket clients (default 50, also test 100, 500).
  2. Each client records every received frame's latency
     (``server_send_ts_ms`` -> client recv).
  3. The harness aggregates: total frames received, mean/p50/p95/p99 latency,
     drop count (clients that disconnected before duration expired),
     average frames-per-client, and (if --backend-pid given) backend
     CPU% and resident memory.

Usage
-----
    # Pick concurrency level (typical: 50 / 100 / 500)
    python concurrency_benchmark.py \
        --url ws://127.0.0.1:8000/ws \
        --clients 100 \
        --duration 30 \
        --backend-pid $(pgrep -f 'uvicorn.*main:app' | head -1) \
        --out concurrency_100.json

Outputs a JSON summary to stdout (and to --out if specified).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import List, Dict, Any, Optional

import websockets

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None


# ----------------------------- helpers ----------------------------- #
def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float('nan')
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# ----------------------------- worker ------------------------------ #
async def client_worker(client_id: int,
                        url: str,
                        stop_at: float,
                        results: Dict[str, Any]) -> None:
    """Single WebSocket client worker. Records latencies, frame count, status."""
    latencies: List[float] = []
    frame_count = 0
    status = 'ok'
    err: Optional[str] = None
    try:
        async with websockets.connect(url, max_size=2 ** 24, open_timeout=10,
                                      ping_interval=20, ping_timeout=20) as ws:
            while True:
                remaining = stop_at - time.time()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining + 1.0)
                except asyncio.TimeoutError:
                    break
                recv_ts = time.time() * 1000.0
                try:
                    decoded = json.loads(raw)
                except Exception:
                    continue
                send_ts = decoded.get('server_send_ts_ms')
                if send_ts is not None:
                    latencies.append(recv_ts - send_ts)
                frame_count += 1
    except Exception as e:
        status = 'dropped'
        err = f"{type(e).__name__}: {e}"

    results['clients'][client_id] = {
        'frames': frame_count,
        'status': status,
        'error': err,
        'mean_latency_ms': round(statistics.fmean(latencies), 3) if latencies else None,
    }
    # Aggregate raw latencies
    results['_all_latencies'].extend(latencies)
    results['_total_frames'] += frame_count


# ------------------------- backend monitor ------------------------- #
async def backend_monitor(pid: int,
                          stop_at: float,
                          samples: List[Dict[str, float]],
                          interval: float = 1.0) -> None:
    if psutil is None:
        print("[concurrency] psutil unavailable -- skipping backend resource sampling")
        return
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"[concurrency] backend PID {pid} not found")
        return
    proc.cpu_percent(interval=None)  # prime
    while time.time() < stop_at:
        await asyncio.sleep(interval)
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / (1024 ** 2)
            samples.append({
                'ts': time.time(),
                'cpu_percent': cpu,
                'rss_mb': round(mem, 2),
                'num_threads': proc.num_threads(),
            })
        except psutil.NoSuchProcess:
            break


# --------------------------- main entry ---------------------------- #
async def run_benchmark(url: str, n_clients: int, duration: float,
                        backend_pid: Optional[int],
                        ramp_seconds: float) -> Dict[str, Any]:
    print(f"[concurrency] target = {n_clients} clients, duration = {duration:.1f}s, "
          f"ramp = {ramp_seconds:.1f}s, url = {url}")

    results: Dict[str, Any] = {
        'clients': {},
        '_all_latencies': [],
        '_total_frames': 0,
    }

    stop_at = time.time() + ramp_seconds + duration
    backend_samples: List[Dict[str, float]] = []
    monitor_task: Optional[asyncio.Task] = None
    if backend_pid:
        monitor_task = asyncio.create_task(
            backend_monitor(backend_pid, stop_at, backend_samples)
        )

    # Stagger client connects across ramp_seconds to avoid SYN flood spike
    tasks: List[asyncio.Task] = []
    per_delay = (ramp_seconds / max(n_clients, 1)) if ramp_seconds > 0 else 0.0
    t_start = time.time()
    for i in range(n_clients):
        if per_delay > 0:
            await asyncio.sleep(per_delay)
        tasks.append(asyncio.create_task(client_worker(i, url, stop_at, results)))

    connect_done_ts = time.time()
    print(f"[concurrency] all {n_clients} client tasks launched in "
          f"{connect_done_ts - t_start:.2f}s; collecting data ...")

    await asyncio.gather(*tasks, return_exceptions=True)
    if monitor_task is not None:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    t_end = time.time()

    # ----- aggregate -----
    latencies = results.pop('_all_latencies')
    total_frames = results.pop('_total_frames')
    actual_duration = max(t_end - t_start, 1e-9)

    dropped = sum(1 for c in results['clients'].values() if c['status'] == 'dropped')
    ok_clients = n_clients - dropped
    frames_per_client = [c['frames'] for c in results['clients'].values()]

    summary = {
        'url': url,
        'target_clients': n_clients,
        'ok_clients': ok_clients,
        'dropped_clients': dropped,
        'duration_s': round(actual_duration, 3),
        'ramp_s': ramp_seconds,
        'total_frames': total_frames,
        'throughput_msgs_per_s': round(total_frames / actual_duration, 2),
        'frames_per_client_mean': round(statistics.fmean(frames_per_client), 2)
            if frames_per_client else 0.0,
        'frames_per_client_min': min(frames_per_client) if frames_per_client else 0,
        'frames_per_client_max': max(frames_per_client) if frames_per_client else 0,
        'latency_ms': {
            'samples': len(latencies),
            'mean': round(statistics.fmean(latencies), 3) if latencies else None,
            'p50': round(_percentile(latencies, 50), 3) if latencies else None,
            'p95': round(_percentile(latencies, 95), 3) if latencies else None,
            'p99': round(_percentile(latencies, 99), 3) if latencies else None,
            'max': round(max(latencies), 3) if latencies else None,
        },
        'backend_resource_samples': backend_samples,
    }

    if backend_samples:
        cpu_vals = [s['cpu_percent'] for s in backend_samples]
        rss_vals = [s['rss_mb'] for s in backend_samples]
        summary['backend_cpu_percent_mean'] = round(statistics.fmean(cpu_vals), 2)
        summary['backend_cpu_percent_max'] = round(max(cpu_vals), 2)
        summary['backend_rss_mb_mean'] = round(statistics.fmean(rss_vals), 2)
        summary['backend_rss_mb_max'] = round(max(rss_vals), 2)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n========= Concurrency Benchmark Summary =========")
    print(f"clients (target/ok/dropped) : "
          f"{summary['target_clients']}/{summary['ok_clients']}/{summary['dropped_clients']}")
    print(f"duration (s)                : {summary['duration_s']}")
    print(f"total frames received       : {summary['total_frames']}")
    print(f"aggregate throughput (msg/s): {summary['throughput_msgs_per_s']}")
    print(f"frames/client (min/mean/max): "
          f"{summary['frames_per_client_min']}/"
          f"{summary['frames_per_client_mean']}/"
          f"{summary['frames_per_client_max']}")
    lat = summary['latency_ms']
    print(f"latency mean/p50/p95/p99/max: "
          f"{lat['mean']}/{lat['p50']}/{lat['p95']}/{lat['p99']}/{lat['max']} ms")
    if 'backend_cpu_percent_mean' in summary:
        print(f"backend cpu  mean/max       : "
              f"{summary['backend_cpu_percent_mean']}% / {summary['backend_cpu_percent_max']}%")
        print(f"backend rss  mean/max       : "
              f"{summary['backend_rss_mb_mean']} MB / {summary['backend_rss_mb_max']} MB")
    print("=================================================\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='ws://127.0.0.1:8000/ws')
    p.add_argument('--clients', type=int, default=50,
                   help='Number of concurrent WebSocket clients')
    p.add_argument('--duration', type=float, default=30.0,
                   help='Sustained measurement duration in seconds')
    p.add_argument('--ramp', type=float, default=2.0,
                   help='Ramp-up time in seconds (stagger client connects)')
    p.add_argument('--backend-pid', type=int, default=0,
                   help='If set, sample CPU/RSS of this PID via psutil')
    p.add_argument('--out', default='',
                   help='Optional JSON output path')
    args = p.parse_args()

    summary = asyncio.run(run_benchmark(
        url=args.url,
        n_clients=args.clients,
        duration=args.duration,
        backend_pid=args.backend_pid or None,
        ramp_seconds=args.ramp,
    ))
    print_summary(summary)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"[concurrency] wrote summary JSON: {args.out}")


if __name__ == '__main__':
    main()
