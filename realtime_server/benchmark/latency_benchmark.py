# -*- coding:utf-8 -*-
"""
End-to-End Latency Benchmark for the Real-time Traffic Prediction System.

Measures the full streaming pipeline latency:

    Virtual gear trigger (data slice)
        -> GPU forward inference
        -> Payload serialize
        -> WebSocket broadcast
        -> Frontend (this client) receives & decodes

The server simulation loop is already instrumented (see
``realtime_server/backend/simulation.py``) to embed:

  * ``server_send_ts_ms`` -- wall-clock just before ``json.dumps``
  * ``stage_timings_ms``  -- per-stage server-side timings (slice / inference / metrics)

This client-side script connects via WebSocket, decodes each frame,
and computes:

  * **server stage timings**: slice / inference / serialize-ish (= recv - send - net)
  * **network + decode latency**: client_recv_ts - server_send_ts
  * **end-to-end latency**: server slice -> client decode-complete

It then prints aggregate statistics (mean, p50, p95, p99, max) and
optionally writes a per-frame CSV to ``--out``.

Usage
-----
    python latency_benchmark.py \
        --url ws://127.0.0.1:8000/ws \
        --frames 200 \
        --out latency_frames.csv

Note: this script measures up to *frontend decode complete*. Browser-side
DOM/canvas paint is environment-dependent; we treat JSON-decode completion
as a faithful proxy for "data available to render", which is the actionable
moment for the frontend application code.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from typing import List, Dict, Any

import websockets


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


def _summarize(name: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {'metric': name, 'count': 0}
    return {
        'metric': name,
        'count': len(values),
        'mean': round(statistics.fmean(values), 3),
        'p50': round(_percentile(values, 50), 3),
        'p95': round(_percentile(values, 95), 3),
        'p99': round(_percentile(values, 99), 3),
        'max': round(max(values), 3),
        'min': round(min(values), 3),
    }


async def collect_frames(url: str, n_frames: int, warmup: int) -> List[Dict[str, Any]]:
    print(f"[latency] connecting to {url} ...")
    async with websockets.connect(url, max_size=2 ** 24) as ws:
        print(f"[latency] connected. capturing {n_frames} frames (warmup={warmup}) ...")
        frames: List[Dict[str, Any]] = []
        captured = 0
        skipped = 0
        while captured < n_frames:
            recv_ts_pre = time.time() * 1000.0
            raw = await ws.recv()
            decoded = json.loads(raw)
            decode_done_ts = time.time() * 1000.0

            server_send_ts = decoded.get('server_send_ts_ms')
            stages = decoded.get('stage_timings_ms') or {}
            if server_send_ts is None:
                # Old payload format -- skip
                continue

            if skipped < warmup:
                skipped += 1
                continue

            net_plus_decode_ms = decode_done_ts - server_send_ts
            net_only_ms = recv_ts_pre - server_send_ts
            decode_ms = decode_done_ts - recv_ts_pre

            slice_ms = float(stages.get('slice', 0.0))
            inference_ms = float(stages.get('inference', 0.0))
            metrics_ms = float(stages.get('metrics', 0.0))
            server_pipeline_ms = slice_ms + inference_ms + metrics_ms
            end_to_end_ms = server_pipeline_ms + net_plus_decode_ms

            frames.append({
                'frame': captured,
                'server_send_ts_ms': server_send_ts,
                'client_recv_ts_ms': recv_ts_pre,
                'client_decode_done_ts_ms': decode_done_ts,
                'slice_ms': slice_ms,
                'inference_ms': inference_ms,
                'metrics_ms': metrics_ms,
                'server_pipeline_ms': server_pipeline_ms,
                'net_only_ms': net_only_ms,
                'decode_ms': decode_ms,
                'net_plus_decode_ms': net_plus_decode_ms,
                'end_to_end_ms': end_to_end_ms,
                'payload_bytes': len(raw) if isinstance(raw, (bytes, bytearray)) else len(raw.encode('utf-8')),
            })
            captured += 1
            if captured % 25 == 0:
                print(f"  captured {captured}/{n_frames}  "
                      f"e2e={end_to_end_ms:.1f}ms  infer={inference_ms:.1f}ms")
        return frames


def report(frames: List[Dict[str, Any]]) -> None:
    if not frames:
        print("[latency] no frames captured.")
        return

    cols = ['slice_ms', 'inference_ms', 'metrics_ms', 'server_pipeline_ms',
            'net_only_ms', 'decode_ms', 'net_plus_decode_ms', 'end_to_end_ms',
            'payload_bytes']
    summaries = [_summarize(c, [f[c] for f in frames]) for c in cols]

    print("\n=========== End-to-End Latency Report ===========")
    print(f"frames captured : {len(frames)}")
    header = f"{'metric':<22}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'max':>10}{'min':>10}"
    print(header)
    print('-' * len(header))
    for s in summaries:
        if s['count'] == 0:
            continue
        print(f"{s['metric']:<22}{s['mean']:>10.3f}{s['p50']:>10.3f}"
              f"{s['p95']:>10.3f}{s['p99']:>10.3f}{s['max']:>10.3f}{s['min']:>10.3f}")
    print('=================================================\n')

    e2e_mean = statistics.fmean([f['end_to_end_ms'] for f in frames])
    print(f"==> Average End-to-End Latency = {e2e_mean:.2f} ms")
    print(f"    (slice + inference + metrics + network + decode)")


def write_csv(frames: List[Dict[str, Any]], path: str) -> None:
    if not frames:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(frames[0].keys()))
        writer.writeheader()
        writer.writerows(frames)
    print(f"[latency] wrote per-frame CSV: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='ws://127.0.0.1:8000/ws',
                        help='WebSocket URL of the backend')
    parser.add_argument('--frames', type=int, default=120,
                        help='Number of frames to capture (after warmup)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Frames to discard before starting measurement')
    parser.add_argument('--out', default='',
                        help='Optional CSV output path for per-frame rows')
    args = parser.parse_args()

    frames = asyncio.run(collect_frames(args.url, args.frames, args.warmup))
    report(frames)
    if args.out:
        write_csv(frames, args.out)


if __name__ == '__main__':
    main()
