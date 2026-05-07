# Real-time System Benchmarks

Performance evidence for Chapter 6 ("实时系统研究与实现"). Two benchmarks
back the thesis claims of a *millisecond-class streaming bus* (§6.2.2)
and a *non-blocking ASGI architecture under high concurrency* (§2.5.1):

| # | Benchmark | Script | Output |
|---|-----------|--------|--------|
| 1 | **End-to-end latency** (data slice → GPU inference → WS push → frontend decode) | `latency_benchmark.py` | per-frame CSV + stage breakdown |
| 2 | **Concurrent WebSocket throughput** (50 / 100 / 500 clients) | `concurrency_benchmark.py` | JSON summary + backend CPU/RSS |

A convenience wrapper `run_sweep.sh` chains both at concurrency levels 50/100/500.

---

## 1. Setup

The benchmark scripts only need two extra packages on top of the backend's runtime:

```bash
pip install websockets psutil
```

Start the backend (which now defaults to the checkpoint
`results/ASTGCN_lr0p001/1_1_20260503100704/checkpoints/best_model.pth`,
as requested):

```bash
bash realtime_server/start_backend.sh
# or:
cd realtime_server/backend && python main.py
```

The simulation loop has been instrumented: every WebSocket frame now
carries `server_send_ts_ms` and a `stage_timings_ms` block (slice /
inference / metrics). A REST endpoint `GET /api/benchmark/stage_timings`
also exposes the most recent tick's full server-side breakdown
(slice / inference / metrics / serialize / broadcast / tick_total).

---

## 2. End-to-end latency

```bash
python realtime_server/benchmark/latency_benchmark.py \
  --url ws://127.0.0.1:8000/ws \
  --frames 120 \
  --warmup 5 \
  --out realtime_server/benchmark/results/latency_frames.csv
```

The script connects as a single WebSocket client, captures `--frames`
broadcast frames after a `--warmup` burn-in, and reports mean / p50 /
p95 / p99 / max for:

| Metric | Meaning |
|--------|---------|
| `slice_ms` | virtual gear: pull frame, update sliding window |
| `inference_ms` | ASTGCN forward pass on GPU |
| `metrics_ms` | online MAE/RMSE update |
| `server_pipeline_ms` | sum of above (server-side compute) |
| `net_only_ms` | server `send_ts` → client `recv_ts` (kernel + LAN) |
| `decode_ms` | client `recv_ts` → JSON-decode complete |
| `net_plus_decode_ms` | net + decode, i.e. *time-to-render-ready* |
| `end_to_end_ms` | full pipeline: trigger → frontend has decoded data |

The final line prints an `Average End-to-End Latency` which is the
number to quote alongside §6.2.2 (e.g., *"Total Latency = 45 ms"*).

---

## 3. Concurrency / QPS

```bash
# Test 100 simultaneous WebSocket clients for 30s, sample backend CPU/MEM
python realtime_server/benchmark/concurrency_benchmark.py \
  --url ws://127.0.0.1:8000/ws \
  --clients 100 \
  --duration 30 \
  --ramp 3.0 \
  --backend-pid "$(pgrep -f 'uvicorn.*main:app' | head -1)" \
  --out realtime_server/benchmark/results/concurrency_100.json
```

Each client records every frame's `server_send_ts → client recv` latency
and frame count. The harness aggregates:

* **Connection success** — `ok_clients` / `dropped_clients`.
* **Aggregate throughput** — total frames / wall time = msgs/s pushed by
  the ASGI broadcaster across all sockets (this is the system's effective
  outbound QPS for the server-driven push channel).
* **Per-client fairness** — min / mean / max frames received per client.
  An ASGI broadcaster with no head-of-line blocking should produce
  near-identical frame counts across clients.
* **Latency distribution** — mean / p50 / p95 / p99 / max.
* **Backend resource usage** — process CPU % and RSS MB sampled at 1 Hz
  via `psutil` (only if `--backend-pid` is supplied).

### Recommended sweep

```bash
bash realtime_server/benchmark/run_sweep.sh ws://127.0.0.1:8000/ws 30
```

This runs the latency capture once, then concurrency tests at 50, 100,
and 500 clients sequentially, dropping all artifacts in
`realtime_server/benchmark/results/`.

---

## 4. Interpreting the numbers for the thesis

* **§6.2.2 *"毫秒级流式通信总线"*.** The `end_to_end_ms` mean from
  `latency_benchmark.py` is the headline figure. The component breakdown
  (`inference_ms` vs `net_plus_decode_ms`) lets the paper attribute the
  budget to GPU compute vs. transport.

* **§2.5.1 *"ASGI 解决同步框架并发阻塞"*.** Compare the three concurrency
  runs (50 / 100 / 500). On a non-blocking ASGI stack you should see:
  (a) `dropped_clients` stays at 0 across all levels, (b) `frames_per_client`
  remains tightly distributed (min ≈ max ≈ mean), (c) per-client latency
  p95 grows sub-linearly with N, (d) backend CPU rises gradually rather
  than saturating one core. These four observations are the empirical
  rebuttal to a synchronous WSGI baseline.

> JMeter / Locust were considered but use a thread-per-virtual-user
> model that is poorly suited for measuring sub-millisecond WebSocket
> push latency. The asyncio-native client here scales to 500+ sockets
> within a single process and gives microsecond-resolution timestamps,
> matching the granularity of the server instrumentation.
