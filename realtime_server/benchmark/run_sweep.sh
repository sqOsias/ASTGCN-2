#!/usr/bin/env bash
# Run the full benchmark sweep against an already-running backend.
#
#   1. End-to-end latency capture
#   2. Concurrency sweep at 50 / 100 / 500 simultaneous WebSocket clients
#
# Prerequisites:
#   * Backend already running on $URL (default ws://127.0.0.1:8000/ws)
#   * `pip install websockets psutil` in the active Python env
#
# Usage:
#   bash run_sweep.sh [URL] [DURATION_SECONDS]
set -euo pipefail

URL="${1:-ws://127.0.0.1:8000/ws}"
DURATION="${2:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results"
mkdir -p "$OUT_DIR"

# Try to discover backend PID for CPU/MEM sampling
BACKEND_PID="$(pgrep -f 'uvicorn.*main:app' | head -1 || true)"
if [[ -z "$BACKEND_PID" ]]; then
  BACKEND_PID="$(pgrep -f 'python.*main.py' | head -1 || true)"
fi
echo "[sweep] backend pid = ${BACKEND_PID:-<none>}"
echo "[sweep] target url  = $URL"
echo "[sweep] duration    = ${DURATION}s per concurrency level"

# ---- 1. End-to-end latency ----
echo
echo "==[ 1/2 ] end-to-end latency =="
python "$SCRIPT_DIR/latency_benchmark.py" \
  --url "$URL" \
  --frames 120 \
  --warmup 5 \
  --out "$OUT_DIR/latency_frames.csv"

# ---- 2. Concurrency sweep ----
for N in 50 100 500; do
  echo
  echo "==[ 2/2 ] concurrency = $N =="
  PID_ARG=()
  if [[ -n "$BACKEND_PID" ]]; then
    PID_ARG=(--backend-pid "$BACKEND_PID")
  fi
  python "$SCRIPT_DIR/concurrency_benchmark.py" \
    --url "$URL" \
    --clients "$N" \
    --duration "$DURATION" \
    --ramp 3.0 \
    "${PID_ARG[@]}" \
    --out "$OUT_DIR/concurrency_${N}.json"
done

echo
echo "[sweep] done. results in: $OUT_DIR"
