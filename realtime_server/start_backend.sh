#!/bin/bash
# Start the FastAPI backend server
# Usage: ./start_backend.sh [port]
# Default port: 8000

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/backend"

# Kill existing process on target port
existing=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$existing" ]; then
    echo "Killing existing process on port ${PORT} (PID: $existing)..."
    kill -9 "$existing" 2>/dev/null
    sleep 1
fi

# Install dependencies (skip if already satisfied)
echo "Checking dependencies..."
pip install -r requirements.txt networkx -q 2>/dev/null

echo "=========================================="
echo "  Starting Backend on port ${PORT}"
echo "=========================================="
echo "  REST API:   http://localhost:${PORT}/api"
echo "  WebSocket:  ws://localhost:${PORT}/ws"
echo "  API Docs:   http://localhost:${PORT}/docs"
echo "=========================================="
echo ""

python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT}
