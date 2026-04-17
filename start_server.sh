#!/bin/bash
# ============================================================
#  实时交通态势感知系统 - 一键启动
#  启动后端 (FastAPI, port 8000) + 前端 (Vite, port 3000)
#  Usage: ./start_server.sh
#  Stop:  Ctrl+C (will stop both services)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=3000

BACKEND_PID=""
FRONTEND_PID=""

# Cleanup function: kill both services on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null
    wait 2>/dev/null
    echo "All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Kill existing processes on target ports
for port in $BACKEND_PORT $FRONTEND_PORT; do
    existing=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -n "$existing" ]; then
        echo "Killing existing process on port ${port} (PID: $existing)..."
        kill -9 "$existing" 2>/dev/null
        sleep 1
    fi
done

echo "=========================================="
echo "  实时交通态势感知系统"
echo "=========================================="

# Start backend
echo "[1/2] Starting backend (port ${BACKEND_PORT})..."
cd "$SCRIPT_DIR/realtime_server/backend"
python3 -m uvicorn main:app --host 0.0.0.0 --port ${BACKEND_PORT} &
BACKEND_PID=$!

# Wait for backend to be ready
echo "  Waiting for backend to initialize..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:${BACKEND_PORT}/" > /dev/null 2>&1; then
        echo "  Backend ready!"
        break
    fi
    sleep 1
done

# Start frontend
echo "[2/2] Starting frontend (port ${FRONTEND_PORT})..."
cd "$SCRIPT_DIR/realtime_server/frontend"
if [ ! -d "node_modules" ]; then
    echo "  Installing npm dependencies..."
    npm install
fi
npm run dev -- --host 0.0.0.0 --port ${FRONTEND_PORT} &
FRONTEND_PID=$!

sleep 3

echo ""
echo "=========================================="
echo "  System Ready!"
echo "=========================================="
echo "  Dashboard:  http://localhost:${FRONTEND_PORT}"
echo "  Backend:    http://localhost:${BACKEND_PORT}"
echo "  API Docs:   http://localhost:${BACKEND_PORT}/docs"
echo "  WebSocket:  ws://localhost:${BACKEND_PORT}/ws"
echo "=========================================="
echo "  Press Ctrl+C to stop all services"
echo "=========================================="

# Wait for either process to exit
wait