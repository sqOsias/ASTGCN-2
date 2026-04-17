#!/bin/bash
# Start the Vue3 frontend development server
# Usage: ./start_frontend.sh [port]
# Default port: 3000

PORT=${1:-3000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

echo "=========================================="
echo "  Starting Frontend on port ${PORT}"
echo "=========================================="
echo "  Dashboard:  http://localhost:${PORT}"
echo "=========================================="
echo ""

npm run dev -- --host 0.0.0.0 --port ${PORT}
