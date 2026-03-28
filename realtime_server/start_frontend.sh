#!/bin/bash
# Start the Vue3 frontend development server

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Start the dev server
echo "Starting Vue3 frontend on http://localhost:3000"
echo ""
npm run dev -- --host 0.0.0.0
