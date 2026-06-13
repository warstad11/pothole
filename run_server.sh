#!/bin/bash
# run_server.sh - Run server with correct virtual environment

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found!"
    echo "Please create the virtual environment first."
    exit 1
fi

echo "Starting server with .venv..."
# Using port 8000 as default for FastAPI/Uvicorn
./.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload > server.log 2>&1 &
PID=$!
echo $PID > server.pid
echo "Server started with PID $PID. Logs in server.log"
