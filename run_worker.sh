#!/bin/bash
# run_worker.sh - Run worker with correct virtual environment

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found!"
    echo "Please create the virtual environment first."
    exit 1
fi

echo "Starting worker with .venv..."
./.venv/bin/python -u worker.py "$@"
