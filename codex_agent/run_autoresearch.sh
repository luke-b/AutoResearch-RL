#!/bin/bash
set -e

# Change to the root of the project
cd "$(dirname "$0")/.."

echo "Installing dependencies..."
pip install -r codex_agent/requirements.txt

echo "Running tests..."
./codex_agent/ci.sh

echo "Loading environment variables..."
if [ -f codex_agent/.env.example ]; then
  source codex_agent/.env.example
fi

# Fallback values if not provided in .env
MAX_ITERATIONS=${MAX_ITERATIONS:-1}

echo "Running AutoResearch-RL for $MAX_ITERATIONS iteration(s)..."
python3 main.py --max_iterations=$MAX_ITERATIONS
