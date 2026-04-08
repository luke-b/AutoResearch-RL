#!/bin/bash
set -e

# Change to the root of the project
cd "$(dirname "$0")/.."

echo "Running pytest suite..."
pytest -q tests/
