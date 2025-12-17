#!/bin/bash
# Test Runner Script for Bluejay Terminator

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$BACKEND_DIR/.." && pwd )"

# Activate venv if it exists
if [ -d "$BACKEND_DIR/venv" ]; then
    source "$BACKEND_DIR/venv/bin/activate"
elif [ -d "$BACKEND_DIR/.venv" ]; then
    source "$BACKEND_DIR/.venv/bin/activate"
fi

echo "🧪 Running Bluejay Terminator Tests"
echo "===================================="

# Check if backend server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "⚠️  Warning: Backend server not running on port 8080"
    echo "   Start it with: python backend/token_server.py"
    echo ""
fi

# Run API endpoint tests
echo ""
echo "📡 Running API Endpoint Tests..."
echo "---------------------------------"
cd "$PROJECT_ROOT"
python -m pytest backend/tests/test_api_endpoints.py -v || echo "⚠️  Some API tests failed"

# Run LiveKit WebRTC tests (if LiveKit SDK available)
echo ""
echo "🎤 Running LiveKit WebRTC Tests..."
echo "-----------------------------------"
python -m pytest backend/tests/test_livekit_voice.py -v -m "not skip" || echo "⚠️  Some LiveKit tests failed"

echo ""
echo "✅ Tests completed!"

