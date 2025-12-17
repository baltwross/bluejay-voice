#!/bin/bash
# Restart script for Bluejay Terminator
# Kills all running services and restarts them

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BLUEJAY TERMINATOR - RESTART SERVICES                 ║"
echo "║               T-800 Voice Agent System                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Determine project root (script is in scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Verify we're in the right place
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Could not find project root${NC}"
    exit 1
fi

echo -e "${YELLOW}[PHASE 1] Stopping existing services...${NC}"

# Kill token server (port 8080)
TOKEN_PIDS=$(lsof -ti:8080 2>/dev/null || true)
if [ -n "$TOKEN_PIDS" ]; then
    echo -e "  ${RED}Killing token server (PIDs: $TOKEN_PIDS)${NC}"
    echo "$TOKEN_PIDS" | xargs kill -9 2>/dev/null || true
else
    echo -e "  ${YELLOW}Token server not running${NC}"
fi

# Kill frontend dev server (port 5173)
FRONTEND_PIDS=$(lsof -ti:5173 2>/dev/null || true)
if [ -n "$FRONTEND_PIDS" ]; then
    echo -e "  ${RED}Killing frontend server (PIDs: $FRONTEND_PIDS)${NC}"
    echo "$FRONTEND_PIDS" | xargs kill -9 2>/dev/null || true
else
    echo -e "  ${YELLOW}Frontend server not running${NC}"
fi

# Kill LiveKit agent processes
AGENT_PIDS=$(pgrep -f "python.*agent.py" 2>/dev/null || true)
if [ -n "$AGENT_PIDS" ]; then
    echo -e "  ${RED}Killing LiveKit agent (PIDs: $AGENT_PIDS)${NC}"
    echo "$AGENT_PIDS" | xargs kill -9 2>/dev/null || true
else
    echo -e "  ${YELLOW}LiveKit agent not running${NC}"
fi

# Brief pause to ensure ports are released
sleep 2

echo -e "\n${GREEN}[PHASE 2] Starting services...${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start token server
echo -e "  ${GREEN}[1/3] Starting token server...${NC}"
cd backend
source venv/bin/activate 2>/dev/null || {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
}
python token_server.py &
TOKEN_PID=$!
cd ..

# Wait for token server
sleep 2

# Start LiveKit agent
echo -e "  ${GREEN}[2/3] Starting LiveKit agent...${NC}"
cd backend
python agent.py dev &
AGENT_PID=$!
cd ..

sleep 2

# Start frontend
echo -e "  ${GREEN}[3/3] Starting frontend dev server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All services restarted!${NC}"
echo -e ""
echo -e "  ${CYAN}Frontend:${NC}     http://localhost:5173"
echo -e "  ${CYAN}Token API:${NC}    http://localhost:8080"
echo -e ""
echo -e "  PIDs: Token=$TOKEN_PID, Agent=$AGENT_PID, Frontend=$FRONTEND_PID"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# Wait for any process to exit
wait


