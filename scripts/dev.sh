#!/bin/bash
# Development startup script for Bluejay Terminator
# Starts both the backend token server and frontend dev server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BLUEJAY TERMINATOR - DEVELOPMENT MODE                ║"
echo "║               T-800 Voice Agent System                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the project root
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend services
echo -e "${GREEN}[1/3] Starting token server...${NC}"
cd backend
source venv/bin/activate 2>/dev/null || {
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
}

# Start token server in background
python token_server.py &
TOKEN_PID=$!
cd ..

# Wait for token server to be ready
echo -e "${YELLOW}Waiting for token server to start...${NC}"
sleep 2

# Start LiveKit agent in background
echo -e "${GREEN}[2/3] Starting LiveKit agent...${NC}"
cd backend
python agent.py dev &
AGENT_PID=$!
cd ..

sleep 2

# Start frontend
echo -e "${GREEN}[3/3] Starting frontend dev server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All services started!${NC}"
echo -e ""
echo -e "  ${CYAN}Frontend:${NC}     http://localhost:5173"
echo -e "  ${CYAN}Token API:${NC}    http://localhost:8080"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# Wait for any process to exit
wait




