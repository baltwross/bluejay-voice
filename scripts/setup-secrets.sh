#!/bin/bash
# =============================================================================
# Bluejay Terminator - AWS Secrets Setup Script
# =============================================================================
# Creates or updates secrets in AWS Secrets Manager for deployment
#
# Usage:
#   ./scripts/setup-secrets.sh              # Interactive mode
#   ./scripts/setup-secrets.sh --from-env   # Read from backend/.env file
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="bluejay-terminator"
SECRET_NAME="${PROJECT_NAME}-secrets"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BLUEJAY TERMINATOR - AWS SECRETS SETUP                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the project root
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    echo -e "${YELLOW}Run 'aws configure' to set up credentials${NC}"
    exit 1
fi

# Parse arguments
FROM_ENV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --from-env)
            FROM_ENV=true
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to read value from .env file
read_env_value() {
    local key=$1
    local env_file="backend/.env"
    
    if [ -f "$env_file" ]; then
        grep "^${key}=" "$env_file" | cut -d'=' -f2- | tr -d '"' | tr -d "'"
    fi
}

# Function to prompt for value with default
prompt_value() {
    local prompt=$1
    local default=$2
    local var_name=$3
    
    if [ -n "$default" ]; then
        read -p "${prompt} [${default}]: " value
        value="${value:-$default}"
    else
        read -p "${prompt}: " value
    fi
    
    eval "$var_name='$value'"
}

# Collect secrets
echo -e "${GREEN}Collecting API keys...${NC}"
echo ""

if [ "$FROM_ENV" = true ]; then
    echo -e "${YELLOW}Reading from backend/.env file...${NC}"
    
    if [ ! -f "backend/.env" ]; then
        echo -e "${RED}Error: backend/.env file not found${NC}"
        echo -e "${YELLOW}Create it by copying backend/env.template${NC}"
        exit 1
    fi
    
    LIVEKIT_URL=$(read_env_value "LIVEKIT_URL")
    LIVEKIT_API_KEY=$(read_env_value "LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET=$(read_env_value "LIVEKIT_API_SECRET")
    OPENAI_API_KEY=$(read_env_value "OPENAI_API_KEY")
    ELEVEN_API_KEY=$(read_env_value "ELEVEN_API_KEY")
    DEEPGRAM_API_KEY=$(read_env_value "DEEPGRAM_API_KEY")
    TAVILY_API_KEY=$(read_env_value "TAVILY_API_KEY")
else
    echo -e "${CYAN}Enter your API keys (press Enter to skip optional ones):${NC}"
    echo ""
    
    prompt_value "LiveKit WebSocket URL (wss://...)" "" "LIVEKIT_URL"
    prompt_value "LiveKit API Key" "" "LIVEKIT_API_KEY"
    prompt_value "LiveKit API Secret" "" "LIVEKIT_API_SECRET"
    prompt_value "OpenAI API Key" "" "OPENAI_API_KEY"
    prompt_value "ElevenLabs API Key" "" "ELEVEN_API_KEY"
    prompt_value "Deepgram API Key" "" "DEEPGRAM_API_KEY"
    prompt_value "Tavily API Key (optional)" "" "TAVILY_API_KEY"
fi

# Validate required keys
if [ -z "$LIVEKIT_URL" ] || [ -z "$LIVEKIT_API_KEY" ] || [ -z "$LIVEKIT_API_SECRET" ]; then
    echo -e "${RED}Error: LiveKit credentials are required${NC}"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OpenAI API key is required${NC}"
    exit 1
fi

if [ -z "$ELEVEN_API_KEY" ]; then
    echo -e "${RED}Error: ElevenLabs API key is required${NC}"
    exit 1
fi

if [ -z "$DEEPGRAM_API_KEY" ]; then
    echo -e "${RED}Error: Deepgram API key is required${NC}"
    exit 1
fi

# Create secret JSON
SECRET_JSON=$(cat <<EOF
{
    "LIVEKIT_URL": "${LIVEKIT_URL}",
    "LIVEKIT_API_KEY": "${LIVEKIT_API_KEY}",
    "LIVEKIT_API_SECRET": "${LIVEKIT_API_SECRET}",
    "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    "ELEVEN_API_KEY": "${ELEVEN_API_KEY}",
    "DEEPGRAM_API_KEY": "${DEEPGRAM_API_KEY}",
    "TAVILY_API_KEY": "${TAVILY_API_KEY:-}"
}
EOF
)

# Check if secret exists
echo ""
echo -e "${GREEN}Checking AWS Secrets Manager...${NC}"

if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" &> /dev/null; then
    echo -e "${YELLOW}Secret already exists. Updating...${NC}"
    
    aws secretsmanager update-secret \
        --secret-id "$SECRET_NAME" \
        --secret-string "$SECRET_JSON" \
        --region "$AWS_REGION"
    
    echo -e "${GREEN}✓ Secret updated successfully${NC}"
else
    echo -e "${YELLOW}Creating new secret...${NC}"
    
    aws secretsmanager create-secret \
        --name "$SECRET_NAME" \
        --description "API keys for Bluejay Terminator voice agent" \
        --secret-string "$SECRET_JSON" \
        --region "$AWS_REGION"
    
    echo -e "${GREEN}✓ Secret created successfully${NC}"
fi

# Get secret ARN
SECRET_ARN=$(aws secretsmanager describe-secret \
    --secret-id "$SECRET_NAME" \
    --region "$AWS_REGION" \
    --query 'ARN' \
    --output text)

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Secrets setup complete!${NC}"
echo ""
echo -e "  ${CYAN}Secret Name:${NC} ${SECRET_NAME}"
echo -e "  ${CYAN}Secret ARN:${NC}  ${SECRET_ARN}"
echo -e "  ${CYAN}Region:${NC}      ${AWS_REGION}"
echo ""
echo -e "${YELLOW}Use this ARN when deploying to ECS:${NC}"
echo -e "  --parameters ParameterKey=SecretsArn,ParameterValue=${SECRET_ARN}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

