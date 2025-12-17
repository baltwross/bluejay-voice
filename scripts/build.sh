#!/bin/bash
# =============================================================================
# Bluejay Terminator - Docker Build Script
# =============================================================================
# Builds Docker images for backend and frontend
#
# Usage:
#   ./scripts/build.sh                    # Build all images
#   ./scripts/build.sh backend            # Build backend only
#   ./scripts/build.sh frontend           # Build frontend only
#   ./scripts/build.sh --push             # Build and push to ECR
#   ./scripts/build.sh --platform linux/amd64  # Build for specific platform
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
BACKEND_IMAGE="${PROJECT_NAME}-agent"
FRONTEND_IMAGE="${PROJECT_NAME}-frontend"
VERSION="${VERSION:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"  # Default for AWS

# Parse arguments
PUSH_TO_ECR=false
BUILD_TARGET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_ECR=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        backend|frontend)
            BUILD_TARGET="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BLUEJAY TERMINATOR - DOCKER BUILD                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the project root
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to build backend
build_backend() {
    echo -e "${GREEN}Building backend Docker image...${NC}"
    echo -e "  Image: ${BACKEND_IMAGE}:${VERSION}"
    echo -e "  Platform: ${PLATFORM}"
    
    docker buildx build \
        --platform "${PLATFORM}" \
        -t "${BACKEND_IMAGE}:${VERSION}" \
        -t "${BACKEND_IMAGE}:latest" \
        -f backend/Dockerfile \
        ./backend
    
    echo -e "${GREEN}✓ Backend image built successfully${NC}"
}

# Function to build frontend
build_frontend() {
    echo -e "${GREEN}Building frontend Docker image...${NC}"
    echo -e "  Image: ${FRONTEND_IMAGE}:${VERSION}"
    echo -e "  Platform: ${PLATFORM}"
    
    docker buildx build \
        --platform "${PLATFORM}" \
        --target production \
        -t "${FRONTEND_IMAGE}:${VERSION}" \
        -t "${FRONTEND_IMAGE}:latest" \
        -f frontend/Dockerfile \
        ./frontend
    
    echo -e "${GREEN}✓ Frontend image built successfully${NC}"
}

# Function to push to ECR
push_to_ecr() {
    local image_name=$1
    
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        echo -e "${YELLOW}AWS_ACCOUNT_ID not set. Attempting to get from AWS CLI...${NC}"
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
        if [ -z "$AWS_ACCOUNT_ID" ]; then
            echo -e "${RED}Error: Could not determine AWS Account ID${NC}"
            echo -e "${YELLOW}Please set AWS_ACCOUNT_ID environment variable${NC}"
            exit 1
        fi
    fi
    
    if [ -z "$AWS_REGION" ]; then
        AWS_REGION="us-east-1"
    fi
    
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    FULL_IMAGE="${ECR_URI}/${image_name}:${VERSION}"
    
    echo -e "${GREEN}Pushing ${image_name} to ECR...${NC}"
    echo -e "  ECR URI: ${FULL_IMAGE}"
    
    # Login to ECR
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_URI}"
    
    # Ensure repository exists
    aws ecr describe-repositories --repository-names "${image_name}" 2>/dev/null || \
        aws ecr create-repository --repository-name "${image_name}"
    
    # Tag and push
    docker tag "${image_name}:${VERSION}" "${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    
    # Also push latest tag
    docker tag "${image_name}:${VERSION}" "${ECR_URI}/${image_name}:latest"
    docker push "${ECR_URI}/${image_name}:latest"
    
    echo -e "${GREEN}✓ ${image_name} pushed to ECR${NC}"
}

# Build based on target
if [ -z "$BUILD_TARGET" ] || [ "$BUILD_TARGET" = "backend" ]; then
    build_backend
    if [ "$PUSH_TO_ECR" = true ]; then
        push_to_ecr "${BACKEND_IMAGE}"
    fi
fi

if [ -z "$BUILD_TARGET" ] || [ "$BUILD_TARGET" = "frontend" ]; then
    build_frontend
    if [ "$PUSH_TO_ECR" = true ]; then
        push_to_ecr "${FRONTEND_IMAGE}"
    fi
fi

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Build completed!${NC}"
echo -e ""
if [ "$PUSH_TO_ECR" = true ]; then
    echo -e "  ${CYAN}Images pushed to ECR${NC}"
else
    echo -e "  ${CYAN}Local images:${NC}"
    echo -e "    - ${BACKEND_IMAGE}:${VERSION}"
    echo -e "    - ${FRONTEND_IMAGE}:${VERSION}"
fi
echo -e ""
echo -e "${YELLOW}To test locally: docker-compose up${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

