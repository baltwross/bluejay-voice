#!/bin/bash
# =============================================================================
# Bluejay Terminator - AWS Deployment Script
# =============================================================================
# Deploys the application to AWS App Runner or ECS Fargate
#
# Usage:
#   ./scripts/deploy-aws.sh                    # Deploy to App Runner (default)
#   ./scripts/deploy-aws.sh --ecs              # Deploy to ECS Fargate
#   ./scripts/deploy-aws.sh --frontend-only    # Deploy frontend to S3/CloudFront
#   ./scripts/deploy-aws.sh --all              # Deploy everything
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
FRONTEND_BUCKET="${PROJECT_NAME}-frontend"
STACK_NAME="${PROJECT_NAME}-stack"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Parse arguments
DEPLOY_TARGET="apprunner"
DEPLOY_FRONTEND=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --ecs)
            DEPLOY_TARGET="ecs"
            shift
            ;;
        --apprunner)
            DEPLOY_TARGET="apprunner"
            shift
            ;;
        --frontend-only)
            DEPLOY_FRONTEND=true
            DEPLOY_TARGET="none"
            shift
            ;;
        --all)
            DEPLOY_FRONTEND=true
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

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BLUEJAY TERMINATOR - AWS DEPLOYMENT                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI is not installed${NC}"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}Error: AWS credentials not configured${NC}"
        exit 1
    fi
    
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
    echo -e "  AWS Account: ${AWS_ACCOUNT_ID}"
    echo -e "  Region: ${AWS_REGION}"
}

# Build and push Docker image
build_and_push() {
    echo -e "\n${GREEN}Building and pushing Docker image...${NC}"
    
    # Run build script with push flag
    ./scripts/build.sh backend --push --platform linux/amd64
}

# Deploy to App Runner
deploy_apprunner() {
    echo -e "\n${GREEN}Deploying to AWS App Runner...${NC}"
    
    IMAGE_URI="${ECR_URI}/${BACKEND_IMAGE}:latest"
    
    # Check if service exists
    SERVICE_ARN=$(aws apprunner list-services \
        --query "ServiceSummaryList[?ServiceName=='${PROJECT_NAME}'].ServiceArn" \
        --output text 2>/dev/null)
    
    if [ -n "$SERVICE_ARN" ] && [ "$SERVICE_ARN" != "None" ]; then
        echo -e "${YELLOW}Service exists, updating...${NC}"
        
        # Trigger new deployment with latest image
        aws apprunner start-deployment \
            --service-arn "${SERVICE_ARN}"
        
        echo -e "${GREEN}✓ Deployment triggered${NC}"
    else
        echo -e "${YELLOW}Creating new App Runner service...${NC}"
        
        # Check if access role exists
        ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole \
            --query 'Role.Arn' --output text 2>/dev/null) || true
        
        if [ -z "$ROLE_ARN" ]; then
            echo -e "${YELLOW}Creating ECR access role...${NC}"
            
            # Create trust policy
            cat > /tmp/trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "build.apprunner.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
            
            aws iam create-role \
                --role-name AppRunnerECRAccessRole \
                --assume-role-policy-document file:///tmp/trust-policy.json
            
            aws iam attach-role-policy \
                --role-name AppRunnerECRAccessRole \
                --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
            
            ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole \
                --query 'Role.Arn' --output text)
            
            # Wait for role to propagate
            sleep 10
        fi
        
        # Create service
        aws apprunner create-service \
            --service-name "${PROJECT_NAME}" \
            --source-configuration "{
                \"AuthenticationConfiguration\": {
                    \"AccessRoleArn\": \"${ROLE_ARN}\"
                },
                \"ImageRepository\": {
                    \"ImageIdentifier\": \"${IMAGE_URI}\",
                    \"ImageRepositoryType\": \"ECR\",
                    \"ImageConfiguration\": {
                        \"Port\": \"8080\",
                        \"RuntimeEnvironmentSecrets\": {
                            \"LIVEKIT_URL\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:LIVEKIT_URL::\",
                            \"LIVEKIT_API_KEY\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:LIVEKIT_API_KEY::\",
                            \"LIVEKIT_API_SECRET\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:LIVEKIT_API_SECRET::\",
                            \"OPENAI_API_KEY\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:OPENAI_API_KEY::\",
                            \"DEEPGRAM_API_KEY\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:DEEPGRAM_API_KEY::\",
                            \"ELEVEN_API_KEY\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:ELEVEN_API_KEY::\",
                            \"TAVILY_API_KEY\": \"arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${PROJECT_NAME}-secrets:TAVILY_API_KEY::\"
                        }
                    }
                },
                \"AutoDeploymentsEnabled\": true
            }" \
            --instance-configuration "{
                \"Cpu\": \"1024\",
                \"Memory\": \"2048\"
            }" \
            --health-check-configuration "{
                \"Protocol\": \"HTTP\",
                \"Path\": \"/health\",
                \"Interval\": 10,
                \"Timeout\": 5,
                \"HealthyThreshold\": 1,
                \"UnhealthyThreshold\": 5
            }"
        
        echo -e "${GREEN}✓ App Runner service created${NC}"
    fi
    
    # Get service URL
    echo -e "\n${YELLOW}Waiting for service to be ready...${NC}"
    sleep 30
    
    SERVICE_URL=$(aws apprunner describe-service \
        --service-arn "$(aws apprunner list-services \
            --query "ServiceSummaryList[?ServiceName=='${PROJECT_NAME}'].ServiceArn" \
            --output text)" \
        --query 'Service.ServiceUrl' \
        --output text 2>/dev/null)
    
    if [ -n "$SERVICE_URL" ] && [ "$SERVICE_URL" != "None" ]; then
        echo -e "${GREEN}Service URL: https://${SERVICE_URL}${NC}"
    fi
}

# Deploy frontend to S3 + CloudFront
deploy_frontend() {
    echo -e "\n${GREEN}Deploying frontend to S3 + CloudFront...${NC}"
    
    # Build frontend
    echo -e "${YELLOW}Building frontend...${NC}"
    cd frontend
    npm ci
    npm run build
    cd ..
    
    # Create S3 bucket if it doesn't exist
    if ! aws s3api head-bucket --bucket "${FRONTEND_BUCKET}" 2>/dev/null; then
        echo -e "${YELLOW}Creating S3 bucket...${NC}"
        aws s3 mb "s3://${FRONTEND_BUCKET}" --region "${AWS_REGION}"
        
        # Configure for static website hosting
        aws s3 website "s3://${FRONTEND_BUCKET}" \
            --index-document index.html \
            --error-document index.html
        
        # Set bucket policy for public read
        cat > /tmp/bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::${FRONTEND_BUCKET}/*"
        }
    ]
}
EOF
        
        # Disable block public access
        aws s3api put-public-access-block \
            --bucket "${FRONTEND_BUCKET}" \
            --public-access-block-configuration \
            "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
        
        # Apply policy
        aws s3api put-bucket-policy \
            --bucket "${FRONTEND_BUCKET}" \
            --policy file:///tmp/bucket-policy.json
    fi
    
    # Sync build to S3
    echo -e "${YELLOW}Uploading to S3...${NC}"
    aws s3 sync frontend/dist/ "s3://${FRONTEND_BUCKET}" --delete
    
    echo -e "${GREEN}✓ Frontend deployed to S3${NC}"
    echo -e "  S3 Website: http://${FRONTEND_BUCKET}.s3-website-${AWS_REGION}.amazonaws.com"
    
    # Check if CloudFront distribution exists
    DIST_ID=$(aws cloudfront list-distributions \
        --query "DistributionList.Items[?Origins.Items[?DomainName=='${FRONTEND_BUCKET}.s3.amazonaws.com']].Id" \
        --output text 2>/dev/null)
    
    if [ -z "$DIST_ID" ] || [ "$DIST_ID" = "None" ]; then
        echo -e "${YELLOW}Creating CloudFront distribution...${NC}"
        echo -e "${YELLOW}Note: CloudFront creation can take 10-15 minutes${NC}"
        
        # Create CloudFront distribution
        aws cloudfront create-distribution \
            --origin-domain-name "${FRONTEND_BUCKET}.s3.amazonaws.com" \
            --default-root-object index.html
    else
        echo -e "${YELLOW}Invalidating CloudFront cache...${NC}"
        aws cloudfront create-invalidation \
            --distribution-id "${DIST_ID}" \
            --paths "/*"
    fi
    
    echo -e "${GREEN}✓ Frontend deployment complete${NC}"
}

# Main execution
check_prerequisites

if [ "$DEPLOY_TARGET" != "none" ]; then
    build_and_push
    
    if [ "$DEPLOY_TARGET" = "apprunner" ]; then
        deploy_apprunner
    elif [ "$DEPLOY_TARGET" = "ecs" ]; then
        echo -e "${YELLOW}For ECS deployment, use the CloudFormation template:${NC}"
        echo -e "  aws cloudformation create-stack \\"
        echo -e "    --stack-name ${STACK_NAME} \\"
        echo -e "    --template-body file://infrastructure/cloudformation.yaml \\"
        echo -e "    --capabilities CAPABILITY_NAMED_IAM"
    fi
fi

if [ "$DEPLOY_FRONTEND" = true ]; then
    deploy_frontend
fi

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

