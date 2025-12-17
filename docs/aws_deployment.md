# AWS Deployment Guide

## Overview

This project is designed for **AWS deployment** to meet the Bluejay take-home interview bonus requirements. The backend agent runs in a Docker container on AWS infrastructure, with the frontend served via S3 and CloudFront.

## Quick Start

The fastest way to deploy is using the provided scripts:

```bash
# 1. Setup secrets in AWS Secrets Manager (one-time)
./scripts/setup-secrets.sh

# 2. Build and deploy to App Runner
./scripts/deploy-aws.sh

# 3. Deploy frontend to S3/CloudFront
./scripts/deploy-aws.sh --frontend-only
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                            AWS Cloud                                 │
│                                                                       │
│  ┌─────────────────────┐          ┌────────────────────────┐        │
│  │   CloudFront CDN    │          │  App Runner / ECS      │        │
│  │    (Frontend)       │          │   (Backend Agent)      │        │
│  └──────────┬──────────┘          └───────────┬────────────┘        │
│             │                                  │                      │
│  ┌──────────▼──────────┐          ┌───────────▼────────────┐        │
│  │    S3 Bucket        │          │   EFS Volume           │        │
│  │   (React Build)     │          │  (ChromaDB Data)       │        │
│  └─────────────────────┘          └────────────────────────┘        │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │              AWS Secrets Manager                            │      │
│  │  - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET       │      │
│  │  - OPENAI_API_KEY                                          │      │
│  │  - ELEVEN_API_KEY (ElevenLabs)                            │      │
│  │  - DEEPGRAM_API_KEY                                        │      │
│  │  - TAVILY_API_KEY                                          │      │
│  └───────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  LiveKit Cloud  │
                    │   (External)    │
                    └─────────────────┘
```

## Deployment Options

### Option 1: AWS App Runner (Recommended - Easiest)

**Best for:** Quick deployment, automatic scaling, minimal configuration

**Pros:**
- Automatic scaling and load balancing
- Simple configuration (no VPC/ALB required)
- Pay-per-use pricing
- Built-in health checks
- Easy rollback

**Cons:**
- Less control over infrastructure
- Limited networking options
- No EFS support (must use S3 for persistence)

### Option 2: ECS Fargate (More Control)

**Best for:** Production workloads, full VPC control, persistent storage

**Pros:**
- Full VPC and networking control
- EFS support for ChromaDB persistence
- More monitoring and logging options
- Better for production scale
- Auto-scaling based on metrics

**Cons:**
- More complex setup (VPC, ALB, target groups)
- Requires CloudFormation or Terraform

## Prerequisites

Before deploying, ensure you have:

1. **AWS CLI** installed and configured
   ```bash
   aws --version
   aws configure  # Set up credentials
   ```

2. **Docker** installed locally
   ```bash
   docker --version
   docker buildx version  # For multi-platform builds
   ```

3. **API Keys** ready for:
   - LiveKit Cloud (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
   - OpenAI (OPENAI_API_KEY)
   - ElevenLabs (ELEVEN_API_KEY)
   - Deepgram (DEEPGRAM_API_KEY)
   - Tavily (TAVILY_API_KEY) - optional

4. **AWS Permissions** for:
   - ECR (Elastic Container Registry)
   - App Runner or ECS
   - Secrets Manager
   - S3 and CloudFront (for frontend)
   - IAM (for creating roles)

## Step-by-Step Deployment

### Step 1: Store Secrets in AWS Secrets Manager

Create a secret containing all API keys:

```bash
aws secretsmanager create-secret \
  --name bluejay-terminator-secrets \
  --description "API keys for Bluejay Terminator voice agent" \
  --secret-string '{
    "LIVEKIT_URL": "wss://your-project.livekit.cloud",
    "LIVEKIT_API_KEY": "your_livekit_api_key",
    "LIVEKIT_API_SECRET": "your_livekit_api_secret",
    "OPENAI_API_KEY": "your_openai_api_key",
    "ELEVEN_API_KEY": "your_elevenlabs_api_key",
    "DEEPGRAM_API_KEY": "your_deepgram_api_key",
    "TAVILY_API_KEY": "your_tavily_api_key"
  }'
```

To update an existing secret:
```bash
aws secretsmanager update-secret \
  --secret-id bluejay-terminator-secrets \
  --secret-string '{...}'
```

### Step 2: Build and Push Docker Image

```bash
# Navigate to project root
cd bluejay-voice

# Make scripts executable
chmod +x scripts/*.sh

# Build and push to ECR
./scripts/build.sh --push

# Or manually:
cd backend

# Build image
docker build -t bluejay-terminator-agent .

# Create ECR repository (if not exists)
aws ecr create-repository --repository-name bluejay-terminator-agent

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag bluejay-terminator-agent:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest

docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest
```

### Step 3A: Deploy with App Runner (Recommended)

Using the deployment script:
```bash
./scripts/deploy-aws.sh --apprunner
```

Or manually via AWS Console:
1. Go to **AWS App Runner** in the AWS Console
2. Click **Create service**
3. Select **Container registry** > **Amazon ECR**
4. Choose your image: `bluejay-terminator-agent:latest`
5. Configure:
   - **Port**: 8080
   - **CPU**: 1 vCPU
   - **Memory**: 2 GB
6. Add environment variables from Secrets Manager
7. Enable health check on `/health`
8. Deploy!

Or via CLI:
```bash
aws apprunner create-service \
  --service-name bluejay-terminator \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8080"
      }
    },
    "AutoDeploymentsEnabled": true
  }' \
  --instance-configuration '{
    "Cpu": "1024",
    "Memory": "2048"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 10,
    "Timeout": 5
  }'
```

### Step 3B: Deploy with ECS Fargate (Alternative)

Using CloudFormation:
```bash
# Deploy the complete stack
aws cloudformation create-stack \
  --stack-name bluejay-terminator-stack \
  --template-body file://infrastructure/cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=SecretsArn,ParameterValue=arn:aws:secretsmanager:us-east-1:<account-id>:secret:bluejay-terminator-secrets

# Wait for stack creation
aws cloudformation wait stack-create-complete \
  --stack-name bluejay-terminator-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name bluejay-terminator-stack \
  --query 'Stacks[0].Outputs'
```

The CloudFormation template creates:
- VPC with public subnets
- ECS Cluster (Fargate)
- ECR Repository
- Application Load Balancer
- EFS file system for ChromaDB persistence
- Auto-scaling based on CPU utilization
- CloudWatch log groups

### Step 4: Deploy Frontend to S3 + CloudFront

```bash
# Build frontend
cd frontend
npm ci
npm run build

# Create S3 bucket
aws s3 mb s3://bluejay-terminator-frontend --region us-east-1

# Configure for static website hosting
aws s3 website s3://bluejay-terminator-frontend \
  --index-document index.html \
  --error-document index.html

# Upload build
aws s3 sync dist/ s3://bluejay-terminator-frontend --delete

# Create CloudFront distribution (for HTTPS and caching)
aws cloudfront create-distribution \
  --origin-domain-name bluejay-terminator-frontend.s3.amazonaws.com \
  --default-root-object index.html
```

**Configure Frontend API URL:**

Before building the frontend for production, update the API URL:
```bash
# In frontend/.env.production
VITE_API_URL=https://your-apprunner-url.awsapprunner.com
```

Or set it during build:
```bash
VITE_API_URL=https://your-backend-url npm run build
```

## Persistent Storage for ChromaDB

### Option 1: EFS (ECS Fargate only)

The CloudFormation template automatically configures EFS. The EFS mount is:
- Mounted at `/app/chroma_db` in the container
- Persists across container restarts
- Encrypted at rest
- Automatically backed up

### Option 2: S3 Snapshots (App Runner)

For App Runner (which doesn't support EFS), use S3 for backups:

```bash
# Add to container startup script or cron
aws s3 sync /app/chroma_db s3://bluejay-chroma-backups/$(date +%Y%m%d)/

# Restore on container start
aws s3 sync s3://bluejay-chroma-backups/latest/ /app/chroma_db/
```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `LIVEKIT_URL` | LiveKit WebSocket URL | Yes |
| `LIVEKIT_API_KEY` | LiveKit API key | Yes |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `ELEVEN_API_KEY` | ElevenLabs API key | Yes |
| `DEEPGRAM_API_KEY` | Deepgram API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | No |
| `ENABLE_NOISE_CANCELLATION` | Enable audio noise cancellation | No |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | No (default: /app/chroma_db) |

## Monitoring & Logging

### CloudWatch Logs

```bash
# View App Runner logs
aws logs tail /aws/apprunner/bluejay-terminator --follow

# View ECS Fargate logs
aws logs tail /ecs/bluejay-terminator --follow

# Filter for errors
aws logs filter-log-events \
  --log-group-name /ecs/bluejay-terminator \
  --filter-pattern "ERROR"
```

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| CPU Utilization | < 70% | > 85% |
| Memory Utilization | < 80% | > 90% |
| Request Latency (P99) | < 1.5s | > 3s |
| Health Check Failures | 0 | > 2 |
| Error Rate | < 1% | > 5% |

### Create CloudWatch Alarms

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name bluejay-cpu-high \
  --alarm-description "CPU utilization is too high" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 85 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Cost Estimation

| Service | Configuration | Estimated Monthly Cost |
|---------|---------------|------------------------|
| App Runner | 1 vCPU, 2GB RAM | ~$30-50 |
| ECS Fargate | 1 vCPU, 2GB RAM | ~$35-55 |
| EFS | 5GB storage | ~$1.50 |
| S3 + CloudFront | Frontend hosting | ~$5-10 |
| Secrets Manager | 1 secret | ~$0.40 |
| CloudWatch Logs | 5GB/month | ~$2.50 |
| **Total** | | **~$40-75/month** |

*Note: Does not include API costs for OpenAI, ElevenLabs, Deepgram, LiveKit*

## Troubleshooting

### Agent won't connect to LiveKit

1. **Check Secrets Manager** has correct `LIVEKIT_URL`
2. **Verify security groups** allow outbound HTTPS (443)
3. **Check CloudWatch logs** for SSL errors
4. **Test connection** locally first with same credentials

```bash
# Check if secrets are readable
aws secretsmanager get-secret-value \
  --secret-id bluejay-terminator-secrets \
  --query SecretString
```

### ChromaDB data not persisting

1. **EFS:** Verify mount is configured correctly
   ```bash
   # Check EFS mount targets
   aws efs describe-mount-targets --file-system-id fs-xxxxx
   ```
2. **Check IAM permissions** for EFS access
3. **Ensure EFS** is in same VPC/subnets as ECS tasks

### High latency

1. **Region:** Deploy in same region as LiveKit
2. **CPU throttling:** Check CloudWatch metrics, scale up if needed
3. **Cold start:** App Runner has cold start delay; keep min instances at 1
4. **Network:** Check for packet loss or high latency to external APIs

### Container fails to start

1. Check CloudWatch logs for startup errors
2. Verify all required environment variables are set
3. Test Docker image locally:
   ```bash
   docker run -p 8080:8080 \
     -e LIVEKIT_URL=wss://... \
     -e LIVEKIT_API_KEY=... \
     bluejay-terminator-agent:latest
   ```

## Rollback Strategy

### App Runner

```bash
# List previous deployments
aws apprunner list-operations --service-arn <service-arn>

# Rollback to previous image
aws apprunner update-service \
  --service-arn <service-arn> \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:previous-tag"
    }
  }'
```

### ECS Fargate

```bash
# List task definition revisions
aws ecs list-task-definitions --family-prefix bluejay-terminator-task

# Update service to previous revision
aws ecs update-service \
  --cluster bluejay-terminator-cluster \
  --service bluejay-terminator-service \
  --task-definition bluejay-terminator-task:1
```

## Deployment Checklist

- [ ] All API keys stored in AWS Secrets Manager
- [ ] Docker image built for `linux/amd64` platform
- [ ] Docker image pushed to ECR with version tag
- [ ] App Runner/ECS service created and running
- [ ] Health check passing (`/health` endpoint)
- [ ] EFS volume mounted (for ECS) or S3 backup configured (for App Runner)
- [ ] Frontend built with correct API URL
- [ ] Frontend deployed to S3 + CloudFront
- [ ] CloudFront invalidation completed
- [ ] CloudWatch logs configured
- [ ] Tested end-to-end: frontend → backend → LiveKit
- [ ] Verified RAG queries work with persisted data
- [ ] Monitored first 24 hours for errors

## Security Best Practices

1. **Never commit secrets** to Git - use Secrets Manager
2. **Enable encryption** at rest (EFS, S3) and in transit (HTTPS)
3. **Use least privilege** IAM roles
4. **Enable VPC** flow logs for network monitoring
5. **Regular updates** - rebuild Docker images with security patches
6. **Enable CloudTrail** for API audit logging

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Build and push Docker image
        run: |
          docker build -t bluejay-terminator-agent ./backend
          docker tag bluejay-terminator-agent:latest \
            ${{ steps.login-ecr.outputs.registry }}/bluejay-terminator-agent:${{ github.sha }}
          docker push ${{ steps.login-ecr.outputs.registry }}/bluejay-terminator-agent:${{ github.sha }}
      
      - name: Deploy to App Runner
        run: |
          aws apprunner start-deployment \
            --service-arn ${{ secrets.APPRUNNER_SERVICE_ARN }}
```
