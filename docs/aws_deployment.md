# AWS Deployment Guide

## Overview

This project is designed for **AWS deployment** to meet the Bluejay take-home interview bonus requirements. The backend agent runs in a Docker container on AWS infrastructure.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
│                                                                   │
│  ┌───────────────────┐          ┌──────────────────┐            │
│  │  CloudFront CDN   │          │   App Runner     │            │
│  │   (Frontend)      │          │  (Backend Agent) │            │
│  └─────────┬─────────┘          └────────┬─────────┘            │
│            │                              │                       │
│  ┌─────────▼─────────┐          ┌────────▼─────────┐            │
│  │   S3 Bucket       │          │   EFS Volume     │            │
│  │  (React Build)    │          │  (ChromaDB Data) │            │
│  └───────────────────┘          └──────────────────┘            │
│                                                                   │
│  ┌────────────────────────────────────────────────┐             │
│  │         AWS Secrets Manager                     │             │
│  │  - LIVEKIT_API_KEY                             │             │
│  │  - OPENAI_API_KEY                              │             │
│  │  - ELEVENLABS_API_KEY                          │             │
│  │  - DEEPGRAM_API_KEY                            │             │
│  │  - TAVILY_API_KEY                              │             │
│  └────────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  LiveKit Cloud  │
                    │   (External)    │
                    └─────────────────┘
```

## Deployment Options

### Option 1: AWS App Runner (Recommended - Easiest)

**Pros:**
- Automatic scaling
- Built-in load balancing
- Simple configuration
- Pay-per-use pricing

**Cons:**
- Less control over infrastructure
- Limited customization

### Option 2: ECS Fargate (More Control)

**Pros:**
- Full VPC control
- Custom networking
- More monitoring options
- Better for production scale

**Cons:**
- More complex setup
- Requires ALB/VPC configuration

## Deployment Steps

### Prerequisites

1. AWS CLI installed and configured
2. Docker installed locally
3. AWS Account with appropriate permissions

### Step 1: Prepare Secrets in AWS Secrets Manager

```bash
# Create secret for all API keys
aws secretsmanager create-secret \
  --name bluejay-terminator-secrets \
  --description "API keys for Bluejay Terminator voice agent" \
  --secret-string '{
    "LIVEKIT_API_KEY": "your_livekit_api_key",
    "LIVEKIT_API_SECRET": "your_livekit_api_secret",
    "LIVEKIT_URL": "wss://your-project.livekit.cloud",
    "OPENAI_API_KEY": "your_openai_api_key",
    "ELEVENLABS_API_KEY": "your_elevenlabs_api_key",
    "DEEPGRAM_API_KEY": "your_deepgram_api_key",
    "TAVILY_API_KEY": "your_tavily_api_key"
  }'
```

### Step 2: Build and Push Docker Image

```bash
# Navigate to backend directory
cd backend

# Build Docker image
docker build -t bluejay-terminator-agent .

# Create ECR repository
aws ecr create-repository --repository-name bluejay-terminator-agent

# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <your-account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag bluejay-terminator-agent:latest \
  <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest

# Push to ECR
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest
```

### Step 3A: Deploy with App Runner

```bash
# Create App Runner service
aws apprunner create-service \
  --service-name bluejay-terminator \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest",
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
  }'
```

### Step 3B: Deploy with ECS Fargate (Alternative)

1. **Create ECS Cluster:**
```bash
aws ecs create-cluster --cluster-name bluejay-terminator-cluster
```

2. **Create Task Definition:**
```json
{
  "family": "bluejay-terminator-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "agent",
      "image": "<your-account-id>.dkr.ecr.us-east-1.amazonaws.com/bluejay-terminator-agent:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "secrets": [
        {
          "name": "LIVEKIT_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:bluejay-terminator-secrets:LIVEKIT_API_KEY::"
        }
      ]
    }
  ]
}
```

3. **Create Service with ALB**

### Step 4: Deploy Frontend to S3 + CloudFront

```bash
# Build frontend
cd frontend
npm run build

# Create S3 bucket
aws s3 mb s3://bluejay-terminator-frontend

# Upload build
aws s3 sync dist/ s3://bluejay-terminator-frontend --delete

# Configure bucket for static website hosting
aws s3 website s3://bluejay-terminator-frontend \
  --index-document index.html \
  --error-document index.html

# Create CloudFront distribution
aws cloudfront create-distribution \
  --origin-domain-name bluejay-terminator-frontend.s3.amazonaws.com \
  --default-root-object index.html
```

## Persistent Storage for ChromaDB

### Option 1: EFS (Elastic File System)

```bash
# Create EFS file system
aws efs create-file-system --tags Key=Name,Value=bluejay-chroma-db

# Mount EFS to App Runner/ECS task
# In task definition, add volume mount:
{
  "volumes": [
    {
      "name": "chroma-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxx",
        "rootDirectory": "/chroma_db"
      }
    }
  ]
}
```

### Option 2: S3 Snapshots

```bash
# Backup ChromaDB periodically to S3
aws s3 sync /app/chroma_db s3://bluejay-chroma-backups/$(date +%Y%m%d)/
```

## Environment Variables

The backend expects these environment variables (via Secrets Manager):

| Variable | Description | Required |
|----------|-------------|----------|
| `LIVEKIT_API_KEY` | LiveKit API key | Yes |
| `LIVEKIT_API_SECRET` | LiveKit API secret | Yes |
| `LIVEKIT_URL` | LiveKit WebSocket URL | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Yes |
| `DEEPGRAM_API_KEY` | Deepgram API key | Yes |
| `TAVILY_API_KEY` | Tavily search API key | Yes |

## Monitoring & Logs

### CloudWatch Logs

```bash
# View App Runner logs
aws logs tail /aws/apprunner/bluejay-terminator --follow

# View ECS Fargate logs
aws logs tail /ecs/bluejay-terminator-task --follow
```

### Metrics to Monitor

- CPU/Memory utilization
- Request latency
- LiveKit connection failures
- RAG query performance

## Cost Estimation

| Service | Configuration | Estimated Monthly Cost |
|---------|---------------|------------------------|
| App Runner | 1 vCPU, 2GB RAM | ~$30-50 |
| EFS | 5GB storage | ~$1.50 |
| S3 + CloudFront | Frontend hosting | ~$5-10 |
| Secrets Manager | 1 secret | ~$0.40 |
| **Total** | | **~$40-65/month** |

*Note: Does not include API costs for OpenAI, ElevenLabs, Deepgram, LiveKit*

## Troubleshooting

### Agent won't connect to LiveKit

1. Check Secrets Manager has correct `LIVEKIT_URL`
2. Verify security groups allow outbound HTTPS (443)
3. Check CloudWatch logs for SSL errors

### ChromaDB data not persisting

1. Verify EFS mount is configured correctly
2. Check IAM permissions for EFS access
3. Ensure EFS is in same availability zone

### High latency

1. Check App Runner/ECS is in same region as LiveKit
2. Monitor CloudWatch metrics for CPU throttling
3. Consider scaling up instance size

## Deployment Checklist

- [ ] All API keys stored in AWS Secrets Manager
- [ ] Docker image built and pushed to ECR
- [ ] App Runner/ECS service created and running
- [ ] EFS volume mounted for ChromaDB persistence
- [ ] Frontend deployed to S3 + CloudFront
- [ ] CloudWatch logs configured
- [ ] Tested end-to-end: frontend → backend → LiveKit
- [ ] Verified RAG queries work with persisted data
- [ ] Monitored first 24 hours for errors

## Rollback Strategy

If deployment fails:

```bash
# App Runner: Deploy previous revision
aws apprunner start-deployment --service-arn <service-arn> --previous-image-repository-type ECR

# ECS: Update service to previous task definition
aws ecs update-service --cluster bluejay-terminator-cluster \
  --service bluejay-terminator-service \
  --task-definition bluejay-terminator-task:1
```

