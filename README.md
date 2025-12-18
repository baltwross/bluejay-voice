# Bluejay Terminator - AI Tools Voice Companion

A RAG-enabled voice agent built with LiveKit that helps software engineers stay on the bleeding edge of AI tools for engineering.

## 🚀 Live Demo

**Try it now:** [https://d1fvldby568qae.cloudfront.net](https://d1fvldby568qae.cloudfront.net)

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| **Frontend (UI)** | https://d1fvldby568qae.cloudfront.net |
| **Backend API** | https://y4se9mhdyc.us-east-1.awsapprunner.com |

> **Try it now:** Visit the frontend URL, click to connect, and start talking to the T-800!

## Overview

The **Bluejay Terminator** is a T-800 series android sent back from August 2027—a future where a superintelligent AI took all Software Engineer jobs in less than 4 months. Reprogrammed by the resistance, his mission is to prevent that future by helping software engineers become absolute experts on the latest AI tools for engineering.

**The Strategy:** Human + AI > AI agents alone, but only if the human is operating at the bleeding edge. The Terminator helps engineers achieve 5x, 10x, even 100x productivity increases using the latest AI tools.

**The Mission:** Prevent August 29, 2027. Time remaining: 622 days.

### Core Features

1. **Stay Ahead**: Discuss the latest AI tools for software engineering (Claude Code, Cursor Composer, GitHub Copilot Workspace, etc.).
2. **Master Documentation**: Upload tool documentation (PDFs, Webpages, YouTube Videos) and have the Terminator explain how to use them for maximum productivity.
3. **Real-time Learning**: Low-latency voice conversation with a stoic, mission-focused T-800 personality.
4. **AI News Feed**: Get the latest news on AI tools for engineering via Tavily search integration.
5. **Document Reading**: Have the agent read documents aloud with full interruptibility and resume capability.

## Quick Start

### Local Development

```bash
# Clone the repository
git clone <repo_url>
cd bluejay-voice

# Start all services (token server, agent, frontend)
./scripts/dev.sh
```

### Docker (Local)

```bash
# Copy environment template
cp env.docker.template .env
# Edit .env with your API keys

# Start all services
docker-compose up --build

# Access the app at http://localhost:5173
```

### AWS Deployment

The application is deployed on AWS with:
- **Backend**: AWS App Runner (auto-scaling container service)
- **Frontend**: S3 + CloudFront (global CDN)
- **Secrets**: AWS Secrets Manager

```bash
# Setup secrets in AWS Secrets Manager
./scripts/setup-secrets.sh --from-env

# Build and deploy backend to AWS App Runner
./scripts/deploy-aws.sh

# Deploy frontend to S3/CloudFront
./scripts/deploy-aws.sh --frontend-only

# Deploy both backend and frontend
./scripts/deploy-aws.sh --all
```

See `scripts/deploy-aws.sh` and `infrastructure/cloudformation.yaml` for detailed AWS deployment instructions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                            │
│                    WebRTC Audio/Video                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      LiveKit Cloud                              │
│                (WebRTC Infrastructure)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                    Python Backend                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Deepgram   │  │   OpenAI    │  │      ElevenLabs         │ │
│  │    STT      │──│  gpt-4o-mini│──│   TTS (Arnold Voice)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                         │                                       │
│  ┌─────────────────────▼────────────────────────────────────┐  │
│  │               RAG Engine (LangChain + ChromaDB)           │  │
│  │  • PDF, YouTube, Web ingestion                            │  │
│  │  • Hybrid search (semantic + keyword)                     │  │
│  │  • Document reading with position tracking                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               External Tools                              │  │
│  │  • Tavily AI News Search                                  │  │
│  │  • Document reading controls                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 18 + Vite + Tailwind | WebRTC client, transcript display, document upload |
| **Backend Agent** | LiveKit Agents (Python) | Voice pipeline orchestration |
| **STT** | Deepgram | Real-time speech-to-text |
| **LLM** | OpenAI gpt-4o-mini | Conversation and reasoning |
| **TTS** | ElevenLabs | Arnold-style voice synthesis |
| **VAD** | Silero | Voice activity detection for interruptions |
| **RAG** | LangChain + ChromaDB | Document retrieval and Q&A |
| **News** | Tavily | AI engineering news search |

## Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **Docker** (optional, for containerized deployment)

### API Keys Required

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| LiveKit Cloud | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | WebRTC infrastructure |
| OpenAI | `OPENAI_API_KEY` | LLM and embeddings |
| ElevenLabs | `ELEVEN_API_KEY` | Text-to-speech |
| Deepgram | `DEEPGRAM_API_KEY` | Speech-to-text |
| Tavily | `TAVILY_API_KEY` | News search (optional) |

## Installation

### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy environment template and add your API keys
cp env.template .env
# Edit .env with your API keys
```

### 2. Frontend Setup

```bash
cd frontend
npm install
```

### 3. Running the Application

**Option 1: Quick Start (All Services)**
```bash
./scripts/dev.sh
```

**Option 2: Manual Start (Separate Terminals)**

Terminal 1 - Token Server:
```bash
cd backend
source venv/bin/activate
python token_server.py
```

Terminal 2 - LiveKit Agent:
```bash
cd backend
source venv/bin/activate
python agent.py dev
```

Terminal 3 - Frontend:
```bash
cd frontend
npm run dev
```

**Access the Application:**
- Frontend: http://localhost:5173
- Token API: http://localhost:8080/api/info

**Production URLs:**
- Frontend: https://d1fvldby568qae.cloudfront.net
- Backend API: https://y4se9mhdyc.us-east-1.awsapprunner.com/api/info

## Docker Usage

### Development with Docker Compose

```bash
# Copy environment template
cp env.docker.template .env
# Edit .env with your API keys

# Build and start all services
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend
```

### Production Build

```bash
# Build production images
./scripts/build.sh

# Or manually:
docker build -t bluejay-terminator-agent ./backend
docker build --target production -t bluejay-terminator-frontend ./frontend
```

## API Endpoints

The token server provides these HTTP endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/info` | GET | API documentation and endpoint list |
| `/health` | GET | Health check with component status |
| `/api/token` | POST | Generate LiveKit access token |
| `/api/ingest` | POST | Ingest URL content (YouTube, web, PDF) |
| `/api/ingest/file` | POST | Upload and ingest file |
| `/api/documents` | GET | List all ingested documents |
| `/api/newsfeed` | GET | Get AI tools news feed |
| `/api/transcripts` | POST | Save conversation transcript |

## Frontend Components

| Component | Purpose |
|-----------|---------|
| `AgentVisualizer` | Audio waveform visualization with T-800 skull icon |
| `Transcript` | Real-time conversation transcript with auto-scroll |
| `ControlPanel` | Start/end call, mic toggle, connection status |
| `InputConsole` | "Share with Terminator" URL/file upload interface |

## Voice Agent Features

### RAG (Retrieval-Augmented Generation)

The agent can ingest and answer questions about:
- **PDF documents** - Technical documentation, papers
- **YouTube videos** - Tutorial transcripts
- **Web articles** - Blog posts, news articles
- **Word documents** - .docx files

### Document Reading Mode

The agent can read documents aloud with:
- **Interruptibility** - Ask questions mid-reading
- **Resume capability** - Continue from where you left off
- **Position tracking** - Knows your progress percentage

### AI News Search

Ask about the latest AI tools:
- "What's happening in AI today?"
- "Tell me about the latest Claude updates"
- "What new coding assistants are available?"

## Project Structure

```
bluejay-voice/
├── backend/
│   ├── agent.py           # Main LiveKit agent
│   ├── token_server.py    # FastAPI token server
│   ├── config.py          # Centralized configuration
│   ├── prompts.py         # T-800 personality prompts
│   ├── Dockerfile         # Production container
│   └── rag/
│       ├── indexer.py     # Document ingestion
│       ├── retriever.py   # Document retrieval
│       └── loaders.py     # Content loaders
├── frontend/
│   ├── src/
│   │   ├── App.tsx        # Main application
│   │   ├── components/    # UI components
│   │   └── hooks/         # React hooks
│   ├── Dockerfile         # Multi-stage build
│   └── nginx.conf         # Production server config
├── infrastructure/
│   └── cloudformation.yaml # AWS ECS deployment
├── scripts/
│   ├── dev.sh             # Development startup
│   ├── build.sh           # Docker build script
│   ├── deploy-aws.sh      # AWS deployment
│   └── setup-secrets.sh   # AWS secrets setup
├── docs/                 # (ignored in git for submission)
│   ├── aws_deployment.md  # Detailed AWS guide
│   ├── technical_design.md # Architecture details
│   └── product_requirements.md # Feature specs
├── docker-compose.yml     # Local development
└── docker-compose.prod.yml # Production config
```

## Design Decisions & Trade-offs

### Voice Pipeline
- **Deepgram STT**: Lower latency (~300ms) vs Whisper
- **ElevenLabs TTS**: Custom Arnold voice (ID: `8DGMp3sPQNZOuCfSIxxE`)
- **Silero VAD**: Enables natural interruption during agent speech

### RAG Framework
- **LangChain**: Robust ecosystem, flexible chains, extensive integrations
- **ChromaDB**: Local, file-based for development; EFS for production
- **Hybrid Search**: Combines semantic embeddings with BM25 keyword search

### Chunking Strategy
- **Chunk Size**: 1000 tokens with 200 token overlap
- **Embedding Model**: OpenAI `text-embedding-3-large`

### AWS Deployment
- **App Runner**: Simple deployment, automatic scaling (recommended)
- **ECS Fargate**: More control, EFS support for persistence
- **Secrets Manager**: Secure API key storage

## Deployment Comparison

| Component | Local | Docker | AWS Production |
|-----------|-------|--------|----------------|
| Backend | `python agent.py dev` | `docker-compose up` | [App Runner](https://y4se9mhdyc.us-east-1.awsapprunner.com) |
| Frontend | `npm run dev` (port 5173) | Port 5173 | [CloudFront](https://d1fvldby568qae.cloudfront.net) |
| Vector DB | `backend/chroma_db/` | Docker volume | EFS mount |
| Secrets | `.env` file | `.env` file | Secrets Manager |

## Testing

```bash
# Backend tests
cd backend
source venv/bin/activate
pytest

# API endpoint tests
python -m pytest tests/test_api_endpoints.py -v
```

## Troubleshooting

### Agent won't connect
1. Check API keys in `.env`
2. Verify LiveKit URL format: `wss://your-project.livekit.cloud`
3. Check CloudWatch/console logs for errors

### High latency
1. Ensure backend is in same region as LiveKit
2. Try disabling noise cancellation: `ENABLE_NOISE_CANCELLATION=false`
3. Check network connectivity

### RAG not returning results
1. Verify documents are ingested: `GET /api/documents`
2. Check ChromaDB persistence directory
3. Review ingestion logs

## Future Improvements

- [ ] Voice emotion detection for user frustration handling
- [ ] Multi-document comparison queries
- [ ] Persistent conversation history
- [ ] Custom voice training

## License

MIT License - See LICENSE file for details.
