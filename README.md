# Bluejay Terminator - AI Tools Voice Companion

A RAG-enabled voice agent built with LiveKit that helps software engineers stay on the bleeding edge of AI tools for engineering.

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

```bash
# Setup secrets in AWS Secrets Manager
./scripts/setup-secrets.sh --from-env

# Build and deploy to AWS App Runner
./scripts/deploy-aws.sh

# Deploy frontend to S3/CloudFront
./scripts/deploy-aws.sh --frontend-only
```

See [docs/aws_deployment.md](docs/aws_deployment.md) for detailed AWS deployment instructions.

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
├── docs/
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
- **Embedding Model**: OpenAI `text-embedding-3-small` for efficiency

### AWS Deployment
- **App Runner**: Simple deployment, automatic scaling (recommended)
- **ECS Fargate**: More control, EFS support for persistence
- **Secrets Manager**: Secure API key storage

## Deployment Comparison

| Component | Local | Docker | AWS |
|-----------|-------|--------|-----|
| Backend | `python agent.py dev` | `docker-compose up` | App Runner/ECS |
| Frontend | `npm run dev` | Port 5173 | S3 + CloudFront |
| Vector DB | `./chroma_db/` | Docker volume | EFS mount |
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

## AI Tools Used

This project was built with assistance from:
- **Claude (Anthropic)** - Architecture design and code generation
- **Cursor** - IDE with AI-powered development
- **Context7** - Documentation lookup
- **Tavily** - Web search integration

## Future Improvements

- [ ] Voice emotion detection for user frustration handling
- [ ] Multi-document comparison queries
- [ ] Persistent conversation history
- [ ] Custom voice training

## Submission (Bluejay Take-Home)

Per the take-home instructions in `docs/Bluejay Take Home Interview.pdf` ([PDF](file:///Users/rossbaltimore/bluejay-voice/docs/Bluejay%20Take%20Home%20Interview.pdf)):

1. **Push this repository to GitHub**
   - Ensure `git status` is clean (no uncommitted changes).
   - Confirm no secrets are committed (keep `.env` files out of git).

2. **Record a short demo video**
   - Show the **Start Call → live transcript → End Call** flow.
   - Show the required **tool call** in-narrative (e.g., “AI news feed”).
   - Show **RAG over a large PDF**: upload/ingest, then ask a specific fact question.
   - Optional (bonus): show the agent running on **AWS**.

3. **Share the repo**
   - Share the GitHub repo with **farazs27@gmail.com**.

4. **Email the demo**
   - Email the video + repo link to **rohan@getbluejay.ai** and **faraz@getbluejay.ai**.

## License

MIT License - See LICENSE file for details.

## Contact

For questions about this take-home interview submission, contact the Bluejay team.
