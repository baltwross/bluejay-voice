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

This section details the architectural decisions, assumptions, and limitations of the Bluejay Terminator voice agent system.

### Trade-offs & Limitations

#### Voice Pipeline Trade-offs
- **STT Choice (Deepgram vs Whisper)**: Chose Deepgram `nova-3` for lower latency (~300ms) and real-time streaming capabilities. Trade-off: Requires API key and incurs per-minute costs vs. free local Whisper, but latency is critical for natural conversation flow.
- **TTS Choice (ElevenLabs vs OpenAI)**: Selected ElevenLabs for custom Arnold-style voice synthesis. Trade-off: Higher cost per character vs. OpenAI TTS, but provides superior voice quality and character consistency.
- **VAD (Silero)**: Uses Silero VAD for voice activity detection to enable natural interruptions. Trade-off: Adds ~50-100ms processing overhead, but essential for reading mode where users can interrupt mid-sentence.
- **Preemptive Generation**: Enabled preemptive LLM response generation to reduce perceived latency. Trade-off: May generate responses for incomplete user turns, but significantly improves conversation responsiveness.

#### RAG System Limitations
- **Single Vector Store**: All users share the same ChromaDB collection. **Limitation**: No multi-user isolation; documents uploaded by one user are visible to all users. For production, would need per-user collections or document-level access control.
- **In-Memory News Cache**: News search results cached in memory with 5-minute TTL. **Limitation**: Cache is lost on agent restart; no persistent cache across deployments.
- **Reading State Persistence**: Reading position stored in local JSON file. **Limitation**: Not suitable for multi-instance deployments; should use Redis or database for shared state.
- **Re-ranking Dependency**: CrossEncoder re-ranking significantly improves precision for "specific fact" queries but requires `sentence-transformers` package. **Limitation**: Optional dependency adds ~500MB to Docker image; system degrades gracefully without it.
- **Chunk Boundary Issues**: Fixed chunk size (1000 tokens) may split answers across chunks. **Mitigation**: Hybrid search with neighbor expansion (±1 chunk) helps, but complex answers spanning multiple chunks may require multiple retrieval passes.

#### Hosting & Infrastructure Limitations
- **EFS for ChromaDB**: Uses AWS EFS for persistent vector store in production. **Limitation**: EFS has higher latency than local disk; first query after cold start may be slower. Acceptable trade-off for persistence across container restarts.
- **No Database**: Reading state and session data stored in files, not a database. **Limitation**: Not suitable for horizontal scaling; would need Redis/DynamoDB for multi-instance deployments.
- **Secrets Management**: AWS Secrets Manager for API keys. **Assumption**: Secrets are pre-configured; deployment script doesn't create secrets automatically.

### Hosting Assumptions

#### Development Environment
- **Local Development**: Assumes developers run backend locally with Python 3.9+, ChromaDB stored in `backend/chroma_db/` directory.
- **Docker Development**: Assumes Docker Compose for local testing; ChromaDB persisted in Docker volume.
- **LiveKit Cloud**: Assumes LiveKit Cloud account with project URL, API key, and secret configured.

#### Production Deployment (AWS)
- **Backend Hosting**: Assumes deployment to either:
  - **AWS App Runner** (recommended): Auto-scaling, managed service, simpler setup. Assumes ECR repository exists and Secrets Manager secret is pre-configured.
  - **ECS Fargate**: More control, supports EFS mounts. Assumes CloudFormation stack deployment with VPC, EFS, and IAM roles.
- **Frontend Hosting**: Assumes S3 bucket for static assets and CloudFront distribution for CDN. Bucket must be configured for public read access.
- **Vector Store Persistence**: 
  - **App Runner**: Assumes ChromaDB stored in container filesystem (ephemeral) OR external EFS mount (requires custom setup).
  - **ECS Fargate**: Assumes EFS file system mounted at `/app/chroma_db` for persistence across container restarts.
- **Secrets**: Assumes AWS Secrets Manager secret named `bluejay-terminator-secrets` with keys: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVEN_API_KEY`, `TAVILY_API_KEY`.
- **Network**: Assumes backend can make outbound HTTPS connections to LiveKit Cloud, OpenAI, Deepgram, ElevenLabs, and Tavily APIs.

### RAG Assumptions

#### Vector Database Choice: ChromaDB
- **Rationale**: ChromaDB chosen for simplicity, local-first architecture, and LangChain integration. File-based persistence works well for single-instance deployments.
- **Alternative Considered**: Pinecone (managed) or Weaviate (self-hosted) for better scalability, but added complexity and cost.
- **Assumption**: Single ChromaDB collection (`bluejay_knowledge_base`) shared across all documents. For multi-tenant, would need per-user collections.

#### Chunking Strategy
- **Chunk Size**: 1000 tokens (approximately 750-1000 words) with 200 token overlap.
  - **Rationale**: Balances context completeness (larger chunks) with retrieval precision (smaller chunks). 1000 tokens captures most paragraph-level concepts without excessive noise.
  - **Overlap**: 200 tokens ensures continuity across chunk boundaries, critical for answers spanning multiple sections.
- **Splitter**: `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`.
  - **Rationale**: Preserves paragraph structure (prefers double newlines) while ensuring chunks don't exceed size limit.
- **Anchor Injection**: Structural anchors (Figure 37, Table 2.3, Section 4.1) extracted and appended to chunks for BM25 keyword boost.
  - **Rationale**: Enables precise retrieval of specific references mentioned in queries like "What does Figure 37 show?"

#### Embedding Model
- **Model**: OpenAI `text-embedding-3-large` (3072 dimensions).
  - **Rationale**: High-quality embeddings with strong semantic understanding. Alternative: `text-embedding-3-small` (1536 dims) for lower cost, but `-large` provides better accuracy for technical documents.
- **Assumption**: OpenAI API key available and embedding costs are acceptable for the use case.

#### Retrieval Strategy
- **Hybrid Search**: Combines semantic similarity (embedding-based) with BM25 keyword search.
  - **Semantic Weight**: 0.4-0.5 (default 0.5) for general queries; 0.4 for document-specific queries to favor keyword matching.
  - **Rationale**: Semantic search handles conceptual queries ("How does RAG work?"), while BM25 excels at specific references ("reference 15", "section 3.2").
- **Re-ranking**: CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision.
  - **Rationale**: Re-ranker scores query-document pairs more accurately than embedding similarity alone. Critical for "specific fact in specific chapter" style questions.
  - **Pipeline**: Retrieve top 20 candidates → Re-rank to top 8 → Inject top 4 into LLM context.
- **Confidence Thresholds**:
  - **Automatic RAG injection**: 0.35 minimum relevance (avoids injecting weakly-related context).
  - **Tool calls**: 0.25 minimum (more permissive for explicit user requests).
  - **Low confidence warning**: Below 0.2, agent admits uncertainty.

#### Framework Choice: LangChain
- **Rationale**: 
  - Robust ecosystem with extensive integrations (ChromaDB, OpenAI, document loaders).
  - Flexible chains for complex retrieval pipelines.
  - Active community and documentation.
- **Alternative Considered**: LlamaIndex (strong RAG focus) or direct ChromaDB API (more control, less abstraction). Chose LangChain for balance of features and maintainability.

### LiveKit Agent Design

#### Architecture Overview
The agent follows LiveKit's recommended architecture:
1. **AgentSession**: Orchestrates STT → LLM → TTS pipeline with VAD for turn-taking.
2. **TerminatorAssistant**: Extends `Agent` class with RAG integration and tool calls.
3. **RAG Integration**: Context injected via `on_user_turn_completed` hook before LLM processes user message.

#### STT Configuration (Deepgram)
- **Model**: `nova-3` (latest, optimized for accuracy and latency).
- **Language**: `en-US` (assumes English-only conversations).
- **Interim Results**: Enabled for real-time transcript display.
- **Endpointing**: 25ms silence threshold for quick turn detection.
- **No Delay**: `no_delay=True` for minimal latency.

#### LLM Configuration (OpenAI)
- **Model**: `gpt-4o-mini` (fast, cost-effective, sufficient for voice conversations).
  - **Alternative Considered**: `gpt-4o` for better reasoning, but `-mini` provides 10x lower cost with acceptable quality for voice.
- **Temperature**: 0.2 (low for consistent, deterministic responses).
- **System Prompt**: T-800 Terminator personality defined in `prompts.py` with clear instructions for tool usage and RAG context handling.

#### TTS Configuration (ElevenLabs)
- **Voice ID**: `8DGMp3sPQNZOuCfSIxxE` (custom Arnold-style voice).
- **Model**: `eleven_multilingual_v2` (supports multiple languages, though primarily English).
- **Streaming**: Optimized for low latency with `optimize_streaming_latency=4`.
- **Voice Switching**: Supports dynamic voice mode switching (Terminator/Inspire/Fate) via tool calls.

#### VAD Configuration (Silero)
- **Min Speech Duration**: 0.05s (quick interruption detection).
- **Min Silence Duration**: 0.2s (low-latency mode) or 0.5s (standard mode).
- **Turn Detection**: `turn_detection="vad"` (relies on VAD, not EOU model, for lower latency).
- **Interruption Handling**: `min_interruption_duration=0.25s` for quick user interruptions during agent speech.

#### Performance Optimizations
- **Preemptive Generation**: Starts generating LLM response before user turn fully committed (reduces perceived latency by ~500ms-1s).
- **Endpointing Delays**: `min_endpointing_delay=0.1s`, `max_endpointing_delay=0.6s` (low-latency mode) for faster response initiation.
- **Noise Cancellation**: Disabled by default (`ENABLE_NOISE_CANCELLATION=false`) to avoid processing delay; can be enabled for noisy environments.

#### RAG Integration Pattern
- **Automatic Injection**: `on_user_turn_completed` hook triggers RAG retrieval for document-related queries (keyword-based heuristic).
- **Tool-Based Retrieval**: LLM can explicitly call `search_documents` tool for precise control.
- **Active Document Context**: Tracks "active document" from recent searches/reads to filter RAG to relevant document.
- **Hybrid Search for Active Docs**: When active document is set, uses hybrid search (semantic + BM25) with higher k (12 vs 6) for better recall on specific references.

#### Tool Call Architecture
- **Function Tools**: LLM decides when to invoke tools (`@function_tool` decorator).
- **Available Tools**:
  - `search_documents`: RAG retrieval from knowledge base.
  - `ingest_url`: Process YouTube/web/PDF URLs.
  - `read_document`: Read documents aloud with position tracking.
  - `search_ai_news`: Tavily integration for latest AI news.
  - `switch_voice`: Dynamic voice mode switching.
- **Tool Error Handling**: Tools raise `ToolError` with user-friendly messages; LLM handles gracefully.

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
