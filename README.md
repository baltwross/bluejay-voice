# Bluejay Terminator - AI Tools Voice Companion

A RAG-enabled voice agent built with LiveKit that helps software engineers stay on the bleeding edge of AI tools for engineering.

## Overview

The **Bluejay Terminator** is a T-800 series android sent back from August 2027—a future where a superintelligent AI took all Software Engineer jobs in less than 4 months. Reprogrammed by the resistance, his mission is to prevent that future by helping software engineers become absolute experts on the latest AI tools for engineering.

**The Strategy:** Human + AI > AI agents alone, but only if the human is operating at the bleeding edge. The Terminator helps engineers achieve 5x, 10x, even 100x productivity increases using the latest AI tools.

**The Mission:** Prevent August 29, 2027. Time remaining: 622 days.

It allows users to:
1.  **Stay Ahead**: Discuss the latest AI tools for software engineering (Claude Code, Cursor Composer, GitHub Copilot Workspace, etc.).
2.  **Master Documentation**: Upload tool documentation (PDFs, Webpages, YouTube Videos) and have the Terminator explain how to use them for maximum productivity.
3.  **Real-time Learning**: Low-latency voice conversation with a stoic, mission-focused personality.

## System Architecture

### 1. Frontend (React + Vite)
- **Framework**: React 18 with TypeScript and Vite.
- **Styling**: Tailwind CSS with custom Terminator HUD theme.
- **Core Features**:
    - WebRTC connection via `livekit-client`.
    - Real-time transcript display.
    - Audio visualization.
    - Document upload & URL input interface.
- **Deployment**: AWS S3 + CloudFront (static hosting).

### 2. Backend (Python)
- **Framework**: LiveKit Agents (Python SDK).
- **Voice Pipeline**:
    - **STT (Speech-to-Text)**: Deepgram (for speed/interruptibility).
    - **LLM**: OpenAI `gpt-5-nano-2025-08-07` (for reasoning).
    - **TTS (Text-to-Speech)**: ElevenLabs (custom Arnold voice, Voice ID: `8DGMp3sPQNZOuCfSIxxE`).
    - **VAD**: Silero (for natural interruption handling).
- **RAG Engine**:
    - **Framework**: LangChain.
    - **Vector Store**: ChromaDB (persistent storage).
    - **Ingestion**: Custom pipeline for PDF, YouTube, and Web scraping.
- **Deployment**: AWS App Runner or ECS Fargate (Dockerized).

### 3. AWS Deployment Strategy
- **Backend Agent**: Docker container on AWS App Runner or ECS Fargate
- **Environment Variables**: AWS Secrets Manager
- **Vector Store**: ChromaDB with persistent volume (EFS) or S3 backup
- **Frontend**: S3 + CloudFront (static hosting)
- **LiveKit**: LiveKit Cloud (external SaaS)

### 4. Data Flow
1.  **Ingestion**: User uploads file/link → Backend processes & chunks → Embeddings → ChromaDB.
2.  **Retrieval**: Agent identifies need for external info → Queries Vector Store → Synthesizes answer.

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- LiveKit Cloud Account
- OpenAI API Key
- ElevenLabs API Key
- Deepgram API Key

### Installation

1.  **Clone the repo**
    ```bash
    git clone <repo_url>
    cd bluejay-voice
    ```

2.  **Backend Setup**
    ```bash
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp env.template .env
    # Edit .env with your API keys
    ```

3.  **Frontend Setup**
    ```bash
    cd frontend
    npm install
    ```

### Running the Application

**Option 1: Quick Start (All Services)**
```bash
./scripts/dev.sh
```
This starts the token server, LiveKit agent, and frontend dev server together.

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
- Token API: http://localhost:8080

### Frontend Architecture

The React frontend (`frontend/`) includes:

| Component | Purpose |
|-----------|---------|
| `AgentVisualizer` | Audio waveform visualization with T-800 skull icon |
| `Transcript` | Real-time conversation transcript with auto-scroll |
| `ControlPanel` | Start/end call, mic toggle, connection status |
| `InputConsole` | "Share with Terminator" URL/file upload interface |

**Key Hooks:**
- `useConnection` - Manages token fetching and connection state
- `useAgent` - LiveKit room connection and agent detection
- `useDocuments` - Document ingestion and listing

**State Flow:**
1. User clicks "Initialize Connection"
2. Frontend fetches token from `/api/token`
3. LiveKitRoom connects with token
4. Agent joins the room automatically
5. Audio streaming begins via WebRTC

## Design Decisions & Trade-offs

### RAG Framework
- **LangChain**: Chosen for its robust ecosystem, flexible chains, and extensive integration support for RAG pipelines. Provides excellent document loaders for PDF, YouTube, and web content.

### Vector Store
- **ChromaDB**: Local, file-based vector store for development. In production, uses persistent volumes (EFS) or S3 snapshots. Chosen for simplicity and no external database dependencies.

### Voice Pipeline
- **Deepgram STT**: Selected for lower latency (~300ms) compared to Whisper, essential for natural conversation flow.
- **ElevenLabs TTS**: Custom Arnold Schwarzenegger voice (Voice ID: `8DGMp3sPQNZOuCfSIxxE`) for authentic T-800 personality.
- **Silero VAD**: Enables natural interruption during agent speech - critical for reading mode.

### AWS Deployment
- **App Runner vs ECS Fargate**: App Runner chosen for simpler deployment and automatic scaling. ECS Fargate provides more control but requires VPC/ALB setup.
- **Persistent Storage**: ChromaDB data stored on EFS mount or S3 with periodic snapshots.
- **Secrets Management**: All API keys stored in AWS Secrets Manager, injected as environment variables.

### Chunking Strategy
- **Chunk Size**: 1000 tokens with 200 token overlap for optimal retrieval accuracy.
- **Embedding Model**: `text-embedding-3-large` (OpenAI) - best quality embeddings, recommended by LangChain for RAG.

## Local Development vs Production

| Component | Local | Production (AWS) |
|-----------|-------|------------------|
| Backend Agent | `python agent.py dev` | Docker on App Runner/Fargate |
| Frontend | `npm run dev` (localhost:5173) | S3 + CloudFront |
| Vector DB | `./chroma_db/` directory | EFS mount or S3 |
| Secrets | `.env` file | AWS Secrets Manager |
| LiveKit | LiveKit Cloud | LiveKit Cloud |

## Future Improvements
- Persistent user conversation history across sessions.
- Multi-document comparison queries.
- Voice emotion detection for user frustration handling.

