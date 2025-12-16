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
- **Styling**: Tailwind CSS + Shadcn/UI (planned).
- **Core Features**:
    - WebRTC connection via `livekit-client`.
    - Real-time transcript display.
    - Audio visualization.
    - Document upload & URL input interface.

### 2. Backend (Python)
- **Framework**: LiveKit Agents (Python SDK).
- **Voice Pipeline**:
    - **STT (Speech-to-Text)**: Deepgram (for speed/interruptibility).
    - **LLM**: OpenAI gpt-5-nano-2025-08-07 (for reasoning).
    - **TTS (Text-to-Speech)**: ElevenLabs (custom Terminator/Arnold voice) or OpenAI.
- **RAG Engine**:
    - **Framework**: LangChain.
    - **Vector Store**: ChromaDB (Persistent local storage).
    - **Ingestion**: Custom pipeline for PDF, YouTube, and Web scraping.

### 3. Data Flow
1.  **Ingestion**: User uploads file/link -> Backend processes & chunks -> Embeddings -> ChromaDB.
2.  **Retrieval**: Agent identifies need for external info -> Queries Vector Store -> Synthesizes answer.

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- LiveKit Cloud Account
- OpenAI API Key

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
    cp env.example .env
    # Edit .env with your API keys
    ```

3.  **Frontend Setup**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## Design Decisions & Trade-offs
- **LangChain**: Chosen for its robust ecosystem, flexible chains, and extensive integration support for RAG pipelines.
- **ChromaDB**: Chosen as a local, file-based vector store to simplify the "take-home" deployment without needing an external vector DB service.
- **Deepgram STT**: Selected for lower latency compared to Whisper, essential for a fluid voice conversation.

## Future Improvements
- AWS Deployment (ECS/Fargate).
- Persistent user history across sessions.
- Multi-agent collaboration (e.g., a "Debater" agent).

