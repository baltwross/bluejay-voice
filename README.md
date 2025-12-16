# Bluejay Terminator Voice Agent (T-800)

A RAG-enabled voice agent with the personality of the T-800 Terminator, designed to help software engineers stay ahead of AI obsolescence. Built with LiveKit, React, and Python.

## 🏗 System Architecture

The system consists of three main components:
1.  **Frontend (React)**: A futuristic "Terminator HUD" interface for voice interaction and real-time transcription.
2.  **Backend Agent (Python)**: The core logic running on `livekit-agents`. It handles:
    *   **STT**: Deepgram
    *   **LLM**: OpenAI GPT-4o
    *   **TTS**: ElevenLabs
    *   **RAG**: LlamaIndex + ChromaDB
    *   **Tools**: Tavily (News), Custom RAG, Article Reader.
3.  **Token Server (FastAPI)**: Issues LiveKit access tokens to the frontend.

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- API Keys for: LiveKit, OpenAI, ElevenLabs, Deepgram, Tavily.

### 1. Backend Setup
Create a `.env` file in `backend/` with your keys (see `backend/.env.example`).

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the Token Server (Terminal 1)
python server.py

# Run the Voice Agent (Terminal 2)
python agent.py dev
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Click "Initialize Uplink" to connect.

## 🧠 Features

### 1. RAG (Retrieval Augmented Generation)
- **Ingestion**: Supports PDFs and Web URLs.
- **Usage**: Paste a URL into the frontend input box. The agent will ingest it and say "Intelligence received."
- **Query**: Ask questions about the uploaded content.
- **Reading Mode**: Ask "Read the article [Title]" and the agent will read it aloud.

### 2. News Feed (Tool Use)
- Ask "What's the latest in AI engineering?" or "Scan for intelligence."
- The agent uses Tavily to search for the latest software engineering AI tools and reports back.

### 3. Personality
- Stoic, literal, mission-focused T-800.
- Uses audio visualizers and CRT effects for immersion.

## ☁️ AWS Deployment (Bonus)

To deploy this agent on AWS:

1.  **Containerize**: Use the provided `backend/Dockerfile`.
    ```bash
    docker build -t bluejay-terminator-backend ./backend
    ```
2.  **Push to ECR**: Push the image to Amazon Elastic Container Registry.
3.  **Deploy Agent**:
    - Use **AWS ECS (Fargate)**.
    - Create a Task Definition with the environment variables (API Keys).
    - Command: `python agent.py start`.
    - Set up a Service. The agent will connect outbound to LiveKit Cloud (WebSocket), so no inbound ports needed for the agent itself.
4.  **Deploy Token Server**:
    - Use **AWS App Runner** or ECS.
    - Command: `python server.py`.
    - Expose port 8000.
    - Update Frontend `App.tsx` to point to the deployed Token Server URL.

## 📜 Design Decisions
- **LlamaIndex**: Chosen for robust data ingestion and retrieval capabilities.
- **ChromaDB**: Used as a simple, persistent local vector store.
- **LiveKit Agents**: Provides a powerful, real-time voice pipeline with built-in VAD and turn-taking.
- **Separation of Concerns**: Agent logic is decoupled from the web server.

## ⚠️ Notes
- Ensure `TAVILY_API_KEY` is set for the News feature.
- The "Reading Mode" chunks text to allow for interruptibility.
