# System Architecture & Technical Design

## 1. High-Level Architecture

```mermaid
graph TD
    User[User (Browser)] <-->|WebRTC Audio/Data| LiveKit[LiveKit Cloud]
    LiveKit <-->|WebSocket| Agent[Python Agent]
    
    subgraph "Python Backend"
        Agent -->|Audio Stream| STT[Deepgram STT]
        Agent -->|Text| LLM[OpenAI gpt-5-nano-2025-08-07]
        LLM -->|Text| TTS[ElevenLabs TTS]
        TTS -->|Audio| Agent
        
        Agent -- Tool Call --> Tools
        
        subgraph "RAG Engine"
            Tools -->|Query| RAG[LangChain + ChromaDB]
            RAG <-->|Embeddings| Chroma[ChromaDB]
            Ingest[Ingestion Pipeline] -->|Chunks| Chroma
        end
        
        subgraph "External World"
            Ingest -->|Scrape| Web[Websites/YouTube]
            Tools -->|Search| Tavily[Tavily Search API]
        end
    end
```

## 2. Component Detail

### 2.1 Backend (Python Agent)
*   **Framework:** `livekit-agents`
*   **Entrypoint:** `agent.py` - defines the `AgentSession` with voice pipeline.
*   **Personality:** Defined in `prompts.py` as the system prompt.

### 2.2 Agent Architecture (LiveKit Conversational Flow)

LiveKit's architecture naturally supports conversational flow without requiring explicit "modes" or complex orchestration like LangGraph. The LLM acts as the orchestrator:

#### Separation of Concerns
| Component | Location | Purpose |
|-----------|----------|---------|
| **System Prompt** | `prompts.py` | Static personality + behavior instructions |
| **RAG Context** | `on_user_turn_completed` hook | Dynamic, injected per-turn based on user query |
| **Tool Calls** | `@function_tool` decorators | LLM decides when to invoke external APIs |
| **Chat History** | `ChatContext` | Managed by LiveKit automatically |

#### RAG Integration Pattern
Per [LiveKit External Data docs](https://docs.livekit.io/agents/build/external-data), RAG context is injected dynamically using the `on_user_turn_completed` hook:

```python
async def on_user_turn_completed(
    self, turn_ctx: ChatContext, new_message: ChatMessage,
) -> None:
    # Perform RAG lookup based on user's message
    rag_content = await my_rag_lookup(new_message.text_content())
    # Inject as hidden assistant message before LLM generates response
    turn_ctx.add_message(
        role="assistant", 
        content=f"Relevant information from documents: {rag_content}"
    )
```

#### Decision Flow (No Explicit Modes)
The LLM naturally decides based on the user's message:
- **Document question** → RAG search triggered via `on_user_turn_completed`
- **News/latest AI tools** → LLM invokes `search_ai_news` tool
- **General chat** → Direct response from LLM knowledge

This eliminates the need for mode switching or complex state machines.

### 2.3 The Ingestion Pipeline (`backend/rag/`)
This is a critical subsystem. It decouples *getting content* from *using content*.

1.  **Input:** Raw File or URL.
2.  **Loader:**
    *   `PDFReader` (via `pymupdf`) for PDFs.
    *   `YoutubeTranscriptReader` for YouTube URLs.
    *   `Trafilatura` for generic Web URLs.
    *   `DocxReader` (via `python-docx`) for Word documents.
3.  **Transformation:**
    *   Clean text (remove headers/footers/timestamps).
    *   Chunking: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200).
4.  **Embedding:** `OpenAIEmbeddings` (text-embedding-3-large) - OpenAI's best embedding model, recommended by LangChain.
5.  **Storage:** `Chroma` via LangChain (persisted to `./chroma_db`).

### 2.4 The Voice Agent Configuration
*   **VAD (Voice Activity Detection):** `Silero VAD`. Essential for interruptibility.
*   **Turn Detection:** Eager. If the user pauses for > 600ms, the agent takes the turn.
*   **Voice Pipeline:**
    *   **STT:** Deepgram (real-time transcription)
    *   **LLM:** OpenAI `gpt-5-nano-2025-08-07`
    *   **TTS:** ElevenLabs with Arnold-style voice (Voice ID: configurable)

### 2.5 Frontend (React)
*   **Connection Management:** `useLiveKitRoom` hook.
*   **State:**
    *   `isRecording`: boolean.
    *   `transcript`: array of `{ sender: 'user' | 'agent', text: string }`.
    *   `activeDocument`: metadata of the currently discussed doc.
*   **RPC / Data Messages:**
    *   The Frontend sends a data packet: `{ type: "UPLOAD_URL", url: "..." }`.
    *   The Backend receives this, triggers ingestion, and sends back: `{ type: "UPLOAD_STATUS", status: "success", summary: "..." }`.
    *   The Agent then verbally acknowledges: "Got it. I just read that article..."

## 3. Data Schema (Vector DB)

**Collection:** `knowledge_base`

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID | Unique chunk ID |
| `embedding` | Vector[3072] | The semantic vector (text-embedding-3-large default) |
| `text` | String | The actual content chunk |
| `metadata.source_type` | String | "pdf", "youtube", "web" |
| `metadata.source_url` | String | Original link or filename |
| `metadata.title` | String | Title of the content |
| `metadata.timestamp` | Float | (Optional) For video transcripts |

## 4. Security & Deployment

### Local Development
*   **API Keys:** Stored in `backend/.env` file (gitignored).
*   **Run Command:** `python agent.py dev` connects to LiveKit Cloud from localhost.
*   **Vector DB:** ChromaDB persisted to `./chroma_db` directory.

### Production Deployment (AWS - REQUIRED)

**⚠️ AWS deployment is REQUIRED for this take-home interview (bonus points).**

#### Backend Agent
*   **Container:** Docker image pushed to AWS ECR
*   **Hosting:** AWS App Runner (recommended) or ECS Fargate
*   **Configuration:**
    *   CPU: 1 vCPU
    *   Memory: 2GB RAM
    *   Port: 8080
*   **Secrets:** All API keys stored in **AWS Secrets Manager**, injected as environment variables
*   **Persistence:** ChromaDB data stored on EFS mount or S3 snapshots

#### Frontend
*   **Hosting:** AWS S3 (static hosting) + CloudFront (CDN)
*   **Build:** `npm run build` → Upload to S3
*   **Domain:** CloudFront distribution URL

#### Architecture Benefits
*   **App Runner:** Automatic scaling, built-in load balancing, pay-per-use
*   **EFS:** Persistent vector store across container restarts
*   **Secrets Manager:** Secure API key management, no hardcoded secrets
*   **CloudFront:** Global CDN for low-latency frontend access

See `docs/aws_deployment.md` for detailed deployment instructions.

