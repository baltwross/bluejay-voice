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

### 2.3 Agent State Management

The agent maintains in-memory state for tracking reading sessions. This is implemented in `backend/agent.py`.

#### ReadingState Dataclass

```python
@dataclass
class ReadingState:
    is_reading: bool = False        # Currently reading aloud?
    is_paused: bool = False         # Paused (can resume)?
    document_id: str | None         # Which document?
    document_title: str | None      # Human-readable title
    current_chunk: int = 0          # Current position (0-indexed)
    total_chunks: int = 0           # Total chunks in document
```

#### State Management Methods (on TerminatorAssistant)

| Method | Purpose |
|--------|---------|
| `start_reading(doc_id, title, total, start)` | Begin reading a document |
| `update_reading_position(chunk)` | Update current position |
| `advance_reading_position()` | Move to next chunk (returns False at end) |
| `pause_reading()` | Pause reading (preserves position) |
| `resume_reading()` | Resume from paused position |
| `stop_reading()` | Stop and reset all state |
| `get_reading_status()` | Get human-readable status string |

#### Properties

- `is_reading` — Quick check if in reading mode
- `reading_state` — Access the full `ReadingState` object
- `reading_state.can_resume` — Check if there's a paused session
- `reading_state.progress_percent` — Reading progress as percentage

#### Usage Example (for Task 5 implementation)

```python
# When user says "read the Claude Code article"
doc = retriever.find_document_by_title("Claude Code")
if doc:
    # Get document chunks for reading
    result = retriever.retrieve_for_reading(doc["document_id"], start_chunk=0, num_chunks=5)
    
    # Initialize reading state
    self.start_reading(
        document_id=doc["document_id"],
        document_title=doc["title"],
        total_chunks=result.sources[0].get("total_chunks", len(result.documents)),
    )
    
    # Read first chunk aloud...
    
# When user interrupts
self.pause_reading()

# When user says "continue reading"
if self.reading_state.can_resume:
    self.resume_reading()
    # Get next chunks from where we left off
    result = retriever.retrieve_for_reading(
        self.reading_state.document_id,
        start_chunk=self.reading_state.current_chunk,
        num_chunks=5,
    )
```

### 2.5 The Ingestion Pipeline (`backend/rag/`)
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

### 2.6 The Voice Agent Configuration
*   **VAD (Voice Activity Detection):** `Silero VAD`. Essential for interruptibility.
*   **Turn Detection:** Eager. If the user pauses for > 600ms, the agent takes the turn.
*   **Voice Pipeline:**
    *   **STT:** Deepgram (real-time transcription)
    *   **LLM:** OpenAI `gpt-5-nano-2025-08-07`
    *   **TTS:** ElevenLabs with Arnold-style voice (Voice ID: configurable)

### 2.7 Frontend (React)

The frontend is built with React 18, Vite, and Tailwind CSS with a Terminator HUD aesthetic.

#### Component Architecture

| Component | File | Purpose |
|-----------|------|---------|
| `AgentVisualizer` | `components/AgentVisualizer.tsx` | LiveKit BarVisualizer with T-800 skull icon, state indicators |
| `Transcript` | `components/Transcript.tsx` | Real-time conversation log with auto-scrolling |
| `ControlPanel` | `components/ControlPanel.tsx` | Call controls (start/end, mic toggle) |
| `InputConsole` | `components/InputConsole.tsx` | URL/file upload for RAG ingestion |
| `ErrorBoundary` | `components/ErrorBoundary.tsx` | Error catching and recovery UI |

#### Custom Hooks

| Hook | File | Purpose |
|------|------|---------|
| `useConnection` | `hooks/useConnection.ts` | Token fetching, connection state management |
| `useAgent` | `hooks/useAgent.ts` | LiveKit room connection, agent detection |
| `useDocuments` | `hooks/useDocuments.ts` | Document ingestion, listing |

#### State Management

- **Connection State:** Token fetching → Room connection → Agent presence
- **Agent State:** Tracked via `lk.agent.state` participant attribute (`listening`, `thinking`, `speaking`)
- **Transcript:** Real-time via `useVoiceAssistant` hook (agent + user transcriptions)

#### API Endpoints (Token Server)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/token` | POST | Generate LiveKit access token |
| `/api/ingest` | POST | Ingest URL content (YouTube, web, PDF) |
| `/api/ingest/file` | POST | Upload and ingest file |
| `/api/documents` | GET | List all ingested documents |

#### LiveKit Integration

```tsx
<LiveKitRoom
  token={token}
  serverUrl={serverUrl}
  connect={true}
  audio={true}
>
  <RoomAudioRenderer />      {/* Plays agent audio */}
  <AgentVisualizer />        {/* Shows audio bars + state */}
  <Transcript />             {/* Real-time transcription */}
  <ControlPanel />           {/* User controls */}
</LiveKitRoom>
```

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

---

## 5. Task 5 Implementation Guide: Article Reading Mode

> **For the next agent working on Task 5** — This section provides everything you need.

### What's Already Done (Subtask 5.1 ✅)

The **ReadingState** dataclass and all state management methods are **already implemented** in `backend/agent.py`. You don't need to create them. See section 2.3 above for the full API.

### Key Files to Know

| File | What It Contains |
|------|------------------|
| `backend/agent.py` | `TerminatorAssistant` class with state management methods |
| `backend/rag/retriever.py` | `DocumentRetriever.retrieve_for_reading()` for sequential chunks |
| `backend/config.py` | `ElevenLabsConfig.model_turbo` = `eleven_turbo_v2` for fast TTS |
| `docs/product_requirements.md` | Section 3.1 has Article Reading Mode requirements |

### read_document Tool Implementation (Subtask 5.2) — IMPLEMENTED ✅

The `read_document` tool is fully implemented in `backend/agent.py`. Key features:
- Finds documents by partial title match using `find_document_by_title()`
- Retrieves chunks sequentially using `retrieve_for_reading()`
- Initializes reading state with `start_reading()`
- Returns first batch of chunks (3 at a time) for TTS narration
- Provides progress information and user instructions

```python
@function_tool()
async def read_document(self, context: RunContext, document_title: str) -> str:
    """Read a document aloud to the user."""
    retriever = self._get_retriever()
    doc = retriever.find_document_by_title(document_title)
    
    if not doc:
        available_docs = retriever.list_documents()
        # Return helpful message with available documents
        return f"I couldn't find '{document_title}'. Available: {titles}"
    
    result = retriever.retrieve_for_reading(
        document_id=doc["document_id"],
        start_chunk=0,
        num_chunks=3,  # 3 chunks at a time
    )
    
    self.start_reading(
        document_id=doc["document_id"],
        document_title=doc["title"],
        total_chunks=doc.get("total_chunks", len(result.documents)),
    )
    
    return f"Beginning to read '{doc['title']}':\n\n{result.context}"
```

### How to Handle Interruptions (Subtask 5.3, 5.4) — IMPLEMENTED ✅

LiveKit's VAD automatically stops TTS when user speaks. The `on_user_turn_completed` hook now handles reading mode interruptions:

```python
async def on_user_turn_completed(self, turn_ctx, new_message):
    user_text = new_message.text_content.lower()
    
    # If reading and user interrupts
    if self._reading_state.is_reading:
        self.pause_reading()  # Save position
        # Inject context so LLM knows what's happening
        turn_ctx.add_message(
            role="assistant",
            content=f"[Reading paused. {self.get_reading_status()}. "
                    f"Respond to the user. Use continue_reading or end_reading tools.]"
        )
        return
    
    # If paused and user wants to continue
    if self._reading_state.can_resume:
        continue_keywords = ["continue", "keep going", "resume", ...]
        if any(kw in user_text for kw in continue_keywords):
            turn_ctx.add_message(
                role="assistant",
                content="[User wants to continue reading. Use continue_reading tool.]"
            )
```

### Available Reading Tools

| Tool | Purpose |
|------|---------|
| `read_document(document_title)` | Start reading a document by title |
| `continue_reading()` | Continue from paused position |
| `end_reading()` | Stop reading and reset state |
| `list_available_documents()` | List all documents in knowledge base |

### Reading Session Persistence (Subtask 5.5) — IMPLEMENTED ✅

Reading state is now persisted to a JSON file (`backend/.reading_state.json`), enabling resume across agent restarts:

**Persistence Features:**
- State is saved automatically on: `start_reading()`, `update_reading_position()`, `pause_reading()`, `resume_reading()`
- State file is cleaned up when `stop_reading()` is called
- Sessions older than 24 hours are automatically expired and cleaned up
- On agent startup, if a resumable session exists, the agent offers to continue

**Storage Format:**
```json
{
  "is_reading": false,
  "is_paused": true,
  "document_id": "abc123",
  "document_title": "Claude Code Article",
  "current_chunk": 5,
  "total_chunks": 15,
  "last_updated": "2025-12-16T06:30:00.000000"
}
```

**New Tool: `check_reading_session()`**
- Returns status of any resumable reading session
- Useful when user asks "was I reading something?" or "can we continue?"

### Reference Documentation

- **LiveKit External Data**: https://docs.livekit.io/agents/build/external-data
- **LiveKit VAD/Interruption**: https://docs.livekit.io/agents/build/turns/vad  
- **LiveKit Tools**: https://docs.livekit.io/agents/build/tools
- **Product Requirements**: `docs/product_requirements.md` section 3.1

