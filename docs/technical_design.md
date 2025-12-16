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

### How to Implement read_document Tool (Subtask 5.2)

```python
@function_tool()
async def read_document(
    self,
    context: RunContext,
    document_title: str,
) -> str:
    """Read a document aloud to the user.
    
    Args:
        document_title: The title or partial title of the document to read.
    """
    # 1. Find the document
    retriever = self._get_retriever()
    doc = retriever.find_document_by_title(document_title)
    
    if not doc:
        return f"I couldn't find a document matching '{document_title}'."
    
    # 2. Get chunks for reading
    result = retriever.retrieve_for_reading(
        document_id=doc["document_id"],
        start_chunk=0,
        num_chunks=5,  # Read 5 chunks at a time
    )
    
    # 3. Initialize reading state
    self.start_reading(
        document_id=doc["document_id"],
        document_title=doc["title"],
        total_chunks=len(result.documents),
    )
    
    # 4. Return first chunk for TTS
    # The agent will speak this, and you can implement
    # continuation logic in on_user_turn_completed
    return result.context
```

### How to Handle Interruptions (Subtask 5.3, 5.4)

LiveKit's VAD automatically stops TTS when user speaks. In `on_user_turn_completed`:

```python
async def on_user_turn_completed(self, turn_ctx, new_message):
    user_text = new_message.text_content.lower()
    
    # If reading and user interrupts
    if self.is_reading:
        self.pause_reading()  # Save position
        
        # Check for control commands
        if "stop" in user_text or "that's enough" in user_text:
            self.stop_reading()
            return  # Let LLM respond naturally
        
        if "continue" in user_text or "keep going" in user_text:
            if self.reading_state.can_resume:
                self.resume_reading()
                # Get next chunks and continue reading...
```

### What to Defer (Subtask 5.5)

**Persistence is NOT needed for MVP.** ReadingState is in-memory only. If the agent restarts, reading state is lost — this is acceptable for the interview demo.

### Reference Documentation

- **LiveKit External Data**: https://docs.livekit.io/agents/build/external-data
- **LiveKit VAD/Interruption**: https://docs.livekit.io/agents/build/turns/vad  
- **LiveKit Tools**: https://docs.livekit.io/agents/build/tools
- **Product Requirements**: `docs/product_requirements.md` section 3.1

