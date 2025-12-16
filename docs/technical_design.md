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
            Tools -->|Query| RAG[LlamaIndex]
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
*   **Entrypoint:** `agent.py` - defines the `VoicePipelineAgent`.
*   **State Management:**
    *   The agent needs to know if it's currently "discussing a document" or "chatting generally".
    *   We will use the LLM's context window (chat history) to maintain this state implicitly.

### 2.2 The Ingestion Pipeline (`backend/rag/`)
This is a critical subsystem. It decouples *getting content* from *using content*.

1.  **Input:** Raw File or URL.
2.  **Loader:**
    *   `PDFReader` (via `pymupdf`) for PDFs.
    *   `YoutubeTranscriptReader` for YouTube URLs.
    *   `Trafilatura` for generic Web URLs.
    *   `DocxReader` (via `python-docx`) for Word documents.
3.  **Transformation:**
    *   Clean text (remove headers/footers/timestamps).
    *   Chunking: SentenceSplitter (chunk_size=1024, overlap=20).
4.  **Embedding:** `OpenAIEmbedding` (text-embedding-3-small).
5.  **Storage:** `ChromaVectorStore` (persisted to `./chroma_db`).

### 2.3 The Voice Agent Configuration
*   **VAD (Voice Activity Detection):** `Silero VAD`. Essential for interruptibility.
*   **Turn Detection:** Eager. If the user pauses for > 600ms, the agent takes the turn.

### 2.4 Frontend (React)
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
| `embedding` | Vector[1536] | The semantic vector |
| `text` | String | The actual content chunk |
| `metadata.source_type` | String | "pdf", "youtube", "web" |
| `metadata.source_url` | String | Original link or filename |
| `metadata.title` | String | Title of the content |
| `metadata.timestamp` | Float | (Optional) For video transcripts |

## 4. Security & Deployment (Take-Home Scope)
*   **API Keys:** Stored in `.env` (local) / AWS Secrets Manager (prod).
*   **Local Run:** `python agent.py dev` connects to LiveKit Cloud from localhost.
*   **Production Deployment (AWS):**
    *   **Strategy:** Dockerized application deployed to **AWS App Runner** or **ECS Fargate**.
    *   **Benefit:** Provides a publicly accessible URL (no local install required for users).
    *   **Persistence:** Local ChromaDB is ephemeral in containers. For this MVP, data resets on deployment (acceptable). Ideally would use persistent volume or S3.

