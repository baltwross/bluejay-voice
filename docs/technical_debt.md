# Technical Debt & Codebase Improvements

## 🚨 Critical Priority

### 1. Frontend Type Safety (`Transcript.tsx`)
- **Issue**: The current `Transcript.tsx` uses `as any` casts to bypass type checking for LiveKit transcript segments (e.g., `segment.id`, `segment.firstReceivedTime`).
- **Risk**: This defeats TypeScript's safety guarantees and may lead to runtime errors if the LiveKit SDK API changes.
- **Recommendation**: 
  - Extend the LiveKit types or create a robust interface that matches the actual runtime shape of `segment`.
  - Remove all `as any` casts.

### 2. LiveKit Type Definitions
- **Issue**: `backend/agent.py` ignores import errors for `MultilingualModel` (`# type: ignore[import-not-found]`).
- **Risk**: Static analysis tools cannot verify this dependency, potential for runtime import failures.
- **Recommendation**: Verify package installation or create/install proper type stubs.

### 3. Blocking I/O in Async Paths (Backend)
- **Issue**: 
  - `backend/agent.py`: `ReadingState.save()` uses `json.dump` (synchronous file I/O) inside async methods.
  - `backend/token_server.py`: `ingest_file` reads entire files into memory (`await file.read()`) and writes to temp files synchronously.
  - `backend/rag/indexer.py`: PDF/DOCX processing performs heavy synchronous I/O and CPU work on the main thread.
- **Risk**: Blocking the event loop in an async application (FastAPI/LiveKit) causes high latency and can make the server unresponsive to health checks or other requests.
- **Recommendation**: 
  - Use `aiofiles` for asynchronous file I/O.
  - Offload heavy CPU/IO tasks (like PDF parsing or Chroma ingestion) to a thread pool using `run_in_executor`.

### 4. Dependency Injection & Global State (Backend)
- **Issue**: `backend/token_server.py` uses global variables (`_indexer`, `_retriever`) with lazy loading functions. `agent.py` also instantiates its own retriever.
- **Risk**: Makes testing difficult (cannot mock dependencies easily) and leads to "fight over resources" (e.g., multiple ChromaDB connections).
- **Recommendation**: 
  - Implement a proper Dependency Injection container or factory pattern.
  - Create a single `RAGService` that manages the Chroma client and is passed to both the API and the Agent.

## ⚠️ Backend Improvements

### 5. Architecture & Separation of Concerns (`agent.py`)
- **Issue**: `backend/agent.py` is a "God Object" (~1000 lines). It handles:
  - Agent Lifecycle
  - State Management (`ReadingState` dataclass defined inside)
  - Tool Definitions (`search_documents`, `read_document`)
  - RAG Hooks
  - Interruption Logic
- **Recommendation**:
  - Extract `ReadingState` to `backend/state/reading_manager.py`.
  - Move tool definitions to `backend/tools/` (e.g., `document_tools.py`, `search_tools.py`).
  - Move RAG hooks/interruption logic to a `ConversationManager` class.
  - Keep `agent.py` focused purely on wiring dependencies and defining the `entrypoint`.

### 6. RAG Pipeline Efficiency
- **Issue**: `DocumentRetriever.retrieve` assigns a hardcoded score of `1.0` to all results because "Chroma L2 distance isn't ideal".
- **Risk**: The LLM receives context without knowing which documents are actually most relevant, potentially hallucinating on weak matches.
- **Recommendation**: 
  - Switch ChromaDB metric to `cosine` similarity.
  - Implement a proper scoring/re-ranking step (e.g., using a Cross-Encoder) if high precision is needed.
  - Normalize scores so the agent can filter out low-quality matches.

### 7. Vector Store Singleton
- **Issue**: `DocumentIndexer` and `DocumentRetriever` both initialize their own `Chroma` client instances pointing to the same SQLite file.
- **Risk**: SQLite locking issues (database is locked) during concurrent reads/writes. High memory usage from duplicate embedding models.
- **Recommendation**: Create a true singleton for the Vector Store in `backend/rag/store.py` and inject it into both classes.

### 8. Configuration Validation
- **Issue**: `config.py` uses `os.getenv` with simple defaults.
- **Recommendation**: Migrate to Pydantic's `BaseSettings`. This ensures the app crashes early (at startup) if critical keys (like `OPENAI_API_KEY`) are missing, rather than failing at runtime.

### 9. Hardcoded Prompts & Strings
- **Issue**: `backend/agent.py` contains hardcoded strings for RAG context injection, greetings, and system prompts.
- **Recommendation**: Move all prompt templates and static messages to `backend/prompts.py` or a YAML/JSON resource file to separate content from logic.

## 🛠 Frontend Refactoring

### 10. Component Decomposition (`App.tsx`)
- **Issue**: `App.tsx` is large and contains logic for both Connected and Disconnected states, plus "scanline" effects, header, footer, and layout.
- **Recommendation**:
  - Extract `<ConnectedLayout />` and `<DisconnectedLayout />` into separate components.
  - Move the "Terminator" visual effects (scanlines, grid) into a generic `<TerminatorShell />` wrapper.
  - Extract `<Header />` and `<Footer />` to their own files.

### 11. Hardcoded Styles (Tailwind)
- **Issue**: Repetitive use of complex class strings like `hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4`.
- **Recommendation**: 
  - Define custom generic components: `<Card>`, `<Button variant="hud">`, `<Badge>`.
  - Use Tailwind `@layer components` or a theme configuration to standardize colors and border styles.

### 12. State Logic Duplication & Complexity
- **Issue**: 
  - `useAgent.ts` manually tracks connection states (`connectionState`, `isAgentConnected`) that partially duplicate LiveKit's internal state.
  - `App.tsx` has complex logic to reconcile "token state" vs "room state" (`effectiveConnectionState`).
  - `Transcript.tsx` performs complex sorting and merging of arrays inside `useMemo`.
- **Recommendation**: 
  - Simplify `useAgent` to rely more directly on LiveKit's `useRoomContext` and `useConnectionState` hooks.
  - Extract chat history logic to a custom `useChatHistory` hook.
  - Use a React Context (`AgentContext`) to share state between `ControlPanel`, `Transcript`, and `Visualizer` to avoid prop drilling.

### 13. Hardcoded Type Casts in API Calls
- **Issue**: `backend/token_server.py` defines `IngestUrlRequest` but hardcodes `type: str`.
- **Recommendation**: Use Python `Enum` (e.g., `class SourceType(str, Enum): YOUTUBE = 'youtube' ...`) for API request models to ensure type safety and automatic validation documentation (Swagger UI).

## 🧪 Testing & Quality

### 14. Missing Unit Tests
- **Issue**: 
  - Backend `test_agent.py` is an integration smoke test, not a unit test.
  - No frontend unit tests (Vitest/Jest) visible.
- **Recommendation**:
  - Add `pytest` unit tests for `ReadingState` logic (it has complex resume/pause rules).
  - Add component tests for `InputConsole` to verify file upload logic without a real backend.
  - Test RAG retriever logic with mocked ChromaDB.

### 15. Linting & Formatting
- **Issue**: Inconsistent explicit type imports in TypeScript.
- **Recommendation**: Enforce strict ESLint rules for `import type` usage.

## 📝 Documentation

### 16. Design Drift
- **Issue**: `docs/technical_design.md` needs to be cross-referenced with `ReadingState` implementation to ensure the "resume reading" feature matches the original spec.
- **Recommendation**: Update technical design docs to reflect the actual implemented state management strategy.

### 17. Security & Input Validation
- **Issue**: File upload endpoint (`ingest_file`) in `token_server.py` trusts the file extension/MIME type without deep inspection.
- **Risk**: Malicious file upload.
- **Recommendation**: Use a library like `python-magic` to verify file types before processing. Implement file size limits.
