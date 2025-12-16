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

## ⚠️ Backend Improvements

### 3. Architecture & Separation of Concerns
- **Issue**: `backend/agent.py` is a "God Object" (~1000 lines). It handles:
  - Agent Lifecycle
  - State Management (`ReadingState`)
  - Tool Definitions (`search_documents`, `read_document`)
  - RAG Hooks
- **Recommendation**:
  - Move `ReadingState` to `backend/state.py`.
  - Move tool definitions to `backend/tools/` (e.g., `document_tools.py`, `search_tools.py`).
  - Keep `agent.py` focused on wiring dependencies and defining the `entrypoint`.

### 4. RAG Pipeline Efficiency
- **Issue**: `DocumentRetriever.retrieve` assigns a hardcoded score of `1.0` to all results because "Chroma L2 distance isn't ideal".
- **Risk**: The LLM receives context without knowing which documents are actually most relevant.
- **Recommendation**: 
  - Switch ChromaDB metric to `cosine` similarity.
  - Normalize scores properly so the agent can filter out low-quality matches.

### 5. Vector Store Singleton
- **Issue**: `DocumentIndexer` and `DocumentRetriever` both initialize their own `Chroma` client instances.
- **Risk**: Multiple connections to the local SQLite DB file can cause locking issues or performance degradation.
- **Recommendation**: Create a true singleton for the Vector Store in `backend/rag/store.py` and inject it into both classes.

### 6. Configuration Validation
- **Issue**: `config.py` uses `os.getenv` with simple defaults.
- **Recommendation**: Migrate to Pydantic's `BaseSettings`. This ensures the app crashes early (at startup) if critical keys (like `OPENAI_API_KEY`) are missing, rather than failing at runtime.

## 🛠 Frontend Refactoring

### 7. Component Decomposition
- **Issue**: `App.tsx` is large and contains logic for both Connected and Disconnected states, plus "scanline" effects and layout.
- **Recommendation**:
  - Extract `<ConnectedLayout />` and `<DisconnectedLayout />` into separate components.
  - Move the "Terminator" visual effects (scanlines, grid) into a generic `<TerminatorShell />` wrapper.

### 8. Hardcoded Styles (Tailwind)
- **Issue**: Repetitive use of complex class strings like `hud-border rounded-lg bg-terminator-surface/50 backdrop-blur-sm p-4`.
- **Recommendation**: 
  - Define custom generic components: `<Card>`, `<Button variant="hud">`.
  - Or use Tailwind `@layer components` to create utility classes like `.panel-hud`.

### 9. State Logic Duplication
- **Issue**: `useAgent.ts` manually tracks connection states that partially duplicate LiveKit's internal state. `App.tsx` has complex logic to reconcile "token state" vs "room state".
- **Recommendation**: Simplify `useAgent` to rely more directly on LiveKit's `useRoomContext` and `useConnectionState` hooks to avoid "state of truth" conflicts.

## 🧪 Testing & Quality

### 10. Missing Unit Tests
- **Issue**: 
  - Backend `test_agent.py` is an integration smoke test, not a unit test.
  - No frontend unit tests (Vitest/Jest) visible.
- **Recommendation**:
  - Add `pytest` unit tests for `ReadingState` logic (it has complex resume/pause rules).
  - Add component tests for `InputConsole` to verify file upload logic without a real backend.

### 11. Linting & Formatting
- **Issue**: Inconsistent explicit type imports in TypeScript.
- **Recommendation**: Enforce strict ESLint rules for `import type` usage.

## 📝 Documentation

### 12. Design Drift
- **Issue**: `docs/technical_design.md` needs to be cross-referenced with `ReadingState` implementation to ensure the "resume reading" feature matches the original spec.
- **Recommendation**: Update technical design docs to reflect the actual implemented state management strategy.

