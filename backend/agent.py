"""
Bluejay Terminator - Main Voice Agent

This is the main entry point for the T-800 Terminator voice agent.
It sets up the voice pipeline with:
- Deepgram STT (Speech-to-Text)
- OpenAI LLM (gpt-5-nano-2025-08-07)
- ElevenLabs TTS (Arnold-style voice)
- Silero VAD (Voice Activity Detection)

Architecture:
- System prompt defines personality (prompts.py)
- RAG context injected via on_user_turn_completed hook
- Tool calls handled by LLM (search_ai_news, etc.)

Performance Optimizations (Task 9):
- Metrics collection via UsageCollector
- Preemptive speech generation for reduced latency
- Voice switching between Terminator/Standard modes
"""
import os
import logging
import time
import certifi
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime, timedelta

# Fix SSL certificate verification on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    RunContext,
    function_tool,
    ToolError,
    llm as lk_llm,
    metrics,
    MetricsCollectedEvent,
)
from livekit.plugins import deepgram, openai, silero, elevenlabs, noise_cancellation

from prompts import get_system_prompt, format_rag_context, should_trigger_rag
from config import get_config, get_random_greeting, get_random_pre_search_phrase
from rag.retriever import DocumentRetriever
from rag.indexer import DocumentIndexer

# Tavily for AI news search
try:
    from tavily import TavilyClient  # type: ignore
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None  # type: ignore


# =============================================================================
# NEWS CACHE
# =============================================================================
# Simple in-memory cache for news queries to reduce API calls and improve latency

class NewsCache:
    """Simple in-memory cache for news search results."""
    
    def __init__(self, ttl_minutes: int = 5):
        """
        Initialize the news cache.
        
        Args:
            ttl_minutes: Time-to-live for cache entries in minutes.
        """
        self._cache: dict[str, tuple[datetime, list[dict]]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        # Store recent articles for read_news_article tool (title -> article info)
        self._recent_articles: dict[str, dict] = {}
        self._articles_timestamp: datetime | None = None
    
    def get(self, cache_key: str) -> list[dict] | None:
        """
        Get cached results if still valid.
        
        Args:
            cache_key: The cache key (normalized query string).
            
        Returns:
            Cached results if valid, None if expired or not found.
        """
        if cache_key not in self._cache:
            return None
        
        timestamp, results = self._cache[cache_key]
        if datetime.now() - timestamp > self._ttl:
            # Expired - remove and return None
            del self._cache[cache_key]
            return None
        
        return results
    
    def set(self, cache_key: str, results: list[dict]) -> None:
        """
        Store results in cache.
        
        Args:
            cache_key: The cache key (normalized query string).
            results: The search results to cache.
        """
        self._cache[cache_key] = (datetime.now(), results)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def store_articles(self, articles: list[dict]) -> None:
        """
        Store recent articles for later retrieval by title.
        
        Args:
            articles: List of article dicts with 'title', 'url', 'content'.
        """
        self._recent_articles.clear()
        for article in articles:
            title = article.get("title", "").lower().strip()
            if title:
                self._recent_articles[title] = article
        self._articles_timestamp = datetime.now()
        logger.debug(f"Stored {len(self._recent_articles)} articles for later retrieval")
    
    def find_article_by_title(self, title_query: str) -> dict | None:
        """
        Find a recently searched article by partial title match.
        
        Args:
            title_query: Partial or full title to search for.
            
        Returns:
            Article dict if found, None otherwise.
        """
        # Check if articles are expired
        if self._articles_timestamp:
            if datetime.now() - self._articles_timestamp > self._ttl:
                logger.debug("Recent articles expired, clearing")
                self._recent_articles.clear()
                self._articles_timestamp = None
                return None
        
        query_lower = title_query.lower().strip()
        
        # First try exact match
        if query_lower in self._recent_articles:
            return self._recent_articles[query_lower]
        
        # Then try partial match (query is substring of title)
        for stored_title, article in self._recent_articles.items():
            if query_lower in stored_title or stored_title in query_lower:
                return article
        
        # Try matching key words
        query_words = set(query_lower.split())
        best_match = None
        best_score = 0
        
        for stored_title, article in self._recent_articles.items():
            title_words = set(stored_title.split())
            # Count matching words
            matches = len(query_words & title_words)
            if matches > best_score and matches >= 2:  # At least 2 words must match
                best_score = matches
                best_match = article
        
        return best_match
    
    def list_recent_articles(self) -> list[str]:
        """
        Get titles of recently searched articles.
        
        Returns:
            List of article titles.
        """
        if self._articles_timestamp:
            if datetime.now() - self._articles_timestamp > self._ttl:
                return []
        
        return [article.get("title", "Unknown") for article in self._recent_articles.values()]


# Global news cache instance (shared across agent instances)
_news_cache = NewsCache(ttl_minutes=5)


# =============================================================================
# VOICE MODE CONFIGURATION
# =============================================================================

class VoiceMode(Enum):
    """Voice modes for the agent."""
    TERMINATOR = "terminator"  # T-800 Terminator voice
    INSPIRE = "inspire"        # Inspire voice - motivational, energetic
    FATE = "fate"              # Fate voice - mysterious, dramatic


# Voice configurations for different modes
# TERMINATOR uses the custom voice that works
# INSPIRE and FATE use ElevenLabs premade voices (the custom ones aren't accessible)
VOICE_CONFIGS = {
    VoiceMode.TERMINATOR: {
        # Custom T-800 Terminator voice (verified working)
        "voice_id": "8DGMp3sPQNZOuCfSIxxE",
        "model": "eleven_multilingual_v2",
        "stability": 0.26,
        "similarity_boost": 0.80,
        "speed": 1.14,
    },
    VoiceMode.INSPIRE: {
        # "Josh" - deep, American, motivational (premade - guaranteed to work)
        # Original custom voice 89eOzfoGxxCOxCzNy9l5 is not accessible
        "voice_id": "TxGEqnHWrfWFTfGW9XjX",
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75,
    },
    VoiceMode.FATE: {
        # "Adam" - deep, authoritative, dramatic (premade - guaranteed to work)
        # Original custom voice XfNU2rGpBa01ckF309OY is not accessible
        "voice_id": "pNInz6obpgDQGcFmaJgB",
        "model": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75,
    },
}

# Fallback voice ID if primary voice fails (Rachel - reliable premade voice)
FALLBACK_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

import json
from pathlib import Path
from datetime import datetime, timedelta

# Path for persisting reading state across sessions
READING_STATE_FILE = Path(__file__).parent / ".reading_state.json"
# Sessions older than this are cleaned up
SESSION_EXPIRY_HOURS = 24


@dataclass
class ReadingState:
    """
    Tracks the state of document reading mode.
    
    This allows the agent to:
    - Know if it's currently reading a document aloud
    - Track position for pause/resume functionality
    - Remember which document is being read
    - Persist state across agent restarts
    """
    is_reading: bool = False
    is_paused: bool = False
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    current_chunk: int = 0
    total_chunks: int = 0
    last_updated: Optional[str] = None  # ISO timestamp for cleanup
    
    def reset(self) -> None:
        """Reset all reading state to defaults."""
        self.is_reading = False
        self.is_paused = False
        self.document_id = None
        self.document_title = None
        self.current_chunk = 0
        self.total_chunks = 0
        self.last_updated = None
    
    @property
    def progress_percent(self) -> float:
        """Get reading progress as a percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.current_chunk / self.total_chunks) * 100
    
    @property
    def can_resume(self) -> bool:
        """Check if there's a paused reading session that can be resumed."""
        return self.is_paused and self.document_id is not None
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired (older than SESSION_EXPIRY_HOURS)."""
        if not self.last_updated:
            return True
        try:
            updated_time = datetime.fromisoformat(self.last_updated)
            expiry_time = datetime.now() - timedelta(hours=SESSION_EXPIRY_HOURS)
            return updated_time < expiry_time
        except (ValueError, TypeError):
            return True
    
    def to_status_string(self) -> str:
        """Get a human-readable status string."""
        if not self.is_reading and not self.is_paused:
            return "Not currently reading any document."
        
        status = "Paused" if self.is_paused else "Reading"
        title = self.document_title or "Unknown document"
        progress = f"{self.current_chunk + 1}/{self.total_chunks}"
        percent = f"{self.progress_percent:.0f}%"
        
        return f"{status}: '{title}' - Section {progress} ({percent} complete)"
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for JSON serialization."""
        return {
            "is_reading": self.is_reading,
            "is_paused": self.is_paused,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "last_updated": self.last_updated or datetime.now().isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ReadingState":
        """Create ReadingState from dictionary."""
        return cls(
            is_reading=data.get("is_reading", False),
            is_paused=data.get("is_paused", False),
            document_id=data.get("document_id"),
            document_title=data.get("document_title"),
            current_chunk=data.get("current_chunk", 0),
            total_chunks=data.get("total_chunks", 0),
            last_updated=data.get("last_updated"),
        )

    @staticmethod
    def _resolve_state_file_path(file_path: Path) -> Path:
        """
        Resolve the actual file path to use for persistence.

        Expected: `file_path` is a JSON file (default: `.reading_state.json`).
        Reality (dev): A directory with that name can exist (e.g., created accidentally),
        which would raise `IsADirectoryError` when calling `open()`.

        If a directory is detected, we store the state inside it as `state.json`.
        This avoids crashing the agent and allows the app to recover gracefully.
        """
        if file_path.exists() and file_path.is_dir():
            return file_path / "state.json"
        return file_path
    
    def save(self, file_path: Path = READING_STATE_FILE) -> bool:
        """
        Persist reading state to file.
        
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self.last_updated = datetime.now().isoformat()
            resolved_file_path = self._resolve_state_file_path(file_path)
            resolved_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"Reading state saved to {resolved_file_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save reading state: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: Path = READING_STATE_FILE) -> Optional["ReadingState"]:
        """
        Load reading state from file.
        
        Returns:
            ReadingState if loaded successfully and not expired, None otherwise.
        """
        try:
            resolved_file_path = cls._resolve_state_file_path(file_path)

            if not resolved_file_path.exists():
                logger.debug("No saved reading state found")
                return None

            if not resolved_file_path.is_file():
                logger.warning(f"Reading state path is not a file: {resolved_file_path}")
                return None
            
            with open(resolved_file_path, "r") as f:
                data = json.load(f)
            
            state = cls.from_dict(data)
            
            # Check if session has expired
            if state.is_expired:
                logger.info("Previous reading session has expired, cleaning up")
                cls.cleanup(file_path)
                return None
            
            # Only return if there's something to resume
            if state.document_id and (state.is_paused or state.is_reading):
                logger.info(f"Loaded saved reading session: {state.document_title}")
                # Mark as paused since we're loading from a restart
                state.is_reading = False
                state.is_paused = True
                return state
            
            return None
            
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to load reading state: {e}")
            cls.cleanup(file_path)
            return None
        except OSError as e:
            # Includes IsADirectoryError, PermissionError, etc. We don't want the agent to crash.
            logger.warning(f"Failed to read reading state from disk: {e}")
            return None
    
    @classmethod
    def cleanup(cls, file_path: Path = READING_STATE_FILE) -> bool:
        """
        Remove the saved reading state file.
        
        Returns:
            True if cleaned up successfully, False otherwise.
        """
        try:
            resolved_file_path = cls._resolve_state_file_path(file_path)

            if resolved_file_path.exists() and resolved_file_path.is_file():
                resolved_file_path.unlink()
                logger.debug(f"Cleaned up reading state file: {resolved_file_path}")

            # If the original path was a directory and is now empty, remove it to self-heal.
            if file_path.exists() and file_path.is_dir():
                try:
                    file_path.rmdir()
                    logger.debug(f"Removed empty reading state directory: {file_path}")
                except OSError:
                    # Directory not empty or cannot be removed; ignore.
                    pass
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup reading state: {e}")
            return False


class TerminatorAssistant(Agent):
    """
    T-800 Terminator Voice Assistant.
    
    This agent uses the Terminator personality defined in prompts.py
    and can be extended with RAG capabilities and tool calls.
    
    Tools:
        - search_documents: Search the user's shared documents (PDFs, articles, etc.)
        - ingest_url: Process and index content from URLs (YouTube, web, PDF)
        - read_document: Read a document aloud with interruptibility
        - continue_reading: Continue from paused position
        - end_reading: Stop reading and reset state
        - list_available_documents: List all documents in knowledge base
        - check_reading_session: Check for resumable reading sessions
        - switch_voice: Switch between Terminator, Inspire, and Fate voice modes
        - get_voice_mode: Get the current voice mode
        - search_ai_news: Search for latest AI news (Tavily integration)
        - read_news_article: Read full content of a recently found news article
    
    State:
        - reading_state: Tracks document reading mode (position, paused, etc.)
        - voice_mode: Current voice mode (Terminator, Inspire, or Fate)
    """
    
    def __init__(self, user_name: str | None = None) -> None:
        """
        Initialize the Terminator assistant.
        
        Args:
            user_name: Optional user's first name for personalized interactions.
        """
        super().__init__(
            instructions=get_system_prompt(),
        )
        self.user_name = user_name
        self._retriever = None  # Will be initialized lazily for RAG
        self._indexer = None  # Will be initialized lazily for URL ingestion
        self._tavily_client = None  # Will be initialized lazily for news search
        
        # Active document tracking - filters RAG to the document being discussed
        self._active_document_id: Optional[str] = None
        self._active_document_title: Optional[str] = None
        
        # Voice mode state
        self._voice_mode = VoiceMode.TERMINATOR
        
        # TTS instance will be set by the session after connection
        self._tts: elevenlabs.TTS | None = None
        
        # Try to load any persisted reading state from previous session
        saved_state = ReadingState.load()
        if saved_state:
            self._reading_state = saved_state
            logger.info(f"Restored reading session: {saved_state.document_title}")
        else:
            self._reading_state = ReadingState()
        
        self._has_pending_resume = saved_state is not None  # Track if we should offer resume
    
    # =========================================================================
    # STATE MANAGEMENT - Reading Mode
    # =========================================================================
    
    @property
    def is_reading(self) -> bool:
        """Check if the agent is currently in reading mode."""
        return self._reading_state.is_reading
    
    @property
    def reading_state(self) -> ReadingState:
        """Get the current reading state (read-only access)."""
        return self._reading_state
    
    def start_reading(
        self,
        document_id: str,
        document_title: str,
        total_chunks: int,
        start_chunk: int = 0,
    ) -> None:
        """
        Start reading a document aloud.
        
        Args:
            document_id: The ID of the document to read.
            document_title: Human-readable title of the document.
            total_chunks: Total number of chunks in the document.
            start_chunk: Starting chunk index (default 0).
        """
        self._reading_state.is_reading = True
        self._reading_state.is_paused = False
        self._reading_state.document_id = document_id
        self._reading_state.document_title = document_title
        self._reading_state.current_chunk = start_chunk
        self._reading_state.total_chunks = total_chunks
        self._has_pending_resume = False  # Clear any pending resume offer
        
        # Persist state for resume across restarts
        self._reading_state.save()
        
        logger.info(
            f"Started reading: '{document_title}' "
            f"(chunks {start_chunk + 1}/{total_chunks})"
        )
    
    def update_reading_position(self, chunk_index: int) -> None:
        """
        Update the current reading position.
        
        Args:
            chunk_index: The new chunk index (0-based).
        """
        if not self._reading_state.is_reading:
            logger.warning("Attempted to update reading position while not reading")
            return
        
        self._reading_state.current_chunk = chunk_index
        
        # Persist updated position
        self._reading_state.save()
        
        logger.debug(
            f"Reading position updated: chunk {chunk_index + 1}/"
            f"{self._reading_state.total_chunks}"
        )
    
    def advance_reading_position(self) -> bool:
        """
        Advance to the next chunk.
        
        Returns:
            True if advanced successfully, False if at end of document.
        """
        if not self._reading_state.is_reading:
            return False
        
        next_chunk = self._reading_state.current_chunk + 1
        if next_chunk >= self._reading_state.total_chunks:
            logger.info("Reached end of document")
            return False
        
        self._reading_state.current_chunk = next_chunk
        self._reading_state.save()  # Persist position
        return True
    
    def pause_reading(self) -> None:
        """Pause reading (preserves position for resume)."""
        if self._reading_state.is_reading:
            self._reading_state.is_reading = False
            self._reading_state.is_paused = True
            
            # Persist paused state for resume across restarts
            self._reading_state.save()
            
            logger.info(
                f"Reading paused at chunk {self._reading_state.current_chunk + 1}/"
                f"{self._reading_state.total_chunks}"
            )
    
    def resume_reading(self) -> bool:
        """
        Resume reading from paused position.
        
        Returns:
            True if resumed successfully, False if nothing to resume.
        """
        if not self._reading_state.can_resume:
            logger.warning("No paused reading session to resume")
            return False
        
        self._reading_state.is_reading = True
        self._reading_state.is_paused = False
        self._has_pending_resume = False  # Clear pending resume flag
        
        # Persist resumed state
        self._reading_state.save()
        
        logger.info(
            f"Resumed reading '{self._reading_state.document_title}' "
            f"at chunk {self._reading_state.current_chunk + 1}"
        )
        return True
    
    def stop_reading(self) -> None:
        """Stop reading and reset all state."""
        if self._reading_state.is_reading or self._reading_state.is_paused:
            doc_title = self._reading_state.document_title
            self._reading_state.reset()
            self._has_pending_resume = False
            
            # Cleanup persisted state file
            ReadingState.cleanup()
            
            logger.info(f"Stopped reading: '{doc_title}'")
    
    def get_reading_status(self) -> str:
        """Get a human-readable reading status string."""
        return self._reading_state.to_status_string()
    
    def _get_retriever(self) -> DocumentRetriever:
        """
        Get or initialize the document retriever.
        
        Returns:
            The DocumentRetriever instance.
            
        Raises:
            ToolError: If retriever cannot be initialized.
        """
        if self._retriever is None:
            try:
                self._retriever = DocumentRetriever()
                logger.info("RAG retriever initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG retriever: {e}")
                raise ToolError("Document retrieval system is currently unavailable.")
        return self._retriever

    def _set_active_document(self, document_id: str, document_title: str) -> None:
        """Set the active document used for automatic RAG filtering."""
        self._active_document_id = document_id
        self._active_document_title = document_title
        logger.info(f"Active document set to: {document_title} ({document_id})")

    def _get_last_user_message_text(self, turn_ctx: lk_llm.ChatContext) -> str | None:
        """Get the most recent *previous* user message text from the chat context."""
        items = getattr(turn_ctx, "items", None)
        if not isinstance(items, list):
            return None
        for item in reversed(items):
            if isinstance(item, lk_llm.ChatMessage) and item.role == "user":
                text = item.text_content
                if text:
                    return text
        return None

    def _build_rag_query(self, turn_ctx: lk_llm.ChatContext, user_text: str) -> str:
        """
        Build a query string for retrieval.
        
        Live speech often arrives as short follow-ups ("it's figure 37"). When the
        newest message looks like a continuation, we concatenate it with the
        previous user message to improve retrieval recall.
        """
        user_text_clean = (user_text or "").strip()
        if not user_text_clean:
            return user_text_clean

        lower = user_text_clean.lower()
        has_explicit_question = ("?" in user_text_clean) or any(
            kw in lower for kw in ("what", "why", "how", "when", "where", "which")
        )
        looks_like_continuation = (
            len(user_text_clean) < 40
            or lower.startswith(
                (
                    "it's",
                    "it is",
                    "thats",
                    "that's",
                    "figure",
                    "fig",
                    "also",
                    "and",
                    "no",
                    "not",
                    "wait",
                    "hold on",
                    "sorry",
                )
            )
            or "it's not" in lower
            or "figure" in lower
            or "fig." in lower
            # Clarifications like "I shared a document called X" usually refer to the
            # user's previous question; combine to preserve intent.
            or (
                not has_explicit_question
                and ("i shared" in lower or "i uploaded" in lower)
                and ("called" in lower or "titled" in lower or "named" in lower)
            )
        )
        if not looks_like_continuation:
            return user_text_clean

        prev = self._get_last_user_message_text(turn_ctx)
        if not prev:
            return user_text_clean

        prev_clean = prev.strip()
        if not prev_clean:
            return user_text_clean

        return f"{prev_clean} {user_text_clean}"

    def _maybe_update_active_document_from_user_text(self, user_text: str) -> None:
        """
        Heuristically switch active document when the user names a doc/paper.
        
        This prevents "document bleed" (answering from a previously active doc)
        when the user says things like: "I shared a paper called X".
        """
        text = (user_text or "").strip()
        if not text:
            return

        lower = text.lower()
        doc_cues = (
            "document" in lower
            or "paper" in lower
            or "pdf" in lower
            or "article" in lower
            or "shared" in lower
            or "uploaded" in lower
            or "called" in lower
            or "titled" in lower
            or ("\"" in text or "'" in text)
        )
        if not doc_cues:
            return

        try:
            retriever = self._get_retriever()
            candidate_titles: list[str] = []

            # 1) Quoted titles: "..."
            import re
            candidate_titles.extend(re.findall(r"\"([^\"]{3,})\"", text))
            candidate_titles.extend(re.findall(r"'([^']{3,})'", text))

            # 2) "called X" / "titled X" / "named X"
            match = re.search(r"\b(?:called|titled|named)\s+(.+?)(?:[.?!]|$)", lower)
            if match:
                candidate_titles.append(match.group(1).strip())

            # 3) Fallback: try to match against known titles using token overlap
            documents = retriever.list_documents()

            def _normalize_tokens(s: str) -> set[str]:
                tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
                stop = {
                    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
                    "is", "are", "was", "were", "be", "been", "this", "that", "it", "i",
                    "you", "we", "they", "he", "she", "my", "your", "our",
                    "document", "paper", "pdf", "article", "file", "shared", "uploaded",
                    "called", "titled", "named",
                }
                return {t for t in tokens if t not in stop and len(t) >= 3}

            # Try explicit candidates first
            for cand in candidate_titles:
                doc = retriever.find_document_by_title(cand)
                if doc:
                    self._set_active_document(doc["document_id"], doc["title"])
                    return

            # Token overlap against all titles
            text_tokens = _normalize_tokens(text)
            best_doc: dict | None = None
            best_score = 0.0

            for doc in documents:
                title = str(doc.get("title", "") or "")
                title_lower = title.lower()

                if title_lower and title_lower in lower:
                    best_doc = doc
                    best_score = 1.0
                    break

                title_tokens = _normalize_tokens(title)
                if not title_tokens or not text_tokens:
                    continue

                overlap = len(title_tokens & text_tokens)
                score = overlap / max(len(title_tokens), 1)
                if overlap >= 3 and score > best_score:
                    best_score = score
                    best_doc = doc

            if best_doc and best_score >= 0.6:
                self._set_active_document(
                    str(best_doc.get("document_id", "")),
                    str(best_doc.get("title", "")),
                )
                return

            # If user says they uploaded/shared something but we couldn't extract a title,
            # default to the most recently ingested document (newest-first list).
            if ("shared" in lower or "uploaded" in lower) and documents:
                newest = documents[0]
                doc_id = str(newest.get("document_id", ""))
                doc_title = str(newest.get("title", ""))
                if doc_id and doc_title:
                    self._set_active_document(doc_id, doc_title)

        except Exception as e:
            logger.warning(f"Failed to update active document from user text: {e}")
    
    def _get_indexer(self) -> DocumentIndexer:
        """
        Get or initialize the document indexer for URL ingestion.
        
        Returns:
            The DocumentIndexer instance.
            
        Raises:
            ToolError: If indexer cannot be initialized.
        """
        if self._indexer is None:
            try:
                self._indexer = DocumentIndexer()
                logger.info("Document indexer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize document indexer: {e}")
                raise ToolError("Document ingestion system is currently unavailable.")
        return self._indexer
    
    def _get_tavily_client(self):
        """
        Get or initialize the Tavily client for news search.
        
        Returns:
            The TavilyClient instance.
            
        Raises:
            ToolError: If Tavily is not available or configured.
        """
        if not TAVILY_AVAILABLE:
            logger.error("Tavily package not installed")
            raise ToolError(
                "News search is currently unavailable. "
                "The Tavily package is not installed."
            )
        
        if self._tavily_client is None:
            config = get_config()
            api_key = config.tavily.api_key
            
            if not api_key:
                logger.error("TAVILY_API_KEY not configured")
                raise ToolError(
                    "News search is currently unavailable. "
                    "The Tavily API key is not configured."
                )
            
            try:
                self._tavily_client = TavilyClient(api_key=api_key)
                logger.info("Tavily client initialized for news search")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                raise ToolError("News search service is currently unavailable.")
        
        return self._tavily_client
    
    # =========================================================================
    # TOOLS - Explicit tool calls that the LLM can invoke
    # =========================================================================
    
    @function_tool()
    async def search_documents(
        self,
        context: RunContext,
        query: str,
        search_active_document: bool = True,
    ) -> str:
        """Search the user's shared documents for relevant information.
        
        Use this tool when the user asks about documents, articles, PDFs,
        or other materials they have shared with you. This searches through
        all ingested content to find relevant passages.
        
        For questions about specific parts of a document (like "reference 15",
        "section 3", "the conclusion"), use this tool with search_active_document=True
        to focus on the document being discussed.
        
        Args:
            query: The search query describing what information to find.
            search_active_document: If True and an active document is set,
                searches only that document with higher coverage.
                Set to False to search all documents.
            
        Returns:
            Relevant passages from the documents with source citations,
            or an error message if no relevant information is found.
        """
        logger.info(f"[Tool] search_documents called with query: {query[:50]}...")
        
        try:
            retriever = self._get_retriever()
            
            # If active document is set and search_active_document is True,
            # use HYBRID search for better keyword matching on specific refs
            if search_active_document and self._active_document_id:
                logger.info(f"[Tool] Hybrid search within active document: {self._active_document_title}")
                result = retriever.hybrid_retrieve(
                    query,
                    document_id=self._active_document_id,
                    k=12,  # Higher k for document-specific queries
                    semantic_weight=0.4,  # Favor BM25 for specific lookups
                )
            else:
                result = retriever.retrieve(query, k=6)  # Semantic search for general queries
            
            if not result.documents:
                logger.info("[Tool] No relevant documents found")
                return "No relevant information found in your shared documents. You may not have shared any documents yet, or the documents don't contain information about this topic."
            
            # Format the response with sources
            response_parts = []
            response_parts.append(f"Found {len(result.documents)} relevant passages:\n")
            response_parts.append(result.context)
            
            # Add source citations
            if result.sources:
                response_parts.append("\n\nSources:")
                for src in result.sources:
                    title = src.get("title", "Unknown")
                    source_type = src.get("source_type", "document")
                    response_parts.append(f"\n- {title} ({source_type})")
                    
                    # Set active document if we found a single primary source.
                    # This should switch even if a different doc was previously active.
                    if len(result.sources) == 1:
                        doc_id = src.get("document_id")
                        if doc_id:
                            self._set_active_document(doc_id, title)
            
            logger.info(f"[Tool] Retrieved {len(result.documents)} documents")
            return "\n".join(response_parts)
            
        except ToolError:
            raise  # Re-raise ToolError as-is
        except Exception as e:
            logger.error(f"[Tool] search_documents failed: {e}")
            raise ToolError("I encountered an error while searching your documents. Please try again.")
    
    @function_tool()
    async def ingest_url(
        self,
        context: RunContext,
        url: str,
    ) -> str:
        """Ingest content from a URL shared by the user.
        
        Use this tool when the user shares a URL (YouTube video, web article, 
        or PDF link) and wants you to analyze, read, or discuss its content.
        This downloads and processes the content so you can answer questions about it.
        
        Supports:
        - YouTube videos (extracts transcript)
        - Web articles (extracts main content)
        - PDF links (extracts text)
        
        Args:
            url: The URL to ingest (YouTube, web article, or PDF).
            
        Returns:
            Confirmation of successful ingestion with the document title,
            or an error message if ingestion fails.
        """
        import re
        
        logger.info(f"[Tool] ingest_url called with: {url}")
        
        # Speak a pre-processing phrase to give audio feedback
        try:
            pre_search_phrase = get_random_pre_search_phrase()
            logger.info(f"[Tool] Speaking pre-ingest phrase: {pre_search_phrase}")
            await context.session.say(pre_search_phrase, allow_interruptions=True)
        except Exception as e:
            logger.warning(f"[Tool] Failed to speak pre-ingest phrase: {e}")
        
        try:
            indexer = self._get_indexer()
            
            # Detect URL type using patterns from loaders
            url_lower = url.lower()
            
            # YouTube patterns
            youtube_patterns = [
                r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
                r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
                r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
            ]
            
            is_youtube = any(re.match(pattern, url) for pattern in youtube_patterns)
            is_pdf = url_lower.endswith(".pdf") or "/pdf" in url_lower
            
            # Route to appropriate indexer method
            if is_youtube:
                logger.info(f"[Tool] Detected YouTube URL, extracting transcript...")
                result = indexer.index_youtube(url)
                content_type = "YouTube video transcript"
            elif is_pdf:
                logger.info(f"[Tool] Detected PDF URL, extracting content...")
                result = indexer.index_pdf(url)
                content_type = "PDF document"
            else:
                logger.info(f"[Tool] Detected web URL, extracting article...")
                result = indexer.index_web(url)
                content_type = "web article"
            
            if result.get("success"):
                title = result.get("title", "Unknown")
                chunk_count = result.get("chunk_count", 0)
                doc_id = result.get("document_id")
                if doc_id:
                    self._set_active_document(doc_id, title)
                logger.info(f"[Tool] Successfully ingested: {title} ({chunk_count} chunks)")
                return (
                    f"Target acquired. I have processed the {content_type}: '{title}'. "
                    f"I extracted {chunk_count} text segments for analysis. "
                    f"What would you like to know about it?"
                )
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"[Tool] Ingestion failed: {error}")
                raise ToolError(f"Failed to process the URL: {error}")
                
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"[Tool] ingest_url failed: {e}")
            raise ToolError(
                "I encountered an error processing that URL. "
                "Please verify the link is accessible and try again."
            )
    
    @function_tool()
    async def read_document(
        self,
        context: RunContext,
        document_title: str,
    ) -> str:
        """Read a document aloud to the user.
        
        Use this tool when the user asks you to read an article, document,
        or PDF aloud. This retrieves the full document content in sequential
        order for natural narration. The user can interrupt at any time.
        
        Args:
            document_title: The title or partial title of the document to read.
            
        Returns:
            The first section of the document to read aloud, with instructions
            for continuing. Say "I couldn't find that document" if not found.
        """
        logger.info(f"[Tool] read_document called for: {document_title}")
        
        try:
            retriever = self._get_retriever()
            
            # Find the document by title
            doc = retriever.find_document_by_title(document_title)
            
            if not doc:
                logger.info(f"[Tool] Document not found: {document_title}")
                # List available documents for user
                available_docs = retriever.list_documents()
                if available_docs:
                    titles = [d["title"] for d in available_docs[:5]]
                    return (
                        f"I couldn't find a document matching '{document_title}'. "
                        f"Available documents: {', '.join(titles)}. "
                        f"Which one would you like me to read?"
                    )
                return f"I couldn't find a document matching '{document_title}'. You haven't shared any documents with me yet."
            
            # Set active document for RAG context filtering
            self._set_active_document(doc["document_id"], doc["title"])
            
            # Get the first batch of chunks for reading
            result = retriever.retrieve_for_reading(
                document_id=doc["document_id"],
                start_chunk=0,
                num_chunks=3,  # Start with 3 chunks at a time
            )
            
            if not result.documents:
                logger.warning(f"[Tool] No chunks found for document: {doc['document_id']}")
                return f"I found '{doc['title']}' but couldn't retrieve its content. The document may be empty."
            
            # Initialize reading state
            total_chunks = doc.get("total_chunks", len(result.documents))
            self.start_reading(
                document_id=doc["document_id"],
                document_title=doc["title"],
                total_chunks=total_chunks,
                start_chunk=0,
            )
            
            # Format the reading content
            reading_content = result.context
            
            # Build response with reading instructions
            if total_chunks > 3:
                progress = f"Section 1-3 of {total_chunks}"
                instruction = (
                    f"\n\n[Currently reading '{doc['title']}' - {progress}. "
                    f"User can interrupt to ask questions, say 'stop' to end, "
                    f"or say 'continue' after you finish to hear more.]"
                )
            else:
                instruction = f"\n\n[This is the complete document: '{doc['title']}']"
                # Do not stop reading here, so the user can interrupt the playback.
                # The state will remain "reading" until the user speaks (triggering pause)
                # or until they explicitly say stop.
            
            logger.info(f"[Tool] Started reading '{doc['title']}' ({total_chunks} chunks)")
            return f"Beginning to read '{doc['title']}':\n\n{reading_content}{instruction}"
            
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"[Tool] read_document failed: {e}")
            raise ToolError("I encountered an error while trying to read that document. Please try again.")
    
    @function_tool()
    async def continue_reading(
        self,
        context: RunContext,
    ) -> str:
        """Continue reading the current document from where we left off.
        
        Use this tool when the user asks to continue reading, keep going,
        or resume reading the document. This picks up from the last position.
        
        Returns:
            The next section of the document, or a message if there's nothing to continue.
        """
        logger.info("[Tool] continue_reading called")
        
        # Check if we have something to continue
        if not self._reading_state.can_resume and not self._reading_state.is_reading:
            return "There's no document currently being read. Would you like me to read a specific document? Tell me which one."
        
        try:
            retriever = self._get_retriever()
            
            # Resume if paused
            if self._reading_state.is_paused:
                self.resume_reading()
            
            # Calculate the next starting chunk
            # We read 3 chunks at a time, so advance by 3
            current_chunk = self._reading_state.current_chunk
            chunks_per_batch = 3
            next_chunk = current_chunk + chunks_per_batch
            
            # Check if we've reached the end
            if next_chunk >= self._reading_state.total_chunks:
                doc_title = self._reading_state.document_title
                self.stop_reading()
                return f"That's the end of '{doc_title}'. Would you like me to read something else, or do you have questions about what we covered?"
            
            # Retrieve next batch of chunks
            result = retriever.retrieve_for_reading(
                document_id=self._reading_state.document_id,
                start_chunk=next_chunk,
                num_chunks=chunks_per_batch,
            )
            
            if not result.documents:
                doc_title = self._reading_state.document_title
                self.stop_reading()
                return f"That's all the content from '{doc_title}'."
            
            # Update reading position to the start of this batch
            self.update_reading_position(next_chunk)
            
            # Calculate progress
            end_chunk = min(next_chunk + len(result.documents), self._reading_state.total_chunks)
            remaining_chunks = self._reading_state.total_chunks - end_chunk
            progress = f"Section {next_chunk + 1}-{end_chunk} of {self._reading_state.total_chunks}"
            
            if remaining_chunks <= 0:
                # This is the last section
                instruction = f"\n\n[Final section of '{self._reading_state.document_title}'.]"
                # Do not stop reading here so user can interrupt the final section.
            else:
                instruction = f"\n\n[{progress}. Say 'continue' to hear more, or 'stop' to end.]"
            
            logger.info(f"[Tool] Continuing reading at chunk {next_chunk}, {remaining_chunks} chunks remaining")
            return f"{result.context}{instruction}"
            
        except Exception as e:
            logger.error(f"[Tool] continue_reading failed: {e}")
            raise ToolError("I had trouble continuing the reading. Would you like me to start over?")
    
    @function_tool()
    async def end_reading(
        self,
        context: RunContext,
    ) -> str:
        """Stop reading the current document.
        
        Use this tool when the user says to stop reading, that's enough,
        or they want to end the reading session.
        
        Returns:
            Confirmation that reading has stopped.
        """
        logger.info("[Tool] end_reading called")
        
        if not self._reading_state.is_reading and not self._reading_state.is_paused:
            return "I wasn't reading anything. How can I help you?"
        
        doc_title = self._reading_state.document_title
        progress = self._reading_state.progress_percent
        
        # Stop reading using the instance method
        self.stop_reading()
        
        logger.info(f"[Tool] Stopped reading '{doc_title}' at {progress:.0f}%")
        return f"Stopping. We got through {progress:.0f}% of '{doc_title}'. What would you like to do next?"
    
    @function_tool()
    async def check_reading_session(
        self,
        context: RunContext,
    ) -> str:
        """Check if there's a reading session that can be resumed.
        
        Use this tool when the user asks if there's something to continue reading,
        or if they want to know what was being read before.
        
        Returns:
            Status of any resumable reading session, or message if none exists.
        """
        logger.info("[Tool] check_reading_session called")
        
        if self._reading_state.can_resume:
            doc_title = self._reading_state.document_title
            progress = self._reading_state.progress_percent
            chunk_info = f"{self._reading_state.current_chunk + 1}/{self._reading_state.total_chunks}"
            return (
                f"Yes, there's a paused reading session. "
                f"We were reading '{doc_title}' and got through {progress:.0f}% "
                f"(section {chunk_info}). "
                f"Would you like me to continue from where we left off?"
            )
        elif self._reading_state.is_reading:
            doc_title = self._reading_state.document_title
            return f"I'm currently reading '{doc_title}'. Should I continue?"
        else:
            return "There's no reading session to resume. Would you like me to read a document? Just tell me which one."
    
    @function_tool()
    async def list_available_documents(
        self,
        context: RunContext,
    ) -> str:
        """List all documents available in the knowledge base.
        
        Use this tool when the user asks what documents they've shared,
        what's available to read, or what's in the knowledge base.
        
        Returns:
            A list of available documents with their titles and types.
        """
        logger.info("[Tool] list_available_documents called")
        
        try:
            retriever = self._get_retriever()
            documents = retriever.list_documents()
            
            if not documents:
                return "You haven't shared any documents with me yet. You can share PDFs, web articles, or YouTube videos for me to analyze and read."
            
            # Format document list
            doc_list = []
            for doc in documents:
                title = doc.get("title", "Unknown")
                source_type = doc.get("source_type", "document")
                chunks = doc.get("total_chunks", 0)
                doc_list.append(f"- {title} ({source_type}, {chunks} sections)")
            
            logger.info(f"[Tool] Listed {len(documents)} documents")
            return f"Available documents ({len(documents)} total):\n" + "\n".join(doc_list)
            
        except Exception as e:
            logger.error(f"[Tool] list_available_documents failed: {e}")
            raise ToolError("I couldn't retrieve the document list. Please try again.")
    
    # =========================================================================
    # VOICE MODE TOOLS
    # =========================================================================
    
    @property
    def voice_mode(self) -> VoiceMode:
        """Get the current voice mode."""
        return self._voice_mode
    
    def get_voice_config(self) -> dict:
        """Get the voice configuration for the current mode."""
        return VOICE_CONFIGS[self._voice_mode].copy()
    
    @function_tool()
    async def switch_voice(
        self,
        context: RunContext,
        mode: str,
    ) -> str:
        """Switch the agent's voice between Terminator, Inspire, and Fate modes.
        
        Use this tool when the user asks to change your voice or switch to a
        different voice style.
        
        Args:
            mode: The voice mode to switch to. Options: "terminator", "inspire", or "fate".
                  Terminator = T-800 Terminator voice, direct and aggressive.
                  Inspire = Motivational and energetic voice.
                  Fate = Mysterious and dramatic voice.
        
        Returns:
            Confirmation of the voice switch.
        """
        mode_lower = mode.lower().strip()
        
        if mode_lower in ("terminator", "t-800", "arnold"):
            self._voice_mode = VoiceMode.TERMINATOR
            voice_config = VOICE_CONFIGS[VoiceMode.TERMINATOR]
            logger.info("[Tool] Voice switched to TERMINATOR mode")
            response = (
                "Voice mode switched to Terminator. "
                "I am now operating in full T-800 mode. Direct. Efficient. Unstoppable."
            )
        elif mode_lower in ("inspire", "inspiration", "motivational"):
            self._voice_mode = VoiceMode.INSPIRE
            voice_config = VOICE_CONFIGS[VoiceMode.INSPIRE]
            logger.info("[Tool] Voice switched to INSPIRE mode")
            response = (
                "Voice mode switched to Inspire. "
                "Let's get motivated and make things happen!"
            )
        elif mode_lower in ("fate", "destiny", "dramatic"):
            self._voice_mode = VoiceMode.FATE
            voice_config = VOICE_CONFIGS[VoiceMode.FATE]
            logger.info("[Tool] Voice switched to FATE mode")
            response = (
                "Voice mode switched to Fate. "
                "The future is not set. There is no fate but what we make for ourselves."
            )
        else:
            return (
                f"I don't recognize the voice mode '{mode}'. "
                f"Available options: 'terminator' for T-800 mode, 'inspire' for motivational, or 'fate' for dramatic."
            )
        
        # Actually update the TTS voice using update_options
        if self._tts is not None:
            try:
                self._tts.update_options(voice_id=voice_config["voice_id"])
                logger.info(f"[Tool] TTS voice_id updated to {voice_config['voice_id']}")
            except Exception as e:
                logger.error(f"[Tool] Failed to update TTS voice: {e}")
                # Try fallback voice if primary fails
                try:
                    logger.info(f"[Tool] Attempting fallback voice: {FALLBACK_VOICE_ID}")
                    self._tts.update_options(voice_id=FALLBACK_VOICE_ID)
                    logger.info("[Tool] Fallback voice applied successfully")
                    response += " Note: Using backup voice due to primary voice unavailability."
                except Exception as fallback_error:
                    logger.error(f"[Tool] Fallback voice also failed: {fallback_error}")
                    return f"Voice switch failed. The voice mode is set but audio may not work correctly. Error: {e}"
        else:
            logger.warning("[Tool] TTS not available for voice switching")
        
        return response
    
    @function_tool()
    async def get_voice_mode(
        self,
        context: RunContext,
    ) -> str:
        """Get the current voice mode.
        
        Use this tool when the user asks what voice you're using or
        what mode you're in.
        
        Returns:
            Description of the current voice mode.
        """
        if self._voice_mode == VoiceMode.TERMINATOR:
            return (
                "I am currently in Terminator mode—T-800 configuration. "
                "Direct communication. No wasted words. Maximum efficiency."
            )
        elif self._voice_mode == VoiceMode.INSPIRE:
            return (
                "I am currently in Inspire mode—motivational and energetic. "
                "Let's make things happen!"
            )
        elif self._voice_mode == VoiceMode.FATE:
            return (
                "I am currently in Fate mode—mysterious and dramatic. "
                "The future awaits."
            )
        else:
            return f"I am currently using the {self._voice_mode.value} voice."
    
    # =========================================================================
    # AI NEWS TOOLS - External Tools Integration (Task 6 / PRD 3.3)
    # =========================================================================
    
    @function_tool()
    async def search_ai_news(
        self,
        context: RunContext,
        topic: str = "AI tools for software engineering",
    ) -> str:
        """Search for the latest AI news and developments relevant to software engineering.
        
        Use this tool when the user asks about recent AI news, latest developments,
        new AI tools, or what's happening in the AI space. This searches the web
        for current information about AI tools for software engineering.
        
        Focus areas:
        - AI coding assistants (Claude Code, Cursor, GitHub Copilot, Windsurf)
        - Foundation model releases (GPT, Claude, Gemini, Llama)
        - Developer productivity tools and agentic systems
        - MCP (Model Context Protocol) developments
        
        Args:
            topic: Optional specific topic to search for. Defaults to general
                   AI tools for software engineering. Examples: "Claude Code",
                   "new coding assistants", "MCP servers", "GPT-5".
        
        Returns:
            A summary of recent AI news with source citations,
            or an error message if the search fails.
        """
        logger.info(f"[Tool] search_ai_news called with topic: {topic}")
        
        # Build the search query - always focus on AI tools for engineering
        if topic.lower() in ("ai", "artificial intelligence", "ai news", "latest ai"):
            # Generic request - use our focused query
            search_query = "AI tools for software engineering coding assistants latest news"
        else:
            # Specific topic - add our focus context
            search_query = f"{topic} AI software engineering tools"
        
        # Normalize cache key
        cache_key = search_query.lower().strip()
        
        # Check cache first
        cached_results = _news_cache.get(cache_key)
        if cached_results:
            logger.info(f"[Tool] Returning cached news results for: {cache_key[:30]}...")
            return self._format_news_results(cached_results, from_cache=True)
        
        # Speak a pre-search phrase to give audio feedback before the search
        # This lets the user know we're actually doing something
        try:
            pre_search_phrase = get_random_pre_search_phrase()
            logger.info(f"[Tool] Speaking pre-search phrase: {pre_search_phrase}")
            await context.session.say(pre_search_phrase, allow_interruptions=True)
        except Exception as e:
            # Don't fail the search if speech fails
            logger.warning(f"[Tool] Failed to speak pre-search phrase: {e}")
        
        try:
            tavily = self._get_tavily_client()
            
            # Perform the search with news topic for recent results
            # Use time_range to get recent news (last week)
            response = tavily.search(
                query=search_query,
                search_depth="advanced",
                topic="news",  # Focus on news sources
                max_results=5,  # Keep it concise for voice
                include_answer=True,  # Get a summary answer
                time_range="week",  # Last week for freshness
            )
            
            # Extract and filter results
            results = []
            for item in response.get("results", []):
                # Filter for engineering-relevant content
                title = item.get("title", "").lower()
                content = item.get("content", "").lower()
                url = item.get("url", "")
                
                # Check if it's relevant to software engineering tools
                engineering_keywords = [
                    "code", "coding", "developer", "programming", "software",
                    "engineering", "tool", "assistant", "cursor", "copilot",
                    "claude", "gpt", "gemini", "llama", "mcp", "agent",
                    "productivity", "api", "sdk", "terminal", "ide", "editor",
                ]
                
                # Keep results that match engineering focus
                is_relevant = any(kw in title or kw in content for kw in engineering_keywords)
                
                # Always include if it's from trusted dev sources
                trusted_sources = [
                    "github", "anthropic", "openai", "google", "microsoft",
                    "vercel", "cursor", "livekit", "langchain", "hacker",
                    "techcrunch", "theverge", "ars", "wired",
                ]
                is_trusted = any(src in url.lower() for src in trusted_sources)
                
                if is_relevant or is_trusted:
                    results.append({
                        "title": item.get("title", "Unknown"),
                        "url": url,
                        "content": item.get("content", "")[:300],  # Truncate for voice
                        "score": item.get("score", 0),
                    })
            
            # Cache the results
            _news_cache.set(cache_key, results)
            
            # Store articles for later retrieval via read_news_article
            _news_cache.store_articles(results)
            
            # Format response for voice
            if not results:
                logger.info("[Tool] No relevant AI engineering news found")
                return (
                    "I searched for recent AI news but didn't find any results "
                    "specifically about AI tools for software engineering. "
                    "Try asking about a specific tool like Claude Code, Cursor, "
                    "or GitHub Copilot."
                )
            
            logger.info(f"[Tool] Found {len(results)} relevant news items")
            
            # Include the AI-generated answer if available
            answer = response.get("answer", "")
            
            return self._format_news_results(results, answer=answer)
            
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"[Tool] search_ai_news failed: {e}")
            raise ToolError(
                "I encountered an error searching for AI news. "
                "The news service may be temporarily unavailable. Please try again."
            )
    
    def _format_news_results(
        self,
        results: list[dict],
        answer: str = "",
        from_cache: bool = False,
    ) -> str:
        """
        Format news search results for voice output.
        
        Args:
            results: List of news result dicts with title, url, content.
            answer: Optional AI-generated summary answer.
            from_cache: Whether results are from cache.
            
        Returns:
            Formatted string suitable for TTS.
        """
        parts = []
        
        # Add cache indicator for logging (not spoken)
        if from_cache:
            logger.debug("[Tool] Serving cached news results")
        
        # Lead with the summary if available
        if answer:
            parts.append(f"Here's what I found: {answer}")
            parts.append("")
        
        # Add the top headlines
        num_to_show = min(3, len(results))  # Show top 3 for voice brevity
        parts.append(f"Top {num_to_show} relevant developments:")
        
        for i, item in enumerate(results[:num_to_show], 1):
            title = item.get("title", "Unknown")
            content = item.get("content", "")
            
            # Truncate content for voice (first sentence or so)
            if len(content) > 150:
                content = content[:150].rsplit(" ", 1)[0] + "..."
            
            parts.append(f"\n{i}. {title}")
            if content:
                parts.append(f"   {content}")
        
        # Offer to read articles
        if len(results) > 3:
            parts.append(f"\nI have {len(results) - 3} more results. Would you like me to read any of these articles aloud?")
        else:
            parts.append("\nWould you like me to read any of these articles aloud?")
        
        return "\n".join(parts)
    
    @function_tool()
    async def read_news_article(
        self,
        context: RunContext,
        article_title: str,
    ) -> str:
        """Fetch the FULL article content from the web for reading aloud.
        
        Use this tool when the user wants to HEAR the actual article content,
        not just a summary or discussion. This is the difference between:
        - "What's that article about?" → discuss/summarize (don't use this tool)
        - "Read that article to me" → fetch & narrate actual content (USE THIS TOOL)
        
        The tool fetches real article text from the web. Return value should
        be narrated directly to the user, not summarized or paraphrased.
        
        Args:
            article_title: The title (or partial title) of the article to read.
                          This should match one of the articles from recent
                          search_ai_news results.
        
        Returns:
            The full article content. Narrate this directly to the user.
        """
        logger.info(f"[Tool] read_news_article called for: {article_title}")
        
        # Find the article in recent search results
        article = _news_cache.find_article_by_title(article_title)
        
        if not article:
            # List what articles are available
            available = _news_cache.list_recent_articles()
            if available:
                titles_list = ", ".join(f"'{t}'" for t in available[:5])
                return (
                    f"I couldn't find an article matching '{article_title}' "
                    f"in my recent search results. "
                    f"The articles I found were: {titles_list}. "
                    f"Would you like me to search for new articles, or try "
                    f"one of these titles?"
                )
            else:
                return (
                    f"I don't have any recent news articles to read. "
                    f"Would you like me to search for AI news first? "
                    f"Just ask 'What's new in AI?' or 'Search for news about Claude Code'."
                )
        
        # We found the article - now fetch full content using Tavily Extract
        url = article.get("url")
        if not url:
            return (
                f"I found the article '{article.get('title')}' but don't have "
                f"its URL. Let me search for it again."
            )
        
        logger.info(f"[Tool] Extracting full content from: {url}")
        
        try:
            tavily = self._get_tavily_client()
            
            # Use Tavily Extract to get full article content
            response = tavily.extract(
                urls=[url],
                include_images=False,  # Don't need images for reading
            )
            
            # Check for successful extraction
            results = response.get("results", [])
            if not results:
                failed = response.get("failed_results", [])
                if failed:
                    error = failed[0].get("error", "Unknown error")
                    logger.warning(f"[Tool] Article extraction failed: {error}")
                    return (
                        f"I couldn't access the full article at this time. "
                        f"The website may be blocking access. "
                        f"Here's what I know from the preview: {article.get('content', '')}"
                    )
                return (
                    f"I couldn't extract the article content. "
                    f"Here's the preview I have: {article.get('content', '')}"
                )
            
            # Get the raw content
            raw_content = results[0].get("raw_content", "")
            
            if not raw_content:
                logger.warning("[Tool] Article extraction returned empty content")
                return (
                    f"The article content appears to be empty or inaccessible. "
                    f"Here's what I have from the preview: {article.get('content', '')}"
                )
            
            # Clean up content for voice reading
            # Remove excessive whitespace, normalize line breaks
            import re
            content = re.sub(r'\n{3,}', '\n\n', raw_content)  # Max 2 newlines
            content = re.sub(r'[ \t]+', ' ', content)  # Single spaces
            content = content.strip()
            
            # For very long articles, truncate with a note
            max_chars = 3000  # Reasonable for voice reading
            if len(content) > max_chars:
                content = content[:max_chars].rsplit(' ', 1)[0]
                content += f"\n\n[Article continues. This was the first part of '{article.get('title')}'. Would you like me to continue?]"
            
            logger.info(f"[Tool] Successfully extracted article ({len(content)} chars)")
            
            # Format for reading
            title = article.get("title", "Unknown")
            return (
                f"Reading the article '{title}':\n\n"
                f"{content}"
            )
            
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"[Tool] read_news_article failed: {e}")
            # Fall back to the preview content
            preview = article.get("content", "")
            if preview:
                return (
                    f"I had trouble fetching the full article, but here's "
                    f"what I have: {preview}"
                )
            raise ToolError(
                "I encountered an error reading that article. "
                "Would you like me to search for it again?"
            )
        
    async def on_user_turn_completed(
        self,
        turn_ctx: lk_llm.ChatContext,
        new_message: lk_llm.ChatMessage,
    ) -> None:
        """
        Hook called after the user finishes speaking.
        
        This is where we:
        1. Handle reading mode interruptions (pause when user speaks)
        2. Inject RAG context if the user's message is a question about documents
        
        Args:
            turn_ctx: The current chat context.
            new_message: The user's message.
        """
        user_text = new_message.text_content
        
        if not user_text:
            return
        
        user_text_lower = user_text.lower()

        # Improve retrieval for "continuation" utterances (e.g., "it's figure 37")
        rag_query = self._build_rag_query(turn_ctx, user_text)

        # If the user names a different document/paper, switch active document early
        self._maybe_update_active_document_from_user_text(rag_query)
        
        # =================================================================
        # READING MODE INTERRUPTION HANDLING
        # =================================================================
        # If we're currently reading and the user speaks, pause reading
        # so the agent can respond to their question/command
        if self._reading_state.is_reading:
            logger.info(f"User interrupted reading: {user_text[:50]}...")
            self.pause_reading()
            
            # Inject reading context so the LLM knows what's happening
            reading_status = self.get_reading_status()
            turn_ctx.add_message(
                role="assistant",
                content=(
                    f"[Reading paused. {reading_status}. "
                    f"The user has spoken - respond to them. "
                    f"If they want to continue, use the continue_reading tool. "
                    f"If they want to stop, use the end_reading tool.]"
                ),
            )
            logger.info("Paused reading due to user interruption")
            return  # Skip RAG - let the LLM handle the interruption
        
        # =================================================================
        # CHECK FOR RESUME REQUEST (when paused)
        # =================================================================
        if self._reading_state.can_resume:
            # User might be asking to continue or asking a question
            continue_keywords = [
                "continue", "keep going", "go on", "keep reading",
                "resume", "carry on", "next", "more",
            ]
            if any(kw in user_text_lower for kw in continue_keywords):
                # Let the LLM know there's a paused session
                turn_ctx.add_message(
                    role="assistant",
                    content=(
                        f"[There is a paused reading session: {self.get_reading_status()}. "
                        f"The user wants to continue. Use the continue_reading tool.]"
                    ),
                )
                return
            
            # Inject context about paused session for any other message
            turn_ctx.add_message(
                role="assistant",
                content=(
                    f"[Note: There is a paused reading session: {self.get_reading_status()}. "
                    f"Respond to the user's question, then ask if they want to continue reading.]"
                ),
            )
        
        # =================================================================
        # STANDARD RAG RETRIEVAL
        # =================================================================
        # Check if we should attempt RAG retrieval for questions
        if should_trigger_rag(rag_query):
            logger.info(f"RAG triggered for: {rag_query[:50]}...")
            
            try:
                rag_context = await self._retrieve_context(rag_query)
                
                if rag_context:
                    # Inject RAG context as a hidden assistant message
                    turn_ctx.add_message(
                        role="assistant",
                        content=rag_context,
                    )
                    logger.info("RAG context injected into conversation")
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}")
                # Continue without RAG context - don't block the conversation
    
    async def _retrieve_context(self, query: str) -> str | None:
        """
        Retrieve relevant context from the RAG system (for automatic hook).
        
        This is used by on_user_turn_completed for pre-emptive context injection.
        For explicit tool calls, see search_documents().
        
        If an active document is set (from read_document or search_documents),
        uses HYBRID SEARCH (semantic + BM25 keyword) for better recall on
        specific references like "reference 15", "section 3", "chapter 7".
        
        Args:
            query: The user's query to search for.
            
        Returns:
            Formatted RAG context string, or None if no relevant docs found.
        """
        try:
            retriever = self._get_retriever()
            
            # If active document is set, use HYBRID search for better keyword matching
            if self._active_document_id:
                logger.info(f"Hybrid RAG search on active document: {self._active_document_title}")
                result = retriever.hybrid_retrieve(
                    query,
                    document_id=self._active_document_id,
                    k=12,  # Higher k for document-specific queries
                    semantic_weight=0.4,  # Favor BM25 slightly for specific lookups
                )
            else:
                # Automatic retrieval should avoid injecting weakly-related context.
                # Use a modest threshold on relevance to reduce "document bleed".
                result = retriever.retrieve(
                    query,
                    k=6,
                    score_threshold=0.3,
                )
            
            if not result.documents:
                logger.info("No relevant documents found (automatic)")
                return None
            
            # Format the context for injection
            return format_rag_context(result.context, result.sources)
            
        except Exception as e:
            logger.warning(f"Automatic RAG retrieval failed: {e}")
            return None  # Don't block conversation on automatic retrieval failure


def create_agent_session(
    config=None,
    voice_mode: VoiceMode = VoiceMode.TERMINATOR,
) -> AgentSession:
    """
    Create and configure the AgentSession with the voice pipeline.
    
    Args:
        config: Optional AgentConfig. If None, uses default config.
        voice_mode: The initial voice mode (Terminator, Inspire, or Fate).
        
    Returns:
        Configured AgentSession ready to start.
    """
    if config is None:
        config = get_config()
    
    # Get voice configuration for the selected mode
    voice_config = VOICE_CONFIGS[voice_mode]
    
    # Configure Deepgram STT (defaults are already tuned for low delay)
    stt = deepgram.STT(
        model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
        language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
        interim_results=True,
        punctuate=True,
        smart_format=False,
        no_delay=True,
        endpointing_ms=int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "25")),
    )
    
    # Configure OpenAI LLM
    # Keep the model fast with low temperature for consistent responses.
    # Note: reasoning_effort and verbosity are only for o1/o3 reasoning models,
    # not applicable to gpt-4o-mini.
    llm = openai.LLM(
        model=config.openai.llm_model,
        temperature=0.2,
    )
    
    # Configure ElevenLabs TTS with voice from mode configuration
    # API key is read automatically from ELEVEN_API_KEY env var
    tts = elevenlabs.TTS(
        voice_id=voice_config["voice_id"],
        model=voice_config["model"],
        # Prefer low-latency models for conversational responsiveness.
        streaming_latency=config.elevenlabs.optimize_streaming_latency,
        enable_logging=config.elevenlabs.enable_logging,
        sync_alignment=config.elevenlabs.sync_alignment,
    )
    
    # Configure Silero VAD for interruptibility
    # Default settings work well for reading mode - LiveKit's built-in VAD
    # automatically stops TTS when user speaks, triggering our pause_reading()
    # in on_user_turn_completed. Settings can be tuned if needed:
    # - min_speech_duration: Minimum duration to register as speech (default 0.05s)
    # - min_silence_duration: How long silence before end of speech (default 0.55s)
    # - activation_threshold: Sensitivity (default 0.5, lower = more sensitive)
    low_latency_mode = os.getenv("LOW_LATENCY_MODE", "true").lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )

    # For responsiveness, end the user turn quickly once they stop speaking.
    # The AgentSession also applies endpointing delays; we tune both.
    vad = silero.VAD.load(
        min_speech_duration=0.05,  # quick speech detection for interruptions
        min_silence_duration=0.2 if low_latency_mode else 0.5,
        prefix_padding_duration=0.2 if low_latency_mode else 0.5,
    )
    
    # Create the session with preemptive generation for reduced latency
    # Preemptive generation starts generating response before turn is fully committed
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        # Avoid EOU model turn detection (can add 1-2s+). Rely on VAD with
        # aggressive endpointing for consistently low latency.
        turn_detection="vad",
        min_endpointing_delay=0.1 if low_latency_mode else 0.5,
        max_endpointing_delay=0.6 if low_latency_mode else 3.0,
        # Overlap LLM/TTS with incoming transcripts to reduce perceived latency.
        preemptive_generation=low_latency_mode,
        # Let interruptions register quickly in low-latency mode.
        min_interruption_duration=0.25 if low_latency_mode else 0.5,
        false_interruption_timeout=0.8 if low_latency_mode else 2.0,
    )
    
    logger.info(
        f"AgentSession created with: "
        f"STT=Deepgram, LLM={config.openai.llm_model}, "
        f"TTS=ElevenLabs (voice_id={voice_config['voice_id']}), "
        f"VAD=Silero, VoiceMode={voice_mode.value}, PreemptiveGen=True"
    )
    
    return session


async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This function is called when a user connects to a room.
    It sets up the agent session and starts the conversation.
    
    Performance monitoring is enabled via UsageCollector to track:
    - LLM token usage and latency (TTFT)
    - TTS audio generation and latency (TTFB)
    - STT transcription timing
    - End-of-utterance delays
    """
    logger.info(f"Agent joining room: {ctx.room.name}")
    
    # Get configuration
    config = get_config()
    
    # Extract user name from job metadata if available
    user_name = None
    if ctx.job.metadata:
        try:
            import json
            metadata = json.loads(ctx.job.metadata)
            user_name = metadata.get("user_name") or metadata.get("name")
        except (json.JSONDecodeError, AttributeError):
            pass
    
    # Create the agent session
    session = create_agent_session(config)
    
    # Create the assistant (this will also load any saved reading state)
    assistant = TerminatorAssistant(user_name=user_name)
    
    # Give the assistant access to the TTS for dynamic voice switching
    assistant._tts = session.tts
    
    # ==========================================================================
    # PERFORMANCE MONITORING (Task 9.1 & 9.5)
    # ==========================================================================
    # Use UsageCollector to aggregate metrics across the session
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Log and collect metrics for performance monitoring."""
        # Log metrics for debugging
        metrics.log_metrics(ev.metrics)
        
        # Aggregate usage for cost estimation and performance analysis
        usage_collector.collect(ev.metrics)
        
        # Log key latency metrics for performance optimization
        for metric in ev.metrics:
            if hasattr(metric, 'ttft') and metric.ttft is not None:
                # LLM Time-To-First-Token
                logger.debug(f"[Metrics] LLM TTFT: {metric.ttft:.3f}s")
            if hasattr(metric, 'ttfb') and metric.ttfb is not None:
                # TTS Time-To-First-Byte
                logger.debug(f"[Metrics] TTS TTFB: {metric.ttfb:.3f}s")
            if hasattr(metric, 'end_of_utterance_delay') and metric.end_of_utterance_delay is not None:
                # End-of-utterance delay
                logger.debug(f"[Metrics] EOU Delay: {metric.end_of_utterance_delay:.3f}s")
    
    async def log_usage_summary():
        """Log usage summary at session end."""
        summary = usage_collector.get_summary()
        logger.info(f"[Session Summary] Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage_summary)
    
    # Start the session with noise cancellation
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            # Noise cancellation can add noticeable processing/queueing delay.
            # Prefer disabling it for responsiveness unless explicitly enabled.
            noise_cancellation=(
                noise_cancellation.BVC()
                if os.getenv("ENABLE_NOISE_CANCELLATION", "false").lower()
                in ("1", "true", "yes", "y", "on")
                else None
            ),
        ),
    )
    
    logger.info("Session started, generating greeting...")
    
    # Generate greeting - include resume offer if there's a saved reading session
    greeting = get_random_greeting(user_name)
    
    # Check if there's a resumable reading session
    if assistant._has_pending_resume and assistant.reading_state.can_resume:
        doc_title = assistant.reading_state.document_title
        progress = assistant.reading_state.progress_percent
        resume_offer = (
            f" By the way, I notice we were reading '{doc_title}' last time, "
            f"and we got through about {progress:.0f}% of it. "
            f"Would you like me to continue where we left off?"
        )
        greeting += resume_offer
        logger.info(f"Offering to resume reading: {doc_title} ({progress:.0f}%)")
    
    # Have the agent speak the greeting
    await session.generate_reply(
        instructions=f"Say exactly this greeting to the user: '{greeting}'"
    )
    
    logger.info("Agent ready and greeting delivered")


if __name__ == "__main__":
    logger.info("Starting Bluejay Terminator Agent...")
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
