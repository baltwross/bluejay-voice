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
"""
import os
import logging
import certifi
from dataclasses import dataclass, field
from typing import Optional

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
)
from livekit.plugins import deepgram, openai, silero, elevenlabs, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # type: ignore[import-not-found]

from prompts import get_system_prompt, format_rag_context, should_trigger_rag
from config import get_config, get_random_greeting
from rag.retriever import DocumentRetriever

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
    
    def save(self, file_path: Path = READING_STATE_FILE) -> bool:
        """
        Persist reading state to file.
        
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self.last_updated = datetime.now().isoformat()
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"Reading state saved to {file_path}")
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
            if not file_path.exists():
                logger.debug("No saved reading state found")
                return None
            
            with open(file_path, "r") as f:
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
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load reading state: {e}")
            cls.cleanup(file_path)
            return None
    
    @classmethod
    def cleanup(cls, file_path: Path = READING_STATE_FILE) -> bool:
        """
        Remove the saved reading state file.
        
        Returns:
            True if cleaned up successfully, False otherwise.
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up reading state file: {file_path}")
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
    
    State:
        - reading_state: Tracks document reading mode (position, paused, etc.)
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
    
    # =========================================================================
    # TOOLS - Explicit tool calls that the LLM can invoke
    # =========================================================================
    
    @function_tool()
    async def search_documents(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search the user's shared documents for relevant information.
        
        Use this tool when the user asks about documents, articles, PDFs,
        or other materials they have shared with you. This searches through
        all ingested content to find relevant passages.
        
        Args:
            query: The search query describing what information to find.
            
        Returns:
            Relevant passages from the documents with source citations,
            or an error message if no relevant information is found.
        """
        logger.info(f"[Tool] search_documents called with query: {query[:50]}...")
        
        try:
            retriever = self._get_retriever()
            result = retriever.retrieve(query)
            
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
            
            logger.info(f"[Tool] Retrieved {len(result.documents)} documents")
            return "\n".join(response_parts)
            
        except ToolError:
            raise  # Re-raise ToolError as-is
        except Exception as e:
            logger.error(f"[Tool] search_documents failed: {e}")
            raise ToolError("I encountered an error while searching your documents. Please try again.")
    
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
        if should_trigger_rag(user_text):
            logger.info(f"RAG triggered for: {user_text[:50]}...")
            
            try:
                rag_context = await self._retrieve_context(user_text)
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
        
        Args:
            query: The user's query to search for.
            
        Returns:
            Formatted RAG context string, or None if no relevant docs found.
        """
        try:
            retriever = self._get_retriever()
            result = retriever.retrieve(query)
            
            if not result.documents:
                logger.info("No relevant documents found (automatic)")
                return None
            
            # Format the context for injection
            return format_rag_context(result.context, result.sources)
            
        except Exception as e:
            logger.warning(f"Automatic RAG retrieval failed: {e}")
            return None  # Don't block conversation on automatic retrieval failure


def create_agent_session(config=None) -> AgentSession:
    """
    Create and configure the AgentSession with the voice pipeline.
    
    Args:
        config: Optional AgentConfig. If None, uses default config.
        
    Returns:
        Configured AgentSession ready to start.
    """
    if config is None:
        config = get_config()
    
    # Configure Deepgram STT
    stt = deepgram.STT()
    
    # Configure OpenAI LLM
    llm = openai.LLM(model=config.openai.llm_model)
    
    # Configure ElevenLabs TTS with Arnold voice
    # API key is read automatically from ELEVEN_API_KEY env var
    tts = elevenlabs.TTS(
        voice_id=config.elevenlabs.voice_id,  # 8DGMp3sPQNZOuCfSIxxE
        model=config.elevenlabs.model_default,  # eleven_multilingual_v2 for quality
    )
    
    # Configure Silero VAD for interruptibility
    # Default settings work well for reading mode - LiveKit's built-in VAD
    # automatically stops TTS when user speaks, triggering our pause_reading()
    # in on_user_turn_completed. Settings can be tuned if needed:
    # - min_speech_duration: Minimum duration to register as speech (default 0.05s)
    # - min_silence_duration: How long silence before end of speech (default 0.55s)
    # - activation_threshold: Sensitivity (default 0.5, lower = more sensitive)
    vad = silero.VAD.load(
        min_speech_duration=0.05,    # Quick speech detection for interruptions
        min_silence_duration=0.5,    # Slightly faster end-of-speech detection
    )
    
    # Configure turn detection
    turn_detection = MultilingualModel()
    
    # Create the session
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_detection=turn_detection,
    )
    
    logger.info(
        f"AgentSession created with: "
        f"STT=Deepgram, LLM={config.openai.llm_model}, "
        f"TTS=ElevenLabs (model={config.elevenlabs.model_default}, voice_id={config.elevenlabs.voice_id}), VAD=Silero"
    )
    
    return session


async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This function is called when a user connects to a room.
    It sets up the agent session and starts the conversation.
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
    
    # Start the session with noise cancellation
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
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

