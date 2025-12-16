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
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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

@dataclass
class ReadingState:
    """
    Tracks the state of document reading mode.
    
    This allows the agent to:
    - Know if it's currently reading a document aloud
    - Track position for pause/resume functionality
    - Remember which document is being read
    """
    is_reading: bool = False
    is_paused: bool = False
    document_id: Optional[str] = None
    document_title: Optional[str] = None
    current_chunk: int = 0
    total_chunks: int = 0
    
    def reset(self) -> None:
        """Reset all reading state to defaults."""
        self.is_reading = False
        self.is_paused = False
        self.document_id = None
        self.document_title = None
        self.current_chunk = 0
        self.total_chunks = 0
    
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
    
    def to_status_string(self) -> str:
        """Get a human-readable status string."""
        if not self.is_reading and not self.is_paused:
            return "Not currently reading any document."
        
        status = "Paused" if self.is_paused else "Reading"
        title = self.document_title or "Unknown document"
        progress = f"{self.current_chunk + 1}/{self.total_chunks}"
        percent = f"{self.progress_percent:.0f}%"
        
        return f"{status}: '{title}' - Section {progress} ({percent} complete)"


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
        self._reading_state = ReadingState()  # Track reading mode state
    
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
        return True
    
    def pause_reading(self) -> None:
        """Pause reading (preserves position for resume)."""
        if self._reading_state.is_reading:
            self._reading_state.is_reading = False
            self._reading_state.is_paused = True
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
        
    async def on_user_turn_completed(
        self,
        turn_ctx: lk_llm.ChatContext,
        new_message: lk_llm.ChatMessage,
    ) -> None:
        """
        Hook called after the user finishes speaking.
        
        This is where we inject RAG context if the user's message
        seems to be asking about documents in the knowledge base.
        
        Args:
            turn_ctx: The current chat context.
            new_message: The user's message.
        """
        user_text = new_message.text_content
        if not user_text:
            return
            
        # Check if we should attempt RAG retrieval
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
    tts = elevenlabs.TTS(
        voice_id=config.elevenlabs.voice_id,
        model=config.elevenlabs.model_default,  # eleven_multilingual_v2 for quality
    )
    
    # Configure Silero VAD for interruptibility
    vad = silero.VAD.load()
    
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
        f"TTS=ElevenLabs ({config.elevenlabs.model_default}), VAD=Silero"
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
    
    # Create the assistant
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
    
    # Generate a random greeting
    greeting = get_random_greeting(user_name)
    
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

