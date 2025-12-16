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

# Fix SSL certificate verification on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    llm as lk_llm,
)
from livekit.plugins import deepgram, openai, silero, elevenlabs, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompts import get_system_prompt, format_rag_context, should_trigger_rag
from config import get_config, get_random_greeting

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerminatorAssistant(Agent):
    """
    T-800 Terminator Voice Assistant.
    
    This agent uses the Terminator personality defined in prompts.py
    and can be extended with RAG capabilities and tool calls.
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
        Retrieve relevant context from the RAG system.
        
        Args:
            query: The user's query to search for.
            
        Returns:
            Formatted RAG context string, or None if no relevant docs found.
        """
        # Lazy import and initialization to avoid circular imports
        # and only load RAG when needed
        if self._retriever is None:
            try:
                from rag.retriever import DocumentRetriever
                self._retriever = DocumentRetriever()
                logger.info("RAG retriever initialized")
            except Exception as e:
                logger.warning(f"Could not initialize RAG retriever: {e}")
                return None
        
        # Perform retrieval
        result = self._retriever.retrieve(query)
        
        if not result.documents:
            logger.info("No relevant documents found")
            return None
        
        # Format the context for injection
        return format_rag_context(result.context, result.sources)


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

