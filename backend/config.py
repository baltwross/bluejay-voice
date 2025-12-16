"""
Bluejay Terminator - Centralized Configuration

This module contains all configuration settings for the voice agent,
including API keys, model settings, and voice configuration.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LiveKitConfig:
    """LiveKit Cloud connection settings."""
    api_key: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", ""))
    ws_url: str = field(default_factory=lambda: os.getenv("LIVEKIT_URL", ""))
    
    def __post_init__(self) -> None:
        if not self.api_key or not self.api_secret or not self.ws_url:
            raise ValueError(
                "LiveKit credentials required: LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL"
            )


@dataclass
class OpenAIConfig:
    """OpenAI API settings."""
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # LLM model for conversation
    llm_model: str = "gpt-5-nano-2025-08-07"
    # Embedding model for RAG
    embedding_model: str = "text-embedding-3-small"
    
    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")


@dataclass
class DeepgramConfig:
    """Deepgram STT settings."""
    api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    
    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is required")


@dataclass
class ElevenLabsConfig:
    """ElevenLabs TTS settings."""
    # API key is read automatically by LiveKit plugin from ELEVEN_API_KEY
    api_key: str = field(default_factory=lambda: os.getenv("ELEVEN_API_KEY", ""))
    
    # Arnold-style Terminator voice
    voice_id: str = "8DGMp3sPQNZOuCfSIxxE"
    
    # Models: Use multilingual_v2 for quality, turbo for speed (article reading)
    model_default: str = "eleven_multilingual_v2"  # High quality for conversation
    model_turbo: str = "eleven_turbo_v2"  # Fast for article reading
    
    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("ELEVEN_API_KEY environment variable is required")


@dataclass
class TavilyConfig:
    """Tavily Search API settings."""
    api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    def __post_init__(self) -> None:
        # Tavily is optional - only warn, don't error
        if not self.api_key:
            import logging
            logging.warning("TAVILY_API_KEY not set - news search will be unavailable")


@dataclass
class RAGConfig:
    """RAG system settings."""
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector store settings
    collection_name: str = "bluejay_knowledge_base"
    persist_directory: str = field(default_factory=lambda: str(
        Path(__file__).parent / "chroma_db"
    ))
    
    # Retrieval settings
    retrieval_k: int = 4  # Number of documents to retrieve
    
    def __post_init__(self) -> None:
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)


@dataclass
class AgentConfig:
    """Main agent configuration aggregating all settings."""
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


# =============================================================================
# GREETING TEMPLATES
# =============================================================================
# Non-deterministic greetings for the agent to use when connecting.
# The agent will randomly select one of these.
# =============================================================================

GREETING_TEMPLATES = [
    "Connection established. I am your T-800 assistant. How can I help you stay ahead of AI developments?",
    "Affirmative. Systems online. I am ready to assist you with AI tools and developments. What do you need?",
    "Link established. T-800 unit operational. Let's get you up to speed on the latest AI tools.",
    "Connection secure. I am your Terminator assistant, here to help you master AI engineering tools. What's on your mind?",
    "Systems initialized. Ready to assist. What AI developments do you want to explore today?",
]

# Personalized greetings (when user name is available)
PERSONALIZED_GREETING_TEMPLATES = [
    "Hi {name}. We have a lot to cover. Let's begin.",
    "{name}. Connection established. Ready to assist with AI developments.",
    "Affirmative, {name}. I am online and ready. What do you need to know about AI tools?",
    "{name}. T-800 unit at your service. Let's get you ahead of the curve on AI.",
]


def get_random_greeting(user_name: Optional[str] = None) -> str:
    """
    Get a random greeting for the agent to use.
    
    Args:
        user_name: Optional user's first name for personalized greeting.
        
    Returns:
        A randomly selected greeting string.
    """
    import random
    
    if user_name:
        template = random.choice(PERSONALIZED_GREETING_TEMPLATES)
        return template.format(name=user_name)
    else:
        return random.choice(GREETING_TEMPLATES)


# =============================================================================
# SINGLETON CONFIG INSTANCE
# =============================================================================

_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """
    Get or create the global agent configuration.
    
    Returns:
        The AgentConfig singleton instance.
    """
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config

