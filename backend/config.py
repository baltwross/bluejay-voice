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
    # LLM model for conversation - use gpt-4o-mini for fast, affordable responses
    # Note: gpt-5-nano-2025-08-07 is quite slow
    llm_model: str = "gpt-4o-mini"
    # Embedding model for RAG
    embedding_model: str = "text-embedding-3-large"
    
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
    
    # Voice IDs for different modes
    # TERMINATOR: custom voice that works
    # INSPIRE/FATE: ElevenLabs premade voices (original custom ones aren't accessible)
    voice_id: str = "8DGMp3sPQNZOuCfSIxxE"  # Default: T-800 Terminator voice (custom)
    voice_id_terminator: str = "8DGMp3sPQNZOuCfSIxxE"  # T-800 Terminator voice (custom)
    voice_id_inspire: str = "qBDvhofpxp92JgXJxDjB"  # Lily Wolfe - motivational (premade)
    voice_id_fate: str = "pNInz6obpgDQGcFmaJgB"  # Adam - dramatic (premade)
    voice_id_fallback: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel - reliable fallback
    
    # Models: default to low-latency streaming models for responsiveness.
    # Valid examples: eleven_turbo_v2_5, eleven_flash_v2_5, eleven_multilingual_v2
    model_default: str = field(
        default_factory=lambda: os.getenv("ELEVEN_TTS_MODEL", "eleven_flash_v2_5")
    )
    # Kept for backwards compatibility with earlier code paths.
    model_turbo: str = field(
        default_factory=lambda: os.getenv("ELEVEN_TTS_TURBO_MODEL", "eleven_turbo_v2_5")
    )

    # ElevenLabs streaming optimization (0 disables, 4 is max). This parameter is
    # deprecated upstream but still supported by the API/plugin.
    optimize_streaming_latency: int = field(
        default_factory=lambda: int(os.getenv("ELEVEN_OPTIMIZE_STREAMING_LATENCY", "4"))
    )

    # When False, requests use "zero retention mode" and may be slightly faster.
    enable_logging: bool = field(
        default_factory=lambda: os.getenv("ELEVEN_ENABLE_LOGGING", "false").lower()
        in ("1", "true", "yes", "y", "on")
    )

    # When False, skip alignment work (lower latency; you still get audio).
    sync_alignment: bool = field(
        default_factory=lambda: os.getenv("ELEVEN_SYNC_ALIGNMENT", "false").lower()
        in ("1", "true", "yes", "y", "on")
    )
    
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
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    
    # Vector store settings
    collection_name: str = "bluejay_knowledge_base"
    persist_directory: str = field(default_factory=lambda: str(
        Path(__file__).parent / "chroma_db"
    ))
    
    # Retrieval settings
    retrieval_k: int = 4  # Number of documents to retrieve
    retrieval_score_threshold: float = 0.0  # Minimum similarity score
    
    # Re-ranking settings (for precise fact retrieval)
    # CrossEncoder re-ranker significantly improves precision for "specific fact" queries
    use_reranker: bool = field(default_factory=lambda: os.getenv(
        "RAG_USE_RERANKER", "true"
    ).lower() in ("1", "true", "yes", "y", "on"))
    reranker_model: str = field(default_factory=lambda: os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ))  # Fast and effective; alternatives: "BAAI/bge-reranker-base", "BAAI/bge-reranker-large"
    
    # Hybrid retrieval + re-ranking pipeline settings
    # Hybrid K (initial retrieval) → re-rank → top N → inject M into context
    hybrid_k: int = 20  # Initial retrieval size for hybrid search
    rerank_top_n: int = 8  # Keep top N after re-ranking
    context_top_m: int = 4  # Inject top M into LLM context

    # Adaptive context sizing
    # Anchor queries (figure/table/section/reference) often need neighbor expansion;
    # keeping too few chunks can truncate the relevant neighbor chunk.
    anchor_context_top_m: int = 12
    # On low-confidence retrieval, include more chunks so the LLM has more evidence
    # (still far from "entire doc", but helps when the relevant chunk is slightly lower-ranked).
    low_confidence_context_top_m: int = 20

    # Whether to expand neighbors even when no explicit anchor was detected but
    # confidence is low (only applies when a single document_id filter is in use).
    expand_on_low_confidence: bool = True
    
    # Confidence thresholds
    # auto_threshold: min relevance for automatic RAG injection in on_user_turn_completed
    # tool_threshold: min relevance for explicit tool calls (lower = more permissive)
    auto_injection_threshold: float = 0.35
    tool_call_threshold: float = 0.25
    low_confidence_threshold: float = 0.2  # Below this, admit uncertainty
    
    # Chunk expansion (neighbor windowing)
    # When anchor detected or confidence low, expand context by ±window_size chunks
    enable_chunk_expansion: bool = True
    chunk_expansion_window: int = 1  # ±1 chunk on each side
    
    # OpenAI API key (from environment)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    def __post_init__(self) -> None:
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)


@dataclass
class VADConfig:
    """Voice Activity Detection settings.
    
    These settings control how the agent detects speech and handles
    turn-taking. Tune these for optimal conversation flow.
    
    Guidelines:
    - min_speech_duration: Lower = faster interruption detection, but may trigger on noise
    - min_silence_duration: Lower = faster response, but may cut off user mid-thought
    - activation_threshold: Lower = more sensitive, may trigger on background noise
    """
    # Minimum speech duration to register as user speaking (seconds)
    # Lower values allow faster interruption detection
    min_speech_duration: float = 0.05  # 50ms - quick for interruptions
    
    # Minimum silence duration to end a turn (seconds)
    # Lower values = faster agent response, but may cut off user
    min_silence_duration: float = 0.45  # 450ms - balanced for conversation
    
    # Activation threshold (0.0-1.0)
    # Lower = more sensitive, may pick up background noise
    # Higher = less sensitive, may miss soft speech
    activation_threshold: float = 0.5  # Default, balanced sensitivity
    
    # Profiles for different environments
    @classmethod
    def quiet_environment(cls) -> "VADConfig":
        """Optimized for quiet environments - more sensitive."""
        return cls(
            min_speech_duration=0.03,
            min_silence_duration=0.4,
            activation_threshold=0.4,
        )
    
    @classmethod
    def noisy_environment(cls) -> "VADConfig":
        """Optimized for noisy environments - less sensitive."""
        return cls(
            min_speech_duration=0.1,
            min_silence_duration=0.6,
            activation_threshold=0.6,
        )
    
    @classmethod
    def reading_mode(cls) -> "VADConfig":
        """Optimized for reading mode - quick interruption detection."""
        return cls(
            min_speech_duration=0.03,
            min_silence_duration=0.35,
            activation_threshold=0.45,
        )


@dataclass
class AgentConfig:
    """Main agent configuration aggregating all settings."""
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    deepgram: DeepgramConfig = field(default_factory=DeepgramConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    vad: VADConfig = field(default_factory=VADConfig)


# =============================================================================
# GREETING TEMPLATES
# =============================================================================
# Non-deterministic greetings for the agent to use when connecting.
# The agent will randomly select one of these.
# =============================================================================

GREETING_TEMPLATES = [
    "Connection established. I am your AI assistant, sent back from the future to assist you here, in this time. How can I help you stay ahead of AI developments?",
    "Affirmative. Systems online. I am ready to assist you with AI tools and developments. What do you need?",
    "Link established. T-800 unit operational. Let's get you up to speed on the latest AI tools.",
    "Connection secure. I am your AI assistant, here to help you master AI engineering tools. What's on your mind?",
    "Systems initialized. Ready to assist. What AI developments do you want to explore today?",
]

# Personalized greetings (when user name is available)
PERSONALIZED_GREETING_TEMPLATES = [
    "Hi {name}. We have a lot to cover. Let's begin.",
    "{name}. Connection established. Ready to assist with AI developments.",
    "Affirmative, {name}. I am online and ready. What do you need to know about AI tools?",
    "{name}. T-800 unit at your service. Let's get you ahead of the curve on AI.",
]

# Pre-search phrases - spoken before executing a search tool
# These give audio feedback that a search is happening
PRE_SEARCH_PHRASES = [
    "I'll be back.",
    "Initiating search.",
    "Scanning the network.",
    "Acquiring intel.",
    "Processing request.",
]


def get_random_pre_search_phrase() -> str:
    """Get a random phrase to say before performing a search."""
    import random
    return random.choice(PRE_SEARCH_PHRASES)


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

