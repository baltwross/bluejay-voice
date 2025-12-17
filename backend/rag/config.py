"""
RAG Configuration Settings - Redirects to central config
"""
from typing import Optional

# Try to import from central config
try:
    from config import RAGConfig, get_config as get_main_config
except ImportError:
    # Fallback for when running as a package or from different root
    try:
        from backend.config import RAGConfig, get_config as get_main_config
    except ImportError:
        raise ImportError("Could not import RAGConfig from backend.config")

# Default configuration instance (for backwards compatibility if accessed directly)
default_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the RAG configuration from the central config."""
    global default_config
    if default_config is None:
        # Get the global config and extract RAG config
        main_config = get_main_config()
        default_config = main_config.rag
    return default_config
