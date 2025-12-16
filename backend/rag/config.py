"""
RAG Configuration Settings
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding settings (text-embedding-3-large is OpenAI's best, recommended by LangChain)
    embedding_model: str = "text-embedding-3-large"
    
    # Vector store settings
    collection_name: str = "bluejay_knowledge_base"
    persist_directory: str = field(default_factory=lambda: str(
        Path(__file__).parent.parent / "chroma_db"
    ))
    
    # Retrieval settings
    retrieval_k: int = 4  # Number of documents to retrieve
    retrieval_score_threshold: float = 0.0  # Minimum similarity score (0 = accept all, Chroma uses L2 distance)
    
    # OpenAI API key (from environment)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for RAG system")
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = None


def get_config() -> RAGConfig:
    """Get or create the default RAG configuration."""
    global default_config
    if default_config is None:
        default_config = RAGConfig()
    return default_config

