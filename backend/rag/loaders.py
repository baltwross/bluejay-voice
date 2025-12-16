"""
Document Loaders for multiple content types.
Supports: PDF, YouTube, Web URLs, and text files.
"""
import re
import logging
from typing import List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders import YoutubeLoader
import trafilatura

logger = logging.getLogger(__name__)


SourceType = Literal["pdf", "youtube", "web", "text"]


@dataclass
class LoaderResult:
    """Result from a document loader."""
    documents: List[Document]
    source_type: SourceType
    source_url: str
    title: str
    ingested_at: datetime


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders based on content type."""
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
    ]
    
    @classmethod
    def detect_source_type(cls, source: str) -> SourceType:
        """Detect the source type from a URL or file path."""
        source_lower = source.lower()
        
        # Check if it's a YouTube URL
        for pattern in cls.YOUTUBE_PATTERNS:
            if re.match(pattern, source):
                return "youtube"
        
        # Check if it's a PDF file
        if source_lower.endswith(".pdf") or "/pdf" in source_lower:
            return "pdf"
        
        # Check if it's a local file
        if Path(source).exists():
            if source_lower.endswith(".pdf"):
                return "pdf"
            return "text"
        
        # Check if it's a web URL
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            return "web"
        
        # Default to text
        return "text"
    
    @classmethod
    def load(
        cls,
        source: str,
        source_type: Optional[SourceType] = None,
        title: Optional[str] = None,
    ) -> LoaderResult:
        """
        Load documents from a source.
        
        Args:
            source: URL or file path to load
            source_type: Optional explicit source type (auto-detected if not provided)
            title: Optional title for the document
            
        Returns:
            LoaderResult with documents and metadata
        """
        if source_type is None:
            source_type = cls.detect_source_type(source)
        
        logger.info(f"Loading {source_type} from: {source}")
        
        loader_methods = {
            "pdf": cls._load_pdf,
            "youtube": cls._load_youtube,
            "web": cls._load_web,
            "text": cls._load_text,
        }
        
        loader_method = loader_methods.get(source_type)
        if not loader_method:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        documents, detected_title = loader_method(source)
        
        # Use provided title or detected title
        final_title = title or detected_title or source
        
        # Add metadata to all documents
        for doc in documents:
            doc.metadata.update({
                "source_type": source_type,
                "source_url": source,
                "title": final_title,
                "ingested_at": datetime.now().isoformat(),
            })
        
        return LoaderResult(
            documents=documents,
            source_type=source_type,
            source_url=source,
            title=final_title,
            ingested_at=datetime.now(),
        )
    
    @classmethod
    def _load_pdf(cls, file_path: str) -> tuple[List[Document], Optional[str]]:
        """Load PDF document."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Try to extract title from first page or filename
            title = Path(file_path).stem.replace("_", " ").replace("-", " ").title()
            
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents, title
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    @classmethod
    def _load_youtube(cls, url: str) -> tuple[List[Document], Optional[str]]:
        """Load YouTube video transcript."""
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,
                language=["en"],
                translation="en",
            )
            documents = loader.load()
            
            # Extract title from video info
            title = None
            if documents and "title" in documents[0].metadata:
                title = documents[0].metadata["title"]
            
            logger.info(f"Loaded transcript from YouTube video: {title}")
            return documents, title
            
        except Exception as e:
            logger.error(f"Error loading YouTube transcript: {e}")
            raise
    
    @classmethod
    def _load_web(cls, url: str) -> tuple[List[Document], Optional[str]]:
        """Load web page content using Trafilatura for better extraction."""
        try:
            # First try Trafilatura for cleaner extraction
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
                metadata = trafilatura.extract_metadata(downloaded)
                
                if text:
                    title = metadata.title if metadata else None
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "title": title or url,
                        }
                    )
                    logger.info(f"Loaded web content via Trafilatura: {title}")
                    return [doc], title
            
            # Fallback to WebBaseLoader
            logger.info("Falling back to WebBaseLoader")
            loader = WebBaseLoader(web_paths=[url])
            documents = loader.load()
            
            # Try to extract title from content
            title = url  # Default to URL
            if documents and documents[0].page_content:
                # Try to find title in first 500 chars
                first_line = documents[0].page_content.strip().split('\n')[0][:200]
                if first_line:
                    title = first_line
            
            logger.info(f"Loaded {len(documents)} documents from web")
            return documents, title
            
        except Exception as e:
            logger.error(f"Error loading web content: {e}")
            raise
    
    @classmethod
    def _load_text(cls, content_or_path: str) -> tuple[List[Document], Optional[str]]:
        """Load text content or text file."""
        try:
            path = Path(content_or_path)
            if path.exists() and path.is_file():
                # Load from file
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                title = path.stem.replace("_", " ").replace("-", " ").title()
            else:
                # Treat as raw text content
                content = content_or_path
                title = content[:50] + "..." if len(content) > 50 else content
            
            doc = Document(
                page_content=content,
                metadata={"source": str(content_or_path)}
            )
            
            logger.info(f"Loaded text content: {len(content)} characters")
            return [doc], title
            
        except Exception as e:
            logger.error(f"Error loading text: {e}")
            raise


