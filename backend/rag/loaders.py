"""
Document Loaders for multiple content types.
Supports: PDF, YouTube, Web URLs, and text files.
"""
import os
import re
import ssl
import logging
import urllib.request
from typing import List, Optional, Literal, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import certifi
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders import YoutubeLoader
import trafilatura

logger = logging.getLogger(__name__)

# Set SSL certificate environment variables for all urllib-based libraries
# This fixes macOS SSL certificate verification issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Create SSL context with certifi certificates (fixes macOS SSL issues)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


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
    
    @staticmethod
    def _fetch_youtube_title_oembed(url: str) -> Optional[str]:
        """
        Fetch YouTube video title using the oEmbed API.
        This is the most reliable method as it's an official YouTube API.
        """
        try:
            import json
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            req = urllib.request.Request(oembed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode('utf-8'))
                title = data.get('title', '').strip()
                if title and title not in ['Untitled', 'Unknown', '']:
                    logger.info(f"Got YouTube title from oEmbed API: {title}")
                    return title
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch YouTube title via oEmbed: {e}")
            return None
    
    @staticmethod
    def _fetch_page_title(url: str) -> Optional[str]:
        """
        Fetch the <title> tag from a URL using BeautifulSoup.
        Works for most web pages including YouTube.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=SSL_CONTEXT) as response:
                html = response.read().decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')
                
                # Try <title> tag first
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                    # Clean up common suffixes
                    title = re.sub(r'\s*[-|–]\s*YouTube\s*$', '', title)
                    title = re.sub(r'\s*[-|–]\s*[^-|–]+$', '', title)  # Remove site name suffix
                    if title:
                        return title
                
                # Try Open Graph title
                og_title = soup.find('meta', property='og:title')
                if og_title and og_title.get('content'):
                    return og_title.get('content').strip()
                
                # Try Twitter title
                twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                if twitter_title and twitter_title.get('content'):
                    return twitter_title.get('content').strip()
                    
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch page title from {url}: {e}")
            return None
    
    @staticmethod
    def _extract_youtube_video_id(url: str) -> Optional[str]:
        """Extract video ID from a YouTube URL."""
        patterns = [
            r"(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&?/]|$)",
            r"youtu\.be/([a-zA-Z0-9_-]{11})",
            r"embed/([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
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
        """Load YouTube video transcript with robust title extraction using yt-dlp."""
        title = None
        documents = []
        
        try:
            # Extract video ID and normalize URL (removes extra query params)
            video_id = cls._extract_youtube_video_id(url)
            if video_id:
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
            else:
                clean_url = url
                logger.warning(f"Could not extract video ID from URL: {url}")
            
            # Use yt-dlp to get video info and subtitles
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'subtitlesformat': 'json3',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)
                
                # Validate that info was extracted successfully
                if info is None:
                    raise ValueError(
                        f"Failed to extract video information from YouTube URL: {url}"
                    )
                
                # Get title from yt-dlp
                if info.get('title'):
                    title = info['title']
                    logger.info(f"Got title from yt-dlp: {title}")
                
                # Try to get subtitles/captions
                transcript_text = None
                
                # Check for manual subtitles first
                subtitles = info.get('subtitles', {})
                auto_captions = info.get('automatic_captions', {})
                
                # Try English subtitles first
                sub_data = None
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in subtitles:
                        sub_data = subtitles[lang]
                        break
                    if lang in auto_captions:
                        sub_data = auto_captions[lang]
                        break
                
                if sub_data:
                    # Get the JSON3 or VTT format URL
                    sub_url = None
                    for fmt in sub_data:
                        if fmt.get('ext') in ['json3', 'vtt', 'srv1']:
                            sub_url = fmt.get('url')
                            break
                    
                    if sub_url:
                        # Fetch and parse subtitles
                        req = urllib.request.Request(sub_url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
                            sub_content = response.read().decode('utf-8')
                            
                            # Parse based on format
                            if 'json3' in sub_url or sub_content.strip().startswith('{'):
                                import json
                                sub_json = json.loads(sub_content)
                                events = sub_json.get('events', [])
                                texts = []
                                for event in events:
                                    segs = event.get('segs', [])
                                    for seg in segs:
                                        text = seg.get('utf8', '').strip()
                                        if text and text != '\n':
                                            texts.append(text)
                                transcript_text = ' '.join(texts)
                            else:
                                # VTT format - extract text lines
                                lines = sub_content.split('\n')
                                texts = []
                                for line in lines:
                                    line = line.strip()
                                    if line and not line.startswith('WEBVTT') and '-->' not in line and not line.isdigit():
                                        # Remove VTT tags
                                        import re as _re
                                        clean_line = _re.sub(r'<[^>]+>', '', line)
                                        if clean_line:
                                            texts.append(clean_line)
                                transcript_text = ' '.join(texts)
                
                if transcript_text:
                    # Build fallback title: prefer video_id, else use URL fragment
                    fallback_title = (
                        f"YouTube Video ({video_id})" if video_id 
                        else f"YouTube Video (from {url[:50]})"
                    )
                    doc = Document(
                        page_content=transcript_text,
                        metadata={
                            "source": clean_url,
                            "title": title or fallback_title,
                        }
                    )
                    documents = [doc]
                    logger.info(f"Loaded transcript via yt-dlp: {len(transcript_text)} chars")
                else:
                    # No subtitles available - use video description as fallback
                    description = info.get('description', '')
                    if description:
                        # Build fallback title: prefer video_id, else use URL fragment
                        fallback_title = (
                            f"YouTube Video ({video_id})" if video_id 
                            else f"YouTube Video (from {url[:50]})"
                        )
                        display_title = title or fallback_title
                        doc = Document(
                            page_content=f"Video Title: {display_title}\n\nDescription:\n{description}",
                            metadata={
                                "source": clean_url,
                                "title": display_title,
                                "note": "No transcript available, using video description",
                            }
                        )
                        documents = [doc]
                        logger.info(f"No transcript available, using video description: {len(description)} chars")
                    else:
                        raise ValueError("No transcript or description available for this video")
            
            # Last resort: extract video ID and use as title
            if not title:
                if video_id:
                    title = f"YouTube Video ({video_id})"
                    logger.info(f"Using video ID as fallback title: {title}")
            
            # Update document metadata with the title
            if title:
                for doc in documents:
                    doc.metadata["title"] = title
            
            logger.info(f"Loaded transcript from YouTube video: {title or 'Unknown'}")
            return documents, title
            
        except Exception as e:
            logger.error(f"Error loading YouTube transcript: {e}")
            raise
    
    @classmethod
    def _load_web(cls, url: str) -> tuple[List[Document], Optional[str]]:
        """Load web page content using Trafilatura for better extraction."""
        title = None
        
        try:
            # First, try to get title by scraping the page (most reliable)
            logger.info(f"Fetching web page title: {url}")
            title = cls._fetch_page_title(url)
            if title:
                logger.info(f"Extracted title from page: {title}")
            
            # Try Trafilatura for cleaner content extraction
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
                    # Use Trafilatura title as fallback if we didn't get one
                    if not title and metadata and metadata.title:
                        title = metadata.title
                        logger.info(f"Got title from Trafilatura metadata: {title}")
                    
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "title": title or url,
                        }
                    )
                    logger.info(f"Loaded web content via Trafilatura: {title or 'Unknown'}")
                    return [doc], title
            
            # Fallback to WebBaseLoader
            logger.info("Falling back to WebBaseLoader")
            loader = WebBaseLoader(web_paths=[url])
            documents = loader.load()
            
            # If still no title, try to find it in the first line of content
            if not title and documents and documents[0].page_content:
                first_line = documents[0].page_content.strip().split('\n')[0][:200]
                if first_line and len(first_line) > 5:
                    title = first_line
            
            logger.info(f"Loaded {len(documents)} documents from web: {title or 'Unknown'}")
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



