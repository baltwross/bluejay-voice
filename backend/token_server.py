"""
Bluejay Terminator - Token Server

A simple FastAPI server that generates LiveKit access tokens for the frontend.
This server runs alongside the main agent and provides:
- Token generation endpoint for WebRTC connections
- Document ingestion endpoint for RAG
- Document listing endpoint
"""
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit import api

from config import get_config
from rag.indexer import DocumentIndexer
from rag.retriever import DocumentRetriever

# Try to import Tavily for news feed endpoint
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None  # type: ignore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bluejay Terminator API",
    description="Token and document management API for the T-800 voice agent",
    version="1.0.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components (lazy loading)
_indexer: Optional[DocumentIndexer] = None
_retriever: Optional[DocumentRetriever] = None


def get_indexer() -> DocumentIndexer:
    """Get or create the document indexer."""
    global _indexer
    if _indexer is None:
        _indexer = DocumentIndexer()
    return _indexer


def get_retriever() -> DocumentRetriever:
    """Get or create the document retriever."""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class TokenResponse(BaseModel):
    """Response model for token endpoint."""
    token: str
    roomName: str
    serverUrl: str


class IngestUrlRequest(BaseModel):
    """Request model for URL ingestion."""
    type: str  # 'youtube', 'web', 'pdf'
    url: str


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    id: str
    title: str
    sourceType: str
    sourceUrl: Optional[str]
    ingestedAt: str
    chunkCount: int


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: list


class TranscriptMessageRequest(BaseModel):
    """Request model for a single transcript message."""
    id: str
    sender: str  # 'user' or 'agent'
    text: str
    timestamp: str  # ISO format
    isFinal: bool


class SaveTranscriptRequest(BaseModel):
    """Request model for saving a transcript."""
    roomName: str
    messages: list[TranscriptMessageRequest]
    startTime: str  # ISO format
    endTime: str  # ISO format


class SaveTranscriptResponse(BaseModel):
    """Response model for saved transcript."""
    filename: str
    messageCount: int
    savedAt: str


class NewsFeedRequest(BaseModel):
    """Request model for news feed search."""
    query: str = "AI tools for software engineering"
    max_results: int = 5


class NewsItem(BaseModel):
    """Model for a news item."""
    title: str
    url: str
    content: str
    score: float


class NewsFeedResponse(BaseModel):
    """Response model for news feed."""
    query: str
    results: list[NewsItem]
    count: int
    timestamp: str


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/api/info")
async def api_info():
    """
    Get API information and available endpoints.
    
    Useful for testing and documentation purposes.
    """
    return {
        "name": "Bluejay Terminator API",
        "version": "1.0.0",
        "description": "Token and document management API for the T-800 voice agent",
        "endpoints": {
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Health check with component status"
            },
            "token": {
                "method": "POST",
                "path": "/api/token",
                "description": "Generate LiveKit access token for WebRTC connection"
            },
            "ingest_url": {
                "method": "POST",
                "path": "/api/ingest",
                "description": "Ingest content from URL (YouTube, web, PDF)",
                "body": {"type": "youtube|web|pdf", "url": "string"}
            },
            "ingest_file": {
                "method": "POST",
                "path": "/api/ingest/file",
                "description": "Upload and ingest file (PDF, DOCX, TXT)",
                "body": "multipart/form-data with file"
            },
            "list_documents": {
                "method": "GET",
                "path": "/api/documents",
                "description": "List all ingested documents"
            },
            "news_feed": {
                "method": "GET",
                "path": "/api/newsfeed",
                "description": "Get AI tools news feed (Tavily integration)",
                "query_params": {"query": "string", "max_results": "int"}
            },
            "save_transcript": {
                "method": "POST",
                "path": "/api/transcripts",
                "description": "Save conversation transcript"
            },
            "api_info": {
                "method": "GET",
                "path": "/api/info",
                "description": "Get API information and endpoints"
            }
        },
        "note": "Voice interaction features (STT, TTS, conversation) are accessed via LiveKit WebRTC, not HTTP endpoints"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed status information.
    
    Returns status of various components:
    - API service status
    - RAG system availability
    - Tavily news feed availability
    """
    try:
        config = get_config()
        health_status = {
            "status": "healthy",
            "service": "bluejay-terminator-api",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "healthy",
                "rag_indexer": "available",
                "rag_retriever": "available",
            }
        }
        
        # Check Tavily availability
        if TAVILY_AVAILABLE and config.tavily.api_key:
            health_status["components"]["news_feed"] = "available"
        else:
            health_status["components"]["news_feed"] = "unavailable"
        
        # Check LiveKit config
        if config.livekit.api_key and config.livekit.api_secret and config.livekit.ws_url:
            health_status["components"]["livekit"] = "configured"
        else:
            health_status["components"]["livekit"] = "not_configured"
        
        # Check OpenAI config
        if config.openai.api_key:
            health_status["components"]["openai"] = "configured"
        else:
            health_status["components"]["openai"] = "not_configured"
        
        return health_status
    except Exception as e:
        return {
            "status": "degraded",
            "service": "bluejay-terminator-api",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.post("/api/token", response_model=TokenResponse)
async def create_token():
    """
    Generate a LiveKit access token for the frontend.
    
    This creates a room and returns credentials for the user to join.
    The agent will automatically join the same room.
    """
    try:
        config = get_config()
        
        # Generate unique room name
        room_name = f"terminator-{uuid.uuid4().hex[:8]}"
        
        # Generate participant identity
        participant_identity = f"user-{uuid.uuid4().hex[:8]}"
        
        # Create access token with grants using the new API pattern
        jwt_token = (
            api.AccessToken(
                api_key=config.livekit.api_key,
                api_secret=config.livekit.api_secret,
            )
            .with_identity(participant_identity)
            .with_name("Human Operator")
            .with_grants(
                api.VideoGrants(
                    room=room_name,
                    room_join=True,
                    room_create=True,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
        
        logger.info(f"Generated token for room: {room_name}, participant: {participant_identity}")
        
        return TokenResponse(
            token=jwt_token,
            roomName=room_name,
            serverUrl=config.livekit.ws_url,
        )
        
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_content(request: IngestUrlRequest):
    """
    Ingest content from a URL into the RAG system.
    
    Supports:
    - YouTube videos (extracts transcript)
    - Web articles (extracts content)
    - PDF URLs (downloads and extracts text)
    
    Content type aliases:
    - "web", "web_article", "article" -> web ingestion
    - "youtube", "yt" -> YouTube ingestion
    - "pdf" -> PDF ingestion
    - "url" -> auto-detect type from URL
    """
    try:
        indexer = get_indexer()
        
        # Normalize content type (handle aliases)
        content_type = request.type.lower().strip()
        
        # Handle aliases
        if content_type in ["web_article", "article", "url"]:
            content_type = "web"
        elif content_type == "yt":
            content_type = "youtube"
        
        # Auto-detect type from URL if type is "url" or not specified
        if content_type == "url" or not content_type:
            url_lower = request.url.lower()
            if "youtube.com" in url_lower or "youtu.be" in url_lower:
                content_type = "youtube"
            elif url_lower.endswith(".pdf") or "pdf" in url_lower:
                content_type = "pdf"
            else:
                content_type = "web"
        
        # Map request type to indexer method
        if content_type == "youtube":
            result = indexer.index_youtube(request.url)
        elif content_type == "web":
            result = indexer.index_web(request.url)
        elif content_type == "pdf":
            result = indexer.index_pdf(request.url)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {request.type}")
        
        # Ensure result has required fields
        if not result.get("success", True):
            # Handle error response from indexer
            error_msg = result.get("error", "Unknown error during ingestion")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {error_msg}")
        
        # Ensure document_id exists
        if "document_id" not in result:
            result["document_id"] = str(uuid.uuid4())[:8]
        
        logger.info(f"Ingested {content_type} content: {result.get('title', 'Unknown')}")
        
        # Ensure we have a valid title (handle None, empty, and "Untitled" from loaders)
        doc_title = result.get("title")
        if not doc_title or doc_title in ["Untitled", "Unknown", ""]:
            doc_title = f"Shared {content_type.capitalize()} Content"
        
        # Ensure chunk_count exists
        chunk_count = result.get("chunk_count") or result.get("num_chunks", 0)
        
        # Ensure we have a document_id for the response
        doc_id = result.get("document_id")
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]
        
        return IngestResponse(
            id=doc_id,
            title=doc_title,
            sourceType=content_type,  # Use normalized type
            sourceUrl=request.url,
            ingestedAt=datetime.now().isoformat(),
            chunkCount=chunk_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a file upload into the RAG system.
    
    Supports:
    - PDF files
    - DOCX files
    - TXT files
    """
    try:
        indexer = get_indexer()
        
        # Determine file type
        filename = file.filename or "upload"
        content_type = file.content_type or ""
        
        # Read file content
        content = await file.read()
        
        # Route to appropriate indexer method
        if filename.endswith(".pdf") or "pdf" in content_type:
            result = indexer.index_pdf_bytes(content, filename)
            source_type = "pdf"
        elif filename.endswith(".docx") or "word" in content_type:
            result = indexer.index_docx_bytes(content, filename)
            source_type = "docx"
        elif filename.endswith(".txt") or "text" in content_type:
            result = indexer.index_text(content.decode("utf-8"), filename)
            source_type = "text"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        
        logger.info(f"Ingested file: {filename}")
        
        # Ensure document_id exists
        if "document_id" not in result:
            result["document_id"] = str(uuid.uuid4())[:8]
        
        # Ensure chunk_count exists
        chunk_count = result.get("chunk_count") or result.get("num_chunks", 0)
        
        # Ensure we have a document_id for the response
        doc_id = result.get("document_id")
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]
        
        return IngestResponse(
            id=doc_id,
            title=result.get("title", filename),
            sourceType=source_type,
            sourceUrl=None,
            ingestedAt=datetime.now().isoformat(),
            chunkCount=chunk_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload", response_model=IngestResponse)
async def ingest_file_compat(file: UploadFile = File(...)):
    """
    Compatibility endpoint for /api/documents/upload.
    Redirects to /api/ingest/file for backward compatibility.
    """
    return await ingest_file(file)


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents in the RAG knowledge base.
    """
    try:
        retriever = get_retriever()
        documents = retriever.list_documents()
        
        return DocumentListResponse(
            documents=[
                {
                    "id": doc.get("document_id", ""),
                    "title": doc.get("title", "Untitled"),
                    "sourceType": doc.get("source_type", "unknown"),
                    "sourceUrl": doc.get("source_url"),
                    "chunkCount": doc.get("total_chunks", 0),
                }
                for doc in documents
            ]
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/newsfeed", response_model=NewsFeedResponse)
async def get_news_feed(query: str = "AI tools for software engineering", max_results: int = 5):
    """
    Get AI tools news feed using Tavily search.
    
    This endpoint provides programmatic access to the news feed functionality
    that is normally accessed through voice conversation.
    
    Args:
        query: Search query (defaults to AI tools for software engineering)
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        NewsFeedResponse with filtered results relevant to software engineering AI tools
    """
    if not TAVILY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Tavily package not installed. Install with: pip install tavily-python"
        )
    
    try:
        config = get_config()
        if not config.tavily.api_key:
            raise HTTPException(
                status_code=503,
                detail="Tavily API key not configured. Set TAVILY_API_KEY environment variable."
            )
        
        tavily_client = TavilyClient(api_key=config.tavily.api_key)
        
        # Perform search with news topic for recent results
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            topic="news",
            max_results=max_results,
            include_answer=True,
            time_range="week",  # Last week for freshness
        )
        
        # Extract and filter results (same logic as agent.py)
        results = []
        for item in response.get("results", []):
            title = item.get("title", "").lower()
            content = item.get("content", "").lower()
            url = item.get("url", "")
            
            # Filter for engineering-relevant content
            engineering_keywords = [
                "code", "coding", "developer", "programming", "software",
                "engineering", "tool", "assistant", "cursor", "copilot",
                "claude", "gpt", "gemini", "llama", "mcp", "agent",
                "productivity", "api", "sdk", "terminal", "ide", "editor",
            ]
            
            is_relevant = any(kw in title or kw in content for kw in engineering_keywords)
            
            # Trusted sources
            trusted_sources = [
                "github", "anthropic", "openai", "google", "microsoft",
                "vercel", "cursor", "livekit", "langchain", "hacker",
                "techcrunch", "theverge", "ars", "wired",
            ]
            is_trusted = any(src in url.lower() for src in trusted_sources)
            
            if is_relevant or is_trusted:
                results.append(NewsItem(
                    title=item.get("title", "Unknown"),
                    url=url,
                    content=item.get("content", "")[:300],  # Truncate
                    score=item.get("score", 0.0),
                ))
        
        logger.info(f"News feed search: '{query}' returned {len(results)} results")
        
        return NewsFeedResponse(
            query=query,
            results=results,
            count=len(results),
            timestamp=datetime.now().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch news feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcripts", response_model=SaveTranscriptResponse)
async def save_transcript(request: SaveTranscriptRequest):
    """
    Save a conversation transcript for debugging purposes.
    
    Transcripts are saved as JSON files in the transcripts/ directory
    with format: {timestamp}_{roomName}.json
    """
    try:
        # Determine transcripts directory (relative to project root)
        # token_server.py is in backend/, so go up one level
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "transcripts"
        
        # Ensure directory exists
        transcripts_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp and room name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_room_name = request.roomName.replace("-", "_").replace(" ", "_")
        filename = f"{timestamp}_{safe_room_name}.json"
        filepath = transcripts_dir / filename
        
        # Format transcript data
        transcript_data = {
            "roomName": request.roomName,
            "startTime": request.startTime,
            "endTime": request.endTime,
            "savedAt": datetime.now().isoformat(),
            "messageCount": len(request.messages),
            "messages": [
                {
                    "id": msg.id,
                    "sender": msg.sender,
                    "text": msg.text,
                    "timestamp": msg.timestamp,
                    "isFinal": msg.isFinal,
                }
                for msg in request.messages
            ],
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcript: {filename} ({len(request.messages)} messages)")
        
        return SaveTranscriptResponse(
            filename=filename,
            messageCount=len(request.messages),
            savedAt=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("TOKEN_SERVER_PORT", "8080"))
    logger.info(f"Starting token server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

