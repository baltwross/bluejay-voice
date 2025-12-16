"""
Bluejay Terminator - Token Server

A simple FastAPI server that generates LiveKit access tokens for the frontend.
This server runs alongside the main agent and provides:
- Token generation endpoint for WebRTC connections
- Document ingestion endpoint for RAG
- Document listing endpoint
"""
import os
import uuid
import logging
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


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bluejay-terminator-api"}


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
    """
    try:
        indexer = get_indexer()
        
        # Map request type to indexer method
        if request.type == "youtube":
            result = indexer.index_youtube(request.url)
        elif request.type == "web":
            result = indexer.index_web(request.url)
        elif request.type == "pdf":
            result = indexer.index_pdf(request.url)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {request.type}")
        
        logger.info(f"Ingested {request.type} content: {result.get('title', 'Unknown')}")
        
        return IngestResponse(
            id=result.get("document_id", str(uuid.uuid4())),
            title=result.get("title", "Untitled"),
            sourceType=request.type,
            sourceUrl=request.url,
            ingestedAt=datetime.now().isoformat(),
            chunkCount=result.get("chunk_count", 0),
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
        
        return IngestResponse(
            id=result.get("document_id", str(uuid.uuid4())),
            title=result.get("title", filename),
            sourceType=source_type,
            sourceUrl=None,
            ingestedAt=datetime.now().isoformat(),
            chunkCount=result.get("chunk_count", 0),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

