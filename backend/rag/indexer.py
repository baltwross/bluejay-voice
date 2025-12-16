import os
import logging
from typing import List, Optional
from urllib.parse import urlparse

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"

# Initialize Settings
Settings.embedding_model = OpenAIEmbedding()
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

def get_vector_store():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def index_document(source: str, metadata: Optional[dict] = None) -> None:
    """
    Index a document from a file path or URL.
    """
    logger.info(f"Indexing document from source: {source}")
    
    documents = []
    
    # Determine source type
    if _is_url(source):
        if "youtube.com" in source or "youtu.be" in source:
            documents = _load_youtube(source)
        else:
            documents = _load_web(source)
    else:
        # Assume file path
        if source.endswith(".pdf"):
             # SimpleDirectoryReader with explicit file handling
            reader = SimpleDirectoryReader(input_files=[source])
            documents = reader.load_data()
        elif source.endswith(".docx"):
            reader = SimpleDirectoryReader(input_files=[source])
            documents = reader.load_data()
        else:
            # Try generic loader or treat as text
            reader = SimpleDirectoryReader(input_files=[source])
            documents = reader.load_data()

    if not documents:
        logger.warning(f"No documents loaded from {source}")
        return

    # Enrich metadata
    for doc in documents:
        if metadata:
            doc.metadata.update(metadata)
        doc.metadata["source"] = source
        # Add basic title if not provided
        if "title" not in doc.metadata:
            doc.metadata["title"] = os.path.basename(source) if not _is_url(source) else source

    # Create/Get Vector Store
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    index.storage_context.persist(persist_dir=CHROMA_DB_PATH) # Chroma handles persistence but good practice
    
    logger.info(f"Successfully indexed {len(documents)} chunks from {source}")

def _is_url(source: str) -> bool:
    try:
        result = urlparse(source)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def _load_youtube(url: str) -> List[Document]:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        video_id = url.split("v=")[-1] # simplistic extraction
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript])
        return [Document(text=text, metadata={"source_type": "youtube", "source_url": url})]
    except Exception as e:
        logger.error(f"Failed to load YouTube transcript: {e}")
        return []

def _load_web(url: str) -> List[Document]:
    import trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        if text:
             return [Document(text=text, metadata={"source_type": "web", "source_url": url})]
        return []
    except Exception as e:
         logger.error(f"Failed to load web page: {e}")
         return []

if __name__ == "__main__":
    # Test
    pass

