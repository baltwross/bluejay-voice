"""
Document Indexer - Handles chunking, embedding, and storage in ChromaDB.
"""
import logging
from typing import List, Optional
from datetime import datetime
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import RAGConfig, get_config
from .loaders import DocumentLoaderFactory, LoaderResult

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Handles document ingestion pipeline:
    1. Load documents from various sources
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the indexer with configuration."""
        self.config = config or get_config()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            add_start_index=True,  # Track position in original document
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )
        
        # Initialize or load vector store
        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory,
        )
        
        logger.info(f"DocumentIndexer initialized with persist_directory: {self.config.persist_directory}")
    
    def ingest(
        self,
        source: str,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> dict:
        """
        Ingest a document from a source (URL or file path).
        
        Args:
            source: URL or file path to ingest
            title: Optional title for the document
            document_id: Optional unique ID for the document
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid.uuid4())[:8]
            
            logger.info(f"Starting ingestion for: {source} (ID: {document_id})")
            
            # Load documents
            loader_result = DocumentLoaderFactory.load(source, title=title)
            
            # Split into chunks
            chunks = self._split_documents(loader_result.documents, document_id)
            
            # Add to vector store
            ids = self._store_chunks(chunks, document_id)
            
            result = {
                "success": True,
                "document_id": document_id,
                "title": loader_result.title,
                "source_type": loader_result.source_type,
                "source_url": loader_result.source_url,
                "num_chunks": len(chunks),
                "chunk_ids": ids,
                "ingested_at": datetime.now().isoformat(),
            }
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from: {source}")
            return result
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": source,
            }
    
    def ingest_text(
        self,
        text: str,
        title: str,
        document_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Ingest raw text content directly.
        
        Args:
            text: Raw text content to ingest
            title: Title for the document
            document_id: Optional unique ID
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            if document_id is None:
                document_id = str(uuid.uuid4())[:8]
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "source_type": "text",
                    "source_url": f"text://{document_id}",
                    "title": title,
                    "ingested_at": datetime.now().isoformat(),
                    **(metadata or {}),
                }
            )
            
            # Split into chunks
            chunks = self._split_documents([doc], document_id)
            
            # Store chunks
            ids = self._store_chunks(chunks, document_id)
            
            return {
                "success": True,
                "document_id": document_id,
                "title": title,
                "source_type": "text",
                "num_chunks": len(chunks),
                "chunk_ids": ids,
                "ingested_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _split_documents(
        self,
        documents: List[Document],
        document_id: str,
    ) -> List[Document]:
        """Split documents into chunks with proper metadata."""
        chunks = self.text_splitter.split_documents(documents)
        
        # Add document_id and chunk_index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document_id
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.info(f"Split into {len(chunks)} chunks (avg {sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0} chars)")
        return chunks
    
    def _store_chunks(
        self,
        chunks: List[Document],
        document_id: str,
    ) -> List[str]:
        """Store chunks in the vector store."""
        # Generate unique IDs for each chunk
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Add to vector store
        self.vector_store.add_documents(chunks, ids=ids)
        
        logger.info(f"Stored {len(chunks)} chunks in vector store")
        return ids
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the vector store."""
        try:
            # Get all chunk IDs for this document
            # Note: Chroma doesn't have a direct "delete by metadata" in langchain wrapper
            # We'll need to delete by ID pattern
            collection = self.vector_store._collection
            
            # Get documents with matching document_id
            results = collection.get(
                where={"document_id": document_id}
            )
            
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document: {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def list_documents(self) -> List[dict]:
        """List all unique documents in the vector store."""
        try:
            collection = self.vector_store._collection
            
            # Get all documents
            results = collection.get(include=["metadatas"])
            
            # Extract unique documents
            documents = {}
            for metadata in results.get("metadatas", []):
                if metadata:
                    doc_id = metadata.get("document_id")
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "title": metadata.get("title", "Unknown"),
                            "source_type": metadata.get("source_type", "unknown"),
                            "source_url": metadata.get("source_url", ""),
                            "ingested_at": metadata.get("ingested_at", ""),
                        }
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            documents = self.list_documents()
            
            return {
                "total_chunks": count,
                "total_documents": len(documents),
                "persist_directory": self.config.persist_directory,
                "collection_name": self.config.collection_name,
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

