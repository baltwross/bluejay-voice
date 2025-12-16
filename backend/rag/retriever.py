"""
Document Retriever - Handles semantic search and context retrieval.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import RAGConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    query: str
    documents: List[Document]
    scores: List[float]
    context: str
    sources: List[Dict[str, Any]]


class DocumentRetriever:
    """
    Handles document retrieval from ChromaDB.
    Supports filtering by document_id for "Active Document" context.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the retriever with configuration."""
        self.config = config or get_config()
        
        # Initialize embeddings model
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )
        
        # Load vector store
        self.vector_store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory,
        )
        
        logger.info(f"DocumentRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default from config)
            document_id: Filter to specific document (Active Document context)
            source_type: Filter by source type (pdf, youtube, web, text)
            score_threshold: Minimum similarity score
            
        Returns:
            RetrievalResult with documents, scores, and formatted context
        """
        k = k or self.config.retrieval_k
        score_threshold = score_threshold or self.config.retrieval_score_threshold
        
        logger.info(f"Retrieving for query: '{query[:50]}...' (k={k})")
        
        # Build filter if specified
        filter_dict = self._build_filter(document_id, source_type)
        
        # Perform similarity search
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        documents = self.vector_store.similarity_search(query, **search_kwargs)
        
        # Use placeholder scores (Chroma L2 distance isn't ideal for thresholding)
        scores = [1.0] * len(documents)
        
        if not documents:
            logger.info("No results found")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                context="",
                sources=[],
            )
        
        # Build context string for LLM
        context = self._build_context(documents)
        
        # Extract source information for citation
        sources = self._extract_sources(documents, scores)
        
        logger.info(f"Retrieved {len(documents)} documents")
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            context=context,
            sources=sources,
        )
    
    def retrieve_for_reading(
        self,
        document_id: str,
        start_chunk: int = 0,
        num_chunks: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve sequential chunks for reading mode.
        Used when the agent needs to read a document aloud.
        
        Args:
            document_id: The document to read
            start_chunk: Starting chunk index
            num_chunks: Number of chunks to retrieve
            
        Returns:
            RetrievalResult with sequential chunks
        """
        logger.info(f"Retrieving chunks {start_chunk}-{start_chunk + num_chunks} for document: {document_id}")
        
        try:
            collection = self.vector_store._collection
            
            # Get chunks in order by chunk_index
            results = collection.get(
                where={
                    "$and": [
                        {"document_id": {"$eq": document_id}},
                        {"chunk_index": {"$gte": start_chunk}},
                        {"chunk_index": {"$lt": start_chunk + num_chunks}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            
            if not results or not results.get("documents"):
                logger.warning(f"No chunks found for document: {document_id}")
                return RetrievalResult(
                    query=f"read:{document_id}",
                    documents=[],
                    scores=[],
                    context="",
                    sources=[],
                )
            
            # Convert to Document objects and sort by chunk_index
            docs_with_meta = list(zip(
                results["documents"],
                results["metadatas"],
            ))
            
            docs_with_meta.sort(key=lambda x: x[1].get("chunk_index", 0))
            
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in docs_with_meta
            ]
            
            context = self._build_context(documents)
            sources = self._extract_sources(documents, [1.0] * len(documents))
            
            return RetrievalResult(
                query=f"read:{document_id}",
                documents=documents,
                scores=[1.0] * len(documents),
                context=context,
                sources=sources,
            )
            
        except Exception as e:
            logger.error(f"Error retrieving for reading: {e}")
            return RetrievalResult(
                query=f"read:{document_id}",
                documents=[],
                scores=[],
                context="",
                sources=[],
            )
    
    def find_document_by_title(
        self,
        title_query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a document by partial title match.
        Used to locate documents by name during conversation.
        
        Args:
            title_query: Partial title to search for
            
        Returns:
            Document metadata if found, None otherwise.
            Includes total_chunks count for reading mode.
        """
        try:
            collection = self.vector_store._collection
            
            # Get all unique documents
            results = collection.get(include=["metadatas"])
            
            title_query_lower = title_query.lower()
            best_match = None
            best_score = 0
            
            # Track chunks per document
            doc_chunks: Dict[str, int] = {}
            doc_metadata: Dict[str, Dict[str, Any]] = {}
            
            for metadata in results.get("metadatas", []):
                if metadata:
                    doc_id = metadata.get("document_id")
                    if doc_id:
                        # Count chunks per document
                        doc_chunks[doc_id] = doc_chunks.get(doc_id, 0) + 1
                        
                        # Store first metadata entry for each document
                        if doc_id not in doc_metadata:
                            doc_metadata[doc_id] = metadata
            
            # Find best title match
            for doc_id, metadata in doc_metadata.items():
                doc_title = metadata.get("title", "").lower()
                
                # Simple substring matching score
                if title_query_lower in doc_title:
                    score = len(title_query_lower) / len(doc_title) if doc_title else 0
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "document_id": doc_id,
                            "title": metadata.get("title"),
                            "source_type": metadata.get("source_type"),
                            "source_url": metadata.get("source_url"),
                            "total_chunks": doc_chunks.get(doc_id, 0),
                        }
            
            if best_match:
                logger.info(
                    f"Found document '{best_match['title']}' for query '{title_query}' "
                    f"({best_match['total_chunks']} chunks)"
                )
            else:
                logger.info(f"No document found for query '{title_query}'")
                
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding document by title: {e}")
            return None
    
    def _build_filter(
        self,
        document_id: Optional[str],
        source_type: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build a filter dictionary for Chroma queries."""
        conditions = []
        
        if document_id:
            conditions.append({"document_id": {"$eq": document_id}})
        
        if source_type:
            conditions.append({"source_type": {"$eq": source_type}})
        
        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build a context string from retrieved documents."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("title", "Unknown")
            chunk_info = ""
            if "chunk_index" in doc.metadata:
                chunk_info = f" (Section {doc.metadata['chunk_index'] + 1})"
            
            context_parts.append(
                f"[Source {i}: {source}{chunk_info}]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _extract_sources(
        self,
        documents: List[Document],
        scores: List[float],
    ) -> List[Dict[str, Any]]:
        """Extract source information for citation."""
        sources = []
        seen = set()
        
        for doc, score in zip(documents, scores):
            doc_id = doc.metadata.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                sources.append({
                    "document_id": doc_id,
                    "title": doc.metadata.get("title", "Unknown"),
                    "source_type": doc.metadata.get("source_type", "unknown"),
                    "source_url": doc.metadata.get("source_url", ""),
                    "relevance_score": round(score, 3),
                })
        
        return sources
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents available in the knowledge base.
        
        Returns:
            List of document metadata dicts with document_id, title, source_type, total_chunks.
        """
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["metadatas"])
            
            # Aggregate chunks per document
            doc_chunks: Dict[str, int] = {}
            doc_metadata: Dict[str, Dict[str, Any]] = {}
            
            for metadata in results.get("metadatas", []):
                if metadata:
                    doc_id = metadata.get("document_id")
                    if doc_id:
                        doc_chunks[doc_id] = doc_chunks.get(doc_id, 0) + 1
                        if doc_id not in doc_metadata:
                            doc_metadata[doc_id] = metadata
            
            documents = []
            for doc_id, metadata in doc_metadata.items():
                documents.append({
                    "document_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "total_chunks": doc_chunks.get(doc_id, 0),
                })
            
            logger.info(f"Listed {len(documents)} documents in knowledge base")
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def as_retriever(self, **kwargs):
        """
        Return this retriever as a LangChain Retriever interface.
        Useful for integration with LangChain chains.
        """
        return self.vector_store.as_retriever(
            search_kwargs={
                "k": kwargs.get("k", self.config.retrieval_k),
                **kwargs,
            }
        )

