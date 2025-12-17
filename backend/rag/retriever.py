"""
Document Retriever - Handles semantic search and context retrieval.

Supports two retrieval modes:
1. Semantic search (default) - Uses embeddings for conceptual similarity
2. Hybrid search - Combines semantic + BM25 keyword search for better recall

Hybrid search is recommended for queries with specific identifiers like:
- "reference number 15" → matches "[15]"
- "section 3.2" → matches "3.2 Methodology"
- "Chapter 7" → matches "Chapter 7:"
"""
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# BM25 for keyword search
from rank_bm25 import BM25Okapi

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
    
    def hybrid_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        semantic_weight: float = 0.5,
    ) -> RetrievalResult:
        """
        Hybrid retrieval combining semantic search + BM25 keyword search.
        
        This is especially effective for queries with specific identifiers:
        - "reference number 15" → BM25 finds "[15]"
        - "section 3.2" → BM25 finds "3.2"
        - Conceptual queries still work via semantic search
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default from config)
            document_id: Filter to specific document
            semantic_weight: Weight for semantic results (0.0-1.0)
                            BM25 weight = 1.0 - semantic_weight
            
        Returns:
            RetrievalResult with deduplicated documents from both methods
        """
        k = k or self.config.retrieval_k
        bm25_weight = 1.0 - semantic_weight
        
        logger.info(f"Hybrid retrieve: '{query[:50]}...' (k={k}, semantic={semantic_weight:.1f}, bm25={bm25_weight:.1f})")
        
        # Build filter for document_id
        filter_dict = self._build_filter(document_id, None)
        
        # Get ALL documents from the filtered set for BM25
        try:
            collection = self.vector_store._collection
            
            if document_id:
                all_results = collection.get(
                    where={"document_id": {"$eq": document_id}},
                    include=["documents", "metadatas"],
                )
            else:
                all_results = collection.get(include=["documents", "metadatas"])
            
            all_docs = all_results.get("documents", [])
            all_metas = all_results.get("metadatas", [])
            
            if not all_docs:
                logger.info("No documents in collection for hybrid search")
                return RetrievalResult(
                    query=query,
                    documents=[],
                    scores=[],
                    context="",
                    sources=[],
                )
            
            # --- BM25 KEYWORD SEARCH ---
            # Tokenize documents for BM25
            tokenized_docs = [self._tokenize(doc) for doc in all_docs]
            bm25 = BM25Okapi(tokenized_docs)
            
            # Transform query for better keyword matching
            enhanced_query = self._enhance_query_for_keywords(query)
            tokenized_query = self._tokenize(enhanced_query)
            
            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Get top-k from BM25
            bm25_k = min(k, len(all_docs))
            bm25_top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:bm25_k]
            
            bm25_docs = []
            for idx in bm25_top_indices:
                if bm25_scores[idx] > 0:  # Only include if there's some match
                    bm25_docs.append(Document(
                        page_content=all_docs[idx],
                        metadata=all_metas[idx] if idx < len(all_metas) else {},
                    ))
            
            logger.info(f"BM25 found {len(bm25_docs)} keyword matches")
            
        except Exception as e:
            logger.warning(f"BM25 search failed, falling back to semantic only: {e}")
            bm25_docs = []
        
        # --- SEMANTIC SEARCH ---
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        semantic_docs = self.vector_store.similarity_search(query, **search_kwargs)
        logger.info(f"Semantic found {len(semantic_docs)} matches")
        
        # --- MERGE RESULTS ---
        # Use a scoring approach: rank by position with weights
        doc_scores: Dict[str, tuple[Document, float]] = {}
        
        # Score semantic results (higher rank = higher score)
        for rank, doc in enumerate(semantic_docs):
            doc_key = self._doc_key(doc)
            score = semantic_weight * (len(semantic_docs) - rank) / len(semantic_docs) if semantic_docs else 0
            if doc_key in doc_scores:
                doc_scores[doc_key] = (doc, doc_scores[doc_key][1] + score)
            else:
                doc_scores[doc_key] = (doc, score)
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            doc_key = self._doc_key(doc)
            score = bm25_weight * (len(bm25_docs) - rank) / len(bm25_docs) if bm25_docs else 0
            if doc_key in doc_scores:
                doc_scores[doc_key] = (doc, doc_scores[doc_key][1] + score)
            else:
                doc_scores[doc_key] = (doc, score)
        
        # Sort by combined score and take top-k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        documents = [doc for doc, _ in sorted_docs]
        scores = [score for _, score in sorted_docs]
        
        if not documents:
            logger.info("No results found from hybrid search")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                context="",
                sources=[],
            )
        
        context = self._build_context(documents)
        sources = self._extract_sources(documents, scores)
        
        logger.info(f"Hybrid search returned {len(documents)} documents")
        return RetrievalResult(
            query=query,
            documents=documents,
            scores=scores,
            context=context,
            sources=sources,
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search."""
        # Simple whitespace + punctuation tokenization
        text = text.lower()
        # Keep brackets for reference matching like "[15]"
        tokens = re.findall(r'\[?\w+\]?', text)
        return tokens
    
    def _enhance_query_for_keywords(self, query: str) -> str:
        """
        Enhance query for better BM25 keyword matching.
        
        Transforms conversational queries into keyword patterns:
        - "reference number fifteen" → "reference [15] 15"
        - "section three point two" → "section 3.2"
        """
        enhanced = query.lower()
        
        # Number word to digit mapping
        number_words = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
            'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
            'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
            'fifty': '50',
        }
        
        # Replace number words with digits
        for word, digit in number_words.items():
            if word in enhanced:
                # Add bracket format for references: "fifteen" → "15 [15]"
                enhanced = enhanced.replace(word, f"{digit} [{digit}]")
        
        # Also add standalone number pattern if "reference" is mentioned
        if 'reference' in enhanced or 'ref' in enhanced:
            # Extract any numbers and add bracket versions
            numbers = re.findall(r'\b(\d+)\b', enhanced)
            for num in numbers:
                if f'[{num}]' not in enhanced:
                    enhanced += f' [{num}]'
        
        logger.debug(f"Enhanced query: '{query}' → '{enhanced}'")
        return enhanced
    
    def _doc_key(self, doc: Document) -> str:
        """Generate a unique key for a document for deduplication."""
        # Use document_id + chunk_index as key
        doc_id = doc.metadata.get("document_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        return f"{doc_id}:{chunk_idx}"
    
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
                
                # Bidirectional substring matching:
                # - query in title: user searches partial title
                # - title in query: LLM sends title with extra metadata like "(web, 7 sections)"
                if title_query_lower in doc_title or doc_title in title_query_lower:
                    # Score based on how much of the title matches
                    if title_query_lower in doc_title:
                        score = len(title_query_lower) / len(doc_title) if doc_title else 0
                    else:
                        # Title is substring of query - score based on title coverage
                        score = len(doc_title) / len(title_query_lower) if title_query_lower else 0
                    
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

