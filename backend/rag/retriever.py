"""
Document Retriever - Handles semantic search and context retrieval.

Supports multiple retrieval modes with re-ranking for precision:
1. Semantic search (default) - Uses embeddings for conceptual similarity
2. Hybrid search - Combines semantic + BM25 keyword search for better recall
3. Re-ranked hybrid - Hybrid + CrossEncoder re-ranking for precise fact retrieval

Hybrid search is recommended for queries with specific identifiers like:
- "reference number 15" → matches "[15]"
- "section 3.2" → matches "3.2 Methodology"  
- "Chapter 7" → matches "Chapter 7:"
- "Figure 37" → matches "Figure 37" / "Fig. 37"

Re-ranking is critical for "specific fact in specific chapter" style questions.
"""
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # type: ignore[import-untyped]

# BM25 for keyword search (no type stubs available)
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

# CrossEncoder for re-ranking (optional, improves precision significantly)
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None  # type: ignore

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
    Handles document retrieval from ChromaDB with optional re-ranking.
    
    Supports filtering by document_id for "Active Document" context.
    Includes CrossEncoder re-ranking for precise "specific fact" retrieval.
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
        
        # Initialize CrossEncoder re-ranker (optional but recommended)
        self._reranker: Optional[Any] = None
        if self.config.use_reranker and RERANKER_AVAILABLE:
            try:
                self._reranker = CrossEncoder(
                    self.config.reranker_model,
                    max_length=512,
                )
                logger.info(f"CrossEncoder re-ranker initialized: {self.config.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize re-ranker: {e}. Proceeding without re-ranking.")
                self._reranker = None
        elif self.config.use_reranker and not RERANKER_AVAILABLE:
            logger.warning(
                "Re-ranker requested but sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"DocumentRetriever initialized (reranker={'enabled' if self._reranker else 'disabled'})")
    
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
        if k is None:
            k = self.config.retrieval_k
        if score_threshold is None:
            score_threshold = self.config.retrieval_score_threshold
        
        logger.info(f"Retrieving for query: '{query[:50]}...' (k={k})")
        
        # Build filter if specified
        filter_dict = self._build_filter(document_id, source_type)
        
        # Perform similarity search WITH scores.
        #
        # Per LangChain docs, `similarity_search_with_score` returns a distance
        # metric that varies inversely with similarity (lower = closer/more similar).
        # We convert it to a monotonic "relevance" score in (0, 1] so we can
        # apply a stable threshold: relevance = 1 / (1 + distance).
        search_kwargs: Dict[str, Any] = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        scored_results = self.vector_store.similarity_search_with_score(
            query,
            **search_kwargs,
        )
        
        filtered_documents: List[Document] = []
        filtered_scores: List[float] = []
        for doc, distance in scored_results:
            try:
                dist_val = float(distance)
            except (TypeError, ValueError):
                dist_val = 0.0
            relevance = 1.0 / (1.0 + max(dist_val, 0.0))
            if relevance >= (score_threshold or 0.0):
                filtered_documents.append(doc)
                filtered_scores.append(relevance)
        
        documents = filtered_documents
        scores = filtered_scores
        
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
        use_reranking: Optional[bool] = None,
        expand_neighbors: Optional[bool] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval combining semantic search + BM25 keyword search.
        
        Pipeline: Hybrid (semantic + BM25) → Re-rank → Neighbor expansion → Filter
        
        This is especially effective for queries with specific identifiers:
        - "reference number 15" → BM25 finds "[15]"
        - "section 3.2" → BM25 finds "3.2"
        - "Figure 37" → BM25 finds "Fig. 37", "figure 37"
        - Conceptual queries still work via semantic search
        
        Args:
            query: The search query
            k: Number of documents for initial retrieval (default: config.hybrid_k)
            document_id: Filter to specific document
            semantic_weight: Weight for semantic results (0.0-1.0)
                            BM25 weight = 1.0 - semantic_weight
            use_reranking: Whether to use CrossEncoder re-ranking (default: config)
            expand_neighbors: Whether to expand with neighboring chunks (default: auto-detect)
            score_threshold: Minimum score threshold (default: config.tool_call_threshold)
            
        Returns:
            RetrievalResult with re-ranked, deduplicated documents
        """
        # Use config defaults
        initial_k = k or self.config.hybrid_k
        bm25_weight = 1.0 - semantic_weight
        use_reranking = use_reranking if use_reranking is not None else (self._reranker is not None)
        score_threshold = score_threshold if score_threshold is not None else self.config.tool_call_threshold
        
        # Detect anchors to determine if neighbor expansion is needed
        anchors = self._detect_anchors(query)
        if expand_neighbors is None:
            expand_neighbors = bool(anchors) and self.config.enable_chunk_expansion
        
        logger.info(
            f"Hybrid retrieve: '{query[:50]}...' "
            f"(k={initial_k}, semantic={semantic_weight:.1f}, bm25={bm25_weight:.1f}, "
            f"rerank={use_reranking}, anchors={list(anchors.keys()) if anchors else 'none'})"
        )
        
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
        
        # Sort by combined score and take top candidates for re-ranking
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:initial_k]
        
        candidate_documents = [doc for doc, _ in sorted_docs]
        
        if not candidate_documents:
            logger.info("No results found from hybrid search")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                context="",
                sources=[],
            )
        
        # =================================================================
        # RE-RANKING: Use CrossEncoder to re-order by semantic relevance
        # =================================================================
        if use_reranking and self._reranker:
            reranked = self._rerank_documents(
                query,
                candidate_documents,
                top_n=self.config.rerank_top_n,
            )
            documents = [doc for doc, _ in reranked]
            scores = [score for _, score in reranked]
            logger.info(f"Re-ranked to top {len(documents)} documents")
        else:
            # Without re-ranking, use fusion scores
            # Keep the full candidate set (up to initial_k) so we can expand neighbors
            # and apply adaptive context sizing later without prematurely truncating.
            documents = candidate_documents
            scores = [score for _, score in sorted_docs]
        
        # =================================================================
        # CONFIDENCE GATING: Check if results are reliable
        # =================================================================
        confidence = self._compute_confidence(scores)
        if confidence < self.config.low_confidence_threshold:
            logger.warning(
                f"Low confidence retrieval: {confidence:.3f} < {self.config.low_confidence_threshold}. "
                f"Results may not contain the answer."
            )
            # Still return results, but agent should use them cautiously

        # =================================================================
        # ADAPTIVE CONTEXT SIZING: anchor queries + low-confidence fallback
        # =================================================================
        effective_context_top_m = self.config.context_top_m
        if anchors:
            effective_context_top_m = max(
                effective_context_top_m,
                getattr(self.config, "anchor_context_top_m", self.config.context_top_m),
            )
        if confidence < self.config.low_confidence_threshold:
            effective_context_top_m = max(
                effective_context_top_m,
                getattr(self.config, "low_confidence_context_top_m", self.config.context_top_m),
            )
        
        # =================================================================
        # NEIGHBOR EXPANSION: Expand context when anchors detected
        # =================================================================
        should_expand = bool(document_id) and self.config.enable_chunk_expansion and bool(documents)
        if should_expand:
            should_expand = bool(expand_neighbors)
            if (not should_expand) and (confidence < self.config.low_confidence_threshold):
                should_expand = bool(getattr(self.config, "expand_on_low_confidence", True))

        if should_expand and document_id:
            # Preserve original ranking order while expanding neighbors.
            base_docs = documents
            base_scores = scores
            documents = self._expand_with_neighbors(
                base_docs,
                document_id,
                window_size=self.config.chunk_expansion_window,
            )

            # Re-score expanded documents: keep original score for retrieved chunks,
            # assign a modest default score to neighbor-only chunks.
            score_by_key: Dict[str, float] = {}
            for i, d in enumerate(base_docs[:len(base_scores)]):
                score_by_key[self._doc_key(d)] = base_scores[i]
            scores = [score_by_key.get(self._doc_key(d), 0.3) for d in documents]
        
        # Take final top M for context injection (adaptive)
        documents = documents[:effective_context_top_m]
        scores = scores[:effective_context_top_m]
        
        # Apply score threshold filter
        # NOTE: When re-ranking is used, scores are CrossEncoder scores (can be negative)
        # When not re-ranking, scores are fusion scores (0-1 range)
        # Skip threshold filtering if re-ranking was used (trust the ranking order instead)
        # For anchor queries with neighbor expansion, keep neighbors even if their
        # scores are weaker; truncation is controlled by effective_context_top_m.
        if (use_reranking and self._reranker) or bool(anchors):
            # Re-ranker already ordered by relevance - keep all top M
            filtered_docs = documents
            filtered_scores = scores
        else:
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(documents, scores):
                if score >= score_threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
            
            if not filtered_docs and documents:
                # If all below threshold, keep at least the best one
                filtered_docs = [documents[0]]
                filtered_scores = [scores[0]]
        
        context = self._build_context(filtered_docs)
        sources = self._extract_sources(filtered_docs, filtered_scores)
        
        logger.info(
            f"Hybrid search returned {len(filtered_docs)} documents "
            f"(confidence={confidence:.3f})"
        )
        return RetrievalResult(
            query=query,
            documents=filtered_docs,
            scores=filtered_scores,
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
    
    @staticmethod
    def _enhance_query_for_keywords(query: str) -> str:
        """
        Enhance query for better BM25 keyword matching.
        
        Transforms conversational queries into keyword patterns:
        - "reference number fifteen" → "reference [15] 15"
        - "section three point two" → "section 3.2"
        - "figure thirty seven" → "figure 37 [37]"
        """
        enhanced = query.lower()
        
        # Number word mappings
        units = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
        }
        tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        
        all_numbers = {**units, **tens}
        
        # =================================================================
        # STEP 1: Handle decimal patterns FIRST (before individual numbers)
        # "four point one" → "4.1", "three point two" → "3.2"
        # =================================================================
        def _decimal_repl(match: re.Match) -> str:
            int_word = match.group(1)
            dec_word = match.group(2)
            int_val = all_numbers.get(int_word, 0)
            dec_val = all_numbers.get(dec_word, 0)
            decimal_str = f"{int_val}.{dec_val}"
            return f"{decimal_str}"
        
        # Pattern: (number word) + "point" + (number word)
        number_words_pattern = "|".join(sorted(all_numbers.keys(), key=len, reverse=True))
        enhanced = re.sub(
            rf"\b({number_words_pattern})\s+point\s+({number_words_pattern})\b",
            _decimal_repl,
            enhanced,
        )
        
        # =================================================================
        # STEP 2: Handle compound integers ("thirty seven" → "37")
        # =================================================================
        def _compound_repl(match: re.Match) -> str:
            tens_word = match.group(1)
            unit_word = match.group(2)
            value = tens.get(tens_word, 0) + units.get(unit_word, 0)
            return f"{value} [{value}]"
        
        enhanced = re.sub(
            r"\b("
            + "|".join(sorted(tens.keys(), key=len, reverse=True))
            + r")\s+("
            + "|".join(sorted([k for k in units.keys() if units[k] < 10], key=len, reverse=True))
            + r")\b",
            _compound_repl,
            enhanced,
        )
        
        # =================================================================
        # STEP 3: Replace standalone number words
        # =================================================================
        # Replace standalone units (one..nineteen)
        for word, value in units.items():
            enhanced = re.sub(rf"\b{re.escape(word)}\b", f"{value} [{value}]", enhanced)
        
        # Replace standalone tens (twenty..ninety)
        for word, value in tens.items():
            enhanced = re.sub(rf"\b{re.escape(word)}\b", f"{value} [{value}]", enhanced)
        
        # Also add standalone number pattern if "reference" is mentioned
        if 'reference' in enhanced or 'ref' in enhanced:
            # Extract any numbers and add bracket versions
            numbers = re.findall(r'\b(\d+)\b', enhanced)
            for num in numbers:
                if f'[{num}]' not in enhanced:
                    enhanced += f' [{num}]'
        
        # If the user references a figure/table/section number, add common variants.
        # Example: "figure 37" -> "fig 37 fig. 37".
        fig_nums = re.findall(r"\b(?:figure|fig\.?)\s*(\d+)\b", enhanced)
        for num in fig_nums:
            enhanced += f" fig {num} fig. {num} figure {num}"
        
        # Handle "section X.Y" / "section X point Y" patterns
        # "section four point one" → "section 4.1"
        # First, normalize "point" to "." in decimal patterns
        enhanced = re.sub(r"(\d+)\s+point\s+(\d+)", r"\1.\2", enhanced)
        
        # Add section variants
        section_nums = re.findall(r"\b(?:section|sec\.?)\s*([\d.]+)\b", enhanced)
        for num in section_nums:
            enhanced += f" section {num} sec {num} sec. {num}"
        
        # Handle "chapter X" patterns
        chapter_nums = re.findall(r"\b(?:chapter|ch\.?)\s*(\d+)\b", enhanced)
        for num in chapter_nums:
            enhanced += f" chapter {num} ch {num} ch. {num}"
        
        # Handle "table X.Y" patterns
        table_nums = re.findall(r"\b(?:table|tab\.?)\s*([\d.]+)\b", enhanced)
        for num in table_nums:
            enhanced += f" table {num} tab {num} tab. {num}"
        
        # Handle "equation X" / "eq X" patterns
        eq_nums = re.findall(r"\b(?:equation|eq\.?)\s*(\d+)\b", enhanced)
        for num in eq_nums:
            enhanced += f" equation {num} eq {num} eq. {num}"
        
        # Handle "page X" patterns
        page_nums = re.findall(r"\bpage\s*(\d+)\b", enhanced)
        for num in page_nums:
            enhanced += f" page {num} p. {num} p {num}"
        
        logger.debug(f"Enhanced query: '{query}' → '{enhanced}'")
        return enhanced
    
    def _doc_key(self, doc: Document) -> str:
        """Generate a unique key for a document for deduplication."""
        # Use document_id + chunk_index as key
        doc_id = doc.metadata.get("document_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        return f"{doc_id}:{chunk_idx}"
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using CrossEncoder for improved precision.
        
        Args:
            query: The user query.
            documents: List of documents to re-rank.
            top_n: Number of top documents to return (default from config).
            
        Returns:
            List of (Document, score) tuples, sorted by relevance descending.
        """
        if not documents:
            return []
        
        if top_n is None:
            top_n = self.config.rerank_top_n
        
        if not self._reranker:
            # No re-ranker available, return documents with placeholder scores
            return [(doc, 0.5) for doc in documents[:top_n]]
        
        try:
            # Create query-document pairs for re-ranking
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get re-ranker scores
            scores = self._reranker.predict(pairs)
            
            # Combine with documents and sort by score descending
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(
                f"Re-ranked {len(documents)} docs. "
                f"Top score: {scored_docs[0][1]:.3f}, Bottom: {scored_docs[-1][1]:.3f}"
            )
            
            return scored_docs[:top_n]
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Returning original order.")
            return [(doc, 0.5) for doc in documents[:top_n]]
    
    def _detect_anchors(self, query: str) -> Dict[str, List[str]]:
        """
        Detect structural anchors in the query (figure, table, section, page, etc.).
        
        Args:
            query: The user query.
            
        Returns:
            Dict mapping anchor types to detected values.
            Example: {"figure": ["37"], "section": ["4.1"]}
        """
        query_lower = query.lower()
        anchors: Dict[str, List[str]] = {}
        
        # Figure patterns
        fig_matches = re.findall(r"\b(?:figure|fig\.?)\s*(\d+)\b", query_lower)
        if fig_matches:
            anchors["figure"] = fig_matches
        
        # Table patterns
        table_matches = re.findall(r"\b(?:table|tab\.?)\s*([\d.]+)\b", query_lower)
        if table_matches:
            anchors["table"] = table_matches
        
        # Section patterns
        section_matches = re.findall(r"\b(?:section|sec\.?)\s*([\d.]+)\b", query_lower)
        if section_matches:
            anchors["section"] = section_matches
        
        # Chapter patterns
        chapter_matches = re.findall(r"\b(?:chapter|ch\.?)\s*(\d+)\b", query_lower)
        if chapter_matches:
            anchors["chapter"] = chapter_matches
        
        # Page patterns
        page_matches = re.findall(r"\bpage\s*(\d+)\b", query_lower)
        if page_matches:
            anchors["page"] = page_matches
        
        # Reference patterns [X]
        ref_matches = re.findall(r"\[(\d+)\]|\breference\s*(\d+)\b", query_lower)
        flat_refs = [m[0] or m[1] for m in ref_matches if m[0] or m[1]]
        if flat_refs:
            anchors["reference"] = flat_refs
        
        # Equation patterns
        eq_matches = re.findall(r"\b(?:equation|eq\.?)\s*(\d+)\b", query_lower)
        if eq_matches:
            anchors["equation"] = eq_matches
        
        if anchors:
            logger.debug(f"Detected anchors in query: {anchors}")
        
        return anchors
    
    def _expand_with_neighbors(
        self,
        documents: List[Document],
        document_id: str,
        window_size: Optional[int] = None,
    ) -> List[Document]:
        """
        Expand documents by fetching neighboring chunks for additional context.
        
        This helps when the answer spans a chunk boundary (e.g., Figure 37's
        caption is in one chunk but the key statistic is in the next).
        
        Args:
            documents: List of retrieved documents.
            document_id: The document ID to filter neighbors.
            window_size: Number of chunks on each side (default from config).
            
        Returns:
            Expanded list of documents with neighbors included.
        """
        if not documents or not self.config.enable_chunk_expansion:
            return documents
        
        if window_size is None:
            window_size = self.config.chunk_expansion_window
        
        try:
            collection = self.vector_store._collection
            
            # Build an ordered list of "centers" (retrieved chunks) in the same order
            # as the incoming `documents` list. We will preserve that order while
            # expanding neighbors.
            centers: List[int] = []
            chunk_indices = set()
            for doc in documents:
                if doc.metadata.get("document_id") == document_id:
                    idx = int(doc.metadata.get("chunk_index", 0))
                    centers.append(idx)
                    chunk_indices.add(idx)
                    # Add neighbors
                    for offset in range(-window_size, window_size + 1):
                        if offset != 0:
                            chunk_indices.add(idx + offset)
            
            # Remove negative indices
            chunk_indices = {idx for idx in chunk_indices if idx >= 0}
            
            # Fetch all chunks in the expanded range
            if not chunk_indices:
                return documents
            
            results = collection.get(
                where={
                    "$and": [
                        {"document_id": {"$eq": document_id}},
                        {"chunk_index": {"$in": list(chunk_indices)}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            
            if not results or not results.get("documents"):
                return documents
            
            # Convert to Document objects and index by chunk_index
            by_index: Dict[int, Document] = {}
            for text, meta in zip(results["documents"], results["metadatas"]):
                try:
                    idx = int(meta.get("chunk_index", 0)) if meta else 0
                except (TypeError, ValueError):
                    idx = 0
                by_index[idx] = Document(page_content=text, metadata=meta)

            # Preserve importance: for each center chunk (in rank order),
            # emit the neighbor window around it in chunk_index order.
            expanded_docs: List[Document] = []
            seen: set[int] = set()
            for center in centers:
                for idx in range(center - window_size, center + window_size + 1):
                    if idx < 0 or idx in seen:
                        continue
                    doc = by_index.get(idx)
                    if doc is not None:
                        expanded_docs.append(doc)
                        seen.add(idx)

            # Fallback: if we couldn't build an ordered list, return all in index order.
            if not expanded_docs:
                expanded_docs = [by_index[i] for i in sorted(by_index.keys())]
            
            logger.debug(
                f"Expanded {len(documents)} docs to {len(expanded_docs)} "
                f"with window_size={window_size}"
            )
            
            return expanded_docs
            
        except Exception as e:
            logger.warning(f"Neighbor expansion failed: {e}. Returning original docs.")
            return documents
    
    def _compute_confidence(self, scores: List[float]) -> float:
        """
        Compute a confidence score for the retrieval result.
        
        Uses the top score as the primary indicator.
        
        Args:
            scores: List of relevance scores.
            
        Returns:
            Confidence value between 0 and 1.
        """
        if not scores:
            return 0.0
        return max(scores)
    
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
        """
        Build a context string from retrieved documents with structured citations.
        
        Includes page numbers, section paths, and detected anchors for
        precise citation by the LLM.
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("title", "Unknown")
            
            # Build citation components
            citation_parts = [f"Source {i}: {source}"]
            
            # Add page number if available
            page_num = doc.metadata.get("page_number") or doc.metadata.get("page")
            if page_num is not None:
                citation_parts.append(f"Page {page_num}")
            
            # Add section/chunk info
            if "chunk_index" in doc.metadata:
                total = doc.metadata.get("total_chunks", "?")
                citation_parts.append(f"Section {doc.metadata['chunk_index'] + 1}/{total}")
            
            # Add detected anchors (figure/table/section references)
            # Anchors stored as comma-separated string in metadata
            anchors_raw = doc.metadata.get("anchors", "")
            if anchors_raw:
                anchors = [a.strip() for a in anchors_raw.split(",") if a.strip()]
                if anchors:
                    # Show first 3 anchors to avoid clutter
                    anchor_str = ", ".join(anchors[:3])
                    if len(anchors) > 3:
                        anchor_str += f" (+{len(anchors) - 3} more)"
                    citation_parts.append(f"Contains: {anchor_str}")
            
            citation = " | ".join(citation_parts)
            
            # Remove anchor suffix from content for cleaner display
            # (anchors were appended during indexing for BM25 boost)
            content = doc.page_content
            if "[anchor:" in content:
                content = re.sub(r"\n\n\[anchor:[^\]]+\](?:\s*\[anchor:[^\]]+\])*\s*$", "", content)
            
            context_parts.append(f"[{citation}]\n{content}")
        
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
            
            documents: List[Dict[str, Any]] = []
            for doc_id, metadata in doc_metadata.items():
                documents.append({
                    "document_id": doc_id,
                    "title": metadata.get("title", "Unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "total_chunks": doc_chunks.get(doc_id, 0),
                    "source_url": metadata.get("source_url", ""),
                    "ingested_at": metadata.get("ingested_at", ""),
                })
            
            # Newest-first ordering for "what did I just upload?" style UX.
            documents.sort(key=lambda d: str(d.get("ingested_at", "")), reverse=True)
            
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

