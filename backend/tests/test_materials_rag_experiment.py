"""
Experimental suite: run a lightweight RAG test over PDFs in backend/tests/materials/.
This exercises ingestion of PDFs and a small set of retrieval/tests across different
`context_top_m` values to observe first-pass hit rates and latency.
"""
import os
import sys
import time
from typing import List, Tuple
import importlib
import logging

# Local path helpers
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(TESTS_DIR, "materials")  # backend/tests/materials
PDF_FILES = [
    ("AYAWA-2024-1712241257109.pdf", "AYAWA 2024"),
    ("AYAWA-2025-1745910098236.pdf", "AYAWA 2025"),
    ("Mastering RAG-compressed.pdf", "Mastering RAG"),
    ("Mastering AI Agents-compressed.pdf", "Mastering AI Agents"),
    ("Weaviate Agentic Architectures-ebook.pdf", "Weaviate Agentic"),
]

logger = logging.getLogger(__name__)


def _ingest_pdf(indexer, pdf_path: str, title: str, document_id: str) -> bool:
    """
    Try ingestion via existing API. If not available, fall back to text extraction
    via PyPDF2 (if installed) and ingest_text.
    """
    # Prefer a direct ingest_pdf API if the codebase provides it
    if hasattr(indexer, "ingest_pdf"):
        try:
            indexer.ingest_pdf(pdf_path=pdf_path, title=title, document_id=document_id)  # type: ignore
            return True
        except Exception as e:
            logger.warning(f"ingest_pdf failed for {pdf_path}: {e}")

    # Fallback: extract text and ingest_text
    text = None
    try:
        import PyPDF2  # type: ignore
        with open(pdf_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                pages.append(t)
            text = "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF text extraction failed for {pdf_path}: {e}")

    if text:
        if hasattr(indexer, "ingest_text"):
            return bool(indexer.ingest_text(text=text, title=title, document_id=document_id))
    return False


def _collect_pdf_texts(pdf_dir: str) -> List[Tuple[str, str, str]]:
    """Return a list of tuples: (path, title, document_id) for PDFs in dir."""
    items: List[Tuple[str, str, str]] = []
    if not os.path.isdir(pdf_dir):
        return items
    for fname, title in PDF_FILES:
        path = os.path.join(pdf_dir, fname)
        if os.path.exists(path):
            items.append((path, title, f"test-materials-{os.path.basename(fname)}"))
    return items


def test_materials_rag_experiment():
    """
    Smoke test: ingest PDFs in backend/tests/materials and ensure retrieval works.
    (The full benchmark/report is in `run_materials_rag_experiment.py`.)
    """
    # Import lazily to reuse the project's rag modules
    from rag import DocumentIndexer, DocumentRetriever

    indexer = DocumentIndexer()
    retriever = DocumentRetriever()

    # Ingest all PDFs
    pdfs = _collect_pdf_texts(PDF_DIR)
    ingested = []
    for pdf_path, title, doc_id in pdfs:
        ok = _ingest_pdf(indexer, pdf_path, title, doc_id)
        ingested.append((pdf_path, title, ok))
        if not ok:
            logger.warning(f"Could not ingest PDF: {pdf_path}")
    # Do not hard-fail if a PDF cannot be ingested in this environment;
    # log and proceed with any successfully ingested documents.
    if not all(ok for _, _, ok in ingested):
        logger.warning("Some PDFs failed ingestion; continuing with ingested ones.")

    # Quick retrieval sweep over a few queries across a few context_top_m values
    queries = [
        "What is the main topic of this document?",
        "What is the conclusion or takeaway?",
        "Figure 37 success rate"  # generic representative fact-type query
    ]
    context_values = [4, 6, 8, 12]

    for (pdf_path, title, _) in ingested:
        doc_id = f"test-materials-{os.path.basename(pdf_path)}"
        for query in queries:
            for cm in context_values:
                # Apply the context_top_m by mutating the retriever's config for this test
                try:
                    retriever.config.context_top_m = cm  # type: ignore[attr-defined]
                except Exception:
                    pass
                _start = time.time()
                # Filter to the ingested document to avoid cross-document contamination.
                result = retriever.hybrid_retrieve(query, document_id=doc_id, k=6, score_threshold=0.0)
                took = time.time() - _start
                # Basic sanity: we should produce at least one document and a non-empty context
                assert len(result.documents) > 0, f"No docs for query {query} cm={cm} doc={doc_id}"
                assert result.context and len(result.context) > 20, "Empty or too-short context"
                logger.info(f"[{title}] cm={cm} q='{query}' docs={len(result.documents)} took={took:.2f}s")


