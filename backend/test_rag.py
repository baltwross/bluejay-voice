"""
Comprehensive RAG System Tests.

Tests document ingestion and retrieval across various document types,
query patterns, and edge cases. Includes performance benchmarking.

Test Categories:
1. Basic ingestion and retrieval
2. Diverse document types (PDF, web, YouTube, text)
3. Complex query patterns
4. Edge cases and error handling
5. Performance benchmarks
"""
import os
import sys
import logging
import time
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_DOCUMENTS = {
    "claude_code_guide": """
    Claude Code is a breakthrough in agentic coding systems. It operates as a background process 
    in your terminal, allowing it to execute commands, read files, and make changes without 
    explicit instructions for each action. The system uses advanced context awareness to 
    understand your project structure.
    
    Key features of Claude Code include:
    1. Autonomous terminal operation - runs continuously monitoring your workflow
    2. Context-aware code changes - understands your entire project structure
    3. Multi-file editing - can modify multiple files simultaneously
    4. Test verification - automatically tests changes before committing
    
    To maximize productivity with Claude Code:
    - Configure it with your project context using .claude files
    - Use it for repetitive tasks like refactoring across multiple files
    - Let it handle debugging while you focus on architecture
    - Leverage its ability to read documentation and apply patterns
    
    Studies show that developers using Claude Code can achieve 5-10x productivity improvements
    on repetitive coding tasks, and up to 3x improvement on novel feature development.
    """,
    
    "cursor_tips": """
    Cursor is an AI-powered IDE built on top of VS Code. It integrates AI assistance directly
    into your development workflow with features like:
    
    Chapter 1: Getting Started
    - Install Cursor from cursor.sh
    - Import your VS Code settings automatically
    - Connect your OpenAI or Anthropic API key
    
    Chapter 2: Composer Mode
    Composer is Cursor's multi-file editing feature. It allows you to make changes across
    multiple files with a single natural language instruction. Key tips:
    - Be specific about which files to modify
    - Provide context about your codebase
    - Review changes before accepting them
    
    Chapter 3: Chat with Codebase
    Use Cmd+K to chat with your entire codebase. The AI has access to:
    - All files in your workspace
    - Your recent edits
    - Error messages from the terminal
    
    Chapter 4: Tab Completion
    Cursor provides intelligent tab completion that:
    - Predicts multi-line code blocks
    - Understands your coding style
    - Learns from your codebase patterns
    
    Advanced tip: Use @file references in chat to focus on specific files.
    """,
    
    "mcp_overview": """
    Model Context Protocol (MCP) is an open standard for connecting AI models to external
    data sources and tools. It enables seamless integration between LLMs and:
    
    1. File Systems - Read and write files locally or in the cloud
    2. Databases - Query SQL and NoSQL databases directly
    3. APIs - Connect to REST and GraphQL endpoints
    4. Development Tools - Integrate with git, package managers, etc.
    
    MCP Servers can be:
    - Local processes running on your machine
    - Remote services accessed over HTTPS
    - Docker containers with isolated environments
    
    Key concepts:
    - Resources: Data that can be read (files, database records, API responses)
    - Tools: Actions that can be performed (create file, run query, make API call)
    - Prompts: Pre-configured templates for common tasks
    
    Popular MCP servers include:
    - Filesystem server for local file access
    - GitHub server for repository operations
    - Postgres server for database queries
    - Brave Search for web searches
    
    The MCP specification is maintained by Anthropic and available at modelcontextprotocol.io
    """,
}


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Container for test results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.duration_ms: float = 0
        self.details: dict = {}
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status} {self.name} ({self.duration_ms:.1f}ms)"
        if self.error:
            result += f"\n   Error: {self.error}"
        return result


class TestRunner:
    """Simple test runner with performance tracking."""
    
    def __init__(self):
        self.results: list[TestResult] = []
    
    def run_test(self, name: str, test_func):
        """Run a test function and record results."""
        result = TestResult(name)
        start = time.time()
        
        try:
            details = test_func()
            result.passed = True
            result.details = details or {}
        except AssertionError as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
        
        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        print(result)
        return result
    
    def summary(self) -> str:
        """Get test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.duration_ms for r in self.results)
        
        return (
            f"\n{'='*60}\n"
            f"TEST SUMMARY: {passed}/{total} passed ({total_time:.0f}ms total)\n"
            f"{'='*60}"
        )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_basic_ingestion():
    """Test 1: Basic text ingestion and retrieval."""
    from rag import DocumentIndexer, DocumentRetriever
    
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    
    # Ingest sample document
    result = indexer.ingest_text(
        text=SAMPLE_DOCUMENTS["claude_code_guide"],
        title="Claude Code Guide",
        document_id="test-claude-001",
    )
    
    assert result["success"], f"Ingestion failed: {result.get('error')}"
    assert result["num_chunks"] > 0, "No chunks created"
    
    # Retrieve and verify
    query_result = retriever.retrieve("What are Claude Code features?", k=2)
    assert len(query_result.documents) > 0, "No documents retrieved"
    assert "autonomous" in query_result.context.lower(), "Expected content not found"
    
    return {"chunks_created": result["num_chunks"], "docs_retrieved": len(query_result.documents)}


def test_multi_document_ingestion():
    """Test 2: Multiple document ingestion and cross-document retrieval."""
    from rag import DocumentIndexer, DocumentRetriever
    
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    
    # Ingest multiple documents
    docs = [
        ("Cursor Tips", "test-cursor-001", SAMPLE_DOCUMENTS["cursor_tips"]),
        ("MCP Overview", "test-mcp-001", SAMPLE_DOCUMENTS["mcp_overview"]),
    ]
    
    for title, doc_id, text in docs:
        result = indexer.ingest_text(text=text, title=title, document_id=doc_id)
        assert result["success"], f"Failed to ingest {title}"
    
    # Cross-document query
    query_result = retriever.retrieve("How do I edit multiple files?", k=4)
    assert len(query_result.documents) > 0, "No documents retrieved"
    
    # Should find Cursor's Composer feature
    titles = [s.get("title", "") for s in query_result.sources]
    assert any("Cursor" in t for t in titles), "Expected Cursor document in results"
    
    return {"sources_found": len(query_result.sources)}


def test_document_filtering():
    """Test 3: Filtered retrieval by document ID (Active Document mode)."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Query with filter - should only get Claude Code results
    result = retriever.retrieve(
        "productivity improvements",
        document_id="test-claude-001",
        k=3,
    )
    
    # All results should be from Claude Code document
    for doc in result.documents:
        assert doc.metadata.get("document_id") == "test-claude-001", \
            f"Got document from wrong source: {doc.metadata.get('document_id')}"
    
    return {"filtered_docs": len(result.documents)}


def test_find_document_by_title():
    """Test 4: Find document by partial title match."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Test partial matches
    test_cases = [
        ("claude", "Claude Code Guide"),
        ("cursor", "Cursor Tips"),
        ("mcp", "MCP Overview"),
    ]
    
    for query, expected_title in test_cases:
        doc = retriever.find_document_by_title(query)
        assert doc is not None, f"Document not found for query: {query}"
        assert expected_title.lower() in doc["title"].lower(), \
            f"Expected '{expected_title}' but got '{doc['title']}'"
    
    return {"matches_found": len(test_cases)}


def test_reading_mode_retrieval():
    """Test 5: Sequential chunk retrieval for reading mode."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Get first batch
    result1 = retriever.retrieve_for_reading("test-cursor-001", start_chunk=0, num_chunks=2)
    assert len(result1.documents) > 0, "No chunks retrieved for reading"
    
    # Get second batch
    result2 = retriever.retrieve_for_reading("test-cursor-001", start_chunk=2, num_chunks=2)
    
    # Verify chunks are different (sequential reading works)
    if len(result2.documents) > 0:
        text1 = result1.documents[0].page_content[:100]
        text2 = result2.documents[0].page_content[:100]
        assert text1 != text2, "Chunks should be different"
    
    return {"batch1_size": len(result1.documents), "batch2_size": len(result2.documents)}


def test_list_documents():
    """Test 6: List all documents in knowledge base."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    documents = retriever.list_documents()
    
    assert len(documents) >= 3, f"Expected at least 3 documents, got {len(documents)}"
    
    # Verify document structure
    for doc in documents:
        assert "document_id" in doc, "Missing document_id"
        assert "title" in doc, "Missing title"
        assert "total_chunks" in doc, "Missing total_chunks"
    
    return {"document_count": len(documents)}


def test_complex_queries():
    """Test 7: Complex multi-topic queries."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Complex queries that span topics
    queries = [
        ("What AI tools help with refactoring code?", ["Claude", "Cursor"]),
        ("How can I query databases with AI?", ["MCP"]),
        ("What is the best way to edit multiple files at once?", ["Cursor", "Claude"]),
    ]
    
    for query, expected_sources in queries:
        result = retriever.retrieve(query, k=4)
        assert len(result.documents) > 0, f"No results for: {query}"
        
        # At least one expected source should be found
        found_sources = [s["title"] for s in result.sources]
        found_match = any(
            any(exp.lower() in src.lower() for exp in expected_sources)
            for src in found_sources
        )
        assert found_match, f"Expected sources {expected_sources} not found in {found_sources}"
    
    return {"queries_tested": len(queries)}


def test_edge_cases():
    """Test 8: Edge cases and error handling."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Empty query
    result = retriever.retrieve("", k=2)
    # Should not crash, may return empty or all docs
    
    # Non-existent document ID
    result = retriever.retrieve_for_reading("non-existent-doc-id", start_chunk=0, num_chunks=5)
    assert len(result.documents) == 0, "Should return empty for non-existent doc"
    
    # Very long query
    long_query = "What " * 100 + "are the features?"
    result = retriever.retrieve(long_query, k=2)
    # Should not crash
    
    # Special characters in query
    special_query = "How to use @file & <tag> in Cursor?"
    result = retriever.retrieve(special_query, k=2)
    # Should not crash
    
    return {"edge_cases_passed": 4}


def test_retrieval_performance():
    """Test 9: Retrieval performance benchmark."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    queries = [
        "What is Claude Code?",
        "How to use Cursor Composer?",
        "What are MCP servers?",
        "Best practices for AI coding",
        "Multi-file editing features",
    ]
    
    times = []
    for query in queries:
        start = time.time()
        retriever.retrieve(query, k=4)
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    # Performance assertions (adjust thresholds as needed)
    assert avg_time < 1000, f"Average retrieval too slow: {avg_time:.0f}ms"
    assert max_time < 2000, f"Max retrieval too slow: {max_time:.0f}ms"
    
    return {"avg_ms": round(avg_time, 1), "max_ms": round(max_time, 1)}


def test_chapter_specific_retrieval():
    """Test 10: Chapter/section-specific retrieval (critical for interview)."""
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Query about specific chapter content
    result = retriever.retrieve("What does Chapter 2 say about Composer?", k=3)
    
    assert len(result.documents) > 0, "No results for chapter query"
    assert "composer" in result.context.lower(), "Chapter 2 content not found"
    
    # Query about Chapter 3
    result = retriever.retrieve("How do I chat with my codebase in Cursor?", k=3)
    assert "cmd+k" in result.context.lower() or "chat" in result.context.lower(), \
        "Chapter 3 content not found"
    
    return {"chapter_queries_passed": 2}


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all RAG system tests."""
    print("\n" + "="*60)
    print("🤖 BLUEJAY TERMINATOR - COMPREHENSIVE RAG TESTS")
    print("="*60 + "\n")
    
    runner = TestRunner()
    
    # Run tests in order
    tests = [
        ("Basic Ingestion", test_basic_ingestion),
        ("Multi-Document Ingestion", test_multi_document_ingestion),
        ("Document Filtering", test_document_filtering),
        ("Find by Title", test_find_document_by_title),
        ("Reading Mode", test_reading_mode_retrieval),
        ("List Documents", test_list_documents),
        ("Complex Queries", test_complex_queries),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmark", test_retrieval_performance),
        ("Chapter-Specific Retrieval", test_chapter_specific_retrieval),
    ]
    
    for name, test_func in tests:
        print(f"\n▸ Running: {name}")
        runner.run_test(name, test_func)
    
    print(runner.summary())
    
    # Return exit code
    failed = sum(1 for r in runner.results if not r.passed)
    return failed


def cleanup_test_documents():
    """Remove test documents from the vector store."""
    print("\n🧹 Cleaning up test documents...")
    
    from rag import DocumentIndexer
    
    indexer = DocumentIndexer()
    
    test_doc_ids = ["test-claude-001", "test-cursor-001", "test-mcp-001"]
    for doc_id in test_doc_ids:
        try:
            # Note: This would require implementing a delete method
            # For now, we'll just note that cleanup is needed
            pass
        except Exception as e:
            logger.warning(f"Could not delete {doc_id}: {e}")
    
    print("   (Note: Test documents remain in store - implement delete if needed)")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in backend/.env file")
        sys.exit(1)
    
    # Run tests
    exit_code = run_all_tests()
    
    # Optional cleanup
    # cleanup_test_documents()
    
    sys.exit(exit_code)
