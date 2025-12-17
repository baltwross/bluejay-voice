"""
Test script for RAG system.
Tests document ingestion and retrieval.
"""
import os
import sys
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag_system():
    """Test the RAG ingestion and retrieval system."""
    
    print("\n" + "="*60)
    print("🤖 BLUEJAY TERMINATOR - RAG SYSTEM TEST")
    print("="*60 + "\n")
    
    # Import RAG components
    from rag import DocumentIndexer, DocumentRetriever
    
    # Initialize indexer and retriever
    print("📦 Initializing RAG system...")
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    
    # Get current stats
    stats = indexer.get_stats()
    print(f"📊 Current stats: {stats}")
    
    # Test 1: Ingest sample text
    print("\n" + "-"*40)
    print("TEST 1: Ingesting sample text document")
    print("-"*40)
    
    sample_text = """
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
    """
    
    result = indexer.ingest_text(
        text=sample_text,
        title="Claude Code Guide",
        document_id="claude-code-001",
    )
    
    print(f"✅ Ingestion result: {result}")
    
    # Test 2: Query the ingested document
    print("\n" + "-"*40)
    print("TEST 2: Querying ingested document")
    print("-"*40)
    
    queries = [
        "What are the key features of Claude Code?",
        "How can I maximize productivity with Claude Code?",
        "What productivity improvements can developers expect?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        result = retriever.retrieve(query, k=2)
        
        print(f"   Found {len(result.documents)} relevant chunks")
        if result.sources:
            print(f"   Sources: {[s['title'] for s in result.sources]}")
        if result.documents:
            print(f"   Top match preview: {result.documents[0].page_content[:150]}...")
    
    # Test 3: Filter by document
    print("\n" + "-"*40)
    print("TEST 3: Filtered retrieval (Active Document)")
    print("-"*40)
    
    result = retriever.retrieve(
        "productivity improvements",
        document_id="claude-code-001",
        k=2,
    )
    print(f"✅ Retrieved {len(result.documents)} chunks from specific document")
    
    # Test 4: Find document by title
    print("\n" + "-"*40)
    print("TEST 4: Find document by title")
    print("-"*40)
    
    doc = retriever.find_document_by_title("claude")
    if doc:
        print(f"✅ Found: {doc['title']} (ID: {doc['document_id']})")
    else:
        print("❌ Document not found")
    
    # Test 5: Reading mode retrieval
    print("\n" + "-"*40)
    print("TEST 5: Reading mode (sequential chunks)")
    print("-"*40)
    
    result = retriever.retrieve_for_reading("claude-code-001", start_chunk=0, num_chunks=3)
    print(f"✅ Retrieved {len(result.documents)} sequential chunks for reading")
    
    # Final stats
    print("\n" + "-"*40)
    print("FINAL STATS")
    print("-"*40)
    
    stats = indexer.get_stats()
    print(f"📊 {stats}")
    
    documents = indexer.list_documents()
    print(f"📚 Documents in store: {len(documents)}")
    for doc in documents:
        print(f"   - {doc['title']} ({doc['source_type']})")
    
    print("\n" + "="*60)
    print("✅ RAG SYSTEM TEST COMPLETE")
    print("="*60 + "\n")


def test_web_ingestion():
    """Test web URL ingestion."""
    print("\n" + "="*60)
    print("🌐 WEB INGESTION TEST")
    print("="*60 + "\n")
    
    from rag import DocumentIndexer, DocumentRetriever
    
    indexer = DocumentIndexer()
    retriever = DocumentRetriever()
    
    # Test with a real web page (LangChain docs)
    url = "https://python.langchain.com/docs/introduction/"
    
    print(f"📥 Ingesting: {url}")
    result = indexer.ingest(url, title="LangChain Introduction")
    
    if result["success"]:
        print(f"✅ Success! Ingested {result['num_chunks']} chunks")
        
        # Query the content
        query_result = retriever.retrieve("What is LangChain?", k=2)
        print(f"🔍 Query result: {len(query_result.documents)} relevant chunks found")
    else:
        print(f"❌ Error: {result.get('error')}")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in backend/.env file")
        sys.exit(1)
    
    # Run tests
    test_rag_system()
    
    # Uncomment to test web ingestion (requires network)
    # test_web_ingestion()



