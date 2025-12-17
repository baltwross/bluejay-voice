"""
Performance Testing Suite for Bluejay Terminator.

Tests and benchmarks various components:
1. RAG retrieval latency
2. Document ingestion speed
3. Voice pipeline simulation metrics
4. Memory usage monitoring

This script is for local testing and optimization.
Production monitoring uses LiveKit's built-in metrics (see agent.py).
"""
import os
import sys
import time
import logging
import statistics
from typing import Callable, Any
from dataclasses import dataclass, field

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance benchmark results."""
    name: str
    samples: list[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.samples)
    
    @property
    def avg_ms(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.mean(self.samples) * 1000
    
    @property
    def min_ms(self) -> float:
        if not self.samples:
            return 0.0
        return min(self.samples) * 1000
    
    @property
    def max_ms(self) -> float:
        if not self.samples:
            return 0.0
        return max(self.samples) * 1000
    
    @property
    def p50_ms(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.median(self.samples) * 1000
    
    @property
    def p95_ms(self) -> float:
        if len(self.samples) < 2:
            return self.max_ms
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx] * 1000
    
    @property
    def std_dev_ms(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples) * 1000
    
    def record(self, duration: float) -> None:
        """Record a sample (duration in seconds)."""
        self.samples.append(duration)
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Samples: {self.count}\n"
            f"  Avg: {self.avg_ms:.2f}ms\n"
            f"  Min: {self.min_ms:.2f}ms\n"
            f"  Max: {self.max_ms:.2f}ms\n"
            f"  P50: {self.p50_ms:.2f}ms\n"
            f"  P95: {self.p95_ms:.2f}ms\n"
            f"  StdDev: {self.std_dev_ms:.2f}ms"
        )


def benchmark(func: Callable, iterations: int = 10, warmup: int = 2) -> PerformanceMetrics:
    """Run a benchmark on a function."""
    metrics = PerformanceMetrics(name=func.__name__)
    
    # Warmup runs (not recorded)
    for _ in range(warmup):
        func()
    
    # Benchmark runs
    for i in range(iterations):
        start = time.perf_counter()
        func()
        duration = time.perf_counter() - start
        metrics.record(duration)
        logger.debug(f"  Run {i+1}/{iterations}: {duration*1000:.2f}ms")
    
    return metrics


# =============================================================================
# RAG BENCHMARKS
# =============================================================================

def benchmark_rag_retrieval():
    """Benchmark RAG retrieval performance."""
    print("\n" + "="*60)
    print("📊 RAG RETRIEVAL BENCHMARKS")
    print("="*60 + "\n")
    
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Test queries of varying complexity
    queries = [
        "What is Claude Code?",  # Simple
        "How do I configure multi-file editing in Cursor with Composer?",  # Medium
        "What are the best practices for using AI coding tools to improve productivity on refactoring tasks?",  # Complex
    ]
    
    all_metrics = []
    
    for query in queries:
        print(f"Query: \"{query[:50]}...\"")
        
        def run_query():
            retriever.retrieve(query, k=4)
        
        metrics = benchmark(run_query, iterations=20, warmup=3)
        all_metrics.append(metrics)
        print(f"  → P50: {metrics.p50_ms:.1f}ms, P95: {metrics.p95_ms:.1f}ms\n")
    
    # Aggregate
    all_samples = [s for m in all_metrics for s in m.samples]
    aggregate = PerformanceMetrics(name="All RAG Queries")
    aggregate.samples = all_samples
    
    print("-"*40)
    print(aggregate)
    
    # Check against targets
    print("\n📈 Performance Targets:")
    if aggregate.p95_ms < 500:
        print("  ✅ P95 < 500ms: PASS")
    else:
        print(f"  ❌ P95 < 500ms: FAIL ({aggregate.p95_ms:.1f}ms)")
    
    if aggregate.avg_ms < 300:
        print("  ✅ Avg < 300ms: PASS")
    else:
        print(f"  ❌ Avg < 300ms: FAIL ({aggregate.avg_ms:.1f}ms)")
    
    return aggregate


def benchmark_document_ingestion():
    """Benchmark document ingestion performance."""
    print("\n" + "="*60)
    print("📊 DOCUMENT INGESTION BENCHMARKS")
    print("="*60 + "\n")
    
    from rag import DocumentIndexer
    
    indexer = DocumentIndexer()
    
    # Test documents of different sizes
    small_doc = "This is a small test document. " * 10
    medium_doc = "This is a medium test document with more content. " * 100
    large_doc = "This is a large test document with substantial content. " * 500
    
    test_cases = [
        ("Small (~200 chars)", small_doc),
        ("Medium (~5KB)", medium_doc),
        ("Large (~25KB)", large_doc),
    ]
    
    for name, doc in test_cases:
        print(f"Document: {name}")
        
        def run_ingestion():
            indexer.ingest_text(
                text=doc,
                title=f"Benchmark Doc {time.time()}",
                document_id=f"bench-{time.time()}",
            )
        
        metrics = benchmark(run_ingestion, iterations=5, warmup=1)
        print(f"  → P50: {metrics.p50_ms:.1f}ms, P95: {metrics.p95_ms:.1f}ms\n")


def benchmark_reading_mode():
    """Benchmark reading mode chunk retrieval."""
    print("\n" + "="*60)
    print("📊 READING MODE BENCHMARKS")
    print("="*60 + "\n")
    
    from rag import DocumentRetriever
    
    retriever = DocumentRetriever()
    
    # Get a document to read
    docs = retriever.list_documents()
    if not docs:
        print("⚠️ No documents available for reading benchmark")
        return None
    
    doc_id = docs[0]["document_id"]
    print(f"Testing with document: {docs[0]['title']}")
    
    # Benchmark sequential chunk retrieval
    def run_reading_retrieval():
        retriever.retrieve_for_reading(doc_id, start_chunk=0, num_chunks=3)
    
    metrics = benchmark(run_reading_retrieval, iterations=20, warmup=3)
    print(metrics)
    
    return metrics


# =============================================================================
# LATENCY TARGET ANALYSIS
# =============================================================================

def analyze_latency_budget():
    """Analyze the latency budget for the voice pipeline.
    
    Target: <1.5s total response time
    
    Components:
    - STT processing: ~100-300ms (depends on utterance length)
    - EOU delay: ~200-400ms (end of utterance detection)
    - RAG retrieval: ~200-500ms (when triggered)
    - LLM inference: ~200-800ms (TTFT)
    - TTS generation: ~100-300ms (TTFB)
    """
    print("\n" + "="*60)
    print("📊 LATENCY BUDGET ANALYSIS")
    print("="*60 + "\n")
    
    # Estimate component latencies
    components = {
        "STT Processing": (100, 300),
        "End-of-Utterance Detection": (200, 400),
        "RAG Retrieval (if triggered)": (200, 500),
        "LLM TTFT": (200, 800),
        "TTS TTFB": (100, 300),
    }
    
    print("Component Latency Estimates (min-max ms):\n")
    total_min = 0
    total_max = 0
    
    for component, (min_ms, max_ms) in components.items():
        print(f"  {component}: {min_ms}-{max_ms}ms")
        total_min += min_ms
        total_max += max_ms
    
    print(f"\n  {'='*40}")
    print(f"  TOTAL: {total_min}-{total_max}ms")
    print(f"\n  Target: <1500ms")
    
    if total_max <= 1500:
        print(f"  ✅ Max estimate within budget")
    else:
        print(f"  ⚠️ Max estimate exceeds budget by {total_max - 1500}ms")
    
    print("\n📝 Optimization Recommendations:")
    print("  1. Enable preemptive_generation in AgentSession (✅ done)")
    print("  2. Use streaming LLM responses (✅ default)")
    print("  3. Use streaming TTS (✅ default with ElevenLabs)")
    print("  4. Keep RAG retrieval k=4 for speed (✅ configured)")
    print("  5. Tune VAD for quick EOU detection (✅ configured)")


# =============================================================================
# MAIN
# =============================================================================

def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("\n" + "="*60)
    print("🚀 BLUEJAY TERMINATOR - PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Run benchmarks
    rag_metrics = benchmark_rag_retrieval()
    benchmark_document_ingestion()
    benchmark_reading_mode()
    analyze_latency_budget()
    
    print("\n" + "="*60)
    print("✅ BENCHMARKS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()

