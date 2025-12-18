"""
Benchmark runner: RAG experiment over PDFs in backend/tests/materials/.

What it does:
- Ingest each of the 5 PDFs (into a stable document_id namespace).
- Auto-discovers anchor targets (figure/table/section/reference numbers) per PDF
  by retrieving "Figure"/"Table"/"Section"/"[" and extracting the first matches.
- Sweeps context_top_m and reranking settings and measures:
  - anchor_hit_top1: top retrieved chunk contains the anchor
  - anchor_hit_any: any retrieved chunk contains the anchor
  - latency_ms

Run:
  cd backend && source venv/bin/activate
  python tests/run_materials_rag_experiment.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


PDF_FILES: List[Tuple[str, str]] = [
    ("AYAWA-2024-1712241257109.pdf", "AYAWA 2024"),
    ("AYAWA-2025-1745910098236.pdf", "AYAWA 2025"),
    ("Mastering RAG-compressed.pdf", "Mastering RAG"),
    ("Mastering AI Agents-compressed.pdf", "Mastering AI Agents"),
    ("Weaviate Agentic Architectures-ebook.pdf", "Weaviate Agentic"),
]


@dataclass(frozen=True)
class AnchorTarget:
    anchor_type: str  # figure/table/section/reference
    anchor_value: str  # e.g. "37", "2.3", "4.2", "15"
    query: str
    must_match_patterns: Tuple[re.Pattern[str], ...]


@dataclass
class SweepRow:
    doc_title: str
    doc_id: str
    context_top_m: int
    use_reranking: bool
    queries: int
    hit_top1: int
    hit_any: int
    avg_latency_ms: float
    avg_docs_returned: float


def _slugify_filename(fname: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", fname).strip("-").lower()


def _materials_dir() -> str:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(tests_dir, "materials")


def _doc_id_for_file(fname: str) -> str:
    return f"materials-{_slugify_filename(fname)}"


def _extract_first(patterns: Sequence[re.Pattern[str]], text: str) -> Optional[str]:
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(1)
    return None


def _build_anchor_target(anchor_type: str, value: str) -> AnchorTarget:
    v = value.strip()
    if anchor_type == "figure":
        return AnchorTarget(
            anchor_type="figure",
            anchor_value=v,
            query=f"What does Figure {v} show?",
            must_match_patterns=(
                re.compile(rf"\bfigure\s+{re.escape(v)}\b", re.IGNORECASE),
                re.compile(rf"\bfig\.?\s+{re.escape(v)}\b", re.IGNORECASE),
            ),
        )
    if anchor_type == "table":
        return AnchorTarget(
            anchor_type="table",
            anchor_value=v,
            query=f"What does Table {v} show?",
            must_match_patterns=(
                re.compile(rf"\btable\s+{re.escape(v)}\b", re.IGNORECASE),
                re.compile(rf"\btab\.?\s+{re.escape(v)}\b", re.IGNORECASE),
            ),
        )
    if anchor_type == "section":
        return AnchorTarget(
            anchor_type="section",
            anchor_value=v,
            query=f"What does Section {v} say?",
            must_match_patterns=(
                re.compile(rf"\bsection\s+{re.escape(v)}\b", re.IGNORECASE),
                re.compile(rf"\bsec\.?\s+{re.escape(v)}\b", re.IGNORECASE),
            ),
        )
    # reference
    return AnchorTarget(
        anchor_type="reference",
        anchor_value=v,
        query=f"What does reference [{v}] say?",
        must_match_patterns=(
            re.compile(rf"\[{re.escape(v)}\]"),
        ),
    )


def _discover_anchors(retriever, doc_id: str) -> List[AnchorTarget]:
    """
    Auto-discover one anchor of each type (figure/table/section/reference) if present.
    We do this by running a broad retrieval ("Figure", "Table", ...) and parsing the returned chunks.
    """
    targets: List[AnchorTarget] = []

    # Broad queries to pull anchor-containing chunks
    probes = [
        ("figure", "Figure", [re.compile(r"(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)]),
        ("table", "Table", [re.compile(r"(?:Table|Tab\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)]),
        ("section", "Section", [re.compile(r"(?:Section|Sec\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)]),
        ("reference", "[", [re.compile(r"\[(\d+)\]")]),
    ]

    for anchor_type, probe_query, patterns in probes:
        result = retriever.hybrid_retrieve(
            probe_query,
            document_id=doc_id,
            k=12,
            score_threshold=0.0,
            use_reranking=False,
            expand_neighbors=False,
        )
        combined = "\n".join(d.page_content for d in result.documents)
        val = _extract_first(patterns, combined)
        if val:
            targets.append(_build_anchor_target(anchor_type, val))

    return targets


def _matches_any(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _print_table(rows: List[SweepRow]) -> None:
    print("\n### Materials RAG Experiment Results\n")
    print("| doc | context_top_m | rerank | queries | hit@1 | hit@any | avg latency (ms) | avg docs returned |")
    print("|---|---:|:---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r.doc_title} | {r.context_top_m} | {'on' if r.use_reranking else 'off'} | "
            f"{r.queries} | {r.hit_top1}/{r.queries} | {r.hit_any}/{r.queries} | "
            f"{r.avg_latency_ms:.0f} | {r.avg_docs_returned:.1f} |"
        )


def main() -> int:
    # Ensure backend/ is importable when running as `python tests/...`
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from rag import DocumentIndexer, DocumentRetriever

    mats = _materials_dir()
    pdfs = [(os.path.join(mats, fname), title, fname) for fname, title in PDF_FILES if os.path.exists(os.path.join(mats, fname))]
    if not pdfs:
        print(f"### Materials RAG Experiment\n\nNo PDFs found in `{mats}`.")
        return 1

    indexer = DocumentIndexer()
    retriever = DocumentRetriever()

    # Conservative sweep (this is the thing you asked about)
    context_sweep = [4, 6, 8, 12, 20]
    rerank_sweep = [False, True]

    all_rows: List[SweepRow] = []

    for pdf_path, title, fname in pdfs:
        doc_id = _doc_id_for_file(fname)

        # Fresh ingest for repeatability
        indexer.delete_document(doc_id)
        ingest_result = indexer.ingest(source=pdf_path, title=title, document_id=doc_id)
        if not ingest_result.get("success"):
            print(f"\n### {title}\n\nFailed to ingest `{fname}`: {ingest_result.get('error')}")
            continue

        targets = _discover_anchors(retriever, doc_id)
        if not targets:
            print(f"\n### {title}\n\nNo anchors (figure/table/section/reference) discovered via probes. Skipping sweeps.")
            continue

        print(f"\n### {title}\n")
        print("- **document_id**: " + f"`{doc_id}`")
        print("- **chunks**: " + str(ingest_result.get("num_chunks", "?")))
        print("- **discovered targets**:")
        for t in targets:
            print(f"  - `{t.anchor_type} {t.anchor_value}` → query: \"{t.query}\"")

        for cm in context_sweep:
            for use_rerank in rerank_sweep:
                # Override config for this sweep run
                retriever.config.context_top_m = cm

                hit_top1 = 0
                hit_any = 0
                latencies: List[float] = []
                docs_counts: List[int] = []

                for target in targets:
                    start = time.time()
                    res = retriever.hybrid_retrieve(
                        target.query,
                        document_id=doc_id,
                        k=20,
                        score_threshold=0.0,
                        use_reranking=use_rerank,
                        expand_neighbors=True,
                    )
                    latencies.append((time.time() - start) * 1000.0)
                    docs_counts.append(len(res.documents))

                    docs_texts = [d.page_content for d in res.documents]
                    top1_text = docs_texts[0] if docs_texts else ""
                    any_text = "\n".join(docs_texts)

                    if _matches_any(target.must_match_patterns, top1_text):
                        hit_top1 += 1
                    if _matches_any(target.must_match_patterns, any_text):
                        hit_any += 1

                all_rows.append(
                    SweepRow(
                        doc_title=title,
                        doc_id=doc_id,
                        context_top_m=cm,
                        use_reranking=use_rerank,
                        queries=len(targets),
                        hit_top1=hit_top1,
                        hit_any=hit_any,
                        avg_latency_ms=sum(latencies) / max(len(latencies), 1),
                        avg_docs_returned=sum(docs_counts) / max(len(docs_counts), 1),
                    )
                )

        # Keep the documents around so you can inspect interactively later.
        # (If you want strict isolation, we can delete after each doc.)

    _print_table(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


