"""
Unit tests for lightweight RAG guardrails.

These tests are intentionally pure / fast (no network calls, no OpenAI keys).
They focus on:
- When we should (and should not) trigger automatic RAG injection
- Query keyword normalization for voice-style number phrases (e.g., "thirty seven" -> 37)
"""

from prompts import should_trigger_rag
from rag.retriever import DocumentRetriever


def test_should_trigger_rag_skips_news_requests() -> None:
    assert should_trigger_rag("What's the latest news?") is False
    assert should_trigger_rag("What is the latest news?") is False
    assert should_trigger_rag("AI news today?") is False
    assert should_trigger_rag("Give me the headlines.") is False


def test_should_trigger_rag_triggers_for_document_anchors() -> None:
    assert (
        should_trigger_rag("In figure 37, what's the success rate for text manipulation?")
        is True
    )
    assert should_trigger_rag("According to the paper I uploaded, what does it say?") is True
    assert (
        should_trigger_rag('I shared a document called "Video models are zero-shot learners and reasoners".')
        is True
    )


def test_enhance_query_for_keywords_parses_compound_numbers() -> None:
    enhanced = DocumentRetriever._enhance_query_for_keywords("It's a figure thirty seven.")
    # We should get 37, not "30 7"
    assert "37" in enhanced
    assert "[37]" in enhanced
    assert "30 [30]" not in enhanced
    assert "7 [7]" not in enhanced


