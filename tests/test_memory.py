"""Real local FinancialBERT/ChromaDB tests - no mocks, no network, no
credentials needed. Slower than the mocked graph tests (loads real models)."""

from app.memory.embeddings import embed_texts
from app.memory.sentiment import summarize_sentiment
from app.memory.vector_store import query_similar, upsert_snippets


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def test_related_sentences_score_higher_similarity_than_unrelated():
    a, b, c = embed_texts(
        [
            "Stripe raised its transaction fees for enterprise customers.",
            "Stripe increased pricing on card payments for large merchants.",
            "The weather in Paris was sunny and mild this weekend.",
        ]
    )
    assert _cosine(a, b) > _cosine(a, c)


def test_upsert_then_query_round_trip():
    company = "TestCo Memory Suite"
    upsert_snippets(
        company=company,
        texts=["TestCo announced a new pricing tier for enterprise customers."],
        metadatas=[{"published_date": "2026-01-01"}],
        ids=["test-memory-1"],
    )

    matches = query_similar(company=company, query_text="TestCo pricing changes", top_k=3)

    assert len(matches) >= 1
    assert any("pricing" in m.text.lower() for m in matches)


def test_sentiment_summary_produces_a_breakdown():
    summary = summarize_sentiment(
        [
            "The company reported record profits and strong growth this quarter.",
            "The company faces backlash after major layoffs and declining revenue.",
        ]
    )
    assert summary is not None
    assert "Sentiment breakdown" in summary
