import asyncio

from app.graph.state import AgentState
from app.memory.client_context import query_client_context
from app.memory.client_uploads import get_client_upload
from app.memory.sentiment import summarize_sentiment
from app.memory.vector_store import query_similar, upsert_snippets


async def brain_node(state: AgentState) -> dict:
    company = state["company"]
    findings = state["scouted_data"]

    # transformers/chromadb are CPU-bound and I/O-blocking respectively, with
    # no async clients - offload so concurrently-running companies' graphs
    # (M4 Map-Reduce) aren't stalled by this one's embedding/DB work.
    if findings:
        await asyncio.to_thread(
            upsert_snippets,
            company=company,
            texts=[f.snippet for f in findings],
            metadatas=[{"source_url": f.source_url, "published_date": f.published_date} for f in findings],
            ids=[f.source_url for f in findings],
        )

    query_text = " ".join(f.snippet for f in findings[-3:]) or company
    historical_context = await asyncio.to_thread(query_similar, company=company, query_text=query_text, top_k=5)

    # Consultancy-provided client/competitor roster (scripts/load_client_context.py)
    # - a small exact-match lookup, not a similarity search, so it's cheap
    # regardless of how large the roster grows.
    client_context = await asyncio.to_thread(query_client_context, company=company)

    # The consultancy's own uploaded dossier on this company, if any (see
    # app/memory/client_uploads.py) - always included when present, same as
    # client_context above. Its age (checked by the Writer, not here) is
    # what determines whether it's treated as sufficient on its own or as
    # background to supplement with fresh Scout findings.
    client_upload_context = await asyncio.to_thread(get_client_upload, company)

    if findings:
        sentiment_summary = await asyncio.to_thread(summarize_sentiment, [f.snippet for f in findings])
    else:
        sentiment_summary = state["sentiment_summary"]

    return {
        "historical_context": historical_context,
        "client_context": client_context,
        "client_upload_context": client_upload_context,
        "sentiment_summary": sentiment_summary,
    }
