from datetime import datetime, timezone

from tavily import TavilyClient

from app.config import settings
from app.graph.state import ScoutFinding

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


_TAVILY_MAX_QUERY_CHARS = 400


def search_company(company: str, instructions: str | None = None, days: int = 30) -> list[ScoutFinding]:
    query = f"{company} latest news pricing product launch"
    if instructions:
        query = f"{query}. {instructions}"
    # Tavily rejects queries over 400 chars; instructions come from an LLM's
    # free-text routing output and aren't length-bounded on their own.
    query = query[:_TAVILY_MAX_QUERY_CHARS]

    # topic="news" + days bias results toward genuinely current coverage -
    # without this, Tavily's general search can surface older-but-authoritative
    # content (e.g. a comprehensive prior-year earnings writeup) ahead of
    # what's actually happened recently. days is user-selected (see the
    # Streamlit UI's recency dropdown) rather than a fixed window, since how
    # tight to search is a real tradeoff: tighter means fresher but thinner
    # coverage for less-followed companies - the Writer already handles thin
    # results gracefully via sufficient_data=false either way.
    response = _get_client().search(query=query, max_results=5, search_depth="basic", topic="news", days=days)

    findings = []
    for result in response.get("results", []):
        findings.append(
            ScoutFinding(
                source_url=result["url"],
                title=result.get("title", ""),
                published_date=result.get("published_date"),
                snippet=result.get("content", ""),
                fetched_at=datetime.now(timezone.utc),
            )
        )
    return findings
