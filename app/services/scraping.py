import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from app.config import settings
from app.graph.state import ScoutFinding

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


_TAVILY_MAX_QUERY_CHARS = 400


def _findings_from_tavily_response(response: dict) -> list[ScoutFinding]:
    return [
        ScoutFinding(
            source_url=result["url"],
            title=result.get("title", ""),
            published_date=result.get("published_date"),
            snippet=result.get("content", ""),
            fetched_at=datetime.now(timezone.utc),
        )
        for result in response.get("results", [])
    ]


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
    return _findings_from_tavily_response(response)


# amazon.in/flipkart.com specifically (not a blanket "site:amazon.com OR
# site:flipkart.com" free-text query) because Tavily's include_domains
# restricts results server-side - more reliable than hoping the query text
# alone biases results there, and works the same way regardless of query
# phrasing.
_MARKETPLACE_DOMAINS = ["amazon.in", "amazon.com", "flipkart.com"]


def search_marketplace_reviews(company: str) -> list[ScoutFinding]:
    """Amazon/Flipkart product listings and customer reviews, via Tavily's
    own already-indexed pages rather than scraping either site directly -
    both are notoriously bot-hostile (CAPTCHAs, aggressive rate limiting),
    and a plain httpx fetch would likely just get blocked outright even
    where fetch_company_site_findings works fine on a company's own site.
    Meant as a supplement for brands that sell through these marketplaces
    but have thin coverage everywhere else, particularly ones whose own
    site is JS-rendered (a plain HTML fetch sees an empty shell, no real
    product/review content) - marketplace listings are consistently
    server-rendered and contain real customer sentiment a company's own
    site rarely does."""
    query = f"{company} reviews ratings price"
    response = _get_client().search(
        query=query, max_results=5, search_depth="basic", topic="general", include_domains=_MARKETPLACE_DOMAINS
    )
    return _findings_from_tavily_response(response)


_FETCH_MAX_CHARS = 2000
_FETCH_TIMEOUT_SECONDS = 10.0
_USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; TrackerBot/1.0)"}


def _extract_finding(url: str, html: str) -> ScoutFinding | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Scope text extraction to <body> - get_text() on the whole document
    # would otherwise pull in <head>/<title> text too, which both pollutes
    # a real page's snippet and, worse, can mask a genuinely content-free
    # page as non-empty (the title alone "counts" as text).
    body = soup.body or soup
    text = " ".join(body.get_text(separator=" ", strip=True).split())
    if not text:
        logger.warning("No extractable text found at %s", url)
        return None

    return ScoutFinding(
        source_url=url,
        title=title,
        published_date=None,
        snippet=text[:_FETCH_MAX_CHARS],
        fetched_at=datetime.now(timezone.utc),
    )


def fetch_url_as_finding(url: str) -> ScoutFinding | None:
    """Fetches a specific URL directly (e.g. a company's own website) instead
    of relying on Tavily search turning it up - for smaller/local businesses
    that generic web search finds thin or no coverage for, this is often the
    only reliable source of *anything* concrete to write a report from.
    Returns None on any fetch/parse failure rather than raising - a bad URL
    shouldn't take down the whole company's research run, just fall back to
    Tavily search alone."""
    try:
        response = httpx.get(url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True, headers=_USER_AGENT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch seed URL %s: %s", url, e)
        return None
    return _extract_finding(url, response.text)


# Same-domain links whose href or link text mentions any of these are worth
# a follow-up fetch: product/launch and discount pages fill in what Tavily
# often can't find at all for a private/local business, and review pages
# feed the *existing* sentiment pipeline (brain_node classifies every
# scouted_data snippet) - so customer ratings/reviews on the company's own
# site get picked up without needing separate rating-extraction logic. The
# Writer is better at pulling a mentioned rating out of messy page text
# than a regex would be.
_RELEVANT_LINK_KEYWORDS = [
    "product",
    "shop",
    "collection",
    "new-arrival",
    "new_arrivals",
    "launch",
    "sale",
    "discount",
    "offer",
    "deal",
    "promo",
    "review",
    "testimonial",
    "rating",
]
_MAX_SEED_PAGES = 4


def fetch_company_site_findings(base_url: str, max_pages: int = _MAX_SEED_PAGES) -> list[ScoutFinding]:
    """Beyond the homepage, does a light same-domain crawl for product/
    launch, discount, and review/rating pages - meant for companies with
    little-to-no coverage anywhere else on the web, where the company's own
    site is often the only concrete source available at all. Capped at
    max_pages total fetches (homepage included), same domain only - this is
    a shallow, targeted crawl, not a general-purpose site spider."""
    try:
        response = httpx.get(base_url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True, headers=_USER_AGENT)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Failed to fetch seed URL %s: %s", base_url, e)
        return []

    homepage_finding = _extract_finding(base_url, response.text)
    findings = [homepage_finding] if homepage_finding else []
    seen = {base_url.rstrip("/")}

    link_soup = BeautifulSoup(response.text, "html.parser")
    base_domain = urlparse(base_url).netloc

    candidate_urls = []
    for a in link_soup.find_all("a", href=True):
        full_url = urljoin(base_url, a["href"]).rstrip("/")
        if urlparse(full_url).netloc != base_domain or full_url in seen:
            continue
        haystack = f"{a['href'].lower()} {(a.get_text() or '').lower()}"
        if any(keyword in haystack for keyword in _RELEVANT_LINK_KEYWORDS):
            candidate_urls.append(full_url)
            seen.add(full_url)

    for url in candidate_urls:
        if len(findings) >= max_pages:
            break
        finding = fetch_url_as_finding(url)
        if finding:
            findings.append(finding)

    return findings
