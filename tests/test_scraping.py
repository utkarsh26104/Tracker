from unittest.mock import MagicMock, patch

import httpx

from app.services.scraping import fetch_company_site_findings, fetch_url_as_finding, search_marketplace_reviews


def _fake_response(html: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_url_as_finding_extracts_title_and_text():
    html = """
    <html>
        <head><title>  LocalBrand - Home  </title></head>
        <body>
            <nav>Menu stuff to ignore</nav>
            <script>console.log('ignore me')</script>
            <main>LocalBrand sells handmade candles across India.</main>
            <footer>Footer stuff to ignore</footer>
        </body>
    </html>
    """
    with patch("app.services.scraping.httpx.get", return_value=_fake_response(html)):
        finding = fetch_url_as_finding("https://localbrand.example")

    assert finding is not None
    assert finding.source_url == "https://localbrand.example"
    assert finding.title == "LocalBrand - Home"
    assert "handmade candles" in finding.snippet
    assert "Menu stuff" not in finding.snippet
    assert "Footer stuff" not in finding.snippet
    assert "ignore me" not in finding.snippet
    assert finding.published_date is None


def test_fetch_url_as_finding_returns_none_on_http_error():
    with patch("app.services.scraping.httpx.get", side_effect=httpx.ConnectError("boom")):
        finding = fetch_url_as_finding("https://unreachable.example")

    assert finding is None


def test_fetch_url_as_finding_returns_none_on_empty_text():
    html = "<html><head><title>Empty</title></head><body><script>only script content</script></body></html>"
    with patch("app.services.scraping.httpx.get", return_value=_fake_response(html)):
        finding = fetch_url_as_finding("https://empty.example")

    assert finding is None


def test_fetch_url_as_finding_truncates_long_pages():
    long_text = "word " * 2000  # comfortably over _FETCH_MAX_CHARS
    html = f"<html><head><title>Long Page</title></head><body><main>{long_text}</main></body></html>"
    with patch("app.services.scraping.httpx.get", return_value=_fake_response(html)):
        finding = fetch_url_as_finding("https://long.example")

    assert finding is not None
    assert len(finding.snippet) <= 2000


_HOMEPAGE_HTML = """
<html><head><title>LocalBrand Home</title></head>
<body>
    <a href="/products/new-arrivals">New Arrivals</a>
    <a href="/about">About Us</a>
    <a href="/reviews">Customer Reviews</a>
    <a href="https://external.example/other">External Link</a>
    <main>LocalBrand sells handmade candles.</main>
</body></html>
"""

_PRODUCTS_HTML = """
<html><head><title>New Arrivals</title></head>
<body><main>We just launched our Winter Collection at 20% off.</main></body></html>
"""

_REVIEWS_HTML = """
<html><head><title>Customer Reviews</title></head>
<body><main>Average rating 4.7 out of 5. "Amazing candles!" - a happy customer.</main></body></html>
"""


def test_fetch_company_site_findings_follows_relevant_links_only():
    def _get(url, **kwargs):
        pages = {
            "https://localbrand.example": _HOMEPAGE_HTML,
            "https://localbrand.example/products/new-arrivals": _PRODUCTS_HTML,
            "https://localbrand.example/reviews": _REVIEWS_HTML,
        }
        if url not in pages:
            raise AssertionError(f"unexpected fetch: {url}")
        return _fake_response(pages[url])

    with patch("app.services.scraping.httpx.get", side_effect=_get):
        findings = fetch_company_site_findings("https://localbrand.example")

    urls = {f.source_url for f in findings}
    assert urls == {
        "https://localbrand.example",
        "https://localbrand.example/products/new-arrivals",
        "https://localbrand.example/reviews",
    }
    # "/about" isn't a relevant-keyword match and the external link is a
    # different domain - neither should ever be fetched (the mock's
    # AssertionError above would have fired if they were).
    reviews_finding = next(f for f in findings if f.source_url == "https://localbrand.example/reviews")
    assert "4.7 out of 5" in reviews_finding.snippet


def test_fetch_company_site_findings_respects_max_pages():
    def _get(url, **kwargs):
        return _fake_response(_HOMEPAGE_HTML if url == "https://localbrand.example" else _PRODUCTS_HTML)

    with patch("app.services.scraping.httpx.get", side_effect=_get):
        findings = fetch_company_site_findings("https://localbrand.example", max_pages=2)

    assert len(findings) == 2


def test_fetch_company_site_findings_returns_empty_list_when_homepage_unreachable():
    with patch("app.services.scraping.httpx.get", side_effect=httpx.ConnectError("boom")):
        findings = fetch_company_site_findings("https://unreachable.example")

    assert findings == []


def test_search_marketplace_reviews_restricts_to_marketplace_domains():
    fake_client = MagicMock()
    fake_client.search.return_value = {
        "results": [
            {
                "url": "https://www.amazon.in/dp/xyz",
                "title": "LocalBrand Candle Set - Amazon.in",
                "content": "4.3 out of 5 stars, 1,204 ratings. Great scent, arrived on time.",
                "published_date": None,
            }
        ]
    }

    with patch("app.services.scraping._get_client", return_value=fake_client):
        findings = search_marketplace_reviews("LocalBrand")

    assert len(findings) == 1
    assert findings[0].source_url == "https://www.amazon.in/dp/xyz"
    assert "4.3 out of 5 stars" in findings[0].snippet

    _, kwargs = fake_client.search.call_args
    assert kwargs["include_domains"] == ["amazon.in", "amazon.com", "flipkart.com"]
    assert "LocalBrand" in kwargs["query"]
