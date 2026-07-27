from unittest.mock import AsyncMock, MagicMock

import groq
import httpx
import pytest
import requests

from app.graph.master_graph import _ainvoke_with_retry, _rate_limit_wait_seconds, describe_exception


def _rate_limit_error(message: str) -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return groq.RateLimitError(message, response=response, body=None)


def test_rate_limit_wait_seconds_parses_minutes_and_seconds_format():
    """Regression test: the daily-quota message format ("16m32.304s") wasn't
    captured by the old seconds-only regex at all, silently falling back to
    a much-too-short guess instead of actually waiting long enough."""
    exc = _rate_limit_error("... tokens per day (TPD): Limit 200000, Used 199912. Please try again in 16m32.304s.")
    assert _rate_limit_wait_seconds(exc, attempt=1) == pytest.approx(16 * 60 + 32.304 + 1.0)


def test_rate_limit_wait_seconds_parses_pure_seconds_format():
    exc = _rate_limit_error("... tokens per minute (TPM): Limit 8000, Used 7447. Please try again in 18.5475s.")
    assert _rate_limit_wait_seconds(exc, attempt=1) == pytest.approx(18.5475 + 1.0)


def test_rate_limit_wait_seconds_falls_back_when_unparseable():
    exc = _rate_limit_error("Rate limit reached, no wait-time hint in this message at all.")
    assert _rate_limit_wait_seconds(exc, attempt=2) == 10.0


def test_describe_exception_for_daily_rate_limit():
    exc = _rate_limit_error("... tokens per day (TPD): Limit 200000, Used 199912. Please try again in 16m32.304s.")
    assert describe_exception(exc) == "Groq's daily rate limit was reached. Please try again in 16m 32s."


def test_describe_exception_for_per_minute_rate_limit():
    exc = _rate_limit_error("... tokens per minute (TPM): Limit 8000, Used 7447. Please try again in 18.5475s.")
    assert describe_exception(exc) == "Groq's per-minute rate limit was reached. Please try again in 19s."


def test_describe_exception_for_transient_connection_error():
    exc = requests.exceptions.ConnectionError("boom")
    assert describe_exception(exc) == "A temporary connection issue interrupted research for this company. Please try again."


def test_describe_exception_for_generic_exception():
    exc = ValueError("something else entirely")
    assert describe_exception(exc) == "Research failed unexpectedly: something else entirely"


@pytest.mark.asyncio
async def test_ainvoke_with_retry_fails_fast_on_long_wait():
    """A daily-quota wait can be many minutes - sleeping through that inside
    a live request would be pointless. This must raise immediately on the
    first attempt, not loop through the normal retry/sleep cycle (which
    would make this test itself hang for real minutes if it regressed)."""
    exc = _rate_limit_error("... tokens per day (TPD) ... Please try again in 16m32.304s.")
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=exc)

    with pytest.raises(groq.RateLimitError):
        await _ainvoke_with_retry(graph, {"some": "state"}, {"configurable": {"thread_id": "t1"}})

    assert graph.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_ainvoke_with_retry_still_retries_short_waits():
    exc = _rate_limit_error("... tokens per minute (TPM) ... Please try again in 1.5s.")
    call_count = 0

    async def _ainvoke(_input, config):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise exc
        return {"final_report": "# Done"}

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=_ainvoke)
    graph.aget_state = AsyncMock(return_value=MagicMock(values={"some": "state"}))

    result = await _ainvoke_with_retry(graph, {"some": "state"}, {"configurable": {"thread_id": "t1"}})

    assert result == {"final_report": "# Done"}
    assert graph.ainvoke.call_count == 2
