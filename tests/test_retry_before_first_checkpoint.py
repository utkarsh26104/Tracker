"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py).

Regression test for a real bug: _ainvoke_with_retry's original logic always
retried with input=None ("resume from the last checkpoint") after catching a
transient error. If the very first-ever call for a brand new thread_id fails
before LangGraph writes its first checkpoint, there IS no checkpoint to
resume from, and None input raises langgraph.errors.EmptyInputError instead
of actually retrying - turning a recoverable transient error (e.g. a Groq
rate limit on the first Supervisor call) into a guaranteed hard failure."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from app.graph.master_graph import run_company
from app.graph.state import RouteDecision
from tests.conftest import make_scripted_writer_llm


@pytest.mark.asyncio
async def test_transient_error_on_very_first_call_still_recovers(pg_pool, fake_scout_finding):
    """A brand new thread_id (no prior checkpoint could possibly exist) whose
    first-ever Supervisor call raises a transient error must still recover
    and complete, not raise EmptyInputError."""
    company = "Acme"
    job_id = f"retry-test-{uuid.uuid4()}"

    calls = iter(
        [
            requests.exceptions.ConnectionError("simulated connection blip on the very first call"),
            RouteDecision(next="Search", reasoning="need data"),
            RouteDecision(next="Write", reasoning="enough data"),
            RouteDecision(next="FINISH", reasoning="done"),
        ]
    )

    async def _invoke(_messages):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_invoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=fake_llm),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm("# Acme\nDone.", True)),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        result_company, result = await run_company(pg_pool, company, job_id)

    assert result_company == company
    assert result["final_report"] == "# Acme\nDone."
    assert result["route_history"] == ["Search", "Write", "FINISH"]
