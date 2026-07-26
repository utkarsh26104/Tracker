"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import asyncio
import time
from unittest.mock import patch

import pytest

from app.graph.master_graph import run_map_phase
from app.graph.state import CompanyJobStatus
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm

COMPANIES = ["Stripe", "Adyen", "Acme Corp"]
PER_CALL_DELAY = 0.3


def _delayed_supervisor_llm():
    llm = make_content_driven_supervisor_llm()
    original_side_effect = llm.with_structured_output.return_value.ainvoke.side_effect

    async def _delayed(messages):
        await asyncio.sleep(PER_CALL_DELAY)
        return await original_side_effect(messages)

    llm.with_structured_output.return_value.ainvoke.side_effect = _delayed
    return llm


@pytest.mark.asyncio
async def test_companies_run_concurrently_not_sequentially(pg_pool, fake_scout_finding):
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=_delayed_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        start = time.monotonic()
        result = await run_map_phase(job_id="test-map-reduce", companies=COMPANIES, pool=pg_pool)
        elapsed = time.monotonic() - start

    # Fully sequential would be ~3 companies * (3 supervisor calls * delay +
    # real Postgres checkpoint round-trips) - empirically ~10.5s+ (see
    # app/db/checkpointer.py's docstring for the measured before/after numbers).
    # Concurrent execution isn't instant (real network I/O still applies, and
    # Neon round-trip latency varies noticeably run to run), so this is a fixed
    # ceiling comfortably under the serialized baseline rather than a tight
    # multiple of the tiny mock delay, which was flaky under real network jitter.
    assert elapsed < 9.0, f"took {elapsed:.2f}s - looks sequential (~10.5s serial baseline), not concurrent"
    assert all(status == CompanyJobStatus.AWAITING_APPROVAL for status in result.company_statuses.values())
    assert set(result.company_reports.keys()) == set(COMPANIES)


@pytest.mark.asyncio
async def test_map_phase_does_not_build_comparison_matrix(pg_pool, fake_scout_finding):
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        result = await run_map_phase(job_id="test-map-only", companies=["Stripe"], pool=pg_pool)

    assert result.comparison_matrix is None
