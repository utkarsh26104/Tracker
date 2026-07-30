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


def _delayed_supervisor_llm(call_windows: list):
    llm = make_content_driven_supervisor_llm()
    original_side_effect = llm.with_structured_output.return_value.ainvoke.side_effect

    async def _delayed(messages):
        start = time.monotonic()
        await asyncio.sleep(PER_CALL_DELAY)
        result = await original_side_effect(messages)
        call_windows.append((start, time.monotonic()))
        return result

    llm.with_structured_output.return_value.ainvoke.side_effect = _delayed
    return llm


@pytest.mark.asyncio
async def test_companies_run_concurrently_not_sequentially(pg_pool, fake_scout_finding):
    # Records a (start, end) window per Supervisor call across all
    # companies, used below to prove real concurrency directly instead of
    # via an aggregate wall-clock threshold - a threshold against real
    # Neon network I/O kept drifting (worsened further by the connection
    # pool's per-checkout health check added in checkpointer.py, itself a
    # real extra round-trip, not just occasional jitter) and was
    # fundamentally the wrong tool: how *long* the run took depends on
    # variable network conditions neither this test nor the code controls,
    # but whether two companies' calls genuinely overlapped in time does not.
    call_windows: list[tuple[float, float]] = []

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=_delayed_supervisor_llm(call_windows)),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        result = await run_map_phase(job_id="test-map-reduce", companies=COMPANIES, pool=pg_pool)

    # A single company's own graph calls its Supervisor strictly
    # sequentially, so any overlap can only come from two *different*
    # companies running at once - direct proof of concurrency.
    overlap_found = any(
        a_start < b_end and b_start < a_end
        for i, (a_start, a_end) in enumerate(call_windows)
        for b_start, b_end in call_windows[i + 1 :]
    )
    assert overlap_found, f"no overlapping Supervisor calls found across {len(call_windows)} calls - looks sequential"
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
