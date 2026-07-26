"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.db.history import get_recent_company_report, save_report
from app.graph.master_graph import run_map_phase
from app.graph.state import CompanyJobStatus
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


async def _run_map_phase_with_mocks(pg_pool, fake_scout_finding, **kwargs):
    scout_mock = MagicMock(return_value=[fake_scout_finding])
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", scout_mock),
    ):
        result = await run_map_phase(pool=pg_pool, **kwargs)
    searched_companies = {call.args[0] for call in scout_mock.call_args_list}
    return result, searched_companies


@pytest.mark.asyncio
async def test_client_with_recent_cached_report_skips_fresh_search(pg_pool, fake_scout_finding):
    job_id = f"cache-test-{uuid.uuid4()}"
    cached_content = "# Cached Acme Report\nAlready known."
    await save_report(pg_pool, job_id="prior-job", company="Acme", report_type="company", content=cached_content)

    result, searched = await _run_map_phase_with_mocks(
        pg_pool, fake_scout_finding, job_id=job_id, companies=["Acme", "Rival Co"], client_company="Acme"
    )

    assert result.company_reports["Acme"] == cached_content
    assert result.company_statuses["Acme"] == CompanyJobStatus.AWAITING_APPROVAL
    assert result.company_route_histories["Acme"][0].startswith("Reused cached report")
    assert "Acme" not in searched
    assert "Rival Co" in searched  # the competitor still gets researched fresh


@pytest.mark.asyncio
async def test_client_with_no_cached_report_runs_fresh_search(pg_pool, fake_scout_finding):
    job_id = f"cache-test-{uuid.uuid4()}"
    client_name = f"NewClient-{uuid.uuid4()}"

    result, searched = await _run_map_phase_with_mocks(
        pg_pool, fake_scout_finding, job_id=job_id, companies=[client_name, "Rival Co"], client_company=client_name
    )

    assert client_name in searched
    assert not result.company_route_histories[client_name][0].startswith("Reused cached report")


@pytest.mark.asyncio
async def test_recent_report_cache_never_applies_to_non_client_companies(pg_pool, fake_scout_finding):
    job_id = f"cache-test-{uuid.uuid4()}"
    await save_report(pg_pool, job_id="prior-job", company="Rival Co", report_type="company", content="# Old Rival Co Report")

    _, searched = await _run_map_phase_with_mocks(
        pg_pool, fake_scout_finding, job_id=job_id, companies=["Acme", "Rival Co"], client_company="Acme"
    )

    assert "Rival Co" in searched  # a recent report existing doesn't matter - only client_company is cached


@pytest.mark.asyncio
async def test_get_recent_company_report_ignores_entries_older_than_max_age(pg_pool):
    company = f"StaleCo-{uuid.uuid4()}"
    old_timestamp = datetime.now(timezone.utc) - timedelta(days=45)
    async with pg_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO report_history (job_id, company, report_type, content, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("old-job", company, "company", "# Stale Report", old_timestamp),
        )

    result = await get_recent_company_report(pg_pool, company, max_age_days=30)
    assert result is None
