"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.graph.master_graph import resume_and_finalize, run_map_phase
from app.graph.state import CompanyJobStatus
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


async def _run_map_phase(pg_pool, fake_scout_finding, job_id, companies):
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        await run_map_phase(job_id=job_id, companies=companies, pool=pg_pool)


@pytest.mark.asyncio
async def test_client_company_triggers_strategy_mode_and_gets_saved_as_such(pg_pool, fake_scout_finding):
    job_id = f"strategy-test-{uuid.uuid4()}"
    companies = ["Acme", "Rival Co"]
    await _run_map_phase(pg_pool, fake_scout_finding, job_id, companies)

    with (
        patch("app.graph.master_graph._build_client_strategy", new=AsyncMock(return_value="# Strategy doc")) as mock_strategy,
        patch("app.graph.master_graph._build_comparison_matrix", new=AsyncMock(return_value="# Comparison doc")) as mock_comparison,
    ):
        result = await resume_and_finalize(job_id=job_id, companies=companies, pool=pg_pool, client_company="Acme")

    mock_strategy.assert_called_once()
    mock_comparison.assert_not_called()
    assert result.comparison_matrix == "# Strategy doc"
    assert result.company_statuses["Acme"] == CompanyJobStatus.DONE
    assert result.company_statuses["Rival Co"] == CompanyJobStatus.DONE


@pytest.mark.asyncio
async def test_no_client_company_falls_back_to_neutral_comparison(pg_pool, fake_scout_finding):
    job_id = f"strategy-test-{uuid.uuid4()}"
    companies = ["Acme", "Rival Co"]
    await _run_map_phase(pg_pool, fake_scout_finding, job_id, companies)

    with (
        patch("app.graph.master_graph._build_client_strategy", new=AsyncMock(return_value="# Strategy doc")) as mock_strategy,
        patch("app.graph.master_graph._build_comparison_matrix", new=AsyncMock(return_value="# Comparison doc")) as mock_comparison,
    ):
        result = await resume_and_finalize(job_id=job_id, companies=companies, pool=pg_pool, client_company=None)

    mock_comparison.assert_called_once()
    mock_strategy.assert_not_called()
    assert result.comparison_matrix == "# Comparison doc"


@pytest.mark.asyncio
async def test_client_company_not_among_approved_falls_back_to_comparison(pg_pool, fake_scout_finding):
    """client_company was set on /run but the reviewer didn't approve that
    company on /approve - strategy mode requires the client's own report to
    compare against, so this must gracefully fall back, not error."""
    job_id = f"strategy-test-{uuid.uuid4()}"
    # Only "Rival Co" gets researched/approved - "Acme" (the named client) never is.
    await _run_map_phase(pg_pool, fake_scout_finding, job_id, ["Rival Co"])

    with (
        patch("app.graph.master_graph._build_client_strategy", new=AsyncMock(return_value="# Strategy doc")) as mock_strategy,
        patch("app.graph.master_graph._build_comparison_matrix", new=AsyncMock(return_value="# Comparison doc")) as mock_comparison,
    ):
        result = await resume_and_finalize(
            job_id=job_id, companies=["Rival Co"], pool=pg_pool, client_company="Acme"
        )

    mock_comparison.assert_called_once()
    mock_strategy.assert_not_called()
    assert result.comparison_matrix == "# Comparison doc"
