"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.db.checkpointer import make_checkpointer
from app.graph.build_graph import build_graph
from app.graph.master_graph import resume_and_finalize, run_map_phase
from app.graph.state import CompanyJobStatus
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm

COMPANIES = ["Stripe", "Adyen"]


@pytest.mark.asyncio
async def test_graphs_pause_before_publish_and_approval_resumes_them(pg_pool, fake_scout_finding):
    job_id = f"test-hitl-{uuid.uuid4()}"

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
        patch("app.graph.master_graph.get_writer_llm") as mock_reduce_llm,
    ):
        reduce_response = AsyncMock()
        reduce_response.content = "# Comparison Matrix"
        mock_reduce_llm.return_value.ainvoke = AsyncMock(return_value=reduce_response)

        map_state = await run_map_phase(job_id=job_id, companies=COMPANIES, pool=pg_pool)

        assert all(s == CompanyJobStatus.AWAITING_APPROVAL for s in map_state.company_statuses.values())
        assert map_state.comparison_matrix is None

        # Confirm LangGraph itself reports these runs as paused before publish_report.
        graph = build_graph(checkpointer=make_checkpointer(pg_pool))
        for company in COMPANIES:
            config = {"configurable": {"thread_id": f"{job_id}:{company}"}}
            snapshot = await graph.aget_state(config)
            assert snapshot.next == ("publish_report",)

        # Approve only Stripe; Adyen is implicitly rejected by omission.
        final_state = await resume_and_finalize(job_id=job_id, companies=["Stripe"], pool=pg_pool)

    assert final_state.company_statuses["Stripe"] == CompanyJobStatus.DONE
    assert "Adyen" not in final_state.company_statuses
    assert set(final_state.company_reports.keys()) == {"Stripe"}
    assert final_state.comparison_matrix is not None

    # The rejected company's graph must remain untouched, still paused.
    config = {"configurable": {"thread_id": f"{job_id}:Adyen"}}
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("publish_report",)
