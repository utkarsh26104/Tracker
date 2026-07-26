"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.master_graph import retry_company, run_company
from app.graph.state import CompanyJobStatus, RouteDecision, WriterOutput
from tests.conftest import make_content_driven_supervisor_llm


@pytest.mark.asyncio
async def test_retry_redirects_back_to_search_and_preserves_prior_findings(pg_pool, fake_scout_finding):
    job_id = f"retry-test-{uuid.uuid4()}"
    company = "Acme"

    # Phase 1: a normal run that reaches a first draft.
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm") as mock_writer_llm,
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        first_output = MagicMock()
        first_output.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=WriterOutput(report_markdown="# Acme\nFirst draft.", sufficient_data=True)
        )
        mock_writer_llm.return_value = first_output

        _, first_result = await run_company(pg_pool, company, job_id)

    assert first_result["final_report"] == "# Acme\nFirst draft."
    assert first_result["route_history"] == ["Search", "Write", "FINISH"]
    first_scouted_count = len(first_result["scouted_data"])
    assert first_scouted_count == 1

    # Phase 2: reviewer isn't happy - retry. A fresh scripted supervisor/writer
    # drives the redirected loop; a second (different-URL) scout finding
    # confirms new results get ADDED to, not replace, what was already found.
    second_finding = fake_scout_finding.model_copy(update={"source_url": "https://example.com/second-finding"})

    supervisor_calls = iter(
        [
            RouteDecision(next="Write", reasoning="have more data now"),
            RouteDecision(next="FINISH", reasoning="done"),
        ]
    )

    async def _supervisor_invoke(_messages):
        return next(supervisor_calls)

    fake_supervisor_structured = MagicMock()
    fake_supervisor_structured.ainvoke = AsyncMock(side_effect=_supervisor_invoke)
    fake_supervisor_llm = MagicMock()
    fake_supervisor_llm.with_structured_output.return_value = fake_supervisor_structured

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=fake_supervisor_llm),
        patch("app.graph.writer.get_writer_llm") as mock_writer_llm2,
        patch("app.graph.scout.search_company", return_value=[second_finding]),
    ):
        second_output = MagicMock()
        second_output.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=WriterOutput(report_markdown="# Acme\nBetter second draft.", sufficient_data=True)
        )
        mock_writer_llm2.return_value = second_output

        retry_result = await retry_company(pg_pool, job_id=job_id, company=company)

    assert retry_result["final_report"] == "# Acme\nBetter second draft."
    # The forced redirect entry plus the new loop's own decisions, appended
    # to (not replacing) the original history.
    assert retry_result["route_history"] == [
        "Search",
        "Write",
        "FINISH",
        "Search (forced: user requested re-research)",
        "Write",
        "FINISH",
    ]
    # Both the original and the new finding are present - retry augments
    # prior research rather than discarding it.
    assert len(retry_result["scouted_data"]) == first_scouted_count + 1
    urls = {f.source_url for f in retry_result["scouted_data"]}
    assert fake_scout_finding.source_url in urls
    assert second_finding.source_url in urls


@pytest.mark.asyncio
async def test_retry_on_unknown_thread_raises_value_error(pg_pool):
    with pytest.raises(ValueError):
        await retry_company(pg_pool, job_id="nonexistent-job", company="NoSuchCompany")
