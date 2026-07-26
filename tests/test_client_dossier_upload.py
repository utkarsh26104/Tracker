"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py). Also touches real ChromaDB/
FinancialBERT via client_uploads.py (no mocks there) - slower than the
purely-mocked graph tests, similar to test_memory.py."""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.master_graph import run_map_phase
from app.graph.state import RouteDecision, WriterOutput
from app.memory.client_uploads import upsert_client_upload
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


def _supervisor_llm_that_visits_analyze():
    """Unlike the shared make_content_driven_supervisor_llm (which routes
    straight from Search to Write, never visiting Brain), this one detours
    through Analyze first - needed here because Brain is what populates
    client_upload_context, and the mix-with-stale-upload test needs to see
    it actually reach the Writer's prompt."""

    async def _ainvoke(messages):
        human_text = messages[1][1]
        if "Final report drafted: True" in human_text:
            return RouteDecision(next="FINISH", reasoning="done")
        if "Scouted findings: 0" in human_text:
            return RouteDecision(next="Search", reasoning="need data first")
        if "Sentiment summary set: False" in human_text:
            return RouteDecision(next="Analyze", reasoning="analyze before writing")
        return RouteDecision(next="Write", reasoning="ready")

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


@pytest.mark.asyncio
async def test_fresh_upload_drafts_report_without_web_search(pg_pool, fake_scout_finding):
    company = f"Acme-{uuid.uuid4()}"
    job_id = f"dossier-test-{uuid.uuid4()}"
    upsert_client_upload(company, "dossier.txt", f"{company} is a DTC supplements brand.")

    scout_mock = MagicMock(return_value=[fake_scout_finding])

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch(
            "app.graph.master_graph._draft_report_from_upload",
            new=AsyncMock(return_value="# Drafted From Dossier"),
        ),
        patch("app.graph.scout.search_company", scout_mock),
    ):
        result = await run_map_phase(
            job_id=job_id, companies=[company, "Rival Co"], pool=pg_pool, client_company=company
        )

    searched = {call.args[0] for call in scout_mock.call_args_list}
    assert company not in searched  # fresh upload means no web search for the client
    assert "Rival Co" in searched  # competitor still gets researched fresh

    assert result.company_reports[company] == "# Drafted From Dossier"
    assert result.company_route_histories[company][0].startswith(
        "Reused cached report (drafted from an uploaded dossier"
    )


@pytest.mark.asyncio
async def test_stale_upload_still_gets_mixed_with_fresh_search(pg_pool, fake_scout_finding):
    company = f"Acme-{uuid.uuid4()}"
    job_id = f"dossier-test-{uuid.uuid4()}"
    marker = "UNIQUE_DOSSIER_MARKER_ABOUT_GADGETS"
    upsert_client_upload(
        company,
        "dossier.txt",
        f"{company} background: {marker}.",
        uploaded_at=datetime.now(timezone.utc) - timedelta(days=45),
    )

    captured_messages = []

    async def _capturing_ainvoke(messages):
        captured_messages.append(messages)
        return WriterOutput(report_markdown="# Report\nSome findings.", sufficient_data=True)

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_capturing_ainvoke)
    capturing_writer_llm = MagicMock()
    capturing_writer_llm.with_structured_output.return_value = fake_structured

    scout_mock = MagicMock(return_value=[fake_scout_finding])

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=_supervisor_llm_that_visits_analyze()),
        patch("app.graph.writer.get_writer_llm", return_value=capturing_writer_llm),
        patch("app.graph.scout.search_company", scout_mock),
    ):
        result = await run_map_phase(
            job_id=job_id, companies=[company, "Rival Co"], pool=pg_pool, client_company=company
        )

    searched = {call.args[0] for call in scout_mock.call_args_list}
    assert company in searched  # a stale upload does NOT skip the fresh search
    assert not result.company_route_histories[company][0].startswith("Reused cached report")

    prompt_texts = [msg[1] for call_messages in captured_messages for msg in call_messages if msg[0] == "human"]
    assert any(marker in text for text in prompt_texts)  # the stale dossier still reached the Writer's prompt
