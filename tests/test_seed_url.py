"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.master_graph import run_map_phase
from app.graph.state import RouteDecision, ScoutFinding
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


def _never_satisfied_supervisor_llm():
    """Always routes Search, except once a report exists (routes FINISH) -
    used to deterministically drive a company all the way to the loop cap
    without ever letting the Supervisor's own judgment call Write."""

    async def _ainvoke(messages):
        human_text = messages[1][1]
        if "Final report drafted: True" in human_text:
            return RouteDecision(next="FINISH", reasoning="done")
        return RouteDecision(next="Search", reasoning="still not enough")

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


@pytest.mark.asyncio
async def test_seed_url_populates_scouted_data_before_scout_runs(pg_pool):
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"LocalBrand-{uuid.uuid4()}"
    seed_findings = [
        ScoutFinding(
            source_url="https://localbrand.example",
            title="LocalBrand - Home",
            snippet="LocalBrand sells handmade candles across India.",
            fetched_at=datetime.now(timezone.utc),
        ),
        ScoutFinding(
            source_url="https://localbrand.example/reviews",
            title="Customer Reviews",
            snippet="Average rating 4.7 out of 5.",
            fetched_at=datetime.now(timezone.utc),
        ),
    ]

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[]),
        patch("app.graph.master_graph.fetch_company_site_findings", return_value=seed_findings) as fetch_mock,
        patch("app.graph.master_graph.search_marketplace_reviews", return_value=[]) as marketplace_mock,
        patch("app.graph.master_graph.search_social_media_presence", return_value=[]) as social_mock,
    ):
        result = await run_map_phase(
            job_id=job_id,
            companies=[company],
            pool=pg_pool,
            company_urls={company: "https://localbrand.example"},
        )

    fetch_mock.assert_called_once_with("https://localbrand.example")
    marketplace_mock.assert_called_once_with(company)
    social_mock.assert_called_once_with(company)
    assert result.company_statuses[company].value == "awaiting_approval"
    assert result.company_route_histories[company][0] == (
        "Seeded with 2 page(s) from provided URL: https://localbrand.example"
    )


@pytest.mark.asyncio
async def test_marketplace_findings_are_included_and_noted_in_route_history(pg_pool):
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"LocalBrand-{uuid.uuid4()}"
    site_findings = [
        ScoutFinding(
            source_url="https://localbrand.example",
            title="LocalBrand - Home",
            snippet="LocalBrand sells handmade candles.",
            fetched_at=datetime.now(timezone.utc),
        )
    ]
    marketplace_findings = [
        ScoutFinding(
            source_url="https://www.amazon.in/dp/xyz",
            title="LocalBrand Candle Set - Amazon.in",
            snippet="4.3 out of 5 stars, 1,204 ratings. Great scent, arrived on time.",
            fetched_at=datetime.now(timezone.utc),
        ),
    ]
    social_findings = [
        ScoutFinding(
            source_url="https://www.instagram.com/p/xyz",
            title="LocalBrand on Instagram",
            snippet="New Winter Collection drop - 20% off this week only!",
            fetched_at=datetime.now(timezone.utc),
        ),
    ]

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[]),
        patch("app.graph.master_graph.fetch_company_site_findings", return_value=site_findings),
        patch("app.graph.master_graph.search_marketplace_reviews", return_value=marketplace_findings),
        patch("app.graph.master_graph.search_social_media_presence", return_value=social_findings),
    ):
        result = await run_map_phase(
            job_id=job_id,
            companies=[company],
            pool=pg_pool,
            company_urls={company: "https://localbrand.example"},
        )

    assert result.company_route_histories[company][0] == (
        "Seeded with 1 page(s) from provided URL: https://localbrand.example "
        "and 1 marketplace listing(s)/review(s) and 1 social media post(s)"
    )


@pytest.mark.asyncio
async def test_failed_seed_url_fetch_does_not_break_the_run(pg_pool, fake_scout_finding):
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"LocalBrand-{uuid.uuid4()}"

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
        patch("app.graph.master_graph.fetch_company_site_findings", return_value=[]),
        patch("app.graph.master_graph.search_marketplace_reviews", return_value=[]),
        patch("app.graph.master_graph.search_social_media_presence", return_value=[]),
    ):
        result = await run_map_phase(
            job_id=job_id,
            companies=[company],
            pool=pg_pool,
            company_urls={company: "https://unreachable.example"},
        )

    # A failed fetch shouldn't prevent the normal Scout-driven pipeline from running.
    assert result.company_statuses[company].value == "awaiting_approval"
    assert not result.company_route_histories[company][0].startswith("Seeded with")


@pytest.mark.asyncio
async def test_no_seed_url_skips_the_fetch_entirely(pg_pool, fake_scout_finding):
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"Rival-{uuid.uuid4()}"

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
        patch("app.graph.master_graph.fetch_company_site_findings") as fetch_mock,
        patch("app.graph.master_graph.search_marketplace_reviews") as marketplace_mock,
        patch("app.graph.master_graph.search_social_media_presence") as social_mock,
    ):
        await run_map_phase(job_id=job_id, companies=[company], pool=pg_pool)

    fetch_mock.assert_not_called()
    marketplace_mock.assert_not_called()
    social_mock.assert_not_called()


@pytest.mark.asyncio
async def test_seeded_company_still_produces_a_report_when_loop_cap_is_hit(pg_pool):
    """End-to-end: a seeded company whose Supervisor never feels satisfied
    (always routes Search) still gets a published report once the loop cap
    forces a final Write, instead of failing outright - see writer.py's
    FORCED_FINAL_ATTEMPT_INSTRUCTION and the deterministic override in
    writer_node."""
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"LocalBrand-{uuid.uuid4()}"
    site_findings = [
        ScoutFinding(
            source_url="https://localbrand.example",
            title="LocalBrand - Home",
            snippet="LocalBrand sells handmade candles.",
            fetched_at=datetime.now(timezone.utc),
        )
    ]

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=_never_satisfied_supervisor_llm()),
        patch(
            "app.graph.writer.get_writer_llm",
            return_value=make_scripted_writer_llm(
                "# LocalBrand\nThin but real data from its own site.", sufficient_data=False
            ),
        ),
        patch("app.graph.scout.search_company", return_value=[]),  # Tavily finds nothing, every loop
        patch("app.graph.master_graph.fetch_company_site_findings", return_value=site_findings),
        patch("app.graph.master_graph.search_marketplace_reviews", return_value=[]),
        patch("app.graph.master_graph.search_social_media_presence", return_value=[]),
    ):
        result = await run_map_phase(
            job_id=job_id,
            companies=[company],
            pool=pg_pool,
            company_urls={company: "https://localbrand.example"},
        )

    assert result.company_statuses[company].value == "awaiting_approval"
    assert result.company_reports[company] == "# LocalBrand\nThin but real data from its own site."


@pytest.mark.asyncio
async def test_client_company_gets_the_same_seed_url_treatment_as_a_competitor(pg_pool):
    """The client field accepts the same `Name | https://url` syntax the
    Competitors textarea does (ui/streamlit_app.py) - confirms the backend
    side of that: run_map_phase's _run() only special-cases client_company
    for the report-cache/dossier shortcuts, and falls through to the exact
    same run_company(..., seed_url=...) call used for any competitor once
    neither shortcut applies (fresh company name here, so neither does)."""
    job_id = f"seed-url-test-{uuid.uuid4()}"
    company = f"ClientBrand-{uuid.uuid4()}"
    site_findings = [
        ScoutFinding(
            source_url="https://clientbrand.example",
            title="ClientBrand - Home",
            snippet="ClientBrand sells artisanal soap across India.",
            fetched_at=datetime.now(timezone.utc),
        )
    ]

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch("app.graph.writer.get_writer_llm", return_value=make_scripted_writer_llm()),
        patch("app.graph.scout.search_company", return_value=[]),
        patch("app.graph.master_graph.fetch_company_site_findings", return_value=site_findings) as fetch_mock,
        patch("app.graph.master_graph.search_marketplace_reviews", return_value=[]) as marketplace_mock,
        patch("app.graph.master_graph.search_social_media_presence", return_value=[]) as social_mock,
    ):
        result = await run_map_phase(
            job_id=job_id,
            companies=[company],
            pool=pg_pool,
            client_company=company,
            company_urls={company: "https://clientbrand.example"},
        )

    fetch_mock.assert_called_once_with("https://clientbrand.example")
    marketplace_mock.assert_called_once_with(company)
    social_mock.assert_called_once_with(company)
    assert result.company_statuses[company].value == "awaiting_approval"
    assert result.company_route_histories[company][0] == (
        "Seeded with 1 page(s) from provided URL: https://clientbrand.example"
    )
