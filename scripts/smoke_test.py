"""In-process wiring smoke test for the M1 thin slice (Supervisor -> Scout -> Writer).

Mocks Bedrock and Tavily so it runs with no API keys and no network calls -
it validates graph wiring (routing, state accumulation, loop guard), not
model quality. Run scripts/live_check.py separately once real credentials
are configured to validate against actual Bedrock/Tavily.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.build_graph import build_thin_graph  # noqa: E402
from app.graph.state import RouteDecision, ScoutFinding, WriterOutput, new_agent_state  # noqa: E402


def _scripted_llm(outputs: list):
    """Returns a fake `.with_structured_output(Model).ainvoke(...)` chain that
    yields the given outputs in order."""
    calls = iter(outputs)

    async def _ainvoke(_messages):
        return next(calls)

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


async def run() -> dict:
    supervisor_outputs = [
        RouteDecision(next="Search", reasoning="Need fresh data first."),
        RouteDecision(next="Write", reasoning="Enough data scouted, draft the report."),
        RouteDecision(next="FINISH", reasoning="Report is complete."),
    ]
    writer_outputs = [
        WriterOutput(report_markdown="# Acme Corp\nAcme raised prices 5%.", sufficient_data=True),
    ]
    fake_finding = ScoutFinding(
        source_url="https://example.com/acme-news",
        title="Acme raises prices",
        snippet="Acme Corp announced a 5% price increase across its product line.",
        fetched_at=datetime.now(timezone.utc),
    )

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=_scripted_llm(supervisor_outputs)),
        patch("app.graph.writer.get_writer_llm", return_value=_scripted_llm(writer_outputs)),
        patch("app.graph.scout.search_company", return_value=[fake_finding]),
    ):
        graph = build_thin_graph()
        return await graph.ainvoke(new_agent_state(company="Acme Corp", job_id="smoke-test-1"))


def main() -> None:
    result = asyncio.run(run())

    assert result["final_report"], "expected a non-empty final_report"
    assert result["route_history"] == ["Search", "Write", "FINISH"], result["route_history"]
    assert len(result["scouted_data"]) == 1, "expected the scouted finding to be merged into state"

    print("PASS - route_history:", result["route_history"])
    print("PASS - final_report:\n", result["final_report"])


if __name__ == "__main__":
    main()
