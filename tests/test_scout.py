from unittest.mock import patch

import pytest

from app.graph.scout import scout_node
from app.graph.state import new_agent_state


@pytest.mark.asyncio
async def test_scout_merges_new_findings(fake_scout_finding):
    state = new_agent_state(company="Acme", job_id="t1")

    with patch("app.graph.scout.search_company", return_value=[fake_scout_finding]):
        result = await scout_node(state)

    assert len(result["scouted_data"]) == 1
    assert result["scouted_data"][0].source_url == fake_scout_finding.source_url
    assert result["pending_instructions"] is None


@pytest.mark.asyncio
async def test_scout_dedupes_by_url(fake_scout_finding):
    state = new_agent_state(company="Acme", job_id="t1")
    state["scouted_data"] = [fake_scout_finding]

    with patch("app.graph.scout.search_company", return_value=[fake_scout_finding]):
        result = await scout_node(state)

    assert len(result["scouted_data"]) == 1  # not duplicated
