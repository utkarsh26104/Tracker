import asyncio

from app.graph.state import AgentState
from app.services.scraping import search_company


async def scout_node(state: AgentState) -> dict:
    # tavily-python has no async client - offload the blocking HTTP call so it
    # doesn't stall the event loop (and other concurrently-running companies'
    # graphs in M4's Map-Reduce).
    new_findings = await asyncio.to_thread(
        search_company, state["company"], state["pending_instructions"], state["search_days"]
    )

    seen_urls = {f.source_url for f in state["scouted_data"]}
    merged = state["scouted_data"] + [f for f in new_findings if f.source_url not in seen_urls]

    return {"scouted_data": merged, "pending_instructions": None}
