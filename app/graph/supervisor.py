from app.graph.state import AgentState, RouteDecision
from app.llm.groq_client import get_supervisor_llm

SYSTEM_PROMPT = """You are the Supervisor of a competitive-intelligence research team.
Route to exactly one of: Search, Analyze, Write, FINISH.

- Search: fetch fresh web data via the Scout agent (use when scouted_data is thin or stale).
- Analyze: embed/retrieve historical context and sentiment via the Brain agent \
(use after new data has been scouted but not yet analyzed).
- Write: draft the executive report via the Writer agent (use once you have enough \
scouted + historical context).
- FINISH: use once final_report has been produced and is sufficient.

Respond with a routing decision, brief reasoning, and optional instructions for the next agent."""


def _summarize_state(state: AgentState) -> str:
    return (
        f"Company: {state['company']}\n"
        f"Scouted findings: {len(state['scouted_data'])}\n"
        f"Historical matches: {len(state['historical_context'])}\n"
        f"Sentiment summary set: {state['sentiment_summary'] is not None}\n"
        f"Final report drafted: {state['final_report'] is not None}\n"
        f"Insufficient data flagged by Writer: {state['insufficient_data_flag']}\n"
        f"Writer's note on what's missing: {state['pending_instructions']}\n"
        f"Route history so far: {state['route_history']}\n"
        f"Loop {state['loop_count']} of max {state['max_loops']}"
    )


async def supervisor_node(state: AgentState) -> dict:
    loop_count = state["loop_count"] + 1

    # Deterministic guardrail: a cheap routing model can fail to emit FINISH reliably.
    # Force termination at the cap instead of risking an infinite cyclic loop.
    if loop_count > state["max_loops"]:
        forced = "Write" if state["final_report"] is None else "FINISH"
        return {
            "route_history": state["route_history"] + [f"{forced} (forced: loop cap reached)"],
            "loop_count": loop_count,
        }

    # method="json_schema" + strict=True uses Groq's constrained decoding,
    # which guarantees schema-valid JSON - the default "function_calling"
    # method is best-effort even on models (like gpt-oss-120b) that support
    # strict mode, and was observed to occasionally emit unparseable JSON.
    llm = get_supervisor_llm().with_structured_output(RouteDecision, method="json_schema", strict=True)
    decision: RouteDecision = await llm.ainvoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", _summarize_state(state)),
        ]
    )

    return {
        "route_history": state["route_history"] + [decision.next],
        "pending_instructions": decision.instructions,
        "loop_count": loop_count,
        "messages": [("assistant", f"Route: {decision.next}. Reasoning: {decision.reasoning}")],
    }


def route_from_supervisor(state: AgentState) -> str:
    last = state["route_history"][-1]
    if last.startswith("Search"):
        return "scout"
    if last.startswith("Analyze"):
        return "brain"
    if last.startswith("Write"):
        return "writer"
    return "__end__"
