from app.graph.state import AgentState


async def publish_report_node(state: AgentState) -> dict:
    """No-op - the meaningful behavior is the interrupt_before gate placed in
    front of this node (see build_graph.py), which pauses the graph here
    until a human approves via POST /tracker/approve. Once resumed, reaching
    this node simply means the report is cleared for inclusion in the
    Reduce-phase Comparison Matrix."""
    return {}
