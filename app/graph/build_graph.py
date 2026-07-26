from langgraph.graph import END, StateGraph

from app.graph.brain import brain_node
from app.graph.publish import publish_report_node
from app.graph.scout import scout_node
from app.graph.state import AgentState
from app.graph.supervisor import route_from_supervisor, supervisor_node
from app.graph.writer import writer_node

# M1 thin slice: Supervisor -> Scout -> Writer only, no Brain/RAG, no HITL gate.
# "Analyze" routes straight to Write since there's no Brain node in this graph -
# kept around for the wiring smoke test, not used by the API anymore past M2.
_THIN_SLICE_ROUTES = {
    "scout": "scout",
    "brain": "writer",
    "writer": "writer",
    "__end__": END,
}

_FULL_ROUTES = {
    "scout": "scout",
    "brain": "brain",
    "writer": "writer",
    "__end__": "publish_report",
}


def build_thin_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("scout", scout_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, _THIN_SLICE_ROUTES)
    graph.add_edge("scout", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


def build_graph(checkpointer=None):
    """M2+: full per-company graph with Brain/RAG wired in.

    Pass a LangGraph checkpointer (e.g. AsyncPostgresSaver) to persist state
    across process restarts, keyed by thread_id - required for M3
    resumability and M5's HITL interrupt/resume flow. Without one, state
    only lives in memory for the duration of a single .invoke() call and the
    interrupt_before pause below can't meaningfully be resumed later.

    The graph pauses right before publish_report (M5's HITL gate) once the
    Supervisor decides FINISH - a human must call POST /tracker/approve to
    let the run continue to END and be eligible for the Reduce-phase
    Comparison Matrix.
    """
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("scout", scout_node)
    graph.add_node("brain", brain_node)
    graph.add_node("writer", writer_node)
    graph.add_node("publish_report", publish_report_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, _FULL_ROUTES)
    graph.add_edge("scout", "supervisor")
    graph.add_edge("brain", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("publish_report", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["publish_report"])
