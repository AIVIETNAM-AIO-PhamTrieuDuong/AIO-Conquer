from __future__ import annotations


from langgraph.graph import StateGraph, END
from app.api.schemas import AskRequest, QAResponse
from app.graph.nodes import (
    GraphState,
    node_load_history,
    node_load_eda_context,
    node_generate,
    node_parse,
    node_save_memory
)
SESSION_ID = "default"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph():
    """Build the QA LangGraph with the shared graph state schema."""
    g = StateGraph(GraphState)

    g.add_node("load_history", node_load_history)
    g.add_node("load_eda_context", node_load_eda_context)
    g.add_node("generate", node_generate)
    g.add_node("parse", node_parse)
    g.add_node("save_memory", node_save_memory)

    g.set_entry_point("load_history")
    g.add_edge("load_history", "load_eda_context")
    g.add_edge("load_eda_context", "generate")
    g.add_edge("generate", "parse")
    g.add_edge("parse", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def run_qa_pipeline(request: AskRequest) -> QAResponse:
    """Run the QA graph and return its parsed response."""
    initial: GraphState = {
        "question": request.question,
        "session_id": SESSION_ID,
        "history": [],
        "context": "",
        "prompt": "",
        "raw_response": "",
        "response": None,
    }
    final = await _graph.ainvoke(initial)
    return final["response"]
