from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.api.schemas import AskRequest, QAResponse
from app.graph.nodes import (
    GraphState,
    node_basic_statistical_summary,
    node_column_metadata,
    node_custom_metric,
    node_generate,
    node_load_eda_context,
    node_load_history,
    node_missingness_summary,
    node_parse,
    node_save_memory,
    node_statistical_association,
    node_type_compatibility,
)

SESSION_ID = "default"


def _build_graph():
    """Build the QA LangGraph with the shared graph state schema."""
    g = StateGraph(GraphState)

    g.add_node("load_history", node_load_history)
    g.add_node("load_eda_context", node_load_eda_context)
    g.add_node("column_metadata", node_column_metadata)
    g.add_node("missingness_summary", node_missingness_summary)
    g.add_node("type_compatibility", node_type_compatibility)
    g.add_node("basic_statistical_summary", node_basic_statistical_summary)
    g.add_node("statistical_association", node_statistical_association)
    g.add_node("custom_metric", node_custom_metric)
    g.add_node("generate", node_generate)
    g.add_node("parse", node_parse)
    g.add_node("save_memory", node_save_memory)

    g.set_entry_point("load_history")
    g.add_edge("load_history", "load_eda_context")
    g.add_edge("load_eda_context", "column_metadata")
    g.add_edge("column_metadata", "missingness_summary")
    g.add_edge("missingness_summary", "type_compatibility")
    g.add_edge("type_compatibility", "basic_statistical_summary")
    g.add_edge("basic_statistical_summary", "statistical_association")
    g.add_edge("statistical_association", "custom_metric")
    g.add_edge("custom_metric", "generate")
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
        "eda_result": {},
        "dataset_id": "",
        "dataset_file_path": "",
        "tool_requests": [],
        "tool_results": [],
        "statistical_findings": [],
        "warnings": [],
        "prompt": "",
        "raw_response": "",
        "response": None,
    }
    final = await _graph.ainvoke(initial)
    return final["response"]
