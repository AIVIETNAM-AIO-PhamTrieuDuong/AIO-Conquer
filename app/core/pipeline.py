from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, END
from redis.exceptions import ResponseError

from app.api.schemas import AskRequest
from app.core.config import settings
from app.graph.nodes import (
    GraphState,
    node_basic_statistical_summary,
    node_coding_tool_planner,
    node_column_metadata,
    node_custom_metric,
    node_domain_context_planner,
    node_generate,
    node_load_domain_context,
    node_load_eda_context,
    node_load_history,
    node_load_meta_memory,
    node_missingness_summary,
    node_orchestrator_router,
    node_parse,
    node_query_builder,
    node_save_memory,
    node_save_meta_memory,
    node_statistical_association,
    node_type_compatibility,
)
from app.memory.context_store import context_store

SESSION_ID = "default"
_checkpointer: Any | None = None
_graph: Any | None = None


def _build_graph(checkpointer: Any):
    """Build the active QA LangGraph with deterministic tool nodes."""
    g = StateGraph(GraphState)

    g.add_node("load_history", node_load_history)
    g.add_node("load_eda_context", node_load_eda_context)
    g.add_node("load_domain_context", node_load_domain_context)
    g.add_node("route_multivariate", node_route_multivariate)
    g.add_node("load_meta_memory", node_load_meta_memory)
    g.add_node("column_metadata", node_column_metadata)
    g.add_node("missingness_summary", node_missingness_summary)
    g.add_node("type_compatibility", node_type_compatibility)
    g.add_node("basic_statistical_summary", node_basic_statistical_summary)
    g.add_node("statistical_association", node_statistical_association)
    g.add_node("custom_metric", node_custom_metric)
    g.add_node("generate", node_generate)
    g.add_node("parse", node_parse)
    g.add_node("save_meta_memory", node_save_meta_memory)
    g.add_node("save_memory", node_save_memory)

    g.set_entry_point("load_history")
    g.add_edge("load_history", "load_eda_context")
    g.add_edge("load_eda_context", "load_domain_context")
    g.add_edge("load_domain_context", "route_multivariate")
    g.add_edge("route_multivariate", "load_meta_memory")
    g.add_edge("load_meta_memory", "column_metadata")
    g.add_edge("column_metadata", "missingness_summary")
    g.add_edge("missingness_summary", "type_compatibility")
    g.add_edge("type_compatibility", "basic_statistical_summary")
    g.add_edge("basic_statistical_summary", "statistical_association")
    g.add_edge("statistical_association", "custom_metric")
    g.add_edge("custom_metric", "generate")
    g.add_edge("generate", "parse")
    g.add_edge("parse", "save_meta_memory")
    g.add_edge("save_meta_memory", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=checkpointer)


def _build_multi_agent_graph(checkpointer: Any):
    """Build the reserved inactive Milestone 3 multi-agent graph."""
    g = StateGraph(GraphState)

    g.add_node("load_history", node_load_history)
    g.add_node("load_eda_context", node_load_eda_context)
    g.add_node("load_domain_context", node_load_domain_context)
    g.add_node("load_meta_memory", node_load_meta_memory)
    g.add_node("orchestrator_router", node_orchestrator_router)
    g.add_node("domain_context_planner", node_domain_context_planner)
    g.add_node("query_builder", node_query_builder)
    g.add_node("coding_tool_planner", node_coding_tool_planner)
    g.add_node("column_metadata", node_column_metadata)
    g.add_node("missingness_summary", node_missingness_summary)
    g.add_node("type_compatibility", node_type_compatibility)
    g.add_node("basic_statistical_summary", node_basic_statistical_summary)
    g.add_node("statistical_association", node_statistical_association)
    g.add_node("custom_metric", node_custom_metric)
    g.add_node("generate", node_generate)
    g.add_node("parse", node_parse)
    g.add_node("save_meta_memory", node_save_meta_memory)
    g.add_node("save_memory", node_save_memory)

    g.set_entry_point("load_history")
    g.add_edge("load_history", "load_eda_context")
    g.add_edge("load_eda_context", "load_domain_context")
    g.add_edge("load_domain_context", "load_meta_memory")
    g.add_edge("load_meta_memory", "orchestrator_router")
    g.add_edge("orchestrator_router", "domain_context_planner")
    g.add_edge("domain_context_planner", "query_builder")
    g.add_edge("query_builder", "coding_tool_planner")
    g.add_edge("coding_tool_planner", "column_metadata")
    g.add_edge("column_metadata", "missingness_summary")
    g.add_edge("missingness_summary", "type_compatibility")
    g.add_edge("type_compatibility", "basic_statistical_summary")
    g.add_edge("basic_statistical_summary", "statistical_association")
    g.add_edge("statistical_association", "custom_metric")
    g.add_edge("custom_metric", "generate")
    g.add_edge("generate", "parse")
    g.add_edge("parse", "save_meta_memory")
    g.add_edge("save_meta_memory", "save_memory")
    g.add_edge("save_memory", END)

    return g.compile(checkpointer=checkpointer)


async def _get_checkpointer() -> Any:
    """Create and initialize the Redis checkpointer once per process."""
    global _checkpointer
    if _checkpointer is None:
        redis_checkpointer = RedisSaver(redis_url=settings.redis_url)
        try:
            redis_checkpointer.setup()
        except ResponseError as exc:
            if not _is_missing_redis_stack_command(exc):
                raise
            _checkpointer = _build_memory_checkpointer()
        else:
            _checkpointer = redis_checkpointer
    return _checkpointer


def _is_missing_redis_stack_command(exc: ResponseError) -> bool:
    """Return whether Redis lacks commands required by Redis Stack."""
    message = str(exc).lower()
    return "unknown command" in message and "ft." in message


def _build_memory_checkpointer() -> Any:
    """Build an in-process checkpointer when Redis Stack is unavailable."""
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


async def _get_graph() -> Any:
    """Compile the QA graph with Redis checkpoint persistence."""
    global _graph
    if _graph is None:
        # _graph = _build_graph(await _get_checkpointer())
        _graph = _build_multi_agent_graph(await _get_checkpointer())
    return _graph


async def reset_conversation_thread(thread_id: str = SESSION_ID) -> None:
    """Delete checkpointed conversation state for one LangGraph thread."""
    checkpointer = await _get_checkpointer()
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(thread_id)
        return
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)
        return
    for attr in ("storage", "writes", "blobs"):
        container = getattr(checkpointer, attr, None)
        if isinstance(container, dict):
            container.pop(thread_id, None)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def run_qa_pipeline(request: AskRequest) -> GraphState:
    """Run the QA graph and return the final checkpoint-safe graph state."""
    thread_id = request.thread_id or SESSION_ID
    run_id = str(uuid.uuid4())
    initial: GraphState = {
        "question": request.question,
        "session_id": thread_id,
        "run_id": run_id,
        "context": "",
        "domain_context": [],
        "domain_requirements": {},
        "multivariate_index": [],
        "multivariate_selected": [],
        "tool_memory": [],
        "agent_working_memory": {},
        "curated_context": [],
        "error_memory": [],
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
    config = {"configurable": {"thread_id": thread_id}}
    try:
        final = await (await _get_graph()).ainvoke(initial, config)
    except Exception as exc:
        try:
            await context_store.error_memory.append(
                scope_id=thread_id,
                thread_id=thread_id,
                run_id=run_id,
                source_node="run_qa_pipeline",
                record={
                    "event_type": "pipeline_exception",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                    "retry_context": {
                        "thread_id": thread_id,
                        "question": request.question,
                    },
                    "provenance": {"source": "run_qa_pipeline"},
                },
            )
        except Exception:
            pass
        raise
    return final
