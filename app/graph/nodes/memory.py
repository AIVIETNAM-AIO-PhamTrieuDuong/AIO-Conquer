"""Memory and context loading nodes for the QA graph."""

from app.core.config import settings
from app.graph.nodes.common import (
    append_context,
    as_list,
    extend_unique,
)
from app.graph.schema import GraphState
from app.memory.context_store import context_store
from app.memory.eda_store import eda_store
from app.memory.vector_store import vector_store


async def node_load_history(state: GraphState) -> dict:
    """Normalize checkpointed conversation history for the active thread."""
    return {"history": state.get("history", [])}


async def node_load_eda_context(state: GraphState) -> dict:
    """Load the active EDA summary and structured dataset context."""
    job_id = await eda_store.get_active_eda(state["session_id"])
    if not job_id:
        return {"context": "", "eda_result": {}, "dataset_file_path": ""}
    result = await eda_store.get_eda_result(job_id)
    if not result:
        return {
            "context": "",
            "eda_result": {},
            "dataset_id": job_id,
            "dataset_file_path": "",
        }
    return {
        "context": result.get("summary_md", ""),
        "eda_result": result,
        "dataset_id": job_id,
        "dataset_file_path": result.get("cleaned_file_path", ""),
    }


async def node_load_domain_context(state: GraphState) -> dict:
    """Retrieve dataset-scoped domain memory before tool execution."""
    job_id = state.get("dataset_id", "")
    if not job_id:
        return {"domain_context": [], "domain_requirements": {}}

    results = await vector_store.search(
        job_id=job_id,
        query=state["question"],
        memory_types=["domain_generated", "domain_custom"],
        top_k=3,
    )
    requirements = domain_requirements(results)
    return {
        "domain_context": results,
        "domain_requirements": requirements,
        "context": append_context(
            state.get("context", ""),
            domain_context_summary(results, requirements),
        ),
    }


async def node_load_meta_memory(state: GraphState) -> dict:
    """Load reusable meta-memory and initialize working memory."""
    scope_id = memory_scope_id(state)
    loaded = await context_store.load_meta_memory(scope_id)
    curated_context = loaded.get("curated_context", [])
    return {
        "tool_memory": [],
        "error_memory": [],
        "curated_context": curated_context,
        "agent_working_memory": initial_working_memory(state, loaded),
        "context": append_context(
            state.get("context", ""),
            curated_context_summary(curated_context),
        ),
    }


async def node_save_meta_memory(state: GraphState) -> dict:
    """Persist current-run meta-memory records to operational Redis."""
    scope_id = memory_scope_id(state)
    thread_id = state["session_id"]
    run_id = state.get("run_id", "")
    saved_tools = []
    saved_errors = []
    for record in state.get("tool_memory", []):
        saved_tools.append(
            await context_store.tool_memory.append(
                scope_id=scope_id,
                thread_id=thread_id,
                run_id=run_id,
                source_node=record.get("source_node", "tool_node"),
                record=record,
            )
        )
    for record in state.get("error_memory", []):
        saved_errors.append(
            await context_store.error_memory.append(
                scope_id=scope_id,
                thread_id=thread_id,
                run_id=run_id,
                source_node=record.get("source_node", "tool_node"),
                record=record,
            )
        )

    working_memory = final_working_memory(state)
    saved_working = await context_store.agent_working_memory.save(
        scope_id=scope_id,
        thread_id=thread_id,
        run_id=run_id,
        source_node="node_save_meta_memory",
        snapshot=working_memory,
    )
    curated_context = list(state.get("curated_context", []))
    curated_record = curated_context_record(state)
    if curated_record:
        saved_curated = await context_store.curated_context.append(
            scope_id=scope_id,
            thread_id=thread_id,
            run_id=run_id,
            source_node="node_save_meta_memory",
            record=curated_record,
        )
        curated_context.append(saved_curated)

    return {
        "tool_memory": saved_tools,
        "error_memory": saved_errors,
        "agent_working_memory": saved_working,
        "curated_context": curated_context[-20:],
    }


async def node_save_memory(state: GraphState) -> dict:
    """Append the final answer to checkpointed conversation history."""
    r = state["response"]
    if r:
        history = [
            *state.get("history", []),
            {"q": state["question"], "a": str(r.get("answer", ""))},
        ]
        return {"history": history[-settings.max_history_turns:]}
    return {"history": state.get("history", [])}


def memory_scope_id(state: GraphState) -> str:
    """Return the Redis meta-memory scope for this graph state."""
    return str(state.get("dataset_id") or state.get("session_id") or "default")


def initial_working_memory(
    state: GraphState,
    loaded: dict,
) -> dict:
    """Create the current run working-memory snapshot."""
    prior_working = loaded.get("agent_working_memory") or {}
    return {
        "objective": state["question"],
        "requirements": state.get("domain_requirements", {}),
        "assumptions": prior_working.get("assumptions", []),
        "intermediate_findings": [],
        "unresolved_questions": [],
        "selected_columns": [],
        "selected_metrics": [],
        "prior_context": {
            "tool_memory_count": len(loaded.get("tool_memory", [])),
            "recent_tool_memory": loaded.get("tool_memory", [])[-3:],
            "error_memory_count": len(loaded.get("error_memory", [])),
            "recent_errors": loaded.get("error_memory", [])[-3:],
            "curated_context_count": len(loaded.get("curated_context", [])),
        },
    }


def curated_context_summary(curated_context: list[dict]) -> str:
    """Format recent curated context for prompt augmentation."""
    if not curated_context:
        return ""

    lines = ["Curated context memory:"]
    for item in curated_context[-3:]:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        label = question or item.get("record_type", "context")
        if answer:
            lines.append(f"- {label}: {answer[:500]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def final_working_memory(state: GraphState) -> dict:
    """Build the final working-memory snapshot for persistence."""
    working = dict(state.get("agent_working_memory", {}))
    working["objective"] = state["question"]
    working["requirements"] = state.get("domain_requirements", {})
    working["analysis_intent"] = state.get("analysis_intent", {})
    working["query_plan"] = state.get("query_plan", {})
    working["coding_plan"] = state.get("coding_plan", {})
    working["agent_handoffs"] = state.get("agent_handoffs", [])[-20:]
    working["open_questions"] = state.get("open_questions", [])[-20:]
    working["warnings"] = state.get("warnings", [])
    working["statistical_findings"] = state.get("statistical_findings", [])[-20:]
    working["final_response_present"] = bool(state.get("response"))
    return working


def curated_context_record(state: GraphState) -> dict | None:
    """Build a system-curated final answer record when parsing succeeded."""
    response = state.get("response") or {}
    answer = str(response.get("answer", "")).strip()
    if not answer:
        return None

    return {
        "record_type": "qa_final_answer",
        "validation_status": "system_generated",
        "question": state["question"],
        "answer": answer,
        "explanation": response.get("explanation", ""),
        "confidence": response.get("confidence"),
        "dataset_id": state.get("dataset_id", ""),
        "provenance": {
            "domain_sources": [
                {
                    "memory_type": item.get("memory_type", ""),
                    "source_id": item.get("source_id", ""),
                    "score": item.get("score"),
                }
                for item in state.get("domain_context", [])
            ],
            "tool_memory": [
                {
                    "tool_name": item.get("tool_name", ""),
                    "request_id": item.get("request_id", ""),
                    "status": item.get("status", ""),
                }
                for item in state.get("tool_memory", [])
            ],
        },
    }


def domain_requirements(results: list[dict]) -> dict:
    """Merge domain vector metadata into tool-planning hints."""
    requirements = {
        "metrics": [],
        "features": [],
        "constraints": [],
        "sources": [],
    }
    for result in results:
        metadata = result.get("metadata", {})
        extend_unique(requirements["metrics"], as_list(metadata.get("metrics")))
        extend_unique(requirements["features"], as_list(metadata.get("features")))
        extend_unique(
            requirements["constraints"],
            as_list(metadata.get("constraints")),
        )
        requirements["sources"].append(
            {
                "memory_type": result.get("memory_type", ""),
                "source_type": result.get("source_type", ""),
                "source_id": result.get("source_id", ""),
                "title": result.get("title", ""),
                "score": result.get("score"),
            }
        )
    return requirements


def domain_context_summary(results: list[dict], requirements: dict) -> str:
    """Format retrieved domain memory as compact prompt context."""
    if not results:
        return ""

    lines = ["Domain memory:"]
    if requirements.get("features"):
        lines.append(f"Feature hints: {', '.join(requirements['features'])}")
    if requirements.get("metrics"):
        lines.append(f"Metric hints: {', '.join(requirements['metrics'])}")
    if requirements.get("constraints"):
        lines.append(f"Constraints: {', '.join(requirements['constraints'])}")

    for result in results[:3]:
        label = result.get("title") or result.get("source_id") or "domain"
        text = " ".join(result.get("text", "").split())
        lines.append(
            f"- {label} [{result.get('memory_type', '')}]: {text[:700]}"
        )
    return "\n".join(lines)
