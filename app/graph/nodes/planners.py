"""Deterministic planner nodes for the QA graph."""

from app.graph.nodes.common import as_list, contains_any, unique_values
from app.graph.nodes.tools import custom_metric_inputs
from app.graph.schema import GraphState
from app.tools.dataset_profile import DatasetProfileTool
from app.tools.statistics import StatisticalAnalysisTool


async def node_orchestrator_router(state: GraphState) -> dict:
    """Classify the question and choose deterministic planner paths."""
    intent = _build_analysis_intent(state)
    open_questions = list(state.get("open_questions", []))
    warnings = []
    if not state.get("eda_result"):
        question = "Run EDA or select an active dataset before analysis."
        open_questions = unique_values([*open_questions, question])
        warnings.append("No active structured EDA result was available.")

    selected_agents = _selected_agent_names(intent)
    handoff = _agent_handoff_record(
        state,
        agent="orchestrator_router",
        reads=[
            "question",
            "history",
            "eda_result",
            "domain_requirements",
            "curated_context",
        ],
        writes=[
            "analysis_intent",
            "selected_agents",
            "agent_handoffs",
            "open_questions",
        ],
        summary=(
            "Detected "
            f"{intent['analysis_type']} intent with {intent['risk_level']} risk."
        ),
        warnings=warnings,
        downstream=["domain_context_planner", "query_builder"],
    )
    return {
        "analysis_intent": intent,
        "selected_agents": selected_agents,
        "agent_handoffs": [*state.get("agent_handoffs", []), handoff],
        "open_questions": open_questions,
    }


async def node_domain_context_planner(state: GraphState) -> dict:
    """Convert retrieved domain memory into bounded planning context."""
    requirements = state.get("domain_requirements", {})
    domain_context = state.get("domain_context", [])
    working = dict(state.get("agent_working_memory", {}))
    assumptions = list(working.get("assumptions", []))
    warnings = []

    if domain_context or any(requirements.get(key) for key in requirements):
        assumptions.append(
            "Domain memory hints are supplemental and require tool evidence."
        )
        summary = "Loaded domain memory as skill-style planning context."
    else:
        assumptions.append(
            "No domain memory hints were retrieved; using generic dataset analysis."
        )
        summary = "No domain-specific context was available."
        warnings.append("Domain context planner used generic dataset assumptions.")

    working["requirements"] = requirements
    working["assumptions"] = unique_values(assumptions)[-20:]
    working["domain_context_sources"] = [
        {
            "memory_type": item.get("memory_type", ""),
            "source_id": item.get("source_id", ""),
            "score": item.get("score"),
        }
        for item in domain_context[:5]
    ]

    handoff = _agent_handoff_record(
        state,
        agent="domain_context_planner",
        reads=["domain_context", "domain_requirements", "analysis_intent"],
        writes=["agent_handoffs", "open_questions", "agent_working_memory"],
        summary=summary,
        warnings=warnings,
        downstream=["query_builder", "coding_tool_planner"],
    )
    return {
        "agent_handoffs": [*state.get("agent_handoffs", []), handoff],
        "open_questions": state.get("open_questions", []),
        "agent_working_memory": working,
    }


async def node_query_builder(state: GraphState) -> dict:
    """Build a structured query plan from EDA metadata and domain hints."""
    columns = _structured_eda_columns(state)
    candidate_columns = _rank_candidate_columns(state, columns)
    intent = state.get("analysis_intent", {})
    query_plan = {
        "analysis_type": intent.get("analysis_type", "general"),
        "operation": intent.get("operation", "general_analysis"),
        "dataset_id": state.get("dataset_id", ""),
        "uses_structured_eda": True,
        "candidate_column_names": [
            item["column"] for item in candidate_columns
        ],
        "source_fields": ["eda_result.num_stats", "eda_result.cat_stats"],
    }
    retrieved_context = _query_retrieved_context(state, candidate_columns)
    handoff = _agent_handoff_record(
        state,
        agent="query_builder",
        reads=["question", "eda_result", "domain_requirements", "analysis_intent"],
        writes=[
            "query_plan",
            "candidate_columns",
            "retrieved_context",
            "agent_handoffs",
        ],
        summary=(
            "Mapped the question to "
            f"{len(candidate_columns)} structured EDA candidate columns."
        ),
        warnings=[],
        downstream=["coding_tool_planner"],
    )
    return {
        "query_plan": query_plan,
        "candidate_columns": candidate_columns,
        "retrieved_context": retrieved_context,
        "agent_handoffs": [*state.get("agent_handoffs", []), handoff],
    }


async def node_coding_tool_planner(state: GraphState) -> dict:
    """Plan deterministic tool usage without executing generated code."""
    coding_plan = _deterministic_coding_plan(state)
    working = dict(state.get("agent_working_memory", {}))
    selected_columns = [
        item.get("column", "") for item in state.get("candidate_columns", [])
    ]
    working["selected_columns"] = unique_values(
        [*working.get("selected_columns", []), *selected_columns]
    )
    working["selected_metrics"] = unique_values(
        [
            *working.get("selected_metrics", []),
            *[
                item.get("tool_name", "")
                for item in coding_plan.get("planned_tools", [])
            ],
        ]
    )
    working["coding_plan"] = coding_plan

    handoff = _agent_handoff_record(
        state,
        agent="coding_tool_planner",
        reads=[
            "query_plan",
            "candidate_columns",
            "analysis_intent",
            "domain_requirements",
        ],
        writes=["coding_plan", "agent_handoffs", "agent_working_memory"],
        summary=(
            "Planned "
            f"{len(coding_plan.get('planned_tools', []))} deterministic tools."
        ),
        warnings=coding_plan.get("warnings", []),
        downstream=[
            "column_metadata",
            "missingness_summary",
            "type_compatibility",
            "basic_statistical_summary",
            "statistical_association",
            "custom_metric",
        ],
    )
    return {
        "coding_plan": coding_plan,
        "agent_handoffs": [*state.get("agent_handoffs", []), handoff],
        "agent_working_memory": working,
    }


def _build_analysis_intent(state: GraphState) -> dict:
    """Infer deterministic analysis intent from the user question."""
    question = state["question"]
    lowered = question.lower()
    analysis_type = "general"
    operation = "general_analysis"
    if contains_any(lowered, ("correlation", "relationship")):
        analysis_type = "association"
        operation = "correlation_or_association"
    elif contains_any(lowered, ("compare", "group", "segment", "cohort")):
        analysis_type = "comparison"
        operation = "group_or_segment_comparison"
    elif contains_any(
        lowered,
        ("summary", "summarize", "count", "average", "mean", "total", "sum"),
    ):
        analysis_type = "summary"
        operation = "statistical_summary"

    risk_level = "low"
    if contains_any(lowered, ("predict", "causal", "cause", "forecast")):
        risk_level = "medium"

    return {
        "analysis_type": analysis_type,
        "operation": operation,
        "risk_level": risk_level,
        "requires_dataset": True,
        "classification_method": "deterministic_keyword_router",
        "domain_hints_present": bool(
            state.get("domain_context")
            or any(state.get("domain_requirements", {}).values())
        ),
    }


def _selected_agent_names(intent: dict) -> list[str]:
    """Return planner names needed for the current analysis intent."""
    selected = [
        "orchestrator_router",
        "domain_context_planner",
        "query_builder",
        "coding_tool_planner",
    ]
    if intent.get("analysis_type") in {"association", "comparison"}:
        selected.append("statistical_tool_nodes")
    return unique_values(selected)


def _agent_handoff_record(
    state: GraphState,
    agent: str,
    reads: list[str],
    writes: list[str],
    summary: str,
    warnings: list[str],
    downstream: list[str],
) -> dict:
    """Build a JSON-safe handoff record for one planner node."""
    return {
        "agent": agent,
        "reads": reads,
        "writes": writes,
        "summary": summary,
        "warnings": warnings,
        "provenance": {
            "source_node": f"node_{agent}",
            "run_id": state.get("run_id", ""),
            "dataset_id": state.get("dataset_id", ""),
        },
        "downstream": downstream,
    }


def _structured_eda_columns(state: GraphState) -> list[dict]:
    """Extract candidate columns from structured EDA stats only."""
    eda_result = state.get("eda_result", {})
    columns = []
    for name in (eda_result.get("num_stats") or {}):
        columns.append(
            {
                "column": str(name),
                "kind": "numeric",
                "source": "eda_result.num_stats",
            }
        )
    for name in (eda_result.get("cat_stats") or {}):
        columns.append(
            {
                "column": str(name),
                "kind": "categorical",
                "source": "eda_result.cat_stats",
            }
        )
    return columns


def _rank_candidate_columns(
    state: GraphState,
    columns: list[dict],
) -> list[dict]:
    """Rank EDA columns by question match, domain hint, then availability."""
    question = state["question"].lower()
    hints = unique_values(
        [
            *as_list(state.get("domain_requirements", {}).get("features")),
            *as_list(state.get("domain_requirements", {}).get("metrics")),
        ]
    )
    ranked = []
    for column in columns:
        name = column["column"]
        score = 1
        reasons = ["available in structured EDA"]
        confidence = "low"
        if _question_matches_column(question, name):
            score += 20
            reasons.insert(0, "question exact match")
            confidence = "high"
        if _column_matches_hints(name, hints):
            score += 10
            reasons.append("domain hint")
            if confidence == "low":
                confidence = "medium"
        ranked.append(
            {
                "column": name,
                "role": _candidate_column_role(column["kind"]),
                "source": column["source"],
                "match_reason": " + ".join(reasons),
                "confidence": confidence,
                "_rank": score,
            }
        )

    ranked.sort(key=lambda item: (-item["_rank"], item["column"].lower()))
    return [
        {key: value for key, value in item.items() if key != "_rank"}
        for item in ranked[:10]
    ]


def _candidate_column_role(kind: str) -> str:
    """Return the planning role for one structured EDA column type."""
    if kind == "numeric":
        return "candidate_metric"
    return "candidate_dimension"


def _question_matches_column(question: str, column: str) -> bool:
    """Return whether a question directly names a column."""
    normalized = column.lower().replace("_", " ").replace("-", " ")
    compact = normalized.replace(" ", "")
    question_compact = question.replace("_", " ").replace("-", " ")
    question_compact = question_compact.replace(" ", "")
    return normalized in question or compact in question_compact


def _column_matches_hints(column: str, hints: list[str]) -> bool:
    """Return whether a column matches retrieved domain hints."""
    normalized = column.lower()
    return any(normalized == hint.lower() for hint in hints)


def _query_retrieved_context(
    state: GraphState,
    candidate_columns: list[dict],
) -> list[dict]:
    """Build compact structured retrieval evidence for the query plan."""
    domain_sources = state.get("domain_requirements", {}).get("sources", [])
    return [
        {
            "source": "structured_eda",
            "dataset_id": state.get("dataset_id", ""),
            "candidate_columns": [
                item.get("column", "") for item in candidate_columns
            ],
        },
        {
            "source": "domain_memory",
            "sources": domain_sources[:5],
        },
    ]


def _deterministic_coding_plan(state: GraphState) -> dict:
    """Plan the existing deterministic tools for the current analysis."""
    intent = state.get("analysis_intent", {})
    operation = intent.get("operation", "general_analysis")
    candidate_columns = [
        item.get("column", "") for item in state.get("candidate_columns", [])
    ]
    planned_tools = [
        _planned_tool(
            DatasetProfileTool.COLUMN_METADATA,
            "Inspect structured column metadata for candidate features.",
            "candidate_columns",
            "Column metadata is always useful for grounded analysis.",
        ),
        _planned_tool(
            DatasetProfileTool.MISSINGNESS_SUMMARY,
            "Check missingness before interpreting statistics.",
            "candidate_columns",
            "Missingness affects all downstream tabular analysis.",
        ),
        _planned_tool(
            DatasetProfileTool.TYPE_COMPATIBILITY,
            "Validate candidate column types for the requested operation.",
            "analysis_intent.operation",
            f"Operation is {operation}.",
        ),
    ]
    if operation in {
        "statistical_summary",
        "general_analysis",
        "group_or_segment_comparison",
        "correlation_or_association",
    }:
        planned_tools.append(
            _planned_tool(
                StatisticalAnalysisTool.BASIC_SUMMARY,
                "Compute basic summaries for candidate columns.",
                "candidate_columns",
                "Summary statistics support grounded response generation.",
            )
        )
    if operation in {"group_or_segment_comparison", "correlation_or_association"}:
        planned_tools.append(
            _planned_tool(
                StatisticalAnalysisTool.CORRELATION,
                "Measure compatible relationships or associations.",
                "candidate_columns",
                "The routed intent asks for comparison or relationship evidence.",
            )
        )
    if custom_metric_inputs(state):
        planned_tools.append(
            _planned_tool(
                StatisticalAnalysisTool.CUSTOM_METRIC,
                "Compute an approved deterministic custom metric.",
                "question/domain_requirements",
                "The question implies an approved simple metric.",
            )
        )

    warnings = []
    if not candidate_columns:
        warnings.append("No structured EDA candidate columns were selected.")

    return {
        "strategy": "deterministic_tool_planning",
        "operation": operation,
        "candidate_columns": candidate_columns,
        "planned_tools": planned_tools,
        "warnings": warnings,
        "provenance": {
            "source_node": "node_coding_tool_planner",
            "run_id": state.get("run_id", ""),
        },
    }


def _planned_tool(
    tool_name: str,
    purpose: str,
    inputs_source: str,
    reason: str,
) -> dict:
    """Build one JSON-safe planned-tool descriptor."""
    return {
        "tool_name": tool_name,
        "purpose": purpose,
        "inputs_source": inputs_source,
        "reason": reason,
    }
