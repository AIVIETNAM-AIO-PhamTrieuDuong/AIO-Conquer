"""Deterministic tabular and statistical tool nodes for the QA graph."""

import asyncio
import json

from app.graph.nodes.common import (
    append_context,
    as_list,
    dump_model,
    unique_values,
)
from app.graph.schema import GraphState
from app.tools.dataset_profile import DatasetProfileTool
from app.tools.schema import ToolRequest, ToolResult
from app.tools.statistics import StatisticalAnalysisTool

dataset_profile_tool = DatasetProfileTool()
statistical_tool = StatisticalAnalysisTool()


async def node_column_metadata(state: GraphState) -> dict:
    """Run the column metadata tool for the active EDA dataset."""
    return await _run_tabular_tool(
        state,
        DatasetProfileTool.COLUMN_METADATA,
        "Generate column metadata for the active dataset.",
    )


async def node_missingness_summary(state: GraphState) -> dict:
    """Run the missingness summary tool for the active EDA dataset."""
    return await _run_tabular_tool(
        state,
        DatasetProfileTool.MISSINGNESS_SUMMARY,
        "Summarize missing values for the active dataset.",
    )


async def node_type_compatibility(state: GraphState) -> dict:
    """Run a type compatibility check for the active analysis question."""
    operation = _type_compatibility_operation(state["question"])
    inputs = {"operation": operation}
    columns = _type_compatibility_columns(state, operation)
    if columns:
        inputs["columns"] = columns
    return await _run_tabular_tool(
        state,
        DatasetProfileTool.TYPE_COMPATIBILITY,
        "Check dataset column types against the requested analysis.",
        inputs,
    )


async def node_basic_statistical_summary(state: GraphState) -> dict:
    """Run the basic statistical summary tool for the active dataset."""
    columns = _analysis_columns(state)
    inputs = {"columns": columns} if columns else {}
    return await _run_statistical_tool(
        state,
        StatisticalAnalysisTool.BASIC_SUMMARY,
        "Generate basic statistical summaries for the active dataset.",
        inputs,
    )


async def node_statistical_association(state: GraphState) -> dict:
    """Run correlation or association analysis for compatible columns."""
    columns = _association_columns(state)
    if len(columns) < 2:
        return {}
    return await _run_statistical_tool(
        state,
        StatisticalAnalysisTool.CORRELATION,
        "Measure correlation or association for active dataset columns.",
        {"columns": columns, "method": _association_method(state)},
    )


async def node_custom_metric(state: GraphState) -> dict:
    """Run an approved custom metric when the question implies one."""
    inputs = custom_metric_inputs(state)
    if not inputs:
        return {}
    return await _run_statistical_tool(
        state,
        StatisticalAnalysisTool.CUSTOM_METRIC,
        "Compute an approved deterministic custom metric.",
        inputs,
    )


async def _run_tabular_tool(
    state: GraphState,
    tool_name: str,
    purpose: str,
    extra_inputs: dict | None = None,
) -> dict:
    """Call `DatasetProfileTool` and append its state updates."""
    file_path = state.get("dataset_file_path", "")
    if not file_path:
        return {}

    inputs = {
        "file_path": file_path,
        "dataset_id": state.get("dataset_id"),
    }
    inputs.update(extra_inputs or {})
    request = ToolRequest(
        tool_name=tool_name,
        request_id=f"{tool_name}:{state.get('dataset_id', 'active')}",
        caller="langgraph_qa_pipeline",
        purpose=purpose,
        inputs=inputs,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, dataset_profile_tool.invoke, request)
    request_payload = dump_model(request)
    result_payload = dump_model(result)
    warnings = [*state.get("warnings", []), *result.warnings]
    if result.status == "error" and result.error:
        warnings.append(result.error.message)
    tool_memory = [
        *state.get("tool_memory", []),
        _tool_memory_record(request_payload, result_payload),
    ]
    error_memory = [
        *state.get("error_memory", []),
        *_tool_error_memory_records(request_payload, result_payload),
    ]
    working_memory = _update_working_memory_with_tool(
        state,
        request_payload,
        result_payload,
    )

    return {
        "tool_requests": [*state.get("tool_requests", []), request_payload],
        "tool_results": [*state.get("tool_results", []), result_payload],
        "tool_memory": tool_memory,
        "error_memory": error_memory,
        "agent_working_memory": working_memory,
        "statistical_findings": [
            *state.get("statistical_findings", []),
            _statistical_finding(result),
        ],
        "warnings": warnings,
        "context": append_context(
            state.get("context", ""),
            _tool_context_summary(result),
        ),
    }


async def _run_statistical_tool(
    state: GraphState,
    tool_name: str,
    purpose: str,
    extra_inputs: dict | None = None,
) -> dict:
    """Call `StatisticalAnalysisTool` and append its state updates."""
    file_path = state.get("dataset_file_path", "")
    if not file_path:
        return {}

    inputs = {
        "file_path": file_path,
        "dataset_id": state.get("dataset_id"),
    }
    inputs.update(extra_inputs or {})
    request = ToolRequest(
        tool_name=tool_name,
        request_id=f"{tool_name}:{state.get('dataset_id', 'active')}",
        caller="langgraph_qa_pipeline",
        purpose=purpose,
        inputs=inputs,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, statistical_tool.invoke, request)
    request_payload = dump_model(request)
    result_payload = dump_model(result)
    warnings = [*state.get("warnings", []), *result.warnings]
    if result.status == "error" and result.error:
        warnings.append(result.error.message)
    tool_memory = [
        *state.get("tool_memory", []),
        _tool_memory_record(request_payload, result_payload),
    ]
    error_memory = [
        *state.get("error_memory", []),
        *_tool_error_memory_records(request_payload, result_payload),
    ]
    working_memory = _update_working_memory_with_tool(
        state,
        request_payload,
        result_payload,
    )

    return {
        "tool_requests": [*state.get("tool_requests", []), request_payload],
        "tool_results": [*state.get("tool_results", []), result_payload],
        "tool_memory": tool_memory,
        "error_memory": error_memory,
        "agent_working_memory": working_memory,
        "statistical_findings": [
            *state.get("statistical_findings", []),
            _statistical_finding(result),
        ],
        "warnings": warnings,
        "context": append_context(
            state.get("context", ""),
            _tool_context_summary(result),
        ),
    }


def custom_metric_inputs(state: GraphState) -> dict | None:
    """Infer approved custom metric inputs from question and EDA columns."""
    metric_hints = " ".join(_domain_metric_hints(state))
    question = f"{state['question']} {metric_hints}".lower()
    all_numeric, all_categorical = _eda_column_groups(state)
    numeric_columns = _domain_feature_columns(state, all_numeric) or all_numeric
    categorical_columns = (
        _domain_feature_columns(state, all_categorical) or all_categorical
    )
    if not numeric_columns and "count" not in question:
        return None
    if any(term in question for term in ("ratio", "rate", "percentage")):
        if len(numeric_columns) >= 2:
            return {
                "metric": "ratio_of_sums",
                "numerator_column": numeric_columns[0],
                "denominator_column": numeric_columns[1],
            }
    if "difference" in question and categorical_columns and numeric_columns:
        return {
            "metric": "difference_of_means",
            "group_column": categorical_columns[0],
            "value_column": numeric_columns[0],
        }
    if any(term in question for term in ("sum", "total")) and numeric_columns:
        return {"metric": "sum", "column": numeric_columns[0]}
    if any(term in question for term in ("mean", "average")) and numeric_columns:
        return {"metric": "mean", "column": numeric_columns[0]}
    if "count" in question or "how many" in question:
        return {"metric": "count"}
    return None


def _tool_memory_record(request: dict, result: dict) -> dict:
    """Build a compact state record for one tool call."""
    return {
        "source_node": "tool_node",
        "tool_name": request.get("tool_name", ""),
        "request_id": request.get("request_id", ""),
        "caller": request.get("caller", ""),
        "purpose": request.get("purpose", ""),
        "inputs": request.get("inputs", {}),
        "status": result.get("status", ""),
        "summary": result.get("summary", ""),
        "warnings": result.get("warnings", []),
        "error": result.get("error"),
        "provenance": result.get("provenance", {}),
    }


def _tool_error_memory_records(request: dict, result: dict) -> list[dict]:
    """Build recoverable warning/error memory records for one tool call."""
    records = []
    retry_context = {
        "tool_name": request.get("tool_name", ""),
        "request_id": request.get("request_id", ""),
        "inputs": request.get("inputs", {}),
        "purpose": request.get("purpose", ""),
    }
    for warning in result.get("warnings", []):
        records.append(
            {
                "source_node": "tool_node",
                "event_type": "tool_warning",
                "tool_name": request.get("tool_name", ""),
                "request_id": request.get("request_id", ""),
                "message": warning,
                "retry_context": retry_context,
                "provenance": result.get("provenance", {}),
            }
        )
    if result.get("status") == "error":
        error = result.get("error") or {}
        records.append(
            {
                "source_node": "tool_node",
                "event_type": "tool_error",
                "tool_name": request.get("tool_name", ""),
                "request_id": request.get("request_id", ""),
                "message": error.get("message", result.get("summary", "")),
                "error": error,
                "retry_context": retry_context,
                "provenance": result.get("provenance", {}),
            }
        )
    return records


def _update_working_memory_with_tool(
    state: GraphState,
    request: dict,
    result: dict,
) -> dict:
    """Update working memory with one tool result artifact."""
    working = dict(state.get("agent_working_memory", {}))
    inputs = request.get("inputs", {})
    working["selected_columns"] = unique_values(
        [
            *working.get("selected_columns", []),
            *_input_columns(inputs),
        ]
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    metric = inputs.get("metric") or data.get("metric")
    working["selected_metrics"] = unique_values(
        [
            *working.get("selected_metrics", []),
            *([str(metric)] if metric else []),
            request.get("tool_name", ""),
        ]
    )
    working["intermediate_findings"] = [
        *working.get("intermediate_findings", []),
        {
            "tool_name": request.get("tool_name", ""),
            "status": result.get("status", ""),
            "summary": result.get("summary", ""),
            "warnings": result.get("warnings", []),
        },
    ][-20:]

    unresolved = list(working.get("unresolved_questions", []))
    for warning in result.get("warnings", []):
        unresolved.append(
            f"Review warning from {request.get('tool_name', 'tool')}: {warning}"
        )
    if result.get("status") == "error":
        unresolved.append(
            "Resolve tool error from "
            f"{request.get('tool_name', 'tool')}: {result.get('summary', '')}"
        )
    working["unresolved_questions"] = unique_values(unresolved)[-20:]
    return working


def _input_columns(inputs: dict) -> list[str]:
    """Return all column-like inputs from a tool request payload."""
    columns = []
    for key in (
        "columns",
        "column",
        "value_column",
        "group_column",
        "numerator_column",
        "denominator_column",
    ):
        columns.extend(as_list(inputs.get(key)))
    return columns


def _eda_column_groups(state: GraphState) -> tuple[list[str], list[str]]:
    """Return numeric and categorical EDA columns from graph state."""
    eda_result = state.get("eda_result", {})
    numeric_columns = list((eda_result.get("num_stats") or {}).keys())
    categorical_columns = list((eda_result.get("cat_stats") or {}).keys())
    return numeric_columns, categorical_columns


def _domain_feature_columns(
    state: GraphState,
    available_columns: list[str],
) -> list[str]:
    """Return domain feature hints that match available EDA columns."""
    hints = as_list(state.get("domain_requirements", {}).get("features"))
    lookup = {column.lower(): column for column in available_columns}
    matched: list[str] = []
    for hint in hints:
        column = lookup.get(hint.lower())
        if column and column not in matched:
            matched.append(column)
    return matched


def _domain_metric_hints(state: GraphState) -> list[str]:
    """Return lowercase domain metric hints for metric selection."""
    hints = as_list(state.get("domain_requirements", {}).get("metrics"))
    return [hint.lower() for hint in hints]


def _type_compatibility_operation(question: str) -> str:
    """Infer the narrow compatibility check needed for a question."""
    lowered = question.lower()
    if any(term in lowered for term in ("correlation", "relationship")):
        return "correlation"
    if any(term in lowered for term in ("group", "segment", "cohort")):
        return "groupby_aggregate"
    if any(term in lowered for term in ("trend", "time", "date")):
        return "time_series"
    if "compare" in lowered:
        return "compare_categories"
    return "numeric_summary"


def _type_compatibility_columns(
    state: GraphState,
    operation: str,
) -> list[str]:
    """Select available EDA columns for the compatibility operation."""
    numeric_columns, categorical_columns = _eda_column_groups(state)
    domain_numeric = _domain_feature_columns(state, numeric_columns)
    domain_categorical = _domain_feature_columns(state, categorical_columns)
    if operation == "correlation":
        return domain_numeric or numeric_columns
    if operation in {"groupby_aggregate", "compare_categories"}:
        categories = domain_categorical or categorical_columns
        numbers = domain_numeric or numeric_columns
        return categories[:1] + numbers[:1]
    if operation == "numeric_summary":
        return domain_numeric or numeric_columns
    return []


def _analysis_columns(state: GraphState) -> list[str]:
    """Select EDA-backed columns for statistical summary."""
    numeric_columns, categorical_columns = _eda_column_groups(state)
    available = [*numeric_columns, *categorical_columns]
    return _domain_feature_columns(state, available) or available


def _association_columns(state: GraphState) -> list[str]:
    """Select compatible EDA columns for association analysis."""
    question = state["question"].lower()
    numeric_columns, categorical_columns = _eda_column_groups(state)
    domain_numeric = _domain_feature_columns(state, numeric_columns)
    domain_categorical = _domain_feature_columns(state, categorical_columns)
    domain_columns = _domain_feature_columns(
        state,
        [*numeric_columns, *categorical_columns],
    )
    if "correlation" in question and len(numeric_columns) >= 2:
        columns = domain_numeric if len(domain_numeric) >= 2 else numeric_columns
        return columns[:4]
    if "relationship" in question and len(numeric_columns) >= 2:
        columns = domain_numeric if len(domain_numeric) >= 2 else numeric_columns
        return columns[:4]
    if any(term in question for term in ("group", "segment", "compare")):
        categories = domain_categorical or categorical_columns
        numbers = domain_numeric or numeric_columns
        if categories and numbers:
            return categories[:1] + numbers[:1]
    if len(domain_columns) >= 2:
        return domain_columns[:4]
    if len(numeric_columns) >= 2:
        return numeric_columns[:4]
    if categorical_columns and numeric_columns:
        return categorical_columns[:1] + numeric_columns[:1]
    if len(categorical_columns) >= 2:
        return categorical_columns[:2]
    return []


def _association_method(state: GraphState) -> str:
    """Return the association method, preferring a selected use-case test."""
    forced = str(
        state.get("domain_requirements", {}).get("association_method", "")
    ).strip()
    if forced:
        return forced
    lowered = state["question"].lower()
    if "spearman" in lowered:
        return "spearman"
    if "pearson" in lowered:
        return "pearson"
    return "auto"


def _statistical_finding(result: ToolResult) -> dict:
    """Convert one tool result into a compact graph finding."""
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "summary": result.summary,
        "warnings": result.warnings,
        "data": result.data,
        "provenance": dump_model(result.provenance),
    }


def _tool_context_summary(result: ToolResult) -> str:
    """Format a compact tool result summary for downstream prompting."""
    if result.status == "error":
        message = result.error.message if result.error else result.summary
        return f"Tool {result.tool_name} failed: {message}"
    if result.tool_name == DatasetProfileTool.COLUMN_METADATA:
        return _column_metadata_context(result)
    if result.tool_name == DatasetProfileTool.MISSINGNESS_SUMMARY:
        return _missingness_context(result)
    if result.tool_name == DatasetProfileTool.TYPE_COMPATIBILITY:
        return _type_compatibility_context(result)
    if result.tool_name == StatisticalAnalysisTool.BASIC_SUMMARY:
        return _basic_summary_context(result)
    if result.tool_name == StatisticalAnalysisTool.CORRELATION:
        return _association_context(result)
    if result.tool_name == StatisticalAnalysisTool.CUSTOM_METRIC:
        return _custom_metric_context(result)
    return f"Tool {result.tool_name}: {result.summary}"


def _column_metadata_context(result: ToolResult) -> str:
    """Summarize column metadata without dumping full tool payloads."""
    data = result.data if isinstance(result.data, dict) else {}
    columns = data.get("columns", [])
    details = []
    for column in columns[:30]:
        warnings = "; ".join(column.get("warnings", [])) or "none"
        details.append(
            f"- {column['name']}: type={column['inferred_type']}, "
            f"missing_ratio={column['missing_ratio']}, warnings={warnings}"
        )
    return "\n".join(["Tool tabular.column_metadata:", *details])


def _missingness_context(result: ToolResult) -> str:
    """Summarize missingness output for the LLM prompt context."""
    data = result.data if isinstance(result.data, dict) else {}
    lines = [
        "Tool tabular.missingness_summary:",
        "rows_with_any_missing_ratio="
        f"{data.get('rows_with_any_missing_ratio', 0)}",
    ]
    for column in data.get("columns", [])[:30]:
        if column.get("missing_count", 0) > 0 or column.get("warnings"):
            lines.append(
                f"- {column['name']}: missing_count="
                f"{column['missing_count']}, "
                f"missing_ratio={column['missing_ratio']}"
            )
    return "\n".join(lines)


def _type_compatibility_context(result: ToolResult) -> str:
    """Summarize type compatibility output for the LLM prompt context."""
    data = result.data if isinstance(result.data, dict) else {}
    reasons = "; ".join(data.get("blocking_reasons", [])) or "none"
    return (
        "Tool tabular.type_compatibility: "
        f"operation={data.get('operation')}, "
        f"compatible={data.get('compatible')}, "
        f"blocking_reasons={reasons}"
    )


def _basic_summary_context(result: ToolResult) -> str:
    """Summarize basic statistical summaries for prompt context."""
    data = result.data if isinstance(result.data, dict) else {}
    lines = ["Tool stats.basic_summary:"]
    for column in data.get("columns", [])[:30]:
        kind = column.get("inferred_type")
        if kind == "numeric":
            metric = column.get("numeric", {})
            lines.append(
                f"- {column['name']}: mean={metric.get('mean')}, "
                f"median={metric.get('median')}, "
                f"missing_ratio={column.get('missing_ratio')}"
            )
        else:
            lines.append(
                f"- {column['name']}: type={kind}, "
                f"unique_count={column.get('unique_count')}, "
                f"missing_ratio={column.get('missing_ratio')}"
            )
    return "\n".join(lines)


def _association_context(result: ToolResult) -> str:
    """Summarize association results for prompt context."""
    data = result.data if isinstance(result.data, dict) else {}
    lines = ["Tool stats.correlation:"]
    for pair in data.get("pairs", [])[:20]:
        columns = " vs ".join(pair.get("columns", []))
        lines.append(
            f"- {columns}: method={pair.get('method')}, "
            f"value={pair.get('value')}, rows_used={pair.get('rows_used')}"
        )
    return "\n".join(lines)


def _custom_metric_context(result: ToolResult) -> str:
    """Summarize a custom metric result for prompt context."""
    data = result.data if isinstance(result.data, dict) else {}
    return (
        "Tool stats.custom_metric: "
        f"metric={data.get('metric')}, "
        f"value={data.get('value')}, "
        f"columns={data.get('columns', [])}"
    )
