import asyncio

from app.core.config import settings
from app.memory.eda_store import eda_store
from app.memory.vector_store import vector_store
from app.model.openai_client import llm
from app.model.prompts.qa_system import build_prompt
from app.tools.dataset_profile import DatasetProfileTool
from app.tools.schema import ToolRequest, ToolResult
from app.tools.statistics import StatisticalAnalysisTool
from app.validation.parser import parse_response
from app.graph.schema import GraphState

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

dataset_profile_tool = DatasetProfileTool()
statistical_tool = StatisticalAnalysisTool()


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
    requirements = _domain_requirements(results)
    return {
        "domain_context": results,
        "domain_requirements": requirements,
        "context": _append_context(
            state.get("context", ""),
            _domain_context_summary(results, requirements),
        ),
    }


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
        {"columns": columns, "method": _association_method(state["question"])},
    )


async def node_custom_metric(state: GraphState) -> dict:
    """Run an approved custom metric when the question implies one."""
    inputs = _custom_metric_inputs(state)
    if not inputs:
        return {}
    return await _run_statistical_tool(
        state,
        StatisticalAnalysisTool.CUSTOM_METRIC,
        "Compute an approved deterministic custom metric.",
        inputs,
    )


async def node_generate(state: GraphState) -> dict:
    """Build the QA prompt and request a raw model response."""
    context = state.get("context", "")
    history = state.get("history", [])
    prompt = build_prompt(state["question"], context=context, history=history)
    max_tokens = 4096 if context else 1024
    raw_response = await llm.generate(prompt, max_tokens=max_tokens)
    return {"prompt": prompt, "raw_response": raw_response}


async def node_parse(state: GraphState) -> dict:
    """Parse the raw LLM JSON text into the API response schema."""
    return {"response": _dump_model(parse_response(state["raw_response"]))}


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
    request_payload = _dump_model(request)
    result_payload = _dump_model(result)
    warnings = [*state.get("warnings", []), *result.warnings]
    if result.status == "error" and result.error:
        warnings.append(result.error.message)

    return {
        "tool_requests": [*state.get("tool_requests", []), request_payload],
        "tool_results": [*state.get("tool_results", []), result_payload],
        "statistical_findings": [
            *state.get("statistical_findings", []),
            _statistical_finding(result),
        ],
        "warnings": warnings,
        "context": _append_context(
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
    request_payload = _dump_model(request)
    result_payload = _dump_model(result)
    warnings = [*state.get("warnings", []), *result.warnings]
    if result.status == "error" and result.error:
        warnings.append(result.error.message)

    return {
        "tool_requests": [*state.get("tool_requests", []), request_payload],
        "tool_results": [*state.get("tool_results", []), result_payload],
        "statistical_findings": [
            *state.get("statistical_findings", []),
            _statistical_finding(result),
        ],
        "warnings": warnings,
        "context": _append_context(
            state.get("context", ""),
            _tool_context_summary(result),
        ),
    }


def _domain_requirements(results: list[dict]) -> dict:
    """Merge domain vector metadata into tool-planning hints."""
    requirements = {
        "metrics": [],
        "features": [],
        "constraints": [],
        "sources": [],
    }
    for result in results:
        metadata = result.get("metadata", {})
        _extend_unique(requirements["metrics"], _as_list(metadata.get("metrics")))
        _extend_unique(requirements["features"], _as_list(metadata.get("features")))
        _extend_unique(
            requirements["constraints"],
            _as_list(metadata.get("constraints")),
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


def _domain_context_summary(results: list[dict], requirements: dict) -> str:
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


def _extend_unique(target: list[str], values: list[str]) -> None:
    """Append unique non-empty values while preserving existing order."""
    existing = {item.lower() for item in target}
    for value in values:
        normalized = value.strip()
        if normalized and normalized.lower() not in existing:
            target.append(normalized)
            existing.add(normalized.lower())


def _as_list(value: object) -> list[str]:
    """Normalize metadata values to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


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
    hints = _as_list(state.get("domain_requirements", {}).get("features"))
    lookup = {column.lower(): column for column in available_columns}
    matched: list[str] = []
    for hint in hints:
        column = lookup.get(hint.lower())
        if column and column not in matched:
            matched.append(column)
    return matched


def _domain_metric_hints(state: GraphState) -> list[str]:
    """Return lowercase domain metric hints for metric selection."""
    hints = _as_list(state.get("domain_requirements", {}).get("metrics"))
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


def _association_method(question: str) -> str:
    """Return the requested association method when explicitly named."""
    lowered = question.lower()
    if "spearman" in lowered:
        return "spearman"
    if "pearson" in lowered:
        return "pearson"
    return "auto"


def _custom_metric_inputs(state: GraphState) -> dict | None:
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


def _statistical_finding(result: ToolResult) -> dict:
    """Convert one tool result into a compact graph finding."""
    return {
        "tool_name": result.tool_name,
        "status": result.status,
        "summary": result.summary,
        "warnings": result.warnings,
        "data": result.data,
        "provenance": _dump_model(result.provenance),
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


def _append_context(context: str, addition: str) -> str:
    """Append a structured tool summary to the existing prompt context."""
    if not addition:
        return context
    if not context:
        return addition
    return f"{context}\n\n{addition}"


def _dump_model(model: object) -> dict:
    """Return a dict from a Pydantic model across supported versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
