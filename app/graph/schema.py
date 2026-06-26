from typing import Any

from typing_extensions import NotRequired, Required, TypedDict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ConversationTurn(TypedDict):
    """Represent one stored conversation turn used as prompt history."""

    q: str
    a: str


class GraphState(TypedDict, total=False):
    """Carry shared LangGraph state across QA pipeline nodes."""

    question: Required[str]
    session_id: Required[str]
    run_id: NotRequired[str]
    history: NotRequired[list[ConversationTurn]]
    context: NotRequired[str]
    domain_context: NotRequired[list[dict[str, Any]]]
    domain_requirements: NotRequired[dict[str, Any]]
    multivariate_index: NotRequired[list[dict[str, Any]]]
    multivariate_selected: NotRequired[list[dict[str, Any]]]
    analysis_intent: NotRequired[dict[str, Any]]
    selected_agents: NotRequired[list[str]]
    agent_handoffs: NotRequired[list[dict[str, Any]]]
    query_plan: NotRequired[dict[str, Any]]
    candidate_columns: NotRequired[list[dict[str, Any]]]
    retrieved_context: NotRequired[list[dict[str, Any]]]
    coding_plan: NotRequired[dict[str, Any]]
    open_questions: NotRequired[list[str]]
    tool_memory: NotRequired[list[dict[str, Any]]]
    agent_working_memory: NotRequired[dict[str, Any]]
    curated_context: NotRequired[list[dict[str, Any]]]
    error_memory: NotRequired[list[dict[str, Any]]]
    eda_result: NotRequired[dict[str, Any]]
    dataset_id: NotRequired[str]
    dataset_file_path: NotRequired[str]
    tool_requests: NotRequired[list[dict[str, Any]]]
    tool_results: NotRequired[list[dict[str, Any]]]
    statistical_findings: NotRequired[list[dict[str, Any]]]
    warnings: NotRequired[list[str]]
    prompt: NotRequired[str]
    raw_response: NotRequired[str]
    response: NotRequired[dict[str, Any] | None]
