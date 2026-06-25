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
    history: NotRequired[list[ConversationTurn]]
    context: NotRequired[str]
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
