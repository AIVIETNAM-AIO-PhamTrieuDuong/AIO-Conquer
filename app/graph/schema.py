from typing_extensions import NotRequired, Required, TypedDict

from app.api.schemas import QAResponse
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
    prompt: NotRequired[str]
    raw_response: NotRequired[str]
    response: NotRequired[QAResponse | None]
