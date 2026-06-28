"""LangSmith tracing bootstrap.

The application reads configuration via pydantic ``Settings`` (from ``.env``),
but the LangChain/langsmith SDK discovers its configuration from ``os.environ``.
``pydantic-settings`` does *not* populate ``os.environ``, so without this bridge
the ``LANGSMITH_*`` values in ``.env`` would never reach the tracer and nothing
would be logged.

``init_tracing`` copies the relevant settings into ``os.environ`` (without
clobbering values an operator may have already exported) so that:

* ``ChatOpenAI`` calls in the QA graph nodes are auto-traced, and
* ``@traceable``-decorated functions (e.g. the raw httpx ``NineRouterClient``)
  are captured as well.

It is safe to call multiple times and is a no-op when tracing is disabled.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings


def _set_env_default(key: str, value: str | None) -> None:
    """Set ``os.environ[key]`` from ``value`` unless already set or empty."""
    if value and not os.environ.get(key):
        os.environ[key] = value


def init_tracing(settings: "Settings") -> None:
    """Export LangSmith settings into ``os.environ`` for the langsmith SDK."""
    if not settings.langsmith_tracing:
        # Leave the environment untouched; tracing stays disabled.
        return

    # LangChain checks both LANGSMITH_TRACING and the legacy LANGCHAIN_TRACING_V2.
    _set_env_default("LANGSMITH_TRACING", "true")
    _set_env_default("LANGCHAIN_TRACING_V2", "true")
    _set_env_default("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    _set_env_default("LANGSMITH_API_KEY", settings.langsmith_api_key)
    _set_env_default("LANGSMITH_PROJECT", settings.langsmith_project)
    _set_env_default("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id)
