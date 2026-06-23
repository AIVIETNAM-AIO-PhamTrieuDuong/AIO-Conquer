from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "warning", "error"]
ToolErrorType = Literal[
    "invalid_column",
    "incompatible_type",
    "insufficient_rows",
    "missing_values",
    "timeout",
    "invalid_code",
    "invalid_result_shape",
    "unsupported_method",
    "invalid_input",
    "execution_error",
]


class ToolRequest(BaseModel):
    """Describe one tool call requested by an agent or graph node."""

    tool_name: str = Field(..., min_length=1)
    request_id: str = Field(..., min_length=1)
    caller: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_output_schema: str = "ToolResult"


class ToolProvenance(BaseModel):
    """Capture the data source and context used to produce a tool result."""

    dataset_id: str | None = None
    source: str | None = None
    columns: list[str] = Field(default_factory=list)


class ToolError(BaseModel):
    """Represent a normalized tool failure for downstream handling."""

    type: ToolErrorType
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Return one normalized result envelope from any tool implementation."""

    tool_name: str
    request_id: str
    status: ToolStatus
    data: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    summary: str
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    provenance: ToolProvenance = Field(default_factory=ToolProvenance)

    @classmethod
    def ok(
        cls,
        request: ToolRequest,
        data: dict[str, Any] | list[Any] | str | int | float | bool | None,
        summary: str,
        provenance: ToolProvenance | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolResult":
        """Build a successful tool result with optional warnings."""
        result_warnings = warnings or []
        return cls(
            tool_name=request.tool_name,
            request_id=request.request_id,
            status="warning" if result_warnings else "ok",
            data=data,
            summary=summary,
            warnings=result_warnings,
            provenance=provenance or ToolProvenance(),
        )

    @classmethod
    def fail(
        cls,
        request: ToolRequest,
        error_type: ToolErrorType,
        message: str,
        details: dict[str, Any] | None = None,
        provenance: ToolProvenance | None = None,
    ) -> "ToolResult":
        """Build a failed tool result with a normalized error payload."""
        return cls(
            tool_name=request.tool_name,
            request_id=request.request_id,
            status="error",
            data=None,
            summary=message,
            error=ToolError(
                type=error_type,
                message=message,
                details=details or {},
            ),
            provenance=provenance or ToolProvenance(),
        )


class DataLoaderTool(ABC):
    """Define the contract for loading tabular data from any source."""

    @abstractmethod
    def load(self) -> ToolResult:
        """Load or validate the source before feature retrieval."""

    @abstractmethod
    def fetch_all(self) -> ToolResult:
        """Return all available records from the loaded source."""

    @abstractmethod
    def fetch_features(self, features: list[str]) -> ToolResult:
        """Return records containing only the requested feature columns."""
