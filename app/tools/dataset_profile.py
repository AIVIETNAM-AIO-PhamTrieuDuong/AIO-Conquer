from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from app.tools.schema import ToolProvenance, ToolRequest, ToolResult


class DatasetProfileTool:
    """Profile a CSV dataset through the shared tool request/result contract."""

    tool_name = "tabular.dataset_profile"

    def __call__(self, request: ToolRequest | dict[str, Any]) -> ToolResult:
        """Run the profiler so graph nodes can call the tool directly."""
        return self.invoke(request)

    def invoke(self, request: ToolRequest | dict[str, Any]) -> ToolResult:
        """Return dataset shape and column-level profiling metadata.

        Args:
            request: A `ToolRequest` or equivalent dictionary containing
                `file_path` in `inputs`. Optional inputs are `dataset_id`,
                `columns`, and `sample_size`.

        Returns:
            A `ToolResult` containing JSON-compatible profile data, warnings,
            normalized errors, and provenance.
        """
        tool_request = self._request(request)
        inputs = tool_request.inputs
        file_path = inputs.get("file_path")
        dataset_id = inputs.get("dataset_id")
        columns = inputs.get("columns")
        sample_size = inputs.get("sample_size")

        if not isinstance(file_path, str) or not file_path:
            return ToolResult.fail(
                tool_request,
                "invalid_input",
                "Dataset profile requires a non-empty file_path input.",
                {"file_path": file_path},
            )

        path = Path(file_path)
        provenance = ToolProvenance(dataset_id=dataset_id, source=file_path)
        if not path.exists():
            return ToolResult.fail(
                tool_request,
                "invalid_input",
                "Dataset file does not exist.",
                {"file_path": file_path},
                provenance,
            )
        if path.suffix.lower() != ".csv":
            return ToolResult.fail(
                tool_request,
                "unsupported_method",
                "Dataset profile currently supports CSV files only.",
                {"file_path": file_path},
                provenance,
            )

        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - normalize tool boundary.
            return ToolResult.fail(
                tool_request,
                "execution_error",
                "Dataset profiling failed while reading the CSV file.",
                {"exception": type(exc).__name__, "message": str(exc)},
                provenance,
            )

        selected_columns = self._selected_columns(
            columns,
            dataframe,
            tool_request,
            provenance,
        )
        if isinstance(selected_columns, ToolResult):
            return selected_columns

        profile_frame = dataframe[selected_columns]
        rows_used = len(profile_frame)
        warnings = []
        if isinstance(sample_size, int):
            if sample_size <= 0:
                return ToolResult.fail(
                    tool_request,
                    "invalid_input",
                    "sample_size must be a positive integer.",
                    {"sample_size": sample_size},
                    provenance,
                )
            if sample_size < rows_used:
                profile_frame = profile_frame.head(sample_size)
                rows_used = sample_size
                warnings.append(
                    "Dataset profile used the first sample_size rows."
                )
        elif sample_size is not None:
            return ToolResult.fail(
                tool_request,
                "invalid_input",
                "sample_size must be an integer when provided.",
                {"sample_size": sample_size},
                provenance,
            )

        if dataframe.empty:
            warnings.append("Dataset is empty.")

        data = {
            "shape": {
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
            },
            "rows_used": int(rows_used),
            "columns": [
                self._column_profile(column, profile_frame[column], len(dataframe))
                for column in selected_columns
            ],
        }
        return ToolResult.ok(
            tool_request,
            data,
            "Dataset profile was generated.",
            ToolProvenance(
                dataset_id=dataset_id,
                source=file_path,
                columns=list(selected_columns),
            ),
            warnings,
        )

    def as_langchain_callable(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Return a dict-in/dict-out callable suitable for LangChain wrappers."""

        def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
            """Profile a dataset and return a serializable ToolResult payload."""
            return self.invoke(payload).model_dump(mode="json")

        return call_tool

    def _request(self, request: ToolRequest | dict[str, Any]) -> ToolRequest:
        """Normalize supported request inputs into a ToolRequest instance."""
        if isinstance(request, ToolRequest):
            return request
        payload = dict(request)
        if "inputs" not in payload:
            payload = {"inputs": payload}
        payload.setdefault("tool_name", self.tool_name)
        payload.setdefault("request_id", f"{self.tool_name}:default")
        payload.setdefault("caller", "dataset_profile_tool")
        payload.setdefault("purpose", "Generate dataset profile metadata.")
        return ToolRequest(**payload)

    def _selected_columns(
        self,
        columns: Any,
        dataframe: pd.DataFrame,
        request: ToolRequest,
        provenance: ToolProvenance,
    ) -> list[str] | ToolResult:
        """Validate and return the columns that should be profiled."""
        if columns is None:
            return [str(column) for column in dataframe.columns]
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            return ToolResult.fail(
                request,
                "invalid_input",
                "columns must be a list of strings when provided.",
                {"columns": columns},
                provenance,
            )
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            return ToolResult.fail(
                request,
                "invalid_column",
                "Requested profile columns are not present in the dataset.",
                {"missing_columns": missing},
                provenance,
            )
        return columns

    def _column_profile(
        self,
        column: str,
        series: pd.Series,
        total_rows: int,
    ) -> dict[str, Any]:
        """Build JSON-compatible metadata for one dataframe column."""
        missing_count = int(series.isna().sum())
        profile = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_ratio": self._safe_ratio(missing_count, total_rows),
            "unique_count": int(series.nunique(dropna=True)),
            "inferred_type": self._inferred_type(series),
        }
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            profile["numeric_summary"] = {
                "count": int(clean.count()),
                "mean": self._safe_number(clean.mean()),
                "std": self._safe_number(clean.std()),
                "min": self._safe_number(clean.min()),
                "median": self._safe_number(clean.median()),
                "max": self._safe_number(clean.max()),
            }
        else:
            profile["sample_values"] = [
                self._safe_value(value)
                for value in series.dropna().drop_duplicates().head(5).tolist()
            ]
        return profile

    @staticmethod
    def _inferred_type(series: pd.Series) -> str:
        """Infer a compact semantic type label for profile consumers."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"

    @staticmethod
    def _safe_number(value: Any) -> float | int | None:
        """Convert pandas numeric outputs into JSON-compatible values."""
        if pd.isna(value):
            return None
        number = float(value)
        if number.is_integer():
            return int(number)
        return number

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Return a rounded ratio while avoiding division by zero."""
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 6)

    @staticmethod
    def _safe_value(value: Any) -> Any:
        """Convert dataframe cell values into JSON-compatible scalar values."""
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value
