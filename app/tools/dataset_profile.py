from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from app.tools.schema import ToolProvenance, ToolRequest, ToolResult


class DatasetProfileTool:
    """Profile a CSV dataset through the shared tool request/result contract."""

    tool_name = "tabular.dataset_profile"
    COLUMN_METADATA = "tabular.column_metadata"
    MISSINGNESS_SUMMARY = "tabular.missingness_summary"
    TYPE_COMPATIBILITY = "tabular.type_compatibility"
    _SUPPORTED_PROFILE_TOOLS = {
        tool_name,
        COLUMN_METADATA,
        MISSINGNESS_SUMMARY,
        TYPE_COMPATIBILITY,
    }

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
        if tool_request.tool_name != self.tool_name:
            return self._invoke_dataset_inspection(tool_request)

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
        payload.setdefault(
            "request_id",
            f"{payload['tool_name']}:"
            f"{payload['inputs'].get('dataset_id', 'default')}",
        )
        payload.setdefault("caller", "dataset_profile_tool")
        payload.setdefault("purpose", "Generate dataset profile metadata.")
        return ToolRequest(**payload)

    def _invoke_dataset_inspection(self, request: ToolRequest) -> ToolResult:
        """Dispatch column metadata, missingness, and compatibility tools."""
        if request.tool_name not in self._SUPPORTED_PROFILE_TOOLS:
            return ToolResult.fail(
                request,
                "unsupported_method",
                "Unsupported dataset profile tool.",
                {"tool_name": request.tool_name},
            )

        dataframe = self._read_csv(request)
        if isinstance(dataframe, ToolResult):
            return dataframe

        columns = self._selected_inspection_columns(request, dataframe)
        if isinstance(columns, ToolResult):
            return columns

        if request.tool_name == self.COLUMN_METADATA:
            return self._column_metadata(request, dataframe, columns)
        if request.tool_name == self.MISSINGNESS_SUMMARY:
            return self._missingness_summary(request, dataframe, columns)
        return self._type_compatibility(request, dataframe, columns)

    def _read_csv(self, request: ToolRequest) -> pd.DataFrame | ToolResult:
        """Read the CSV file named by the request or return a tool error."""
        file_path = request.inputs.get("file_path")
        dataset_id = request.inputs.get("dataset_id")
        if not isinstance(file_path, str) or not file_path:
            return ToolResult.fail(
                request,
                "invalid_input",
                "Dataset profile requires a non-empty file_path input.",
                {"file_path": file_path},
            )

        path = Path(file_path)
        provenance = ToolProvenance(dataset_id=dataset_id, source=file_path)
        if not path.exists():
            return ToolResult.fail(
                request,
                "invalid_input",
                "Dataset file does not exist.",
                {"file_path": file_path},
                provenance,
            )
        if path.suffix.lower() != ".csv":
            return ToolResult.fail(
                request,
                "unsupported_method",
                "Dataset profile currently supports CSV files only.",
                {"file_path": file_path},
                provenance,
            )
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - normalize tool boundary.
            return ToolResult.fail(
                request,
                "execution_error",
                "CSV reading failed during dataset profile inspection.",
                {"exception": type(exc).__name__, "message": str(exc)},
                provenance,
            )

    def _selected_inspection_columns(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> list[str] | ToolResult:
        """Validate requested inspection columns and return their names."""
        requested = request.inputs.get("columns")
        all_columns = [str(column) for column in dataframe.columns]
        if requested is None:
            return all_columns
        if not isinstance(requested, list) or not all(
            isinstance(column, str) for column in requested
        ):
            return ToolResult.fail(
                request,
                "invalid_input",
                "columns must be a list of strings when provided.",
                {"columns": requested},
                self._provenance(request),
            )
        missing = [column for column in requested if column not in all_columns]
        if missing:
            return ToolResult.fail(
                request,
                "invalid_column",
                "Requested columns are not present in the dataset.",
                {"missing_columns": missing},
                self._provenance(request, requested),
            )
        return requested

    def _column_metadata(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> ToolResult:
        """Return per-column structural metadata and capability flags."""
        data = {
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": [
                self._metadata_for_column(dataframe, column)
                for column in columns
            ],
        }
        return ToolResult.ok(
            request,
            data,
            "Column metadata was generated.",
            self._provenance(request, columns),
        )

    def _missingness_summary(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> ToolResult:
        """Return missing-value counts and ratios for selected columns."""
        selected = dataframe[columns]
        total_rows = int(dataframe.shape[0])
        rows_with_any_missing = int(selected.isna().any(axis=1).sum())
        data = {
            "row_count": total_rows,
            "rows_with_any_missing": rows_with_any_missing,
            "rows_with_any_missing_ratio": self._safe_ratio(
                rows_with_any_missing,
                total_rows,
            ),
            "columns": [
                {
                    "name": column,
                    "missing_count": int(selected[column].isna().sum()),
                    "missing_ratio": self._safe_ratio(
                        int(selected[column].isna().sum()),
                        total_rows,
                    ),
                    "non_null_count": int(selected[column].notna().sum()),
                    "warnings": self._missingness_warnings(
                        column,
                        selected[column],
                        total_rows,
                    ),
                }
                for column in columns
            ],
        }
        return ToolResult.ok(
            request,
            data,
            "Missingness summary was generated.",
            self._provenance(request, columns),
            self._aggregate_warnings(data["columns"]),
        )

    def _type_compatibility(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> ToolResult:
        """Validate selected columns against a requested analysis operation."""
        operation = request.inputs.get("operation")
        if not isinstance(operation, str) or not operation:
            return ToolResult.fail(
                request,
                "invalid_input",
                "Type compatibility requires a non-empty operation input.",
                {"operation": operation},
                self._provenance(request, columns),
            )

        metadata = [
            self._metadata_for_column(dataframe, column)
            for column in columns
        ]
        compatible, reasons = self._compatibility_for_operation(
            operation,
            metadata,
        )
        data = {
            "operation": operation,
            "compatible": compatible,
            "columns": [
                {
                    "name": item["name"],
                    "inferred_type": item["inferred_type"],
                    "capabilities": item["capabilities"],
                }
                for item in metadata
            ],
            "blocking_reasons": reasons,
        }
        status_warnings = [] if compatible else reasons
        return ToolResult.ok(
            request,
            data,
            "Type compatibility was evaluated.",
            self._provenance(request, columns),
            status_warnings,
        )

    def _metadata_for_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> dict[str, Any]:
        """Build JSON-safe metadata for one dataframe column."""
        series = dataframe[column]
        total_rows = int(dataframe.shape[0])
        non_null = series.dropna()
        inferred_type = self._inferred_type(series)
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        metadata: dict[str, Any] = {
            "name": column,
            "index": int(dataframe.columns.get_loc(column)),
            "dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "row_count": total_rows,
            "non_null_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_ratio": self._safe_ratio(missing_count, total_rows),
            "unique_count": unique_count,
            "unique_ratio": self._safe_ratio(
                unique_count,
                max(int(series.notna().sum()), 1),
            ),
        }
        metadata["numeric"] = self._numeric_metadata(non_null, inferred_type)
        metadata["categorical"] = self._categorical_metadata(
            non_null,
            inferred_type,
        )
        metadata["datetime"] = self._datetime_metadata(non_null, inferred_type)
        metadata["capabilities"] = self._capabilities(
            metadata,
            inferred_type,
        )
        metadata["warnings"] = self._column_warnings(series, metadata)
        return metadata

    def _numeric_metadata(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        """Return numeric summary data for numeric columns."""
        if inferred_type != "numeric":
            return None
        return {
            "min": self._safe_number(series.min()),
            "max": self._safe_number(series.max()),
            "mean": self._safe_number(series.mean()),
            "median": self._safe_number(series.median()),
            "std": self._safe_number(series.std()),
            "has_zero": bool((series == 0).any()),
            "has_negative": bool((series < 0).any()),
        }

    def _categorical_metadata(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        """Return sample values and frequencies for categorical-like columns."""
        if inferred_type not in {"categorical", "text", "boolean"}:
            return None
        values = series.astype(str)
        counts = values.value_counts().head(5)
        return {
            "top_values": [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ],
            "sample_values": values.drop_duplicates().head(5).tolist(),
            "average_length": self._safe_number(values.str.len().mean()),
        }

    def _datetime_metadata(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any] | None:
        """Return datetime bounds for datetime columns."""
        if inferred_type != "datetime":
            return None
        parsed = self._parse_datetime(series)
        valid = parsed.dropna()
        return {
            "parse_success_ratio": self._safe_ratio(len(valid), len(series)),
            "min": valid.min().isoformat() if not valid.empty else None,
            "max": valid.max().isoformat() if not valid.empty else None,
        }

    def _capabilities(
        self,
        metadata: dict[str, Any],
        inferred_type: str,
    ) -> dict[str, bool]:
        """Return operation capability flags for one column."""
        unique_ratio = metadata["unique_ratio"]
        return {
            "can_aggregate": inferred_type == "numeric",
            "can_group_by": inferred_type in {"categorical", "boolean"}
            and (metadata["unique_count"] <= 50 or unique_ratio <= 0.8),
            "can_filter": True,
            "can_sort": inferred_type in {"numeric", "datetime", "categorical"},
            "can_correlate": inferred_type == "numeric",
            "can_time_group": inferred_type == "datetime",
        }

    def _compatibility_for_operation(
        self,
        operation: str,
        metadata: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Return compatibility and blocking reasons for one operation."""
        operation = operation.lower()
        types = [item["inferred_type"] for item in metadata]
        if operation in {"correlation", "association_numeric"}:
            if len(metadata) < 2:
                return False, ["Correlation requires at least two columns."]
            if not all(item["capabilities"]["can_correlate"] for item in metadata):
                return False, ["Correlation requires numeric columns."]
            return True, []
        if operation in {"aggregate", "numeric_summary"}:
            if not any(item["capabilities"]["can_aggregate"] for item in metadata):
                return False, ["Aggregation requires at least one numeric column."]
            return True, []
        if operation in {"groupby_aggregate", "compare_categories"}:
            has_group = any(item["capabilities"]["can_group_by"] for item in metadata)
            has_numeric = any(
                item["capabilities"]["can_aggregate"]
                for item in metadata
            )
            if not has_group or not has_numeric:
                return False, [
                    "Group-by aggregation requires a groupable column "
                    "and a numeric column."
                ]
            return True, []
        if operation == "time_series":
            if "datetime" not in types:
                return False, ["Time-series analysis requires a datetime column."]
            return True, []
        if operation in {"filter", "sort"}:
            return True, []
        return False, [f"Unsupported operation: {operation}."]

    def _column_warnings(
        self,
        series: pd.Series,
        metadata: dict[str, Any],
    ) -> list[str]:
        """Return structural warnings for one column."""
        warnings_list = []
        if metadata["missing_count"] == metadata["row_count"]:
            warnings_list.append("Column contains only missing values.")
        elif metadata["missing_ratio"] >= 0.3:
            warnings_list.append("Column has high missingness.")
        if metadata["unique_count"] <= 1 and metadata["non_null_count"] > 0:
            warnings_list.append("Column is constant.")
        if (
            metadata["inferred_type"] in {"categorical", "text"}
            and metadata["unique_ratio"] > 0.8
        ):
            warnings_list.append("Column has high cardinality.")
        if self._looks_like_identifier(series, metadata):
            warnings_list.append("Column may be an identifier.")
        return warnings_list

    def _missingness_warnings(
        self,
        column: str,
        series: pd.Series,
        total_rows: int,
    ) -> list[str]:
        """Return missingness warnings for one selected column."""
        missing_count = int(series.isna().sum())
        ratio = self._safe_ratio(missing_count, total_rows)
        if missing_count == total_rows:
            return [f"{column} contains only missing values."]
        if ratio >= 0.3:
            return [f"{column} has high missingness."]
        return []

    @staticmethod
    def _aggregate_warnings(columns: list[dict[str, Any]]) -> list[str]:
        """Flatten column-level warning lists into result-level warnings."""
        return [
            warning
            for column in columns
            for warning in column.get("warnings", [])
        ]

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
        parsed = DatasetProfileTool._parse_datetime(series.dropna())
        if not parsed.empty and parsed.notna().mean() >= 0.8:
            return "datetime"
        non_null = series.dropna().astype(str)
        if not non_null.empty and non_null.str.len().mean() > 50:
            return "text"
        return "categorical"

    @staticmethod
    def _looks_like_identifier(
        series: pd.Series,
        metadata: dict[str, Any],
    ) -> bool:
        """Return whether a column has identifier-like cardinality."""
        name = metadata["name"].lower()
        if name.endswith("id") or name in {"id", "uuid", "guid"}:
            return True
        return (
            metadata["unique_ratio"] >= 0.95
            and metadata["inferred_type"] in {"numeric", "categorical", "text"}
            and int(series.notna().sum()) > 0
        )

    @staticmethod
    def _parse_datetime(series: pd.Series) -> pd.Series:
        """Parse datetimes without surfacing pandas format warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce")

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

    @staticmethod
    def _provenance(
        request: ToolRequest,
        columns: list[str] | None = None,
    ) -> ToolProvenance:
        """Build provenance for a dataset profile result."""
        return ToolProvenance(
            dataset_id=request.inputs.get("dataset_id"),
            source=request.inputs.get("file_path"),
            columns=columns or [],
        )
