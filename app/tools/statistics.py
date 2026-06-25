from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.tools.schema import ToolProvenance, ToolRequest, ToolResult


class StatisticalAnalysisTool:
    """Run deterministic statistical analysis through ToolRequest payloads."""

    CORRELATION = "stats.correlation"
    BASIC_SUMMARY = "stats.basic_summary"
    CUSTOM_METRIC = "stats.custom_metric"
    APPROVED_CUSTOM_METRICS = {
        "sum",
        "mean",
        "count",
        "ratio_of_sums",
        "difference_of_means",
    }
    _SUPPORTED_TOOLS = {
        CORRELATION,
        BASIC_SUMMARY,
        CUSTOM_METRIC,
    }

    def __call__(self, request: ToolRequest | dict[str, Any]) -> ToolResult:
        """Run the requested statistical operation."""
        return self.invoke(request)

    def invoke(self, request: ToolRequest | dict[str, Any]) -> ToolResult:
        """Dispatch one supported statistical tool request.

        Args:
            request: A `ToolRequest` or dictionary containing `tool_name` and
                `inputs`. `file_path` is required in the inputs.

        Returns:
            A normalized `ToolResult` with JSON-safe statistical output.
        """
        tool_request = self._request(request)
        if tool_request.tool_name not in self._SUPPORTED_TOOLS:
            return ToolResult.fail(
                tool_request,
                "unsupported_method",
                "Unsupported statistical analysis tool.",
                {"tool_name": tool_request.tool_name},
            )

        dataframe = self._read_csv(tool_request)
        if isinstance(dataframe, ToolResult):
            return dataframe

        if tool_request.tool_name == self.CUSTOM_METRIC:
            return self._custom_metric(tool_request, dataframe)

        columns = self._selected_columns(tool_request, dataframe)
        if isinstance(columns, ToolResult):
            return columns

        if tool_request.tool_name == self.CORRELATION:
            return self._correlation(tool_request, dataframe, columns)
        return self._basic_summary(tool_request, dataframe, columns)

    def _request(self, request: ToolRequest | dict[str, Any]) -> ToolRequest:
        """Normalize supported request payloads into a ToolRequest."""
        if isinstance(request, ToolRequest):
            return request
        payload = dict(request)
        if "inputs" not in payload:
            payload = {"inputs": payload}
        payload.setdefault("tool_name", self.BASIC_SUMMARY)
        payload.setdefault(
            "request_id",
            f"{payload['tool_name']}:"
            f"{payload['inputs'].get('dataset_id', 'default')}",
        )
        payload.setdefault("caller", "statistical_analysis_tool")
        payload.setdefault("purpose", "Run deterministic statistical analysis.")
        return ToolRequest(**payload)

    def _read_csv(self, request: ToolRequest) -> pd.DataFrame | ToolResult:
        """Read the requested CSV file or return a normalized failure."""
        file_path = request.inputs.get("file_path")
        dataset_id = request.inputs.get("dataset_id")
        if not isinstance(file_path, str) or not file_path:
            return ToolResult.fail(
                request,
                "invalid_input",
                "Statistical analysis requires a non-empty file_path input.",
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
                "Statistical analysis currently supports CSV files only.",
                {"file_path": file_path},
                provenance,
            )
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - normalize tool boundary.
            return ToolResult.fail(
                request,
                "execution_error",
                "CSV reading failed during statistical analysis.",
                {"exception": type(exc).__name__, "message": str(exc)},
                provenance,
            )

    def _selected_columns(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> list[str] | ToolResult:
        """Validate requested columns and return their string names."""
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

    def _correlation(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> ToolResult:
        """Compute correlation or association for selected column pairs."""
        if len(columns) < 2:
            return ToolResult.fail(
                request,
                "invalid_input",
                "Correlation requires at least two columns.",
                {"columns": columns},
                self._provenance(request, columns),
            )

        method = request.inputs.get("method", "auto")
        if method not in {
            "auto",
            "pearson",
            "spearman",
            "cramers_v",
            "correlation_ratio",
        }:
            return ToolResult.fail(
                request,
                "unsupported_method",
                "Unsupported correlation or association method.",
                {"method": method},
                self._provenance(request, columns),
            )

        pairs = []
        warnings = []
        for left, right in combinations(columns, 2):
            result, warning = self._association_pair(
                dataframe,
                left,
                right,
                method,
            )
            if result:
                pairs.append(result)
            if warning:
                warnings.append(warning)

        if not pairs:
            return ToolResult.fail(
                request,
                "incompatible_type",
                "No compatible column pairs were available for association.",
                {"columns": columns, "warnings": warnings},
                self._provenance(request, columns),
            )

        data = {
            "requested_method": method,
            "columns": columns,
            "pairs": pairs,
        }
        return ToolResult.ok(
            request,
            data,
            f"Generated {len(pairs)} correlation or association result(s).",
            self._provenance(request, columns),
            warnings,
        )

    def _association_pair(
        self,
        dataframe: pd.DataFrame,
        left: str,
        right: str,
        method: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compute one supported association result for a column pair."""
        left_kind = self._inferred_type(dataframe[left])
        right_kind = self._inferred_type(dataframe[right])
        pair_method = self._pair_method(method, left_kind, right_kind)
        if pair_method is None:
            return None, (
                f"{left} and {right} are not compatible with method "
                f"{method}."
            )
        if pair_method in {"pearson", "spearman"}:
            return self._numeric_correlation(dataframe, left, right, pair_method)
        if pair_method == "cramers_v":
            return self._cramers_v(dataframe, left, right)
        return self._correlation_ratio(dataframe, left, right)

    def _numeric_correlation(
        self,
        dataframe: pd.DataFrame,
        left: str,
        right: str,
        method: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compute a numeric Pearson or Spearman correlation."""
        pair = dataframe[[left, right]].apply(pd.to_numeric, errors="coerce")
        valid = pair.dropna()
        if len(valid) < 2:
            return None, f"{left} and {right} have fewer than two valid rows."
        if valid[left].nunique() < 2 or valid[right].nunique() < 2:
            return None, f"{left} and {right} include a constant numeric input."
        value = valid[left].corr(valid[right], method=method)
        if pd.isna(value):
            return None, f"{left} and {right} correlation could not be computed."
        return {
            "columns": [left, right],
            "method": method,
            "association_type": "numeric_numeric",
            "value": self._safe_number(value),
            "rows_used": int(len(valid)),
            "missing_rows_dropped": int(len(pair) - len(valid)),
        }, None

    def _cramers_v(
        self,
        dataframe: pd.DataFrame,
        left: str,
        right: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compute Cramer's V for two categorical-like columns."""
        valid = dataframe[[left, right]].dropna()
        if valid.empty:
            return None, f"{left} and {right} have no complete rows."
        table = pd.crosstab(valid[left].astype(str), valid[right].astype(str))
        if min(table.shape) < 2:
            return None, f"{left} and {right} need at least two categories."
        values = table.to_numpy(dtype=float)
        chi_square = self._chi_square(values)
        total = values.sum()
        denominator = total * (min(table.shape) - 1)
        if denominator == 0:
            return None, f"{left} and {right} Cramer's V denominator is zero."
        value = np.sqrt(chi_square / denominator)
        return {
            "columns": [left, right],
            "method": "cramers_v",
            "association_type": "categorical_categorical",
            "value": self._safe_number(value),
            "rows_used": int(len(valid)),
            "missing_rows_dropped": int(len(dataframe) - len(valid)),
        }, None

    def _correlation_ratio(
        self,
        dataframe: pd.DataFrame,
        left: str,
        right: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compute correlation ratio for categorical and numeric columns."""
        left_kind = self._inferred_type(dataframe[left])
        if left_kind == "numeric":
            numeric_column = left
            category_column = right
        else:
            numeric_column = right
            category_column = left

        valid = dataframe[[category_column, numeric_column]].copy()
        valid[numeric_column] = pd.to_numeric(
            valid[numeric_column],
            errors="coerce",
        )
        valid = valid.dropna()
        if valid.empty:
            return None, (
                f"{category_column} and {numeric_column} have no complete rows."
            )
        grouped = valid.groupby(category_column, dropna=True)[numeric_column]
        if grouped.ngroups < 2:
            return None, f"{category_column} needs at least two groups."
        overall_mean = valid[numeric_column].mean()
        total_ss = ((valid[numeric_column] - overall_mean) ** 2).sum()
        if total_ss == 0:
            return None, f"{numeric_column} is constant across groups."
        between_ss = sum(
            len(group) * (group.mean() - overall_mean) ** 2
            for _, group in grouped
        )
        value = np.sqrt(between_ss / total_ss)
        return {
            "columns": [category_column, numeric_column],
            "method": "correlation_ratio",
            "association_type": "categorical_numeric",
            "value": self._safe_number(value),
            "rows_used": int(len(valid)),
            "missing_rows_dropped": int(len(dataframe) - len(valid)),
        }, None

    def _basic_summary(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> ToolResult:
        """Return basic statistical summaries for selected columns."""
        data = {
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "columns": [
                self._summary_for_column(dataframe[column])
                for column in columns
            ],
        }
        return ToolResult.ok(
            request,
            data,
            "Basic statistical summary was generated.",
            self._provenance(request, columns),
        )

    def _summary_for_column(self, series: pd.Series) -> dict[str, Any]:
        """Build a JSON-safe basic summary for one column."""
        kind = self._inferred_type(series)
        summary: dict[str, Any] = {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "inferred_type": kind,
            "row_count": int(len(series)),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": self._safe_ratio(
                int(series.isna().sum()),
                int(len(series)),
            ),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if kind == "numeric":
            summary["numeric"] = self._numeric_summary(series)
        elif kind == "datetime":
            summary["datetime"] = self._datetime_summary(series)
        else:
            summary["categorical"] = self._categorical_summary(series)
        return summary

    def _numeric_summary(self, series: pd.Series) -> dict[str, Any]:
        """Return basic numeric distribution metrics."""
        values = pd.to_numeric(series, errors="coerce").dropna()
        quantiles = values.quantile([0.25, 0.5, 0.75]) if not values.empty else {}
        return {
            "count": int(values.count()),
            "mean": self._safe_number(values.mean()),
            "std": self._safe_number(values.std()),
            "min": self._safe_number(values.min()),
            "q1": self._safe_number(quantiles.get(0.25)),
            "median": self._safe_number(quantiles.get(0.5)),
            "q3": self._safe_number(quantiles.get(0.75)),
            "max": self._safe_number(values.max()),
            "sum": self._safe_number(values.sum()),
        }

    def _categorical_summary(self, series: pd.Series) -> dict[str, Any]:
        """Return frequencies for categorical-like columns."""
        values = series.dropna().astype(str)
        counts = values.value_counts().head(10)
        return {
            "top_values": [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ],
            "mode": str(counts.index[0]) if not counts.empty else None,
        }

    def _datetime_summary(self, series: pd.Series) -> dict[str, Any]:
        """Return bounds for datetime-like columns."""
        values = self._parse_datetime(series.dropna()).dropna()
        return {
            "count": int(values.count()),
            "min": values.min().isoformat() if not values.empty else None,
            "max": values.max().isoformat() if not values.empty else None,
        }

    def _custom_metric(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> ToolResult:
        """Compute an approved deterministic custom metric."""
        metric = request.inputs.get("metric")
        if metric not in self.APPROVED_CUSTOM_METRICS:
            return ToolResult.fail(
                request,
                "unsupported_method",
                "Custom metric is not approved.",
                {
                    "metric": metric,
                    "approved_metrics": sorted(self.APPROVED_CUSTOM_METRICS),
                },
                self._provenance(request),
            )

        if metric in {"sum", "mean"}:
            return self._single_column_metric(request, dataframe, metric)
        if metric == "count":
            return self._count_metric(request, dataframe)
        if metric == "ratio_of_sums":
            return self._ratio_of_sums(request, dataframe)
        return self._difference_of_means(request, dataframe)

    def _single_column_metric(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        metric: str,
    ) -> ToolResult:
        """Compute sum or mean for one numeric column."""
        column = self._required_column(request, dataframe, "column")
        if isinstance(column, ToolResult):
            return column
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            return ToolResult.fail(
                request,
                "insufficient_rows",
                "Custom metric requires at least one numeric value.",
                {"column": column},
                self._provenance(request, [column]),
            )
        value = values.sum() if metric == "sum" else values.mean()
        data = {
            "metric": metric,
            "value": self._safe_number(value),
            "columns": [column],
            "rows_used": int(values.count()),
            "missing_rows_dropped": int(len(dataframe) - values.count()),
        }
        return ToolResult.ok(
            request,
            data,
            f"Custom metric {metric} was computed.",
            self._provenance(request, [column]),
        )

    def _count_metric(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> ToolResult:
        """Count rows or non-null values for one optional column."""
        column = request.inputs.get("column")
        if column is None:
            data = {
                "metric": "count",
                "value": int(len(dataframe)),
                "columns": [],
                "rows_used": int(len(dataframe)),
            }
            return ToolResult.ok(
                request,
                data,
                "Custom metric count was computed.",
                self._provenance(request),
            )
        if not isinstance(column, str) or column not in dataframe.columns:
            return ToolResult.fail(
                request,
                "invalid_column",
                "count column is not present in the dataset.",
                {"column": column},
                self._provenance(request),
            )
        value = int(dataframe[column].notna().sum())
        data = {
            "metric": "count",
            "value": value,
            "columns": [column],
            "rows_used": int(len(dataframe)),
        }
        return ToolResult.ok(
            request,
            data,
            "Custom metric count was computed.",
            self._provenance(request, [column]),
        )

    def _ratio_of_sums(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> ToolResult:
        """Compute numerator sum divided by denominator sum."""
        numerator = self._required_column(
            request,
            dataframe,
            "numerator_column",
        )
        if isinstance(numerator, ToolResult):
            return numerator
        denominator = self._required_column(
            request,
            dataframe,
            "denominator_column",
        )
        if isinstance(denominator, ToolResult):
            return denominator
        values = dataframe[[numerator, denominator]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        valid = values.dropna()
        if valid.empty:
            return ToolResult.fail(
                request,
                "insufficient_rows",
                "ratio_of_sums requires complete numeric rows.",
                {"columns": [numerator, denominator]},
                self._provenance(request, [numerator, denominator]),
            )
        denominator_sum = valid[denominator].sum()
        if denominator_sum == 0:
            return ToolResult.fail(
                request,
                "invalid_result_shape",
                "ratio_of_sums denominator sum is zero.",
                {"denominator_column": denominator},
                self._provenance(request, [numerator, denominator]),
            )
        numerator_sum = valid[numerator].sum()
        data = {
            "metric": "ratio_of_sums",
            "value": self._safe_number(numerator_sum / denominator_sum),
            "numerator_sum": self._safe_number(numerator_sum),
            "denominator_sum": self._safe_number(denominator_sum),
            "columns": [numerator, denominator],
            "rows_used": int(len(valid)),
            "missing_rows_dropped": int(len(values) - len(valid)),
        }
        return ToolResult.ok(
            request,
            data,
            "Custom metric ratio_of_sums was computed.",
            self._provenance(request, [numerator, denominator]),
        )

    def _difference_of_means(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
    ) -> ToolResult:
        """Compute the mean difference between two selected groups."""
        value_column = self._required_column(request, dataframe, "value_column")
        if isinstance(value_column, ToolResult):
            return value_column
        group_column = self._required_column(request, dataframe, "group_column")
        if isinstance(group_column, ToolResult):
            return group_column
        values = dataframe[[group_column, value_column]].copy()
        values[value_column] = pd.to_numeric(
            values[value_column],
            errors="coerce",
        )
        values[group_column] = values[group_column].astype(str)
        values = values.dropna()
        groups = self._selected_groups(request, values[group_column])
        if isinstance(groups, ToolResult):
            return groups
        grouped = values[values[group_column].isin(groups)].groupby(group_column)
        means = grouped[value_column].mean()
        if len(means) != 2:
            return ToolResult.fail(
                request,
                "insufficient_rows",
                "difference_of_means requires rows for exactly two groups.",
                {"groups": groups},
                self._provenance(request, [group_column, value_column]),
            )
        difference = means.loc[groups[0]] - means.loc[groups[1]]
        data = {
            "metric": "difference_of_means",
            "value": self._safe_number(difference),
            "groups": groups,
            "group_means": {
                str(group): self._safe_number(means.loc[group])
                for group in groups
            },
            "columns": [group_column, value_column],
            "rows_used": int(len(values[values[group_column].isin(groups)])),
            "missing_rows_dropped": int(len(dataframe) - len(values)),
        }
        return ToolResult.ok(
            request,
            data,
            "Custom metric difference_of_means was computed.",
            self._provenance(request, [group_column, value_column]),
        )

    def _required_column(
        self,
        request: ToolRequest,
        dataframe: pd.DataFrame,
        input_name: str,
    ) -> str | ToolResult:
        """Return a required column input or a normalized error."""
        column = request.inputs.get(input_name)
        if not isinstance(column, str) or column not in dataframe.columns:
            return ToolResult.fail(
                request,
                "invalid_column",
                f"{input_name} is not present in the dataset.",
                {input_name: column},
                self._provenance(request),
            )
        return column

    def _selected_groups(
        self,
        request: ToolRequest,
        series: pd.Series,
    ) -> list[Any] | ToolResult:
        """Return two groups supplied by input or selected deterministically."""
        requested = request.inputs.get("groups")
        if requested is not None:
            if not isinstance(requested, list) or len(requested) != 2:
                return ToolResult.fail(
                    request,
                    "invalid_input",
                    "groups must contain exactly two values when provided.",
                    {"groups": requested},
                    self._provenance(request),
                )
            return [str(value) for value in requested]
        counts = series.value_counts()
        if len(counts) < 2:
            return ToolResult.fail(
                request,
                "insufficient_rows",
                "difference_of_means requires at least two groups.",
                {},
                self._provenance(request),
            )
        return counts.index[:2].tolist()

    @staticmethod
    def _pair_method(
        method: str,
        left_kind: str,
        right_kind: str,
    ) -> str | None:
        """Choose the concrete association method for a pair."""
        if method != "auto":
            return method
        if left_kind == "numeric" and right_kind == "numeric":
            return "pearson"
        categorical = {"categorical", "text", "boolean"}
        if left_kind in categorical and right_kind in categorical:
            return "cramers_v"
        if "numeric" in {left_kind, right_kind}:
            if left_kind in categorical or right_kind in categorical:
                return "correlation_ratio"
        return None

    @staticmethod
    def _inferred_type(series: pd.Series) -> str:
        """Infer the compact data type needed for statistical routing."""
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        parsed = StatisticalAnalysisTool._parse_datetime(series.dropna())
        if not parsed.empty and parsed.notna().mean() >= 0.8:
            return "datetime"
        non_null = series.dropna().astype(str)
        if not non_null.empty and non_null.str.len().mean() > 50:
            return "text"
        return "categorical"

    @staticmethod
    def _chi_square(values: np.ndarray) -> float:
        """Compute chi-square statistic for a contingency table."""
        total = values.sum()
        row_totals = values.sum(axis=1, keepdims=True)
        column_totals = values.sum(axis=0, keepdims=True)
        expected = row_totals @ column_totals / total
        mask = expected > 0
        return float(((values - expected) ** 2 / expected)[mask].sum())

    @staticmethod
    def _safe_number(value: Any) -> float | int | None:
        """Convert numpy or pandas numeric outputs into JSON-safe values."""
        if value is None or pd.isna(value):
            return None
        number = float(value)
        if number.is_integer():
            return int(number)
        return round(number, 12)

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        """Return a rounded ratio while avoiding division by zero."""
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 6)

    @staticmethod
    def _parse_datetime(series: pd.Series) -> pd.Series:
        """Parse datetime-like values for summary output."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def _provenance(
        request: ToolRequest,
        columns: list[str] | None = None,
    ) -> ToolProvenance:
        """Build provenance for a statistical analysis result."""
        return ToolProvenance(
            dataset_id=request.inputs.get("dataset_id"),
            source=request.inputs.get("file_path"),
            columns=columns or [],
        )
