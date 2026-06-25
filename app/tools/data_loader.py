from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.tools.schema import DataLoaderTool, ToolProvenance, ToolRequest, ToolResult
from langchain.tools import tool

class CSVDataLoaderTool(DataLoaderTool):
    """Load CSV data and expose JSON-safe row retrieval methods."""

    def __init__(
        self,
        file_path: str,
        dataset_id: str | None = None,
        encoding: str | None = None,
    ) -> None:
        """Initialize the loader with a CSV path and optional dataset metadata."""
        self.file_path = file_path
        self.dataset_id = dataset_id
        self.encoding = encoding
        self._dataframe: pd.DataFrame | None = None
    @tool("csv_dataloader.load")
    def load(self) -> ToolResult:
        """Load the configured CSV file into memory."""
        request = self._request("tabular.csv.load", "Load CSV data.")
        path = Path(self.file_path)
        if not path.exists():
            return ToolResult.fail(
                request,
                "invalid_input",
                "CSV file does not exist.",
                {"file_path": self.file_path},
                self._provenance(),
            )
        if path.suffix.lower() != ".csv":
            return ToolResult.fail(
                request,
                "invalid_input",
                "Data loader only supports CSV files.",
                {"file_path": self.file_path},
                self._provenance(),
            )
        try:
            self._dataframe = pd.read_csv(path, encoding=self.encoding)
        except Exception as exc:  # noqa: BLE001 - normalize loader boundary.
            return ToolResult.fail(
                request,
                "execution_error",
                "CSV loading failed.",
                {"exception": type(exc).__name__, "message": str(exc)},
                self._provenance(),
            )
        data = {
            "shape": {
                "rows": int(self._dataframe.shape[0]),
                "columns": int(self._dataframe.shape[1]),
            },
            "columns": [str(column) for column in self._dataframe.columns],
        }
        return ToolResult.ok(
            request,
            data,
            "CSV file was loaded.",
            self._provenance(rows_used=int(self._dataframe.shape[0])),
        )

    def fetch_all(self) -> ToolResult:
        """Return all records from the loaded CSV file."""
        request = self._request("tabular.csv.fetch_all", "Fetch all CSV rows.")
        dataframe = self._loaded_dataframe(request)
        if isinstance(dataframe, ToolResult):
            return dataframe
        return ToolResult.ok(
            request,
            self._records(dataframe),
            "All CSV rows were fetched.",
            self._provenance(rows_used=int(dataframe.shape[0])),
        )

    def fetch_features(self, features: list[str]) -> ToolResult:
        """Return records containing only the requested CSV feature columns."""
        request = self._request(
            "tabular.csv.fetch_features",
            "Fetch selected CSV feature columns.",
        )
        dataframe = self._loaded_dataframe(request)
        if isinstance(dataframe, ToolResult):
            return dataframe
        missing = [feature for feature in features if feature not in dataframe.columns]
        if missing:
            return ToolResult.fail(
                request,
                "invalid_column",
                "Requested features are not present in the CSV file.",
                {"missing_features": missing},
                self._provenance(columns=features, rows_used=int(dataframe.shape[0])),
            )
        selected = dataframe[features]
        return ToolResult.ok(
            request,
            self._records(selected),
            "Selected CSV feature columns were fetched.",
            self._provenance(columns=features, rows_used=int(selected.shape[0])),
        )

    def _loaded_dataframe(self, request: ToolRequest) -> pd.DataFrame | ToolResult:
        """Return the loaded dataframe or a normalized error result."""
        if self._dataframe is not None:
            return self._dataframe
        load_result = self.load()
        if load_result.status == "error":
            return ToolResult.fail(
                request,
                load_result.error.type if load_result.error else "execution_error",
                load_result.summary,
                load_result.error.details if load_result.error else {},
                load_result.provenance,
            )
        return self._dataframe

    def _request(self, tool_name: str, purpose: str) -> ToolRequest:
        """Create a ToolRequest for one CSV loader operation."""
        return ToolRequest(
            tool_name=tool_name,
            request_id=f"{tool_name}:{self.dataset_id or self.file_path}",
            caller="csv_data_loader",
            purpose=purpose,
            inputs={
                "file_path": self.file_path,
                "dataset_id": self.dataset_id,
            },
        )

    def _provenance(
        self,
        columns: list[str] | None = None,
        rows_used: int | None = None,
    ) -> ToolProvenance:
        """Build provenance for one CSV loader result."""
        return ToolProvenance(
            dataset_id=self.dataset_id,
            source=self.file_path,
            columns=columns or [],
        )

    @staticmethod
    def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        """Convert a dataframe into JSON-safe row records."""
        safe = dataframe.where(pd.notna(dataframe), None)
        return safe.to_dict(orient="records")
