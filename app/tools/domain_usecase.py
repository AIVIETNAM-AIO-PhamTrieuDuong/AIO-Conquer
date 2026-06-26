from __future__ import annotations

from typing import Any

from app.memory.domain_store import domain_store
from app.tools.schema import ToolProvenance, ToolRequest, ToolResult


class DomainUsecaseTool:
    """Look up precomputed multivariate use-cases from the domain store.

    Fetches full use-case records by id from the dedicated domain knowledge
    Redis store, resolves their ``comparison_pair`` columns against the active
    dataset columns, and returns a normalized ``ToolResult`` so the lookup is
    machine-readable and traceable like every other MVP tool.

    The id selection itself (which records are relevant to the question) stays in
    the calling agent/node; this tool is the deterministic fetch + column
    resolution step.
    """

    tool_name = "domain.usecase_lookup"

    # Map a use-case statistical test name to a supported association method.
    _TEST_METHODS = {"spearman": "spearman", "pearson": "pearson"}

    async def invoke(self, request: ToolRequest | dict[str, Any]) -> ToolResult:
        """Resolve requested use-case ids into dataset columns and a method."""
        tool_request = self._request(request)
        inputs = tool_request.inputs
        job_id = inputs.get("job_id")
        ids = inputs.get("ids")
        available_columns = inputs.get("available_columns") or []

        provenance = ToolProvenance(dataset_id=job_id, source="redis-domain-store")
        if not isinstance(job_id, str) or not job_id:
            return ToolResult.fail(
                tool_request,
                "invalid_input",
                "domain.usecase_lookup requires a non-empty job_id input.",
                {"job_id": job_id},
                provenance,
            )
        if not isinstance(ids, list) or not all(
            isinstance(item, (str, int)) for item in ids
        ):
            return ToolResult.fail(
                tool_request,
                "invalid_input",
                "ids must be a list of use-case record ids.",
                {"ids": ids},
                provenance,
            )

        records = await domain_store.get_records(job_id, [str(item) for item in ids])
        if not records:
            return ToolResult.ok(
                tool_request,
                {
                    "records": [],
                    "columns": [],
                    "unresolved_columns": [],
                    "association_method": "",
                },
                "No multivariate use-case records matched the requested ids.",
                provenance,
                ["No domain use-case records were found for the requested ids."],
            )

        resolved, unresolved = self._resolve_columns(records, available_columns)
        method = self._method(records)
        warnings: list[str] = []
        if unresolved:
            warnings.append(
                "Use-case columns not found in the active dataset: "
                + ", ".join(unresolved)
            )

        data = {
            "records": records,
            "columns": resolved,
            "unresolved_columns": unresolved,
            "association_method": method,
        }
        return ToolResult.ok(
            tool_request,
            data,
            f"Resolved {len(records)} multivariate use-case(s) into "
            f"{len(resolved)} dataset column(s).",
            ToolProvenance(
                dataset_id=job_id,
                source="redis-domain-store",
                columns=resolved,
            ),
            warnings,
        )

    def _resolve_columns(
        self,
        records: list[dict],
        available_columns: list[str],
    ) -> tuple[list[str], list[str]]:
        """Split use-case columns into ones that exist in the dataset and ones that don't.

        Matching is case-insensitive. When no ``available_columns`` are supplied,
        every referenced column is returned as resolved (no validation possible).
        """
        lookup = {str(column).lower(): str(column) for column in available_columns}
        resolved: list[str] = []
        unresolved: list[str] = []
        for record in records:
            pair = record.get("comparison_pair", {})
            for key in ("variable_a", "variable_b"):
                raw = str(pair.get(key, "")).strip()
                if not raw:
                    continue
                if not available_columns:
                    if raw not in resolved:
                        resolved.append(raw)
                    continue
                match = lookup.get(raw.lower())
                if match:
                    if match not in resolved:
                        resolved.append(match)
                elif raw not in unresolved:
                    unresolved.append(raw)
        return resolved, unresolved

    def _method(self, records: list[dict]) -> str:
        """Map a selected use-case statistical test to a supported tool method."""
        for record in records:
            test = str(
                record.get("metrics_and_significance", {}).get(
                    "statistical_test_type", ""
                )
            ).lower()
            for needle, method in self._TEST_METHODS.items():
                if needle in test:
                    return method
        return ""

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
            f"{payload['tool_name']}:{payload['inputs'].get('job_id', 'active')}",
        )
        payload.setdefault("caller", "domain_usecase_tool")
        payload.setdefault("purpose", "Look up multivariate use-cases by id.")
        return ToolRequest(**payload)


domain_usecase_tool = DomainUsecaseTool()
