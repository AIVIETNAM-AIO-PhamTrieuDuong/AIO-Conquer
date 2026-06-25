from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "app" / "data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_path: Path
    benchmark_path: Path


DEFAULT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="EmployeeAttrition",
        data_path=DATA_ROOT / "EmployeeAttrition" / "EmployeeAttrition.csv",
        benchmark_path=DATA_ROOT / "EmployeeAttrition" / "EmployeeAttrition_QA_Benchmark.csv",
    ),
    DatasetSpec(
        name="MobileGameChurn",
        data_path=DATA_ROOT / "MobileGameChurn" / "MobileGameChurn.csv",
        benchmark_path=DATA_ROOT / "MobileGameChurn" / "MobileGameChurn_QA_Benchmark.csv",
    ),
    DatasetSpec(
        name="SuperStore",
        data_path=DATA_ROOT / "SuperStore" / "SuperStores.csv",
        benchmark_path=DATA_ROOT / "SuperStore" / "SuperStore_QA_Benchmark.csv",
    ),
)

GENERATED_RESPONSE_COLUMNS = [
    "llm_answer",
    "generated_answer",
    "generated_explanation",
    "generated_confidence",
    "generated_fol",
    "generated_cot",
    "generated_premises",
    "generated_raw_response",
]

FAILED_ANSWER = "FAILED_ANSWER"


class TestBuilder:
    """Build chatbot evaluation outputs from benchmark Q/A CSV files."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
        output_dir: str | Path = REPO_ROOT / "app" / "evaluation" / "results",
        request_timeout: float = 120.0,
        eda_poll_interval: float = 2.0,
        eda_max_wait: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.datasets = tuple(datasets)
        self.output_dir = self._resolve_path(output_dir)
        self.request_timeout = request_timeout
        self.eda_poll_interval = eda_poll_interval
        self.eda_max_wait = eda_max_wait

    def upload_file_to_eda(self, file_path: str | Path, wait: bool = True) -> dict[str, Any]:
        """Upload one CSV/XLSX file to /eda/analyze and optionally wait for completion."""
        path = self._resolve_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with httpx.Client(timeout=self.request_timeout) as client:
            with path.open("rb") as file_obj:
                response = client.post(
                    f"{self.base_url}/eda/analyze",
                    files={"file": (path.name, file_obj, self._mime_type(path))},
                )
            response.raise_for_status()
            payload = response.json()

            job_id = payload.get("job_id")
            if not job_id:
                raise RuntimeError(f"/eda/analyze did not return a job_id: {payload}")

            if wait:
                payload["result"] = self._poll_eda_result(client, job_id)
            return payload

    def upload_dataset_files_to_eda(self, wait: bool = True) -> dict[str, dict[str, Any]]:
        """Upload all benchmark source datasets to /eda/analyze."""
        uploads: dict[str, dict[str, Any]] = {}
        for dataset in self.datasets:
            uploads[dataset.name] = self.upload_file_to_eda(dataset.data_path, wait=wait)
        return uploads

    def generate_answers(
        self,
        upload_first: bool = True,
        reset_history: bool = True,
        reset_before_each_question: bool = True,
    ) -> dict[str, Path]:
        """Generate chatbot answers for every configured benchmark CSV.

        Each output CSV contains the original question and expected answer plus
        the chatbot answer, explanation, confidence, and raw response fields.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}

        for dataset in self.datasets:
            if reset_history:
                self.reset_session()
            if upload_first:
                self.upload_file_to_eda(dataset.data_path, wait=True)

            output_paths[dataset.name] = self.generate_answers_for_dataset(
                dataset,
                reset_before_each_question=reset_before_each_question,
            )
        return output_paths

    def generate_answers_for_dataset(
        self,
        dataset: DatasetSpec,
        reset_before_each_question: bool = True,
    ) -> Path:
        benchmark_path = self._resolve_path(dataset.benchmark_path)
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

        rows = self._read_benchmark_rows(benchmark_path)
        output_path = self.output_dir / f"{dataset.name}_generated_answers.csv"

        with output_path.open("w", newline="", encoding="utf-8") as file_obj:
            fieldnames = [
                "dataset",
                "question",
                "expected_answer",
                "generated_answer",
                "explanation",
                "confidence",
                "fol",
                "cot",
                "premises",
                "raw_response",
            ]
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                question = row["Q"]
                candidate_answer = self._candidate_answer(row)
                if candidate_answer:
                    response = {"answer": candidate_answer}
                else:
                    if reset_before_each_question:
                        self.reset_session()
                    response = self.ask(question)
                writer.writerow(
                    {
                        "dataset": dataset.name,
                        "question": question,
                        "expected_answer": row["A"],
                        "generated_answer": response.get("answer", ""),
                        "explanation": response.get("explanation", ""),
                        "confidence": response.get("confidence", ""),
                        "fol": response.get("fol", ""),
                        "cot": self._join_list(response.get("cot")),
                        "premises": self._join_list(response.get("premises")),
                        "raw_response": json.dumps(response, ensure_ascii=False),
                    }
                )

        return output_path

    def generate_answers_into_original_files(
        self,
        upload_first: bool = True,
        reset_history: bool = True,
        reset_before_each_question: bool = True,
        skip_existing: bool = False,
    ) -> dict[str, Path]:
        """Generate answers via the API and save them back into each benchmark CSV.

        The original Q/A columns are preserved. Candidate answers are written
        to llm_answer for scoring, with generated_* columns mirroring the /ask
        response schema.
        """
        updated_paths: dict[str, Path] = {}

        for dataset in self.datasets:
            if reset_history:
                self.reset_session()
            if upload_first:
                self.upload_file_to_eda(dataset.data_path, wait=True)

            updated_paths[dataset.name] = self.generate_answers_into_original_file_for_dataset(
                dataset,
                reset_before_each_question=reset_before_each_question,
                skip_existing=skip_existing,
            )
        return updated_paths

    def generate_answers_into_original_file_for_dataset(
        self,
        dataset: DatasetSpec,
        reset_before_each_question: bool = True,
        skip_existing: bool = False,
    ) -> Path:
        benchmark_path = self._resolve_path(dataset.benchmark_path)
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

        rows, fieldnames = self._read_csv_rows_with_fieldnames(benchmark_path)
        self._validate_benchmark_columns(benchmark_path, fieldnames)

        for row in rows:
            if self._existing_generated_answer(row):
                continue
            if self._has_candidate_answer_column(row):
                row.update(self._format_failed_response())
                continue

            if reset_before_each_question:
                self.reset_session()
            response = self.ask(row["Q"])
            row.update(self._format_generated_response(response))

        output_fieldnames = list(fieldnames)
        for column in GENERATED_RESPONSE_COLUMNS:
            if column not in output_fieldnames:
                output_fieldnames.append(column)

        with benchmark_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return benchmark_path

    def ask(self, question: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.request_timeout) as client:
            response = client.post(f"{self.base_url}/ask", json={"question": question})
            response.raise_for_status()
            payload = response.json()
            graph_response = payload.get("response")
            if isinstance(graph_response, dict):
                return graph_response
            return payload

    def reset_session(self) -> None:
        with httpx.Client(timeout=self.request_timeout) as client:
            response = client.post(f"{self.base_url}/dev/reset")
            response.raise_for_status()

    def _poll_eda_result(self, client: httpx.Client, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.eda_max_wait

        while True:
            response = client.get(f"{self.base_url}/eda/result/{job_id}")
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")

            if status == "done":
                return payload
            if status and status != "pending":
                raise RuntimeError(f"EDA job {job_id} returned status {status!r}: {payload}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for EDA job {job_id} after {self.eda_max_wait} seconds."
                )

            time.sleep(self.eda_poll_interval)

    @staticmethod
    def _read_benchmark_rows(path: Path) -> list[dict[str, str]]:
        rows, fieldnames = TestBuilder._read_csv_rows_with_fieldnames(path)
        TestBuilder._validate_benchmark_columns(path, fieldnames)
        return rows

    @staticmethod
    def _read_csv_rows_with_fieldnames(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames

    @staticmethod
    def _validate_benchmark_columns(path: Path, fieldnames: Iterable[str]) -> None:
        missing_columns = {"Q", "A"} - set(fieldnames)
        if missing_columns:
            raise ValueError(f"{path} is missing benchmark columns: {sorted(missing_columns)}")

    @staticmethod
    def _format_generated_response(response: dict[str, Any]) -> dict[str, str]:
        answer = str(response.get("answer", ""))
        return {
            "llm_answer": answer,
            "generated_answer": answer,
            "generated_explanation": str(response.get("explanation", "")),
            "generated_confidence": "" if response.get("confidence") is None else str(response["confidence"]),
            "generated_fol": "" if response.get("fol") is None else str(response["fol"]),
            "generated_cot": TestBuilder._join_list(response.get("cot")),
            "generated_premises": TestBuilder._join_list(response.get("premises")),
            "generated_raw_response": json.dumps(response, ensure_ascii=False),
        }

    @staticmethod
    def _format_failed_response() -> dict[str, str]:
        return {
            "llm_answer": FAILED_ANSWER,
            "generated_answer": FAILED_ANSWER,
            "generated_explanation": "",
            "generated_confidence": "",
            "generated_fol": "",
            "generated_cot": "",
            "generated_premises": "",
            "generated_raw_response": json.dumps({"answer": FAILED_ANSWER}, ensure_ascii=False),
        }

    @staticmethod
    def _candidate_answer(row: dict[str, str]) -> str:
        existing_answer = TestBuilder._existing_generated_answer(row)
        if existing_answer:
            return existing_answer
        if TestBuilder._has_candidate_answer_column(row):
            return FAILED_ANSWER
        return ""

    @staticmethod
    def _existing_generated_answer(row: dict[str, str]) -> str:
        for column in ("llm_answer", "generated_answer"):
            answer = row.get(column, "").strip()
            if answer:
                return answer
        return ""

    @staticmethod
    def _has_candidate_answer_column(row: dict[str, str]) -> bool:
        return any(column in row for column in ("llm_answer", "generated_answer"))

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path

    @staticmethod
    def _mime_type(path: Path) -> str:
        return "text/csv" if path.suffix.lower() == ".csv" else "application/octet-stream"

    @staticmethod
    def _join_list(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return "" if value is None else str(value)


def upload_dataset_files_to_eda(
    base_url: str = "http://localhost:8000",
    wait: bool = True,
) -> dict[str, dict[str, Any]]:
    """Upload SuperStore, EmployeeAttrition, and MobileGameChurn CSVs to /eda/analyze."""
    return TestBuilder(base_url=base_url).upload_dataset_files_to_eda(wait=wait)


def generate_answers_into_original_files(
    base_url: str = "http://localhost:8000",
    upload_first: bool = True,
    skip_existing: bool = False,
) -> dict[str, Path]:
    """Generate API answers and append them to the original benchmark CSV files."""
    return TestBuilder(base_url=base_url).generate_answers_into_original_files(
        upload_first=upload_first,
        skip_existing=skip_existing,
    )
