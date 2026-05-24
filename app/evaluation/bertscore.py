from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.evaluation.utils import DEFAULT_DATASETS, FAILED_ANSWER, REPO_ROOT, DatasetSpec


@dataclass(frozen=True)
class BertScoreTestCase:
    dataset: str
    row_number: int
    question: str
    reference: str
    candidate: str


class BertScore:
    """Calculate BERTScore for generated benchmark answers."""

    REQUIRED_COLUMNS = {"Q", "A", "llm_answer"}

    def __init__(
        self,
        datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
        output_dir: str | Path = REPO_ROOT / "app" / "evaluation" / "results",
        model_type: str = "distilbert-base-uncased",
        lang: str = "en",
        batch_size: int = 16,
        rescale_with_baseline: bool = True,
        experiment_name: str = "rag_chatbot_bertscore",
    ) -> None:
        self.datasets = tuple(datasets)
        self.output_dir = self._resolve_path(output_dir)
        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline
        self.experiment_name = experiment_name

    def validate_files(self) -> dict[str, dict[str, Any]]:
        """Validate all benchmark CSVs and return per-file validation details."""
        validation: dict[str, dict[str, Any]] = {}

        for dataset in self.datasets:
            path = self._resolve_path(dataset.benchmark_path)
            if not path.exists():
                raise FileNotFoundError(f"Benchmark file not found: {path}")

            rows, fieldnames = self._read_csv(path)
            missing_columns = sorted(self.REQUIRED_COLUMNS - set(fieldnames))
            if missing_columns:
                raise ValueError(f"{path} is missing required columns: {missing_columns}")

            validation[dataset.name] = {
                "path": str(path),
                "row_count": len(rows),
                "columns": fieldnames,
                "required_columns": sorted(self.REQUIRED_COLUMNS),
            }

        return validation

    def load_test_cases(self) -> list[BertScoreTestCase]:
        """Load benchmark rows after validating required columns are present."""
        self.validate_files()
        test_cases: list[BertScoreTestCase] = []

        for dataset in self.datasets:
            rows, _ = self._read_csv(self._resolve_path(dataset.benchmark_path))
            for index, row in enumerate(rows, start=2):
                test_cases.append(
                    BertScoreTestCase(
                        dataset=dataset.name,
                        row_number=index,
                        question=row["Q"].strip(),
                        reference=row["A"].strip(),
                        candidate=row["llm_answer"].strip() or FAILED_ANSWER,
                    )
                )

        return test_cases

    def calculate(
        self,
        output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate BERTScore and save a JSON report."""
        test_cases = self.load_test_cases()
        if not test_cases:
            raise ValueError("No test cases found in benchmark CSV files.")

        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'bert-score'. Install it with "
                "`pip install bert-score` or from requirements.txt."
            ) from exc
        scorer = BERTScorer(
            model_type=self.model_type, 
            lang=self.lang,
            batch_size=1,
            rescale_with_baseline=self.rescale_with_baseline
        )
        case_scores = []
        for case in test_cases:
            try:
                precision, recall, f1 = scorer.score(
                    [case.candidate],
                    [case.reference],
                    verbose=True
                )
            except Exception as exc:
                raise RuntimeError(
                    "BERTScore failed for "
                    f"dataset={case.dataset!r}, row_number={case.row_number}, "
                    f"candidate_length={len(case.candidate)}, "
                    f"reference_length={len(case.reference)}, "
                    f"candidate_preview={case.candidate[:500]!r}, "
                    f"reference_preview={case.reference[:500]!r}"
                ) from exc

            case_scores.append(
                {
                    "dataset": case.dataset,
                    "row_number": case.row_number,
                    "question": case.question,
                    "reference": case.reference,
                    "candidate": case.candidate,
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                }
            )

        report = {
            "experiment_metadata": self._build_metadata(test_cases, extra_metadata),
            "metrics": {
                "name": "BERTScore",
                "model_type": self.model_type,
                "lang": self.lang,
                "batch_size": self.batch_size,
                "rescale_with_baseline": self.rescale_with_baseline,
                "library": "bert-score",
            },
            "score": {
                "overall": self._aggregate_scores(case_scores),
                "by_dataset": self._aggregate_scores_by_dataset(case_scores),
            },
            "cases": case_scores,
        }

        output_file = self._resolve_output_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)

        return report

    def _build_metadata(
        self,
        test_cases: list[BertScoreTestCase],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_test_cases": len(test_cases),
            "datasets": [
                {
                    "name": dataset.name,
                    "benchmark_path": str(self._resolve_path(dataset.benchmark_path)),
                }
                for dataset in self.datasets
            ],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    @staticmethod
    def _aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(scores),
            "precision": mean(score["precision"] for score in scores),
            "recall": mean(score["recall"] for score in scores),
            "f1": mean(score["f1"] for score in scores),
        }

    def _aggregate_scores_by_dataset(
        self,
        scores: list[dict[str, Any]],
    ) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for score in scores:
            grouped.setdefault(score["dataset"], []).append(score)
        return {
            dataset_name: self._aggregate_scores(dataset_scores)
            for dataset_name, dataset_scores in grouped.items()
        }

    def _resolve_output_path(self, output_path: str | Path | None) -> Path:
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            return self.output_dir / f"bertscore_{timestamp}.json"
        return self._resolve_path(output_path)

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path


def calculate_bertscore(
    output_path: str | Path | None = None,
    datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for calculating BERTScore over prepared benchmark CSVs."""
    return BertScore(datasets=datasets, **kwargs).calculate(output_path=output_path)
