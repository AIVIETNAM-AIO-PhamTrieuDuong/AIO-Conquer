from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.evaluation.utils import DATA_ROOT, FAILED_ANSWER, REPO_ROOT


@dataclass(frozen=True)
class AnswerCotPairSpec:
    """Describe one actual-response log paired with one reference QA CSV."""

    name: str
    actual_path: Path
    reference_path: Path


DEFAULT_ANSWER_COT_PAIRS: tuple[AnswerCotPairSpec, ...] = (
    AnswerCotPairSpec(
        name="EmployeeAttrition",
        actual_path=DATA_ROOT
        / "EmployeeAttrition"
        / "ask_EmployeeAttrition_QA_Benchmark_20260628_180255.json",
        reference_path=DATA_ROOT
        / "EmployeeAttrition"
        / "EmployeeAttrition_QA_with_Premise.csv",
    ),
    AnswerCotPairSpec(
        name="MobileGameChurn",
        actual_path=DATA_ROOT
        / "MobileGameChurn"
        / "ask_MobileGameChurn_QA_Benchmark_20260628_174821.json",
        reference_path=DATA_ROOT
        / "MobileGameChurn"
        / "MobileGameChurn_QA_with_Premise.csv",
    ),
    AnswerCotPairSpec(
        name="SuperStore",
        actual_path=DATA_ROOT
        / "SuperStore"
        / "ask_SuperStore_QA_Benchmark_20260628_165847.json",
        reference_path=DATA_ROOT
        / "SuperStore"
        / "SuperStore_QA_with_Premise.csv",
    ),
)


@dataclass(frozen=True)
class AnswerCotTestCase:
    """Hold one aligned answer and CoT evaluation case."""

    dataset: str
    row_number: int
    index: int
    question: str
    reference_answer: str
    actual_answer: str
    reference_cot: str
    actual_cot: str
    actual_premises: str
    thread_id: str
    run_id: str


class AnswerCotScore:
    """Evaluate final answers with BERTScore and visible CoT with G-Eval."""

    REQUIRED_REFERENCE_COLUMNS = {"Q", "A", "premise"}

    def __init__(
        self,
        pairs: Iterable[AnswerCotPairSpec] = DEFAULT_ANSWER_COT_PAIRS,
        output_dir: str | Path = REPO_ROOT / "app" / "evaluation" / "results",
        bertscore_model_type: str = "distilbert-base-uncased",
        bertscore_lang: str = "en",
        bertscore_batch_size: int = 16,
        bertscore_rescale_with_baseline: bool = True,
        experiment_name: str = "rag_chatbot_answer_cot",
        answer_scorer: Any | None = None,
        cot_metric: Any | None = None,
    ) -> None:
        self.pairs = tuple(pairs)
        self.output_dir = self._resolve_path(output_dir)
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_lang = bertscore_lang
        self.bertscore_batch_size = bertscore_batch_size
        self.bertscore_rescale_with_baseline = bertscore_rescale_with_baseline
        self.experiment_name = experiment_name
        self.answer_scorer = answer_scorer
        self.cot_metric = cot_metric

    def validate_files(self) -> dict[str, dict[str, Any]]:
        """Validate configured reference and actual files."""
        validation: dict[str, dict[str, Any]] = {}

        for pair in self.pairs:
            reference_path = self._resolve_path(pair.reference_path)
            actual_path = self._resolve_path(pair.actual_path)
            reference_rows, fieldnames = self._read_csv(reference_path)
            actual_rows = self._read_actual_rows(actual_path)
            missing_columns = sorted(
                self.REQUIRED_REFERENCE_COLUMNS - set(fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    f"{reference_path} is missing required columns: "
                    f"{missing_columns}"
                )
            if len(reference_rows) != len(actual_rows):
                raise ValueError(
                    f"{pair.name} row count mismatch: reference has "
                    f"{len(reference_rows)}, actual has {len(actual_rows)}"
                )
            self._validate_row_alignment(pair.name, reference_rows, actual_rows)

            validation[pair.name] = {
                "reference_path": str(reference_path),
                "actual_path": str(actual_path),
                "row_count": len(reference_rows),
                "columns": fieldnames,
                "required_columns": sorted(self.REQUIRED_REFERENCE_COLUMNS),
            }

        return validation

    def load_test_cases(self) -> list[AnswerCotTestCase]:
        """Load row-aligned answer and CoT test cases."""
        self.validate_files()
        test_cases: list[AnswerCotTestCase] = []

        for pair in self.pairs:
            reference_rows, _ = self._read_csv(self._resolve_path(pair.reference_path))
            actual_rows = self._read_actual_rows(self._resolve_path(pair.actual_path))
            for actual in actual_rows:
                index = int(actual["index"])
                reference = reference_rows[index]
                response = actual.get("response_state", {}).get("response", {})
                actual_answer = str(
                    actual.get("answer") or response.get("answer") or FAILED_ANSWER
                )
                test_cases.append(
                    AnswerCotTestCase(
                        dataset=pair.name,
                        row_number=index + 2,
                        index=index,
                        question=reference["Q"].strip(),
                        reference_answer=reference["A"].strip(),
                        actual_answer=actual_answer.strip() or FAILED_ANSWER,
                        reference_cot=reference["premise"].strip(),
                        actual_cot=self._join_list(response.get("cot")).strip(),
                        actual_premises=self._join_list(
                            response.get("premises")
                        ).strip(),
                        thread_id=str(actual.get("thread_id", "")),
                        run_id=str(actual.get("run_id", "")),
                    )
                )

        return test_cases

    def calculate(
        self,
        output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Calculate answer and CoT scores and save a combined JSON report."""
        test_cases = self.load_test_cases()
        if not test_cases:
            raise ValueError("No test cases found in answer/CoT pair files.")

        answer_scores = self._score_answers(test_cases)
        cot_scores = self._score_cot(test_cases)
        case_scores = []
        for test_case, answer_score, cot_score in zip(
            test_cases,
            answer_scores,
            cot_scores,
            strict=True,
        ):
            case_scores.append(
                {
                    "dataset": test_case.dataset,
                    "row_number": test_case.row_number,
                    "index": test_case.index,
                    "question": test_case.question,
                    "reference_answer": test_case.reference_answer,
                    "actual_answer": test_case.actual_answer,
                    "reference_cot": test_case.reference_cot,
                    "actual_cot": test_case.actual_cot,
                    "actual_premises": test_case.actual_premises,
                    "thread_id": test_case.thread_id,
                    "run_id": test_case.run_id,
                    "answer_score": answer_score,
                    "cot_score": cot_score,
                }
            )

        report = {
            "experiment_metadata": self._build_metadata(test_cases, extra_metadata),
            "metrics": {
                "answer": {
                    "name": "BERTScore",
                    "model_type": self.bertscore_model_type,
                    "lang": self.bertscore_lang,
                    "batch_size": self.bertscore_batch_size,
                    "rescale_with_baseline": self.bertscore_rescale_with_baseline,
                    "library": "bert-score",
                },
                "cot": {
                    "name": "DeepEval G-Eval",
                    "missing_cot_policy": "score_zero",
                    "library": "deepeval",
                },
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

    def _score_answers(
        self,
        test_cases: list[AnswerCotTestCase],
    ) -> list[dict[str, float]]:
        """Calculate BERTScore values for final answers."""
        scorer = self.answer_scorer or self._build_answer_scorer()
        scores: list[dict[str, float]] = []

        for test_case in test_cases:
            try:
                precision, recall, f1 = scorer.score(
                    [test_case.actual_answer],
                    [test_case.reference_answer],
                    verbose=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "BERTScore failed for "
                    f"dataset={test_case.dataset!r}, "
                    f"row_number={test_case.row_number}, "
                    f"candidate_preview={test_case.actual_answer[:500]!r}, "
                    f"reference_preview={test_case.reference_answer[:500]!r}"
                ) from exc

            scores.append(
                {
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "f1": float(f1[0]),
                }
            )

        return scores

    def _score_cot(
        self,
        test_cases: list[AnswerCotTestCase],
    ) -> list[dict[str, Any]]:
        """Calculate DeepEval G-Eval scores for visible CoT text."""
        metric = self.cot_metric
        injected_metric = metric is not None
        scores: list[dict[str, Any]] = []

        for test_case in test_cases:
            if not test_case.actual_cot:
                scores.append(
                    {
                        "score": 0.0,
                        "reason": "missing_cot",
                        "evaluated": False,
                    }
                )
                continue

            if metric is None:
                metric = self._build_cot_metric()
            if injected_metric:
                deepeval_case = test_case
            else:
                deepeval_case = self._build_deepeval_case(test_case)
            try:
                metric.measure(deepeval_case)
            except Exception as exc:
                raise RuntimeError(
                    "DeepEval G-Eval failed for "
                    f"dataset={test_case.dataset!r}, "
                    f"row_number={test_case.row_number}, "
                    f"actual_cot_preview={test_case.actual_cot[:500]!r}, "
                    f"reference_cot_preview={test_case.reference_cot[:500]!r}"
                ) from exc

            scores.append(
                {
                    "score": float(getattr(metric, "score", 0.0)),
                    "reason": str(getattr(metric, "reason", "")),
                    "evaluated": True,
                }
            )

        return scores

    def _build_answer_scorer(self) -> Any:
        """Build the BERTScore scorer used by the existing answer evaluator."""
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'bert-score'. Install it with "
                "`pip install bert-score` or from requirements.txt."
            ) from exc

        return BERTScorer(
            model_type=self.bertscore_model_type,
            lang=self.bertscore_lang,
            batch_size=self.bertscore_batch_size,
            rescale_with_baseline=self.bertscore_rescale_with_baseline,
        )

    @staticmethod
    def _build_cot_metric() -> Any:
        """Build a DeepEval G-Eval metric for CoT quality."""
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCaseParams
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'deepeval'. Install it with "
                "`pip install deepeval` or from requirements.txt."
            ) from exc

        return GEval(
            name="CoT Quality",
            criteria=(
                "Evaluate whether the actual visible chain-of-thought matches "
                "the expected premise for the same data-analysis question. "
                "Reward factual alignment, coverage of required reasoning "
                "steps, logical consistency, correct use of numbers and "
                "columns, and concise relevant reasoning. Penalize unsupported "
                "values, irrelevant steps, contradictions, and conclusions that "
                "do not follow from the steps."
            ),
            evaluation_steps=[
                "Compare the actual CoT against the expected premise.",
                "Check whether required facts, numbers, and operations appear.",
                "Check whether the steps logically support the final answer.",
                "Penalize hallucinated or irrelevant reasoning.",
                "Return a score from 0 to 1 with a concise reason.",
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )

    @staticmethod
    def _build_deepeval_case(test_case: AnswerCotTestCase) -> Any:
        """Build one DeepEval test case without importing DeepEval at module load."""
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'deepeval'. Install it with "
                "`pip install deepeval` or from requirements.txt."
            ) from exc

        return LLMTestCase(
            input=test_case.question,
            actual_output=test_case.actual_cot,
            expected_output=test_case.reference_cot,
        )

    def _build_metadata(
        self,
        test_cases: list[AnswerCotTestCase],
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build report metadata for the combined evaluation run."""
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_test_cases": len(test_cases),
            "pairs": [
                {
                    "name": pair.name,
                    "actual_path": str(self._resolve_path(pair.actual_path)),
                    "reference_path": str(self._resolve_path(pair.reference_path)),
                }
                for pair in self.pairs
            ],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    @classmethod
    def _aggregate_scores(cls, scores: list[dict[str, Any]]) -> dict[str, float | int]:
        """Aggregate answer and CoT scores across a group of cases."""
        cot_scores = [score["cot_score"] for score in scores]
        return {
            "count": len(scores),
            "answer_precision": mean(
                score["answer_score"]["precision"] for score in scores
            ),
            "answer_recall": mean(score["answer_score"]["recall"] for score in scores),
            "answer_f1": mean(score["answer_score"]["f1"] for score in scores),
            "cot_score": mean(score["score"] for score in cot_scores),
            "cot_evaluated_count": sum(
                1 for score in cot_scores if score["evaluated"]
            ),
            "cot_missing_count": sum(
                1 for score in cot_scores if not score["evaluated"]
            ),
        }

    @classmethod
    def _aggregate_scores_by_dataset(
        cls,
        scores: list[dict[str, Any]],
    ) -> dict[str, dict[str, float | int]]:
        """Aggregate combined scores separately for each dataset."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for score in scores:
            grouped.setdefault(score["dataset"], []).append(score)
        return {
            dataset_name: cls._aggregate_scores(dataset_scores)
            for dataset_name, dataset_scores in grouped.items()
        }

    def _resolve_output_path(self, output_path: str | Path | None) -> Path:
        """Resolve a caller-provided report path or create a timestamped default."""
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            return self.output_dir / f"answer_cot_{timestamp}.json"
        return self._resolve_path(output_path)

    @classmethod
    def _validate_row_alignment(
        cls,
        dataset: str,
        reference_rows: list[dict[str, str]],
        actual_rows: list[dict[str, Any]],
    ) -> None:
        """Validate JSON indexes align to CSV row order and question text."""
        for position, actual in enumerate(actual_rows):
            index = actual.get("index")
            if not isinstance(index, int):
                raise ValueError(
                    f"{dataset} actual row {position} has invalid index: {index!r}"
                )
            if index < 0 or index >= len(reference_rows):
                raise ValueError(
                    f"{dataset} actual row {position} index out of range: {index}"
                )
            reference_question = reference_rows[index]["Q"].strip()
            actual_question = str(actual.get("question", "")).strip()
            if reference_question != actual_question:
                raise ValueError(
                    f"{dataset} question mismatch at index {index}: "
                    f"reference={reference_question!r}, actual={actual_question!r}"
                )

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        """Read a UTF-8 CSV file and return rows plus fieldnames."""
        if not path.exists():
            raise FileNotFoundError(f"Reference file not found: {path}")
        with path.open(newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        return rows, fieldnames

    @staticmethod
    def _read_actual_rows(path: Path) -> list[dict[str, Any]]:
        """Read response rows from a JSON report or JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"Actual response file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as file_obj:
                for line in file_obj:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows

        with path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload["results"]
        raise ValueError(f"Unsupported actual response JSON shape: {path}")

    @staticmethod
    def _join_list(value: Any) -> str:
        """Normalize list or scalar response fields into display text."""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return "" if value is None else str(value)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """Resolve repository-relative paths to absolute paths."""
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path


def calculate_answer_cot(
    output_path: str | Path | None = None,
    pairs: Iterable[AnswerCotPairSpec] = DEFAULT_ANSWER_COT_PAIRS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for calculating answer and CoT metrics."""
    return AnswerCotScore(pairs=pairs, **kwargs).calculate(output_path=output_path)
