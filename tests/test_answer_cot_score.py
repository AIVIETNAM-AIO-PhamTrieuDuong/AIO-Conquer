from __future__ import annotations

import csv
import json
import unittest
import uuid
from pathlib import Path
from typing import Any

from app.evaluation.answer_cot_score import AnswerCotPairSpec, AnswerCotScore


class FakeAnswerScorer:
    """Provide deterministic BERTScore-shaped outputs for tests."""

    def score(
        self,
        candidates: list[str],
        references: list[str],
        verbose: bool = False,
    ) -> tuple[list[float], list[float], list[float]]:
        """Return fixed precision, recall, and F1 values."""
        return [0.5], [0.6], [0.7]


class FakeCotMetric:
    """Provide deterministic DeepEval-shaped CoT scores for tests."""

    def __init__(self) -> None:
        """Initialize the fake metric result fields."""
        self.score = 0.8
        self.reason = "aligned"
        self.calls = 0

    def measure(self, test_case: Any) -> None:
        """Record that one non-empty CoT case was judged."""
        self.calls += 1


class AnswerCotScoreTests(unittest.TestCase):
    """Verify answer and CoT evaluation over JSON/CSV pair fixtures."""

    def test_calculate_scores_answers_and_cot(self) -> None:
        """Score answers, non-empty CoT, and missing CoT in one report."""
        reference_path, actual_path, output_path = self._case_paths()
        try:
            self._write_reference(reference_path)
            self._write_actual(actual_path)

            cot_metric = FakeCotMetric()
            scorer = AnswerCotScore(
                pairs=(
                    AnswerCotPairSpec(
                        name="Fixture",
                        actual_path=actual_path,
                        reference_path=reference_path,
                    ),
                ),
                output_dir=output_path.parent,
                answer_scorer=FakeAnswerScorer(),
                cot_metric=cot_metric,
            )
            report = scorer.calculate(output_path=output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(report["score"]["overall"]["count"], 2)
            self.assertEqual(report["score"]["overall"]["cot_evaluated_count"], 1)
            self.assertEqual(report["score"]["overall"]["cot_missing_count"], 1)
            self.assertAlmostEqual(report["score"]["overall"]["answer_f1"], 0.7)
            self.assertAlmostEqual(report["score"]["overall"]["cot_score"], 0.4)
            self.assertEqual(report["cases"][0]["cot_score"]["reason"], "missing_cot")
            self.assertEqual(report["cases"][1]["cot_score"]["reason"], "aligned")
            self.assertEqual(cot_metric.calls, 1)
        finally:
            self._cleanup_paths(reference_path, actual_path, output_path)

    def test_validate_rejects_question_mismatch(self) -> None:
        """Reject JSON rows whose index points at a different CSV question."""
        reference_path, actual_path, output_path = self._case_paths()
        try:
            self._write_reference(reference_path)
            self._write_actual(actual_path, second_question="Wrong question")

            scorer = AnswerCotScore(
                pairs=(
                    AnswerCotPairSpec(
                        name="Fixture",
                        actual_path=actual_path,
                        reference_path=reference_path,
                    ),
                ),
                output_dir=output_path.parent,
                answer_scorer=FakeAnswerScorer(),
                cot_metric=FakeCotMetric(),
            )

            with self.assertRaisesRegex(ValueError, "question mismatch"):
                scorer.validate_files()
        finally:
            self._cleanup_paths(reference_path, actual_path, output_path)

    @staticmethod
    def _case_paths() -> tuple[Path, Path, Path]:
        """Return unique transient paths in an existing workspace directory."""
        root = Path.cwd() / "app" / "evaluation" / "results"
        suffix = uuid.uuid4().hex
        return (
            root / f"test_answer_cot_{suffix}.csv",
            root / f"test_answer_cot_{suffix}.json",
            root / f"test_answer_cot_{suffix}_report.json",
        )

    @staticmethod
    def _cleanup_paths(*paths: Path) -> None:
        """Remove transient test files created by this test case."""
        for path in paths:
            if path.exists():
                path.unlink()

    @staticmethod
    def _write_reference(path: Path) -> None:
        """Write a two-row reference CSV with answer and premise fields."""
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["Q", "A", "premise"])
            writer.writeheader()
            writer.writerow(
                {
                    "Q": "How many rows?",
                    "A": "There are 2 rows.",
                    "premise": "Step 1: Count rows. Step 2: Total rows = 2.",
                }
            )
            writer.writerow(
                {
                    "Q": "What is the rate?",
                    "A": "The rate is 50%.",
                    "premise": "Step 1: Count positives. Step 2: Divide by total.",
                }
            )

    @staticmethod
    def _write_actual(path: Path, second_question: str = "What is the rate?") -> None:
        """Write a two-row actual response JSON report."""
        payload = {
            "results": [
                {
                    "index": 0,
                    "question": "How many rows?",
                    "answer": "There are two rows.",
                    "thread_id": "thread-1",
                    "run_id": "run-1",
                    "response_state": {"response": {"cot": None}},
                },
                {
                    "index": 1,
                    "question": second_question,
                    "answer": "The rate is one half.",
                    "thread_id": "thread-2",
                    "run_id": "run-2",
                    "response_state": {
                        "response": {
                            "cot": [
                                "Step 1: Count positives.",
                                "Step 2: Divide by total.",
                            ],
                            "premises": ["target"],
                        }
                    },
                },
            ]
        }
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj)


if __name__ == "__main__":
    unittest.main()
