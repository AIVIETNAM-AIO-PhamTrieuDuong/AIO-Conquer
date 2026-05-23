from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.evaluation.bertscore import BertScore
from app.evaluation.utils import DEFAULT_DATASETS, REPO_ROOT, DatasetSpec, TestBuilder


@dataclass(frozen=True)
class RunnerResult:
    prepared_files: dict[str, Path]
    bertscore_report: dict[str, Any]
    bertscore_output_path: Path


class Runner:
    """Run benchmark answer preparation and BERTScore evaluation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
        output_dir: str | Path = REPO_ROOT / "app" / "evaluation" / "results",
        request_timeout: float = 120.0,
        eda_poll_interval: float = 2.0,
        eda_max_wait: float = 300.0,
        bertscore_model_type: str = "distilbert-base-uncased",
        bertscore_lang: str = "en",
        bertscore_batch_size: int = 16,
        bertscore_rescale_with_baseline: bool = True,
        experiment_name: str = "rag_chatbot_bertscore",
    ) -> None:
        self.datasets = tuple(datasets)
        self.output_dir = self._resolve_path(output_dir)
        self.test_builder = TestBuilder(
            base_url=base_url,
            datasets=self.datasets,
            output_dir=self.output_dir,
            request_timeout=request_timeout,
            eda_poll_interval=eda_poll_interval,
            eda_max_wait=eda_max_wait,
        )
        self.bertscore = BertScore(
            datasets=self.datasets,
            output_dir=self.output_dir,
            model_type=bertscore_model_type,
            lang=bertscore_lang,
            batch_size=bertscore_batch_size,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            experiment_name=experiment_name,
        )

    def run(
        self,
        prepare: bool = True,
        upload_first: bool = True,
        reset_history: bool = True,
        reset_before_each_question: bool = True,
        skip_existing: bool = False,
        bertscore_output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RunnerResult:
        """Prepare generated answers, then calculate BERTScore."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prepared_files: dict[str, Path] = {}
        if prepare:
            prepared_files = self.test_builder.generate_answers_into_original_files(
                upload_first=upload_first,
                reset_history=reset_history,
                reset_before_each_question=reset_before_each_question,
                skip_existing=skip_existing,
            )

        output_path = self._resolve_bertscore_output_path(bertscore_output_path)
        report = self.bertscore.calculate(
            output_path=output_path,
            extra_metadata={
                "prepared_files": {name: str(path) for name, path in prepared_files.items()},
                **(extra_metadata or {}),
            },
        )

        return RunnerResult(
            prepared_files=prepared_files,
            bertscore_report=report,
            bertscore_output_path=output_path,
        )

    def prepare(self, **kwargs: Any) -> dict[str, Path]:
        """Only generate API answers into the benchmark CSV files."""
        return self.test_builder.generate_answers_into_original_files(**kwargs)

    def score(
        self,
        output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Only calculate BERTScore for already prepared benchmark CSV files."""
        return self.bertscore.calculate(
            output_path=self._resolve_bertscore_output_path(output_path),
            extra_metadata=extra_metadata,
        )

    def _resolve_bertscore_output_path(self, output_path: str | Path | None) -> Path:
        if output_path is None:
            return self.output_dir / "bertscore_report.json"
        return self._resolve_path(output_path)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path


def run_evaluation(
    base_url: str = "http://localhost:8000",
    prepare: bool = True,
    skip_existing: bool = False,
    bertscore_output_path: str | Path | None = None,
) -> RunnerResult:
    """Convenience wrapper for the full evaluation workflow."""
    return Runner(base_url=base_url).run(
        prepare=prepare,
        skip_existing=skip_existing,
        bertscore_output_path=bertscore_output_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG benchmark preparation and BERTScore.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", default=None, help="Path for the BERTScore JSON report.")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip answer generation and score the already prepared CSV files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not regenerate rows that already have generated_answer. Rows with llm_answer are always skipped.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload datasets before generating answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = Runner(base_url=args.base_url).run(
        prepare=not args.score_only,
        upload_first=not args.no_upload,
        skip_existing=args.skip_existing,
        bertscore_output_path=args.output,
    )
    overall = result.bertscore_report["score"]["overall"]
    print(f"BERTScore report: {result.bertscore_output_path}")
    print(
        "Overall: "
        f"P={overall['precision']:.4f}, "
        f"R={overall['recall']:.4f}, "
        f"F1={overall['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
