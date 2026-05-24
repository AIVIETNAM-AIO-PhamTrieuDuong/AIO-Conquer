from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from app.evaluation.bertscore import BertScore
from app.evaluation.ragas_score import RAGAScore
from app.evaluation.utils import DEFAULT_DATASETS, REPO_ROOT, DatasetSpec, TestBuilder

ScoreMetric = Literal["bertscore", "ragas"]


@dataclass(frozen=True)
class RunnerResult:
    prepared_files: dict[str, Path]
    metric: str
    score_report: dict[str, Any]
    score_output_path: Path

    @property
    def bertscore_report(self) -> dict[str, Any]:
        return self.score_report

    @property
    def bertscore_output_path(self) -> Path:
        return self.score_output_path


class Runner:
    """Run benchmark answer preparation and score evaluation."""

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
        ragas_batch_size: int = 1,
        ragas_experiment_name: str = "rag_chatbot_ragas",
        ragas_show_progress: bool = True,
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
        self.ragas = RAGAScore(
            datasets=self.datasets,
            output_dir=self.output_dir,
            batch_size=ragas_batch_size,
            experiment_name=ragas_experiment_name,
            show_progress=ragas_show_progress,
        )

    def run(
        self,
        prepare: bool = True,
        upload_first: bool = True,
        reset_history: bool = True,
        reset_before_each_question: bool = True,
        skip_existing: bool = False,
        metric: ScoreMetric = "bertscore",
        output_path: str | Path | None = None,
        bertscore_output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RunnerResult:
        """Prepare generated answers, then calculate the selected score metric."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        prepared_files: dict[str, Path] = {}
        if prepare:
            prepared_files = self.test_builder.generate_answers_into_original_files(
                upload_first=upload_first,
                reset_history=reset_history,
                reset_before_each_question=reset_before_each_question,
                skip_existing=skip_existing,
            )

        resolved_output_path = self._resolve_score_output_path(
            metric,
            output_path if output_path is not None else bertscore_output_path,
        )
        report = self._score_with_metric(
            metric,
            output_path=resolved_output_path,
            extra_metadata={
                "metric": metric,
                "prepared_files": {name: str(path) for name, path in prepared_files.items()},
                **(extra_metadata or {}),
            },
        )

        return RunnerResult(
            prepared_files=prepared_files,
            metric=metric,
            score_report=report,
            score_output_path=resolved_output_path,
        )

    def prepare(self, **kwargs: Any) -> dict[str, Path]:
        """Only generate API answers into the benchmark CSV files."""
        return self.test_builder.generate_answers_into_original_files(**kwargs)

    def score(
        self,
        metric: ScoreMetric = "bertscore",
        output_path: str | Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Only calculate the selected score metric for prepared benchmark CSV files."""
        return self._score_with_metric(
            metric,
            output_path=self._resolve_score_output_path(metric, output_path),
            extra_metadata=extra_metadata,
        )

    def _score_with_metric(
        self,
        metric: ScoreMetric,
        output_path: Path,
        extra_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if metric == "bertscore":
            return self.bertscore.calculate(output_path=output_path, extra_metadata=extra_metadata)
        if metric == "ragas":
            return self.ragas.calculate(output_path=output_path, extra_metadata=extra_metadata)
        raise ValueError(f"Unsupported metric: {metric}")

    def _resolve_score_output_path(
        self,
        metric: ScoreMetric,
        output_path: str | Path | None,
    ) -> Path:
        if output_path is None:
            return self.output_dir / f"{metric}_report.json"
        return self._resolve_path(output_path)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else REPO_ROOT / path


def run_evaluation(
    base_url: str = "http://localhost:8000",
    prepare: bool = True,
    skip_existing: bool = False,
    metric: ScoreMetric = "bertscore",
    output_path: str | Path | None = None,
    bertscore_output_path: str | Path | None = None,
) -> RunnerResult:
    """Convenience wrapper for the full evaluation workflow."""
    return Runner(base_url=base_url).run(
        prepare=prepare,
        skip_existing=skip_existing,
        metric=metric,
        output_path=output_path,
        bertscore_output_path=bertscore_output_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG benchmark preparation and scoring.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--metric",
        choices=("bertscore", "ragas"),
        default="bertscore",
        help="Scoring metric to calculate.",
    )
    parser.add_argument("--output", default=None, help="Path for the JSON score report.")
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
        metric=args.metric,
        output_path=args.output,
    )
    print(f"{result.metric} report: {result.score_output_path}")
    print(_format_overall_score(result.score_report["score"]["overall"]))


def _format_overall_score(overall: dict[str, Any]) -> str:
    metric_values = [
        f"{name}={value:.4f}"
        for name, value in overall.items()
        if name != "count" and isinstance(value, int | float)
    ]
    return f"Overall: count={overall['count']}, " + ", ".join(metric_values)


if __name__ == "__main__":
    main()
