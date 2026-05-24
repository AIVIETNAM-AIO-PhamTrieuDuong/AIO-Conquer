from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.evaluation.utils import DEFAULT_DATASETS, FAILED_ANSWER, REPO_ROOT, DatasetSpec


class HashEmbeddings(Embeddings):
    """Small deterministic embedding fallback for RAGAS response relevancy."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


@dataclass(frozen=True)
class RAGASTestCase:
    dataset: str
    row_number: int
    question: str
    reference: str
    candidate: str


class RAGAScore:
    """Calculate RAGAS metrics for generated benchmark answers."""

    REQUIRED_COLUMNS = {"Q", "A", "llm_answer"}
    METRIC_COLUMNS = ("response_relevancy", "answer_accuracy")

    def __init__(
        self,
        datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
        output_dir: str | Path = REPO_ROOT / "app" / "evaluation" / "results",
        batch_size: int = 1,
        experiment_name: str = "rag_chatbot_ragas",
        llm: Any | None = None,
        embeddings: Any | None = None,
        show_progress: bool = True,
    ) -> None:
        self.datasets = tuple(datasets)
        self.output_dir = self._resolve_path(output_dir)
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.llm = llm
        self.embeddings = embeddings
        self.show_progress = show_progress

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

    def load_test_cases(self) -> list[RAGASTestCase]:
        """Load benchmark rows after validating required columns are present."""
        self.validate_files()
        test_cases: list[RAGASTestCase] = []

        for dataset in self.datasets:
            rows, _ = self._read_csv(self._resolve_path(dataset.benchmark_path))
            for index, row in enumerate(rows, start=2):
                test_cases.append(
                    RAGASTestCase(
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
        """Calculate RAGAS metrics and save a JSON report."""
        test_cases = self.load_test_cases()
        if not test_cases:
            raise ValueError("No test cases found in benchmark CSV files.")

        try:
            self._install_langchain_vertexai_compat()
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset
            from ragas.metrics import AnswerAccuracy, ResponseRelevancy
        except ImportError as exc:
            raise ImportError(
                "Missing or incompatible RAGAS dependencies. Install them with "
                "`pip install ragas` or from requirements.txt."
            ) from exc

        metrics = [
            ResponseRelevancy(name="response_relevancy"),
            AnswerAccuracy(name="answer_accuracy"),
        ]
        llm = self._resolve_llm()
        embeddings = self.embeddings or HashEmbeddings()

        case_scores = []
        for case in test_cases:
            sample = {
                "user_input": case.question,
                "response": case.candidate,
                "reference": case.reference,
            }
            try:
                result = evaluate(
                    dataset=EvaluationDataset.from_list([sample]),
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings,
                    batch_size=self.batch_size,
                    show_progress=self.show_progress,
                    raise_exceptions=True,
                    experiment_name=self.experiment_name,
                )
            except Exception as exc:
                raise RuntimeError(
                    "RAGAS failed for "
                    f"dataset={case.dataset!r}, row_number={case.row_number}, "
                    f"candidate_length={len(case.candidate)}, "
                    f"reference_length={len(case.reference)}, "
                    f"candidate_preview={case.candidate[:500]!r}, "
                    f"reference_preview={case.reference[:500]!r}, "
                    f"cause={type(exc).__name__}: {exc}"
                ) from exc

            scores = result.scores[0]
            case_scores.append(
                {
                    "dataset": case.dataset,
                    "row_number": case.row_number,
                    "question": case.question,
                    "reference": case.reference,
                    "candidate": case.candidate,
                    "response_relevancy": self._as_float(scores.get("response_relevancy")),
                    "answer_accuracy": self._as_float(scores.get("answer_accuracy")),
                }
            )

        report = {
            "experiment_metadata": self._build_metadata(test_cases, extra_metadata),
            "metrics": {
                "name": "RAGAS",
                "metric_columns": list(self.METRIC_COLUMNS),
                "batch_size": self.batch_size,
                "library": "ragas",
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
        test_cases: list[RAGASTestCase],
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

    @classmethod
    def _aggregate_scores(cls, scores: list[dict[str, Any]]) -> dict[str, float | int]:
        aggregate: dict[str, float | int] = {"count": len(scores)}
        for metric in cls.METRIC_COLUMNS:
            aggregate[metric] = mean(score[metric] for score in scores)
        return aggregate

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
            return self.output_dir / f"ragas_{timestamp}.json"
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

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None:
            return float("nan")
        return float(value)

    def _resolve_llm(self) -> Any | None:
        if self.llm is not None:
            return self.llm
        self._load_dotenv_into_environment()
        if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY"):
            return None
        if not settings.ninerouter_key:
            raise ValueError(
                "RAGAS requires a judge LLM. Set ninerouter_key in .env, "
                "set OPENAI_API_KEY for RAGAS defaults, or pass an explicit llm to RAGAScore."
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.ninerouter_model,
            # api_key=settings.ninerouter_key,
            base_url=settings.ninerouter_url,
            temperature=0,
        )

    @staticmethod
    def _load_dotenv_into_environment() -> None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv(REPO_ROOT / ".env", override=False)

    @staticmethod
    def _install_langchain_vertexai_compat() -> None:
        """Provide the legacy VertexAI chat module path expected by RAGAS 0.4.x."""
        module_name = "langchain_community.chat_models.vertexai"
        if module_name in sys.modules:
            return

        try:
            __import__(module_name)
            return
        except ModuleNotFoundError:
            pass

        module = types.ModuleType(module_name)

        class ChatVertexAI:  # pragma: no cover - compatibility placeholder
            pass

        module.ChatVertexAI = ChatVertexAI # pyright: ignore[reportAttributeAccessIssue]
        sys.modules[module_name] = module


def calculate_ragas(
    output_path: str | Path | None = None,
    datasets: Iterable[DatasetSpec] = DEFAULT_DATASETS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for calculating RAGAS over prepared benchmark CSVs."""
    return RAGAScore(datasets=datasets, **kwargs).calculate(output_path=output_path)
