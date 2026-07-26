from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .anti_contamination import assert_no_private_keys, validate_prediction_file
from .common import (
    CorpusDocument,
    PublicQuestion,
    load_public_corpus,
    load_public_questions,
    write_jsonl,
)
from .metrics import evaluate_predictions_file, parse_ks, write_csv


PredictionFn = Callable[[PublicQuestion], Mapping[str, Any]]


@dataclass(slots=True)
class BoxResult:
    method: str
    split: str
    prediction_path: Path
    metrics_path: Path | None = None
    summary_path: Path | None = None
    question_count: int = 0
    evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "split": self.split,
            "prediction_path": str(self.prediction_path),
            "metrics_path": None if self.metrics_path is None else str(self.metrics_path),
            "summary_path": None if self.summary_path is None else str(self.summary_path),
            "question_count": self.question_count,
            "evaluated": self.evaluated,
        }


class ExperimentBox:
    """Runtime/evaluation boundary for final retrieval experiments.

    Runtime methods receive only PublicQuestion and public corpus views.
    Full gold labels are opened only inside evaluate_predictions().
    """

    def __init__(self, *, config: Mapping[str, Any], split: str) -> None:
        self.config = config
        self.split = split
        self.results_dir = Path(str(config["paths"]["results_dir"]))
        self.indexes_dir = Path(str(config["paths"]["indexes_dir"]))
        self.questions_path = Path(str(config["paths"]["questions_path"]))
        self.corpus_path = Path(str(config["paths"]["corpus_path"]))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

    def runtime_questions(self) -> list[PublicQuestion]:
        return [
            question
            for question in load_public_questions(self.questions_path)
            if self.split == "all" or question.split == self.split
        ]

    def runtime_corpus(self) -> list[CorpusDocument]:
        return load_public_corpus(self.corpus_path)

    def prediction_path(self, method: str) -> Path:
        return self.results_dir / f"{method}_{self.split}_predictions.jsonl"

    def metrics_path(self, method: str) -> Path:
        return self.results_dir / f"{method}_{self.split}_metrics.csv"

    def summary_path(self, method: str) -> Path:
        return self.results_dir / f"{method}_{self.split}_summary.json"

    def run_method(
        self,
        *,
        method: str,
        predict: PredictionFn,
        output_path: str | Path | None = None,
        evaluate: bool = True,
        ks: str | None = None,
    ) -> BoxResult:
        questions = self.runtime_questions()
        rows = []
        for question in questions:
            row = dict(predict(question))
            row.setdefault("question_id", question.question_id)
            row.setdefault("method", method)
            assert_no_private_keys(row, context=f"{method}:{question.question_id}")
            rows.append(row)

        prediction_path = Path(output_path) if output_path else self.prediction_path(method)
        write_jsonl(prediction_path, rows)
        validate_prediction_file(prediction_path)

        result = BoxResult(
            method=method,
            split=self.split,
            prediction_path=prediction_path,
            question_count=len(rows),
        )
        if evaluate:
            result.metrics_path = self.metrics_path(method)
            result.summary_path = self.summary_path(method)
            self.evaluate_predictions(
                predictions_path=prediction_path,
                metrics_path=result.metrics_path,
                summary_path=result.summary_path,
                ks=ks,
            )
            result.evaluated = True
        return result

    def write_predictions(
        self,
        *,
        method: str,
        rows: Iterable[Mapping[str, Any]],
        output_path: str | Path | None = None,
        evaluate: bool = True,
        ks: str | None = None,
    ) -> BoxResult:
        materialized = [dict(row) for row in rows]
        for index, row in enumerate(materialized, start=1):
            assert_no_private_keys(row, context=f"{method}:row_{index}")
        prediction_path = Path(output_path) if output_path else self.prediction_path(method)
        write_jsonl(prediction_path, materialized)
        validate_prediction_file(prediction_path)

        result = BoxResult(
            method=method,
            split=self.split,
            prediction_path=prediction_path,
            question_count=len(materialized),
        )
        if evaluate:
            result.metrics_path = self.metrics_path(method)
            result.summary_path = self.summary_path(method)
            self.evaluate_predictions(
                predictions_path=prediction_path,
                metrics_path=result.metrics_path,
                summary_path=result.summary_path,
                ks=ks,
            )
            result.evaluated = True
        return result

    def evaluate_predictions(
        self,
        *,
        predictions_path: str | Path,
        metrics_path: str | Path,
        summary_path: str | Path,
        ks: str | None = None,
    ) -> None:
        validate_prediction_file(predictions_path)
        rows, summary = evaluate_predictions_file(
            questions_path=self.questions_path,
            predictions_path=predictions_path,
            ks=parse_ks(ks, None),
            bootstrap_samples=int(self.config["experiment"]["bootstrap_samples"]),
            bootstrap_confidence=float(self.config["experiment"]["bootstrap_confidence"]),
            seed=int(self.config["project"]["seed"]),
        )
        write_csv(metrics_path, rows)
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def validate_runtime_inputs(self) -> dict[str, int | str]:
        questions = self.runtime_questions()
        corpus = self.runtime_corpus()
        for question in questions:
            assert_no_private_keys(question.to_dict(), context=f"runtime_question:{question.question_id}")
        for index, doc in enumerate(corpus, start=1):
            assert_no_private_keys(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "sentences": doc.sentences,
                    "full_text": doc.full_text,
                    "metadata": doc.metadata,
                },
                context=f"runtime_doc:{index}",
            )
        return {
            "split": self.split,
            "runtime_questions": len(questions),
            "runtime_documents": len(corpus),
        }
