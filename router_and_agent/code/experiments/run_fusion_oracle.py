from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anti_contamination import assert_no_private_keys
from src.common import SearchOutput, iter_jsonl, load_questions, rrf_merge, write_jsonl
from src.config import load_config
from src.metrics import evaluate_prediction


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in iter_jsonl(path)}


def build_fusion_rows(
    *,
    predictions_by_method: Mapping[str, Mapping[str, dict[str, Any]]],
    question_ids: list[str],
    top_k: int,
    rrf_k0: int,
) -> list[dict[str, Any]]:
    methods = list(predictions_by_method)
    rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        outputs = [
            SearchOutput.from_dict(predictions_by_method[method][question_id])
            for method in methods
        ]
        fusion = rrf_merge(outputs, top_k=top_k, k0=rrf_k0, question_id=question_id)
        row = fusion.to_dict()
        row["method"] = "fusion"
        row["tool"] = "fusion"
        row["tool_calls"] = len(methods)
        row["tool_sequence"] = methods
        row["analysis_only"] = False
        assert_no_private_keys(row, context=f"fusion:{question_id}")
        rows.append(row)
    return rows


def oracle_score(question, prediction: Mapping[str, Any], *, k: int) -> tuple[float, float, float, float, float]:
    metric = evaluate_prediction(question, prediction, k=k)
    return (
        float(metric["complete_recall_at_k"]),
        float(metric["matched_gold_count"]),
        float(metric["evidence_recall_at_k"]),
        float(metric["mrr"]),
        -float(metric["latency_ms"]),
    )


def build_oracle_rows(
    *,
    questions_path: Path,
    predictions_by_method: Mapping[str, Mapping[str, dict[str, Any]]],
    question_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    questions = {question.question_id: question for question in load_questions(questions_path)}
    methods = list(predictions_by_method)
    rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        question = questions[question_id]
        best_method = max(
            methods,
            key=lambda method: oracle_score(question, predictions_by_method[method][question_id], k=top_k),
        )
        best_row = json.loads(json.dumps(predictions_by_method[best_method], ensure_ascii=False))[question_id]
        selected_tool_calls = int(best_row.get("tool_calls", 1))
        best_row["method"] = "oracle"
        best_row["tool"] = "oracle"
        best_row["tool_calls"] = selected_tool_calls
        best_row["selected_oracle_method"] = best_method
        best_row["candidate_methods"] = methods
        best_row["analysis_only"] = True
        best_row.setdefault("metadata", {})
        best_row["metadata"]["selected_oracle_method"] = best_method
        best_row["metadata"]["oracle_policy"] = "best_existing_output_by_complete_recall_then_mrr"
        for rank, doc in enumerate(best_row.get("ranked_documents", []), start=1):
            doc["rank"] = rank
            doc["tool"] = "oracle"
        assert_no_private_keys(best_row, context=f"oracle:{question_id}")
        rows.append(best_row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RRF fusion and analysis-only oracle outputs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="final")
    parser.add_argument("--methods", default="dense,file_fts,graph_path")
    parser.add_argument("--include-fusion-in-oracle", action="store_true")
    parser.add_argument("--fusion-output", default=None)
    parser.add_argument("--oracle-output", default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    questions_path = Path(config["paths"]["questions_path"])
    top_k = int(config["experiment"]["top_k"])
    rrf_k0 = int(config["experiment"]["rrf_k0"])
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method")

    predictions_by_method = {}
    for method in methods:
        path = results_dir / f"{method}_{args.split}_predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {method}: {path}")
        predictions_by_method[method] = load_predictions(path)

    question_ids = sorted(set.intersection(*(set(rows) for rows in predictions_by_method.values())))
    if not question_ids:
        raise ValueError("No shared question IDs found across prediction files.")

    fusion_rows = build_fusion_rows(
        predictions_by_method=predictions_by_method,
        question_ids=question_ids,
        top_k=top_k,
        rrf_k0=rrf_k0,
    )
    fusion_output = Path(args.fusion_output) if args.fusion_output else results_dir / f"fusion_{args.split}_predictions.jsonl"
    write_jsonl(fusion_output, fusion_rows)

    oracle_inputs = dict(predictions_by_method)
    if args.include_fusion_in_oracle:
        oracle_inputs["fusion"] = {str(row["question_id"]): row for row in fusion_rows}
    oracle_rows = build_oracle_rows(
        questions_path=questions_path,
        predictions_by_method=oracle_inputs,
        question_ids=question_ids,
        top_k=top_k,
    )
    oracle_output = Path(args.oracle_output) if args.oracle_output else results_dir / f"oracle_{args.split}_predictions.jsonl"
    write_jsonl(oracle_output, oracle_rows)

    print(
        json.dumps(
            {
                "fusion_output": str(fusion_output),
                "oracle_output": str(oracle_output),
                "question_count": len(question_ids),
                "fusion_methods": methods,
                "oracle_candidate_methods": list(oracle_inputs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
