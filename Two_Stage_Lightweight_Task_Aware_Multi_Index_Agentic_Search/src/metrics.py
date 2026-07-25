from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .anti_contamination import assert_no_private_keys
from .common import Question, iter_jsonl, load_questions


MetricRow = dict[str, Any]


def doc_ids_from_ranked_documents(
    ranked_documents: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in ranked_documents[:k]:
        doc_id = str(row["doc_id"])
        if doc_id not in seen:
            ids.append(doc_id)
            seen.add(doc_id)
    return ids


def doc_ids_from_any(value: Any, *, k: int | None = None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        ids = []
        for item in value[:k]:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, Mapping) and "doc_id" in item:
                ids.append(str(item["doc_id"]))
        return ids
    return []


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_ids: set[str]) -> float:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / index
    return 0.0


def infer_tool_calls(prediction: Mapping[str, Any]) -> int:
    if prediction.get("tool_calls") is not None:
        return int(prediction["tool_calls"])
    for key in ("tool_sequence", "tools", "used_tools"):
        if isinstance(prediction.get(key), list):
            return len(prediction[key])
    if isinstance(prediction.get("rounds"), list):
        return len(prediction["rounds"])
    return 1


def infer_first_round_doc_ids(prediction: Mapping[str, Any], *, k: int) -> list[str]:
    for key in ("first_round_documents", "first_round_doc_ids"):
        ids = doc_ids_from_any(prediction.get(key), k=k)
        if ids:
            return ids

    metadata = prediction.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("first_round_documents", "first_round_doc_ids"):
            ids = doc_ids_from_any(metadata.get(key), k=k)
            if ids:
                return ids

    rounds = prediction.get("rounds")
    if isinstance(rounds, list) and rounds:
        first = rounds[0]
        if isinstance(first, Mapping):
            return doc_ids_from_any(first.get("ranked_documents"), k=k)
    return []


def infer_stop_accuracy(
    prediction: Mapping[str, Any],
    *,
    final_complete: bool,
    first_round_complete: bool,
) -> float | None:
    judge_decision = prediction.get("judge_decision")
    if not isinstance(judge_decision, Mapping):
        return None
    stopped = bool(judge_decision.get("sufficient"))
    should_stop = first_round_complete
    if should_stop:
        return 1.0 if stopped and final_complete else 0.0
    return 1.0 if not stopped else 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_prediction(
    question: Question,
    prediction: Mapping[str, Any],
    *,
    k: int = 5,
) -> MetricRow:
    ranked_documents = prediction.get("ranked_documents", [])
    if not isinstance(ranked_documents, list):
        raise ValueError(f"ranked_documents must be a list for {question.question_id}")

    retrieved_doc_ids = doc_ids_from_ranked_documents(ranked_documents, k=k)
    gold_doc_ids = set(question.gold_documents)
    matched_gold = gold_doc_ids.intersection(retrieved_doc_ids)
    first_round_doc_ids = infer_first_round_doc_ids(prediction, k=k)

    evidence_recall = 1.0 if matched_gold else 0.0
    complete_recall = 1.0 if gold_doc_ids and gold_doc_ids.issubset(retrieved_doc_ids) else 0.0
    first_round_gold = gold_doc_ids.intersection(first_round_doc_ids)
    first_round_complete = bool(gold_doc_ids) and gold_doc_ids.issubset(first_round_doc_ids)
    evidence_gain_step2 = (
        len(matched_gold) - len(first_round_gold) if first_round_doc_ids else None
    )
    stop_accuracy = infer_stop_accuracy(
        prediction,
        final_complete=bool(complete_recall),
        first_round_complete=first_round_complete,
    )

    return {
        "question_id": question.question_id,
        "task_type": question.task_type,
        "split": question.split,
        "method": str(prediction.get("method") or prediction.get("tool") or "unknown"),
        "k": k,
        "gold_count": len(gold_doc_ids),
        "retrieved_count": len(retrieved_doc_ids),
        "matched_gold_count": len(matched_gold),
        "evidence_recall_at_k": evidence_recall,
        "complete_recall_at_k": complete_recall,
        "search_success": complete_recall,
        "mrr": reciprocal_rank(retrieved_doc_ids, gold_doc_ids),
        "tool_calls": infer_tool_calls(prediction),
        "latency_ms": float(prediction.get("latency_ms", 0.0)),
        "text_read_tokens": int(prediction.get("text_read_tokens", 0)),
        "evidence_gain_step2": evidence_gain_step2,
        "stop_accuracy": stop_accuracy,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | None]:
    if not values:
        return {"low": None, "high": None}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(draw))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(alpha * samples))
    high_index = min(samples - 1, int((1.0 - alpha) * samples) - 1)
    return {"low": means[low_index], "high": means[high_index]}


def summarize_rows(
    rows: Sequence[MetricRow],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    if not rows:
        return {"count": 0}

    metric_keys = [
        "evidence_recall_at_k",
        "complete_recall_at_k",
        "search_success",
        "mrr",
        "tool_calls",
        "latency_ms",
        "latency_ms_p95",
        "text_read_tokens",
        "evidence_gain_step2",
        "stop_accuracy",
    ]
    summary: dict[str, Any] = {"count": len(rows)}
    for key in metric_keys:
        if key == "latency_ms_p95":
            latency_values = [
                float(row["latency_ms"])
                for row in rows
                if row.get("latency_ms") is not None
            ]
            summary[key] = percentile(latency_values, 0.95)
            continue
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
        ]
        if not values:
            summary[key] = None
            continue
        summary[key] = mean(values)
        if key in {"evidence_recall_at_k", "complete_recall_at_k", "search_success"}:
            summary[f"{key}_ci95"] = bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=seed,
            )
    return summary


def summarize_by_group(
    rows: Sequence[MetricRow],
    group_key: str,
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[MetricRow]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    return {
        key: summarize_rows(
            group_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            seed=seed,
        )
        for key, group_rows in sorted(groups.items())
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictions_file(
    *,
    questions_path: str | Path,
    predictions_path: str | Path,
    ks: Sequence[int],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    seed: int,
) -> tuple[list[MetricRow], dict[str, Any]]:
    questions = {q.question_id: q for q in load_questions(questions_path)}
    rows: list[MetricRow] = []
    missing_question_ids: list[str] = []

    for prediction in iter_jsonl(predictions_path):
        question_id = str(prediction.get("question_id", ""))
        assert_no_private_keys(prediction, context=f"prediction:{question_id}")
        question = questions.get(question_id)
        if question is None:
            missing_question_ids.append(question_id)
            continue
        for k in ks:
            rows.append(evaluate_prediction(question, prediction, k=k))

    primary_k = max(ks)
    primary_rows = [row for row in rows if row["k"] == primary_k]
    summary = {
        "primary_k": primary_k,
        "overall": summarize_rows(
            primary_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            seed=seed,
        ),
        "by_task_type": summarize_by_group(
            primary_rows,
            "task_type",
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            seed=seed,
        ),
        "by_k": {
            str(k): summarize_rows(
                [row for row in rows if row["k"] == k],
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                seed=seed,
            )
            for k in ks
        },
        "by_task_type_by_k": {
            str(k): summarize_by_group(
                [row for row in rows if row["k"] == k],
                "task_type",
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence=bootstrap_confidence,
                seed=seed,
            )
            for k in ks
        },
        "missing_question_ids": missing_question_ids,
    }
    return rows, summary


def parse_ks(raw_value: str | None, fallback_k: int | None) -> list[int]:
    if raw_value:
        values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    elif fallback_k is not None:
        values = [fallback_k]
    else:
        values = [1, 3, 5]
    values = sorted(set(values))
    if not values or any(value <= 0 for value in values):
        raise ValueError("--ks/--k must contain positive integers")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions.")
    parser.add_argument("--questions", required=True, help="Path to data/questions.jsonl")
    parser.add_argument("--predictions", required=True, help="Path to prediction JSONL")
    parser.add_argument("--out", required=True, help="Output per-question CSV path")
    parser.add_argument(
        "--summary-out",
        required=True,
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Single evaluation cutoff. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated evaluation cutoffs, for example: 1,3,5. Defaults to 1,3,5.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ks = parse_ks(args.ks, args.k)
    rows, summary = evaluate_predictions_file(
        questions_path=args.questions,
        predictions_path=args.predictions,
        ks=ks,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        seed=args.seed,
    )
    write_csv(args.out, rows)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if len(ks) == 1:
        print(json.dumps(summary["overall"], ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary["by_k"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
