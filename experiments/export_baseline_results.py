from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import iter_jsonl, load_corpus, load_questions
from src.config import load_config


FORMAL_METHODS = {"dense", "file_fts", "graph_path", "fusion"}
ANALYSIS_ONLY_METHODS = {"oracle"}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(path: str | Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def float_or_blank(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    return float(value)


def number_or_na(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def percent_or_na(value: Any) -> str:
    if value == "" or value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def normalized_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    method = str(row["method"])
    return {
        "method": method,
        "method_role": "analysis_only" if method in ANALYSIS_ONLY_METHODS else "formal_baseline",
        "split": row.get("split", ""),
        "k": row.get("k", ""),
        "group_type": row.get("group_type", ""),
        "task_type": "" if row.get("group_name") == "overall" else row.get("group_name", ""),
        "count": row.get("count", ""),
        "evidence_recall": float_or_blank(row.get("evidence_recall_at_k")),
        "complete_evidence_recall": float_or_blank(row.get("complete_recall_at_k")),
        "mrr": float_or_blank(row.get("mrr")),
        "search_success_rate": float_or_blank(row.get("search_success")),
        "average_tool_calls": float_or_blank(row.get("tool_calls")),
        "latency_avg_ms": float_or_blank(row.get("latency_ms")),
        "latency_p95_ms": float_or_blank(row.get("latency_ms_p95")),
        "evidence_gain_per_step": float_or_blank(row.get("evidence_gain_step2")),
        "stop_accuracy": float_or_blank(row.get("stop_accuracy")),
    }


def result_method_name(path: Path) -> str | None:
    for suffix in (
        "_final_predictions.jsonl",
        "_final_metrics.csv",
        "_final_summary.json",
    ):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def export_artifact_name(path: Path) -> str:
    if path.name.endswith("_final_predictions.jsonl"):
        return path.name.replace("_final_predictions.jsonl", "_final_retrieval_output.jsonl")
    return path.name


def copy_result_artifacts(results_dir: Path, out_dir: Path) -> None:
    artifact_dir = out_dir / "retrieval_outputs_and_metrics"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*")):
        if not path.is_file():
            continue
        method = result_method_name(path)
        if path.name != "main_results.csv" and method not in FORMAL_METHODS | ANALYSIS_ONLY_METHODS:
            continue
        exported_name = export_artifact_name(path)
        shutil.copy2(path, artifact_dir / exported_name)
        manifest_rows.append(
            {
                "source_file": path.name,
                "exported_file": exported_name,
                "note": "Top-k retrieval output"
                if exported_name.endswith("_retrieval_output.jsonl")
                else "metric/summary artifact",
            }
        )
    write_csv(
        out_dir / "result_file_manifest.csv",
        manifest_rows,
        ["source_file", "exported_file", "note"],
    )


def build_metric_exports(results_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    main_rows = read_csv(results_dir / "main_results.csv")
    normalized = [
        normalized_metric_row(row)
        for row in main_rows
        if str(row.get("method", "")) in FORMAL_METHODS | ANALYSIS_ONLY_METHODS
    ]
    fieldnames = [
        "method",
        "method_role",
        "split",
        "k",
        "group_type",
        "task_type",
        "count",
        "evidence_recall",
        "complete_evidence_recall",
        "mrr",
        "search_success_rate",
        "average_tool_calls",
        "latency_avg_ms",
        "latency_p95_ms",
        "evidence_gain_per_step",
        "stop_accuracy",
    ]
    write_csv(out_dir / "baseline_metrics_all.csv", normalized, fieldnames)
    write_csv(
        out_dir / "baseline_metrics_overall.csv",
        [row for row in normalized if row["group_type"] == "overall"],
        fieldnames,
    )
    write_csv(
        out_dir / "baseline_metrics_by_task.csv",
        [row for row in normalized if row["group_type"] == "task_type"],
        fieldnames,
    )
    return normalized


def retrieval_output_doc_titles(
    retrieval_output: Mapping[str, Any],
    doc_title_by_id: Mapping[str, str],
) -> list[str]:
    titles = []
    for doc in retrieval_output.get("ranked_documents", []):
        doc_id = str(doc.get("doc_id", ""))
        titles.append(str(doc.get("title") or doc_title_by_id.get(doc_id, "")))
    return titles


def build_failure_exports(config: dict, results_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    questions = {question.question_id: question for question in load_questions(config["paths"]["questions_path"])}
    corpus = load_corpus(config["paths"]["corpus_path"])
    doc_title_by_id = {doc.doc_id: doc.title for doc in corpus}

    failure_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("*_final_metrics.csv")):
        method = metrics_path.name.replace("_final_metrics.csv", "")
        if method not in FORMAL_METHODS | ANALYSIS_ONLY_METHODS:
            continue
        retrieval_path = results_dir / f"{method}_final_predictions.jsonl"
        if not retrieval_path.exists():
            continue
        retrieval_outputs = {str(row["question_id"]): row for row in iter_jsonl(retrieval_path)}
        for metric in read_csv(metrics_path):
            if str(metric.get("k")) != "5":
                continue
            if float(metric.get("complete_recall_at_k") or 0.0) >= 1.0:
                continue
            question_id = str(metric["question_id"])
            question = questions[question_id]
            retrieval_output = retrieval_outputs[question_id]
            retrieved_doc_ids = [
                str(doc.get("doc_id", ""))
                for doc in retrieval_output.get("ranked_documents", [])[:5]
            ]
            gold_doc_ids = list(question.gold_documents)
            missed_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id not in set(retrieved_doc_ids)]
            failure_rows.append(
                {
                    "method": method,
                    "method_role": "analysis_only" if method in ANALYSIS_ONLY_METHODS else "formal_baseline",
                    "question_id": question_id,
                    "task_type": question.task_type,
                    "question": question.question,
                    "gold_count": metric.get("gold_count", ""),
                    "matched_gold_count": metric.get("matched_gold_count", ""),
                    "gold_titles": " | ".join(doc_title_by_id.get(doc_id, doc_id) for doc_id in gold_doc_ids),
                    "missed_gold_titles": " | ".join(doc_title_by_id.get(doc_id, doc_id) for doc_id in missed_doc_ids),
                    "retrieved_titles_top5": " | ".join(
                        retrieval_output_doc_titles(retrieval_output, doc_title_by_id)[:5]
                    ),
                    "mrr_at_5": metric.get("mrr", ""),
                    "latency_ms": metric.get("latency_ms", ""),
                }
            )

    fieldnames = [
        "method",
        "method_role",
        "question_id",
        "task_type",
        "question",
        "gold_count",
        "matched_gold_count",
        "gold_titles",
        "missed_gold_titles",
        "retrieved_titles_top5",
        "mrr_at_5",
        "latency_ms",
    ]
    write_csv(out_dir / "failure_cases_k5.csv", failure_rows, fieldnames)
    return failure_rows


def metric_table_row(row: Mapping[str, Any]) -> str:
    return (
        f"| {row['method']} | {row.get('task_type', '') or 'overall'} | "
        f"{percent_or_na(row['evidence_recall'])} | "
        f"{percent_or_na(row['complete_evidence_recall'])} | "
        f"{number_or_na(row['mrr'])} | "
        f"{percent_or_na(row['search_success_rate'])} | "
        f"{number_or_na(row['average_tool_calls'], 2)} | "
        f"{number_or_na(row['latency_avg_ms'], 2)} | "
        f"{number_or_na(row['latency_p95_ms'], 2)} |"
    )


def append_metric_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Method | Group | Evidence Recall@5 | Complete Recall@5 | MRR | Search Success | Avg Tool Calls | Latency Avg ms | Latency P95 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(metric_table_row(row))


def build_markdown_report(
    *,
    out_dir: Path,
    metrics_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    overall_k5 = [
        row
        for row in metrics_rows
        if row["group_type"] == "overall" and str(row["k"]) == "5"
    ]
    task_k5 = [
        row
        for row in metrics_rows
        if row["group_type"] == "task_type" and str(row["k"]) == "5"
    ]
    failure_counts = Counter(row["method"] for row in failure_rows)
    failure_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failure_rows:
        failure_by_task[row["method"]][row["task_type"]] += 1

    method_order = ["dense", "file_fts", "graph_path", "fusion", "oracle"]
    lines = [
        "# Baseline Results And Failure Analysis",
        "",
        "This report contains the frozen retrieval baselines plus two supplemental references: fusion and oracle.",
        "",
        "All rows are Top-k retrieval outputs, not answer-generation predictions. Gold documents are read only by the evaluator after retrieval is complete.",
        "",
        "## Metric Definitions",
        "",
        "- Evidence Recall@5: whether Top-5 contains at least one gold document.",
        "- Complete Evidence Recall@5: whether Top-5 contains all gold documents required by the question.",
        "- MRR: reciprocal rank of the first correct evidence document, averaged over questions.",
        "- Search Success Rate: equivalent to Complete Evidence Recall@k in this project.",
        "- Average Tool Calls: average number of core retrieval backends called per question.",
        "- Latency: average and P95 end-to-end retrieval latency.",
        "- Oracle: analysis-only upper bound that chooses the best existing output per question using evaluation labels.",
        "",
        "## Overall @5",
        "",
    ]
    append_metric_table(
        lines,
        sorted(overall_k5, key=lambda item: method_order.index(item["method"])),
    )

    lines.extend(["", "## Task-wise @5", ""])
    append_metric_table(
        lines,
        sorted(task_k5, key=lambda item: (method_order.index(item["method"]), item["task_type"])),
    )

    lines.extend(
        [
            "",
            "## Failure Counts @5",
            "",
            "| Method | Total Failures | semantic_fact | multi_hop_relation | exact_file_lookup |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in method_order:
        counts = failure_by_task[method]
        total = failure_counts[method]
        lines.append(
            f"| {method} | {total} | {counts['semantic_fact']} | {counts['multi_hop_relation']} | {counts['exact_file_lookup']} |"
        )

    lines.extend(["", "## Representative Failure Cases", ""])
    for method in method_order:
        lines.append(f"### {method}")
        examples = [row for row in failure_rows if row["method"] == method][:5]
        if not examples:
            lines.append("")
            lines.append("No failure cases at @5.")
            continue
        for row in examples:
            lines.extend(
                [
                    "",
                    f"- `{row['question_id']}` / `{row['task_type']}`",
                    f"  - Question: {row['question']}",
                    f"  - Gold: {row['gold_titles']}",
                    f"  - Missed: {row['missed_gold_titles']}",
                    f"  - Retrieved top-5: {row['retrieved_titles_top5']}",
                ]
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `baseline_metrics_all.csv`: all metrics across overall/task groups and k=1/3/5.",
            "- `baseline_metrics_overall.csv`: overall metrics only.",
            "- `baseline_metrics_by_task.csv`: task-wise metrics.",
            "- `failure_cases_k5.csv`: all Complete@5 failure cases.",
            "- `result_file_manifest.csv`: source-to-exported artifact mapping.",
            "- `retrieval_outputs_and_metrics/`: copied Top-k retrieval outputs, metric files, and summary files.",
        ]
    )
    (out_dir / "baseline_failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export baseline results and failure analysis.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-dir", default="results/analysis/baselines")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else Path(config["paths"]["results_dir"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_result_artifacts(results_dir, out_dir)
    metrics_rows = build_metric_exports(results_dir, out_dir)
    failure_rows = build_failure_exports(config, results_dir, out_dir)
    build_markdown_report(out_dir=out_dir, metrics_rows=metrics_rows, failure_rows=failure_rows)
    shutil.copy2(Path(args.config), out_dir / "config.yaml")
    shutil.copy2(PROJECT_ROOT / "docs" / "BASELINES.md", out_dir / "BASELINES.md")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "metrics_rows": len(metrics_rows),
                "failure_rows": len(failure_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
