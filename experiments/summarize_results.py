from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


SUMMARY_NAME_PATTERN = re.compile(r"(?P<method>.+?)_(?P<split>dev|test|final|all)_summary\.json$")


METRIC_KEYS = [
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


def method_split_from_name(path: Path) -> tuple[str, str]:
    match = SUMMARY_NAME_PATTERN.match(path.name)
    if not match:
        return path.stem.replace("_summary", ""), "unknown"
    return match.group("method"), match.group("split")


def summary_to_rows(path: Path) -> list[dict[str, Any]]:
    method, split = method_split_from_name(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    if "by_k" in data:
        for k, group_summary in sorted(data["by_k"].items(), key=lambda item: int(item[0])):
            rows.append(flatten_group(method, split, k, "overall", "overall", group_summary))
        for k, task_summaries in sorted(
            data.get("by_task_type_by_k", {}).items(),
            key=lambda item: int(item[0]),
        ):
            for task_type, group_summary in task_summaries.items():
                rows.append(flatten_group(method, split, k, "task_type", task_type, group_summary))
    else:
        rows.append(flatten_group(method, split, data.get("primary_k", ""), "overall", "overall", data["overall"]))
        for task_type, group_summary in data.get("by_task_type", {}).items():
            rows.append(flatten_group(method, split, data.get("primary_k", ""), "task_type", task_type, group_summary))
    return rows


def flatten_group(
    method: str,
    split: str,
    k: str | int,
    group_type: str,
    group_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "method": method,
        "split": split,
        "k": k,
        "group_type": group_type,
        "group_name": group_name,
        "count": summary.get("count", 0),
    }
    for key in METRIC_KEYS:
        row[key] = summary.get(key)
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize metric JSON files into CSV.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--glob", default="*_summary.json")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    results_dir = Path(args.results_dir) if args.results_dir else Path(config["paths"]["results_dir"])
    rows = []
    for path in sorted(results_dir.glob(args.glob)):
        rows.extend(summary_to_rows(path))

    out_path = Path(args.out) if args.out else results_dir / "main_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "split",
        "k",
        "group_type",
        "group_name",
        "count",
        *METRIC_KEYS,
    ]
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
