from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import write_jsonl
from .config import load_config


TASK_TYPES = ("semantic_fact", "multi_hop", "exact_file_lookup")


def stable_doc_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    slug = slug[:48] or "untitled"
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"doc_{slug}_{digest}"


def stable_sample_id(sample: Mapping[str, Any], fallback_index: int) -> str:
    raw_id = (
        sample.get("id")
        or sample.get("_id")
        or sample.get("qid")
        or f"sample_{fallback_index:06d}"
    )
    return str(raw_id)


def extract_context(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = sample.get("context", {})

    if isinstance(context, Mapping):
        titles = list(context.get("title", []))
        sentence_groups = list(context.get("sentences", []))
        return [
            {
                "title": str(title),
                "sentences": [str(sentence) for sentence in sentences],
            }
            for title, sentences in zip(titles, sentence_groups)
            if str(title).strip()
        ]

    if isinstance(context, list):
        rows = []
        for item in context:
            if isinstance(item, Mapping):
                title = str(item.get("title", "")).strip()
                sentences = item.get("sentences", [])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0]).strip()
                sentences = item[1]
            else:
                continue
            if title:
                rows.append(
                    {
                        "title": title,
                        "sentences": [str(sentence) for sentence in sentences],
                    }
                )
        return rows

    return []


def extract_supporting_facts(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    supporting_facts = sample.get("supporting_facts", {})

    if isinstance(supporting_facts, Mapping):
        titles = list(supporting_facts.get("title", []))
        sent_ids = list(supporting_facts.get("sent_id", []))
        return [
            {"title": str(title), "sent_id": int(sent_id)}
            for title, sent_id in zip(titles, sent_ids)
        ]

    rows = []
    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, Mapping):
                title = item.get("title")
                sent_id = item.get("sent_id")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sent_id = item[1]
            else:
                continue
            if title is not None and sent_id is not None:
                rows.append({"title": str(title), "sent_id": int(sent_id)})
    return rows


def support_rows_with_text(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    context_by_title = {row["title"]: row for row in extract_context(sample)}
    rows = []
    for fact in extract_supporting_facts(sample):
        context_row = context_by_title.get(fact["title"])
        if not context_row:
            continue
        sentences = context_row["sentences"]
        sent_id = fact["sent_id"]
        text = sentences[sent_id] if 0 <= sent_id < len(sentences) else ""
        rows.append(
            {
                "title": fact["title"],
                "doc_id": stable_doc_id(fact["title"]),
                "sent_id": sent_id,
                "text": text,
            }
        )
    return rows


def load_hotpot_dataset(
    *,
    dataset_name: str,
    fallback_dataset_name: str | None,
    config_name: str,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required. Run: pip install -r requirements.txt"
        ) from exc

    dataset_names = [dataset_name]
    if fallback_dataset_name and fallback_dataset_name not in dataset_names:
        dataset_names.append(fallback_dataset_name)

    last_error: Exception | None = None
    for name in dataset_names:
        try:
            return load_dataset(name, config_name, split=split)
        except Exception as exc:  # pragma: no cover - depends on network/cache.
            last_error = exc
    raise RuntimeError(
        f"Could not load HotpotQA dataset using {dataset_names} config={config_name!r} "
        f"split={split!r}. Last error: {last_error}"
    )


def is_usable_seed(sample: Mapping[str, Any]) -> bool:
    support_rows = support_rows_with_text(sample)
    support_titles = {row["title"] for row in support_rows}
    if len(support_titles) < 2:
        return False
    if not str(sample.get("question", "")).strip():
        return False
    return any(row["text"].strip() for row in support_rows)


def collect_seed_samples(
    dataset: Iterable[Mapping[str, Any]],
    *,
    needed_seeds: int,
    seed: int,
    scan_limit: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    rng = random.Random(seed)
    candidates: list[tuple[int, Mapping[str, Any]]] = []

    for index, sample in enumerate(dataset):
        if index >= scan_limit:
            break
        if is_usable_seed(sample):
            candidates.append((index, sample))

    rng.shuffle(candidates)
    if len(candidates) < needed_seeds:
        raise RuntimeError(
            f"Only found {len(candidates)} usable HotpotQA seeds, need {needed_seeds}. "
            "Increase --scan-limit or lower --target-questions."
        )
    return candidates[:needed_seeds]


def task_quotas(target_questions: int) -> dict[str, int]:
    base = target_questions // len(TASK_TYPES)
    remainder = target_questions % len(TASK_TYPES)
    quotas = {task_type: base for task_type in TASK_TYPES}
    for task_type in TASK_TYPES[:remainder]:
        quotas[task_type] += 1
    return quotas


def choose_semantic_support(support_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_text = [row for row in support_rows if row["text"].strip()]
    return rows_with_text[0] if rows_with_text else support_rows[0]


def choose_exact_support(support_rows: list[dict[str, Any]]) -> dict[str, Any]:
    number_or_date = re.compile(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b")
    for row in support_rows:
        if number_or_date.search(row["text"]):
            return row
    return choose_semantic_support(support_rows)


def gold_sentences_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": str(row["doc_id"]),
            "sent_id": int(row["sent_id"]),
            "text": str(row.get("text", "")),
        }
        for row in rows
    ]


def build_questions_and_corpus(
    seed_samples: list[tuple[int, Mapping[str, Any]]],
    *,
    target_questions: int,
    dev_ratio: float,
    seed: int,
    mark_quality_checked: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    quotas = task_quotas(target_questions)
    counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    corpus_by_doc_id: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []

    for fallback_index, sample in seed_samples:
        source_id = stable_sample_id(sample, fallback_index)
        context_rows = extract_context(sample)
        support_rows = support_rows_with_text(sample)
        if not context_rows or not support_rows:
            continue

        for context_row in context_rows:
            doc_id = stable_doc_id(context_row["title"])
            existing = corpus_by_doc_id.get(doc_id)
            if existing is None:
                corpus_by_doc_id[doc_id] = {
                    "doc_id": doc_id,
                    "title": context_row["title"],
                    "sentences": context_row["sentences"],
                    "full_text": " ".join(context_row["sentences"]),
                    "source_question_ids": [source_id],
                }
            elif source_id not in existing["source_question_ids"]:
                existing["source_question_ids"].append(source_id)

        support_doc_ids = sorted({row["doc_id"] for row in support_rows})
        support_titles = sorted({row["title"] for row in support_rows})
        original_question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()

        if counts["multi_hop"] < quotas["multi_hop"]:
            counters["multi_hop"] += 1
            questions.append(
                {
                    "question_id": f"mh_{counters['multi_hop']:04d}",
                    "question": original_question,
                    "task_type": "multi_hop",
                    "gold_documents": support_doc_ids,
                    "gold_sentences": gold_sentences_from_rows(support_rows),
                    "source_hotpot_id": source_id,
                    "split": "unset",
                    "quality_checked": mark_quality_checked,
                    "metadata": {
                        "answer": answer,
                        "support_titles": support_titles,
                        "construction": "hotpot_original_question",
                    },
                }
            )
            counts["multi_hop"] += 1

        if counts["semantic_fact"] < quotas["semantic_fact"]:
            support = choose_semantic_support(support_rows)
            counters["semantic_fact"] += 1
            claim = support["text"].strip()
            questions.append(
                {
                    "question_id": f"sf_{counters['semantic_fact']:04d}",
                    "question": (
                        "Which source document provides evidence for this claim: "
                        f"{claim}"
                    ),
                    "task_type": "semantic_fact",
                    "gold_documents": [support["doc_id"]],
                    "gold_sentences": gold_sentences_from_rows([support]),
                    "source_hotpot_id": source_id,
                    "split": "unset",
                    "quality_checked": mark_quality_checked,
                    "metadata": {
                        "answer": answer,
                        "support_titles": [support["title"]],
                        "construction": "bootstrap_claim_question",
                        "needs_manual_paraphrase": not mark_quality_checked,
                    },
                }
            )
            counts["semantic_fact"] += 1

        if counts["exact_file_lookup"] < quotas["exact_file_lookup"]:
            support = choose_exact_support(support_rows)
            counters["exact_file_lookup"] += 1
            questions.append(
                {
                    "question_id": f"ex_{counters['exact_file_lookup']:04d}",
                    "question": (
                        f'Find the document titled "{support["title"]}" and locate '
                        f"the evidence related to: {original_question}"
                    ),
                    "task_type": "exact_file_lookup",
                    "gold_documents": [support["doc_id"]],
                    "gold_sentences": gold_sentences_from_rows([support]),
                    "source_hotpot_id": source_id,
                    "split": "unset",
                    "quality_checked": mark_quality_checked,
                    "metadata": {
                        "answer": answer,
                        "support_titles": [support["title"]],
                        "construction": "bootstrap_title_lookup",
                    },
                }
            )
            counts["exact_file_lookup"] += 1

        if sum(counts.values()) >= target_questions:
            break

    if sum(counts.values()) < target_questions:
        raise RuntimeError(
            f"Built {sum(counts.values())} questions, target is {target_questions}."
        )

    assign_stratified_splits(questions, dev_ratio=dev_ratio, seed=seed)
    splits = {
        "dev": [row["question_id"] for row in questions if row["split"] == "dev"],
        "test": [row["question_id"] for row in questions if row["split"] == "test"],
    }
    corpus = sorted(corpus_by_doc_id.values(), key=lambda row: row["doc_id"])
    return corpus, questions, splits


def assign_stratified_splits(
    questions: list[dict[str, Any]],
    *,
    dev_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_task[question["task_type"]].append(question)

    for task_questions in by_task.values():
        rng.shuffle(task_questions)
        dev_count = round(len(task_questions) * dev_ratio)
        if len(task_questions) > 1:
            dev_count = max(1, min(dev_count, len(task_questions) - 1))
        for index, question in enumerate(task_questions):
            question["split"] = "dev" if index < dev_count else "test"


def write_data_check_csv(path: str | Path, questions: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "task_type",
        "split",
        "source_hotpot_id",
        "quality_checked",
        "check_status",
        "question",
        "gold_documents",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for question in questions:
            checked = bool(question.get("quality_checked"))
            writer.writerow(
                {
                    "question_id": question["question_id"],
                    "task_type": question["task_type"],
                    "split": question["split"],
                    "source_hotpot_id": question.get("source_hotpot_id", ""),
                    "quality_checked": checked,
                    "check_status": "checked_by_flag" if checked else "needs_manual_review",
                    "question": question["question"],
                    "gold_documents": "|".join(question["gold_documents"]),
                    "notes": "",
                }
            )


def write_outputs(
    *,
    corpus: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    splits: dict[str, list[str]],
    corpus_path: str | Path,
    questions_path: str | Path,
    splits_path: str | Path,
    data_check_path: str | Path,
) -> None:
    write_jsonl(corpus_path, corpus)
    write_jsonl(questions_path, questions)
    splits_path = Path(splits_path)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_data_check_csv(data_check_path, questions)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the HotpotQA-derived MVP data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target-questions", type=int, default=None)
    parser.add_argument("--dev-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--fallback-dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument(
        "--mark-quality-checked",
        action="store_true",
        help="Mark generated questions as checked. Use only after manual review.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    target_questions = args.target_questions or int(config["data"]["target_questions"])
    dev_ratio = args.dev_ratio if args.dev_ratio is not None else float(config["data"]["dev_ratio"])
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    dataset_name = args.dataset_name or str(config["data"]["hotpot_dataset_name"])
    fallback_dataset_name = args.fallback_dataset_name
    if fallback_dataset_name is None:
        fallback_dataset_name = str(config["data"].get("hotpot_dataset_fallback_name", ""))
    dataset_config = args.dataset_config or str(config["data"]["hotpot_config"])
    dataset_split = args.dataset_split or str(config["data"]["hotpot_split"])

    needed_seeds = (target_questions + len(TASK_TYPES) - 1) // len(TASK_TYPES)
    dataset = load_hotpot_dataset(
        dataset_name=dataset_name,
        fallback_dataset_name=fallback_dataset_name,
        config_name=dataset_config,
        split=dataset_split,
    )
    seed_samples = collect_seed_samples(
        dataset,
        needed_seeds=needed_seeds,
        seed=seed,
        scan_limit=args.scan_limit,
    )
    corpus, questions, splits = build_questions_and_corpus(
        seed_samples,
        target_questions=target_questions,
        dev_ratio=dev_ratio,
        seed=seed,
        mark_quality_checked=args.mark_quality_checked,
    )

    paths = config["paths"]
    write_outputs(
        corpus=corpus,
        questions=questions,
        splits=splits,
        corpus_path=paths["corpus_path"],
        questions_path=paths["questions_path"],
        splits_path=paths["splits_path"],
        data_check_path=paths["data_check_path"],
    )

    print(
        json.dumps(
            {
                "corpus_documents": len(corpus),
                "questions": len(questions),
                "dev_questions": len(splits["dev"]),
                "test_questions": len(splits["test"]),
                "quality_checked": args.mark_quality_checked,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
