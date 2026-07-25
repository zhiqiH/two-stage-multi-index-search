from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import iter_jsonl, write_jsonl
from .config import load_config


EXACT_SUBTYPE_ORDER = (
    "title_anchor",
    "date_number_lookup",
    "exact_phrase_lookup",
)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_pool(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = list(iter_jsonl(path))
    pool = {str(row["candidate_id"]): row for row in rows}
    if len(pool) != len(rows):
        raise ValueError(f"Duplicate candidate_id values in {path}")
    return pool


def accepted_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    question_field: str,
) -> list[dict[str, str]]:
    accepted = []
    for row in rows:
        if row.get("status") != "accepted":
            continue
        if not row.get(question_field, "").strip():
            continue
        accepted.append(dict(row))
    return accepted


def sample_rows(
    rows: Sequence[dict[str, str]],
    *,
    quota: int,
    rng: random.Random,
    label: str,
) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda row: row["question_id"])
    if len(rows) < quota:
        raise ValueError(f"Not enough accepted rows for {label}: {len(rows)} < {quota}")
    return sorted(rng.sample(rows, quota), key=lambda row: row["question_id"])


def build_question(
    *,
    final_question_id: str,
    question_text: str,
    pool_row: Mapping[str, Any],
    source_row: Mapping[str, str],
    construction: str,
    subtype: str,
    source_sheet: str,
) -> dict[str, Any]:
    metadata = dict(pool_row.get("metadata", {}))
    metadata.update(
        {
            "anchor_value": source_row.get("anchor_value", metadata.get("anchor_value", "")),
            "candidate_id": pool_row["candidate_id"],
            "construction": construction,
            "llm_attempts": source_row.get("llm_attempts", ""),
            "llm_verdict": source_row.get("llm_verdict", ""),
            "source_sheet": source_sheet,
            "source_status": source_row.get("status", ""),
            "subtype": subtype,
            "verifier_model": source_row.get("verifier_model", ""),
        }
    )
    if "requires_manual_rewrite" in metadata:
        metadata["requires_manual_rewrite"] = False
    if "requires_manual_review" in metadata:
        metadata["requires_manual_review"] = False
    if source_row.get("generator_model"):
        metadata["generator_model"] = source_row["generator_model"]

    return {
        "question_id": final_question_id,
        "question": question_text.strip(),
        "task_type": str(pool_row["task_type"]),
        "gold_documents": [str(doc_id) for doc_id in pool_row["gold_documents"]],
        "gold_sentences": list(pool_row.get("gold_sentences", [])),
        "source_hotpot_id": pool_row.get("source_hotpot_id"),
        "split": "final",
        "quality_checked": True,
        "metadata": metadata,
    }


def select_questions(
    *,
    candidate_pool: Mapping[str, dict[str, Any]],
    semantic_rows: Sequence[dict[str, str]],
    exact_rows: Sequence[dict[str, str]],
    multi_rows: Sequence[dict[str, str]],
    semantic_quota: int,
    multi_quota: int,
    exact_subtype_quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    semantic_selected = sample_rows(
        accepted_rows(semantic_rows, question_field="rewritten_question"),
        quota=semantic_quota,
        rng=rng,
        label="semantic_fact",
    )
    for index, row in enumerate(semantic_selected, start=1):
        pool_row = candidate_pool[row["question_id"]]
        selected.append(
            build_question(
                final_question_id=f"sf_{index:04d}",
                question_text=row["rewritten_question"],
                pool_row=pool_row,
                source_row=row,
                construction="llm_semantic_rewrite_accepted",
                subtype="semantic_rewrite",
                source_sheet="semantic_rewrite_sheet_llm.csv",
            )
        )

    multi_selected = sample_rows(
        accepted_rows(multi_rows, question_field="question"),
        quota=multi_quota,
        rng=rng,
        label="multi_hop_relation",
    )
    for index, row in enumerate(multi_selected, start=1):
        pool_row = candidate_pool[row["question_id"]]
        selected.append(
            build_question(
                final_question_id=f"mh_{index:04d}",
                question_text=row["question"],
                pool_row=pool_row,
                source_row=row,
                construction="llm_verified_hotpot_multi_hop",
                subtype="hotpot_relation_verified",
                source_sheet="multi_hop_candidate_sheet_llm.csv",
            )
        )

    exact_accepted = accepted_rows(exact_rows, question_field="final_question")
    exact_by_subtype: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_accepted:
        exact_by_subtype[row["subtype"]].append(row)

    exact_counter = 0
    exact_id_prefix = {
        "title_anchor": "ex_title",
        "date_number_lookup": "ex_date",
        "exact_phrase_lookup": "ex_phrase",
    }
    for subtype in EXACT_SUBTYPE_ORDER:
        quota = int(exact_subtype_quotas[subtype])
        subtype_selected = sample_rows(
            exact_by_subtype[subtype],
            quota=quota,
            rng=rng,
            label=f"exact_file_lookup/{subtype}",
        )
        for index, row in enumerate(subtype_selected, start=1):
            exact_counter += 1
            pool_row = candidate_pool[row["question_id"]]
            selected.append(
                build_question(
                    final_question_id=f"{exact_id_prefix[subtype]}_{index:04d}",
                    question_text=row["final_question"],
                    pool_row=pool_row,
                    source_row=row,
                    construction="llm_exact_file_lookup_accepted",
                    subtype=subtype,
                    source_sheet="exact_lookup_sheet_llm.csv",
                )
            )

    return selected


def load_corpus_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build_final_corpus(
    staging_corpus: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final_question_ids_by_gold_doc: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        for doc_id in question["gold_documents"]:
            final_question_ids_by_gold_doc[str(doc_id)].append(str(question["question_id"]))

    final_gold_doc_ids = set(final_question_ids_by_gold_doc)
    final_corpus = []
    for row in staging_corpus:
        doc_id = str(row["doc_id"])
        metadata = dict(row.get("metadata", {}))
        if "corpus_role" in metadata:
            metadata["staging_corpus_role"] = metadata["corpus_role"]
        metadata["corpus_role"] = "selected_gold" if doc_id in final_gold_doc_ids else "noise"
        metadata["final_gold_question_ids"] = sorted(final_question_ids_by_gold_doc.get(doc_id, []))

        final_corpus.append(
            {
                "doc_id": doc_id,
                "title": row.get("title", ""),
                "sentences": list(row.get("sentences", [])),
                "full_text": row.get("full_text", " ".join(row.get("sentences", []))),
                "source_question_ids": sorted(final_question_ids_by_gold_doc.get(doc_id, [])),
                "metadata": metadata,
            }
        )
    return sorted(final_corpus, key=lambda row: row["doc_id"])


def validate_final_dataset(
    *,
    questions: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    target_questions: int,
    target_corpus_documents: int,
    semantic_quota: int,
    multi_quota: int,
    exact_subtype_quotas: Mapping[str, int],
) -> dict[str, Any]:
    errors: list[str] = []
    doc_ids = {str(row["doc_id"]) for row in corpus}
    question_ids = [str(row["question_id"]) for row in questions]

    task_counts = Counter(str(row["task_type"]) for row in questions)
    exact_counts = Counter(str(row.get("metadata", {}).get("subtype", "")) for row in questions if row["task_type"] == "exact_file_lookup")

    if len(questions) != target_questions:
        errors.append(f"questions={len(questions)} expected={target_questions}")
    if len(corpus) != target_corpus_documents:
        errors.append(f"corpus_documents={len(corpus)} expected={target_corpus_documents}")
    if len(question_ids) != len(set(question_ids)):
        errors.append("duplicate final question_id values")
    if task_counts["semantic_fact"] != semantic_quota:
        errors.append(f"semantic_fact={task_counts['semantic_fact']} expected={semantic_quota}")
    if task_counts["multi_hop_relation"] != multi_quota:
        errors.append(f"multi_hop_relation={task_counts['multi_hop_relation']} expected={multi_quota}")
    if task_counts["exact_file_lookup"] != sum(int(x) for x in exact_subtype_quotas.values()):
        errors.append("exact_file_lookup count mismatch")
    for subtype, quota in exact_subtype_quotas.items():
        if exact_counts[subtype] != int(quota):
            errors.append(f"exact subtype {subtype}={exact_counts[subtype]} expected={quota}")

    for question in questions:
        if question.get("split") != "final":
            errors.append(f"{question['question_id']} split is not final")
        if not question.get("quality_checked"):
            errors.append(f"{question['question_id']} quality_checked is false")
        if not str(question.get("question", "")).strip():
            errors.append(f"{question['question_id']} has empty question")
        gold_documents = [str(doc_id) for doc_id in question.get("gold_documents", [])]
        if not gold_documents:
            errors.append(f"{question['question_id']} has no gold_documents")
        for doc_id in gold_documents:
            if doc_id not in doc_ids:
                errors.append(f"{question['question_id']} missing gold doc in corpus: {doc_id}")
        gold_sentences = question.get("gold_sentences", [])
        if not gold_sentences:
            errors.append(f"{question['question_id']} has no gold_sentences")
        for sentence in gold_sentences:
            if not str(sentence.get("text", "")).strip():
                errors.append(f"{question['question_id']} has empty gold sentence text")

    if errors:
        raise ValueError("Final dataset validation failed:\n" + "\n".join(errors[:50]))

    return {
        "corpus_documents": len(corpus),
        "exact_subtype_counts": dict(sorted(exact_counts.items())),
        "question_count": len(questions),
        "task_counts": dict(sorted(task_counts.items())),
    }


def write_data_check(path: str | Path, questions: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for question in questions:
        metadata = dict(question.get("metadata", {}))
        rows.append(
            {
                "question_id": question["question_id"],
                "task_type": question["task_type"],
                "subtype": metadata.get("subtype", ""),
                "split": question["split"],
                "quality_checked": question["quality_checked"],
                "candidate_id": metadata.get("candidate_id", ""),
                "source_hotpot_id": question.get("source_hotpot_id", ""),
                "source_status": metadata.get("source_status", ""),
                "check_status": "accepted_llm_verified",
                "question": question["question"],
                "gold_documents": "|".join(question["gold_documents"]),
                "notes": "",
            }
        )
    write_csv_rows(path, rows)


def write_dataset_card(
    path: str | Path,
    *,
    seed: int,
    validation: Mapping[str, Any],
    frozen: bool,
    manifest_path: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Final Dataset Card",
                "",
                f"Status: {'frozen' if frozen else 'built_not_frozen'}",
                f"Build seed: {seed}",
                f"Selection manifest: `{manifest_path}`",
                "",
                "## Size",
                "",
                f"- Questions: {validation['question_count']}",
                f"- Corpus documents: {validation['corpus_documents']}",
                "",
                "## Task Counts",
                "",
                *[
                    f"- {task_type}: {count}"
                    for task_type, count in validation["task_counts"].items()
                ],
                "",
                "## Exact Lookup Subtypes",
                "",
                *[
                    f"- {subtype}: {count}"
                    for subtype, count in validation["exact_subtype_counts"].items()
                ],
                "",
                "## Isolation Rule",
                "",
                "This final dataset is for evaluation only. Do not tune Router, Judge, retriever parameters, or question wording on final experiment failures.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_freeze_record(
    path: str | Path,
    *,
    seed: int,
    manifest_path: str,
    manifest_hash: str,
    validation: Mapping[str, Any],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    Path(path).write_text(
        "\n".join(
            [
                "FINAL DATASET FROZEN",
                f"timestamp_utc: {timestamp}",
                f"seed: {seed}",
                f"manifest_path: {manifest_path}",
                f"manifest_sha256: {manifest_hash}",
                f"question_count: {validation['question_count']}",
                f"corpus_documents: {validation['corpus_documents']}",
                "method_tuning_on_final_allowed: false",
                "",
                "After this file exists, do not tune Router, Judge, retriever parameters, or question wording using final experiment results.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def guard_outputs(paths: Mapping[str, Path], *, overwrite: bool, freeze: bool) -> None:
    frozen_path = paths["frozen"]
    if frozen_path.exists() and not overwrite:
        raise FileExistsError(f"Final dataset is already frozen: {frozen_path}")
    output_keys = ["corpus", "questions", "splits", "data_check", "dataset_card", "manifest"]
    if freeze:
        output_keys.append("frozen")
    existing = [str(paths[key]) for key in output_keys if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist. Use --overwrite if intentional:\n" + "\n".join(existing))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and optionally freeze the final 180-question dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--semantic-in", default="data/final/manual/semantic_rewrite_sheet_llm.csv")
    parser.add_argument("--exact-in", default="data/final/manual/exact_lookup_sheet_llm.csv")
    parser.add_argument("--multi-hop-in", default="data/final/manual/multi_hop_candidate_sheet_llm.csv")
    parser.add_argument("--candidate-pool", default="data/final/staging/candidate_pool.jsonl")
    parser.add_argument("--staging-corpus", default="data/final/staging/corpus.jsonl")
    parser.add_argument("--manifest-out", default="data/final/selection_manifest.json")
    parser.add_argument("--dataset-card-out", default="data/final/DATASET_CARD.md")
    parser.add_argument("--freeze", action="store_true", help="Create FROZEN.txt and remove NOT_FROZEN_YET.txt.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing final outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])

    paths_config = config["paths"]
    final_dir = Path(paths_config["corpus_path"]).parent
    output_paths = {
        "corpus": Path(paths_config["corpus_path"]),
        "questions": Path(paths_config["questions_path"]),
        "splits": Path(paths_config["splits_path"]),
        "data_check": Path(paths_config["data_check_path"]),
        "dataset_card": Path(args.dataset_card_out),
        "manifest": Path(args.manifest_out),
        "frozen": final_dir / "FROZEN.txt",
        "not_frozen": final_dir / "NOT_FROZEN_YET.txt",
    }
    if not args.dry_run:
        guard_outputs(output_paths, overwrite=args.overwrite, freeze=args.freeze)

    semantic_rows = read_csv_rows(args.semantic_in)
    exact_rows = read_csv_rows(args.exact_in)
    multi_rows = read_csv_rows(args.multi_hop_in)
    candidate_pool = load_candidate_pool(args.candidate_pool)
    staging_corpus = load_corpus_rows(args.staging_corpus)

    data_config = config["data"]
    semantic_quota = int(data_config["task_quotas"]["semantic_fact"])
    multi_quota = int(data_config["task_quotas"]["multi_hop_relation"])
    exact_subtype_quotas = {
        subtype: int(data_config["exact_file_lookup_subtypes"][subtype])
        for subtype in EXACT_SUBTYPE_ORDER
    }
    questions = select_questions(
        candidate_pool=candidate_pool,
        semantic_rows=semantic_rows,
        exact_rows=exact_rows,
        multi_rows=multi_rows,
        semantic_quota=semantic_quota,
        multi_quota=multi_quota,
        exact_subtype_quotas=exact_subtype_quotas,
        seed=seed,
    )
    corpus = build_final_corpus(staging_corpus, questions)
    validation = validate_final_dataset(
        questions=questions,
        corpus=corpus,
        target_questions=int(data_config["target_questions"]),
        target_corpus_documents=int(data_config["target_corpus_documents"]),
        semantic_quota=semantic_quota,
        multi_quota=multi_quota,
        exact_subtype_quotas=exact_subtype_quotas,
    )

    selected_manifest_rows = [
        {
            "candidate_id": question["metadata"]["candidate_id"],
            "final_question_id": question["question_id"],
            "source_hotpot_id": question.get("source_hotpot_id", ""),
            "subtype": question["metadata"].get("subtype", ""),
            "task_type": question["task_type"],
        }
        for question in questions
    ]
    input_hashes = {
        "candidate_pool": sha256_file(args.candidate_pool),
        "exact_sheet": sha256_file(args.exact_in),
        "multi_hop_sheet": sha256_file(args.multi_hop_in),
        "semantic_sheet": sha256_file(args.semantic_in),
        "staging_corpus": sha256_file(args.staging_corpus),
    }

    summary = {
        "freeze": bool(args.freeze),
        "input_hashes": input_hashes,
        "output_paths": {key: str(value) for key, value in output_paths.items() if key != "not_frozen"},
        "seed": seed,
        "selected": selected_manifest_rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    write_jsonl(output_paths["corpus"], corpus)
    write_jsonl(output_paths["questions"], questions)
    output_paths["splits"].write_text(
        json.dumps({"final": [row["question_id"] for row in questions]}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_data_check(output_paths["data_check"], questions)
    write_dataset_card(
        output_paths["dataset_card"],
        seed=seed,
        validation=validation,
        frozen=bool(args.freeze),
        manifest_path=str(output_paths["manifest"]),
    )

    output_hashes = {
        "corpus": sha256_file(output_paths["corpus"]),
        "data_check": sha256_file(output_paths["data_check"]),
        "dataset_card": sha256_file(output_paths["dataset_card"]),
        "questions": sha256_file(output_paths["questions"]),
        "splits": sha256_file(output_paths["splits"]),
    }
    summary["output_hashes"] = output_hashes
    output_paths["manifest"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.freeze:
        write_freeze_record(
            output_paths["frozen"],
            seed=seed,
            manifest_path=str(output_paths["manifest"]),
            manifest_hash=sha256_file(output_paths["manifest"]),
            validation=validation,
        )
        if output_paths["not_frozen"].exists():
            output_paths["not_frozen"].unlink()

    print(json.dumps({"output_hashes": output_hashes, **validation}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
