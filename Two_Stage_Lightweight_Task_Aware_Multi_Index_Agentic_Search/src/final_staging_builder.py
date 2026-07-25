from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import write_jsonl
from .config import load_config
from .data_builder import (
    extract_context,
    load_hotpot_dataset,
    stable_doc_id,
    stable_sample_id,
    support_rows_with_text,
)
from .text_utils import keywords, numbers_and_dates


SEMANTIC_CANDIDATES = 90
MULTI_HOP_CANDIDATES = 90
EXACT_CANDIDATES_PER_SUBTYPE = 30


def is_good_sample(sample: Mapping[str, Any]) -> bool:
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip().lower()
    support_rows = support_rows_with_text(sample)
    support_doc_ids = {row["doc_id"] for row in support_rows}
    return (
        bool(question)
        and answer not in {"yes", "no"}
        and len(support_doc_ids) == 2
        and all(row["text"].strip() for row in support_rows)
    )


def choose_support_row(sample: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in support_rows_with_text(sample)
        if is_clean_support_sentence(row["text"])
    ]
    if not rows:
        rows = [row for row in support_rows_with_text(sample) if row["text"].strip()]
    if not rows:
        return None
    return sorted(rows, key=lambda row: (abs(len(row["text"]) - 140), row["title"]))[0]


def is_clean_support_sentence(text: str) -> bool:
    text = text.strip()
    if not 60 <= len(text) <= 260:
        return False
    if any(char in text for char in "[]{}"):
        return False
    return len(text.split()) >= 8


def choose_number_support_row(sample: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    for row in support_rows_with_text(sample):
        values = numbers_and_dates(row["text"])
        if values:
            return row, values[0]
    return None


def choose_phrase_support_row(sample: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    for row in support_rows_with_text(sample):
        phrase = choose_exact_phrase(row["text"], row["title"])
        if phrase:
            return row, phrase
    return None


def choose_exact_phrase(text: str, title: str) -> str | None:
    title_norm = normalize_for_compare(title)
    proper_phrase_pattern = re.compile(
        r"\b(?:[A-Z][a-zA-Z0-9'&.-]+)(?:\s+(?:[A-Z][a-zA-Z0-9'&.-]+|of|the|and|de|la)){1,5}"
    )
    entity_candidates = []
    for match in proper_phrase_pattern.finditer(text):
        phrase = match.group(0).strip()
        if normalize_for_compare(phrase) != title_norm:
            entity_candidates.append(phrase)
    if entity_candidates:
        return max(entity_candidates, key=len)

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'’-]+", text)
    stop = {
        "the",
        "and",
        "was",
        "were",
        "that",
        "this",
        "with",
        "from",
        "into",
        "also",
        "later",
        "which",
    }
    best: tuple[int, str] | None = None
    for window_size in range(6, 2, -1):
        for start in range(0, max(0, len(tokens) - window_size + 1)):
            window = tokens[start : start + window_size]
            useful = [token for token in window if token.lower() not in stop]
            phrase = " ".join(window)
            if len(useful) >= 2 and 18 <= len(phrase) <= 90:
                score = len(useful) * 10 + len(phrase)
                if best is None or score > best[0]:
                    best = (score, phrase)
        if best:
            return best[1]
    return None


def normalize_for_compare(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def collect_usable_samples(
    dataset: Iterable[Mapping[str, Any]],
    *,
    scan_limit: int,
    seed: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    samples = []
    for index, sample in enumerate(dataset):
        if index >= scan_limit:
            break
        if is_good_sample(sample):
            samples.append((index, sample))
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


def build_candidates(
    samples: list[tuple[int, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], set[str]]:
    used_source_ids: set[str] = set()
    pool: list[dict[str, Any]] = []
    sheets: dict[str, list[dict[str, Any]]] = {
        "semantic": [],
        "multi_hop": [],
        "exact": [],
    }

    for fallback_index, sample in samples:
        if len(sheets["semantic"]) >= SEMANTIC_CANDIDATES:
            break
        source_id = stable_sample_id(sample, fallback_index)
        if source_id in used_source_ids:
            continue
        support = choose_support_row(sample)
        if not support:
            continue
        candidate_id = f"sf_cand_{len(sheets['semantic']) + 1:04d}"
        claim_question = f"Rewrite this fact into a natural semantic question: {support['text']}"
        row = {
            "question_id": candidate_id,
            "source_hotpot_id": source_id,
            "gold_doc_id": support["doc_id"],
            "gold_title": support["title"],
            "gold_sentence": support["text"],
            "current_question": claim_question,
            "rewritten_question": "",
            "status": "todo",
            "notes": "",
        }
        sheets["semantic"].append(row)
        pool.append(
            candidate_to_pool_row(
                candidate_id=candidate_id,
                task_type="semantic_fact",
                subtype="manual_semantic_rewrite",
                sample=sample,
                source_id=source_id,
                gold_rows=[support],
                candidate_question=claim_question,
                metadata={"requires_manual_rewrite": True},
            )
        )
        used_source_ids.add(source_id)

    for fallback_index, sample in samples:
        if len(sheets["multi_hop"]) >= MULTI_HOP_CANDIDATES:
            break
        source_id = stable_sample_id(sample, fallback_index)
        if source_id in used_source_ids:
            continue
        support_rows = support_rows_with_text(sample)
        support_doc_ids = sorted({row["doc_id"] for row in support_rows})
        if len(support_doc_ids) != 2:
            continue
        candidate_id = f"mh_cand_{len(sheets['multi_hop']) + 1:04d}"
        question = str(sample.get("question", "")).strip()
        row = {
            "question_id": candidate_id,
            "source_hotpot_id": source_id,
            "question": question,
            "answer": str(sample.get("answer", "")),
            "gold_doc_ids": "|".join(support_doc_ids),
            "gold_titles": "|".join(sorted({row["title"] for row in support_rows})),
            "gold_sentences": " || ".join(row["text"] for row in support_rows),
            "status": "todo",
            "notes": "",
        }
        sheets["multi_hop"].append(row)
        pool.append(
            candidate_to_pool_row(
                candidate_id=candidate_id,
                task_type="multi_hop_relation",
                subtype="hotpot_relation_candidate",
                sample=sample,
                source_id=source_id,
                gold_rows=support_rows,
                candidate_question=question,
                metadata={"answer": str(sample.get("answer", ""))},
            )
        )
        used_source_ids.add(source_id)

    exact_builders = [
        ("title_anchor", build_title_anchor_candidate),
        ("date_number_lookup", build_date_number_candidate),
        ("exact_phrase_lookup", build_exact_phrase_candidate),
    ]
    for subtype, builder in exact_builders:
        subtype_count = 0
        for fallback_index, sample in samples:
            if subtype_count >= EXACT_CANDIDATES_PER_SUBTYPE:
                break
            source_id = stable_sample_id(sample, fallback_index)
            if source_id in used_source_ids:
                continue
            built = builder(sample, source_id, subtype_count + 1)
            if not built:
                continue
            sheet_row, pool_row = built
            sheets["exact"].append(sheet_row)
            pool.append(pool_row)
            used_source_ids.add(source_id)
            subtype_count += 1

    assert_minimum_counts(sheets)
    return pool, sheets, used_source_ids


def build_title_anchor_candidate(
    sample: Mapping[str, Any],
    source_id: str,
    number: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    support = choose_support_row(sample)
    if not support:
        return None
    candidate_id = f"ex_title_cand_{number:04d}"
    original_question = str(sample.get("question", "")).strip()
    suggested = (
        f'In the "{support["title"]}" document, what evidence helps answer: '
        f"{original_question}"
    )
    return build_exact_rows(
        candidate_id=candidate_id,
        subtype="title_anchor",
        sample=sample,
        source_id=source_id,
        support=support,
        anchor_value=support["title"],
        suggested_question=suggested,
    )


def build_date_number_candidate(
    sample: Mapping[str, Any],
    source_id: str,
    number: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    chosen = choose_number_support_row(sample)
    if not chosen:
        return None
    support, value = chosen
    candidate_id = f"ex_date_cand_{number:04d}"
    suggested = (
        f'Which document mentions "{value}" in the evidence needed to answer: '
        f'{str(sample.get("question", "")).strip()}'
    )
    return build_exact_rows(
        candidate_id=candidate_id,
        subtype="date_number_lookup",
        sample=sample,
        source_id=source_id,
        support=support,
        anchor_value=value,
        suggested_question=suggested,
    )


def build_exact_phrase_candidate(
    sample: Mapping[str, Any],
    source_id: str,
    number: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    chosen = choose_phrase_support_row(sample)
    if not chosen:
        return None
    support, phrase = chosen
    candidate_id = f"ex_phrase_cand_{number:04d}"
    suggested = (
        f'Which document contains the exact phrase "{phrase}" in evidence related to: '
        f'{str(sample.get("question", "")).strip()}'
    )
    return build_exact_rows(
        candidate_id=candidate_id,
        subtype="exact_phrase_lookup",
        sample=sample,
        source_id=source_id,
        support=support,
        anchor_value=phrase,
        suggested_question=suggested,
    )


def build_exact_rows(
    *,
    candidate_id: str,
    subtype: str,
    sample: Mapping[str, Any],
    source_id: str,
    support: dict[str, Any],
    anchor_value: str,
    suggested_question: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sheet_row = {
        "question_id": candidate_id,
        "source_hotpot_id": source_id,
        "subtype": subtype,
        "gold_doc_id": support["doc_id"],
        "gold_title": support["title"],
        "gold_sentence": support["text"],
        "anchor_value": anchor_value,
        "current_question": suggested_question,
        "final_question": "",
        "status": "todo",
        "notes": "",
    }
    pool_row = candidate_to_pool_row(
        candidate_id=candidate_id,
        task_type="exact_file_lookup",
        subtype=subtype,
        sample=sample,
        source_id=source_id,
        gold_rows=[support],
        candidate_question=suggested_question,
        metadata={"anchor_value": anchor_value, "requires_manual_review": True},
    )
    return sheet_row, pool_row


def candidate_to_pool_row(
    *,
    candidate_id: str,
    task_type: str,
    subtype: str,
    sample: Mapping[str, Any],
    source_id: str,
    gold_rows: list[dict[str, Any]],
    candidate_question: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "task_type": task_type,
        "subtype": subtype,
        "question": candidate_question,
        "gold_documents": sorted({row["doc_id"] for row in gold_rows}),
        "gold_sentences": [
            {
                "doc_id": row["doc_id"],
                "sent_id": int(row["sent_id"]),
                "text": row["text"],
            }
            for row in gold_rows
        ],
        "source_hotpot_id": source_id,
        "split": "final",
        "quality_checked": False,
        "metadata": {
            **metadata,
            "answer": str(sample.get("answer", "")),
            "support_titles": sorted({row["title"] for row in gold_rows}),
        },
    }


def assert_minimum_counts(sheets: dict[str, list[dict[str, Any]]]) -> None:
    exact_counts = Counter(row["subtype"] for row in sheets["exact"])
    required = {
        "semantic": SEMANTIC_CANDIDATES,
        "multi_hop": MULTI_HOP_CANDIDATES,
        "title_anchor": EXACT_CANDIDATES_PER_SUBTYPE,
        "date_number_lookup": EXACT_CANDIDATES_PER_SUBTYPE,
        "exact_phrase_lookup": EXACT_CANDIDATES_PER_SUBTYPE,
    }
    actual = {
        "semantic": len(sheets["semantic"]),
        "multi_hop": len(sheets["multi_hop"]),
        **exact_counts,
    }
    missing = {
        key: required_count - actual.get(key, 0)
        for key, required_count in required.items()
        if actual.get(key, 0) < required_count
    }
    if missing:
        raise RuntimeError(f"Not enough candidates: {missing}. Increase --scan-limit.")


def build_corpus(
    samples: list[tuple[int, Mapping[str, Any]]],
    *,
    candidate_source_ids: set[str],
    candidate_gold_doc_ids: set[str],
    target_docs: int,
) -> list[dict[str, Any]]:
    corpus_by_doc_id: dict[str, dict[str, Any]] = {}

    for fallback_index, sample in samples:
        source_id = stable_sample_id(sample, fallback_index)
        if source_id in candidate_source_ids:
            add_context_documents(
                corpus_by_doc_id,
                sample=sample,
                source_id=source_id,
                corpus_role="candidate_seed_noise",
                candidate_gold_doc_ids=candidate_gold_doc_ids,
            )

    for fallback_index, sample in samples:
        if len(corpus_by_doc_id) >= target_docs:
            break
        source_id = stable_sample_id(sample, fallback_index)
        if source_id in candidate_source_ids:
            continue
        add_context_documents(
            corpus_by_doc_id,
            sample=sample,
            source_id=source_id,
            corpus_role="noise",
            candidate_gold_doc_ids=candidate_gold_doc_ids,
            max_total=target_docs,
        )

    if len(corpus_by_doc_id) < target_docs:
        raise RuntimeError(
            f"Only built {len(corpus_by_doc_id)} corpus documents, target is {target_docs}."
        )

    return sorted(corpus_by_doc_id.values(), key=lambda row: row["doc_id"])[:target_docs]


def add_context_documents(
    corpus_by_doc_id: dict[str, dict[str, Any]],
    *,
    sample: Mapping[str, Any],
    source_id: str,
    corpus_role: str,
    candidate_gold_doc_ids: set[str],
    max_total: int | None = None,
) -> None:
    for context_row in extract_context(sample):
        if max_total is not None and len(corpus_by_doc_id) >= max_total:
            return
        doc_id = stable_doc_id(context_row["title"])
        role = "candidate_gold" if doc_id in candidate_gold_doc_ids else corpus_role
        existing = corpus_by_doc_id.get(doc_id)
        if existing is None:
            corpus_by_doc_id[doc_id] = {
                "doc_id": doc_id,
                "title": context_row["title"],
                "sentences": context_row["sentences"],
                "full_text": " ".join(context_row["sentences"]),
                "source_question_ids": [source_id],
                "metadata": {"corpus_role": role},
            }
        else:
            if source_id not in existing["source_question_ids"]:
                existing["source_question_ids"].append(source_id)
            if doc_id in candidate_gold_doc_ids:
                existing["metadata"]["corpus_role"] = "candidate_gold"
            elif existing["metadata"].get("corpus_role") == "noise" and role != "noise":
                existing["metadata"]["corpus_role"] = role


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final staging candidate data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scan-limit", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--semantic-candidates", type=int, default=SEMANTIC_CANDIDATES)
    parser.add_argument("--multi-hop-candidates", type=int, default=MULTI_HOP_CANDIDATES)
    parser.add_argument("--exact-candidates-per-subtype", type=int, default=EXACT_CANDIDATES_PER_SUBTYPE)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    global SEMANTIC_CANDIDATES, MULTI_HOP_CANDIDATES, EXACT_CANDIDATES_PER_SUBTYPE
    SEMANTIC_CANDIDATES = args.semantic_candidates
    MULTI_HOP_CANDIDATES = args.multi_hop_candidates
    EXACT_CANDIDATES_PER_SUBTYPE = args.exact_candidates_per_subtype

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    dataset = load_hotpot_dataset(
        dataset_name=str(config["data"]["hotpot_dataset_name"]),
        fallback_dataset_name=str(config["data"].get("hotpot_dataset_fallback_name", "")),
        config_name=str(config["data"]["hotpot_config"]),
        split=str(config["data"]["hotpot_split"]),
    )
    samples = collect_usable_samples(dataset, scan_limit=args.scan_limit, seed=seed)
    pool, sheets, candidate_source_ids = build_candidates(samples)
    candidate_gold_doc_ids = {
        doc_id
        for candidate in pool
        for doc_id in candidate["gold_documents"]
    }
    corpus = build_corpus(
        samples,
        candidate_source_ids=candidate_source_ids,
        candidate_gold_doc_ids=candidate_gold_doc_ids,
        target_docs=int(config["data"]["target_corpus_documents"]),
    )

    final_dir = Path(config["paths"]["corpus_path"]).parent
    staging_dir = final_dir / "staging"
    manual_dir = final_dir / "manual"
    staging_dir.mkdir(parents=True, exist_ok=True)
    manual_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(staging_dir / "candidate_pool.jsonl", pool)
    write_jsonl(staging_dir / "corpus.jsonl", corpus)
    write_csv(manual_dir / "semantic_rewrite_sheet.csv", sheets["semantic"])
    write_csv(manual_dir / "multi_hop_candidate_sheet.csv", sheets["multi_hop"])
    write_csv(manual_dir / "exact_lookup_sheet.csv", sheets["exact"])

    manifest = {
        "candidate_counts": {
            "semantic_fact": len(sheets["semantic"]),
            "multi_hop_relation": len(sheets["multi_hop"]),
            "exact_file_lookup": len(sheets["exact"]),
            "exact_subtypes": dict(Counter(row["subtype"] for row in sheets["exact"])),
        },
        "corpus_documents": len(corpus),
        "candidate_source_ids": len(candidate_source_ids),
        "scan_limit": args.scan_limit,
        "seed": seed,
        "outputs": {
            "candidate_pool": str(staging_dir / "candidate_pool.jsonl"),
            "staging_corpus": str(staging_dir / "corpus.jsonl"),
            "semantic_sheet": str(manual_dir / "semantic_rewrite_sheet.csv"),
            "multi_hop_sheet": str(manual_dir / "multi_hop_candidate_sheet.csv"),
            "exact_sheet": str(manual_dir / "exact_lookup_sheet.csv"),
        },
    }
    (staging_dir / "staging_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
