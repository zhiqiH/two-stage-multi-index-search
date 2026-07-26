from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import iter_jsonl


RUNTIME_FORBIDDEN_KEYS = {
    "answer",
    "corpus_role",
    "final_gold_question_ids",
    "gold_documents",
    "gold_sentences",
    "quality_checked",
    "source_hotpot_id",
    "source_question_ids",
    "staging_corpus_role",
}


class ContaminationError(RuntimeError):
    pass


def assert_no_private_keys(value: Any, *, context: str = "payload") -> None:
    for path, key in private_key_hits(value):
        raise ContaminationError(
            f"Private evaluation key {key!r} found at {context}{path}."
        )


def private_key_hits(value: Any, *, path: str = "") -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else f".{key_text}"
            if key_text in RUNTIME_FORBIDDEN_KEYS:
                hits.append((child_path, key_text))
            hits.extend(private_key_hits(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(private_key_hits(child, path=f"{path}[{index}]"))
    return hits


def validate_prediction_file(path: str | Path) -> int:
    count = 0
    for count, row in enumerate(iter_jsonl(path), start=1):
        assert_no_private_keys(row, context=f"{path}:{count}")
    return count
