from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import normalize_text
from .config import load_config
from .text_utils import STOPWORDS


PROMPT_VERSION = "final_data_llm_v1"
LOOKUP_TERMS = ("document", "file", "record", "entry", "article", "page", "corpus")
MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)
DATE_CONTEXT_PATTERN = re.compile(
    rf"\b(?:{MONTH_PATTERN})\s+\d{{1,2}},?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{4}}\b"
)
SEMANTIC_FORBIDDEN_TERMS = (
    "document",
    "file",
    "evidence",
    "sentence",
    "passage",
    "source",
    "text",
    "article",
    "according to",
    "rewrite",
    "fact",
)
MULTI_HOP_FORBIDDEN_TERMS = (
    "document",
    "file",
    "evidence",
    "sentence",
    "passage",
    "source",
    "article",
)


class LLMResponseError(RuntimeError):
    pass


class DeepInfraJSONClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {api_key_env}. Set it in the current shell before running LLM preparation."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai first: pip install -r requirements.txt") from exc
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def complete_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self._complete_json_once(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= 3:
                    break
                time.sleep(min(10, 2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    def _complete_json_once(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_object(content)
        usage = response.usage.model_dump() if response.usage is not None else {}
        return parsed, {"content": content, "usage": usage}


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise LLMResponseError("Empty model response.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError(f"No JSON object found in response: {text[:200]}")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise LLMResponseError("Model response JSON is not an object.")
    return value


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: PermissionError | None = None
    for attempt in range(1, 11):
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(5.0, 0.5 * attempt))
    assert last_error is not None
    raise last_error


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        file.write("\n")


def content_words(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in STOPWORDS
    }


def contains_normalized(haystack: str, needle: str) -> bool:
    normalized_haystack = normalize_text(haystack)
    normalized_needle = normalize_text(needle)
    return bool(normalized_needle) and normalized_needle in normalized_haystack


def has_forbidden_term(text: str, terms: Sequence[str]) -> bool:
    lower = text.lower()
    for term in terms:
        term = term.lower()
        if " " in term:
            if term in lower:
                return True
            continue
        if re.search(rf"\b{re.escape(term)}\b", lower):
            return True
    return False


def preferred_anchor_value(row: Mapping[str, str]) -> str:
    subtype = row.get("subtype", "")
    anchor = row.get("anchor_value", "").strip()
    sentence = row.get("gold_sentence", "")
    if subtype != "date_number_lookup" or not anchor:
        return anchor
    for match in DATE_CONTEXT_PATTERN.finditer(sentence):
        candidate = match.group(0).strip()
        if contains_normalized(candidate, anchor):
            return candidate
    return anchor


def exact_candidate_preflight_issues(row: Mapping[str, str]) -> list[str]:
    subtype = row.get("subtype", "")
    title = row.get("gold_title", "")
    anchor = row.get("anchor_value", "")
    preferred_anchor = preferred_anchor_value(row)
    issues = []
    if subtype in {"date_number_lookup", "exact_phrase_lookup"}:
        if contains_normalized(anchor, title) or contains_normalized(preferred_anchor, title):
            issues.append("anchor_leaks_gold_title")
    return issues


def question_basic_issues(question: str, *, lookup_style: bool) -> list[str]:
    question = question.strip()
    lower = question.lower()
    issues = []
    if not question:
        issues.append("empty_question")
        return issues
    if not question.endswith("?"):
        issues.append("missing_question_mark")
    if len(question) < 25:
        issues.append("question_too_short")
    if len(question) > 240:
        issues.append("question_too_long")
    if "\n" in question or "\r" in question:
        issues.append("contains_newline")
    if lower.startswith(("rewrite ", "write ", "generate ")):
        issues.append("instruction_like_question")
    if not lookup_style and has_forbidden_term(lower, MULTI_HOP_FORBIDDEN_TERMS):
        issues.append("contains_file_or_evidence_language")
    return issues


def semantic_deterministic_issues(row: Mapping[str, str], question: str) -> list[str]:
    issues = question_basic_issues(question, lookup_style=False)
    lower = question.lower()
    if has_forbidden_term(lower, SEMANTIC_FORBIDDEN_TERMS):
        issues.append("semantic_question_uses_file_or_rewrite_language")
    if is_title_definition_question(row, question):
        issues.append("semantic_question_is_title_definition")

    q_words = content_words(question)
    s_words = content_words(row.get("gold_sentence", ""))
    if len(q_words) >= 5 and s_words:
        overlap = len(q_words & s_words) / max(1, len(q_words))
        if overlap >= 0.78:
            issues.append("question_copies_gold_sentence_too_closely")
        if overlap <= 0.10:
            issues.append("question_has_too_little_overlap_with_gold_sentence")
    return sorted(set(issues))


def is_title_definition_question(row: Mapping[str, str], question: str) -> bool:
    title = normalize_text(row.get("gold_title", ""))
    if not title:
        return False
    normalized_question = normalize_text(question)
    definition_prefixes = (
        f"what is {title}",
        f"what is the {title}",
        f"what was {title}",
        f"what was the {title}",
        f"who is {title}",
        f"who was {title}",
    )
    if not normalized_question.startswith(definition_prefixes):
        return False
    remaining_words = content_words(normalized_question) - content_words(title)
    return len(remaining_words) <= 1


def exact_deterministic_issues(row: Mapping[str, str], question: str) -> list[str]:
    subtype = row.get("subtype", "")
    anchor = row.get("anchor_value", "")
    required_anchor = preferred_anchor_value(row)
    title = row.get("gold_title", "")
    sentence = row.get("gold_sentence", "")
    lower = question.lower()
    issues = question_basic_issues(question, lookup_style=True)

    if "evidence needed to answer" in lower or "evidence related to" in lower:
        issues.append("boilerplate_evidence_template")
    if not any(term in lower for term in LOOKUP_TERMS):
        issues.append("missing_file_lookup_language")
    if not contains_normalized(question, required_anchor):
        issues.append("missing_required_anchor")
    if subtype != "title_anchor" and anchor and not contains_normalized(sentence, anchor):
        issues.append("anchor_not_in_gold_sentence")

    if subtype == "title_anchor":
        if not contains_normalized(question, title):
            issues.append("title_anchor_missing_gold_title")
    elif subtype in {"date_number_lookup", "exact_phrase_lookup"}:
        if contains_normalized(question, title):
            issues.append("non_title_lookup_leaks_gold_title")
    else:
        issues.append(f"unknown_exact_subtype:{subtype}")

    if subtype == "exact_phrase_lookup" and '"' not in question:
        issues.append("exact_phrase_not_quoted")
    return sorted(set(issues))


def multi_hop_deterministic_issues(row: Mapping[str, str]) -> list[str]:
    question = row.get("question", "")
    answer = row.get("answer", "")
    doc_ids = [value for value in row.get("gold_doc_ids", "").split("|") if value.strip()]
    sentences = [value for value in row.get("gold_sentences", "").split("||") if value.strip()]
    issues = question_basic_issues(question, lookup_style=False)
    if len(doc_ids) != 2:
        issues.append("multi_hop_requires_exactly_two_gold_documents")
    if len(sentences) < 2:
        issues.append("multi_hop_requires_at_least_two_gold_sentences")
    if not answer.strip():
        issues.append("empty_answer")
    elif contains_normalized(question, answer):
        issues.append("answer_leaked_in_question")
    if not re.search(r"\b(who|what|which|where|when|whose|that|from|by|of)\b", question.lower()):
        issues.append("question_has_weak_relation_signal")
    return sorted(set(issues))


def verifier_accepts_semantic(payload: Mapping[str, Any]) -> bool:
    payload = verifier_decision(payload)
    return (
        str(payload.get("verdict", "")).lower() == "accept"
        and bool(payload.get("answerable_from_gold_sentence"))
        and bool(payload.get("natural_question"))
        and not bool(payload.get("requires_file_or_title_lookup"))
        and not bool(payload.get("introduces_new_fact"))
        and not bool(payload.get("copies_sentence_too_closely"))
    )


def verifier_accepts_exact(payload: Mapping[str, Any]) -> bool:
    payload = verifier_decision(payload)
    return (
        str(payload.get("verdict", "")).lower() == "accept"
        and bool(payload.get("answerable_from_gold_sentence"))
        and bool(payload.get("matches_subtype"))
        and bool(payload.get("uses_required_anchor"))
        and bool(payload.get("natural_lookup_question"))
        and not bool(payload.get("leaks_title_when_forbidden"))
    )


def verifier_accepts_multi_hop(payload: Mapping[str, Any]) -> bool:
    payload = verifier_decision(payload)
    return (
        str(payload.get("verdict", "")).lower() == "accept"
        and bool(payload.get("requires_two_documents"))
        and bool(payload.get("both_gold_sentences_relevant"))
        and bool(payload.get("answerable_from_given_evidence"))
        and bool(payload.get("natural_question"))
        and bool(payload.get("answer_not_leaked"))
    )


def verifier_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    decision = dict(payload)
    nested = payload.get("decision_schema")
    if isinstance(nested, Mapping) and any(
        key in nested
        for key in (
            "verdict",
            "answerable_from_gold_sentence",
            "answerable_from_given_evidence",
            "matches_subtype",
            "requires_two_documents",
        )
    ):
        decision.update(dict(nested))
    return decision


def reason_from_verifier(payload: Mapping[str, Any]) -> str:
    payload = verifier_decision(payload)
    for key in ("reason", "revision_instruction", "notes"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def semantic_generator_messages(
    row: Mapping[str, str],
    *,
    previous_question: str = "",
    feedback: str = "",
) -> list[dict[str, str]]:
    user = {
        "gold_title": row.get("gold_title", ""),
        "gold_sentence": row.get("gold_sentence", ""),
        "current_question": row.get("current_question", ""),
        "previous_question": previous_question,
        "feedback": feedback,
        "rules": [
            "Write exactly one natural English question.",
            "The question must be answerable from the gold sentence alone.",
            "Do not mention document, file, evidence, sentence, passage, source, article, text, or rewrite.",
            "Do not copy the gold sentence with only small word changes.",
            "Do not introduce facts that are not present in the gold sentence.",
            "Ask for a concrete answer, not for a summary of the sentence.",
            "Do not ask only 'Who is [title]?' or 'What is [title]?'; ask about a concrete attribute, relation, role, date, location, or ownership.",
            "Return strict JSON only: {\"question\": \"...\", \"rationale\": \"...\"}.",
        ],
    }
    return [
        {
            "role": "system",
            "content": "You are preparing high-quality retrieval evaluation questions. Return only valid JSON.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def semantic_verifier_messages(row: Mapping[str, str], candidate_question: str) -> list[dict[str, str]]:
    user = {
        "gold_title": row.get("gold_title", ""),
        "gold_sentence": row.get("gold_sentence", ""),
        "candidate_question": candidate_question,
        "decision_schema": {
            "answerable_from_gold_sentence": "boolean",
            "natural_question": "boolean",
            "requires_file_or_title_lookup": "boolean",
            "introduces_new_fact": "boolean",
            "copies_sentence_too_closely": "boolean",
            "verdict": "accept|repair|reject",
            "reason": "short string",
            "revision_instruction": "short string",
        },
    }
    return [
        {
            "role": "system",
            "content": "You are an independent dataset quality verifier. Return only valid JSON.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def exact_generator_messages(
    row: Mapping[str, str],
    *,
    previous_question: str = "",
    feedback: str = "",
) -> list[dict[str, str]]:
    subtype = row.get("subtype", "")
    preferred_anchor = preferred_anchor_value(row)
    subtype_rules = {
        "title_anchor": [
            "The question must include the exact gold title, preferably in quotation marks.",
            "The question must explicitly refer to the document or file.",
            "Ask for a specific detail answerable from the gold sentence.",
            "A good pattern is: In the \"TITLE\" document, what ...?",
        ],
        "date_number_lookup": [
            "The question must include the preferred date/number anchor exactly.",
            "The question must not include the gold title.",
            "Ask which document or record contains that anchor and the described fact.",
        ],
        "exact_phrase_lookup": [
            "The question must include the preferred exact phrase anchor in quotation marks.",
            "The question must not include the gold title.",
            "Ask which document or record contains that exact phrase and the described fact.",
        ],
    }
    user = {
        "subtype": subtype,
        "gold_title": row.get("gold_title", ""),
        "gold_sentence": row.get("gold_sentence", ""),
        "anchor_value": row.get("anchor_value", ""),
        "preferred_anchor_value": preferred_anchor,
        "current_question": row.get("current_question", ""),
        "previous_question": previous_question,
        "feedback": feedback,
        "rules": [
            "Write exactly one natural English exact/file lookup question.",
            "The target gold document must be identifiable from the question.",
            "Do not introduce facts that are not present in the gold sentence.",
            "Do not paste the current question.",
            "Do not use the phrases 'evidence needed to answer' or 'evidence related to'.",
            "Prefer a compact standalone question that a retrieval system could run directly.",
            "Return strict JSON only: {\"question\": \"...\", \"rationale\": \"...\"}.",
            *subtype_rules.get(subtype, ["Match the requested subtype."]),
        ],
    }
    return [
        {
            "role": "system",
            "content": "You are preparing exact/file lookup retrieval evaluation questions. Return only valid JSON.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def exact_verifier_messages(row: Mapping[str, str], candidate_question: str) -> list[dict[str, str]]:
    user = {
        "subtype": row.get("subtype", ""),
        "gold_title": row.get("gold_title", ""),
        "gold_sentence": row.get("gold_sentence", ""),
        "anchor_value": row.get("anchor_value", ""),
        "preferred_anchor_value": preferred_anchor_value(row),
        "candidate_question": candidate_question,
        "decision_schema": {
            "answerable_from_gold_sentence": "boolean",
            "matches_subtype": "boolean",
            "uses_required_anchor": "boolean",
            "leaks_title_when_forbidden": "boolean",
            "natural_lookup_question": "boolean",
            "verdict": "accept|repair|reject",
            "reason": "short string",
            "revision_instruction": "short string",
        },
    }
    return [
        {
            "role": "system",
            "content": "You are an independent exact/file lookup dataset verifier. Return only valid JSON.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def multi_hop_verifier_messages(row: Mapping[str, str]) -> list[dict[str, str]]:
    user = {
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "gold_doc_ids": row.get("gold_doc_ids", ""),
        "gold_titles": row.get("gold_titles", ""),
        "gold_sentences": row.get("gold_sentences", ""),
        "rules": [
            "Do not rewrite the question.",
            "Verify whether the question needs both gold documents/sentences.",
            "Reject if it can be answered from only one gold document.",
            "Reject if the answer is leaked in the question.",
            "Reject if the evidence is insufficient or unrelated.",
        ],
        "decision_schema": {
            "requires_two_documents": "boolean",
            "both_gold_sentences_relevant": "boolean",
            "answerable_from_given_evidence": "boolean",
            "answer_not_leaked": "boolean",
            "natural_question": "boolean",
            "verdict": "accept|needs_review|reject",
            "reason": "short string",
        },
    }
    return [
        {
            "role": "system",
            "content": "You are an independent multi-hop retrieval dataset verifier. Return only valid JSON.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def call_and_log(
    *,
    client: DeepInfraJSONClient,
    log_path: Path,
    task: str,
    question_id: str,
    stage: str,
    attempt: int,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        payload, raw = client.complete_json(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        append_jsonl(
            log_path,
            {
                "prompt_version": PROMPT_VERSION,
                "task": task,
                "question_id": question_id,
                "stage": stage,
                "attempt": attempt,
                "model": model,
                "ok": True,
                "elapsed_seconds": round(time.time() - started, 3),
                "usage": raw.get("usage", {}),
                "payload": payload,
            },
        )
        return payload
    except Exception as exc:
        append_jsonl(
            log_path,
            {
                "prompt_version": PROMPT_VERSION,
                "task": task,
                "question_id": question_id,
                "stage": stage,
                "attempt": attempt,
                "model": model,
                "ok": False,
                "elapsed_seconds": round(time.time() - started, 3),
                "error": str(exc),
            },
        )
        raise


def process_semantic_row(
    row: Mapping[str, str],
    *,
    client: DeepInfraJSONClient,
    settings: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    generator_model = str(settings["generator_model"])
    verifier_model = str(settings["verifier_model"])
    generator_temperature = float(settings.get("generator_temperature", 0.2))
    verifier_temperature = float(settings.get("verifier_temperature", 0.0))
    max_tokens = int(settings.get("max_tokens", 700))
    max_repair_attempts = int(settings.get("max_repair_attempts", 2))

    output = dict(row)
    best_question = str(row.get("rewritten_question", "")).strip()
    feedback = ""
    verifier_payload: dict[str, Any] = {}
    issues: list[str] = []

    for attempt in range(1, max_repair_attempts + 2):
        generator_payload = call_and_log(
            client=client,
            log_path=log_path,
            task="semantic_fact",
            question_id=str(row["question_id"]),
            stage="generate",
            attempt=attempt,
            model=generator_model,
            messages=semantic_generator_messages(
                row,
                previous_question=best_question,
                feedback=feedback,
            ),
            temperature=generator_temperature,
            max_tokens=max_tokens,
        )
        best_question = str(generator_payload.get("question", "")).strip()
        issues = semantic_deterministic_issues(row, best_question)
        if issues:
            feedback = "Deterministic checks failed: " + ", ".join(issues)
            append_jsonl(
                log_path,
                {
                    "prompt_version": PROMPT_VERSION,
                    "task": "semantic_fact",
                    "question_id": row["question_id"],
                    "stage": "deterministic_check",
                    "attempt": attempt,
                    "ok": False,
                    "issues": issues,
                },
            )
            continue

        verifier_payload = call_and_log(
            client=client,
            log_path=log_path,
            task="semantic_fact",
            question_id=str(row["question_id"]),
            stage="verify",
            attempt=attempt,
            model=verifier_model,
            messages=semantic_verifier_messages(row, best_question),
            temperature=verifier_temperature,
            max_tokens=max_tokens,
        )
        if verifier_accepts_semantic(verifier_payload):
            output.update(
                {
                    "rewritten_question": best_question,
                    "status": "accepted",
                    "notes": "",
                    "llm_verdict": "accept",
                    "llm_attempts": attempt,
                    "deterministic_issues": "",
                    "verifier_reason": reason_from_verifier(verifier_payload),
                    "generator_model": generator_model,
                    "verifier_model": verifier_model,
                }
            )
            return output
        feedback = reason_from_verifier(verifier_payload)

    output.update(
        {
            "rewritten_question": best_question,
            "status": "needs_review",
            "notes": feedback,
            "llm_verdict": str(verifier_decision(verifier_payload).get("verdict", "repair")),
            "llm_attempts": max_repair_attempts + 1,
            "deterministic_issues": "|".join(issues),
            "verifier_reason": reason_from_verifier(verifier_payload) if verifier_payload else feedback,
            "generator_model": generator_model,
            "verifier_model": verifier_model,
        }
    )
    return output


def process_exact_row(
    row: Mapping[str, str],
    *,
    client: DeepInfraJSONClient,
    settings: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    generator_model = str(settings["generator_model"])
    verifier_model = str(settings["verifier_model"])
    generator_temperature = float(settings.get("generator_temperature", 0.2))
    verifier_temperature = float(settings.get("verifier_temperature", 0.0))
    max_tokens = int(settings.get("max_tokens", 700))
    max_repair_attempts = int(settings.get("max_repair_attempts", 2))

    output = dict(row)
    preflight_issues = exact_candidate_preflight_issues(row)
    if preflight_issues:
        append_jsonl(
            log_path,
            {
                "prompt_version": PROMPT_VERSION,
                "task": "exact_file_lookup",
                "question_id": row["question_id"],
                "stage": "preflight_check",
                "attempt": 0,
                "ok": False,
                "issues": preflight_issues,
            },
        )
        output.update(
            {
                "status": "drop",
                "notes": "Candidate cannot satisfy subtype constraints: " + ", ".join(preflight_issues),
                "llm_verdict": "preflight_drop",
                "llm_attempts": 0,
                "deterministic_issues": "|".join(preflight_issues),
                "verifier_reason": "",
                "generator_model": generator_model,
                "verifier_model": verifier_model,
            }
        )
        return output

    best_question = str(row.get("final_question", "")).strip()
    feedback = ""
    verifier_payload: dict[str, Any] = {}
    issues: list[str] = []

    for attempt in range(1, max_repair_attempts + 2):
        generator_payload = call_and_log(
            client=client,
            log_path=log_path,
            task="exact_file_lookup",
            question_id=str(row["question_id"]),
            stage="generate",
            attempt=attempt,
            model=generator_model,
            messages=exact_generator_messages(
                row,
                previous_question=best_question,
                feedback=feedback,
            ),
            temperature=generator_temperature,
            max_tokens=max_tokens,
        )
        best_question = str(generator_payload.get("question", "")).strip()
        issues = exact_deterministic_issues(row, best_question)
        if issues:
            feedback = "Deterministic checks failed: " + ", ".join(issues)
            append_jsonl(
                log_path,
                {
                    "prompt_version": PROMPT_VERSION,
                    "task": "exact_file_lookup",
                    "question_id": row["question_id"],
                    "stage": "deterministic_check",
                    "attempt": attempt,
                    "ok": False,
                    "issues": issues,
                },
            )
            continue

        verifier_payload = call_and_log(
            client=client,
            log_path=log_path,
            task="exact_file_lookup",
            question_id=str(row["question_id"]),
            stage="verify",
            attempt=attempt,
            model=verifier_model,
            messages=exact_verifier_messages(row, best_question),
            temperature=verifier_temperature,
            max_tokens=max_tokens,
        )
        if verifier_accepts_exact(verifier_payload):
            output.update(
                {
                    "final_question": best_question,
                    "status": "accepted",
                    "notes": "",
                    "llm_verdict": "accept",
                    "llm_attempts": attempt,
                    "deterministic_issues": "",
                    "verifier_reason": reason_from_verifier(verifier_payload),
                    "generator_model": generator_model,
                    "verifier_model": verifier_model,
                }
            )
            return output
        feedback = reason_from_verifier(verifier_payload)

    output.update(
        {
            "final_question": best_question,
            "status": "needs_review",
            "notes": feedback,
            "llm_verdict": str(verifier_decision(verifier_payload).get("verdict", "repair")),
            "llm_attempts": max_repair_attempts + 1,
            "deterministic_issues": "|".join(issues),
            "verifier_reason": reason_from_verifier(verifier_payload) if verifier_payload else feedback,
            "generator_model": generator_model,
            "verifier_model": verifier_model,
        }
    )
    return output


def process_multi_hop_row(
    row: Mapping[str, str],
    *,
    client: DeepInfraJSONClient,
    settings: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    verifier_model = str(settings["verifier_model"])
    verifier_temperature = float(settings.get("verifier_temperature", 0.0))
    max_tokens = int(settings.get("max_tokens", 700))
    output = dict(row)
    issues = multi_hop_deterministic_issues(row)
    if issues:
        append_jsonl(
            log_path,
            {
                "prompt_version": PROMPT_VERSION,
                "task": "multi_hop_relation",
                "question_id": row["question_id"],
                "stage": "deterministic_check",
                "attempt": 1,
                "ok": False,
                "issues": issues,
            },
        )
        output.update(
            {
                "status": "needs_review",
                "notes": "Deterministic checks failed: " + ", ".join(issues),
                "llm_verdict": "",
                "llm_attempts": 0,
                "deterministic_issues": "|".join(issues),
                "verifier_reason": "",
                "verifier_model": verifier_model,
            }
        )
        return output

    verifier_payload = call_and_log(
        client=client,
        log_path=log_path,
        task="multi_hop_relation",
        question_id=str(row["question_id"]),
        stage="verify",
        attempt=1,
        model=verifier_model,
        messages=multi_hop_verifier_messages(row),
        temperature=verifier_temperature,
        max_tokens=max_tokens,
    )
    decision = verifier_decision(verifier_payload)
    if verifier_accepts_multi_hop(decision):
        status = "accepted"
    elif str(decision.get("verdict", "")).lower() == "reject":
        status = "drop"
    else:
        status = "needs_review"

    output.update(
        {
            "status": status,
            "notes": "" if status == "accepted" else reason_from_verifier(decision),
            "llm_verdict": str(decision.get("verdict", "")),
            "llm_attempts": 1,
            "deterministic_issues": "",
            "verifier_reason": reason_from_verifier(decision),
            "verifier_model": verifier_model,
        }
    )
    return output


def limit_rows(rows: Sequence[dict[str, str]], limit: int | None, *, balance_by: str | None = None) -> list[dict[str, str]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)
    if balance_by is None:
        return list(rows[:limit])

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(balance_by, "")].append(row)

    selected: list[dict[str, str]] = []
    offsets = defaultdict(int)
    keys = sorted(groups)
    while len(selected) < limit and keys:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                offsets[key] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def with_extra_fields(rows: Sequence[Mapping[str, Any]], extra_fields: Sequence[str]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def processing_error_row(
    row: Mapping[str, Any],
    *,
    task: str,
    exc: Exception,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "status": "needs_review",
            "notes": f"LLM processing error: {type(exc).__name__}: {exc}",
            "llm_verdict": "error",
            "llm_attempts": "",
            "deterministic_issues": "",
            "verifier_reason": "",
            "generator_model": settings.get("generator_model", "") if task != "multi_hop_relation" else "",
            "verifier_model": settings.get("verifier_model", ""),
        }
    )
    return output


def process_rows_incrementally(
    *,
    rows: Sequence[dict[str, str]],
    output_path: str | Path,
    extra_fields: Sequence[str],
    log_path: Path,
    task: str,
    settings: Mapping[str, Any],
    processor: Any,
    existing_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = list(existing_rows or [])
    processed_ids = {str(row.get("question_id", "")) for row in processed}
    for row in rows:
        if str(row.get("question_id", "")) in processed_ids:
            continue
        try:
            processed_row = processor(row)
        except Exception as exc:
            append_jsonl(
                log_path,
                {
                    "prompt_version": PROMPT_VERSION,
                    "task": task,
                    "question_id": row.get("question_id", ""),
                    "stage": "row_error",
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            processed_row = processing_error_row(
                row,
                task=task,
                exc=exc,
                settings=settings,
            )
        processed.append(processed_row)
        processed_ids.add(str(row.get("question_id", "")))
        write_csv_rows(output_path, processed, with_extra_fields(processed, extra_fields))
    return processed


def load_resume_rows(output_path: str | Path, *, resume: bool) -> list[dict[str, Any]]:
    path = Path(output_path)
    if not resume or not path.exists():
        return []
    return [dict(row) for row in read_csv_rows(path)]


def parse_tasks(raw: str) -> set[str]:
    aliases = {
        "semantic": "semantic",
        "semantic_fact": "semantic",
        "exact": "exact",
        "exact_file_lookup": "exact",
        "multi": "multi_hop",
        "multi_hop": "multi_hop",
        "multi_hop_relation": "multi_hop",
        "all": "all",
    }
    tasks = set()
    for token in raw.split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized not in aliases:
            raise ValueError(f"Unknown task: {token}")
        tasks.add(aliases[normalized])
    if "all" in tasks or not tasks:
        return {"semantic", "exact", "multi_hop"}
    return tasks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use DeepInfra LLMs to prepare and verify final dataset candidate sheets."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tasks", default="all", help="Comma-separated: semantic,exact,multi_hop,all")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum rows per selected sheet for a cost-controlled run. Use --all-rows for full processing.",
    )
    parser.add_argument("--all-rows", action="store_true", help="Process all rows in selected sheets.")
    parser.add_argument("--resume", action="store_true", help="Keep existing output rows and process only missing question_ids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print planned work without API calls.")
    parser.add_argument("--semantic-in", default="data/final/manual/semantic_rewrite_sheet.csv")
    parser.add_argument("--exact-in", default="data/final/manual/exact_lookup_sheet.csv")
    parser.add_argument("--multi-hop-in", default="data/final/manual/multi_hop_candidate_sheet.csv")
    parser.add_argument("--semantic-out", default="data/final/manual/semantic_rewrite_sheet_llm.csv")
    parser.add_argument("--exact-out", default="data/final/manual/exact_lookup_sheet_llm.csv")
    parser.add_argument("--multi-hop-out", default="data/final/manual/multi_hop_candidate_sheet_llm.csv")
    parser.add_argument("--log-out", default="data/final/staging/llm_generation_log.jsonl")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    settings = dict(config.get("data_llm", {}))
    if not settings:
        raise RuntimeError("Missing data_llm settings in config.yaml.")

    tasks = parse_tasks(args.tasks)
    limit = None if args.all_rows else args.limit

    semantic_rows = read_csv_rows(args.semantic_in) if "semantic" in tasks else []
    exact_rows = read_csv_rows(args.exact_in) if "exact" in tasks else []
    multi_rows = read_csv_rows(args.multi_hop_in) if "multi_hop" in tasks else []

    semantic_rows = limit_rows(semantic_rows, limit)
    exact_rows = limit_rows(exact_rows, limit, balance_by="subtype")
    multi_rows = limit_rows(multi_rows, limit)

    plan = {
        "tasks": sorted(tasks),
        "limit": "all" if limit is None else limit,
        "rows": {
            "semantic_fact": len(semantic_rows),
            "exact_file_lookup": len(exact_rows),
            "multi_hop_relation": len(multi_rows),
        },
        "outputs": {
            "semantic": args.semantic_out,
            "exact": args.exact_out,
            "multi_hop": args.multi_hop_out,
            "log": args.log_out,
        },
        "generator_model": settings.get("generator_model"),
        "verifier_model": settings.get("verifier_model"),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    client = DeepInfraJSONClient(
        base_url=str(settings["base_url"]),
        api_key_env=str(settings["api_key_env"]),
        timeout_seconds=float(settings.get("timeout_seconds", 60)),
    )
    log_path = Path(args.log_out)

    summary: dict[str, Any] = {"plan": plan, "status_counts": {}}

    if semantic_rows:
        existing_rows = load_resume_rows(args.semantic_out, resume=bool(args.resume))
        processed = process_rows_incrementally(
            rows=semantic_rows,
            output_path=args.semantic_out,
            extra_fields=[
                "llm_verdict",
                "llm_attempts",
                "deterministic_issues",
                "verifier_reason",
                "generator_model",
                "verifier_model",
            ],
            log_path=log_path,
            task="semantic_fact",
            settings=settings,
            existing_rows=existing_rows,
            processor=lambda row: process_semantic_row(
                row,
                client=client,
                settings=settings,
                log_path=log_path,
            ),
        )
        summary["status_counts"]["semantic_fact"] = status_counts(processed)

    if exact_rows:
        existing_rows = load_resume_rows(args.exact_out, resume=bool(args.resume))
        processed = process_rows_incrementally(
            rows=exact_rows,
            output_path=args.exact_out,
            extra_fields=[
                "llm_verdict",
                "llm_attempts",
                "deterministic_issues",
                "verifier_reason",
                "generator_model",
                "verifier_model",
            ],
            log_path=log_path,
            task="exact_file_lookup",
            settings=settings,
            existing_rows=existing_rows,
            processor=lambda row: process_exact_row(
                row,
                client=client,
                settings=settings,
                log_path=log_path,
            ),
        )
        summary["status_counts"]["exact_file_lookup"] = status_counts(processed)

    if multi_rows:
        existing_rows = load_resume_rows(args.multi_hop_out, resume=bool(args.resume))
        processed = process_rows_incrementally(
            rows=multi_rows,
            output_path=args.multi_hop_out,
            extra_fields=[
                "llm_verdict",
                "llm_attempts",
                "deterministic_issues",
                "verifier_reason",
                "verifier_model",
            ],
            log_path=log_path,
            task="multi_hop_relation",
            settings=settings,
            existing_rows=existing_rows,
            processor=lambda row: process_multi_hop_row(
                row,
                client=client,
                settings=settings,
                log_path=log_path,
            ),
        )
        summary["status_counts"]["multi_hop_relation"] = status_counts(processed)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def cli_main(argv: Iterable[str] | None = None) -> int:
    try:
        return main(argv)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
