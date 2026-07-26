from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .text_utils import numbers_and_dates, quoted_phrases, simple_entities


@dataclass(slots=True)
class RouteDecision:
    tool: str
    reason: str
    signals: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FILE_CUES = {
    "article",
    "document",
    "entry",
    "evidence",
    "file",
    "page",
    "record",
    "source",
}
LOOKUP_CUES = {
    "contains",
    "find",
    "identify",
    "locate",
    "mentions",
    "named",
    "retrieve",
    "states",
    "titled",
    "title",
}
WEAK_EXACT_CUES = {
    "date",
    "number",
    "older",
    "year",
    "younger",
}
GRAPH_CUES = {
    "acted",
    "actor",
    "actress",
    "adopted",
    "album",
    "appeared",
    "band",
    "born",
    "capital",
    "city",
    "country",
    "developed",
    "directed",
    "director",
    "died",
    "film",
    "founded",
    "founder",
    "friend",
    "genre",
    "held",
    "located",
    "movie",
    "played",
    "produced",
    "screenwriter",
    "served",
    "starred",
    "starring",
    "whose",
    "wrote",
    "written",
}
BRIDGE_PATTERNS = [
    r"\bwho (?:was|is|were|are|wrote|directed|founded|played|starred)",
    r"\bwhose\b",
    r"\bwhich .*\b(was|is|were|are|has|had)\b",
    r"\bperson who\b",
    r"\bone of (?:his|her|their|the)\b",
    r"\bin which\b",
    r"\bwhere was\b",
    r"\bwhat .* by .*\b",
]


def route_question(question: str) -> RouteDecision:
    question_lower = question.lower()
    tokens = set(re.findall(r"[a-z0-9]+", question_lower))
    entities = simple_entities(question)
    phrases = quoted_phrases(question)
    numeric_anchors = numbers_and_dates(question)

    exact_title = has_exact_title_pattern(question)
    claim_evidence = is_claim_evidence_question(question)
    file_cues = sorted(tokens.intersection(FILE_CUES))
    lookup_cues = sorted(tokens.intersection(LOOKUP_CUES))
    exact_cues = sorted(tokens.intersection(WEAK_EXACT_CUES))
    graph_cues = sorted(tokens.intersection(GRAPH_CUES))
    bridge_hits = [pattern for pattern in BRIDGE_PATTERNS if re.search(pattern, question_lower)]

    file_score = score_file_signal(
        exact_title=exact_title,
        phrases=phrases,
        numeric_anchors=numeric_anchors,
        file_cues=file_cues,
        lookup_cues=lookup_cues,
        exact_cues=exact_cues,
    )
    graph_score = score_graph_signal(
        entities=entities,
        graph_cues=graph_cues,
        bridge_hits=bridge_hits,
        claim_evidence=claim_evidence,
    )
    dense_score = 1.0 + (2.0 if claim_evidence else 0.0)
    if file_score < 3.0 and graph_score < 3.0:
        dense_score += 1.0

    signals = {
        "entities": entities,
        "quoted_phrases": phrases,
        "numbers_dates": numeric_anchors,
        "file_cues": file_cues,
        "lookup_cues": lookup_cues,
        "weak_exact_cues": exact_cues,
        "graph_cues": graph_cues,
        "bridge_patterns": bridge_hits,
        "exact_title_signal": exact_title,
        "claim_evidence_signal": claim_evidence,
        "file_score": file_score,
        "graph_score": graph_score,
        "dense_score": dense_score,
    }

    if claim_evidence and file_score < 4.0:
        return RouteDecision(tool="dense", reason="semantic_claim_signal", signals=signals)

    if file_score >= 4.0 and file_score >= graph_score:
        return RouteDecision(tool="file_fts", reason="strong_file_lookup_signal", signals=signals)

    if graph_score >= 3.0:
        return RouteDecision(tool="graph_path", reason="entity_relation_or_bridge_signal", signals=signals)

    if file_score >= 3.0:
        return RouteDecision(tool="file_fts", reason="moderate_exact_lookup_signal", signals=signals)

    return RouteDecision(tool="dense", reason="default_semantic_signal", signals=signals)


def score_file_signal(
    *,
    exact_title: bool,
    phrases: list[str],
    numeric_anchors: list[str],
    file_cues: list[str],
    lookup_cues: list[str],
    exact_cues: list[str],
) -> float:
    score = 0.0
    if exact_title:
        score += 5.0
    if phrases:
        score += 2.2
    if file_cues:
        score += 2.0
    if lookup_cues:
        score += 1.5
    if numeric_anchors and (file_cues or lookup_cues or exact_cues):
        score += 1.3
    if exact_cues and (file_cues or lookup_cues):
        score += 0.8
    return score


def score_graph_signal(
    *,
    entities: list[str],
    graph_cues: list[str],
    bridge_hits: list[str],
    claim_evidence: bool,
) -> float:
    if claim_evidence:
        return 0.0
    score = 0.0
    if len(entities) >= 2:
        score += 1.6
    elif len(entities) == 1:
        score += 0.6
    if graph_cues:
        score += min(2.4, 0.8 * len(graph_cues))
    if bridge_hits:
        score += min(2.4, 1.2 * len(bridge_hits))
    return score


def is_claim_evidence_question(question: str) -> bool:
    question_lower = question.lower()
    return (
        "evidence for this claim" in question_lower
        or "provides evidence for this claim" in question_lower
    )


def has_exact_title_pattern(question: str) -> bool:
    return bool(
        re.search(r'\bfind the document titled\s+"[^"]+"', question, flags=re.IGNORECASE)
        or re.search(r'\bdocument titled\s+"[^"]+"', question, flags=re.IGNORECASE)
        or re.search(r'\b(?:file|document|article|entry) (?:named|called|titled)\s+"[^"]+"', question, flags=re.IGNORECASE)
    )


def has_strong_file_signal(question: str) -> bool:
    decision = route_question(question)
    return bool(decision.signals["file_score"] >= 4.0)


def has_graph_signal(question: str) -> bool:
    decision = route_question(question)
    return bool(decision.signals["graph_score"] >= 3.0)
