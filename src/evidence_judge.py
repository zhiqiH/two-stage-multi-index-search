from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .common import SearchOutput, normalize_text
from .router import route_question
from .text_utils import keywords, normalize_entity, numbers_and_dates, quoted_phrases


@dataclass(slots=True)
class JudgeDecision:
    sufficient: bool
    missing_information: str
    next_tool: str | None
    revised_query: str
    confidence: float
    strategy: str = "heuristic_coverage_protected_final"
    signals: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def heuristic_judge(
    *,
    question: str,
    first_output: SearchOutput,
    used_tool: str,
    available_tools: list[str],
) -> JudgeDecision:
    unused_tools = [tool for tool in available_tools if tool != used_tool]
    route_decision = route_question(question)
    route_signals = route_decision.signals
    output_signals = first_output_signals_v2(question, first_output, route_signals)
    signals = {**route_signals, **output_signals, "used_tool": used_tool}

    if not first_output.ranked_documents:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool=fallback_next_tool(used_tool, unused_tools),
            reason="No ranked documents were returned.",
            confidence=0.95,
            signals={**signals, "protect_first_count": 0},
        )

    if used_tool == "file_fts":
        return judge_after_file(question, first_output, unused_tools, signals)
    if used_tool == "dense":
        return judge_after_dense(question, first_output, unused_tools, signals)
    if used_tool == "graph_path":
        return judge_after_graph(question, first_output, unused_tools, signals)

    return continue_with(
        question=question,
        first_output=first_output,
        next_tool=fallback_next_tool(used_tool, unused_tools),
        reason=f"Unsupported first tool {used_tool}; use fallback.",
        confidence=0.50,
        signals={**signals, "protect_first_count": 0},
    )


def judge_after_dense(
    question: str,
    first_output: SearchOutput,
    unused_tools: list[str],
    signals: dict[str, object],
) -> JudgeDecision:
    file_score = float(signals["file_score"])
    graph_score = float(signals["graph_score"])
    dense_confident = bool(signals["dense_confident"])
    semantic_confident = bool(signals["semantic_confident"])
    graph_need = bool(signals["graph_need"])
    file_need = bool(signals["file_need"])
    entity_coverage = float(signals["entity_coverage_ratio"])
    bridge_patterns = bool(signals["bridge_patterns"])

    if semantic_confident and not file_need and not graph_need:
        return stop(
            question=question,
            reason="Dense result has enough score, title, and entity coverage for a semantic question.",
            confidence=0.84 if dense_confident else 0.74,
            signals={**signals, "protect_first_count": 1},
        )

    if file_need and "file_fts" in unused_tools:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="file_fts",
            reason="The query contains exact/file lookup anchors that dense retrieval should verify with file_fts.",
            confidence=0.76,
            signals={**signals, "protect_first_count": 1 if dense_confident else 0},
        )

    if graph_need and "graph_path" in unused_tools:
        protect_first = 1 if (dense_confident or entity_coverage > 0.0) else 0
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="graph_path",
            reason="Dense found partial evidence, but entity/bridge coverage suggests a missing relational hop.",
            confidence=0.84 if graph_score >= 3.0 or bridge_patterns else 0.74,
            signals={**signals, "protect_first_count": protect_first},
        )

    if file_score >= 3.5 and "file_fts" in unused_tools:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="file_fts",
            reason="Moderate file/exact signal remains after dense retrieval.",
            confidence=0.68,
            signals={**signals, "protect_first_count": 1 if dense_confident else 0},
        )

    return stop(
        question=question,
        reason="No second tool has a stronger expected evidence gain than the first dense result.",
        confidence=0.72 if dense_confident else 0.62,
        signals={**signals, "protect_first_count": 1 if dense_confident else 0},
    )


def judge_after_file(
    question: str,
    first_output: SearchOutput,
    unused_tools: list[str],
    signals: dict[str, object],
) -> JudgeDecision:
    file_score = float(signals["file_score"])
    graph_need = bool(signals["graph_need"])
    exact_anchor = bool(signals["exact_anchor_in_top3"])
    entity_coverage = float(signals["entity_coverage_ratio"])

    if file_score >= 4.0 and exact_anchor and not graph_need:
        return stop(
            question=question,
            reason="file_fts already anchors the exact file/title/number evidence.",
            confidence=0.90,
            signals={**signals, "protect_first_count": 1},
        )

    if graph_need and "graph_path" in unused_tools:
        protect_first = 1 if exact_anchor or entity_coverage > 0.0 else 0
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="graph_path",
            reason="File retrieval appears partial for a relational query; graph_path should test bridge coverage.",
            confidence=0.78,
            signals={**signals, "protect_first_count": protect_first},
        )

    if file_score >= 3.5 and exact_anchor:
        return stop(
            question=question,
            reason="Exact lookup evidence is represented in the top file_fts results.",
            confidence=0.82,
            signals={**signals, "protect_first_count": 1},
        )

    if "dense" in unused_tools:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="dense",
            reason="file_fts did not produce a clearly anchored result; semantic retrieval can cross-check it.",
            confidence=0.66,
            signals={**signals, "protect_first_count": 1 if exact_anchor else 0},
        )

    return stop(
        question=question,
        reason="No useful second tool is available.",
        confidence=0.55,
        signals={**signals, "protect_first_count": 0},
    )


def judge_after_graph(
    question: str,
    first_output: SearchOutput,
    unused_tools: list[str],
    signals: dict[str, object],
) -> JudgeDecision:
    file_need = bool(signals["file_need"])
    graph_score = float(signals["graph_score"])
    graph_sufficient = bool(signals["graph_sufficient"])
    entity_coverage = float(signals["entity_coverage_ratio"])

    if file_need and "file_fts" in unused_tools:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="file_fts",
            reason="Graph was selected, but exact/file anchors should be verified with file_fts.",
            confidence=0.78,
            signals={**signals, "protect_first_count": 1 if entity_coverage > 0.0 else 0},
        )

    if graph_sufficient:
        return stop(
            question=question,
            reason="Graph-path results cover enough relation/entity evidence for this query.",
            confidence=0.82,
            signals={**signals, "protect_first_count": 1},
        )

    if "dense" in unused_tools:
        return continue_with(
            question=question,
            first_output=first_output,
            next_tool="dense",
            reason="Graph retrieval is under-covered; dense retrieval can recover semantic evidence.",
            confidence=0.68,
            signals={**signals, "protect_first_count": 1 if graph_score >= 2.4 else 0},
        )

    return stop(
        question=question,
        reason="No useful second tool is available.",
        confidence=0.55,
        signals={**signals, "protect_first_count": 0},
    )


def first_output_signals_v2(
    question: str,
    output: SearchOutput,
    route_signals: dict[str, object],
) -> dict[str, object]:
    docs = output.ranked_documents
    top_titles = [doc.title for doc in docs[:3]]
    top5_titles = [doc.title for doc in docs[:5]]
    top3_text = " ".join(top_titles)
    top5_text = " ".join(top5_titles)
    normalized_top3 = normalize_text(top3_text)
    normalized_top5 = normalize_text(top5_text)

    scores = [float(doc.score) for doc in docs[:5]]
    top_score = scores[0] if scores else 0.0
    second_score = scores[1] if len(scores) > 1 else 0.0
    score_gap = top_score - second_score
    score_spread = top_score - scores[-1] if len(scores) > 1 else 0.0

    question_terms = set(keywords(question, max_keywords=12))
    title_terms = set(keywords(top5_text, max_keywords=35))
    title_keyword_overlap = len(question_terms.intersection(title_terms))
    keyword_coverage_ratio = safe_ratio(title_keyword_overlap, len(question_terms))

    entities = [str(entity) for entity in route_signals.get("entities", [])]
    entity_hits = [
        entity
        for entity in entities
        if entity_is_covered(entity, normalized_top5)
    ]
    entity_coverage_ratio = safe_ratio(len(entity_hits), len(entities))
    entity_under_covered = bool(len(entities) >= 2 and entity_coverage_ratio < 0.67)

    phrases = [normalize_entity(phrase) for phrase in quoted_phrases(question)]
    nums = set(numbers_and_dates(question))
    exact_anchor = bool(
        any(phrase and phrase in normalized_top3 for phrase in phrases)
        or any(number in top3_text for number in nums)
        or any(
            phrase
            and (
                phrase == normalize_entity(title)
                or phrase in normalize_entity(title)
                or normalize_entity(title) in phrase
            )
            for phrase in phrases
            for title in top_titles
        )
    )

    graph_score = float(route_signals["graph_score"])
    file_score = float(route_signals["file_score"])
    bridge_patterns = bool(route_signals.get("bridge_patterns", []))
    graph_cues = bool(route_signals.get("graph_cues", []))

    dense_confident = bool(top_score >= 0.72 and score_gap >= 0.14)
    dense_plausible = bool(
        top_score >= 0.58
        and (score_gap >= 0.08 or title_keyword_overlap >= 2 or entity_coverage_ratio >= 0.5)
    )
    semantic_confident = bool(
        file_score < 3.0
        and not bridge_patterns
        and graph_score < 3.2
        and dense_plausible
        and (entity_coverage_ratio >= 0.5 or keyword_coverage_ratio >= 0.25 or title_keyword_overlap >= 2)
    )
    graph_need = bool(
        graph_score >= 3.0
        or (
            (bridge_patterns or graph_cues or graph_score >= 2.4)
            and (entity_under_covered or not dense_confident or score_gap < 0.12)
        )
    )
    file_need = bool(file_score >= 4.0 or (file_score >= 3.5 and exact_anchor))
    graph_sufficient = bool(
        graph_score >= 3.0
        and (
            entity_coverage_ratio >= 0.5
            or len(entities) <= 1
            or title_keyword_overlap >= 2
        )
    )

    return {
        "top_titles": top_titles,
        "top_score": top_score,
        "score_gap": score_gap,
        "score_spread": score_spread,
        "title_keyword_overlap": title_keyword_overlap,
        "keyword_coverage_ratio": keyword_coverage_ratio,
        "entity_hits_top5": entity_hits,
        "entity_coverage_ratio": entity_coverage_ratio,
        "entity_under_covered": entity_under_covered,
        "exact_anchor_in_top3": exact_anchor,
        "top3_has_quoted_phrase": any(phrase and phrase in normalized_top3 for phrase in phrases),
        "top3_has_title_like_phrase": any(
            phrase
            and (
                phrase == normalize_entity(title)
                or phrase in normalize_entity(title)
                or normalize_entity(title) in phrase
            )
            for phrase in phrases
            for title in top_titles
        ),
        "top3_has_number_date": any(number in top3_text for number in nums),
        "dense_confident": dense_confident,
        "dense_plausible": dense_plausible,
        "semantic_confident": semantic_confident,
        "graph_need": graph_need,
        "file_need": file_need,
        "graph_sufficient": graph_sufficient,
    }


def stop(
    *,
    question: str,
    reason: str,
    confidence: float,
    signals: dict[str, object],
) -> JudgeDecision:
    return JudgeDecision(
        sufficient=True,
        missing_information="",
        next_tool=None,
        revised_query=question,
        confidence=confidence,
        signals=signals,
    )


def continue_with(
    *,
    question: str,
    first_output: SearchOutput,
    next_tool: str | None,
    reason: str,
    confidence: float,
    signals: dict[str, object],
) -> JudgeDecision:
    return JudgeDecision(
        sufficient=False,
        missing_information=reason,
        next_tool=next_tool,
        revised_query=revised_query(question, first_output, next_tool),
        confidence=confidence,
        signals=signals,
    )


def revised_query(question: str, first_output: SearchOutput, next_tool: str | None) -> str:
    if next_tool == "graph_path":
        top_titles = " ".join(doc.title for doc in first_output.ranked_documents[:2])
        return f"{question} {top_titles}".strip()
    return question


def fallback_next_tool(used_tool: str, unused_tools: list[str]) -> str | None:
    if not unused_tools:
        return None
    preferences = {
        "dense": ["graph_path", "file_fts"],
        "file_fts": ["dense", "graph_path"],
        "graph_path": ["dense", "file_fts"],
    }
    for tool in preferences.get(used_tool, []):
        if tool in unused_tools:
            return tool
    return unused_tools[0]


def entity_is_covered(entity: str, normalized_text: str) -> bool:
    normalized_entity = normalize_entity(entity)
    if not normalized_entity:
        return False
    if normalized_entity in normalized_text:
        return True
    terms = [term for term in normalized_entity.split() if len(term) > 2]
    return bool(terms and all(term in normalized_text for term in terms))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
