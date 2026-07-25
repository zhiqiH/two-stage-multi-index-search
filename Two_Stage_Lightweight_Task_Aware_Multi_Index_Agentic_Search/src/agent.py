from __future__ import annotations

from .common import PublicQuestion, RetrievedDocument, SearchOutput
from .evidence_judge import JudgeDecision, heuristic_judge
from .retriever_factory import RetrieverBundle
from .router import RouteDecision, route_question


CORE_TOOLS = ["dense", "graph_path", "file_fts"]


class TwoStageAgent:
    def __init__(
        self,
        *,
        retrievers: RetrieverBundle,
        top_k: int,
        rrf_k0: int = 60,
        tools: list[str] | None = None,
    ) -> None:
        self.retrievers = retrievers
        self.top_k = top_k
        self.rrf_k0 = rrf_k0
        self.tools = tools or CORE_TOOLS

    def run(self, question: PublicQuestion) -> dict:
        route_decision = route_question(question.question)
        first_tool = route_decision.tool
        if first_tool not in self.tools:
            first_tool = self.tools[0]

        first_output = self.retrievers.search(
            tool=first_tool,
            question_id=question.question_id,
            query=question.question,
            top_k=self.top_k,
        )
        judge_decision = heuristic_judge(
            question=question.question,
            first_output=first_output,
            used_tool=first_tool,
            available_tools=self.tools,
        )

        if judge_decision.sufficient or not judge_decision.next_tool:
            return self.format_prediction(
                question=question,
                route_decision=route_decision,
                judge_decision=judge_decision,
                final_output=first_output,
                first_output=first_output,
                second_output=None,
            )

        second_tool = judge_decision.next_tool
        second_output = self.retrievers.search(
            tool=second_tool,
            question_id=question.question_id,
            query=judge_decision.revised_query,
            top_k=self.top_k,
        )
        final_output = protected_rrf_merge(
            first_output=first_output,
            second_output=second_output,
            judge_decision=judge_decision,
            top_k=self.top_k,
            k0=self.rrf_k0,
            question_id=question.question_id,
        )
        return self.format_prediction(
            question=question,
            route_decision=route_decision,
            judge_decision=judge_decision,
            final_output=final_output,
            first_output=first_output,
            second_output=second_output,
        )

    def format_prediction(
        self,
        *,
        question: PublicQuestion,
        route_decision: RouteDecision,
        judge_decision: JudgeDecision,
        final_output: SearchOutput,
        first_output: SearchOutput,
        second_output: SearchOutput | None,
    ) -> dict:
        row = final_output.to_dict()
        row["method"] = "two_stage_agent"
        row["tool_calls"] = 1 if second_output is None else 2
        row["tool_sequence"] = (
            [first_output.tool] if second_output is None else [first_output.tool, second_output.tool]
        )
        row["first_round_documents"] = [
            doc.to_dict() for doc in first_output.ranked_documents
        ]
        row["route_decision"] = route_decision.to_dict()
        row["judge_decision"] = judge_decision.to_dict()
        row["rounds"] = [first_output.to_dict()]
        if second_output is not None:
            row["rounds"].append(second_output.to_dict())
        return row


def protected_rrf_merge(
    *,
    first_output: SearchOutput,
    second_output: SearchOutput,
    judge_decision: JudgeDecision,
    top_k: int,
    k0: int,
    question_id: str,
) -> SearchOutput:
    signals = judge_decision.signals
    protect_first_count = int(signals.get("protect_first_count", 0) or 0)
    protect_first_count = max(0, min(protect_first_count, top_k, len(first_output.ranked_documents)))
    first_weight, second_weight = fusion_weights(first_output.tool, second_output.tool, signals)

    outputs = [(first_output, first_weight), (second_output, second_weight)]
    scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDocument] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    source_weights: dict[str, dict[str, float]] = {}

    for output, weight in outputs:
        for doc in output.ranked_documents:
            if doc.rank <= 0:
                continue
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + weight * (1.0 / (k0 + doc.rank))
            docs_by_id.setdefault(doc.doc_id, doc)
            source_ranks.setdefault(doc.doc_id, {})[output.tool] = doc.rank
            source_weights.setdefault(doc.doc_id, {})[output.tool] = weight

    protected_ids: list[str] = []
    for doc in first_output.ranked_documents[:protect_first_count]:
        if doc.doc_id not in protected_ids:
            protected_ids.append(doc.doc_id)

    ranked_doc_ids = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], docs_by_id[doc_id].title, doc_id),
    )
    final_ids = protected_ids[:]
    for doc_id in ranked_doc_ids:
        if doc_id not in final_ids:
            final_ids.append(doc_id)
        if len(final_ids) >= top_k:
            break

    merged_docs = []
    for rank, doc_id in enumerate(final_ids[:top_k], start=1):
        original = docs_by_id[doc_id]
        merged_docs.append(
            RetrievedDocument(
                doc_id=doc_id,
                title=original.title,
                score=scores[doc_id],
                rank=rank,
                tool="fusion",
                metadata={
                    "source_ranks": source_ranks[doc_id],
                    "source_tools": sorted(source_ranks[doc_id]),
                    "source_weights": source_weights[doc_id],
                    "protected_first_round": doc_id in protected_ids,
                },
            )
        )

    return SearchOutput(
        question_id=question_id,
        tool="fusion",
        ranked_documents=merged_docs,
        latency_ms=first_output.latency_ms + second_output.latency_ms,
        text_read_tokens=first_output.text_read_tokens + second_output.text_read_tokens,
        trace=first_output.trace
        + second_output.trace
        + ["protected_weighted_rrf_merge"],
        metadata={
            "merged_tools": [first_output.tool, second_output.tool],
            "rrf_k0": k0,
            "protect_first_count": protect_first_count,
            "first_weight": first_weight,
            "second_weight": second_weight,
            "fusion_policy": "protected_weighted_rrf_v1",
        },
    )


def fusion_weights(first_tool: str, second_tool: str, signals: dict[str, object]) -> tuple[float, float]:
    first_weight = 1.0
    second_weight = 1.0

    if second_tool == "graph_path" and bool(signals.get("graph_need", False)):
        second_weight = 1.35
    if second_tool == "file_fts" and bool(signals.get("file_need", False)):
        second_weight = 1.25
    if first_tool == "file_fts" and bool(signals.get("exact_anchor_in_top3", False)):
        first_weight = 1.20
    if first_tool == "dense" and bool(signals.get("dense_confident", False)):
        first_weight = max(first_weight, 1.10)

    return first_weight, second_weight
