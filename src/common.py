from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


JSONDict = dict[str, Any]


@dataclass(slots=True)
class CorpusDocument:
    doc_id: str
    title: str
    sentences: list[str]
    full_text: str
    source_question_ids: list[str] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CorpusDocument":
        sentences = [str(x) for x in row.get("sentences", [])]
        full_text = str(row.get("full_text") or " ".join(sentences))
        return cls(
            doc_id=str(row["doc_id"]),
            title=str(row.get("title", "")),
            sentences=sentences,
            full_text=full_text,
            source_question_ids=[str(x) for x in row.get("source_question_ids", [])],
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class GoldSentence:
    doc_id: str
    sent_id: int
    text: str | None = None

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "GoldSentence":
        return cls(
            doc_id=str(row["doc_id"]),
            sent_id=int(row["sent_id"]),
            text=None if row.get("text") is None else str(row["text"]),
        )


@dataclass(slots=True)
class Question:
    question_id: str
    question: str
    task_type: str
    gold_documents: list[str]
    gold_sentences: list[GoldSentence] = field(default_factory=list)
    source_hotpot_id: str | None = None
    split: str = "test"
    quality_checked: bool = False
    metadata: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Question":
        gold_sentences = [
            GoldSentence.from_dict(x) for x in row.get("gold_sentences", [])
        ]
        gold_documents = [str(x) for x in row.get("gold_documents", [])]
        if not gold_documents and gold_sentences:
            gold_documents = sorted({x.doc_id for x in gold_sentences})
        return cls(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            task_type=str(row.get("task_type", "unknown")),
            gold_documents=gold_documents,
            gold_sentences=gold_sentences,
            source_hotpot_id=(
                None
                if row.get("source_hotpot_id") is None
                else str(row.get("source_hotpot_id"))
            ),
            split=str(row.get("split", "test")),
            quality_checked=bool(row.get("quality_checked", False)),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        row = asdict(self)
        row["gold_sentences"] = [asdict(x) for x in self.gold_sentences]
        return row


@dataclass(slots=True)
class PublicQuestion:
    question_id: str
    question: str
    split: str = "test"

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "PublicQuestion":
        return cls(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            split=str(row.get("split", "test")),
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class RetrievedDocument:
    doc_id: str
    title: str = ""
    score: float = 0.0
    rank: int = 0
    tool: str | None = None
    metadata: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, Any],
        *,
        fallback_rank: int = 0,
        fallback_tool: str | None = None,
    ) -> "RetrievedDocument":
        return cls(
            doc_id=str(row["doc_id"]),
            title=str(row.get("title", "")),
            score=float(row.get("score", 0.0)),
            rank=int(row.get("rank", fallback_rank)),
            tool=None if row.get("tool", fallback_tool) is None else str(row.get("tool", fallback_tool)),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class SearchOutput:
    question_id: str
    tool: str
    ranked_documents: list[RetrievedDocument]
    evidence_sentences: list[JSONDict] = field(default_factory=list)
    latency_ms: float = 0.0
    text_read_tokens: int = 0
    trace: list[str] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SearchOutput":
        tool = str(row.get("tool") or row.get("method") or "unknown")
        docs = [
            RetrievedDocument.from_dict(
                doc,
                fallback_rank=i + 1,
                fallback_tool=tool,
            )
            for i, doc in enumerate(row.get("ranked_documents", []))
        ]
        return cls(
            question_id=str(row["question_id"]),
            tool=tool,
            ranked_documents=docs,
            evidence_sentences=list(row.get("evidence_sentences", [])),
            latency_ms=float(row.get("latency_ms", 0.0)),
            text_read_tokens=int(row.get("text_read_tokens", 0)),
            trace=[str(x) for x in row.get("trace", [])],
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> JSONDict:
        row = asdict(self)
        row["ranked_documents"] = [x.to_dict() for x in self.ranked_documents]
        return row

    def doc_ids(self, k: int | None = None) -> list[str]:
        docs = self.ranked_documents if k is None else self.ranked_documents[:k]
        return [doc.doc_id for doc in docs]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_jsonl(path: str | Path) -> list[JSONDict]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[JSONDict]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def load_corpus(path: str | Path) -> list[CorpusDocument]:
    return [CorpusDocument.from_dict(row) for row in iter_jsonl(path)]


def load_public_corpus(path: str | Path) -> list[CorpusDocument]:
    """Load documents for runtime retrieval without evaluation-only metadata."""
    public_rows = []
    for row in iter_jsonl(path):
        public_row = dict(row)
        public_row["source_question_ids"] = []
        public_row["metadata"] = {}
        public_rows.append(CorpusDocument.from_dict(public_row))
    return public_rows


def load_questions(path: str | Path) -> list[Question]:
    return [Question.from_dict(row) for row in iter_jsonl(path)]


def load_public_questions(path: str | Path) -> list[PublicQuestion]:
    """Load questions for runtime retrieval without gold labels or answer metadata."""
    return [PublicQuestion.from_dict(row) for row in iter_jsonl(path)]


def rrf_merge(
    outputs: Sequence[SearchOutput],
    *,
    top_k: int = 5,
    k0: int = 60,
    question_id: str | None = None,
) -> SearchOutput:
    if not outputs:
        raise ValueError("rrf_merge requires at least one SearchOutput")

    scores: dict[str, float] = {}
    docs_by_id: dict[str, RetrievedDocument] = {}
    source_ranks: dict[str, dict[str, int]] = {}

    for output in outputs:
        for doc in output.ranked_documents:
            if doc.rank <= 0:
                continue
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + (1.0 / (k0 + doc.rank))
            docs_by_id.setdefault(doc.doc_id, doc)
            source_ranks.setdefault(doc.doc_id, {})[output.tool] = doc.rank

    ranked_doc_ids = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], docs_by_id[doc_id].title, doc_id),
    )[:top_k]

    merged_docs = []
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
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
                },
            )
        )

    return SearchOutput(
        question_id=question_id or outputs[0].question_id,
        tool="fusion",
        ranked_documents=merged_docs,
        latency_ms=sum(output.latency_ms for output in outputs),
        text_read_tokens=sum(output.text_read_tokens for output in outputs),
        trace=[item for output in outputs for item in output.trace] + ["rrf_merge"],
        metadata={
            "merged_tools": [output.tool for output in outputs],
            "rrf_k0": k0,
        },
    )
