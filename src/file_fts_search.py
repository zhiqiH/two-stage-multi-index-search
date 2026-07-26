from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    CorpusDocument,
    RetrievedDocument,
    SearchOutput,
    estimate_tokens,
    normalize_text,
    now_ms,
)
from .text_utils import STOPWORDS, keywords, numbers_and_dates, quoted_phrases, simple_entities


QUERY_STOPWORDS = STOPWORDS.union(
    {
        "also",
        "did",
        "does",
        "find",
        "give",
        "list",
        "name",
        "named",
        "tell",
        "what",
        "when",
        "where",
        "whose",
    }
)


@dataclass(slots=True)
class FTSChunk:
    chunk_id: int
    doc_id: str
    title: str
    body: str
    sentence_start: int
    sentence_end: int


@dataclass(slots=True)
class DocumentFeatures:
    doc_index: int
    doc_id: str
    title_norm: str
    title_tokens: set[str]
    text_norm: str
    numbers_dates: set[str]


@dataclass(slots=True)
class DocumentScoreState:
    doc_index: int
    chunk_scores: list[float] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    raw_fts_scores: list[float] = field(default_factory=list)
    title_match: bool = False
    title_overlap: float = 0.0
    matched_numbers_dates: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    exact_title_phrases: list[str] = field(default_factory=list)


class FileFTSSearcher:
    """SQLite FTS5 chunk-level file retriever with document-level aggregation."""

    def __init__(
        self,
        *,
        corpus: list[CorpusDocument],
        connection: sqlite3.Connection,
        chunks_by_id: dict[int, FTSChunk],
        doc_features: list[DocumentFeatures],
        chunk_sentences: int,
        chunk_stride: int,
        candidate_chunks: int,
        max_query_terms: int,
        max_chunks_per_doc: int,
        title_weight: float,
        body_weight: float,
        entity_weight: float,
        number_date_weight: float,
        chunk_sum_weight: float,
        title_match_bonus: float,
        exact_title_bonus: float,
        exact_phrase_bonus: float,
        number_date_bonus: float,
        title_overlap_bonus: float,
        max_open_files: int,
    ) -> None:
        self.corpus = corpus
        self.connection = connection
        self.chunks_by_id = chunks_by_id
        self.doc_features = doc_features
        self.doc_index_by_id = {doc.doc_id: index for index, doc in enumerate(corpus)}
        self.chunk_sentences = chunk_sentences
        self.chunk_stride = chunk_stride
        self.candidate_chunks = candidate_chunks
        self.max_query_terms = max_query_terms
        self.max_chunks_per_doc = max_chunks_per_doc
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.entity_weight = entity_weight
        self.number_date_weight = number_date_weight
        self.chunk_sum_weight = chunk_sum_weight
        self.title_match_bonus = title_match_bonus
        self.exact_title_bonus = exact_title_bonus
        self.exact_phrase_bonus = exact_phrase_bonus
        self.number_date_bonus = number_date_bonus
        self.title_overlap_bonus = title_overlap_bonus
        self.max_open_files = max_open_files

    @classmethod
    def build(
        cls,
        corpus: list[CorpusDocument],
        *,
        chunk_sentences: int = 3,
        chunk_stride: int = 2,
        candidate_chunks: int = 400,
        max_query_terms: int = 18,
        max_chunks_per_doc: int = 3,
        title_weight: float = 6.0,
        body_weight: float = 1.0,
        entity_weight: float = 2.0,
        number_date_weight: float = 3.0,
        chunk_sum_weight: float = 0.20,
        title_match_bonus: float = 0.30,
        exact_title_bonus: float = 0.75,
        exact_phrase_bonus: float = 0.20,
        number_date_bonus: float = 0.10,
        title_overlap_bonus: float = 0.20,
        max_open_files: int = 5,
    ) -> "FileFTSSearcher":
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        create_schema(connection)
        chunks_by_id: dict[int, FTSChunk] = {}
        doc_features: list[DocumentFeatures] = []

        chunk_id = 0
        for doc_index, doc in enumerate(corpus):
            text = doc.full_text or " ".join(doc.sentences)
            feature_text = f"{doc.title}. {text}"
            doc_features.append(
                DocumentFeatures(
                    doc_index=doc_index,
                    doc_id=doc.doc_id,
                    title_norm=normalize_text(doc.title),
                    title_tokens=set(tokenize_query_text(doc.title, max_terms=32)),
                    text_norm=normalize_text(feature_text),
                    numbers_dates=set(numbers_and_dates(feature_text)),
                )
            )

            for sentence_start, sentence_end, body in iter_chunks(
                doc,
                chunk_sentences=chunk_sentences,
                chunk_stride=chunk_stride,
            ):
                chunk_id += 1
                entities = " ".join(simple_entities(f"{doc.title}. {body}"))
                number_values = " ".join(numbers_and_dates(f"{doc.title}. {body}"))
                chunks_by_id[chunk_id] = FTSChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    body=body,
                    sentence_start=sentence_start,
                    sentence_end=sentence_end,
                )
                connection.execute(
                    """
                    INSERT INTO chunks(
                        rowid, title, body, entities, numbers_dates, doc_id, chunk_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc.title, body, entities, number_values, doc.doc_id, chunk_id),
                )
        connection.commit()

        return cls(
            corpus=corpus,
            connection=connection,
            chunks_by_id=chunks_by_id,
            doc_features=doc_features,
            chunk_sentences=chunk_sentences,
            chunk_stride=chunk_stride,
            candidate_chunks=candidate_chunks,
            max_query_terms=max_query_terms,
            max_chunks_per_doc=max_chunks_per_doc,
            title_weight=title_weight,
            body_weight=body_weight,
            entity_weight=entity_weight,
            number_date_weight=number_date_weight,
            chunk_sum_weight=chunk_sum_weight,
            title_match_bonus=title_match_bonus,
            exact_title_bonus=exact_title_bonus,
            exact_phrase_bonus=exact_phrase_bonus,
            number_date_bonus=number_date_bonus,
            title_overlap_bonus=title_overlap_bonus,
            max_open_files=max_open_files,
        )

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        destination = sqlite3.connect(path)
        try:
            self.connection.backup(destination)
        finally:
            destination.close()

    def search(
        self,
        *,
        question_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchOutput:
        start_ms = now_ms()
        query_terms = tokenize_query_text(query, max_terms=self.max_query_terms)
        fts_query = build_fts_or_query(query_terms)
        states: dict[str, DocumentScoreState] = {}

        chunk_hits = self.search_chunks(fts_query) if fts_query else []
        for rank, hit in enumerate(chunk_hits, start=1):
            doc_id = str(hit["doc_id"])
            doc_index = self.doc_index_by_id[doc_id]
            state = states.setdefault(doc_id, DocumentScoreState(doc_index=doc_index))
            state.chunk_scores.append(rank_to_score(rank))
            state.chunk_ids.append(int(hit["chunk_id"]))
            state.raw_fts_scores.append(float(hit["raw_fts_score"]))

        self.add_exact_feature_candidates(query=query, query_terms=query_terms, states=states)

        scored_docs = []
        for doc_id, state in states.items():
            score = aggregate_document_score(
                state,
                max_chunks_per_doc=self.max_chunks_per_doc,
                chunk_sum_weight=self.chunk_sum_weight,
                title_match_bonus=self.title_match_bonus,
                exact_title_bonus=self.exact_title_bonus,
                exact_phrase_bonus=self.exact_phrase_bonus,
                number_date_bonus=self.number_date_bonus,
                title_overlap_bonus=self.title_overlap_bonus,
            )
            scored_docs.append((score, doc_id, state))

        ranked = sorted(
            scored_docs,
            key=lambda item: (-item[0], self.corpus[item[2].doc_index].title, item[1]),
        )[:top_k]

        docs = []
        text_read_tokens = 0
        for rank, (score, _doc_id, state) in enumerate(ranked, start=1):
            doc = self.corpus[state.doc_index]
            chunk_token_count = self.estimate_selected_chunk_tokens(state)
            text_read_tokens += chunk_token_count or estimate_tokens(doc.full_text)
            docs.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=score,
                    rank=rank,
                    tool="file_fts",
                    metadata={
                        "chunk_ids": state.chunk_ids[: self.max_chunks_per_doc],
                        "chunk_scores": state.chunk_scores[: self.max_chunks_per_doc],
                        "raw_fts_scores": state.raw_fts_scores[: self.max_chunks_per_doc],
                        "title_match": state.title_match,
                        "title_overlap": state.title_overlap,
                        "matched_numbers_dates": state.matched_numbers_dates,
                        "matched_phrases": state.matched_phrases,
                        "exact_title_phrases": state.exact_title_phrases,
                    },
                )
            )

        return SearchOutput(
            question_id=question_id,
            tool="file_fts",
            ranked_documents=docs,
            latency_ms=now_ms() - start_ms,
            text_read_tokens=text_read_tokens,
            trace=[
                "sqlite_fts5_match",
                f"candidate_chunks:{self.candidate_chunks}",
                "chunk_to_document_aggregation",
                "exact_feature_rescore",
                f"open_top_{min(top_k, self.max_open_files)}",
                f"return_top_{top_k}",
            ],
            metadata={
                "fts_query_terms": query_terms,
                "chunk_sentences": self.chunk_sentences,
                "chunk_stride": self.chunk_stride,
            },
        )

    def search_chunks(self, fts_query: str) -> list[sqlite3.Row]:
        cursor = self.connection.execute(
            """
            SELECT
                rowid,
                doc_id,
                chunk_id,
                bm25(chunks, ?, ?, ?, ?, 0.0, 0.0) AS raw_fts_score
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY raw_fts_score
            LIMIT ?
            """,
            (
                self.title_weight,
                self.body_weight,
                self.entity_weight,
                self.number_date_weight,
                fts_query,
                self.candidate_chunks,
            ),
        )
        return list(cursor.fetchall())

    def add_exact_feature_candidates(
        self,
        *,
        query: str,
        query_terms: list[str],
        states: dict[str, DocumentScoreState],
    ) -> None:
        query_norm = normalize_text(query)
        query_token_set = set(query_terms)
        query_numbers = set(numbers_and_dates(query))
        phrases = [phrase for phrase in quoted_phrases(query) if normalize_text(phrase)]

        for features in self.doc_features:
            doc = self.corpus[features.doc_index]
            title_match = bool(features.title_norm and features.title_norm in query_norm)
            title_overlap = 0.0
            if features.title_tokens and query_token_set:
                title_overlap = len(features.title_tokens.intersection(query_token_set)) / len(features.title_tokens)

            exact_title_phrases = []
            matched_phrases = []
            for phrase in phrases:
                phrase_norm = normalize_text(phrase)
                if not phrase_norm:
                    continue
                if phrase_norm == features.title_norm or phrase_norm in features.title_norm:
                    exact_title_phrases.append(phrase)
                if phrase_norm in features.text_norm:
                    matched_phrases.append(phrase)

            matched_numbers = sorted(query_numbers.intersection(features.numbers_dates))
            should_add = (
                title_match
                or title_overlap >= 0.50
                or bool(exact_title_phrases)
                or bool(matched_phrases)
                or bool(matched_numbers)
            )
            if not should_add:
                continue

            state = states.setdefault(doc.doc_id, DocumentScoreState(doc_index=features.doc_index))
            state.title_match = state.title_match or title_match
            state.title_overlap = max(state.title_overlap, title_overlap)
            state.exact_title_phrases = sorted(set(state.exact_title_phrases).union(exact_title_phrases))
            state.matched_phrases = sorted(set(state.matched_phrases).union(matched_phrases))
            state.matched_numbers_dates = sorted(set(state.matched_numbers_dates).union(matched_numbers))

    def estimate_selected_chunk_tokens(self, state: DocumentScoreState) -> int:
        tokens = 0
        for chunk_id in state.chunk_ids[: self.max_chunks_per_doc]:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is not None:
                tokens += estimate_tokens(chunk.body)
        return tokens


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE VIRTUAL TABLE chunks USING fts5(
            title,
            body,
            entities,
            numbers_dates,
            doc_id UNINDEXED,
            chunk_id UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )


def iter_chunks(
    doc: CorpusDocument,
    *,
    chunk_sentences: int,
    chunk_stride: int,
) -> list[tuple[int, int, str]]:
    sentences = [sentence.strip() for sentence in doc.sentences if sentence.strip()]
    if not sentences:
        text = doc.full_text.strip()
        return [(0, 0, text)] if text else [(0, 0, doc.title)]

    chunk_sentences = max(1, chunk_sentences)
    chunk_stride = max(1, chunk_stride)
    chunks = []
    for start in range(0, len(sentences), chunk_stride):
        end = min(start + chunk_sentences, len(sentences))
        body = " ".join(sentences[start:end]).strip()
        if body:
            chunks.append((start, end - 1, body))
        if end >= len(sentences):
            break
    return chunks


def tokenize_query_text(text: str, *, max_terms: int) -> list[str]:
    tokens = []
    seen = set()
    keyword_tokens = keywords(text, max_keywords=max_terms * 2)
    fallback_tokens = normalize_text(text).split()
    for token in keyword_tokens + fallback_tokens:
        if token in seen or token in QUERY_STOPWORDS:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_terms:
            break
    return tokens


def build_fts_or_query(terms: list[str]) -> str:
    safe_terms = [quote_fts_term(term) for term in terms if term]
    return " OR ".join(safe_terms)


def quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def rank_to_score(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def aggregate_document_score(
    state: DocumentScoreState,
    *,
    max_chunks_per_doc: int,
    chunk_sum_weight: float,
    title_match_bonus: float,
    exact_title_bonus: float,
    exact_phrase_bonus: float,
    number_date_bonus: float,
    title_overlap_bonus: float,
) -> float:
    chunk_scores = sorted(state.chunk_scores, reverse=True)
    base_score = chunk_scores[0] if chunk_scores else 0.0
    if len(chunk_scores) > 1:
        base_score += chunk_sum_weight * sum(chunk_scores[1:max_chunks_per_doc])

    bonus = 0.0
    if state.title_match:
        bonus += title_match_bonus
    if state.title_overlap:
        bonus += title_overlap_bonus * state.title_overlap
    bonus += exact_title_bonus * len(state.exact_title_phrases)
    bonus += exact_phrase_bonus * len(state.matched_phrases)
    bonus += number_date_bonus * len(state.matched_numbers_dates)
    return base_score + bonus
