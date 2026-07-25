from __future__ import annotations

import math
import pickle
from collections import Counter
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
from .text_utils import keywords, numbers_and_dates, quoted_phrases, simple_entities


RELATION_LEXICON = {
    "birth": {"born", "birth", "birthplace", "native"},
    "death": {"died", "death", "deceased"},
    "location": {"located", "location", "city", "country", "state", "province", "region", "where"},
    "founding": {"founder", "founded", "established", "created"},
    "authorship": {"author", "wrote", "writer", "novel", "book"},
    "film": {"director", "directed", "film", "movie", "starring", "actor", "actress"},
    "music": {"album", "song", "singer", "band", "music", "composer"},
    "sports": {"team", "club", "league", "season", "coach", "player", "played"},
    "organization": {"company", "organization", "university", "school", "member"},
    "family": {"spouse", "wife", "husband", "father", "mother", "son", "daughter"},
    "award": {"award", "prize", "won", "winner", "nominated"},
    "population": {"population", "inhabitants", "census"},
    "time": {"year", "date", "when"},
}


NodeId = tuple[str, str]


@dataclass(slots=True)
class SentenceNode:
    doc_id: str
    sent_id: int
    entities: set[str]
    keywords: set[str]
    relations: set[str]


@dataclass(slots=True)
class GraphPathState:
    nodes: tuple[NodeId, ...]
    score: float
    doc_ids: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    support: tuple[dict[str, Any], ...] = ()

    @property
    def last_node(self) -> NodeId:
        return self.nodes[-1]


@dataclass(slots=True)
class GraphPathCandidate:
    path: GraphPathState
    final_score: float
    query_coverage: float
    relation_coverage: float


@dataclass(slots=True)
class QuerySignals:
    query: str
    normalized: str
    keywords: set[str]
    entities: set[str]
    numbers_dates: set[str]
    quoted_phrases: list[str]
    relations: set[str]


class GraphPathSearcher:
    """Pure graph path retriever over a document-sentence-entity graph."""

    def __init__(
        self,
        *,
        corpus: list[CorpusDocument],
        sentence_nodes_by_doc: dict[str, list[SentenceNode]],
        entities_by_doc: dict[str, set[str]],
        docs_by_entity: dict[str, set[str]],
        sentences_by_entity: dict[str, list[tuple[str, int]]],
        doc_keywords: dict[str, set[str]],
        doc_relations: dict[str, set[str]],
        title_norm_by_doc: dict[str, str],
        doc_ids_by_title_norm: dict[str, str],
        title_keywords_by_doc: dict[str, set[str]],
        doc_links: dict[str, set[str]],
        entity_idf: dict[str, float],
        max_depth: int,
        beam_size: int,
        max_start_entities: int,
        max_start_docs: int,
        max_entity_degree: int,
        max_doc_entity_neighbors: int,
        max_doc_sentence_neighbors: int,
        max_sentence_entity_neighbors: int,
        max_title_link_neighbors: int,
        path_top_n: int,
        min_entity_chars: int,
        score_weights: dict[str, float],
    ) -> None:
        self.corpus = corpus
        self.sentence_nodes_by_doc = sentence_nodes_by_doc
        self.entities_by_doc = entities_by_doc
        self.docs_by_entity = docs_by_entity
        self.sentences_by_entity = sentences_by_entity
        self.doc_keywords = doc_keywords
        self.doc_relations = doc_relations
        self.title_norm_by_doc = title_norm_by_doc
        self.doc_ids_by_title_norm = doc_ids_by_title_norm
        self.title_keywords_by_doc = title_keywords_by_doc
        self.doc_links = doc_links
        self.entity_idf = entity_idf
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.max_start_entities = max_start_entities
        self.max_start_docs = max_start_docs
        self.max_entity_degree = max_entity_degree
        self.max_doc_entity_neighbors = max_doc_entity_neighbors
        self.max_doc_sentence_neighbors = max_doc_sentence_neighbors
        self.max_sentence_entity_neighbors = max_sentence_entity_neighbors
        self.max_title_link_neighbors = max_title_link_neighbors
        self.path_top_n = path_top_n
        self.min_entity_chars = min_entity_chars
        self.score_weights = score_weights
        self.doc_index_by_id = {doc.doc_id: index for index, doc in enumerate(corpus)}

    @classmethod
    def build(
        cls,
        corpus: list[CorpusDocument],
        *,
        max_depth: int = 4,
        beam_size: int = 45,
        max_start_entities: int = 16,
        max_start_docs: int = 12,
        max_entity_degree: int = 75,
        max_doc_entity_neighbors: int = 18,
        max_doc_sentence_neighbors: int = 8,
        max_sentence_entity_neighbors: int = 8,
        max_title_link_neighbors: int = 6,
        path_top_n: int = 30,
        min_entity_chars: int = 3,
        score_weights: dict[str, float] | None = None,
    ) -> "GraphPathSearcher":
        sentence_nodes_by_doc: dict[str, list[SentenceNode]] = {}
        entities_by_doc: dict[str, set[str]] = {}
        docs_by_entity: dict[str, set[str]] = {}
        sentences_by_entity: dict[str, list[tuple[str, int]]] = {}
        doc_keywords: dict[str, set[str]] = {}
        doc_relations: dict[str, set[str]] = {}
        title_norm_by_doc: dict[str, str] = {}
        doc_ids_by_title_norm: dict[str, str] = {}
        title_keywords_by_doc: dict[str, set[str]] = {}

        for doc in corpus:
            title_norm = normalize_text(doc.title)
            title_norm_by_doc[doc.doc_id] = title_norm
            if title_norm:
                doc_ids_by_title_norm.setdefault(title_norm, doc.doc_id)
            title_keywords_by_doc[doc.doc_id] = set(keywords(doc.title, max_keywords=12))

        for doc in corpus:
            sentence_nodes = build_sentence_nodes(doc, min_entity_chars=min_entity_chars)
            sentence_nodes_by_doc[doc.doc_id] = sentence_nodes
            doc_entities = extract_document_entities(
                doc,
                sentence_nodes=sentence_nodes,
                min_entity_chars=min_entity_chars,
            )
            entities_by_doc[doc.doc_id] = doc_entities
            for entity in doc_entities:
                docs_by_entity.setdefault(entity, set()).add(doc.doc_id)

            for sentence in sentence_nodes:
                for entity in sentence.entities:
                    sentences_by_entity.setdefault(entity, []).append((doc.doc_id, sentence.sent_id))

            text = f"{doc.title}. {doc.full_text}"
            doc_keywords[doc.doc_id] = set(keywords(text, max_keywords=96))
            doc_relations[doc.doc_id] = relation_cues(text)

        entity_idf = {
            entity: math.log((1.0 + len(corpus)) / (1.0 + len(doc_ids))) + 1.0
            for entity, doc_ids in docs_by_entity.items()
        }
        doc_links = build_title_links(
            corpus=corpus,
            title_norm_by_doc=title_norm_by_doc,
            doc_ids_by_title_norm=doc_ids_by_title_norm,
            max_links_per_doc=max_title_link_neighbors,
        )
        weights = score_weights or {
            "start_entity": 1.00,
            "start_doc": 0.95,
            "entity_to_doc": 0.72,
            "doc_to_sentence": 0.62,
            "sentence_to_entity": 0.70,
            "doc_to_doc": 0.82,
            "query_coverage": 0.55,
            "relation_coverage": 0.45,
            "multi_doc_bonus": 0.35,
            "title_match": 0.45,
            "high_degree_penalty": 0.22,
            "path_length_penalty": 0.04,
        }
        return cls(
            corpus=corpus,
            sentence_nodes_by_doc=sentence_nodes_by_doc,
            entities_by_doc=entities_by_doc,
            docs_by_entity=docs_by_entity,
            sentences_by_entity=sentences_by_entity,
            doc_keywords=doc_keywords,
            doc_relations=doc_relations,
            title_norm_by_doc=title_norm_by_doc,
            doc_ids_by_title_norm=doc_ids_by_title_norm,
            title_keywords_by_doc=title_keywords_by_doc,
            doc_links=doc_links,
            entity_idf=entity_idf,
            max_depth=max_depth,
            beam_size=beam_size,
            max_start_entities=max_start_entities,
            max_start_docs=max_start_docs,
            max_entity_degree=max_entity_degree,
            max_doc_entity_neighbors=max_doc_entity_neighbors,
            max_doc_sentence_neighbors=max_doc_sentence_neighbors,
            max_sentence_entity_neighbors=max_sentence_entity_neighbors,
            max_title_link_neighbors=max_title_link_neighbors,
            path_top_n=path_top_n,
            min_entity_chars=min_entity_chars,
            score_weights=weights,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "sentence_nodes_by_doc": self.sentence_nodes_by_doc,
                    "entities_by_doc": self.entities_by_doc,
                    "docs_by_entity": self.docs_by_entity,
                    "sentences_by_entity": self.sentences_by_entity,
                    "doc_keywords": self.doc_keywords,
                    "doc_relations": self.doc_relations,
                    "title_norm_by_doc": self.title_norm_by_doc,
                    "doc_ids_by_title_norm": self.doc_ids_by_title_norm,
                    "title_keywords_by_doc": self.title_keywords_by_doc,
                    "doc_links": self.doc_links,
                    "entity_idf": self.entity_idf,
                    "settings": {
                        "max_depth": self.max_depth,
                        "beam_size": self.beam_size,
                        "max_start_entities": self.max_start_entities,
                        "max_start_docs": self.max_start_docs,
                        "max_entity_degree": self.max_entity_degree,
                        "max_doc_entity_neighbors": self.max_doc_entity_neighbors,
                        "max_doc_sentence_neighbors": self.max_doc_sentence_neighbors,
                        "max_sentence_entity_neighbors": self.max_sentence_entity_neighbors,
                        "max_title_link_neighbors": self.max_title_link_neighbors,
                        "path_top_n": self.path_top_n,
                        "min_entity_chars": self.min_entity_chars,
                        "score_weights": self.score_weights,
                    },
                    "graph_stats": self.graph_stats(),
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "GraphPathSearcher":
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        settings = payload["settings"]
        return cls(
            corpus=payload["corpus"],
            sentence_nodes_by_doc=payload["sentence_nodes_by_doc"],
            entities_by_doc=payload["entities_by_doc"],
            docs_by_entity=payload["docs_by_entity"],
            sentences_by_entity=payload["sentences_by_entity"],
            doc_keywords=payload["doc_keywords"],
            doc_relations=payload["doc_relations"],
            title_norm_by_doc=payload["title_norm_by_doc"],
            doc_ids_by_title_norm=payload["doc_ids_by_title_norm"],
            title_keywords_by_doc=payload["title_keywords_by_doc"],
            doc_links=payload["doc_links"],
            entity_idf=payload["entity_idf"],
            max_depth=int(settings["max_depth"]),
            beam_size=int(settings["beam_size"]),
            max_start_entities=int(settings["max_start_entities"]),
            max_start_docs=int(settings["max_start_docs"]),
            max_entity_degree=int(settings["max_entity_degree"]),
            max_doc_entity_neighbors=int(settings["max_doc_entity_neighbors"]),
            max_doc_sentence_neighbors=int(settings["max_doc_sentence_neighbors"]),
            max_sentence_entity_neighbors=int(settings["max_sentence_entity_neighbors"]),
            max_title_link_neighbors=int(settings["max_title_link_neighbors"]),
            path_top_n=int(settings["path_top_n"]),
            min_entity_chars=int(settings["min_entity_chars"]),
            score_weights={key: float(value) for key, value in settings["score_weights"].items()},
        )

    def graph_stats(self) -> dict[str, int | float]:
        sentence_count = sum(len(rows) for rows in self.sentence_nodes_by_doc.values())
        entity_doc_edges = sum(len(doc_ids) for doc_ids in self.docs_by_entity.values())
        sentence_entity_edges = sum(len(rows) for rows in self.sentences_by_entity.values())
        doc_sentence_edges = sentence_count
        doc_doc_edges = sum(len(targets) for targets in self.doc_links.values())
        relation_sentence_edges = sum(
            len(sentence.relations)
            for sentences in self.sentence_nodes_by_doc.values()
            for sentence in sentences
        )
        edge_count = (
            entity_doc_edges
            + sentence_entity_edges
            + doc_sentence_edges
            + doc_doc_edges
            + relation_sentence_edges
        )
        return {
            "documents": len(self.corpus),
            "sentences": sentence_count,
            "entities": len(self.docs_by_entity),
            "relations": len(RELATION_LEXICON),
            "entity_doc_edges": entity_doc_edges,
            "sentence_entity_edges": sentence_entity_edges,
            "doc_sentence_edges": doc_sentence_edges,
            "doc_doc_title_link_edges": doc_doc_edges,
            "relation_sentence_edges": relation_sentence_edges,
            "total_edges_estimate": edge_count,
            "avg_entities_per_doc": entity_doc_edges / max(1, len(self.corpus)),
        }

    def search(
        self,
        *,
        question_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchOutput:
        start_ms = now_ms()
        signals = self.extract_query_signals(query)
        start_states = self.build_start_states(signals)
        beam = start_states[: self.beam_size]
        completed: list[GraphPathState] = list(beam)

        for _depth in range(self.max_depth):
            expanded: list[GraphPathState] = []
            for state in beam:
                expanded.extend(self.expand_state(state, signals))
            if not expanded:
                break
            ranked = sorted(expanded, key=lambda item: (-self.final_path_score(item, signals), path_tiebreak(item)))
            beam = ranked[: self.beam_size]
            completed.extend(beam)

        path_candidates = self.rank_completed_paths(completed, signals)
        ranked_doc_ids, doc_scores, doc_metadata = self.rank_documents_from_paths(
            path_candidates=path_candidates,
            signals=signals,
            top_k=top_k,
        )

        docs: list[RetrievedDocument] = []
        text_read_tokens = 0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            doc = self.corpus[self.doc_index_by_id[doc_id]]
            text_read_tokens += estimate_tokens(doc.full_text)
            docs.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=float(doc_scores.get(doc_id, 0.0)),
                    rank=rank,
                    tool="graph_path",
                    metadata=doc_metadata.get(doc_id, {}),
                )
            )

        return SearchOutput(
            question_id=question_id,
            tool="graph_path",
            ranked_documents=docs,
            latency_ms=now_ms() - start_ms,
            text_read_tokens=text_read_tokens,
            trace=[
                "link_query_to_graph_nodes",
                f"start_states:{len(start_states)}",
                f"beam_size:{self.beam_size}",
                f"max_depth:{self.max_depth}",
                "rank_explicit_graph_paths",
                f"return_top_{top_k}",
            ],
            metadata={
                "query_entities": sorted(signals.entities),
                "query_numbers_dates": sorted(signals.numbers_dates),
                "query_relations": sorted(signals.relations),
                "path_count": len(path_candidates),
                "graph_stats": self.graph_stats(),
            },
        )

    def extract_query_signals(self, query: str) -> QuerySignals:
        query_norm = normalize_text(query)
        query_keywords = set(keywords(query, max_keywords=32))
        query_entities = {
            entity
            for entity in simple_entities(query)
            if len(entity) >= self.min_entity_chars
        }
        for token in query_keywords:
            if token in self.docs_by_entity:
                query_entities.add(token)
        query_numbers = set(numbers_and_dates(query))
        query_entities.update(value for value in query_numbers if value in self.docs_by_entity)
        phrases = [phrase for phrase in quoted_phrases(query) if normalize_text(phrase)]
        return QuerySignals(
            query=query,
            normalized=query_norm,
            keywords=query_keywords,
            entities=query_entities,
            numbers_dates=query_numbers,
            quoted_phrases=phrases,
            relations=relation_cues(query),
        )

    def build_start_states(self, signals: QuerySignals) -> list[GraphPathState]:
        starts: list[GraphPathState] = []
        seen_nodes: set[NodeId] = set()

        entity_scores = []
        for entity in signals.entities:
            if entity not in self.docs_by_entity:
                continue
            degree = len(self.docs_by_entity[entity])
            if degree > self.max_entity_degree:
                continue
            score = self.score_weights["start_entity"] * self.entity_specificity(entity)
            score += self.query_entity_bonus(entity, signals)
            score -= self.high_degree_penalty(entity)
            entity_scores.append((score, entity))

        for score, entity in sorted(entity_scores, key=lambda item: (-item[0], item[1]))[: self.max_start_entities]:
            node = entity_node(entity)
            seen_nodes.add(node)
            starts.append(
                GraphPathState(
                    nodes=(node,),
                    score=score,
                    entities=(entity,),
                    support=({"start": "query_entity", "entity": entity, "score": score},),
                )
            )

        for doc_id, score, reason in self.query_linked_docs(signals):
            node = doc_node(doc_id)
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            starts.append(
                GraphPathState(
                    nodes=(node,),
                    score=score,
                    doc_ids=(doc_id,),
                    support=({"start": reason, "doc_id": doc_id, "score": score},),
                )
            )

        if starts:
            return sorted(starts, key=lambda item: (-item.score, path_tiebreak(item)))

        fallback_docs = self.query_fallback_docs(signals)
        return [
            GraphPathState(
                nodes=(doc_node(doc_id),),
                score=score,
                doc_ids=(doc_id,),
                support=({"start": "query_keyword_title_fallback", "doc_id": doc_id, "score": score},),
            )
            for doc_id, score in fallback_docs[: self.max_start_docs]
        ]

    def query_linked_docs(self, signals: QuerySignals) -> list[tuple[str, float, str]]:
        candidates: dict[str, tuple[float, str]] = {}
        query_keywords = signals.keywords
        for doc in self.corpus:
            doc_id = doc.doc_id
            title_norm = self.title_norm_by_doc.get(doc_id, "")
            title_keywords = self.title_keywords_by_doc.get(doc_id, set())
            score = 0.0
            reason = "query_title_overlap"

            if title_norm and title_norm in signals.normalized:
                score += self.score_weights["start_doc"] + self.score_weights["title_match"]
                reason = "query_exact_title"
            phrase_hits = [
                phrase
                for phrase in signals.quoted_phrases
                if normalize_text(phrase) and normalize_text(phrase) in title_norm
            ]
            if phrase_hits:
                score += self.score_weights["title_match"] * len(phrase_hits)
                reason = "query_quoted_title"
            if title_keywords and query_keywords:
                overlap = len(title_keywords.intersection(query_keywords)) / len(title_keywords)
                if overlap >= 0.45:
                    score += self.score_weights["start_doc"] * overlap

            if score <= 0.0:
                continue
            candidates[doc_id] = (score, reason)

        ranked = sorted(
            ((doc_id, score_reason[0], score_reason[1]) for doc_id, score_reason in candidates.items()),
            key=lambda item: (-item[1], self.doc_title(item[0]), item[0]),
        )
        return ranked[: self.max_start_docs]

    def query_fallback_docs(self, signals: QuerySignals) -> list[tuple[str, float]]:
        rows = []
        for doc in self.corpus:
            doc_id = doc.doc_id
            coverage = self.doc_query_coverage(doc_id, signals)
            if coverage <= 0.0:
                continue
            relation = self.doc_relation_coverage(doc_id, signals)
            score = self.score_weights["query_coverage"] * coverage + self.score_weights["relation_coverage"] * relation
            rows.append((doc_id, score))
        return sorted(rows, key=lambda item: (-item[1], self.doc_title(item[0]), item[0]))

    def expand_state(self, state: GraphPathState, signals: QuerySignals) -> list[GraphPathState]:
        kind, value = state.last_node
        if kind == "entity":
            return self.expand_entity_state(state, value, signals)
        if kind == "doc":
            return self.expand_doc_state(state, value, signals)
        if kind == "sent":
            doc_id, sent_id_text = value.rsplit("::", 1)
            return self.expand_sentence_state(state, doc_id, int(sent_id_text), signals)
        return []

    def expand_entity_state(
        self,
        state: GraphPathState,
        entity: str,
        signals: QuerySignals,
    ) -> list[GraphPathState]:
        doc_ids = self.docs_by_entity.get(entity, set())
        if not doc_ids or len(doc_ids) > self.max_entity_degree:
            return []
        rows = []
        for doc_id in doc_ids:
            node = doc_node(doc_id)
            if node in state.nodes:
                continue
            edge_score = self.score_weights["entity_to_doc"] * self.entity_specificity(entity)
            edge_score += self.doc_query_coverage(doc_id, signals) * self.score_weights["query_coverage"]
            edge_score += self.doc_relation_coverage(doc_id, signals) * self.score_weights["relation_coverage"]
            if self.title_norm_by_doc.get(doc_id) == entity:
                edge_score += self.score_weights["title_match"]
            edge_score -= self.high_degree_penalty(entity)
            rows.append(self.extend_state(state, node, edge_score, doc_id=doc_id, entity=entity, support_kind="entity_to_doc"))
        return sorted(rows, key=lambda item: (-item.score, path_tiebreak(item)))[: self.beam_size]

    def expand_doc_state(
        self,
        state: GraphPathState,
        doc_id: str,
        signals: QuerySignals,
    ) -> list[GraphPathState]:
        rows: list[GraphPathState] = []
        for link_doc_id in self.rank_title_link_neighbors(doc_id, signals):
            node = doc_node(link_doc_id)
            if node in state.nodes:
                continue
            edge_score = self.score_weights["doc_to_doc"]
            edge_score += self.doc_query_coverage(link_doc_id, signals) * self.score_weights["query_coverage"]
            edge_score += self.doc_relation_coverage(link_doc_id, signals) * self.score_weights["relation_coverage"]
            rows.append(
                self.extend_state(
                    state,
                    node,
                    edge_score,
                    doc_id=link_doc_id,
                    support_kind="doc_title_link",
                )
            )

        for sentence in self.rank_doc_sentences(doc_id, signals):
            node = sent_node(doc_id, sentence.sent_id)
            if node in state.nodes:
                continue
            sentence_score = self.sentence_query_score(sentence, signals)
            edge_score = self.score_weights["doc_to_sentence"] * sentence_score
            rows.append(
                self.extend_state(
                    state,
                    node,
                    edge_score,
                    relations=sentence.relations,
                    support_kind="doc_to_sentence",
                    support_extra={"sent_id": sentence.sent_id},
                )
            )

        for entity in self.rank_doc_entities(doc_id, signals):
            node = entity_node(entity)
            if node in state.nodes:
                continue
            edge_score = self.score_weights["sentence_to_entity"] * self.entity_specificity(entity)
            edge_score += self.query_entity_bonus(entity, signals)
            edge_score -= self.high_degree_penalty(entity)
            rows.append(self.extend_state(state, node, edge_score, entity=entity, support_kind="doc_to_entity"))

        return sorted(rows, key=lambda item: (-item.score, path_tiebreak(item)))[: self.beam_size]

    def expand_sentence_state(
        self,
        state: GraphPathState,
        doc_id: str,
        sent_id: int,
        signals: QuerySignals,
    ) -> list[GraphPathState]:
        sentence = self.sentence_nodes_by_doc[doc_id][sent_id]
        rows = []
        if doc_node(doc_id) not in state.nodes:
            rows.append(
                self.extend_state(
                    state,
                    doc_node(doc_id),
                    self.score_weights["doc_to_sentence"] * 0.50,
                    doc_id=doc_id,
                    support_kind="sentence_to_doc",
                )
            )

        ranked_entities = sorted(
            sentence.entities,
            key=lambda entity: (
                -self.sentence_entity_score(entity, sentence, signals),
                entity,
            ),
        )[: self.max_sentence_entity_neighbors]
        for entity in ranked_entities:
            node = entity_node(entity)
            if node in state.nodes:
                continue
            edge_score = self.sentence_entity_score(entity, sentence, signals)
            rows.append(
                self.extend_state(
                    state,
                    node,
                    edge_score,
                    entity=entity,
                    relations=sentence.relations,
                    support_kind="sentence_to_entity",
                    support_extra={"sent_id": sent_id},
                )
            )
        return sorted(rows, key=lambda item: (-item.score, path_tiebreak(item)))[: self.beam_size]

    def extend_state(
        self,
        state: GraphPathState,
        node: NodeId,
        edge_score: float,
        *,
        doc_id: str | None = None,
        entity: str | None = None,
        relations: set[str] | None = None,
        support_kind: str,
        support_extra: dict[str, Any] | None = None,
    ) -> GraphPathState:
        doc_ids = append_unique(state.doc_ids, doc_id)
        entities = append_unique(state.entities, entity)
        relation_values = state.relations
        if relations:
            for relation in sorted(relations):
                relation_values = append_unique(relation_values, relation)
        support = {
            "edge": support_kind,
            "to_node": node_to_text(node),
            "edge_score": edge_score,
        }
        if doc_id is not None:
            support["doc_id"] = doc_id
        if entity is not None:
            support["entity"] = entity
        if relations:
            support["relations"] = sorted(relations)
        if support_extra:
            support.update(support_extra)
        return GraphPathState(
            nodes=state.nodes + (node,),
            score=state.score + edge_score - self.score_weights["path_length_penalty"],
            doc_ids=doc_ids,
            entities=entities,
            relations=relation_values,
            support=state.support + (support,),
        )

    def rank_completed_paths(
        self,
        paths: list[GraphPathState],
        signals: QuerySignals,
    ) -> list[GraphPathCandidate]:
        unique_paths: dict[tuple[NodeId, ...], GraphPathState] = {}
        for path in paths:
            if path.doc_ids:
                current = unique_paths.get(path.nodes)
                if current is None or path.score > current.score:
                    unique_paths[path.nodes] = path

        candidates = []
        for path in unique_paths.values():
            query_coverage = self.path_query_coverage(path, signals)
            relation_coverage = self.path_relation_coverage(path, signals)
            final_score = self.final_path_score(path, signals)
            candidates.append(
                GraphPathCandidate(
                    path=path,
                    final_score=final_score,
                    query_coverage=query_coverage,
                    relation_coverage=relation_coverage,
                )
            )
        return sorted(
            candidates,
            key=lambda item: (-item.final_score, path_tiebreak(item.path)),
        )[: self.path_top_n]

    def final_path_score(self, path: GraphPathState, signals: QuerySignals) -> float:
        query_coverage = self.path_query_coverage(path, signals)
        relation_coverage = self.path_relation_coverage(path, signals)
        multi_doc_bonus = self.score_weights["multi_doc_bonus"] if len(set(path.doc_ids)) >= 2 else 0.0
        return (
            path.score
            + self.score_weights["query_coverage"] * query_coverage
            + self.score_weights["relation_coverage"] * relation_coverage
            + multi_doc_bonus
        )

    def rank_documents_from_paths(
        self,
        *,
        path_candidates: list[GraphPathCandidate],
        signals: QuerySignals,
        top_k: int,
    ) -> tuple[list[str], dict[str, float], dict[str, dict[str, Any]]]:
        doc_scores: dict[str, float] = {}
        doc_metadata: dict[str, dict[str, Any]] = {}

        for path_rank, candidate in enumerate(path_candidates, start=1):
            for doc_position, doc_id in enumerate(candidate.path.doc_ids):
                position_bonus = 1.0 / (1.0 + doc_position)
                doc_score = candidate.final_score + position_bonus
                doc_score += self.doc_query_coverage(doc_id, signals) * self.score_weights["query_coverage"]
                doc_score += self.doc_relation_coverage(doc_id, signals) * self.score_weights["relation_coverage"]
                if doc_score > doc_scores.get(doc_id, float("-inf")):
                    doc_scores[doc_id] = doc_score
                    doc_metadata[doc_id] = {
                        "source": "graph_path",
                        "best_path_rank": path_rank,
                        "best_path_score": candidate.final_score,
                        "query_coverage": candidate.query_coverage,
                        "relation_coverage": candidate.relation_coverage,
                        "path_nodes": [node_to_text(node) for node in candidate.path.nodes],
                        "path_doc_ids": list(candidate.path.doc_ids),
                        "path_entities": list(candidate.path.entities),
                        "path_relations": list(candidate.path.relations),
                        "support": list(candidate.path.support),
                    }

        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for candidate in path_candidates:
            for doc_id in candidate.path.doc_ids:
                add_doc_id(ranked_doc_ids, seen, doc_id, top_k=top_k)
                if len(ranked_doc_ids) >= top_k:
                    return ranked_doc_ids, doc_scores, doc_metadata

        for doc_id, _score in sorted(
            doc_scores.items(),
            key=lambda item: (-item[1], self.doc_title(item[0]), item[0]),
        ):
            add_doc_id(ranked_doc_ids, seen, doc_id, top_k=top_k)
            if len(ranked_doc_ids) >= top_k:
                return ranked_doc_ids, doc_scores, doc_metadata

        for doc_id, score in self.query_fallback_docs(signals):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = score
                doc_metadata[doc_id] = {
                    "source": "query_graph_fallback",
                    "query_coverage": self.doc_query_coverage(doc_id, signals),
                    "relation_coverage": self.doc_relation_coverage(doc_id, signals),
                }
            add_doc_id(ranked_doc_ids, seen, doc_id, top_k=top_k)
            if len(ranked_doc_ids) >= top_k:
                return ranked_doc_ids, doc_scores, doc_metadata

        return ranked_doc_ids, doc_scores, doc_metadata

    def rank_title_link_neighbors(self, doc_id: str, signals: QuerySignals) -> list[str]:
        targets = self.doc_links.get(doc_id, set())
        return sorted(
            targets,
            key=lambda target: (
                -self.doc_query_coverage(target, signals),
                -self.doc_relation_coverage(target, signals),
                self.doc_title(target),
                target,
            ),
        )[: self.max_title_link_neighbors]

    def rank_doc_sentences(self, doc_id: str, signals: QuerySignals) -> list[SentenceNode]:
        sentences = self.sentence_nodes_by_doc.get(doc_id, [])
        ranked = sorted(
            sentences,
            key=lambda sentence: (-self.sentence_query_score(sentence, signals), sentence.sent_id),
        )
        useful = [
            sentence
            for sentence in ranked
            if self.sentence_query_score(sentence, signals) > 0.0
            or sentence.entities.intersection(self.entities_by_doc.get(doc_id, set()))
        ]
        return useful[: self.max_doc_sentence_neighbors]

    def rank_doc_entities(self, doc_id: str, signals: QuerySignals) -> list[str]:
        entities = self.entities_by_doc.get(doc_id, set())
        ranked = sorted(
            entities,
            key=lambda entity: (
                -self.doc_entity_score(entity, doc_id, signals),
                entity,
            ),
        )
        return [
            entity
            for entity in ranked
            if len(self.docs_by_entity.get(entity, set())) <= self.max_entity_degree
        ][: self.max_doc_entity_neighbors]

    def sentence_query_score(self, sentence: SentenceNode, signals: QuerySignals) -> float:
        keyword_overlap = len(sentence.keywords.intersection(signals.keywords)) / max(1, len(signals.keywords))
        entity_overlap = len(sentence.entities.intersection(signals.entities)) / max(1, len(signals.entities))
        relation_overlap = len(sentence.relations.intersection(signals.relations)) / max(1, len(signals.relations))
        return keyword_overlap + 0.70 * entity_overlap + 0.55 * relation_overlap

    def sentence_entity_score(self, entity: str, sentence: SentenceNode, signals: QuerySignals) -> float:
        score = self.score_weights["sentence_to_entity"] * self.entity_specificity(entity)
        score += self.query_entity_bonus(entity, signals)
        if sentence.relations.intersection(signals.relations):
            score += self.score_weights["relation_coverage"] * 0.60
        if sentence.keywords.intersection(signals.keywords):
            score += self.score_weights["query_coverage"] * 0.30
        score -= self.high_degree_penalty(entity)
        return score

    def doc_entity_score(self, entity: str, doc_id: str, signals: QuerySignals) -> float:
        score = self.entity_specificity(entity)
        score += self.query_entity_bonus(entity, signals)
        if self.title_norm_by_doc.get(doc_id) == entity:
            score += self.score_weights["title_match"]
        if set(entity.split()).intersection(signals.keywords):
            score += 0.15
        score -= self.high_degree_penalty(entity)
        return score

    def query_entity_bonus(self, entity: str, signals: QuerySignals) -> float:
        if entity in signals.entities:
            return 0.55
        entity_tokens = set(entity.split())
        if entity_tokens and entity_tokens.intersection(signals.keywords):
            return 0.20 * len(entity_tokens.intersection(signals.keywords)) / len(entity_tokens)
        return 0.0

    def entity_specificity(self, entity: str) -> float:
        max_idf = math.log(1.0 + len(self.corpus)) + 1.0
        return self.entity_idf.get(entity, 0.0) / max(1.0, max_idf)

    def high_degree_penalty(self, entity: str) -> float:
        degree = len(self.docs_by_entity.get(entity, set()))
        if degree <= 1:
            return 0.0
        return self.score_weights["high_degree_penalty"] * math.log1p(degree) / math.log1p(len(self.corpus))

    def doc_query_coverage(self, doc_id: str, signals: QuerySignals) -> float:
        if not signals.keywords:
            return 0.0
        coverage = len(self.doc_keywords.get(doc_id, set()).intersection(signals.keywords)) / len(signals.keywords)
        title_keywords = self.title_keywords_by_doc.get(doc_id, set())
        if title_keywords:
            coverage += 0.30 * len(title_keywords.intersection(signals.keywords)) / len(title_keywords)
        if self.title_norm_by_doc.get(doc_id, "") in signals.normalized:
            coverage += 0.30
        return min(1.0, coverage)

    def doc_relation_coverage(self, doc_id: str, signals: QuerySignals) -> float:
        if not signals.relations:
            return 0.0
        return len(self.doc_relations.get(doc_id, set()).intersection(signals.relations)) / len(signals.relations)

    def path_query_coverage(self, path: GraphPathState, signals: QuerySignals) -> float:
        if not signals.keywords:
            return 0.0
        path_keywords = set()
        for doc_id in path.doc_ids:
            path_keywords.update(self.doc_keywords.get(doc_id, set()))
        for entity in path.entities:
            path_keywords.update(entity.split())
        return min(1.0, len(path_keywords.intersection(signals.keywords)) / len(signals.keywords))

    def path_relation_coverage(self, path: GraphPathState, signals: QuerySignals) -> float:
        if not signals.relations:
            return 0.0
        path_relations = set(path.relations)
        for doc_id in path.doc_ids:
            path_relations.update(self.doc_relations.get(doc_id, set()))
        return len(path_relations.intersection(signals.relations)) / len(signals.relations)

    def doc_title(self, doc_id: str) -> str:
        return self.corpus[self.doc_index_by_id[doc_id]].title


def build_sentence_nodes(doc: CorpusDocument, *, min_entity_chars: int) -> list[SentenceNode]:
    sentences = doc.sentences or [doc.full_text]
    rows = []
    for sent_id, sentence in enumerate(sentences):
        entities = {
            entity
            for entity in simple_entities(sentence)
            if len(entity) >= min_entity_chars and not entity.isdigit()
        }
        for value in numbers_and_dates(sentence):
            if len(value) >= min_entity_chars:
                entities.add(value)
        rows.append(
            SentenceNode(
                doc_id=doc.doc_id,
                sent_id=sent_id,
                entities=entities,
                keywords=set(keywords(sentence, max_keywords=32)),
                relations=relation_cues(sentence),
            )
        )
    return rows


def extract_document_entities(
    doc: CorpusDocument,
    *,
    sentence_nodes: list[SentenceNode],
    min_entity_chars: int,
) -> set[str]:
    text = f"{doc.title}. {doc.full_text}"
    entities = set(simple_entities(text))
    title_norm = normalize_text(doc.title)
    if title_norm:
        entities.add(title_norm)
    entities.update(entity for sentence in sentence_nodes for entity in sentence.entities)
    for token in keywords(doc.title, max_keywords=12):
        if len(token) >= min_entity_chars:
            entities.add(token)
    for value in numbers_and_dates(text):
        if len(value) >= min_entity_chars:
            entities.add(value)
    return {
        entity
        for entity in entities
        if len(entity) >= min_entity_chars
    }


def build_title_links(
    *,
    corpus: list[CorpusDocument],
    title_norm_by_doc: dict[str, str],
    doc_ids_by_title_norm: dict[str, str],
    max_links_per_doc: int,
) -> dict[str, set[str]]:
    links: dict[str, set[str]] = {doc.doc_id: set() for doc in corpus}
    title_rows = [
        (title_norm, doc_id)
        for title_norm, doc_id in doc_ids_by_title_norm.items()
        if len(title_norm) >= 4
    ]
    for doc in corpus:
        text_norm = normalize_text(doc.full_text)
        matches = []
        for title_norm, target_doc_id in title_rows:
            if target_doc_id == doc.doc_id:
                continue
            if title_norm and title_norm in text_norm:
                matches.append((len(title_norm.split()), title_norm, target_doc_id))
        ranked = sorted(matches, key=lambda item: (-item[0], item[1], item[2]))[:max_links_per_doc]
        links[doc.doc_id].update(target_doc_id for _length, _title, target_doc_id in ranked)
    return links


def relation_cues(text: str) -> set[str]:
    tokens = set(normalize_text(text).split())
    cues = set()
    for cue_name, lexemes in RELATION_LEXICON.items():
        if tokens.intersection(lexemes):
            cues.add(cue_name)
    return cues


def doc_node(doc_id: str) -> NodeId:
    return ("doc", doc_id)


def entity_node(entity: str) -> NodeId:
    return ("entity", entity)


def sent_node(doc_id: str, sent_id: int) -> NodeId:
    return ("sent", f"{doc_id}::{sent_id}")


def node_to_text(node: NodeId) -> str:
    return f"{node[0]}::{node[1]}"


def append_unique(values: tuple[str, ...], value: str | None) -> tuple[str, ...]:
    if value is None or value in values:
        return values
    return values + (value,)


def path_tiebreak(path: GraphPathState) -> str:
    return " > ".join(node_to_text(node) for node in path.nodes)


def add_doc_id(ranked: list[str], seen: set[str], doc_id: str, *, top_k: int) -> None:
    if doc_id in seen or len(ranked) >= top_k:
        return
    seen.add(doc_id)
    ranked.append(doc_id)
