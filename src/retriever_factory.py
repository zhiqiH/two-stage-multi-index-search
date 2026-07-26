from __future__ import annotations

from pathlib import Path

from .common import CorpusDocument, SearchOutput, load_public_corpus
from .config import load_config
from .dense_search import DenseSearcher
from .file_fts_search import FileFTSSearcher
from .graph_path_search import GraphPathSearcher


class RetrieverBundle:
    def __init__(self, *, corpus: list[CorpusDocument], searchers: dict[str, object]) -> None:
        self.corpus = corpus
        self.searchers = searchers

    def search(
        self,
        *,
        tool: str,
        question_id: str,
        query: str,
        top_k: int,
    ) -> SearchOutput:
        if tool not in self.searchers:
            raise KeyError(f"Retriever is not available: {tool}")
        searcher = self.searchers[tool]
        return searcher.search(question_id=question_id, query=query, top_k=top_k)


def build_retriever_bundle(
    *,
    config_path: str | Path = "config.yaml",
    tools: list[str],
    dense_provider: str | None = None,
    save_indexes: bool = False,
    rebuild_dense: bool = False,
) -> tuple[RetrieverBundle, dict]:
    config = load_config(config_path)
    corpus = load_public_corpus(config["paths"]["corpus_path"])
    searchers: dict[str, object] = {}

    if "dense" in tools:
        searchers["dense"] = build_dense_searcher(
            config=config,
            corpus=corpus,
            dense_provider=dense_provider,
            save_indexes=save_indexes,
            rebuild_dense=rebuild_dense,
        )

    if "graph_path" in tools:
        settings = config["retrievers"]["graph_path"]
        searchers["graph_path"] = GraphPathSearcher.build(
            corpus,
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
        if save_indexes:
            searchers["graph_path"].save(Path(config["paths"]["indexes_dir"]) / "graph_path.pkl")

    if "file_fts" in tools:
        settings = config["retrievers"]["file_fts"]
        searchers["file_fts"] = FileFTSSearcher.build(
            corpus,
            chunk_sentences=int(settings["chunk_sentences"]),
            chunk_stride=int(settings["chunk_stride"]),
            candidate_chunks=int(settings["candidate_chunks"]),
            max_query_terms=int(settings["max_query_terms"]),
            max_chunks_per_doc=int(settings["max_chunks_per_doc"]),
            title_weight=float(settings["title_weight"]),
            body_weight=float(settings["body_weight"]),
            entity_weight=float(settings["entity_weight"]),
            number_date_weight=float(settings["number_date_weight"]),
            chunk_sum_weight=float(settings["chunk_sum_weight"]),
            title_match_bonus=float(settings["title_match_bonus"]),
            exact_title_bonus=float(settings["exact_title_bonus"]),
            exact_phrase_bonus=float(settings["exact_phrase_bonus"]),
            number_date_bonus=float(settings["number_date_bonus"]),
            title_overlap_bonus=float(settings["title_overlap_bonus"]),
            max_open_files=int(settings["max_open_files"]),
        )
        if save_indexes:
            searchers["file_fts"].save_index(Path(config["paths"]["indexes_dir"]) / "file_fts.sqlite")

    missing = sorted(set(tools) - set(searchers))
    if missing:
        raise KeyError(f"Unsupported retriever(s): {', '.join(missing)}")

    return RetrieverBundle(corpus=corpus, searchers=searchers), config


def build_dense_searcher(
    *,
    config: dict,
    corpus: list[CorpusDocument],
    dense_provider: str | None,
    save_indexes: bool,
    rebuild_dense: bool,
) -> DenseSearcher:
    settings = config["retrievers"]["dense"]
    provider = dense_provider or str(settings["provider"])
    indexes_dir = Path(config["paths"]["indexes_dir"])
    embeddings_path = indexes_dir / "dense_embeddings.npy"
    doc_ids_path = indexes_dir / "dense_doc_ids.json"
    common_kwargs = {
        "provider": provider,
        "model_name": str(settings["model_name"]),
        "batch_size": int(settings["batch_size"]),
        "normalize_embeddings": bool(settings["normalize_embeddings"]),
        "max_document_tokens": int(settings["max_document_tokens"]),
        "base_url": str(settings.get("base_url") or ""),
        "api_key_env": str(settings.get("api_key_env") or "DEEPINFRA_TOKEN"),
    }

    if not rebuild_dense and embeddings_path.exists() and doc_ids_path.exists():
        return DenseSearcher.load(
            corpus=corpus,
            embeddings_path=embeddings_path,
            doc_ids_path=doc_ids_path,
            **common_kwargs,
        )

    searcher = DenseSearcher.build(corpus, **common_kwargs)
    if save_indexes:
        searcher.save(embeddings_path=embeddings_path, doc_ids_path=doc_ids_path)
    return searcher
