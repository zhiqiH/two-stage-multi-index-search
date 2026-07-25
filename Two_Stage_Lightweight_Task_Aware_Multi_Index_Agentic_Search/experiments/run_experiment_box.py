from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import CORE_TOOLS, TwoStageAgent
from src.common import now_ms
from src.config import load_config
from src.dense_search import DenseSearcher, encode_texts
from src.experiment_box import ExperimentBox
from src.file_fts_search import FileFTSSearcher
from src.graph_path_search import GraphPathSearcher
from src.retriever_factory import build_retriever_bundle
from src.router import route_question


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run methods inside the anti-contamination experiment box.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--method",
        choices=[
            "dense",
            "graph_path",
            "file_fts",
            "rule_router",
            "two_stage_agent",
        ],
        required=True,
    )
    parser.add_argument("--split", choices=["dev", "test", "final", "all"], default="final")
    parser.add_argument("--output", default=None, help="Optional prediction JSONL output path.")
    parser.add_argument("--no-evaluate", action="store_true", help="Write predictions only.")
    parser.add_argument("--ks", default=None, help="Evaluation cutoffs, e.g. 1,3,5. Defaults to 1,3,5.")
    parser.add_argument("--save-indexes", action="store_true")
    parser.add_argument("--rebuild-dense", action="store_true")
    parser.add_argument("--dense-provider", choices=["deepinfra", "local"], default=None)
    parser.add_argument(
        "--validate-inputs-only",
        action="store_true",
        help="Validate the public runtime view and exit without running a method.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    box = ExperimentBox(config=config, split=args.split)

    if args.validate_inputs_only:
        print(json.dumps(box.validate_runtime_inputs(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.method == "dense":
        result = run_dense(box, config, args)
    elif args.method == "graph_path":
        result = run_graph_path(box, config, args)
    elif args.method == "file_fts":
        result = run_file_fts(box, config, args)
    elif args.method == "rule_router":
        result = run_rule_router(box, config, args)
    elif args.method == "two_stage_agent":
        result = run_two_stage_agent(box, config, args)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported method: {args.method}")

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_dense(box: ExperimentBox, config: dict, args: argparse.Namespace):
    corpus = box.runtime_corpus()
    settings = config["retrievers"]["dense"]
    provider = args.dense_provider or str(settings["provider"])
    embeddings_path = box.indexes_dir / "dense_embeddings.npy"
    doc_ids_path = box.indexes_dir / "dense_doc_ids.json"
    common_kwargs = {
        "provider": provider,
        "model_name": str(settings["model_name"]),
        "batch_size": int(settings["batch_size"]),
        "normalize_embeddings": bool(settings["normalize_embeddings"]),
        "max_document_tokens": int(settings["max_document_tokens"]),
        "base_url": str(settings.get("base_url") or ""),
        "api_key_env": str(settings.get("api_key_env") or "DEEPINFRA_TOKEN"),
    }

    if not args.rebuild_dense and embeddings_path.exists() and doc_ids_path.exists():
        searcher = DenseSearcher.load(
            corpus=corpus,
            embeddings_path=embeddings_path,
            doc_ids_path=doc_ids_path,
            **common_kwargs,
        )
    else:
        searcher = DenseSearcher.build(corpus, **common_kwargs)
        if args.save_indexes:
            searcher.save(embeddings_path=embeddings_path, doc_ids_path=doc_ids_path)

    top_k = int(config["experiment"]["top_k"])
    questions = box.runtime_questions()
    query_start_ms = now_ms()
    query_embeddings = encode_texts(
        [question.question for question in questions],
        provider=provider,
        model_name=str(settings["model_name"]),
        batch_size=int(settings["batch_size"]),
        normalize_embeddings=bool(settings["normalize_embeddings"]),
        base_url=str(settings.get("base_url") or ""),
        api_key_env=str(settings.get("api_key_env") or "DEEPINFRA_TOKEN"),
    )
    query_latency_ms = (now_ms() - query_start_ms) / max(1, len(questions))

    rows = []
    for question, query_embedding in zip(questions, query_embeddings):
        output = searcher.search_with_embedding(
            question_id=question.question_id,
            query_embedding=query_embedding,
            top_k=top_k,
            query_latency_ms=query_latency_ms,
            trace_prefix=f"encode_query_batch:{provider}",
        )
        output.tool = "dense"
        for doc in output.ranked_documents:
            doc.tool = "dense"
        row = output.to_dict()
        row["method"] = "dense"
        row["tool_calls"] = 1
        rows.append(row)

    return box.write_predictions(
        method="dense",
        rows=rows,
        output_path=args.output,
        evaluate=not args.no_evaluate,
        ks=args.ks,
    )


def run_graph_path(box: ExperimentBox, config: dict, args: argparse.Namespace):
    corpus = box.runtime_corpus()
    settings = config["retrievers"]["graph_path"]
    searcher = GraphPathSearcher.build(
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
    if args.save_indexes:
        searcher.save(box.indexes_dir / "graph_path.pkl")
    top_k = int(config["experiment"]["top_k"])

    def predict(question):
        output = searcher.search(question_id=question.question_id, query=question.question, top_k=top_k)
        row = output.to_dict()
        row["method"] = "graph_path"
        row["tool_calls"] = 1
        return row

    return box.run_method(
        method="graph_path",
        predict=predict,
        output_path=args.output,
        evaluate=not args.no_evaluate,
        ks=args.ks,
    )


def run_file_fts(box: ExperimentBox, config: dict, args: argparse.Namespace):
    corpus = box.runtime_corpus()
    settings = config["retrievers"]["file_fts"]
    searcher = FileFTSSearcher.build(
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
    if args.save_indexes:
        searcher.save_index(box.indexes_dir / "file_fts.sqlite")
    top_k = int(config["experiment"]["top_k"])

    def predict(question):
        output = searcher.search(question_id=question.question_id, query=question.question, top_k=top_k)
        row = output.to_dict()
        row["method"] = "file_fts"
        row["tool_calls"] = 1
        return row

    return box.run_method(
        method="file_fts",
        predict=predict,
        output_path=args.output,
        evaluate=not args.no_evaluate,
        ks=args.ks,
    )


def run_rule_router(box: ExperimentBox, config: dict, args: argparse.Namespace):
    bundle, _ = build_retriever_bundle(
        config_path=args.config,
        tools=CORE_TOOLS,
        dense_provider=args.dense_provider,
        save_indexes=args.save_indexes,
        rebuild_dense=args.rebuild_dense,
    )
    top_k = int(config["experiment"]["top_k"])

    def predict(question):
        decision = route_question(question.question)
        output = bundle.search(
            tool=decision.tool,
            question_id=question.question_id,
            query=question.question,
            top_k=top_k,
        )
        row = output.to_dict()
        row["method"] = "rule_router"
        row["tool_calls"] = 1
        row["selected_tool"] = decision.tool
        row["route_decision"] = decision.to_dict()
        return row

    return box.run_method(
        method="rule_router",
        predict=predict,
        output_path=args.output,
        evaluate=not args.no_evaluate,
        ks=args.ks,
    )


def run_two_stage_agent(box: ExperimentBox, config: dict, args: argparse.Namespace):
    bundle, _ = build_retriever_bundle(
        config_path=args.config,
        tools=CORE_TOOLS,
        dense_provider=args.dense_provider,
        save_indexes=args.save_indexes,
        rebuild_dense=args.rebuild_dense,
    )
    agent = TwoStageAgent(
        retrievers=bundle,
        top_k=int(config["experiment"]["top_k"]),
        rrf_k0=int(config["experiment"]["rrf_k0"]),
        tools=CORE_TOOLS,
    )
    return box.run_method(
        method="two_stage_agent",
        predict=agent.run,
        output_path=args.output,
        evaluate=not args.no_evaluate,
        ks=args.ks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
