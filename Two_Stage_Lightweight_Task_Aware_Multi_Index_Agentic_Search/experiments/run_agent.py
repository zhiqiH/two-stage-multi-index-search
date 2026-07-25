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
from src.anti_contamination import assert_no_private_keys
from src.common import load_public_questions, write_jsonl
from src.config import load_config
from src.retriever_factory import build_retriever_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run two-stage dynamic agent.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", choices=["dev", "test", "final", "all"], default="final")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dense-provider", choices=["deepinfra", "local"], default=None)
    parser.add_argument("--save-indexes", action="store_true")
    parser.add_argument("--rebuild-dense", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    top_k = int(config["experiment"]["top_k"])
    output_path = args.output or Path(config["paths"]["results_dir"]) / f"agent_{args.split}_predictions.jsonl"
    questions = [
        question
        for question in load_public_questions(config["paths"]["questions_path"])
        if args.split == "all" or question.split == args.split
    ]
    retrievers, _ = build_retriever_bundle(
        config_path=args.config,
        tools=CORE_TOOLS,
        dense_provider=args.dense_provider,
        save_indexes=args.save_indexes,
        rebuild_dense=args.rebuild_dense,
    )
    agent = TwoStageAgent(
        retrievers=retrievers,
        top_k=top_k,
        rrf_k0=int(config["experiment"]["rrf_k0"]),
        tools=CORE_TOOLS,
    )

    rows = []
    for question in questions:
        row = agent.run(question)
        assert_no_private_keys(row, context=f"two_stage_agent:{question.question_id}")
        rows.append(row)
    write_jsonl(output_path, rows)
    print(
        json.dumps(
            {
                "method": "two_stage_agent",
                "split": args.split,
                "questions": len(rows),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
