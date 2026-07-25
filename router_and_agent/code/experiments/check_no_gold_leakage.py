from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anti_contamination import validate_prediction_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate prediction JSONL files for private gold-label leakage.")
    parser.add_argument("paths", nargs="+", help="Prediction JSONL files to validate.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = {}
    for raw_path in args.paths:
        path = Path(raw_path)
        result[str(path)] = {"rows": validate_prediction_file(path), "status": "ok"}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
