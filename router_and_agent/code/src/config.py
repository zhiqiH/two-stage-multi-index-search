from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_project_path(config: dict[str, Any], key: str) -> Path:
    raw_value = config["paths"][key]
    return Path(raw_value)
