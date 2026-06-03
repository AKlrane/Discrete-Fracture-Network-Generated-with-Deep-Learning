from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(base or PROJECT_ROOT) / resolved


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_point(config: dict[str, Any], name: str) -> tuple[float, float]:
    if "x" not in config or "y" not in config:
        raise ValueError(f"{name} must contain x and y")
    return float(config["x"]), float(config["y"])


def format_command_arg(value: str, replacements: dict[str, str]) -> str:
    formatted = value
    for key, replacement in replacements.items():
        formatted = formatted.replace(f"{{{key}}}", replacement)
    return formatted
