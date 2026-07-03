from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from jsonschema import Draft202012Validator


def load_schema(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_errors(item: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [e.message for e in validator.iter_errors(item)]


def is_valid_schema(item: dict[str, Any], schema: dict[str, Any]) -> bool:
    return not schema_errors(item, schema)


def validation_quality(card: dict[str, Any]) -> float:
    validation = card.get("validation", {}) or {}
    vals = []
    for key in ("faithfulness", "edos_alignment", "usefulness"):
        v = validation.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v) / 5.0)
    if not vals:
        # Candidate or unvalidated cards get neutral-low score, not zero.
        return 0.35
    return sum(vals) / len(vals)
