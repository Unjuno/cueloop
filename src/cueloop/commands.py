from __future__ import annotations

import json


def parse_command_json(raw: str | None, legacy: list[str] | None, *, label: str) -> list[str] | None:
    if raw is not None and legacy:
        raise ValueError(f"use only one of --{label}-command or --{label}-command-json")
    if raw is None:
        return legacy
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} command JSON is invalid") from exc
    if not isinstance(payload, list) or not payload or not all(
        isinstance(item, str) and item for item in payload
    ):
        raise ValueError(f"{label} command JSON must be a non-empty string array")
    return payload
