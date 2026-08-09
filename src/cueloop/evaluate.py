from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .core import Action


@dataclass(frozen=True)
class SessionMetrics:
    turns: int
    silence: int
    continue_count: int
    cue_count: int
    intervention_rate: float
    silence_rate: float
    duration_ms: int
    interventions_per_minute: float
    low_asr_suppression_count: int
    time_cooldown_suppression_count: int
    turn_cooldown_suppression_count: int


def evaluate_records(records: Iterable[dict]) -> SessionMetrics:
    rows = list(records)
    turns = len(rows)
    silence = 0
    continue_count = 0
    cue_count = 0
    low_asr = 0
    time_cooldown = 0
    turn_cooldown = 0
    end_ms = 0

    for row in rows:
        end_ms = max(end_ms, int(row.get("at_ms", 0)))
        decision = row.get("decision", {})
        action = decision.get("action")
        reason = str(decision.get("reason", ""))

        if action == Action.SILENCE.value:
            silence += 1
        elif action == Action.CONTINUE.value:
            continue_count += 1
        elif action == Action.CUE.value:
            cue_count += 1

        analysis = decision.get("analysis", {})
        evidence = " ".join(str(item) for item in analysis.get("evidence", []))
        if "ASR confidence" in evidence:
            low_asr += 1
        if "time cooldown" in reason:
            time_cooldown += 1
        if "turn cooldown" in reason:
            turn_cooldown += 1

    interventions = continue_count + cue_count
    intervention_rate = interventions / turns if turns else 0.0
    silence_rate = silence / turns if turns else 0.0
    minutes = end_ms / 60_000 if end_ms > 0 else 0.0
    interventions_per_minute = interventions / minutes if minutes > 0 else 0.0

    return SessionMetrics(
        turns=turns,
        silence=silence,
        continue_count=continue_count,
        cue_count=cue_count,
        intervention_rate=intervention_rate,
        silence_rate=silence_rate,
        duration_ms=end_ms,
        interventions_per_minute=interventions_per_minute,
        low_asr_suppression_count=low_asr,
        time_cooldown_suppression_count=time_cooldown,
        turn_cooldown_suppression_count=turn_cooldown,
    )


def load_timeline(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: timeline row must be an object")
            records.append(payload)
    return records


def evaluate_timeline(path: Path) -> SessionMetrics:
    return evaluate_records(load_timeline(path))
