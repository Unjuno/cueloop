from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .audio import AudioEvent, BoundaryEvent, BoundaryKind, TranscriptEvent, TranscriptKind


class ReplayFormatError(ValueError):
    pass


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayFormatError(f"{key} must be an integer")
    return value


def event_from_mapping(payload: dict[str, Any]) -> AudioEvent:
    event_type = payload.get("type")
    if event_type == "transcript":
        try:
            kind = TranscriptKind(payload["kind"])
        except (KeyError, ValueError) as exc:
            raise ReplayFormatError("invalid transcript kind") from exc
        text = payload.get("text")
        if not isinstance(text, str):
            raise ReplayFormatError("transcript text must be a string")
        confidence = payload.get("confidence")
        if confidence is not None and not isinstance(confidence, (int, float)):
            raise ReplayFormatError("confidence must be numeric or null")
        return TranscriptEvent(
            kind=kind,
            text=text,
            start_ms=_require_int(payload, "start_ms"),
            end_ms=_require_int(payload, "end_ms"),
            confidence=float(confidence) if confidence is not None else None,
        )

    if event_type == "boundary":
        try:
            kind = BoundaryKind(payload["kind"])
        except (KeyError, ValueError) as exc:
            raise ReplayFormatError("invalid boundary kind") from exc
        silence_ms = payload.get("silence_ms", 0)
        if isinstance(silence_ms, bool) or not isinstance(silence_ms, int):
            raise ReplayFormatError("silence_ms must be an integer")
        return BoundaryEvent(
            kind=kind,
            at_ms=_require_int(payload, "at_ms"),
            silence_ms=silence_ms,
        )

    raise ReplayFormatError("event type must be 'transcript' or 'boundary'")


@dataclass
class JSONLReplaySource:
    """Deterministic event source for recorded or hand-authored audio traces."""

    events: tuple[AudioEvent, ...]
    _index: int = 0

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "JSONLReplaySource":
        events: list[AudioEvent] = []
        previous_time = -1
        for line_number, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReplayFormatError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ReplayFormatError(f"line {line_number}: event must be an object")
            try:
                event = event_from_mapping(payload)
            except ReplayFormatError as exc:
                raise ReplayFormatError(f"line {line_number}: {exc}") from exc
            event_time = event.end_ms if isinstance(event, TranscriptEvent) else event.at_ms
            if event_time < previous_time:
                raise ReplayFormatError(f"line {line_number}: event time moved backwards")
            previous_time = event_time
            events.append(event)
        return cls(tuple(events))

    @classmethod
    def from_path(cls, path: Path) -> "JSONLReplaySource":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_lines(handle)

    def next_event(self) -> AudioEvent | None:
        if self._index >= len(self.events):
            return None
        event = self.events[self._index]
        self._index += 1
        return event
