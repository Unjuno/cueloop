"""Typed values shared by the synchronization loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum


class EventType(StrEnum):
    SPEECH_STARTED = "speech_started"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    SPEECH_ENDED = "speech_ended"
    TICK = "tick"
    SESSION_ENDED = "session_ended"
    INTERVENTIONS_PAUSED = "interventions_paused"
    INTERVENTIONS_RESUMED = "interventions_resumed"
    PARTICIPANT_STOP = "participant_stop"
    ADAPTER_WARNING = "adapter_warning"


@dataclass(frozen=True)
class SyncEvent:
    """One ordered event from the speech adapter."""

    type: EventType
    at_ms: int
    text: str = ""

    def __post_init__(self) -> None:
        if self.at_ms < 0:
            raise ValueError("at_ms must be non-negative")
        if (
            self.type
            in {
                EventType.TRANSCRIPT_PARTIAL,
                EventType.TRANSCRIPT_FINAL,
                EventType.ADAPTER_WARNING,
            }
            and not self.text.strip()
        ):
            raise ValueError(f"{self.type.value} requires non-empty text")

    @classmethod
    def from_json(cls, raw: str) -> SyncEvent:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("event line is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("event must be a JSON object")
        raw_type = payload.get("type")
        if not isinstance(raw_type, str):
            raise TypeError("event requires a string type")
        try:
            event_type = EventType(raw_type)
        except ValueError as exc:
            raise ValueError(f"unknown event type: {raw_type}") from exc
        at_ms = payload.get("at_ms")
        if not isinstance(at_ms, int) or isinstance(at_ms, bool):
            raise TypeError("event requires an integer at_ms")
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise TypeError("event text must be a string")
        return cls(event_type, at_ms, text)


@dataclass(frozen=True)
class Prediction:
    """What the decision model expects the speaker to say next."""

    span: str
    operations: tuple[str, ...]
    confidence: float
    source: str
    created_at_ms: int
    base_length: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("prediction confidence must be between 0 and 1")
        if self.base_length < 0:
            raise ValueError("prediction base_length must be non-negative")


@dataclass(frozen=True)
class SyncState:
    """Snapshot handed to the decision model at a clause boundary."""

    utterance_id: str
    text: str
    context: tuple[str, ...]
    now_ms: int
    started_at_ms: int
    chars_per_second: float
    at_boundary: bool
    prediction: Prediction | None
    prediction_match: int
    diverged: bool
    clause_operations: frozenset[str]
    recent_operations: frozenset[str]


@dataclass(frozen=True)
class Intervention:
    """A scheduled cue that will be spoken if the speaker has not diverged."""

    intervention_id: str
    detected_at_ms: int
    operation: str
    requested_operation: str
    cue: str
    reason: str
    confidence: float
    scheduled_at_ms: int
    expires_at_ms: int
    insertion_delay_ms: int
    assignment: str
    base_text: str
    prediction: Prediction


@dataclass(frozen=True)
class Delivery:
    """The recorded outcome of a scheduled intervention."""

    intervention: Intervention
    status: str
    delivered_at_ms: int
    text_at_delivery: str

    def __post_init__(self) -> None:
        if self.status not in {"DELIVERED", "CANCELLED", "EXPIRED", "SUPPRESSED"}:
            raise ValueError(f"unknown delivery status: {self.status}")


@dataclass
class Outcome:
    """Whether the human continuation followed the prediction or the cue."""

    delivery: Delivery
    window_end_ms: int
    start_index: int = 0
    continuation: str = ""
    target_operation_seen: bool = False
    prediction_followed: bool = False
    carryover: str = ""
    carryover_seen: bool = False
    finalized: bool = False
