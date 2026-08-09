from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class TranscriptKind(StrEnum):
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


class BoundaryKind(StrEnum):
    SPEECH_STARTED = "SPEECH_STARTED"
    SPEECH_ENDED = "SPEECH_ENDED"
    TURN_CANDIDATE = "TURN_CANDIDATE"


@dataclass(frozen=True)
class TranscriptEvent:
    """Provider-neutral transcript update.

    Times are monotonic milliseconds relative to the current audio session.
    Confidence is optional because not every ASR provider exposes a calibrated
    score. `text` may be revised by later PARTIAL events until a FINAL event.
    """

    kind: TranscriptKind
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("invalid transcript time range")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class BoundaryEvent:
    """Turn-taking signal inferred from audio/VAD, not a cognitive-state claim."""

    kind: BoundaryKind
    at_ms: int
    silence_ms: int = 0

    def __post_init__(self) -> None:
        if self.at_ms < 0 or self.silence_ms < 0:
            raise ValueError("audio times must be non-negative")
        if self.kind is not BoundaryKind.TURN_CANDIDATE and self.silence_ms != 0:
            raise ValueError("silence_ms is only valid for TURN_CANDIDATE")


AudioEvent = TranscriptEvent | BoundaryEvent


class AudioEventSource(Protocol):
    """Minimal source boundary to be implemented by future ASR/VAD adapters."""

    def next_event(self) -> AudioEvent | None: ...


@dataclass
class TranscriptBuffer:
    """Accumulates final text and tracks only the latest unstable partial.

    ASR confidence is transport evidence, not a cognitive-state signal.  When
    multiple FINAL segments form one reasoning turn, the conservative turn
    confidence is the minimum known segment confidence.  Providers that do not
    expose confidence remain `None` rather than receiving an invented score.
    """

    _final_segments: list[str] = None  # type: ignore[assignment]
    _final_confidences: list[float] = None  # type: ignore[assignment]
    _partial: str = ""

    def __post_init__(self) -> None:
        if self._final_segments is None:
            self._final_segments = []
        if self._final_confidences is None:
            self._final_confidences = []

    def apply(self, event: TranscriptEvent) -> None:
        text = " ".join(event.text.strip().split())
        if event.kind is TranscriptKind.PARTIAL:
            self._partial = text
            return
        if text:
            self._final_segments.append(text)
            if event.confidence is not None:
                self._final_confidences.append(event.confidence)
        self._partial = ""

    def stable_text(self) -> str:
        return " ".join(self._final_segments)

    def stable_confidence(self) -> float | None:
        if not self._final_confidences:
            return None
        return min(self._final_confidences)

    def live_text(self) -> str:
        parts = [*self._final_segments]
        if self._partial:
            parts.append(self._partial)
        return " ".join(parts)

    def consume_stable_with_confidence(self) -> tuple[str, float | None]:
        text = self.stable_text()
        confidence = self.stable_confidence()
        self._final_segments.clear()
        self._final_confidences.clear()
        return text, confidence

    def consume_stable(self) -> str:
        text, _ = self.consume_stable_with_confidence()
        return text
