from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .audio import AudioEvent, AudioEventSource, BoundaryEvent, BoundaryKind, TranscriptBuffer, TranscriptEvent
from .core import Analyzer, Decision, HeuristicAnalyzer, InterventionController, SessionContext, safe_analysis


@dataclass(frozen=True)
class OfflineRunnerConfig:
    """Runtime safety gates before semantic intervention analysis."""

    minimum_asr_confidence: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_asr_confidence <= 1.0:
            raise ValueError("minimum_asr_confidence must be in [0, 1]")


@dataclass(frozen=True)
class TimelineDecision:
    at_ms: int
    text: str
    decision: Decision
    asr_confidence: float | None = None

    def to_json(self) -> str:
        payload = {
            "at_ms": self.at_ms,
            "text": self.text,
            "asr_confidence": self.asr_confidence,
            "decision": {
                "action": self.decision.action.value,
                "cue": self.decision.cue,
                "reason": self.decision.reason,
                "analysis": {
                    **asdict(self.decision.analysis),
                    "state": self.decision.analysis.state.value,
                    "proposed_action": self.decision.analysis.proposed_action.value,
                },
            },
        }
        return json.dumps(payload, ensure_ascii=False)


class OfflineSessionRunner:
    """Runs the production decision path against deterministic audio events.

    Audio transport uncertainty is handled before semantic analysis. A low
    calibrated ASR confidence therefore cannot be mistaken for human hesitation
    or fixation and cannot cause an intervention. Event timestamps are also
    passed to the controller so recorded and live runs share the same delivery
    spacing rules.
    """

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        controller: InterventionController | None = None,
        context: SessionContext | None = None,
        config: OfflineRunnerConfig | None = None,
    ) -> None:
        self.analyzer = analyzer or HeuristicAnalyzer()
        self.controller = controller or InterventionController()
        self.context = context or SessionContext()
        self.config = config or OfflineRunnerConfig()
        self.buffer = TranscriptBuffer()

    def process(self, event: AudioEvent) -> TimelineDecision | None:
        if isinstance(event, TranscriptEvent):
            self.buffer.apply(event)
            return None

        if not isinstance(event, BoundaryEvent) or event.kind is not BoundaryKind.TURN_CANDIDATE:
            return None

        text, asr_confidence = self.buffer.consume_stable_with_confidence()
        text = text.strip()
        if not text:
            return None

        if asr_confidence is not None and asr_confidence < self.config.minimum_asr_confidence:
            analysis = safe_analysis(
                f"ASR confidence {asr_confidence:.2f} below delivery threshold "
                f"{self.config.minimum_asr_confidence:.2f}"
            )
        else:
            analysis = self.analyzer.analyze(text, self.context)

        decision = self.controller.decide(analysis, now_ms=event.at_ms)
        self.context.add(text)
        return TimelineDecision(
            at_ms=event.at_ms,
            text=text,
            decision=decision,
            asr_confidence=asr_confidence,
        )

    def run_source(self, source: AudioEventSource) -> list[TimelineDecision]:
        timeline: list[TimelineDecision] = []
        while True:
            event = source.next_event()
            if event is None:
                break
            result = self.process(event)
            if result is not None:
                timeline.append(result)
        return timeline

    def run_events(self, events: Iterable[AudioEvent]) -> list[TimelineDecision]:
        class _Source:
            def __init__(self, values: Iterable[AudioEvent]) -> None:
                self._iterator = iter(values)

            def next_event(self) -> AudioEvent | None:
                return next(self._iterator, None)

        return self.run_source(_Source(events))


def write_timeline(path: Path, timeline: Iterable[TimelineDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in timeline:
            handle.write(item.to_json() + "\n")
