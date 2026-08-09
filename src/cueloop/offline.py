from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .audio import AudioEvent, AudioEventSource, BoundaryEvent, BoundaryKind, TranscriptBuffer, TranscriptEvent
from .core import Analyzer, Decision, HeuristicAnalyzer, InterventionController, SessionContext


@dataclass(frozen=True)
class TimelineDecision:
    at_ms: int
    text: str
    decision: Decision

    def to_json(self) -> str:
        payload = {
            "at_ms": self.at_ms,
            "text": self.text,
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
    """Runs the production decision path against deterministic audio events."""

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        controller: InterventionController | None = None,
        context: SessionContext | None = None,
    ) -> None:
        self.analyzer = analyzer or HeuristicAnalyzer()
        self.controller = controller or InterventionController()
        self.context = context or SessionContext()
        self.buffer = TranscriptBuffer()

    def process(self, event: AudioEvent) -> TimelineDecision | None:
        if isinstance(event, TranscriptEvent):
            self.buffer.apply(event)
            return None

        if not isinstance(event, BoundaryEvent) or event.kind is not BoundaryKind.TURN_CANDIDATE:
            return None

        text = self.buffer.consume_stable().strip()
        if not text:
            return None

        analysis = self.analyzer.analyze(text, self.context)
        decision = self.controller.decide(analysis)
        self.context.add(text)
        return TimelineDecision(at_ms=event.at_ms, text=text, decision=decision)

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
