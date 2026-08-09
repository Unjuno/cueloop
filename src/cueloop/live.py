from __future__ import annotations

from .audio import AudioEventSource
from .offline import OfflineSessionRunner, TimelineDecision
from .output import InterventionSink


class LiveSessionRunner:
    """Consumes streaming audio events and emits interventions as decisions occur."""

    def __init__(self, session: OfflineSessionRunner, sink: InterventionSink) -> None:
        self._session = session
        self._sink = sink

    def run(self, source: AudioEventSource) -> list[TimelineDecision]:
        timeline: list[TimelineDecision] = []
        while True:
            event = source.next_event()
            if event is None:
                break
            item = self._session.process(event)
            if item is None:
                continue
            timeline.append(item)
            self._sink.emit(item.decision)
        return timeline
