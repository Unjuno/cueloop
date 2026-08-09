from __future__ import annotations

from .audio import AudioEventSource, TranscriptEvent
from .offline import OfflineSessionRunner, TimelineDecision
from .output import InterventionSink


class LiveSessionRunner:
    """Consumes streaming audio events and emits interventions as decisions occur.

    If the configured analyzer exposes `prefetch(text, context)`, partial/final
    transcript updates are analyzed speculatively while the user is still
    speaking. The normal boundary-time `analyze` call then consumes only an
    exact-text prefetched result, preserving the same controller semantics.
    """

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

            if isinstance(event, TranscriptEvent):
                prefetch = getattr(self._session.analyzer, "prefetch", None)
                if callable(prefetch):
                    text = self._session.buffer.live_text()
                    if text.strip():
                        prefetch(text, self._session.context)

            if item is None:
                continue
            timeline.append(item)
            self._sink.emit(item.decision)
        return timeline
