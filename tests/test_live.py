import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from cueloop.audio import BoundaryEvent, BoundaryKind, TranscriptEvent, TranscriptKind
from cueloop.core import Action
from cueloop.live import LiveSessionRunner
from cueloop.offline import OfflineSessionRunner
from cueloop.output import InterventionSink
from cueloop.stream_command import StreamingCommandEventSource


class RecordingSink(InterventionSink):
    def __init__(self) -> None:
        self.actions = []

    def emit(self, decision) -> None:
        self.actions.append((decision.action, decision.cue))


class LiveSessionRunnerTests(unittest.TestCase):
    def test_silence_is_not_emitted_as_audio_output(self) -> None:
        sink = RecordingSink()
        runner = LiveSessionRunner(OfflineSessionRunner(), sink)

        class Source:
            def __init__(self) -> None:
                self.events = iter([
                    TranscriptEvent(
                        TranscriptKind.FINAL,
                        "前提と依存関係を確認すると、この結論には条件があります。",
                        0,
                        1000,
                    ),
                    BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1400, silence_ms=400),
                ])

            def next_event(self):
                return next(self.events, None)

        timeline = runner.run(Source())
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.SILENCE)
        self.assertEqual(sink.actions, [(Action.SILENCE, None)])

    def test_streaming_command_drives_live_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "stream.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    events = [
                        {"type":"transcript","kind":"FINAL","text":"えっと えっと えっと 性能が","start_ms":0,"end_ms":900},
                        {"type":"boundary","kind":"TURN_CANDIDATE","at_ms":1400,"silence_ms":500},
                    ]
                    for event in events:
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    """
                ),
                encoding="utf-8",
            )
            sink = RecordingSink()
            runner = LiveSessionRunner(OfflineSessionRunner(), sink)
            with StreamingCommandEventSource((sys.executable, str(script))) as source:
                timeline = runner.run(source)

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.CUE)
        self.assertEqual(sink.actions[0][0], Action.CUE)


if __name__ == "__main__":
    unittest.main()
