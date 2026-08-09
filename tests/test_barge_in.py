import sys
import tempfile
import time
import unittest
from pathlib import Path

from cueloop.audio import BoundaryEvent, BoundaryKind
from cueloop.core import Action, Analysis, Decision, State
from cueloop.live import LiveSessionRunner
from cueloop.offline import OfflineSessionRunner
from cueloop.output import AsyncInterventionSink, CommandInterventionSink, CommandSinkConfig


def cue_decision() -> Decision:
    analysis = Analysis(
        state=State.STUCK,
        confidence=0.9,
        evidence=("semantic loop detected",),
        proposed_action=Action.CUE,
        cue="反例は？",
    )
    return Decision(Action.CUE, "反例は？", "test cue", analysis)


class EmptySource:
    def __init__(self, events):
        self._events = iter(events)

    def next_event(self):
        return next(self._events, None)


class BargeInTests(unittest.TestCase):
    def test_command_sink_cancel_terminates_active_process_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "finished.txt"
            command = (
                sys.executable,
                "-c",
                (
                    "import sys,time,pathlib; sys.stdin.read(); "
                    "time.sleep(2); pathlib.Path(sys.argv[1]).write_text('done')"
                ),
                str(marker),
            )
            sink = AsyncInterventionSink(
                CommandInterventionSink(CommandSinkConfig(command, timeout_seconds=5))
            )
            sink.emit(cue_decision())
            time.sleep(0.1)
            started = time.perf_counter()
            sink.cancel()
            sink.close()
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertFalse(marker.exists())

    def test_live_runner_calls_cancel_on_speech_started(self) -> None:
        class RecordingSink:
            def __init__(self):
                self.cancelled = 0

            def emit(self, decision):
                pass

            def cancel(self):
                self.cancelled += 1

        sink = RecordingSink()
        runner = LiveSessionRunner(OfflineSessionRunner(), sink)
        runner.run(EmptySource([BoundaryEvent(BoundaryKind.SPEECH_STARTED, at_ms=100)]))
        self.assertEqual(sink.cancelled, 1)


if __name__ == "__main__":
    unittest.main()
