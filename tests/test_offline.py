import tempfile
import unittest
from pathlib import Path

from cueloop.audio import BoundaryEvent, BoundaryKind, TranscriptEvent, TranscriptKind
from cueloop.core import Action
from cueloop.offline import OfflineSessionRunner, write_timeline
from cueloop.replay import JSONLReplaySource, ReplayFormatError


class ReplaySourceTests(unittest.TestCase):
    def test_parses_monotonic_event_stream(self) -> None:
        source = JSONLReplaySource.from_lines(
            [
                '{"type":"transcript","kind":"FINAL","text":"結論はこうです。","start_ms":0,"end_ms":800}',
                '{"type":"boundary","kind":"TURN_CANDIDATE","at_ms":1200,"silence_ms":400}',
            ]
        )
        self.assertIsInstance(source.next_event(), TranscriptEvent)
        self.assertIsInstance(source.next_event(), BoundaryEvent)
        self.assertIsNone(source.next_event())

    def test_rejects_time_reversal(self) -> None:
        with self.assertRaises(ReplayFormatError):
            JSONLReplaySource.from_lines(
                [
                    '{"type":"boundary","kind":"TURN_CANDIDATE","at_ms":1000,"silence_ms":400}',
                    '{"type":"transcript","kind":"FINAL","text":"backwards","start_ms":0,"end_ms":500}',
                ]
            )


class OfflineSessionRunnerTests(unittest.TestCase):
    def test_turn_candidate_drives_same_controller_path(self) -> None:
        runner = OfflineSessionRunner()
        timeline = runner.run_events(
            [
                TranscriptEvent(
                    TranscriptKind.FINAL,
                    "前提と依存関係を確認すると、この結論には条件があります。",
                    0,
                    1000,
                ),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1400, silence_ms=400),
            ]
        )
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.SILENCE)

    def test_multiple_final_segments_form_one_turn(self) -> None:
        runner = OfflineSessionRunner()
        timeline = runner.run_events(
            [
                TranscriptEvent(TranscriptKind.FINAL, "えっと", 0, 300),
                TranscriptEvent(TranscriptKind.FINAL, "えっと", 320, 600),
                TranscriptEvent(TranscriptKind.FINAL, "えっと性能が", 620, 1000),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1500, silence_ms=500),
            ]
        )
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.analysis.state.value, "STUCK")
        self.assertEqual(timeline[0].decision.action, Action.CUE)

    def test_partial_without_final_does_not_trigger_analysis(self) -> None:
        runner = OfflineSessionRunner()
        timeline = runner.run_events(
            [
                TranscriptEvent(TranscriptKind.PARTIAL, "まだ途中", 0, 500),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=900, silence_ms=400),
            ]
        )
        self.assertEqual(timeline, [])

    def test_timeline_is_jsonl_serializable(self) -> None:
        runner = OfflineSessionRunner()
        timeline = runner.run_events(
            [
                TranscriptEvent(TranscriptKind.FINAL, "これは短いので", 0, 700),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1100, silence_ms=400),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.jsonl"
            write_timeline(path, timeline)
            content = path.read_text(encoding="utf-8")
        self.assertIn('"at_ms": 1100', content)
        self.assertIn('"action": "CONTINUE"', content)


if __name__ == "__main__":
    unittest.main()
