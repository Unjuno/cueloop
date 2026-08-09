import unittest

from cueloop.audio import BoundaryEvent, BoundaryKind, TranscriptBuffer, TranscriptEvent, TranscriptKind
from cueloop.core import Action
from cueloop.offline import OfflineRunnerConfig, OfflineSessionRunner


class TranscriptConfidenceTests(unittest.TestCase):
    def test_turn_confidence_is_minimum_known_segment(self) -> None:
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptEvent(TranscriptKind.FINAL, "first", 0, 100, confidence=0.91))
        buffer.apply(TranscriptEvent(TranscriptKind.FINAL, "second", 120, 240, confidence=0.62))
        text, confidence = buffer.consume_stable_with_confidence()
        self.assertEqual(text, "first second")
        self.assertEqual(confidence, 0.62)

    def test_unknown_confidence_remains_unknown(self) -> None:
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptEvent(TranscriptKind.FINAL, "text", 0, 100, confidence=None))
        _, confidence = buffer.consume_stable_with_confidence()
        self.assertIsNone(confidence)


class ASRSafetyGateTests(unittest.TestCase):
    def test_low_confidence_transcript_cannot_trigger_intervention(self) -> None:
        runner = OfflineSessionRunner(config=OfflineRunnerConfig(minimum_asr_confidence=0.45))
        timeline = runner.run_events(
            [
                TranscriptEvent(
                    TranscriptKind.FINAL,
                    "えっと えっと えっと 性能が 性能が",
                    0,
                    1000,
                    confidence=0.20,
                ),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1400, silence_ms=400),
            ]
        )
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.SILENCE)
        self.assertEqual(timeline[0].decision.analysis.confidence, 0.0)
        self.assertIn("ASR confidence", timeline[0].decision.analysis.evidence[0])

    def test_high_confidence_transcript_reaches_analyzer(self) -> None:
        runner = OfflineSessionRunner(config=OfflineRunnerConfig(minimum_asr_confidence=0.45))
        timeline = runner.run_events(
            [
                TranscriptEvent(
                    TranscriptKind.FINAL,
                    "えっと えっと えっと 性能が 性能が",
                    0,
                    1000,
                    confidence=0.90,
                ),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1400, silence_ms=400),
            ]
        )
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.CUE)
        self.assertEqual(timeline[0].asr_confidence, 0.90)

    def test_unknown_confidence_is_not_invented_or_blocked(self) -> None:
        runner = OfflineSessionRunner(config=OfflineRunnerConfig(minimum_asr_confidence=0.45))
        timeline = runner.run_events(
            [
                TranscriptEvent(
                    TranscriptKind.FINAL,
                    "前提と依存関係を確認すると、この結論には条件があります。",
                    0,
                    1000,
                    confidence=None,
                ),
                BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=1400, silence_ms=400),
            ]
        )
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].decision.action, Action.SILENCE)
        self.assertIsNone(timeline[0].asr_confidence)


if __name__ == "__main__":
    unittest.main()
