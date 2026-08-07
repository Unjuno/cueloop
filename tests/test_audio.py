import unittest

from cueloop.audio import (
    BoundaryEvent,
    BoundaryKind,
    TranscriptBuffer,
    TranscriptEvent,
    TranscriptKind,
)


class TranscriptEventTests(unittest.TestCase):
    def test_rejects_invalid_time_range(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptEvent(TranscriptKind.FINAL, "text", 100, 50)

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValueError):
            TranscriptEvent(TranscriptKind.FINAL, "text", 0, 10, 1.2)


class BoundaryEventTests(unittest.TestCase):
    def test_turn_candidate_can_carry_silence_duration(self) -> None:
        event = BoundaryEvent(BoundaryKind.TURN_CANDIDATE, at_ms=900, silence_ms=450)
        self.assertEqual(event.silence_ms, 450)

    def test_non_turn_boundary_cannot_carry_silence_duration(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryEvent(BoundaryKind.SPEECH_STARTED, at_ms=100, silence_ms=50)


class TranscriptBufferTests(unittest.TestCase):
    def test_partial_is_replaced_not_appended(self) -> None:
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptEvent(TranscriptKind.PARTIAL, "考えて", 0, 100))
        buffer.apply(TranscriptEvent(TranscriptKind.PARTIAL, "考えている", 0, 180))
        self.assertEqual(buffer.live_text(), "考えている")
        self.assertEqual(buffer.stable_text(), "")

    def test_final_commits_text_and_clears_partial(self) -> None:
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptEvent(TranscriptKind.PARTIAL, "途中", 0, 100))
        buffer.apply(TranscriptEvent(TranscriptKind.FINAL, "最終テキスト", 0, 200))
        self.assertEqual(buffer.live_text(), "最終テキスト")
        self.assertEqual(buffer.stable_text(), "最終テキスト")

    def test_consume_stable_clears_only_committed_segments(self) -> None:
        buffer = TranscriptBuffer()
        buffer.apply(TranscriptEvent(TranscriptKind.FINAL, "確定", 0, 100))
        buffer.apply(TranscriptEvent(TranscriptKind.PARTIAL, "次の途中", 110, 150))
        self.assertEqual(buffer.consume_stable(), "確定")
        self.assertEqual(buffer.live_text(), "次の途中")


if __name__ == "__main__":
    unittest.main()
