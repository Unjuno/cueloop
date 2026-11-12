"""Lux replay simulator: real-speech events through the runtime."""

import unittest

from cueloop.sync.models import EventType
from scripts.simulate_lux_replay import events_for, run_session


class EventsForTests(unittest.TestCase):
    def test_events_for_emits_progressive_transcript_partials(self) -> None:
        segments = [
            {
                "words": [
                    {"word": "女", "start": 0.1, "end": 0.3},
                    {"word": "の", "start": 0.3, "end": 0.5},
                    {"word": "子", "start": 0.5, "end": 0.7},
                ]
            }
        ]
        events = events_for("dummy.wav", segments)
        kinds = [event.type for event in events]
        self.assertEqual(kinds[0], EventType.SPEECH_STARTED)
        self.assertEqual(kinds[1], EventType.TRANSCRIPT_PARTIAL)
        self.assertEqual(events[1].text, "女")
        self.assertEqual(kinds[-3], EventType.TRANSCRIPT_FINAL)
        self.assertEqual(kinds[-2], EventType.SPEECH_ENDED)
        self.assertEqual(kinds[-1], EventType.SESSION_ENDED)

    def test_events_for_returns_empty_for_no_words(self) -> None:
        self.assertEqual(events_for("dummy.wav", []), [])
        self.assertEqual(events_for("dummy.wav", [{"words": []}]), [])


class RunSessionTests(unittest.TestCase):
    def _segments(self, text: str):
        # Break the text into per-word chunks by splitting on common kanji word boundaries.
        # For tests only, just split by the first N characters.
        midpoint = max(1, len(text) // 2)
        return [
            {
                "words": [
                    {"word": text[:midpoint], "start": 0.0, "end": 1.0},
                    {"word": text[midpoint:], "start": 1.0, "end": 2.0},
                ]
            }
        ]

    def test_run_session_emits_analyzer_ready_report(self) -> None:
        cue = run_session(
            file_name="test.wav",
            segments=self._segments("この設計は前提と依存関係を分けて考える。"),
            condition="CUE",
            pause_ms=350,
            cooldown_ms=250,
            min_text_characters=4,
        )
        silence = run_session(
            file_name="test.wav",
            segments=self._segments("この設計は前提と依存関係を分けて考える。"),
            condition="SILENCE",
            pause_ms=350,
            cooldown_ms=250,
            min_text_characters=4,
        )
        for key in (
            "predictor",
            "decision_model",
            "condition",
            "session_id",
            "pair_key",
            "silent",
            "utterances",
            "predictions",
            "interventions",
            "delivered",
            "outcomes",
        ):
            self.assertIn(key, cue)
        self.assertEqual(cue["condition"], "CUE")
        self.assertEqual(silence["condition"], "SILENCE")
        self.assertTrue(silence["silent"])


if __name__ == "__main__":
    unittest.main()
