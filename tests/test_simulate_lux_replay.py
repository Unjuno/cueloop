"""Lux/Reazon replay simulator: real-speech events through the runtime."""

import csv
import json
import unittest
from pathlib import Path

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


class DeriveWhisperCacheTests(unittest.TestCase):
    def test_relative_lux_corpus_selects_canonical_lux_cache(self) -> None:
        from scripts.simulate_lux_replay import LUX_METADATA, _derive_whisper_cache

        relative = Path("data/real-speech/lux/metadata.csv")
        expected = LUX_METADATA.with_name("lux-whisper-results.json")
        self.assertEqual(_derive_whisper_cache(relative, None), expected)

    def test_non_lux_corpus_gets_sibling_cache(self) -> None:
        from scripts.simulate_lux_replay import _derive_whisper_cache

        corpus = Path("data/real-speech/reazon/metadata-sample-200.csv")
        self.assertEqual(
            _derive_whisper_cache(corpus, None),
            Path("data/real-speech/reazon/metadata-sample-200-whisper-results.json"),
        )

    def test_explicit_whisper_cache_wins(self) -> None:
        from scripts.simulate_lux_replay import _derive_whisper_cache

        corpus = Path("data/real-speech/lux/metadata.csv")
        self.assertEqual(
            _derive_whisper_cache(corpus, Path("/tmp/custom-cache.json")),
            Path("/tmp/custom-cache.json"),
        )


class CacheSubsetTests(unittest.TestCase):
    def test_cache_with_extra_entries_returns_exactly_the_corpus(self) -> None:
        import tempfile

        from scripts import simulate_lux_replay as sim

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "000").mkdir()
            (tmp / "000/a.flac").write_bytes(b"x")
            (tmp / "000/b.flac").write_bytes(b"x")
            metadata = tmp / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["file_name", "id", "raw_path", "cleaned_path", "transcript"]
                )
                writer.writerow(
                    ["000/a.flac", "a", "000/a.flac", "000/a.flac", "テスト例文"]
                )
                writer.writerow(
                    ["000/b.flac", "b", "000/b.flac", "000/b.flac", "もう一つの例文"]
                )
            cache = tmp / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "000/a.flac": {
                            "segments": [
                                {
                                    "words": [
                                        {"word": "テスト", "start": 0.0, "end": 0.5}
                                    ]
                                }
                            ]
                        },
                        "000/b.flac": {
                            "segments": [
                                {
                                    "words": [
                                        {"word": "もう一つ", "start": 0.0, "end": 0.5}
                                    ]
                                }
                            ]
                        },
                        "000/c.flac": {
                            "segments": [
                                {"words": [{"word": "余分", "start": 0.0, "end": 0.5}]}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = sim.load_or_transcribe(
                metadata,
                cache,
                whisper_hf_repo="/tmp/whisper-base",
                limit=0,
            )
            self.assertEqual(sorted(result), ["000/a.flac", "000/b.flac"])


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
