import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cueloop.sync import cli as sync_cli
from cueloop.sync.models import EventType, SyncEvent
from cueloop.sync.policy import (
    AsyncDecisionModel,
    CanonicalAssigner,
    RuleDecisionModel,
)
from cueloop.sync.predictors import NgramPredictor, ScriptedPredictor
from cueloop.sync.runtime import CallableSpeaker, CommandSpeaker, SyncRuntime


def event(event_type: str, at_ms: int, text: str = "") -> SyncEvent:
    return SyncEvent(EventType(event_type), at_ms, text)


def build_runtime(
    *, insertion_delay_ms: int = 0
) -> tuple[SyncRuntime, CallableSpeaker]:
    speaker = CallableSpeaker(lambda text: None)
    model = RuleDecisionModel(
        CanonicalAssigner(),
        insertion_delay_ms=insertion_delay_ms,
        context_fit=False,
    )
    runtime = SyncRuntime(
        predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.9)}),
        decision_model=model,
        speaker=speaker,
        cooldown_ms=2_500,
    )
    return runtime, speaker


class SlowSpeaker(CallableSpeaker):
    def speak(self, text: str) -> None:
        super().speak(text)
        time.sleep(0.05)


class RuntimeDeliveryTests(unittest.TestCase):
    def test_inserts_cue_at_clause_boundary(self) -> None:
        runtime, speaker = build_runtime()
        runtime.feed(event("speech_started", 0))
        produced = runtime.feed(
            event("transcript_partial", 800, "この設計は前提を分けて考える。")
        )
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].status, "DELIVERED")
        self.assertEqual(produced[0].intervention.cue, "根拠は？")
        self.assertEqual(produced[0].intervention.operation, "ground")
        self.assertEqual(speaker.spoken, ["根拠は？"])
        report = runtime.report()
        self.assertEqual(report["median_boundary_to_delivery_ms"], 0)

    def test_records_uptake_in_continuation(self) -> None:
        runtime, _ = build_runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(
            event(
                "transcript_partial",
                1_500,
                "この設計は前提を分けて考える。根拠は観測データだ。",
            )
        )
        runtime.feed(event("session_ended", 2_000))
        self.assertEqual(len(runtime.outcomes), 1)
        outcome = runtime.outcomes[0]
        self.assertTrue(outcome.finalized)
        self.assertTrue(outcome.target_operation_seen)
        self.assertIn("根拠", outcome.continuation)

    def test_ignores_transcript_during_tts_echo(self) -> None:
        speaker = SlowSpeaker(lambda text: None)
        runtime = SyncRuntime(
            predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.9)}),
            decision_model=RuleDecisionModel(CanonicalAssigner(), context_fit=False),
            speaker=speaker,
            cooldown_ms=2_500,
        )
        runtime.feed(event("speech_started", 0))
        text = "この設計は前提を分けて考える。"
        runtime.feed(event("transcript_partial", 800, text))
        self.assertEqual(runtime.synchronizer.text, text)
        runtime.feed(event("transcript_partial", 1_000, text + "根拠は？"))
        self.assertEqual(runtime.synchronizer.text, text)
        runtime.feed(event("transcript_partial", 1_200, text + "根拠は観測データだ。"))
        self.assertIn("根拠は", runtime.synchronizer.text)

    def test_cancels_when_speaker_diverges(self) -> None:
        runtime, speaker = build_runtime(insertion_delay_ms=500)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        self.assertEqual(speaker.spoken, [])
        produced = runtime.feed(
            event("transcript_partial", 1_000, "この設計は前提を分けて逆に考える")
        )
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].status, "CANCELLED")
        self.assertEqual(speaker.spoken, [])

    def test_replans_after_divergence(self) -> None:
        runtime, _ = build_runtime(insertion_delay_ms=500)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        produced = runtime.feed(
            event("transcript_partial", 2_000, "この設計は前提を分けて逆に考える")
        )
        self.assertEqual([delivery.status for delivery in produced], ["CANCELLED"])
        self.assertEqual(len(runtime.interventions), 2)

    def test_cancels_when_speech_ends_before_delivery(self) -> None:
        runtime, speaker = build_runtime(insertion_delay_ms=500)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(event("speech_ended", 1_000))
        self.assertEqual(len(runtime.deliveries), 1)
        self.assertEqual(runtime.deliveries[0].status, "CANCELLED")
        self.assertEqual(speaker.spoken, [])

    def test_paused_interventions_plan_nothing(self) -> None:
        runtime, _ = build_runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("interventions_paused", 100))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        self.assertEqual(runtime.interventions, [])

    def test_cooldown_blocks_second_plan(self) -> None:
        runtime, _ = build_runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(
            event(
                "transcript_partial",
                1_500,
                "この設計は前提を分けて考える。根拠は観測データだ。",
            )
        )
        self.assertEqual(len(runtime.interventions), 1)
        self.assertEqual(runtime.prediction_count, 1)

    def test_pause_boundary_triggers_prediction(self) -> None:
        runtime, _ = build_runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 400, "この設計は前提を分ける"))
        self.assertEqual(runtime.interventions, [])
        runtime.feed(event("tick", 900))
        self.assertEqual(runtime.prediction_count, 1)
        self.assertEqual(len(runtime.interventions), 1)

    def test_low_confidence_predictor_is_silent(self) -> None:
        runtime = SyncRuntime(
            predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.2)}),
            decision_model=RuleDecisionModel(CanonicalAssigner(), min_confidence=0.5),
        )
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        self.assertEqual(runtime.interventions, [])

    def test_ngram_runtime_reports(self) -> None:
        runtime = SyncRuntime(
            predictor=NgramPredictor.default(),
            decision_model=RuleDecisionModel(CanonicalAssigner(), min_confidence=0.0),
        )
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(event("session_ended", 2_000))
        report = runtime.report()
        self.assertEqual(report["utterances"], 1)
        self.assertIn("per_operation", report)
        self.assertIn("predictions", report)


class EarlyInsertionTests(unittest.TestCase):
    def _runtime(self, early_lead_ms: int) -> SyncRuntime:
        return SyncRuntime(
            predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.9)}),
            decision_model=RuleDecisionModel(CanonicalAssigner(), context_fit=False),
            speaker=CallableSpeaker(lambda text: None),
            pause_ms=350,
            early_lead_ms=early_lead_ms,
        )

    def test_plans_before_the_pause_threshold(self) -> None:
        runtime = self._runtime(early_lead_ms=100)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 400, "この設計は前提を分ける"))
        produced = runtime.feed(event("tick", 650))
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].status, "DELIVERED")
        self.assertEqual(produced[0].delivered_at_ms, 650)
        self.assertEqual(runtime.interventions[0].scheduled_at_ms, 650)

    def test_disabled_early_lead_waits_for_the_pause(self) -> None:
        runtime = self._runtime(early_lead_ms=0)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 400, "この設計は前提を分ける"))
        self.assertEqual(runtime.feed(event("tick", 650)), [])
        self.assertEqual(runtime.interventions, [])
        produced = runtime.feed(event("tick", 750))
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].delivered_at_ms, 750)


class CarryoverTests(unittest.TestCase):
    def _runtime(self) -> SyncRuntime:
        return SyncRuntime(
            predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.9)}),
            decision_model=RuleDecisionModel(CanonicalAssigner(), context_fit=False),
            speaker=CallableSpeaker(lambda text: None),
            cooldown_ms=2_500,
            outcome_window_ms=1_000,
        )

    def test_counts_cued_operation_after_the_outcome_window(self) -> None:
        runtime = self._runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(
            event(
                "transcript_partial",
                1_200,
                "この設計は前提を分けて考える。根拠は観測データだ。",
            )
        )
        runtime.feed(event("tick", 1_900))
        runtime.feed(
            event(
                "transcript_partial",
                2_000,
                "この設計は前提を分けて考える。根拠は観測データだ。根拠を再確認する。",
            )
        )
        runtime.feed(event("session_ended", 2_500))
        outcome = runtime.outcomes[0]
        self.assertTrue(outcome.carryover_seen)
        self.assertIn("根拠を再確認する。", outcome.carryover)
        report = runtime.report()
        self.assertEqual(report["carryover"], 1)
        self.assertEqual(report["novel_carryover"], 1)
        self.assertIn("ground", report["repertoire_growth"])
        self.assertEqual(
            report["outcomes"][0]["base_text"], "この設計は前提を分けて考える。"
        )
        self.assertTrue(report["outcomes"][0]["novel_operation"])

    def test_does_not_count_unrelated_text_after_the_window(self) -> None:
        runtime = self._runtime()
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 800, "この設計は前提を分けて考える。"))
        runtime.feed(
            event(
                "transcript_partial",
                1_200,
                "この設計は前提を分けて考える。根拠は観測データだ。",
            )
        )
        runtime.feed(event("tick", 1_900))
        runtime.feed(
            event(
                "transcript_partial",
                2_000,
                "この設計は前提を分けて考える。根拠は観測データだ。逆に考える。",
            )
        )
        runtime.feed(event("session_ended", 2_500))
        self.assertFalse(runtime.outcomes[0].carryover_seen)


class _SlowDecisionModel:
    label = "slow-decision"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.inner = RuleDecisionModel(CanonicalAssigner(), context_fit=False)

    def decide(self, state):
        time.sleep(self.delay)
        return self.inner.decide(state)


class AsyncDecisionRuntimeTests(unittest.TestCase):
    def _runtime(self, delay: float) -> SyncRuntime:
        return SyncRuntime(
            predictor=ScriptedPredictor({"この設計": ("だから結論は決まる。", 0.9)}),
            decision_model=AsyncDecisionModel(
                _SlowDecisionModel(delay),
                fallback=RuleDecisionModel(CanonicalAssigner(), context_fit=False),
            ),
            speaker=CallableSpeaker(lambda text: None),
            cooldown_ms=2_500,
        )

    def test_precomputed_decision_delivers_at_boundary(self) -> None:
        runtime = self._runtime(delay=0.02)
        runtime.feed(event("speech_started", 0))
        runtime.feed(event("transcript_partial", 400, "この設計は前提を"))
        time.sleep(0.05)
        produced = runtime.feed(
            event("transcript_partial", 800, "この設計は前提を分けて考える。")
        )
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].status, "DELIVERED")
        self.assertEqual(produced[0].delivered_at_ms, 800)
        self.assertEqual(runtime.decision_model.last_source, "precomputed")

    def test_fallback_keeps_delivery_on_time(self) -> None:
        runtime = self._runtime(delay=0.2)
        runtime.feed(event("speech_started", 0))
        produced = runtime.feed(
            event("transcript_partial", 800, "この設計は前提を分けて考える。")
        )
        self.assertEqual(len(produced), 1)
        self.assertEqual(produced[0].status, "DELIVERED")
        self.assertEqual(produced[0].delivered_at_ms, 800)
        self.assertEqual(runtime.decision_model.last_source, "fallback")


class CommandSpeakerTests(unittest.TestCase):
    def test_background_speaker_does_not_block(self) -> None:
        speaker = CommandSpeaker(
            [sys.executable, "-c", "import time; time.sleep(0.2)"],
            background=True,
        )
        try:
            started = time.perf_counter()
            speaker.speak("何に？")
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.15)
            self.assertGreaterEqual(speaker.last_duration_ms, 500)
        finally:
            speaker.close()

    def test_synchronous_speaker_reports_elapsed_time(self) -> None:
        speaker = CommandSpeaker(
            [sys.executable, "-c", "import time; time.sleep(0.05)"],
            background=False,
        )
        started = time.perf_counter()
        speaker.speak("何に？")
        self.assertGreaterEqual(time.perf_counter() - started, 0.05)
        self.assertGreaterEqual(speaker.last_duration_ms, 50)


class SyncCliTests(unittest.TestCase):
    def test_replay_writes_report_and_log(self) -> None:
        lines = [
            {"type": "speech_started", "at_ms": 0},
            {
                "type": "transcript_partial",
                "at_ms": 800,
                "text": "この設計は前提を分けて考える。",
            },
            {"type": "tick", "at_ms": 1200},
            {
                "type": "transcript_partial",
                "at_ms": 1600,
                "text": "この設計は前提を分けて考える。根拠は観測データだ。",
            },
            {"type": "speech_ended", "at_ms": 2000},
            {"type": "session_ended", "at_ms": 2100},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            events.write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
                + "\n",
                encoding="utf-8",
            )
            report_path = root / "report.json"
            log_path = root / "log.jsonl"
            status = sync_cli.main(
                [
                    "--events",
                    str(events),
                    "--predictor",
                    "ngram",
                    "--assignment",
                    "fixed",
                    "--cue",
                    "根拠は？",
                    "--min-confidence",
                    "0.0",
                    "--report",
                    str(report_path),
                    "--log",
                    str(log_path),
                ]
            )
            self.assertEqual(status, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("summary", report)
            self.assertIn("delivered", report)
            self.assertTrue(log_path.exists())

    def test_realtime_paces_file_replay(self) -> None:
        lines = [
            {"type": "speech_started", "at_ms": 0},
            {"type": "tick", "at_ms": 120},
            {"type": "session_ended", "at_ms": 120},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            events.write_text(
                "\n".join(json.dumps(line) for line in lines) + "\n",
                encoding="utf-8",
            )
            started = time.perf_counter()
            status = sync_cli.main(
                [
                    "--events",
                    str(events),
                    "--realtime",
                    "--report",
                    str(root / "report.json"),
                ]
            )
            self.assertEqual(status, 0)
            self.assertGreaterEqual(time.perf_counter() - started, 0.1)

    def test_invalid_event_line_exits_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"
            events.write_text(
                '{"type": "not-a-real-event", "at_ms": 1}\n', encoding="utf-8"
            )
            status = sync_cli.main(
                [
                    "--events",
                    str(events),
                    "--report",
                    str(Path(directory) / "report.json"),
                ]
            )
            self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()


class SilentConditionTests(unittest.TestCase):
    def test_silent_logs_decisions_without_delivering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "silent.jsonl"
            speaker = CallableSpeaker(lambda text: None)
            runtime = SyncRuntime(
                predictor=ScriptedPredictor(
                    {"この設計": ("だから結論は決まる。", 0.9)}
                ),
                decision_model=RuleDecisionModel(
                    CanonicalAssigner(), context_fit=False
                ),
                speaker=speaker,
                log_path=log_path,
                condition="SILENCE",
                assignment="canonical",
                seed=0,
                session_id="silent-session",
                pair_key="rule-seed0-fixture",
                silent=True,
            )
            runtime.feed(event("speech_started", 0))
            produced = runtime.feed(
                event("transcript_partial", 800, "この設計は前提を分けて考える。")
            )
            runtime.feed(event("session_ended", 900))
            self.assertEqual([item.status for item in produced], ["SILENT"])
            self.assertEqual(speaker.spoken, [])
            report = runtime.report()
            self.assertEqual(report["delivered"], 0)
            self.assertEqual(report["held"], 1)
            self.assertIs(report["silent"], True)
            self.assertEqual(report["assignment"], "canonical")
            self.assertEqual(report["seed"], 0)
            self.assertEqual(report["pair_key"], "rule-seed0-fixture")
            self.assertEqual(report["session_id"], "silent-session")
            self.assertEqual(len(report["outcomes"]), 1)
            self.assertEqual(report["median_boundary_to_delivery_ms"], 0)
            payloads = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            kinds = [payload["kind"] for payload in payloads]
            self.assertIn("intervention", kinds)
            self.assertIn("delivery", kinds)
            delivery = next(
                payload for payload in payloads if payload["kind"] == "delivery"
            )
            self.assertEqual(delivery["status"], "SILENT")
            self.assertEqual(delivery["condition"], "SILENCE")
