import io
from pathlib import Path
import sys
import tempfile
import time
import unittest

from cueloop.core import Action, Analysis, ControllerConfig, HeuristicAnalyzer, InterventionController, State
from cueloop.knowledge import GroundedRetrievalAnalyzer, KnowledgeIndex
from cueloop.personalization import AdaptiveController, PersonalizationProfile
from cueloop.voice import (
    CommandSpeaker,
    MidSpeechInjector,
    ProfileSpeaker,
    TurnBoundaryDetector,
    VoiceEvent,
    VoiceEventType,
    VoiceProfile,
    VoiceSession,
    read_events,
    run_event_stream,
    run_live_event_stream,
    command_event_lines,
)
from cueloop.voice_cli import build_parser as build_voice_parser
from cueloop.voice_cli import ObservationOnlyController


ROTATION_FIRST = "何に依存？"


class RecordingSpeaker:
    def __init__(self) -> None:
        self.utterances: list[str] = []

    def speak(self, text: str) -> None:
        self.utterances.append(text)


class TurnBoundaryDetectorTests(unittest.TestCase):
    def test_waits_for_endpoint_silence(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=700)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 500, "考えているので"))
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 600)))
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.TICK, 1200)))
        turn = detector.feed(VoiceEvent(VoiceEventType.TICK, 1300))
        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertEqual(turn.text, "考えているので")

    def test_resume_cancels_old_endpoint(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=500)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "前半"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 400))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 500, "後半。"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 600))
        turn = detector.feed(VoiceEvent(VoiceEventType.TICK, 1100))
        assert turn is not None
        self.assertEqual(turn.text, "前半 後半。")

    def test_new_speech_after_long_gap_closes_previous_turn_without_tick(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=500)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "前の発話。"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200))
        previous = detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 2_000))
        assert previous is not None
        self.assertEqual(previous.text, "前の発話。")
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 2_100, "新しい発話。"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 2_200))
        current = detector.feed(VoiceEvent(VoiceEventType.TICK, 2_700))
        assert current is not None
        self.assertEqual(current.started_at_ms, 2_000)
        self.assertEqual(current.text, "新しい発話。")

    def test_empty_old_turn_is_discarded_before_new_speech(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=500, transcript_grace_ms=1_000)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 100))
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 2_000)))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 2_100, "新しい発話。"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 2_200))
        current = detector.feed(VoiceEvent(VoiceEventType.TICK, 2_700))
        assert current is not None
        self.assertEqual(current.started_at_ms, 2_000)

    def test_rejects_out_of_order_events(self) -> None:
        detector = TurnBoundaryDetector()
        detector.feed(VoiceEvent(VoiceEventType.TICK, 100))
        with self.assertRaises(ValueError):
            detector.feed(VoiceEvent(VoiceEventType.TICK, 99))

    def test_session_end_does_not_bypass_endpoint_silence(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=700)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "考え中"))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200))
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.SESSION_ENDED, 201)))

    def test_late_final_transcript_is_not_dropped_at_first_endpoint_tick(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=700, transcript_grace_ms=3_000)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200))
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.TICK, 900)))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 1_100, "遅れて確定した発話。"))
        turn = detector.feed(VoiceEvent(VoiceEventType.TICK, 1_101))
        assert turn is not None
        self.assertEqual(turn.text, "遅れて確定した発話。")


class VoiceSessionTests(unittest.TestCase):
    def test_never_speaks_before_turn_boundary(self) -> None:
        speaker = RecordingSpeaker()
        session = VoiceSession(detector=TurnBoundaryDetector(500), speaker=speaker)
        session.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        session.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "えっと、その、なんだろう、性能が良い"))
        session.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200))
        session.feed(VoiceEvent(VoiceEventType.TICK, 699))
        self.assertEqual(speaker.utterances, [])
        result = session.feed(VoiceEvent(VoiceEventType.TICK, 700))
        assert result is not None
        self.assertEqual(result.decision.action, Action.QUESTION)
        self.assertEqual(speaker.utterances, ["何に依存？"])
        self.assertGreater(result.state_expires_at_ms, result.state_observed_at_ms)

    def test_jsonl_stream_runs_end_to_end(self) -> None:
        stream = io.StringIO(
            '\n'.join(
                [
                    '{"type":"speech_started","at_ms":0}',
                    '{"type":"transcript_final","at_ms":100,"text":"前提と根拠を分けて考える。"}',
                    '{"type":"speech_ended","at_ms":200}',
                    '{"type":"tick","at_ms":900}',
                    '{"type":"session_ended","at_ms":1000}',
                ]
            )
        )
        results = run_event_stream(stream, VoiceSession())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].decision.action, Action.SILENCE)

    def test_expired_candidate_is_suppressed_and_logged(self) -> None:
        class SlowAnalyzer:
            def analyze(self, text, context=None):
                time.sleep(0.02)
                return Analysis(
                    State.STRUCTURE_GAP,
                    0.9,
                    ("observable gap",),
                    Action.QUESTION,
                    "何に依存？",
                )

        speaker = RecordingSpeaker()
        session = VoiceSession(
            analyzer=SlowAnalyzer(), speaker=speaker, max_decision_latency_ms=710
        )
        for event in (
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 0),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "性能が良い"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 200),
        ):
            self.assertIsNone(session.feed(event))
        result = session.feed(VoiceEvent(VoiceEventType.TICK, 900))
        assert result is not None
        self.assertEqual(result.decision.action, Action.SILENCE)
        self.assertEqual(result.delivery_status, "EXPIRED")
        self.assertEqual(speaker.utterances, [])

    def test_hung_analyzer_is_cut_off_by_delivery_deadline(self) -> None:
        class HungAnalyzer:
            def __init__(self):
                self.cancelled = False

            def analyze(self, text, context=None):
                time.sleep(1)
                return Analysis(State.STRUCTURE_GAP, 0.9, ("late",), Action.QUESTION, "何に依存？")

            def cancel(self):
                self.cancelled = True

        analyzer = HungAnalyzer()
        session = VoiceSession(analyzer=analyzer, max_decision_latency_ms=710)
        for event in (
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 0),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "性能が良い"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 200),
        ):
            session.feed(event)
        started = time.monotonic()
        result = session.feed(VoiceEvent(VoiceEventType.TICK, 900))
        assert result is not None
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(result.delivery_status, "EXPIRED")
        self.assertEqual(session.analyzer_failure_count, 1)
        self.assertTrue(analyzer.cancelled)

    def test_user_pause_control_suppresses_interventions_until_resumed(self) -> None:
        speaker = RecordingSpeaker()
        session = VoiceSession(speaker=speaker)
        session.feed(VoiceEvent(VoiceEventType.INTERVENTIONS_PAUSED, 0))
        for event in (
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 1),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "えっと、その、なんだろう、性能が良い"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 200),
        ):
            session.feed(event)
        result = session.feed(VoiceEvent(VoiceEventType.TICK, 900))
        assert result is not None
        self.assertEqual(result.decision.action, Action.SILENCE)
        self.assertEqual(result.delivery_status, "CANCELLED")
        self.assertEqual(speaker.utterances, [])

    def test_voice_time_cooldown_suppresses_close_interventions(self) -> None:
        session = VoiceSession(cooldown_ms=5_000)
        events = [
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 0),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "えっと、その、なんだろう、性能が良い"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 200),
            VoiceEvent(VoiceEventType.TICK, 900),
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 2_000),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 2_100, "えっと、その、なんだろう、性能が悪い"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 2_200),
            VoiceEvent(VoiceEventType.TICK, 2_900),
        ]
        results = [result for event in events if (result := session.feed(event)) is not None]
        self.assertEqual(results[0].decision.action, Action.QUESTION)
        self.assertEqual(results[1].decision.action, Action.SILENCE)
        self.assertIn("voice intervention cooldown", results[1].decision.reason)

    def test_adapter_warning_is_counted_without_creating_a_turn(self) -> None:
        session = VoiceSession()
        result = session.feed(
            VoiceEvent(VoiceEventType.ADAPTER_WARNING, 1, "recognizer rollover")
        )
        self.assertIsNone(result)
        self.assertEqual(session.adapter_warning_count, 1)

    def test_live_stream_cancels_candidate_when_user_resumes_during_analysis(self) -> None:
        class SlowAnalyzer:
            def analyze(self, text, context=None):
                time.sleep(0.03)
                return Analysis(
                    State.STRUCTURE_GAP,
                    0.9,
                    ("observable gap",),
                    Action.QUESTION,
                    "何に依存？",
                )

        def live_events():
            yield '{"type":"speech_started","at_ms":0}'
            yield '{"type":"transcript_final","at_ms":100,"text":"性能が良い"}'
            yield '{"type":"speech_ended","at_ms":200}'
            yield '{"type":"tick","at_ms":900}'
            time.sleep(0.005)
            yield '{"type":"speech_started","at_ms":910}'
            yield '{"type":"transcript_final","at_ms":1000,"text":"続ける。"}'
            yield '{"type":"speech_ended","at_ms":1100}'
            yield '{"type":"tick","at_ms":1800}'

        speaker = RecordingSpeaker()
        results = run_live_event_stream(
            live_events(),
            VoiceSession(analyzer=SlowAnalyzer(), speaker=speaker, max_decision_latency_ms=1_000),
        )
        self.assertEqual(results[0].decision.action, Action.SILENCE)
        self.assertEqual(results[0].delivery_status, "CANCELLED")
        self.assertIn("resumed speaking", results[0].decision.reason)
        self.assertEqual(speaker.utterances, ["何に依存？"])

    def test_cancelled_live_candidate_is_not_learned_as_delivered(self) -> None:
        class SlowAnalyzer:
            def analyze(self, text, context=None):
                time.sleep(0.02)
                return Analysis(State.STRUCTURE_GAP, 0.9, ("gap",), Action.QUESTION, "何に依存？")

        def live_events():
            yield '{"type":"speech_started","at_ms":0}'
            yield '{"type":"transcript_final","at_ms":100,"text":"性能が良い"}'
            yield '{"type":"speech_ended","at_ms":200}'
            yield '{"type":"tick","at_ms":900}'
            time.sleep(0.002)
            yield '{"type":"speech_started","at_ms":910}'

        profile = PersonalizationProfile()
        controller = AdaptiveController(
            InterventionController(ControllerConfig(cooldown_turns=2)), profile
        )
        results = run_live_event_stream(
            live_events(), VoiceSession(analyzer=SlowAnalyzer(), controller=controller)
        )
        self.assertEqual(results[0].delivery_status, "CANCELLED")
        stats = profile.for_intervention(Action.QUESTION, "何に依存？")
        self.assertEqual(stats.delivered, 0)

    def test_invalid_event_line_reports_line_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(read_events(["{\"type\":\"tick\",\"at_ms\":1}", "not-json"]))

    def test_event_command_reports_bounded_stderr_without_deadlock(self) -> None:
        script = "import sys; sys.stderr.write('x'*100000); print('{\"type\":\"tick\",\"at_ms\":1}'); raise SystemExit(7)"
        with self.assertRaisesRegex(RuntimeError, r"event command failed \(7\)"):
            list(command_event_lines([sys.executable, "-c", script]))

    def test_event_command_timeout_applies_while_output_is_streaming(self) -> None:
        script = "import time; print('{\"type\":\"tick\",\"at_ms\":1}', flush=True); time.sleep(1)"
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            list(command_event_lines([sys.executable, "-c", script], timeout_seconds=0.05))
        self.assertLess(time.monotonic() - started, 0.8)

    def test_command_speaker_can_be_cancelled(self) -> None:
        speaker = CommandSpeaker(
            [sys.executable, "-c", "import time; time.sleep(2)"], timeout_seconds=3
        )
        outcome: list[BaseException] = []

        def speak() -> None:
            try:
                speaker.speak("cue")
            except BaseException as exc:
                outcome.append(exc)

        worker = __import__("threading").Thread(target=speak)
        started = time.monotonic()
        worker.start()
        time.sleep(0.05)
        speaker.cancel()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 1)
        self.assertRegex(str(outcome[0]), "cancelled")

    def test_voice_cli_accepts_semantic_analyzer_and_trusted_sources(self) -> None:
        args = build_voice_parser().parse_args(
            [
                "--events", "events.jsonl", "--analyzer-command", "python", "adapter.py",
                "--sources", "notes.txt", "--tts-command", "say",
            ]
        )
        self.assertEqual(args.analyzer_command, ["python", "adapter.py"])
        self.assertEqual(args.sources, [Path("notes.txt")])

    def test_observation_only_controller_never_claims_delivery(self) -> None:
        profile = PersonalizationProfile()
        decision = ObservationOnlyController(profile).decide(
            Analysis(State.STRUCTURE_GAP, 0.9, ("gap",), Action.QUESTION, "何に依存？")
        )
        self.assertEqual(decision.action, Action.SILENCE)
        self.assertEqual(len(profile.state_observations), 1)


class MidSpeechInjectorTests(unittest.TestCase):
    def test_injects_cue_during_ongoing_speech_at_interval(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=1_000, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 300, "たぶん、この方法が良いと思う")
        )
        self.assertEqual(speaker.utterances, [])
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_200, "たぶん、この方法が良いと思う")
        )
        self.assertEqual(speaker.utterances, ["根拠は？"])
        self.assertEqual(injector.delivered_count, 1)

    def test_respects_interval_between_injections(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=1_000, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 500, "たぶん、この方法が良いと思う")
        )
        self.assertEqual(speaker.utterances, [])
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_500, "たぶん、この方法が良いと思う")
        )
        self.assertEqual(speaker.utterances, ["根拠は？"])
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 2_000, "まだ同じ話を続けている")
        )
        self.assertEqual(speaker.utterances, ["根拠は？"])
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 3_000, "うーん、逆に考えた方が良いかな")
        )
        # The user already inverted the view; the complement asks for its boundary.
        self.assertIn("範囲は？", speaker.utterances)

    def test_cue_is_context_aware_and_gap_driven(self) -> None:
        cases = [
            # Uncertainty tokens assert a claim without support: ask for grounding.
            ("たぶん、そうだと思う", "根拠は？"),
            # A completed operation is not echoed back: it is complemented instead.
            # After a conclusion, the missing relation is its support.
            ("つまり、結論として決まる", "根拠は？"),
            # After concretizing, the missing relation is support/grounds.
            ("具体的な例を挙げる", "根拠は？"),
            # After explaining a mechanism, ask for grounds of the claim.
            ("なぜなら、こうなるはずだ", "根拠は？"),
            # Naming a dependency is not itself performative closing; exploring
            # "依存関係を確認" still warrants a dependency cue.
            ("前提と依存関係を確認", "何に？"),
            # Naming unclear conditions still warrants a boundary cue.
            ("条件と限界が不明瞭", "範囲は？"),
            # After inverting, ask for the boundary of the flipped view.
            ("逆に考えれば答えが見える", "範囲は？"),
        ]
        for text, expected in cases:
            speaker = RecordingSpeaker()
            injector = MidSpeechInjector(speaker=speaker, interval_ms=500, cooldown_ms=0)
            injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
            injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, text))
            self.assertEqual(speaker.utterances, [expected], text)

    def test_stale_echo_is_not_repeated_when_operation_is_performed(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=500, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        # The user is already being concrete, so "具体は？" is a stale echo and
        # must not be spoken. The complement (ground) may still be delivered.
        injector.feed(
            VoiceEvent(
                VoiceEventType.TRANSCRIPT_PARTIAL,
                1_100,
                "具体的な例を挙げて説明する。",
            )
        )
        self.assertNotIn("具体は？", speaker.utterances)
        self.assertEqual(speaker.utterances, ["根拠は？"])

    def test_no_injection_when_silent_or_too_short(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=1_000)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "短い"))
        self.assertEqual(speaker.utterances, [])
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 1_500))
        injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 2_000, "沈黙後は話していないので注入しない"))
        self.assertEqual(speaker.utterances, [])

    def test_no_injection_when_no_bound_operation_matches(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=500, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "今日は天気が良いから外を歩く")
        )
        self.assertEqual(speaker.utterances, [])

    def test_limits_cue_length_to_max_cue_characters(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(
            speaker=speaker, interval_ms=500, cooldown_ms=0, max_cue_characters=4
        )
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "なぜなら、こうなるはずだ")
        )
        # "なぜなら" already performs the mechanism operation; the complement
        # asks for the support of the asserted claim, still within the limit.
        self.assertEqual(speaker.utterances, ["根拠は？"])

    def test_mid_speech_never_delivers_answer_pointer(self) -> None:
        """Mid-speech is a method-steering channel: an explicit knowledge
        question that would upgrade to a grounded RETRIEVE at the turn boundary
        must stay silent during ongoing speech, so the user's own reasoning is
        not derailed by a source fragment."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text(
                "二相コミットプロトコルは複数のノード間で原子性を維持する。\n",
                encoding="utf-8",
            )
            index = KnowledgeIndex.from_paths([source])
            grounded = GroundedRetrievalAnalyzer(HeuristicAnalyzer(), index)
            speaker = RecordingSpeaker()
            injector = MidSpeechInjector(
                speaker=speaker, analyzer=grounded, interval_ms=500, cooldown_ms=0
            )
            injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
            injector.feed(
                VoiceEvent(
                    VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "二相コミットって何？"
                )
            )
            self.assertEqual(speaker.utterances, [])
            self.assertEqual(injector.delivered_count, 0)
        # Method nudges still pass through the same channel.
        speaker = RecordingSpeaker()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "method.txt"
            source.write_text("方法の記述。", encoding="utf-8")
            index = KnowledgeIndex.from_paths([source])
            injector = MidSpeechInjector(
                speaker=speaker,
                analyzer=GroundedRetrievalAnalyzer(HeuristicAnalyzer(), index),
                interval_ms=500,
                cooldown_ms=0,
            )
            injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
            injector.feed(
                VoiceEvent(
                    VoiceEventType.TRANSCRIPT_PARTIAL,
                    1_100,
                    "たぶん、この方法が良いと思う",
                )
            )
            self.assertEqual(speaker.utterances, ["根拠は？"])

    def test_voice_session_wires_mid_speech_injector(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=1_000)
        session = VoiceSession(speaker=speaker, mid_speech=injector)
        session.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        session.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 300, "えっと、その、なんだろう、性能が良い")
        )
        session.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 1_200, "えっと、その、なんだろう、性能が良い")
        )
        session.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 1_400))
        session.feed(VoiceEvent(VoiceEventType.TICK, 2_100))
        self.assertIn("何に依存？", speaker.utterances)

    def test_mid_speech_suppressed_when_interventions_paused(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=1_000)
        session = VoiceSession(speaker=speaker, mid_speech=injector)
        session.feed(VoiceEvent(VoiceEventType.INTERVENTIONS_PAUSED, 0))
        session.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 1))
        session.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_300, "えっと、その、なんだろう、性能が良い")
        )
        session.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 1_500))
        self.assertEqual(speaker.utterances, [])

    def test_voice_cli_parses_mid_speech_flags(self) -> None:
        args = build_voice_parser().parse_args(
            [
                "--events", "events.jsonl",
                "--tts-command", "say",
                "--mid-speech", "--mid-speech-interval-ms", "3000",
            ]
        )
        self.assertTrue(args.mid_speech)
        self.assertEqual(args.mid_speech_interval_ms, 3_000)
        self.assertFalse(args.mid_speech_cooldown_ms == -1)

    def test_rotate_mode_injects_during_productive_speech(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(
            speaker=speaker, interval_ms=1_000, cooldown_ms=0, always_rotate=True
        )
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "前提と根拠を分けて考える。")
        )
        self.assertEqual(speaker.utterances, [ROTATION_FIRST])
        injector.feed(
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 2_200, "さらに条件を比較して検討する。")
        )
        self.assertEqual(len(speaker.utterances), 2)
        self.assertNotEqual(speaker.utterances[0], speaker.utterances[1])

    def test_rotate_mode_never_cues_overload(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(
            speaker=speaker, interval_ms=1_000, cooldown_ms=0, always_rotate=True
        )
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_100, "疲れた、混乱、情報が多すぎ"))
        self.assertEqual(speaker.utterances, [])

    def test_same_operation_rotates_cue_so_it_never_repeats_verbatim(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=500, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        at = 1_000
        for _ in range(4):
            injector.feed(
                VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, at, "それには必要が伴う")
            )
            at += 600
        self.assertGreaterEqual(len(speaker.utterances), 2)
        self.assertEqual(len(set(speaker.utterances)), len(speaker.utterances))

    def test_verbatim_repeat_is_suppressed_even_within_same_operation(self) -> None:
        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(
            speaker=speaker, interval_ms=500, cooldown_ms=0, max_cue_characters=3
        )
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 1_000, "それには必要が伴う"))
        self.assertEqual(speaker.utterances, ["何に？"])
        for at in (1_600, 2_200, 2_800):
            injector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, at, "それには必要が伴う"))
        self.assertEqual(speaker.utterances, ["何に？"])

    def test_voice_cli_appends_tts_voice_to_say(self) -> None:
        args = build_voice_parser().parse_args(
            ["--events", "events.jsonl", "--tts-command", "say", "--tts-voice", "Kyoko"]
        )
        self.assertEqual(args.tts_command, ["say"])
        self.assertEqual(args.tts_voice, "Kyoko")

    def test_voice_profile_to_command_appends_say_options(self) -> None:
        profile = VoiceProfile(
            name="question",
            command=("say",),
            voice="Kyoko",
            rate=180,
            volume=0.6,
            intent="question",
        )
        self.assertEqual(
            profile.to_command("何に？"),
            ["say", "-v", "Kyoko", "-r", "180", "--volume=0.60", "何に？"],
        )

    def test_voice_profile_leaves_custom_command_unchanged(self) -> None:
        custom = VoiceProfile(name="custom", command=("flite", "-t"), intent="question")
        self.assertEqual(custom.to_command("cf"), ["flite", "-t", "cf"])

    def test_profile_speaker_selects_profile_by_intent(self) -> None:
        neutral = VoiceProfile(name="neutral", command=("say",), voice="Hana")
        question = VoiceProfile(
            name="question",
            command=("say",),
            voice="Kyoko",
            rate=160,
            volume=0.5,
            intent="question",
        )
        speaker = ProfileSpeaker([neutral, question])
        self.assertEqual(
            speaker.to_command_for_intent("question", "何に？"),
            ["say", "-v", "Kyoko", "-r", "160", "--volume=0.50", "何に？"],
        )

    def test_profile_speaker_falls_back_to_neutral_for_unknown_intent(self) -> None:
        neutral = VoiceProfile(name="neutral", command=("say",), voice="Hana")
        speaker = ProfileSpeaker([neutral])
        self.assertEqual(speaker.to_command_for_intent("verify", "例は？"), ["say", "-v", "Hana", "例は？"])

    def test_profile_speaker_requires_neutral_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "neutral"):
            ProfileSpeaker([VoiceProfile(name="q", command=("say",), intent="question")])


if __name__ == "__main__":
    unittest.main()
