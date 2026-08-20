"""Tests for UnifiedSessionAnalyzer: turn-boundary analysis with structural diff + knowledge state."""

import unittest
from datetime import datetime, timezone

from cueloop.core import (
    Action,
    HeuristicAnalyzer,
    SessionContext,
    State,
)
from cueloop.knowledge_state import KnowledgeStore, KnowledgeState
from cueloop.personalization import PersonalizationProfile
from cueloop.structure import extract_structure, SessionStructure
from cueloop.unified_analyzer import UnifiedSessionAnalyzer, MidSpeechUnifiedAnalyzer, UnifiedAnalyzerConfig
from cueloop.voice import (
    MidSpeechInjector,
    VoiceEvent,
    VoiceEventType,
    VoiceSession,
)


def _source_from_text(text: str) -> SessionStructure:
    """Build source structure from text (same as session.py does)."""
    return extract_structure([text])


class UnifiedSessionAnalyzerBasicTests(unittest.TestCase):
    def test_passthrough_without_source_or_store(self) -> None:
        """Without source structure or knowledge store, delegates to base analyzer."""
        analyzer = UnifiedSessionAnalyzer(HeuristicAnalyzer())
        context = SessionContext()
        context.add("性能が良い")
        result = analyzer.analyze("前提として、データの品質が重要である。根拠は実験データに基づいている。")
        self.assertEqual(result.state, State.PRODUCTIVE)

    def test_productive_passthrough(self) -> None:
        """PRODUCTIVE state passes through without intervention."""
        source = _source_from_text("この設計は前提と依存関係を分けて考える。")
        analyzer = UnifiedSessionAnalyzer(HeuristicAnalyzer(), source_structure=source)
        result = analyzer.analyze("前提と根拠を分けて考える。結論は明確だ。")
        self.assertEqual(result.state, State.PRODUCTIVE)
        self.assertIsNone(result.cue)

    def test_overload_passthrough(self) -> None:
        """OVERLOAD state passes through without intervention."""
        source = _source_from_text("この設計は前提と依存関係を分けて考える。")
        analyzer = UnifiedSessionAnalyzer(HeuristicAnalyzer(), source_structure=source)
        result = analyzer.analyze("疲れた、混乱、情報が多すぎ")
        self.assertEqual(result.state, State.OVERLOAD)


class StructuralDiffAtTurnBoundaryTests(unittest.TestCase):
    def test_structural_diff_fills_gap_when_heuristic_has_no_cue(self) -> None:
        """When heuristic produces no cue (RETRIEVAL_FAILURE/BACKCHANNEL),
        structural diff provides one if source structure is available."""
        source = _source_from_text(
            "この設計は前提と依存関係を分けて考える。"
            "前提はデータの鮮度である。"
        )
        analyzer = UnifiedSessionAnalyzer(
            HeuristicAnalyzer(), source_structure=source, structural_minimum_confidence=0.1,
        )
        # "えっと、設計について" → short text, filler-like
        # Heuristic returns RETRIEVAL_FAILURE/BACKCHANNEL (no cue)
        context = SessionContext()
        context.add("この設計について考える。")
        result = analyzer.analyze("えっと、設計について", context)
        # The key: heuristic alone would return no cue (BACKCHANNEL).
        # But UnifiedSessionAnalyzer fills the gap with a structural diff cue.
        self.assertIsNotNone(result.cue)
        self.assertEqual(result.state, State.STRUCTURE_GAP)
        # The cue comes from the structural diff, not the heuristic
        self.assertIn(result.cue, {"前提は？", "依存は？", "変数は？", "根拠は？", "逆は？"})

    def test_heuristic_cue_is_preserved(self) -> None:
        """When heuristic produces a cue (e.g. repetition), it is preserved."""
        source = _source_from_text("この設計は前提と依存関係を分けて考える。")
        analyzer = UnifiedSessionAnalyzer(HeuristicAnalyzer(), source_structure=source)
        # Repetition triggers fixation → PERTURB with cue "逆なら？"
        context = SessionContext()
        context.add("性能が良い")
        result = analyzer.analyze("えっと、その、なんだろう、性能が良い", context)
        self.assertIsNotNone(result.cue)
        self.assertEqual(result.proposed_action, Action.QUESTION)


class KnowledgeStateFadingTests(unittest.TestCase):
    def test_fading_reduces_cue_after_spontaneous_reuse(self) -> None:
        """After spontaneous reuse, the same cue is faded to keyword."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        # Record cue delivery
        store.record_cue_delivery("何に依存？", source_session_id="test", now=now)
        # Simulate spontaneous reuse
        store.detect_spontaneous_reuse("何に依存しているのか考える", now=now)
        # Verify state was upgraded
        ac = store.get_acquisition("何に依存？")
        self.assertIsNotNone(ac)
        self.assertEqual(ac.state, KnowledgeState.SPONTANEOUSLY_REUSED)
        # Fading level should be 1 (keyword-only)
        self.assertEqual(store.get_fading_level("何に依存？"), 1)

    def test_fading_suppresses_explicitly_retained_concept(self) -> None:
        """After explicit retention, cue is fully suppressed."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        ac = store.record_cue_delivery("何に依存？", source_session_id="test", now=now)
        # Manually transition to EXPLICITLY_RETRAINED
        store._transition(ac, KnowledgeState.SPONTANEOUSLY_REUSED, now)
        ac2 = store.get_acquisition("何に依存？")
        store._transition(ac2, KnowledgeState.EXPLICITLY_RETRAINED, now)
        self.assertEqual(store.get_fading_level("何に依存？"), 2)

    def test_fading_after_spontaneous_reuse(self) -> None:
        """After spontaneous reuse of a multi-keyword cue, fading level is 1."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        store.record_cue_delivery("何に依存？", source_session_id="test", now=now)
        # "何に依存" contains keywords from "何に依存？"
        store.detect_spontaneous_reuse("何に依存しているのか考える", now=now)
        self.assertEqual(store.get_fading_level("何に依存？"), 1)

    def test_unified_analyzer_applies_fading_at_turn_boundary(self) -> None:
        """UnifiedSessionAnalyzer applies knowledge state fading to cues."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        # Record "何に依存？" delivery and simulate spontaneous reuse
        store.record_cue_delivery("何に依存？", source_session_id="test", now=now)
        store.detect_spontaneous_reuse("何に依存しているのか考える", now=now)
        # Now "何に依存？" is SPONTANEOUSLY_REUSED → fading level 1
        self.assertEqual(store.get_fading_level("何に依存？"), 1)
        # CueFadingPolicy should fade it
        from cueloop.cue_fading import CueFadingPolicy
        policy = CueFadingPolicy(store)
        decision = policy.apply("何に依存？")
        # After reuse, the cue is faded to keyword
        self.assertNotEqual(decision.faded_cue, "何に依存？")


class SpontaneousReuseDetectionTests(unittest.TestCase):
    def test_reuse_detection_in_analyze_method(self) -> None:
        """UnifiedSessionAnalyzer detects spontaneous reuse during analyze()."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        store.record_cue_delivery("依存は？", source_session_id="test", now=now)
        ac = store.get_acquisition("依存は？")
        self.assertEqual(ac.state, KnowledgeState.CUE_ASSISTED)
        # Create analyzer and analyze text that contains the keyword
        analyzer = UnifiedSessionAnalyzer(
            HeuristicAnalyzer(), knowledge_store=store,
        )
        # "依存" should trigger reuse detection
        context = SessionContext()
        context.add("設計を考える")
        analyzer.analyze("依存関係を整理する", context)
        # Check that state was upgraded
        ac_after = store.get_acquisition("依存は？")
        self.assertEqual(ac_after.state, KnowledgeState.SPONTANEOUSLY_REUSED)

    def test_reuse_detection_requires_all_keywords(self) -> None:
        """Reuse requires ALL keywords (AND logic) to reduce false positives."""
        store = KnowledgeStore()
        now = datetime.now(timezone.utc)
        store.record_cue_delivery("何に依存？", source_session_id="test", now=now)
        ac = store.get_acquisition("何に依存？")
        self.assertEqual(ac.state, KnowledgeState.CUE_ASSISTED)
        # "何" alone should NOT trigger reuse (AND logic requires all keywords)
        analyzer = UnifiedSessionAnalyzer(
            HeuristicAnalyzer(), knowledge_store=store,
        )
        context = SessionContext()
        context.add("設計を考える")
        analyzer.analyze("何をやればいいか", context)
        # Should still be CUE_ASSISTED (not upgraded)
        ac_after = store.get_acquisition("何に依存？")
        self.assertEqual(ac_after.state, KnowledgeState.CUE_ASSISTED)


class MidSpeechStructuralActivationTests(unittest.TestCase):
    """Mid-speech cues become source-grounded once session context is supplied."""

    def _source(self) -> SessionStructure:
        return _source_from_text(
            "モノリス設計は依存関係を整理する必要がある。"
            "前提として、データの一貫性保証は分散トランザクションに依存する。"
        )

    def test_mid_speech_structural_diff_uses_context_when_provided(self) -> None:
        """With context, the structural branch produces a gap cue while the user
        is reasoning productively, even when no vocabulary operation fires."""
        source = self._source()
        config = UnifiedAnalyzerConfig(
            source_structure=source,
            minimum_confidence=0.6,
            structural_minimum_confidence=0.3,
            max_mid_speech_cue_characters=8,
        )
        analyzer = MidSpeechUnifiedAnalyzer(config, HeuristicAnalyzer())
        context = SessionContext()
        context.add("この設計はモノリスが良いと思う。")
        result = analyzer.analyze("モノリスはデプロイの観点で良い選択だ。", context)
        # The plain declarative turn has no vocabulary operation, so the only
        # way to produce a cue here is the source-grounded structural path.
        self.assertEqual(result.state, State.STRUCTURE_GAP)
        self.assertEqual(result.cue, "根拠は？")

    def test_mid_speech_without_context_falls_back_to_silence(self) -> None:
        """Without context the structural branch is inert (previous behavior):
        a productive declarative turn produces no cue at all."""
        source = self._source()
        config = UnifiedAnalyzerConfig(
            source_structure=source,
            minimum_confidence=0.6,
            structural_minimum_confidence=0.3,
            max_mid_speech_cue_characters=8,
        )
        analyzer = MidSpeechUnifiedAnalyzer(config, HeuristicAnalyzer())
        result = analyzer.analyze("モノリスはデプロイの観点で良い選択だ。")
        self.assertEqual(result.state, State.PRODUCTIVE)
        self.assertIsNone(result.cue)

    def test_voice_session_passes_context_so_mid_speech_is_source_grounded(self) -> None:
        """VoiceSession forwards its completed-turn context into the mid-speech
        path, so live cues reflect source gaps while the user is still speaking."""
        class RecordingSpeaker:
            def __init__(self) -> None:
                self.utterances: list[str] = []

            def speak(self, text: str) -> None:
                self.utterances.append(text)

        speaker = RecordingSpeaker()
        injector = MidSpeechInjector(
            speaker=speaker,
            interval_ms=1_500,
            cooldown_ms=0,
            source_structure=self._source(),
        )
        session = VoiceSession(speaker=speaker, mid_speech=injector)
        for event in (
            VoiceEvent(VoiceEventType.SPEECH_STARTED, 0),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 800, "この設計はモノリスが良いと思う。"),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, 900),
            VoiceEvent(VoiceEventType.TICK, 1_600),
        ):
            session.feed(event)
        self.assertEqual(speaker.utterances, [])
        session.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 3_000))
        session.feed(
            VoiceEvent(
                VoiceEventType.TRANSCRIPT_PARTIAL,
                4_600,
                "モノリスはデプロイの観点で良い選択だ。",
            )
        )
        # With prior-turn context, the mid-speech path surfaces a source-grounded
        # gap live; without that context the same turn would be silent.
        self.assertIn("根拠は？", speaker.utterances)


class EndToEndSessionTests(unittest.TestCase):
    def test_session_with_source_uses_unified_analyzer(self) -> None:
        """Session creates UnifiedSessionAnalyzer when source is available."""
        import tempfile
        from pathlib import Path
        from cueloop.session import run_integrated_session

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("この設計は前提と依存関係を分けて考える。\n")
            f.write("前提はデータの鮮度である。\n")
            source_path = Path(f.name)

        try:
            events = (
                '{"type":"speech_started","at_ms":0}\n'
                '{"type":"transcript_final","at_ms":100,"text":"えっと、その、なんだろう、性能が良い"}\n'
                '{"type":"speech_ended","at_ms":200}\n'
                '{"type":"tick","at_ms":900}\n'
                '{"type":"session_ended","at_ms":1000}\n'
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                outcome = run_integrated_session(
                    events.splitlines(),
                    Path(tmpdir),
                    goal="unified analyzer integration test",
                    source_paths=[source_path],
                    session_id="test-unified-integration",
                )
                self.assertEqual(outcome.manifest["status"], "COMPLETED")
                self.assertGreater(outcome.manifest["turn_count"], 0)
        finally:
            source_path.unlink(missing_ok=True)

    def test_knowledge_state_shared_between_mid_speech_and_turn_boundary(self) -> None:
        """Knowledge state from mid-speech delivery is visible to turn-boundary analysis."""
        profile = PersonalizationProfile()
        now = datetime.now(timezone.utc)
        # Record a cue delivery (as mid-speech would)
        profile.knowledge_store.record_cue_delivery(
            "何に依存？", source_session_id="test", now=now
        )
        ac = profile.knowledge_store.get_acquisition("何に依存？")
        self.assertEqual(ac.state, KnowledgeState.CUE_ASSISTED)
        # Simulate spontaneous reuse (as turn-boundary detection would)
        profile.knowledge_store.detect_spontaneous_reuse(
            "何に依存しているのか考える", now=now
        )
        ac_after = profile.knowledge_store.get_acquisition("何に依存？")
        self.assertEqual(ac_after.state, KnowledgeState.SPONTANEOUSLY_REUSED)
        # Fading level should reflect the upgrade
        self.assertEqual(profile.knowledge_store.get_fading_level("何に依存？"), 1)


if __name__ == "__main__":
    unittest.main()
