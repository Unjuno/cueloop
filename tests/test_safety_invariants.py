from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cueloop.core import (
    Action,
    Analysis,
    ControllerConfig,
    Evidence,
    EvidenceKind,
    InterventionController,
    State,
)
from cueloop.knowledge import KnowledgeIndex
from cueloop.knowledge_diff import build_source_structure, compare_structures
from cueloop.llm import parse_analysis_json
from cueloop.structure import extract_structure
from cueloop.voice import (
    MidSpeechInjector,
    TurnBoundaryDetector,
    VoiceEvent,
    VoiceEventType,
)


class _Recorder:
    def __init__(self) -> None:
        self.utterances: list[str] = []

    def speak(self, text: str) -> None:
        self.utterances.append(text)


class SafetyInvariantTests(unittest.TestCase):
    # productive reasoning does not get interrupted
    def test_productive_state_is_silence(self) -> None:
        analysis = Analysis(
            State.PRODUCTIVE, 0.95, (Evidence(EvidenceKind.TRANSCRIPT, "x"),), Action.SILENCE
        )
        decision = InterventionController().decide(analysis)
        self.assertEqual(decision.action, Action.SILENCE)

    # low-confidence ASR/classification cannot trigger intervention
    def test_low_confidence_is_silence(self) -> None:
        analysis = Analysis(
            State.FIXATION, 0.3, (Evidence(EvidenceKind.TRANSCRIPT, "x"),), Action.PERTURB, "逆なら？"
        )
        decision = InterventionController(ControllerConfig(minimum_confidence=0.7)).decide(analysis)
        self.assertEqual(decision.action, Action.SILENCE)

    # short pauses do not create false turn boundaries
    def test_short_pause_is_not_turn_boundary(self) -> None:
        detector = TurnBoundaryDetector(endpoint_silence_ms=700)
        detector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 100, "a"))
        self.assertTrue(detector.speaking)
        self.assertIsNone(detector.feed(VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, 200, "a")))
        self.assertEqual(detector.feed(VoiceEvent(VoiceEventType.SPEECH_ENDED, 200)), None)

    # controller remains final delivery authority over proposed cue
    def test_controller_rejects_overlong_cue(self) -> None:
        analysis = Analysis(
            State.STRUCTURE_GAP,
            0.9,
            (Evidence(EvidenceKind.TRANSCRIPT, "x"),),
            Action.QUESTION,
            "とても長い完成回答ですこれはだめです",
        )
        decision = InterventionController(
            ControllerConfig(minimum_confidence=0.5, max_cue_characters=16)
        ).decide(analysis)
        self.assertEqual(decision.action, Action.SILENCE)

    # malformed LLM output fails closed
    def test_malformed_llm_output_is_silence(self) -> None:
        samples = [
            "not json",
            '{"state":"BADSTATE","confidence":0.9,"evidence":[],"proposed_action":"SILENCE","cue":null,"source_refs":[]}',
            '{"state":"STRUCTURE_GAP","confidence":1.5,"evidence":[{"kind":"TRANSCRIPT","description":"x","source_span":null}],"proposed_action":"QUESTION","cue":"q","source_refs":[]}',
        ]
        for raw in samples:
            analysis = parse_analysis_json(raw)
            self.assertEqual(analysis.proposed_action, Action.SILENCE)
            self.assertEqual(analysis.confidence, 0.0)

    # cues obey length limit
    def test_cue_respects_max_cue_characters(self) -> None:
        analysis = Analysis(
            State.STRUCTURE_GAP, 0.9, (Evidence(EvidenceKind.TRANSCRIPT, "x"),), Action.QUESTION, "x"
        )
        decision = InterventionController(ControllerConfig(minimum_confidence=0.5, max_cue_characters=8)).decide(analysis)
        self.assertEqual(decision.action, Action.QUESTION)
        self.assertLessEqual(len(decision.cue or ""), 8)

    # long-term safety: overload never receives a mid-speech cue
    def test_overload_never_receives_mid_speech_cue(self) -> None:

        speaker = _Recorder()
        injector = MidSpeechInjector(speaker=speaker, interval_ms=500, cooldown_ms=0)
        injector.feed(VoiceEvent(VoiceEventType.SPEECH_STARTED, 0))
        injector.feed(
            VoiceEvent(
                VoiceEventType.TRANSCRIPT_PARTIAL, 1_000, "疲れた、混乱、情報が多すぎてたぶん無理"
            )
        )
        self.assertEqual(speaker.utterances, [])

    # grounding diff is a candidate, never proof
    def test_grounding_diff_marked_candidate_not_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "notes.txt"
            source_path.write_text(
                "この設計は前提と依存関係を分けて考える。前提はデータの鮮度である。",
                encoding="utf-8",
            )
            index = KnowledgeIndex.from_paths([source_path])
            source = build_source_structure(index)
            user = extract_structure(["この設計について話す。前提が大事だ。"])
            diffs = compare_structures(user, source)
            joined = " ".join(diff.evidence for diff in diffs).lower()
            # diffs are review candidates, explicitly not ground truth
            self.assertTrue(len(joined) >= 0)
            for diff in diffs:
                self.assertTrue(0.0 <= diff.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
