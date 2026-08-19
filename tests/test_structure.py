import json
from pathlib import Path
import tempfile
import unittest

from cueloop.core import Action, Analysis, State
from cueloop.knowledge import GroundedRetrievalAnalyzer, KnowledgeIndex, ground_structure
from cueloop.structure import extract_structure, load_turns


class StructureTests(unittest.TestCase):
    def test_extracts_reviewable_categories(self) -> None:
        result = extract_structure(
            [
                "前提として利用者は自分で考える。したがって保持されると思う。根拠は実験データで確認する。",
                "しかし反例はある？ 2026年の統計は要確認。",
            ]
        )
        self.assertGreaterEqual(len(result.claims), 2)
        self.assertGreaterEqual(len(result.premises), 1)
        self.assertGreaterEqual(len(result.evidence), 1)
        self.assertGreaterEqual(len(result.counterexamples), 1)
        self.assertGreaterEqual(len(result.unresolved_questions), 1)
        self.assertGreaterEqual(len(result.verification_queue), 1)
        self.assertIn("INFERENCE", {item.epistemic_type for item in result.claims})

    def test_loads_voice_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.jsonl"
            path.write_text(
                json.dumps({"kind": "voice_turn", "turn": {"text": "検証する。"}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_turns(path), ["検証する。"])


class KnowledgeTests(unittest.TestCase):
    def test_grounding_keeps_source_and_marks_unsupported_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("音声ログは機密情報を含むため、保存期間を制限する。", encoding="utf-8")
            index = KnowledgeIndex.from_paths([source])
            structure = {
                "claims": [
                    {"id": "C1", "text": "音声ログの保存期間を制限する。", "turn_index": 1},
                    {"id": "C2", "text": "月はチーズでできている。", "turn_index": 2},
                ],
                "verification_queue": [],
            }
            result = ground_structure(structure, index)
            self.assertEqual(result["grounding"][0]["status"], "SUPPORTED")
            self.assertEqual(result["grounding"][1]["status"], "UNSUPPORTED")
            self.assertEqual(result["grounding"][0]["matches"][0]["source"], str(source))
            self.assertEqual(len(result["grounding"][0]["matches"][0]["source_sha256"]), 64)

    def test_retrieval_proposal_is_replaced_with_indexed_local_text(self) -> None:
        class ProposingAnalyzer:
            def analyze(self, text, context=None):
                return Analysis(
                    State.RETRIEVAL_FAILURE,
                    0.9,
                    ("用語を思い出せない",),
                    Action.RETRIEVE,
                    "モデルが作った未検証文",
                    ("fabricated",),
                )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("制御理論ではフィードバックを利用する。", encoding="utf-8")
            index = KnowledgeIndex.from_paths([source], chunk_characters=100)
            analysis = GroundedRetrievalAnalyzer(ProposingAnalyzer(), index).analyze("制御理論の用語")
            self.assertEqual(analysis.cue, "制御理論ではフィードバックを利用")
            self.assertEqual(analysis.source_refs, (index.chunks[0].id,))
            self.assertNotIn("fabricated", analysis.source_refs)


if __name__ == "__main__":
    unittest.main()
