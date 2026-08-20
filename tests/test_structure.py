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


class KnowledgeRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text(
                "データベースの一貫性を保証するには分散トランザクションが必要である。\n"
                "特に、二相コミットプロトコルは複数のノード間で原子性を維持する。\n"
                "分散環境では、ネットワーク分断が障害モデルの中心になる。\n",
                encoding="utf-8",
            )
            # Copy into a stable path so chunk IDs are deterministic across the
            # index instances created by this test class.
            self._dir = tempfile.TemporaryDirectory()
            stable = Path(self._dir.name) / "source.txt"
            stable.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            self.index = KnowledgeIndex.from_paths([stable])

    def _gra(self) -> GroundedRetrievalAnalyzer:
        from cueloop.core import HeuristicAnalyzer

        return GroundedRetrievalAnalyzer(HeuristicAnalyzer(), self.index, max_characters=16)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_knowledge_seeking_question_surfaces_grounded_retrieve(self) -> None:
        result = self._gra().analyze("データベースの整合性ってどうやって保証するんだろう。")
        self.assertEqual(result.proposed_action, Action.RETRIEVE)
        self.assertIsNotNone(result.cue)
        self.assertTrue(result.source_refs)
        # The cue is a fragment of the source, not its resolution.
        self.assertNotIn("分散トランザクション", result.cue)
        self.assertLessEqual(len(result.cue or ""), 16)

    def test_knowledge_seeking_question_with_question_mark(self) -> None:
        result = self._gra().analyze("二相コミットって何？")
        self.assertEqual(result.proposed_action, Action.RETRIEVE)
        self.assertTrue(result.source_refs)
        self.assertLessEqual(len(result.cue or ""), 16)
        # The cue is a fragment taken from the matched source chunk.
        chunk = next(c for c in self.index.chunks if c.id == result.source_refs[0])
        self.assertTrue(result.cue in chunk.text, "cue should be sourced from the index")

    def test_non_question_productive_turn_stays_silent(self) -> None:
        result = self._gra().analyze("モノリスはデプロイが容易だ。")
        self.assertEqual(result.proposed_action, Action.SILENCE)
        self.assertIsNone(result.cue)

    def test_off_topic_question_with_no_source_match_stays_silent(self) -> None:
        result = self._gra().analyze("APIのレイテンシってどうする？")
        self.assertEqual(result.proposed_action, Action.SILENCE)
        self.assertIsNone(result.cue)

    def test_bare_question_surface_detects_terminal_ha(self) -> None:
        """Japanese bare "Xは？" is an explicit knowledge-seeking question and
        must upgrade to a grounded RETRIEVE when the source answers it."""
        result = self._gra().analyze("二相コミットの扱いは？")
        self.assertEqual(result.proposed_action, Action.RETRIEVE)
        self.assertTrue(result.source_refs)
        self.assertGreater(result.confidence, 0.65)

    def test_strategy_question_does_not_trigger_retrieval(self) -> None:
        """Method/strategy questions are reasoning gaps, not knowledge
        questions; they must stay under the heuristic layer's control."""
        result = self._gra().analyze("どうやって進めていけばいいのかわからない。")
        self.assertEqual(result.proposed_action, Action.SILENCE)
        self.assertIsNone(result.cue)

    def test_cue_points_at_matched_region_not_chunk_head(self) -> None:
        """The grounded cue must come from the part of the passage that
        actually matches the query, never from the chunk opening."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text(
                "まず前提として、全体は顧客要件を中心に考える。"
                "そして、ネットワーク分断が発生したときの可用性と一貫性の"
                "トレードオフが障害モデルの中心になる。"
                "APIレートリミットは1分あたり1000リクエストに設定する。\n",
                encoding="utf-8",
            )
            index = KnowledgeIndex.from_paths([path])
            from cueloop.core import HeuristicAnalyzer
            analyzer = GroundedRetrievalAnalyzer(HeuristicAnalyzer(), index, max_characters=16)
            result = analyzer.analyze("APIレートリミットは？")
            self.assertEqual(result.proposed_action, Action.RETRIEVE)
            self.assertIsNotNone(result.cue)
            self.assertIn("リミット", result.cue or "")
            self.assertNotIn("前提", result.cue or "")

    def test_synonym_weak_question_still_surfaces_above_gate(self) -> None:
        """A knowledge question whose lexical match to the source is weak
        (because the user used a synonym: '整合性' vs the source's
        '一貹性') must still upgrade to a grounded RETRIEVE, because the
        explicit question intent is strong, independent evidence."""
        result = self._gra().analyze("データベースの整合性ってどうやって保証するんだろう。")
        self.assertEqual(result.proposed_action, Action.RETRIEVE)
        self.assertTrue(result.source_refs)
        # Confidence must clear the default delivery gate (0.65) thanks to the
        # question-aware lift, even though the raw match score is only ~0.26.
        self.assertGreater(result.confidence, 0.65)
        self.assertTrue(result.cue in self.index.chunks[0].text)


if __name__ == "__main__":
    unittest.main()
