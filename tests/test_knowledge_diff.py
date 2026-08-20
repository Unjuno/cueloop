from __future__ import annotations

import unittest

from cueloop.core import HeuristicAnalyzer, SessionContext, State
from cueloop.knowledge import KnowledgeIndex
from cueloop.knowledge_diff import (
    StructuralDiffKind,
    StructuralDiffAnalyzer,
    build_source_structure,
    compare_structures,
)
from cueloop.structure import extract_structure, SessionStructure


def _user(*turns: str) -> SessionStructure:
    return extract_structure(list(turns))


class KnowledgeDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        source = ["この設計は前提と依存関係を分けて考える。前提はデータの鮮度である。APIはOpenAIを使用する。"]
        self.source_structure = extract_structure(source)

    def test_same_topic_related_user_emits_diffs(self) -> None:
        user = _user("この設計について話す。前提が大事だ。")
        diffs = compare_structures(user, self.source_structure)
        self.assertGreater(len(diffs), 0)
        self.assertTrue(all(diff.confidence <= 0.9 for diff in diffs))
        self.assertTrue(all(0.0 <= diff.confidence <= 1.0 for diff in diffs))

    def test_unrelated_user_emits_no_diffs(self) -> None:
        user = _user("今日は天気がいいので散歩した。")
        self.assertEqual(compare_structures(user, self.source_structure), ())

    def test_variable_diff_cue_is_generic_not_answer(self) -> None:
        user = _user("この設計について話す。前提が大事だ。")
        variables = [
            diff for diff in compare_structures(user, self.source_structure)
            if diff.kind is StructuralDiffKind.MISSING_VARIABLE
        ]
        for diff in variables:
            self.assertEqual(diff.cue, "変数は？")
            self.assertNotIn("api", diff.cue)

    def test_diffs_never_exceed_cue_length_limit(self) -> None:
        user = _user("この設計について話す。前提が大事だ。")
        diffs = compare_structures(user, self.source_structure, max_cue_characters=8)
        self.assertTrue(all(len(diff.cue) <= 8 for diff in diffs))

    def test_analyzer_proposes_question_on_nonproductive_related_turn(self) -> None:
        from cueloop.core import HeuristicAnalyzer

        analyzer = StructuralDiffAnalyzer(HeuristicAnalyzer(), self.source_structure)
        context = SessionContext()
        context.add("この設計について話す。")
        analysis = analyzer.analyze("えっと、えっと、この設計の前提と依存を分けるのが難しくて、その、うーん", context)
        self.assertNotEqual(analysis.state, State.PRODUCTIVE)
        self.assertGreaterEqual(analysis.confidence, 0.5)
        self.assertIsNotNone(analysis.cue)
        self.assertLessEqual(len(analysis.cue or ""), 8)

    def test_analyzer_keeps_productive_untouched(self) -> None:
        from cueloop.core import HeuristicAnalyzer

        analyzer = StructuralDiffAnalyzer(HeuristicAnalyzer(), self.source_structure)
        analysis = analyzer.analyze("この設計は前提と根拠を分けて整理し、依存関係を丁寧に扱う。結論は明確だ。反例も検討した。")
        self.assertEqual(analysis.state, State.PRODUCTIVE)

    def test_build_source_structure_reuses_session_shape(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "notes.txt"
            source_path.write_text(
                "この設計は前提と依存関係を分けて考える。前提はデータの鮮度である。\n\nAPIはOpenAIを使用する。",
                encoding="utf-8",
            )
            index = KnowledgeIndex.from_paths([source_path])
            built = build_source_structure(index)
            self.assertIsInstance(built, SessionStructure)
            self.assertEqual(len(built.claims), len(self.source_structure.claims))
            self.assertEqual(len(built.premises), len(self.source_structure.premises))

    def test_compare_against_actual_index_based_source(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "notes.txt"
            source_path.write_text(
                "この設計は前提と依存関係を分けて考える。"
                "前提はデータの鮮度である。APIはOpenAIの公開仕様に基づく。",
                encoding="utf-8",
            )
            index = KnowledgeIndex.from_paths([source_path])
            source = build_source_structure(index)
            user = _user("この設計について話す。前提が大事だ。")
            diffs = compare_structures(user, source)
            self.assertIsInstance(diffs, tuple)
            for diff in diffs:
                self.assertGreater(diff.confidence, 0.0)

    def test_user_inference_without_source_support_is_flagged(self) -> None:
        source = extract_structure(
            ["この設計はデータの鮮度を前提に、依存関係を整理する。APIはOpenAI。"]
        )
        user = _user(
            "この設計のAPIをまったく新しい実装に置き換えるべきだと考える。"
        )
        kinds = {diff.kind for diff in compare_structures(user, source)}
        self.assertIn(StructuralDiffKind.UNSUPPORTED_INFERENCE, kinds)

    def test_analayzer_cue_never_carries_source_answer(self) -> None:
        source = extract_structure(
            ["最適なAPIはOpenAIのGPTの公開仕様に基づく。前提は応答時間の許容値。"]
        )
        analyzer = StructuralDiffAnalyzer(
            HeuristicAnalyzer(), source, minimum_confidence=0.1
        )
        context = SessionContext()
        context.add("このAPI選定を考えるのが難しい。")
        analysis = analyzer.analyze("この設計のAPIを置き換えた方が良い、うーん", context)
        self.assertIsNotNone(analysis.cue)
        self.assertNotIn("GPT", analysis.cue or "")
        self.assertNotIn("公開仕様", analysis.cue or "")
        self.assertLessEqual(len(analysis.cue or ""), 16)

    def test_empty_source_structure_yields_no_diffs(self) -> None:
        empty = extract_structure([])
        user = _user("この設計について話す。")
        try:
            diffs = compare_structures(user, empty)
        except IndexError:
            self.fail("empty source must not raise")
        self.assertEqual(diffs, ())


if __name__ == "__main__":
    unittest.main()