import unittest

from cueloop.sync.attention import AttentionTermExtractor, operation_for_term


class AttentionTests(unittest.TestCase):
    def test_operation_for_term_uses_vocabulary(self) -> None:
        self.assertEqual(operation_for_term("前提"), "dependency")
        self.assertEqual(operation_for_term("逆に"), "invert")
        # Terms absent from the vocabulary map to None (safe fallback).
        self.assertIsNone(operation_for_term("三段階"))

    def test_clean_accepts_bare_word_and_json(self) -> None:
        self.assertEqual(AttentionTermExtractor._clean("  前提\n"), "前提")
        self.assertEqual(AttentionTermExtractor._clean('{"term": "前提"}'), "前提")
        self.assertIsNone(AttentionTermExtractor._clean(""))
        self.assertIsNone(AttentionTermExtractor._clean("長" * 20))

    def test_in_context_requires_grounding(self) -> None:
        self.assertTrue(
            AttentionTermExtractor._in_context("前提", "この設計は前提を分ける", ())
        )
        self.assertFalse(
            AttentionTermExtractor._in_context("未知語", "この設計を分ける", ())
        )


if __name__ == "__main__":
    unittest.main()
