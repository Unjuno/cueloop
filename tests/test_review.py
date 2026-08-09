import json
import unittest

from cueloop.review import EpistemicStatus, SessionReviewer, parse_review_json


VALID = {
    "claims": [
        {
            "text": "モデル規模だけでは学習効率を説明できない",
            "status": "INFERENCE",
            "premises": ["効率には複数の指標がある"],
            "assumptions": ["比較指標を分離すべきである"],
            "evidence": [],
            "counterexamples": [],
            "verification_needed": False,
        },
        {
            "text": "人間の認知アーキテクチャは継続学習に有利かもしれない",
            "status": "HYPOTHESIS",
            "premises": [],
            "assumptions": [],
            "evidence": [],
            "counterexamples": [],
            "verification_needed": True,
        },
    ],
    "unresolved_questions": ["効率を何で測るか"],
    "structural_gaps": ["比較指標が未定義"],
}


class ReviewContractTests(unittest.TestCase):
    def test_valid_review_parses(self) -> None:
        result = parse_review_json(json.dumps(VALID, ensure_ascii=False))
        self.assertIsNone(result.error)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].status, EpistemicStatus.INFERENCE)
        self.assertFalse(result.claims[0].verification_needed)
        self.assertTrue(result.claims[1].verification_needed)
        self.assertEqual(result.structural_gaps, ("比較指標が未定義",))

    def test_invalid_contract_fails_closed(self) -> None:
        result = parse_review_json('{"answer": "完成回答"}')
        self.assertIsNotNone(result.error)
        self.assertEqual(result.claims, ())

    def test_reviewer_does_not_call_model_for_empty_session(self) -> None:
        calls = []

        def complete(request):
            calls.append(request)
            return json.dumps(VALID, ensure_ascii=False)

        result = SessionReviewer(complete).review([])
        self.assertIsNotNone(result.error)
        self.assertEqual(calls, [])

    def test_reviewer_passes_numbered_turns(self) -> None:
        captured = []

        def complete(request):
            captured.append(request)
            return json.dumps(VALID, ensure_ascii=False)

        result = SessionReviewer(complete).review(["第一の主張", "第二の主張"])
        self.assertIsNone(result.error)
        self.assertEqual(len(captured), 1)
        self.assertIn("1. 第一の主張", captured[0].user)
        self.assertIn("2. 第二の主張", captured[0].user)
        self.assertIn("Do not rewrite", captured[0].system)


if __name__ == "__main__":
    unittest.main()
