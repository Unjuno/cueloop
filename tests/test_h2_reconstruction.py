import json
import tempfile
import unittest
from pathlib import Path

from scripts.h2_reconstruction import (
    RUBRIC,
    human_scores,
    load_records,
    pair_and_compare,
    proxy_scores,
    score_records,
)


def record(pair_key: str, condition: str, task: str, scores: dict) -> dict:
    return {
        "pair_key": pair_key,
        "condition": condition,
        "task": task,
        "text": "sample",
        "raters": [
            {"rater": "r1", "scores": scores},
            {"rater": "r2", "scores": {k: v + 1 for k, v in scores.items()}},
        ],
    }


class H2ReconstructionTests(unittest.TestCase):
    def test_human_scores_average_across_blind_raters(self) -> None:
        rec = record("p1", "CUE", "design", {k: 1 for k in RUBRIC})
        scores = human_scores(rec)
        self.assertEqual(scores["claims"], 1.5)

    def test_proxy_scores_are_bounded_and_labeled(self) -> None:
        scores = proxy_scores("結論は。前提は根拠。なぜなら。逆に。たぶん。")
        for key in RUBRIC:
            self.assertLessEqual(scores[key], 3.0)
            self.assertGreaterEqual(scores[key], 0.0)

    def test_paired_compare_positive_effect(self) -> None:
        records = []
        for i in range(6):
            records.append(record(f"p{i}", "CUE", "design", {k: 2 for k in RUBRIC}))
            records.append(record(f"p{i}", "SILENCE", "design", {k: 1 for k in RUBRIC}))
        scored = score_records(records, proxy=False)
        payload = pair_and_compare(scored, resamples=200, seed=0, confidence=0.95)
        self.assertEqual(payload["n_pairs"], 6)
        self.assertEqual(payload["scoring_source"], "blinded-raters")
        self.assertEqual(payload["metrics"]["total"]["point"], 7.0)
        self.assertTrue(payload["metrics"]["total"]["ci_excludes_zero"])

    def test_load_records_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.jsonl"
            path.write_text(
                json.dumps(record("p1", "CUE", "t", {k: 1 for k in RUBRIC})) + "\n"
            )
            loaded = load_records([path])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["pair_key"], "p1")


if __name__ == "__main__":
    unittest.main()
