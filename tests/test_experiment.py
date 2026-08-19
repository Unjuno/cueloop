import json
from pathlib import Path
import tempfile
import unittest

from cueloop.core import ControllerConfig
from cueloop.experiment import EvaluationCase, ExpectedTurn, evaluate, load_dataset, run_experiment
from cueloop.core import Action


class ExperimentTests(unittest.TestCase):
    def test_harmful_interruption_is_weighted_more_than_a_miss(self) -> None:
        cases = [
            EvaluationCase("a", "eval", (ExpectedTurn("短い", Action.SILENCE),)),
        ]
        unsafe = evaluate(cases, ControllerConfig(0.55, 1, 16), name="unsafe")
        safe = evaluate(cases, ControllerConfig(), name="safe")
        self.assertEqual(unsafe.metrics.harmful_interruptions, 1)
        self.assertGreater(unsafe.metrics.weighted_loss, safe.metrics.weighted_loss)

    def test_dataset_requires_tune_and_eval_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps({"session_id": "x", "split": "tune", "turns": [{"text": "x", "expected_action": "SILENCE"}]})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both tune and eval"):
                load_dataset(path)

    def test_dataset_rejects_cross_split_text_leakage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rows = [json.loads(line) for line in (root / "experiments/policy-eval.jsonl").read_text().splitlines()]
        rows[-1]["turns"][0]["text"] = rows[0]["turns"][0]["text"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaked.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate turn text"):
                load_dataset(path)

    def test_experiment_writes_reproducible_policy_and_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            policy_path = Path(directory) / "policy.json"
            report = run_experiment(root / "experiments/policy-eval.jsonl", report_path, policy_path)
            self.assertTrue(report_path.exists())
            self.assertTrue(policy_path.exists())
            self.assertEqual(report["candidate_count"], 15)
            selected = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertIn("minimum_confidence", selected)
            self.assertLessEqual(
                report["tuned_eval"]["metrics"]["harmful_interruptions"],
                report["current_eval"]["metrics"]["harmful_interruptions"],
            )

    def test_experiment_rejects_output_path_collision(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "same.json"
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                run_experiment(root / "experiments/policy-eval.jsonl", output, output)


if __name__ == "__main__":
    unittest.main()
