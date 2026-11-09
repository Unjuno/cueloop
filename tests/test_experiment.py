from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExperimentToolTests(unittest.TestCase):
    def test_compare_policies_groups_and_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            rule = {
                "decision_model": "prediction-gap(assigner=randomized(0))",
                "predictor": "ngram(3)",
                "condition": "CUE",
                "delivered": 2,
                "repertoire_growth": ["ground"],
                "boundary_to_delivery_ms": [100, 120],
                "outcomes": [
                    {
                        "operation": "ground",
                        "target_operation_seen": True,
                        "carryover_seen": False,
                        "novel_operation": True,
                    },
                    {
                        "operation": "dependency",
                        "target_operation_seen": False,
                        "carryover_seen": True,
                        "novel_operation": False,
                    },
                ],
            }
            linear = {
                "decision_model": "linear-policy(linear-policy)",
                "predictor": "ngram(3)",
                "condition": "CUE",
                "delivered": 2,
                "repertoire_growth": ["ground", "invert"],
                "boundary_to_delivery_ms": [90],
                "outcomes": [
                    {
                        "operation": "ground",
                        "target_operation_seen": True,
                        "carryover_seen": True,
                        "novel_operation": True,
                    },
                    {
                        "operation": "ground",
                        "target_operation_seen": True,
                        "carryover_seen": True,
                        "novel_operation": True,
                    },
                ],
            }
            llm = {
                "decision_model": "local-llm(qwen2.5:1.5b-instruct)",
                "predictor": "ngram(3)",
                "condition": "CUE",
                "delivered": 1,
                "repertoire_growth": [],
                "boundary_to_delivery_ms": [600],
                "outcomes": [
                    {
                        "operation": "boundary",
                        "target_operation_seen": False,
                        "carryover_seen": False,
                        "novel_operation": False,
                    }
                ],
            }
            for name, payload in (("rule", rule), ("linear", linear), ("llm", llm)):
                (reports / f"{name}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            output = root / "comparison.json"
            markdown = root / "comparison.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/compare_policies.py",
                    str(reports),
                    "--json",
                    str(output),
                    "--markdown",
                    str(markdown),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            groups = payload["groups"]
            self.assertEqual(groups["rule|CUE"]["uptake_rate"], 0.5)
            self.assertEqual(groups["rule|CUE"]["carryover_rate"], 0.5)
            self.assertEqual(groups["rule|CUE"]["novel_uptake_rate"], 1.0)
            self.assertEqual(groups["linear|CUE"]["carryover_rate"], 1.0)
            self.assertEqual(groups["linear|CUE"]["repertoire_growth_mean"], 2.0)
            self.assertEqual(groups["llm|CUE"]["carryover_rate"], 0.0)
            table = markdown.read_text(encoding="utf-8")
            self.assertIn("| rule | CUE |", table)
            self.assertIn("| linear | CUE |", table)
            self.assertIn("| llm | CUE |", table)

    def test_dry_run_uses_random_assignment(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--mode",
                "replay",
                "--decision",
                "rule",
                "--dry-run",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest_text = completed.stdout.split("log=", 1)[0].strip()
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["assignment"], "random")
        self.assertIn("--assignment", manifest["command"])
        self.assertIn("random", manifest["command"])
        self.assertFalse(manifest["tts"])

    def test_train_policy_smoke_fixture(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "policy.json"
            report = Path(directory) / "train-report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_policy.py",
                    str(root / "experiments" / "policy-smoke.jsonl"),
                    "--min-samples",
                    "10",
                    "--epochs",
                    "200",
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            model = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(model["metadata"]["dataset_samples"], 24)
            self.assertEqual(model["metadata"]["evaluated_on"], "holdout")
            self.assertGreaterEqual(model["threshold"], 0.5)
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["evaluated_on"], "holdout")
            self.assertIsNotNone(summary["eval_metrics"])
            self.assertIn("roc_auc", summary["eval_metrics"])
            self.assertIn("train_metrics", model["metadata"])

            evaluated = Path(directory) / "eval.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_policy.py",
                    str(root / "experiments" / "policy-smoke.jsonl"),
                    "--model",
                    str(output),
                    "--report",
                    str(evaluated),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads(evaluated.read_text(encoding="utf-8"))["metrics"]
            self.assertEqual(metrics["samples"], 24)
            self.assertGreater(metrics["accuracy"], 0.5)
            self.assertGreater(metrics["roc_auc"], 0.5)

    def test_missing_policy_model_fails_fast(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--mode",
                "replay",
                "--decision",
                "linear",
                "--policy-model",
                "/nonexistent/policy.json",
                "--dry-run",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("missing linear policy model", completed.stderr)


if __name__ == "__main__":
    unittest.main()
