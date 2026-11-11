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


def _session_report(
    condition: str,
    pair_key: str,
    session_id: str,
    *,
    target: bool,
) -> dict[str, object]:
    return {
        "decision_model": "prediction-gap(assigner=canonical)",
        "predictor": "ngram(3)",
        "condition": condition,
        "assignment": "random-operation",
        "seed": 0,
        "pair_key": pair_key,
        "session_id": session_id,
        "delivered": 1 if condition == "CUE" else 0,
        "boundary_to_delivery_ms": [100],
        "repertoire_growth": ["ground"] if target else [],
        "outcomes": [
            {
                "operation": "ground",
                "requested_operation": "ground",
                "novel_operation": True,
                "target_operation_seen": target,
                "carryover_seen": target,
            }
        ],
    }


class ExperimentAnalysisTests(unittest.TestCase):
    def _run(self, reports: Path, output: Path, markdown: Path) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/analyze_experiment.py",
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

    def _write_pairs(self, reports: Path, count: int) -> None:
        for index in range(count):
            for condition, target in (("CUE", True), ("SILENCE", False)):
                payload = _session_report(
                    condition,
                    f"rule-seed{index}-fixture",
                    f"{condition}-{index}",
                    target=target,
                )
                (reports / f"{condition}-{index}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

    def test_analysis_is_deterministic_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            self._write_pairs(reports, 3)
            self._run(reports, root / "first.json", root / "first.md")
            self._run(reports, root / "second.json", root / "second.md")
            self.assertEqual(
                (root / "first.json").read_text(encoding="utf-8"),
                (root / "second.json").read_text(encoding="utf-8"),
            )
            payload = json.loads((root / "first.json").read_text(encoding="utf-8"))
            comparison = payload["comparisons"]["rule"]
            self.assertEqual(comparison["status"], "ok")
            self.assertEqual(comparison["n_pairs"], 3)
            uptake = comparison["metrics"]["uptake_rate"]
            self.assertEqual(uptake["point"], 1.0)
            self.assertEqual(uptake["n_pairs"], 3)
            self.assertIsInstance(uptake["ci_low"], float)
            self.assertIsInstance(uptake["ci_high"], float)
            self.assertEqual(payload["seed"], 0)
            self.assertEqual(payload["resamples"], 2000)
            table = (root / "first.md").read_text(encoding="utf-8")
            self.assertIn("Paired differences", table)
            self.assertIn("| rule | uptake_rate | 3 |", table)

    def test_single_pair_reports_insufficient_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            self._write_pairs(reports, 1)
            self._run(reports, root / "analysis.json", root / "analysis.md")
            payload = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
            uptake = payload["comparisons"]["rule"]["metrics"]["uptake_rate"]
            self.assertEqual(uptake["point"], 1.0)
            self.assertEqual(uptake["ci_low"], "insufficient")
            self.assertEqual(uptake["ci_high"], "insufficient")
            self.assertIn(
                "insufficient", (root / "analysis.md").read_text(encoding="utf-8")
            )

    def test_unpaired_reports_are_not_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            payload = _session_report("CUE", "rule-seed0-fixture", "cue-0", target=True)
            (reports / "cue.json").write_text(json.dumps(payload), encoding="utf-8")
            self._run(reports, root / "analysis.json", root / "analysis.md")
            result = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
            comparison = result["comparisons"]["rule"]
            self.assertEqual(comparison["status"], "not_comparable")
            self.assertEqual(comparison["n_pairs"], 0)
            self.assertIn(
                "not comparable",
                (root / "analysis.md").read_text(encoding="utf-8"),
            )


class ExperimentRunnerFlagTests(unittest.TestCase):
    def _manifest(self, *extra: str) -> dict[str, object]:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--mode",
                "replay",
                "--dry-run",
                *extra,
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout.split("log=", 1)[0].strip())

    def test_random_operation_manifest_records_assignment_and_seed(self) -> None:
        manifest = self._manifest("--assignment", "random-operation", "--seed", "7")
        self.assertEqual(manifest["assignment"], "random-operation")
        self.assertEqual(manifest["seed"], 7)
        self.assertIn("random-operation", manifest["command"])
        self.assertIn("--pair-key", manifest["command"])
        self.assertIn("rule-seed7-", str(manifest["pair_key"]))

    def test_silence_run_shares_the_pair_key_and_holds_the_cue(self) -> None:
        cue = self._manifest("--seed", "7")
        silence = self._manifest("--seed", "7", "--condition", "SILENCE")
        self.assertEqual(cue["pair_key"], silence["pair_key"])
        self.assertIn("--silent", silence["command"])
        self.assertNotIn("--silent", cue["command"])
        self.assertFalse(silence["tts"])


class SteeringRunnerTests(unittest.TestCase):
    def test_steer_is_accepted_and_recorded_in_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_experiment.py",
                "--mode",
                "replay",
                "--dry-run",
                "--decision",
                "steer",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        manifest = json.loads(completed.stdout.split("log=", 1)[0].strip())
        self.assertEqual(manifest["decision"], "steer")
        self.assertIn("steer", manifest["command"])
