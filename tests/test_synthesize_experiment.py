import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/synthesize_experiment.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class SynthesizeExperimentTests(unittest.TestCase):
    def _generate(self, directory: Path, *extra: str) -> Path:
        completed = run(
            "--output-dir",
            str(directory),
            "--sessions",
            "2",
            "--clauses",
            "4",
            "--backends",
            "rule",
            *extra,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return directory

    def test_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = self._generate(Path(tmp) / "a")
            second = self._generate(Path(tmp) / "b")
            files_a = sorted(path.name for path in first.iterdir())
            files_b = sorted(path.name for path in second.iterdir())
            self.assertEqual(files_a, files_b)
            for name in files_a:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )

    def test_silence_holds_cue_but_keeps_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._generate(Path(tmp) / "run")
            silence = json.loads(
                (directory / "rule-silence-000.json").read_text(encoding="utf-8")
            )
            cue = json.loads(
                (directory / "rule-cue-000.json").read_text(encoding="utf-8")
            )
            self.assertEqual(silence["delivered"], 0)
            self.assertEqual(silence["held"], silence["interventions"])
            self.assertGreaterEqual(silence["interventions"], 1)
            self.assertEqual(len(silence["outcomes"]), silence["interventions"])
            self.assertTrue(silence["synthetic"])
            self.assertEqual(cue["delivered"], cue["interventions"])
            self.assertEqual(cue["pair_key"], silence["pair_key"])
            self.assertNotEqual(cue["session_id"], silence["session_id"])

    def test_declared_effect_is_recovered_by_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._generate(
                Path(tmp) / "effect", "--uptake-effect", "0.4", "--sessions", "6"
            )
            output = Path(tmp) / "analysis.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_experiment.py",
                    str(directory),
                    "--json",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            comparison = json.loads(output.read_text(encoding="utf-8"))["comparisons"][
                "rule"
            ]
            uptake = comparison["metrics"]["uptake_rate"]
            self.assertEqual(comparison["n_pairs"], 6)
            self.assertGreater(uptake["point"], 0.2)
            self.assertGreater(uptake["ci_low"], 0.0)
            delay = comparison["metrics"]["median_boundary_to_delivery_ms"]
            self.assertEqual(delay["point"], 0.0)

    def test_null_arm_reports_zero_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._generate(
                Path(tmp) / "null", "--uptake-effect", "0.0", "--sessions", "6"
            )
            output = Path(tmp) / "analysis.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_experiment.py",
                    str(directory),
                    "--json",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            comparison = json.loads(output.read_text(encoding="utf-8"))["comparisons"][
                "rule"
            ]
            self.assertEqual(comparison["metrics"]["uptake_rate"]["point"], 0.0)

    def test_output_dir_must_be_fresh_and_backend_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._generate(Path(tmp) / "once")
            second = run("--output-dir", str(directory))
            self.assertEqual(second.returncode, 2)
            unknown = run(
                "--output-dir",
                str(Path(tmp) / "unknown"),
                "--backends",
                "rule,magic",
            )
            self.assertEqual(unknown.returncode, 2)
            self.assertIn("unknown backends", unknown.stderr)
