import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tune_operation_weights import _is_gated, derive


def write_cue_log(path: Path, *, operation: str, adopted: bool) -> None:
    lines = [
        json.dumps(
            {"kind": "intervention", "operation": operation, "intervention_id": "i1"}
        ),
        json.dumps(
            {
                "kind": "outcome",
                "operation": operation,
                "intervention_id": "i1",
                "target_operation_seen": adopted,
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TuneOperationWeightsTests(unittest.TestCase):
    def test_is_gated_detects_all_gated_name_tags(self) -> None:
        for name in (
            "self-accuracy-gated01-p1-paired-20261122.jsonl",
            "self-accuracy-v2gate-p1-paired-20261122.jsonl",
            "self-accuracy-v3gate-p1-paired-20261122.jsonl",
            "self-accuracy-floor03-p1-paired-20261122.jsonl",
        ):
            self.assertTrue(_is_gated(Path(name)), name)
        self.assertFalse(_is_gated(Path("self-accuracy-pty-paired-20261121.jsonl")))
        self.assertFalse(_is_gated(Path("self-design-choice-paired-20261115.jsonl")))

    def test_exclude_gated_reproduces_baseline_only_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "self-accuracy-base-paired-20261121.jsonl"
            gated = root / "self-accuracy-gated99-p1-paired-20261122.jsonl"
            # Baseline: 'ground' always adopted -> rate 1.0 -> weight 2.0 (cap).
            write_cue_log(baseline, operation="ground", adopted=True)
            # Gated: 'dependency' never adopted -> would drag weight down if used.
            write_cue_log(gated, operation="dependency", adopted=False)

            all_logs = [baseline, gated]
            base_only = [p for p in all_logs if not _is_gated(p)]

            weights_all, _ = derive(all_logs)
            weights_base, _ = derive(base_only)

            self.assertEqual(base_only, [baseline])
            self.assertIn("ground", weights_base)
            self.assertIn("dependency", weights_all)
            # With exclusion, the gated-only operation disappears entirely.
            self.assertNotIn("dependency", weights_base)
            self.assertEqual(weights_base["ground"], 2.0)

    def test_cli_exclude_gated_flag_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "self-accuracy-base-paired-20261121.jsonl"
            gated = root / "self-accuracy-gated99-p1-paired-20261122.jsonl"
            write_cue_log(baseline, operation="ground", adopted=True)
            write_cue_log(gated, operation="dependency", adopted=False)
            output = root / "weights.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "tune_operation_weights.py"),
                    str(baseline),
                    str(gated),
                    "--exclude-gated",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            weights = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("dependency", weights)
            self.assertEqual(weights["ground"], 2.0)


if __name__ == "__main__":
    unittest.main()
