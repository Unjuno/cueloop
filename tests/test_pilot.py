from pathlib import Path
import json
import tempfile
import unittest

from cueloop.pilot import PilotSession, analyze_pilot, append_pilot_record, generate_assignment, load_pilot_data, pilot_record_from_session
from cueloop.session import SessionCondition, run_integrated_session


class PilotTests(unittest.TestCase):
    def test_assignment_counterbalances_first_condition(self) -> None:
        rows = generate_assignment(["p1", "p2"], ["a", "b"], seed=1)
        first = {row["participant_id"]: row["condition"] for row in rows if row["session_number"] == 1}
        self.assertEqual(first, {"p1": "CONTROL", "p2": "CUELOOP"})

    def test_synthetic_analysis_computes_within_person_difference(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "experiments/pilot-synthetic.jsonl"
        report = analyze_pilot(load_pilot_data(path), path)
        self.assertEqual(report["participants_complete"], 2)
        self.assertEqual(len(report["primary_endpoint"]["participant_differences"]), 2)
        self.assertEqual(report["safety"]["productive_interruptions"], 0)
        self.assertFalse(report["protocol_adherence"]["primary_endpoint_confirmatory"])
        self.assertFalse(report["protocol_adherence"]["design_complete"])
        self.assertEqual(report["safety"]["mean_resume_latency_ms"], 1100)
        self.assertIn("bootstrap_95_interval", report["primary_endpoint"])

    def test_rejects_incomplete_rubric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"condition":"CONTROL","rubric":{}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_pilot_data(path)

    def test_rejects_string_boolean_in_safety_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        record = (root / "experiments/pilot-synthetic.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = record.replace('"requested_stop":false', '"requested_stop":"false"')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-bool.jsonl"
            path.write_text(record + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requested_stop must be boolean"):
                load_pilot_data(path)

    def test_rejects_duplicate_session_ids(self) -> None:
        root = Path(__file__).resolve().parents[1]
        first = (root / "experiments/pilot-synthetic.jsonl").read_text(encoding="utf-8").splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(f"{first}\n{first}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate session_id"):
                load_pilot_data(path)

    def test_rejects_safety_count_above_intervention_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        record = (root / "experiments/pilot-synthetic.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = record.replace('"productive_interruptions":0', '"productive_interruptions":1')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-count.jsonl"
            path.write_text(f"{record}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safety counts"):
                load_pilot_data(path)

    def test_protocol_deviation_is_reported_not_hidden(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lines = (root / "experiments/pilot-synthetic.jsonl").read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"scorer_blinded":true', '"scorer_blinded":false')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deviation.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = analyze_pilot(load_pilot_data(path), path)
            self.assertFalse(report["protocol_adherence"]["primary_endpoint_confirmatory"])
            reasons = next(
                item["reasons"] for item in report["protocol_adherence"]["deviations"]
                if item["session_id"] == "p1-control"
            )
            self.assertIn("scorer was not blinded", reasons)

    def test_assignment_mismatch_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sessions = load_pilot_data(root / "experiments/pilot-synthetic.jsonl")
        with tempfile.TemporaryDirectory() as directory:
            assignment = Path(directory) / "assignment.json"
            rows = generate_assignment(
                ["synthetic-p1", "synthetic-p2"], ["task-a", "task-b"], seed=1
            )
            assignment.write_text(json.dumps(rows), encoding="utf-8")
            report = analyze_pilot(sessions, root / "experiments/pilot-synthetic.jsonl", assignment)
            self.assertFalse(report["protocol_adherence"]["assignment"]["verified"])

    def test_complete_adherent_design_can_be_marked_confirmatory(self) -> None:
        tasks = [f"task-{index}" for index in range(6)]
        rows = generate_assignment(["p1"], tasks, seed=3)
        rubric = {
            "claims": 1.0, "premises": 1.0, "dependencies": 1.0,
            "counterexamples": 1.0, "questions": 1.0,
            "unsupported_claim_calibration": 1.0,
        }
        sessions = [
            PilotSession(
                "p1", f"s-{index}", row["condition"], row["task_id"], rubric,
                0, 0, 0 if row["condition"] == "CONTROL" else 1, 10.0, False,
                10.0, True, (), 0, 0, 0, 0, "LIVE", True,
            )
            for index, row in enumerate(rows)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text("synthetic test fixture\n", encoding="utf-8")
            assignment = root / "assignment.json"
            assignment.write_text(json.dumps(rows), encoding="utf-8")
            report = analyze_pilot(sessions, data, assignment)
            self.assertTrue(report["protocol_adherence"]["primary_endpoint_confirmatory"])

    def test_append_rejects_duplicate_before_corrupting_data(self) -> None:
        root = Path(__file__).resolve().parents[1]
        record = json.loads(
            (root / "experiments/pilot-synthetic.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.jsonl"
            append_pilot_record(path, record)
            with self.assertRaisesRegex(ValueError, "duplicate session_id"):
                append_pilot_record(path, record)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_verified_session_becomes_pilot_record(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            with (root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="pilot",
                    condition=SessionCondition.CUELOOP,
                    session_id="pilot-session",
                )
            record = pilot_record_from_session(
                outcome.directory,
                participant_id="p1",
                task_id="t1",
                rubric={
                    "claims": 1,
                    "premises": 1,
                    "dependencies": 1,
                    "counterexamples": 1,
                    "questions": 1,
                    "unsupported_claim_calibration": 1,
                },
                productive_interruptions=0,
                disruptive_interventions=0,
                recall_delay_minutes=10,
                scorer_blinded=True,
            )
            self.assertEqual(record["condition"], "CUELOOP")
            self.assertEqual(record["intervention_count"], 1)
            self.assertFalse(record["requested_stop"])
            self.assertTrue(record["scorer_blinded"])
            self.assertIn("analyzer_failure_count", record)


if __name__ == "__main__":
    unittest.main()
