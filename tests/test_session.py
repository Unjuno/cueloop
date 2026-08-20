from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import tempfile
import unittest

from cueloop.session import (
    SessionCondition,
    SessionRetention,
    purge_expired_sessions,
    run_integrated_session,
    verify_session,
)
from cueloop.core import Action, Analysis, State


class IntegratedSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_atomic_session_produces_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="設計仮説を検討する",
                    source_paths=[self.root / "README.md"],
                    session_id="test-session",
                )
            self.assertEqual(verify_session(outcome.directory), [])
            self.assertTrue((outcome.directory / "structure.json").exists())
            self.assertTrue((outcome.directory / "grounding.json").exists())
            self.assertTrue((outcome.directory / "structure-diff.json").exists())
            self.assertAlmostEqual(outcome.manifest["session_duration_minutes"], 5_000 / 60_000)

    def test_structure_diff_file_marks_differences_as_candidates_not_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="設計仮説を検討する",
                    source_paths=[self.root / "README.md"],
                    session_id="diff-session",
                )
            diff_payload = json.loads(
                (outcome.directory / "structure-diff.json").read_text(encoding="utf-8")
            )
            self.assertIsInstance(diff_payload["diffs"], list)
            self.assertIn("candidates", diff_payload["notice"].lower())
            for diff in diff_payload["diffs"]:
                self.assertIn(diff["kind"], {"MISSING_PREMISE", "MISSING_VARIABLE", "MISSING_DEPENDENCY", "CONTRADICTION", "UNSUPPORTED_INFERENCE"})
                self.assertTrue(0.0 <= diff["confidence"] <= 1.0)

    def test_integrated_session_delivers_cue_to_speaker(self) -> None:
        class RecordingSpeaker:
            def __init__(self) -> None:
                self.utterances = []

            def speak(self, text: str) -> None:
                self.utterances.append(text)

        with tempfile.TemporaryDirectory() as directory:
            speaker = RecordingSpeaker()
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                run_integrated_session(
                    lines,
                    Path(directory),
                    goal="audible intervention",
                    speaker=speaker,
                    session_id="speaker",
                )
            self.assertEqual(speaker.utterances, ["何に依存？"])

    def test_control_condition_uses_same_pipeline_but_never_intervenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="control",
                    condition=SessionCondition.CONTROL,
                    session_id="control",
                )
            self.assertEqual(outcome.manifest["condition"], "CONTROL")
            self.assertEqual(outcome.manifest["action_counts"], {"SILENCE": 2})
            self.assertEqual(outcome.manifest["candidate_status_counts"], {"CANCELLED": 1, "DELIVERED": 1})
            self.assertFalse(outcome.manifest["participant_requested_stop"])

    def test_participant_stop_is_preserved_in_verified_manifest(self) -> None:
        events = [
            '{"type":"speech_started","at_ms":0}',
            '{"type":"transcript_final","at_ms":100,"text":"前提を確認する。"}',
            '{"type":"speech_ended","at_ms":200}',
            '{"type":"tick","at_ms":900}',
            '{"type":"participant_stop","at_ms":901}',
            '{"type":"session_ended","at_ms":902}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_integrated_session(
                events, Path(directory), goal="stop", session_id="stopped"
            )
            self.assertTrue(outcome.manifest["participant_requested_stop"])
            self.assertTrue(outcome.manifest["interventions_paused_by_user"])
            self.assertEqual(verify_session(outcome.directory), [])

    def test_integrated_retrieval_uses_only_indexed_source_text(self) -> None:
        class RetrievalAnalyzer:
            def analyze(self, text, context=None):
                return Analysis(
                    State.RETRIEVAL_FAILURE,
                    0.9,
                    ("用語を求めている",),
                    Action.RETRIEVE,
                    "未検証の生成文",
                    ("made-up",),
                )

        class RecordingSpeaker:
            def __init__(self) -> None:
                self.utterances = []

            def speak(self, text: str) -> None:
                self.utterances.append(text)

        events = [
            '{"type":"speech_started","at_ms":0}',
            '{"type":"transcript_final","at_ms":100,"text":"制御理論の用語"}',
            '{"type":"speech_ended","at_ms":200}',
            '{"type":"tick","at_ms":900}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("制御理論ではフィードバックを利用する。", encoding="utf-8")
            speaker = RecordingSpeaker()
            outcome = run_integrated_session(
                events,
                root / "sessions",
                goal="用語を取得する",
                source_paths=[source],
                analyzer=RetrievalAnalyzer(),
                speaker=speaker,
                session_id="retrieval",
            )
            log = json.loads((outcome.directory / "voice.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(log["decision"]["action"], "RETRIEVE")
            self.assertEqual(speaker.utterances, ["制御理論ではフィードバックを利用"])
            self.assertNotIn("made-up", log["decision"]["source_refs"])

    def test_no_transcript_retention_omits_voice_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="private session",
                    retention=SessionRetention(transcript_retained=False),
                    session_id="private-session",
                )
            self.assertFalse((outcome.directory / "voice.jsonl").exists())
            self.assertEqual(verify_session(outcome.directory), [])

    def test_failure_leaves_no_partial_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_integrated_session(["not-json"], Path(directory), goal="test", session_id="failed")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_live_treatment_requires_delivery_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "delivery speaker"):
                run_integrated_session(
                    [], Path(directory), goal="live", live_input=True, session_id="live"
                )

    def test_live_control_requires_consent_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "consent"):
                run_integrated_session(
                    [], Path(directory), goal="live", live_input=True,
                    condition=SessionCondition.CONTROL, session_id="live-control",
                )

    def test_initialization_failure_leaves_no_temporary_directory_or_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(OSError):
                run_integrated_session(
                    [], root, goal="test", source_paths=[root / "missing.txt"], session_id="init"
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_session_id_lock_rejects_concurrent_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".same.lock").write_text(
                json.dumps({"pid": __import__("os").getpid(), "created_at": datetime.now(timezone.utc).isoformat()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "already running"):
                run_integrated_session([], root, goal="test", session_id="same")

    def test_stale_session_lock_and_temporary_directory_are_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".stale.lock").write_text(
                json.dumps({"pid": 2_147_483_647, "created_at": "2020-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )
            stale = root / ".stale-abandoned"
            stale.mkdir()
            (stale / "private.txt").write_text("private", encoding="utf-8")
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, root, goal="test", session_id="stale")
            self.assertTrue(outcome.directory.exists())
            self.assertFalse(stale.exists())
            self.assertFalse((root / ".stale.lock").exists())

    def test_verifier_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="tamper")
            (outcome.directory / "structure.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(any("hash mismatch" in error for error in verify_session(outcome.directory)))

    def test_verifier_detects_undeclared_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="extra")
            (outcome.directory / "extra.json").write_text("{}\n", encoding="utf-8")
            self.assertIn("undeclared artifact: extra.json", verify_session(outcome.directory))

    def test_verifier_rejects_unsafe_artifact_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="unsafe")
            manifest_path = outcome.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts_sha256"]["../outside"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("unsafe artifact name: ../outside", verify_session(outcome.directory))

    def test_verifier_rejects_semantically_invalid_voice_log_even_when_rehashed(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="semantic")
            voice_path = outcome.directory / "voice.jsonl"
            records = [json.loads(line) for line in voice_path.read_text(encoding="utf-8").splitlines()]
            records[0]["candidate"]["confidence"] = 2
            voice_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            manifest_path = outcome.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts_sha256"]["voice.jsonl"] = hashlib.sha256(voice_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("candidate confidence" in error for error in verify_session(outcome.directory)))

    def test_verifier_reconciles_manifest_counts_with_voice_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="counts")
            manifest_path = outcome.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["turn_count"] = 99
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("voice log turn count does not match manifest", verify_session(outcome.directory))

    def test_verifier_rejects_invalid_retention_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, Path(directory), goal="test", session_id="retention")
            manifest_path = outcome.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["retention"]["raw_audio_retained"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("manifest retention booleans are invalid", verify_session(outcome.directory))

    def test_expiry_purge_is_dry_run_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(
                    lines,
                    Path(directory),
                    goal="expiry test",
                    retention=SessionRetention(retention_days=1),
                    session_id="expired",
                )
            future = datetime.now(timezone.utc) + timedelta(days=2)
            preview = purge_expired_sessions(Path(directory), now=future)
            self.assertEqual(preview[0].reason, "expired")
            self.assertFalse(preview[0].removed)
            self.assertTrue(outcome.directory.exists())
            removed = purge_expired_sessions(Path(directory), now=future, confirm=True)
            self.assertTrue(removed[0].removed)
            self.assertFalse(outcome.directory.exists())

    def test_expiry_purge_rejects_symlinked_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, root, goal="test", session_id="real")
            (root / "alias").symlink_to(outcome.directory, target_is_directory=True)
            results = purge_expired_sessions(root, now=datetime.now(timezone.utc), confirm=True)
            alias = next(result for result in results if result.session_id == "alias")
            self.assertIn("symlink", alias.reason)
            self.assertTrue(outcome.directory.exists())

    def test_profile_save_failure_leaves_no_completed_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blocked_parent = root / "blocked"
            blocked_parent.write_text("file", encoding="utf-8")
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                with self.assertRaises(OSError):
                    run_integrated_session(
                        lines,
                        root / "sessions",
                        goal="atomic profile test",
                        profile_path=blocked_parent / "profile.json",
                        session_id="profile-failure",
                    )
            sessions = root / "sessions"
            self.assertEqual(list(sessions.iterdir()) if sessions.exists() else [], [])


if __name__ == "__main__":
    unittest.main()
