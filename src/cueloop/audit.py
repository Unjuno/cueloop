from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest

from .core import Action, Analysis, State
from .experiment import run_experiment
from .knowledge import KnowledgeIndex, ground_structure
from .memory import retain_session_structure
from .personalization import PersonalizationProfile
from .pilot import analyze_pilot, load_pilot_data
from .soak import run_soak
from .session import SessionRetention, purge_expired_sessions, run_integrated_session, verify_session
from .structure import extract_structure
from .voice import VoiceEvent, VoiceEventType, VoiceSession, run_event_stream


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    evidence: str


def run_completion_audit(root: Path) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    required = (
        "README.md", "docs/deterministic-policy.md", "docs/voice-mvp.md",
        "docs/experiment-protocol.md", "docs/completion-evidence.md",
        "schemas/human-state.schema.json",
        "schemas/intervention-candidate.schema.json", "schemas/session-state.schema.json",
        "tools/macos_speech_events.swift", "tools/macos_speech_events.Info.plist",
        "src/cueloop/memory.py",
    )
    missing = [path for path in required if not (root / path).is_file()]
    checks.append(AuditCheck("required artifacts", not missing, f"missing={missing}"))

    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TestResult()
    suite.run(result)
    checks.append(AuditCheck("unit and contract tests", result.wasSuccessful(), f"run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}"))

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        experiment = run_experiment(root / "experiments/policy-eval.jsonl", temporary / "experiment.json", temporary / "policy.json")
        baseline = experiment["baseline_eval"]["metrics"]
        current = experiment["current_eval"]["metrics"]
        policy_safe = current["harmful_interruptions"] <= baseline["harmful_interruptions"] and current["weighted_loss"] <= baseline["weighted_loss"]
        checks.append(AuditCheck("held-out policy safety", policy_safe, f"baseline_harmful={baseline['harmful_interruptions']} current_harmful={current['harmful_interruptions']}"))

        with (root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
            voice_results = run_event_stream(lines, VoiceSession(), temporary / "voice.jsonl")
        voice_safe = bool(voice_results) and all(item.turn.boundary_at_ms >= item.turn.ended_at_ms + 700 for item in voice_results)
        checks.append(AuditCheck("voice boundary integration", voice_safe, f"turns={len(voice_results)}"))

        structure = extract_structure(item.turn.text for item in voice_results)
        checks.append(AuditCheck("post-session structure", bool(structure.claims and structure.premises and structure.dependencies), f"claims={len(structure.claims)} dependencies={len(structure.dependencies)}"))
        index = KnowledgeIndex.from_paths([root / "README.md", root / "docs/product-axioms.md"])
        grounded = ground_structure(structure.to_dict(), index)
        grounding_valid = bool(grounded["grounding"]) and all(item["status"] in {"SUPPORTED", "UNSUPPORTED"} for item in grounded["grounding"])
        checks.append(AuditCheck("reviewable grounding", grounding_valid, f"items={len(grounded['grounding'])}"))

        with (root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
            outcome = run_integrated_session(
                lines,
                temporary / "sessions",
                goal="completion audit",
                source_paths=[root / "README.md"],
                session_id="audit-session",
            )
        session_errors = verify_session(outcome.directory)
        first_voice_record = json.loads(
            (outcome.directory / "voice.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        checks.append(
            AuditCheck(
                "atomic integrated session",
                not session_errors,
                f"artifacts={sorted(outcome.manifest['artifacts_sha256'])} errors={session_errors}",
            )
        )
        candidate = first_voice_record.get("candidate", {})
        lifecycle = first_voice_record.get("candidate_lifecycle", {})
        formal_candidate = (
            set(candidate) == {
                "id", "action", "text", "intended_effect", "state", "confidence",
                "created_at", "expires_at", "delivery_constraints", "source_refs",
            }
            and lifecycle.get("status") in {"DELIVERED", "CANCELLED", "EXPIRED"}
            and bool(candidate.get("state", {}).get("evidence"))
        )
        checks.append(
            AuditCheck(
                "formal candidate lifecycle",
                formal_candidate,
                f"candidate_keys={sorted(candidate)} lifecycle={lifecycle.get('status')}",
            )
        )
        memory_path = temporary / "personalization.json"
        retained = retain_session_structure(outcome.directory, "C1", memory_path, retention_days=30)
        memory_profile = PersonalizationProfile.load(memory_path)
        memory_safe = (
            len(memory_profile.retained_structures) == 1
            and memory_profile.retained_structures[0].id == retained.id
            and memory_profile.forget_structure(retained.id)
        )
        checks.append(
            AuditCheck(
                "explicit structure memory",
                memory_safe,
                f"retained={retained.id} forgotten={memory_safe}",
            )
        )

        with (root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
            private = run_integrated_session(
                lines,
                temporary / "private-sessions",
                goal="privacy audit",
                retention=SessionRetention(transcript_retained=False),
                session_id="private",
            )
        private_safe = not (private.directory / "voice.jsonl").exists() and not verify_session(private.directory)
        checks.append(
            AuditCheck(
                "no-transcript retention",
                private_safe,
                f"artifacts={sorted(private.manifest['artifacts_sha256'])}",
            )
        )
        purge_preview = purge_expired_sessions(
            temporary / "sessions",
            now=datetime.now(timezone.utc) + timedelta(days=31),
        )
        checks.append(
            AuditCheck(
                "retention enforcement",
                bool(purge_preview and purge_preview[0].reason == "expired" and not purge_preview[0].removed),
                f"preview={purge_preview}",
            )
        )
        purge_applied = purge_expired_sessions(
            temporary / "sessions",
            now=datetime.now(timezone.utc) + timedelta(days=31),
            confirm=True,
        )
        checks.append(
            AuditCheck(
                "retention deletion applied",
                bool(purge_applied and purge_applied[0].removed and not outcome.directory.exists()),
                f"removed={purge_applied[0].removed if purge_applied else False}",
            )
        )

    class DeadlineAnalyzer:
        def analyze(self, text, context=None):
            del text, context
            time.sleep(0.02)
            return Analysis(State.STRUCTURE_GAP, 0.9, ("late",), Action.QUESTION, "何に依存？")

    deadline_session = VoiceSession(analyzer=DeadlineAnalyzer(), max_decision_latency_ms=701)
    deadline_result = None
    for event in (
        VoiceEvent(VoiceEventType.SPEECH_STARTED, 0),
        VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, 100, "性能が良い"),
        VoiceEvent(VoiceEventType.SPEECH_ENDED, 200),
        VoiceEvent(VoiceEventType.TICK, 900),
    ):
        deadline_result = deadline_session.feed(event) or deadline_result
    deadline_session.feed(VoiceEvent(VoiceEventType.PARTICIPANT_STOP, 901))
    runtime_safe = bool(
        deadline_result
        and deadline_result.delivery_status == "EXPIRED"
        and deadline_result.decision.action is Action.SILENCE
        and deadline_session.participant_requested_stop
        and deadline_session.analyzer_failure_count == 1
    )
    checks.append(
        AuditCheck(
            "runtime deadline and stop controls",
            runtime_safe,
            f"status={deadline_result.delivery_status if deadline_result else None} stop={deadline_session.participant_requested_stop}",
        )
    )

    soak = run_soak(10)
    checks.append(AuditCheck("ten-minute session invariants", bool(soak["passed"]), f"turns={soak['turns']}"))
    pilot_path = root / "experiments/pilot-synthetic.jsonl"
    pilot = analyze_pilot(load_pilot_data(pilot_path), pilot_path)
    pilot_valid = (
        pilot["participants_complete"] >= 2
        and "bootstrap_95_interval" in pilot["primary_endpoint"]
        and pilot["protocol_adherence"]["primary_endpoint_confirmatory"] is False
        and "interpretation_guardrail" in pilot
    )
    checks.append(
        AuditCheck(
            "human-pilot analysis path",
            pilot_valid,
            f"synthetic_participants={pilot['participants_complete']} confirmatory={pilot['protocol_adherence']['primary_endpoint_confirmatory']} (not efficacy evidence)",
        )
    )
    return checks
