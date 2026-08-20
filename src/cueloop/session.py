from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
import uuid

from .core import Action, Analysis, Analyzer, ControllerConfig, Decision, HeuristicAnalyzer, InterventionController
from .core import EvidenceKind, State
from .knowledge import GroundedRetrievalAnalyzer, KnowledgeIndex, ground_structure
from .knowledge_diff import build_source_structure, compare_structures
from .personalization import AdaptiveController, PersonalizationProfile
from .structure import extract_structure, write_structure
from .unified_analyzer import UnifiedSessionAnalyzer
from .voice import VoiceSession, append_voice_result, read_events, run_live_event_stream
from .voice import Speaker, MidSpeechInjector, NullSpeaker


class SessionCondition(StrEnum):
    CONTROL = "CONTROL"
    CUELOOP = "CUELOOP"


class SilentController:
    """Control condition: preserve analysis/logging while delivering no intervention."""

    def decide(self, analysis: Analysis) -> Decision:
        return Decision(Action.SILENCE, None, "control condition suppresses all interventions", analysis)


@dataclass(frozen=True)
class SessionRetention:
    transcript_retained: bool = True
    raw_audio_retained: bool = False
    retention_days: int = 30

    def __post_init__(self) -> None:
        if self.raw_audio_retained:
            raise ValueError("the integrated core does not retain raw audio")
        if not 1 <= self.retention_days <= 365:
            raise ValueError("retention_days must be between 1 and 365")


@dataclass(frozen=True)
class SessionOutcome:
    session_id: str
    directory: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class PurgeResult:
    session_id: str
    expires_at: str | None
    removed: bool
    reason: str


def run_integrated_session(
    event_lines: Iterable[str],
    output_root: Path,
    *,
    goal: str,
    source_paths: list[Path] | None = None,
    policy: ControllerConfig | None = None,
    profile_path: Path | None = None,
    retention: SessionRetention | None = None,
    speaker: Speaker | None = None,
    analyzer: Analyzer | None = None,
    condition: SessionCondition = SessionCondition.CUELOOP,
    max_decision_latency_ms: int = 1_500,
    cooldown_ms: int = 5_000,
    live_input: bool = False,
    consent_acknowledged: bool = False,
    session_id: str | None = None,
) -> SessionOutcome:
    normalized_goal = " ".join(goal.strip().split())
    if not normalized_goal:
        raise ValueError("session goal must not be empty")
    selected_retention = retention or SessionRetention()
    if live_input and condition is SessionCondition.CUELOOP and speaker is None:
        raise ValueError("live CUELOOP sessions require a delivery speaker")
    if live_input and not consent_acknowledged:
        raise ValueError("live sessions require explicit consent acknowledgement")
    identifier = session_id or f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError("session_id must be a single safe path component")

    root = output_root.resolve()
    final_directory = root / identifier
    root.mkdir(parents=True, exist_ok=True)
    if final_directory.exists() or final_directory.is_symlink():
        raise FileExistsError(f"session already exists: {identifier}")
    lock_path = root / f".{identifier}.lock"
    _acquire_session_lock(lock_path, root, identifier)
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=root))
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    started_at = datetime.now(timezone.utc)
    try:
        profile = PersonalizationProfile.load(profile_path) if profile_path else PersonalizationProfile()
        knowledge_index = KnowledgeIndex.from_paths(source_paths) if source_paths else None
        selected_analyzer = analyzer or HeuristicAnalyzer()
        if knowledge_index is not None:
            selected_analyzer = GroundedRetrievalAnalyzer(selected_analyzer, knowledge_index)
        trusted_refs = knowledge_index.source_refs if knowledge_index is not None else frozenset()
        controller = (
            SilentController()
            if condition is SessionCondition.CONTROL
            else AdaptiveController(
                InterventionController(policy or ControllerConfig(), trusted_refs),
                profile,
            )
        )
        # Create mid-speech injector for CUELOOP condition with sources
        mid_speech = None
        if condition is SessionCondition.CUELOOP and knowledge_index is not None:
            source_structure = build_source_structure(knowledge_index)
            mid_speech = MidSpeechInjector(
                analyzer=selected_analyzer,
                speaker=speaker or NullSpeaker(),
                interval_ms=2_500,
                cooldown_ms=2_000,
                minimum_confidence=0.6,
                max_cue_characters=8,
                min_text_characters=4,
                always_rotate=False,
                source_structure=source_structure,
                knowledge_store=profile.knowledge_store,
            )
        # Turn-boundary analyzer: adds structural diff + knowledge state
        # awareness to the primary decision path. Mid-speech uses its own
        # MidSpeechUnifiedAnalyzer, but turn-boundary analysis is the
        # authoritative decision point and must have the same signals.
        turn_boundary_analyzer: Analyzer = selected_analyzer
        source_struct = source_structure if condition is SessionCondition.CUELOOP and knowledge_index is not None else None
        if source_struct is not None or profile.knowledge_store.acquisitions:
            turn_boundary_analyzer = UnifiedSessionAnalyzer(
                selected_analyzer,
                source_structure=source_struct,
                knowledge_store=profile.knowledge_store,
            )
        voice_session = VoiceSession(
            analyzer=turn_boundary_analyzer,
            controller=controller,
            speaker=speaker,
            max_decision_latency_ms=max_decision_latency_ms,
            cooldown_ms=cooldown_ms,
            mid_speech=mid_speech,
        )
    except BaseException:
        shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)
        raise
    results = []
    first_event_at_ms: int | None = None
    last_event_at_ms: int | None = None
    status = "FAILED"
    try:
        if live_input:
            def tracked_lines() -> Iterable[str]:
                nonlocal first_event_at_ms, last_event_at_ms
                for event in read_events(event_lines):
                    if first_event_at_ms is None:
                        first_event_at_ms = event.at_ms
                    last_event_at_ms = event.at_ms
                    yield json.dumps(
                        {key: value for key, value in asdict(event).items() if value is not None},
                        ensure_ascii=False,
                    )

            results = run_live_event_stream(
                tracked_lines(), voice_session, temporary / "voice.jsonl"
            )
        else:
            for event in read_events(event_lines):
                if first_event_at_ms is None:
                    first_event_at_ms = event.at_ms
                last_event_at_ms = event.at_ms
                result = voice_session.feed(event)
                if result is not None:
                    results.append(result)
                    append_voice_result(temporary / "voice.jsonl", result)
        if not results:
            raise ValueError("session produced no completed reasoning turns")
        assert first_event_at_ms is not None and last_event_at_ms is not None

        structure = extract_structure(item.turn.text for item in results)
        write_structure(temporary / "structure.json", structure)
        if knowledge_index is not None:
            grounded = ground_structure(structure.to_dict(), knowledge_index)
            _write_json(temporary / "grounding.json", grounded)
            source_structure = build_source_structure(knowledge_index)
            diffs = compare_structures(structure, source_structure)
            _write_json(
                temporary / "structure-diff.json",
                {
                    "diffs": [diff.to_dict() for diff in diffs],
                    "notice": (
                        "Lexical-structural differences are review candidates, "
                        "not proof that the user lacks knowledge or that the "
                        "source is correct."
                    ),
                },
            )

        if not selected_retention.transcript_retained:
            (temporary / "voice.jsonl").unlink(missing_ok=True)

        completed_at = datetime.now(timezone.utc)
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest: dict[str, object] = {
            "schema_version": 1,
            "session_id": identifier,
            "goal": normalized_goal,
            "condition": condition.value,
            "input_mode": "LIVE" if live_input else "REPLAY",
            "consent_acknowledged": consent_acknowledged,
            "status": "COMPLETED",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "turn_count": len(results),
            "session_duration_minutes": (last_event_at_ms - first_event_at_ms) / 60_000,
            "action_counts": _action_counts(results),
            "candidate_status_counts": _candidate_status_counts(results),
            "interventions_paused_by_user": voice_session.interventions_ever_paused,
            "participant_requested_stop": voice_session.participant_requested_stop,
            "max_decision_latency_ms": max_decision_latency_ms,
            "cooldown_ms": cooldown_ms,
            "runtime_safety": {
                "adapter_warning_count": voice_session.adapter_warning_count,
                "analyzer_failure_count": voice_session.analyzer_failure_count,
                "resume_latencies_ms": voice_session.resume_latencies_ms,
                "expired_candidates": sum(
                    result.delivery_status == "EXPIRED" for result in results
                ),
                "cancelled_candidates": sum(
                    result.delivery_status == "CANCELLED" for result in results
                ),
            },
            "retention": {
                **asdict(selected_retention),
                "expires_at": (completed_at + timedelta(days=selected_retention.retention_days)).isoformat(),
            },
            "artifacts_sha256": files,
            "privacy": "local output; raw audio is never retained by this workflow",
        }
        _write_json(temporary / "manifest.json", manifest)
        if final_directory.exists() or final_directory.is_symlink():
            raise FileExistsError(f"session appeared during execution: {identifier}")
        if profile_path:
            profile.save(profile_path)
        temporary.replace(final_directory)
        status = "COMPLETED"
        return SessionOutcome(identifier, final_directory, manifest)
    finally:
        if status != "COMPLETED" and temporary.exists():
            shutil.rmtree(temporary)
        lock_path.unlink(missing_ok=True)


def verify_session(directory: Path) -> list[str]:
    errors: list[str] = []
    root = directory.resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest unreadable: {exc}"]
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != 1:
        errors.append("unsupported manifest schema version")
    if manifest.get("session_id") != root.name:
        errors.append("manifest session_id does not match directory")
    if manifest.get("condition") not in {condition.value for condition in SessionCondition}:
        errors.append("manifest condition is invalid")
    if manifest.get("status") != "COMPLETED":
        errors.append("session status is not COMPLETED")
    errors.extend(_verify_manifest_semantics(manifest))
    artifacts = manifest.get("artifacts_sha256")
    if not isinstance(artifacts, dict):
        errors.append("artifact hash map is missing")
        return errors
    declared: set[str] = set()
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", "..", "manifest.json"}:
            errors.append(f"unsafe artifact name: {name}")
            continue
        declared.add(name)
        if not isinstance(expected, str) or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            errors.append(f"invalid artifact hash: {name}")
            continue
        path = root / name
        if path.is_symlink():
            errors.append(f"artifact must not be a symlink: {name}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"artifact hash mismatch: {name}")
    actual_entries = {path.name for path in root.iterdir() if path.name != "manifest.json"}
    for name in sorted(actual_entries - declared):
        errors.append(f"undeclared artifact: {name}")
    voice_path = root / "voice.jsonl"
    if voice_path.is_file() and "voice.jsonl" in declared:
        errors.extend(_verify_voice_log(voice_path, manifest))
    return errors


def _verify_manifest_semantics(manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version", "session_id", "goal", "condition", "status", "started_at",
        "completed_at", "turn_count", "session_duration_minutes", "action_counts",
        "input_mode", "consent_acknowledged",
        "candidate_status_counts", "interventions_paused_by_user",
        "participant_requested_stop", "max_decision_latency_ms", "cooldown_ms",
        "runtime_safety", "retention", "artifacts_sha256", "privacy",
    }
    missing = required - set(manifest)
    if missing:
        return [f"manifest required fields are missing: {sorted(missing)}"]
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        completed = datetime.fromisoformat(manifest["completed_at"])
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("manifest session timestamps are invalid")
        completed = None
    turn_count = manifest["turn_count"]
    if isinstance(turn_count, bool) or not isinstance(turn_count, int) or turn_count < 1:
        errors.append("manifest turn_count is invalid")
        turn_count = None
    duration = manifest["session_duration_minutes"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("manifest session duration is invalid")
    for field in ("interventions_paused_by_user", "participant_requested_stop"):
        if not isinstance(manifest[field], bool):
            errors.append(f"manifest {field} is invalid")
    if manifest["input_mode"] not in {"LIVE", "REPLAY"}:
        errors.append("manifest input_mode is invalid")
    if not isinstance(manifest["consent_acknowledged"], bool):
        errors.append("manifest consent acknowledgement is invalid")
    if manifest["input_mode"] == "LIVE" and manifest["consent_acknowledged"] is not True:
        errors.append("LIVE manifest has no consent acknowledgement")
    if manifest["participant_requested_stop"] is True and manifest["interventions_paused_by_user"] is not True:
        errors.append("participant stop must pause interventions")
    for field, minimum in (("max_decision_latency_ms", 1), ("cooldown_ms", 0)):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            errors.append(f"manifest {field} is invalid")
    for field in ("action_counts", "candidate_status_counts"):
        counts = manifest[field]
        if not isinstance(counts, dict) or not counts or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        ):
            errors.append(f"manifest {field} is invalid")
        elif turn_count is not None and sum(counts.values()) != turn_count:
            errors.append(f"manifest {field} does not sum to turn_count")
    action_counts = manifest["action_counts"]
    if isinstance(action_counts, dict):
        if set(action_counts) - {action.value for action in Action}:
            errors.append("manifest action_counts contains an invalid action")
        if manifest["condition"] == SessionCondition.CONTROL.value and set(action_counts) - {Action.SILENCE.value}:
            errors.append("CONTROL manifest contains a non-silence action")
    status_counts = manifest["candidate_status_counts"]
    if isinstance(status_counts, dict) and set(status_counts) - {"DELIVERED", "CANCELLED", "EXPIRED"}:
        errors.append("manifest candidate_status_counts contains an invalid status")
    retention = manifest["retention"]
    if not isinstance(retention, dict) or set(retention) != {
        "transcript_retained", "raw_audio_retained", "retention_days", "expires_at"
    }:
        errors.append("manifest retention policy is invalid")
    else:
        if not isinstance(retention["transcript_retained"], bool) or retention["raw_audio_retained"] is not False:
            errors.append("manifest retention booleans are invalid")
        days = retention["retention_days"]
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
            errors.append("manifest retention_days is invalid")
        try:
            expires = datetime.fromisoformat(retention["expires_at"])
            if expires.tzinfo is None or (completed is not None and expires <= completed):
                raise ValueError
        except (TypeError, ValueError):
            errors.append("manifest retention expiry is invalid")
        artifacts = manifest["artifacts_sha256"]
        if isinstance(artifacts, dict):
            voice_declared = "voice.jsonl" in artifacts
            if voice_declared != retention["transcript_retained"]:
                errors.append("manifest transcript retention disagrees with artifacts")
    runtime = manifest["runtime_safety"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "adapter_warning_count", "analyzer_failure_count", "resume_latencies_ms",
        "expired_candidates", "cancelled_candidates",
    }:
        errors.append("manifest runtime safety metrics are invalid")
    else:
        values = [
            runtime["adapter_warning_count"], runtime["analyzer_failure_count"],
            runtime["expired_candidates"], runtime["cancelled_candidates"],
        ]
        latencies = runtime["resume_latencies_ms"]
        if not isinstance(latencies, list):
            errors.append("manifest resume latencies are invalid")
        else:
            values.extend(latencies)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            errors.append("manifest runtime safety counts are invalid")
    if not isinstance(manifest["goal"], str) or not manifest["goal"].strip():
        errors.append("manifest goal is invalid")
    if not isinstance(manifest["privacy"], str) or not manifest["privacy"].strip():
        errors.append("manifest privacy notice is invalid")
    return errors


def _verify_voice_log(path: Path, manifest: dict[str, object]) -> list[str]:
    errors: list[str] = []
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    allowed_candidate_keys = {
        "id", "action", "text", "intended_effect", "state", "confidence",
        "created_at", "expires_at", "delivery_constraints", "source_refs",
    }
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
            candidate = record["candidate"]
            lifecycle = record["candidate_lifecycle"]
            if not isinstance(candidate, dict) or set(candidate) != allowed_candidate_keys:
                raise ValueError("candidate keys do not match schema")
            if not isinstance(candidate["id"], str) or not candidate["id"]:
                raise ValueError("candidate id is invalid")
            proposed_action = Action(candidate["action"])
            if candidate["text"] is not None and (
                not isinstance(candidate["text"], str) or len(candidate["text"]) > 120
            ):
                raise ValueError("candidate text is invalid")
            if proposed_action in {
                Action.QUESTION, Action.MIRROR, Action.PERTURB, Action.CONSTRAIN, Action.RETRIEVE
            } and not candidate["text"]:
                raise ValueError("text action candidate has no text")
            if not isinstance(candidate["intended_effect"], str) or not candidate["intended_effect"]:
                raise ValueError("candidate intended_effect is invalid")
            confidence = float(candidate["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("candidate confidence is invalid")
            created = datetime.fromisoformat(candidate["created_at"])
            expires = datetime.fromisoformat(candidate["expires_at"])
            if created.tzinfo is None or expires.tzinfo is None or expires < created:
                raise ValueError("candidate timestamps are invalid")
            state = candidate["state"]
            if not isinstance(state, dict) or set(state) != {
                "state", "confidence", "evidence", "observed_at", "expires_at", "notes"
            }:
                raise ValueError("candidate state is invalid")
            State(state["state"])
            state_confidence = float(state["confidence"])
            if state_confidence != confidence:
                raise ValueError("candidate and state confidence disagree")
            observed = datetime.fromisoformat(state["observed_at"])
            state_expires = datetime.fromisoformat(state["expires_at"])
            if observed.tzinfo is None or state_expires.tzinfo is None or state_expires < observed:
                raise ValueError("candidate state timestamps are invalid")
            if state["notes"] is not None and not isinstance(state["notes"], str):
                raise ValueError("candidate state notes are invalid")
            if not isinstance(state["evidence"], list) or not state["evidence"]:
                raise ValueError("candidate evidence is invalid")
            for evidence in state["evidence"]:
                if not isinstance(evidence, dict) or set(evidence) != {
                    "kind", "description", "source_span"
                }:
                    raise ValueError("candidate evidence is invalid")
                EvidenceKind(evidence["kind"])
                if not isinstance(evidence["description"], str) or not evidence["description"]:
                    raise ValueError("candidate evidence description is invalid")
            constraints = candidate["delivery_constraints"]
            if not isinstance(constraints, dict) or set(constraints) != {
                "allow_while_speaking", "requires_pause", "minimum_pause_ms",
                "cooldown_ms", "max_delivery_latency_ms",
            }:
                raise ValueError("candidate delivery constraints are invalid")
            if not isinstance(constraints["allow_while_speaking"], bool) or not isinstance(constraints["requires_pause"], bool):
                raise ValueError("candidate delivery booleans are invalid")
            for key in ("minimum_pause_ms", "cooldown_ms", "max_delivery_latency_ms"):
                value = constraints[key]
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ValueError("candidate delivery timing is invalid")
            if not isinstance(candidate["source_refs"], list) or not all(
                isinstance(value, str) and value for value in candidate["source_refs"]
            ):
                raise ValueError("candidate source refs are invalid")
            if proposed_action is Action.RETRIEVE and not candidate["source_refs"]:
                raise ValueError("retrieval candidate has no source refs")
            if proposed_action is not Action.RETRIEVE and candidate["source_refs"]:
                raise ValueError("non-retrieval candidate has source refs")
            if not isinstance(lifecycle, dict) or set(lifecycle) != {"status", "reason", "timestamp"}:
                raise ValueError("candidate lifecycle keys are invalid")
            if lifecycle["status"] not in {"DELIVERED", "CANCELLED", "EXPIRED"}:
                raise ValueError("candidate lifecycle is invalid")
            if not isinstance(lifecycle["reason"], str) or not lifecycle["reason"]:
                raise ValueError("candidate lifecycle reason is invalid")
            lifecycle_at = datetime.fromisoformat(lifecycle["timestamp"])
            if lifecycle_at.tzinfo is None or lifecycle_at < created:
                raise ValueError("candidate lifecycle timestamp is invalid")
            if lifecycle["status"] == "DELIVERED" and lifecycle_at > expires:
                raise ValueError("candidate was delivered after expiry")
            decision = record["decision"]
            final_action = Action(decision["action"])
            status = lifecycle["status"]
            if status == "DELIVERED" and final_action.value != candidate["action"]:
                raise ValueError("delivered candidate does not match final action")
            if status != "DELIVERED" and final_action is not Action.SILENCE:
                raise ValueError("cancelled or expired candidate must end in SILENCE")
            if decision["state"] != state["state"] or float(decision["confidence"]) != confidence:
                raise ValueError("decision and candidate state disagree")
            action_counts[final_action.value] += 1
            status_counts[status] += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid voice log line {line_number}: {exc}")
    if sum(action_counts.values()) != manifest.get("turn_count"):
        errors.append("voice log turn count does not match manifest")
    if dict(sorted(action_counts.items())) != manifest.get("action_counts"):
        errors.append("voice log action counts do not match manifest")
    if dict(sorted(status_counts.items())) != manifest.get("candidate_status_counts"):
        errors.append("voice log candidate status counts do not match manifest")
    if manifest.get("condition") == SessionCondition.CONTROL.value and set(action_counts) - {Action.SILENCE.value}:
        errors.append("CONTROL session contains a delivered intervention")
    runtime = manifest.get("runtime_safety")
    if isinstance(runtime, dict):
        if runtime.get("expired_candidates") != status_counts["EXPIRED"]:
            errors.append("expired candidate count does not match voice log")
        if runtime.get("cancelled_candidates") != status_counts["CANCELLED"]:
            errors.append("cancelled candidate count does not match voice log")
    else:
        errors.append("runtime safety metrics are missing")
    return errors


def purge_expired_sessions(
    output_root: Path,
    *,
    now: datetime | None = None,
    confirm: bool = False,
) -> list[PurgeResult]:
    root = output_root.resolve()
    if not root.exists():
        return []
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("purge reference time must be timezone-aware")
    results: list[PurgeResult] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if directory.is_symlink():
            results.append(
                PurgeResult(directory.name, None, False, "invalid manifest: session directory is a symlink")
            )
            continue
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires_at_raw = manifest["retention"]["expires_at"]
            expires_at = datetime.fromisoformat(expires_at_raw)
            if expires_at.tzinfo is None:
                raise ValueError("expiry is not timezone-aware")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            results.append(PurgeResult(directory.name, None, False, f"invalid manifest: {exc}"))
            continue
        if expires_at > reference:
            results.append(PurgeResult(directory.name, expires_at_raw, False, "not expired"))
            continue
        if confirm:
            # Re-resolve immediately before deletion and require a direct child.
            resolved = directory.resolve()
            if resolved.parent != root or resolved == root:
                raise ValueError("session deletion target escapes output root")
            shutil.rmtree(resolved)
        results.append(PurgeResult(directory.name, expires_at_raw, confirm, "expired"))
    return results


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _acquire_session_lock(lock_path: Path, root: Path, identifier: str) -> None:
    payload = json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()})
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = existing["pid"]
                if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
                    raise ValueError
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise FileExistsError(f"session lock is invalid: {identifier}") from exc
            if _pid_is_running(pid):
                raise FileExistsError(f"session is already running: {identifier}") from exc
            lock_path.unlink()
            for stale in root.glob(f".{identifier}-*"):
                if stale.parent == root and stale.is_dir() and not stale.is_symlink():
                    shutil.rmtree(stale)
            continue
        try:
            os.write(descriptor, payload.encode("utf-8"))
        finally:
            os.close(descriptor)
        return
    raise FileExistsError(f"could not acquire session lock: {identifier}")


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _action_counts(results: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        action = result.decision.action.value
        counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_status_counts(results: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        status = result.delivery_status
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
