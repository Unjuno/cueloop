from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean
from typing import Mapping

from .session import verify_session


CONDITIONS = ("CONTROL", "CUELOOP")
RUBRIC_FIELDS = (
    "claims", "premises", "dependencies", "counterexamples", "questions", "unsupported_claim_calibration"
)
PILOT_RECORD_FIELDS = {
    "participant_id", "session_id", "condition", "task_id", "rubric",
    "productive_interruptions", "disruptive_interventions", "intervention_count",
    "duration_minutes", "requested_stop", "recall_delay_minutes", "scorer_blinded",
    "resume_latencies_ms", "adapter_warning_count", "analyzer_failure_count",
    "expired_candidates", "cancelled_candidates",
    "input_mode", "consent_acknowledged",
}


@dataclass(frozen=True)
class PilotSession:
    participant_id: str
    session_id: str
    condition: str
    task_id: str
    rubric: dict[str, float]
    productive_interruptions: int
    disruptive_interventions: int
    intervention_count: int
    duration_minutes: float
    requested_stop: bool
    recall_delay_minutes: float
    scorer_blinded: bool
    resume_latencies_ms: tuple[int, ...]
    adapter_warning_count: int
    analyzer_failure_count: int
    expired_candidates: int
    cancelled_candidates: int
    input_mode: str
    consent_acknowledged: bool

    @property
    def delayed_score(self) -> float:
        return sum(self.rubric.values())


def generate_assignment(participant_ids: list[str], tasks: list[str], seed: int = 20260812) -> list[dict[str, str | int]]:
    if not participant_ids or len(tasks) < 2 or len(tasks) % 2:
        raise ValueError("assignment requires participants and an even number of at least two tasks")
    if any(not value.strip() for value in [*participant_ids, *tasks]):
        raise ValueError("participant and task IDs must not be empty")
    if len(set(participant_ids)) != len(participant_ids) or len(set(tasks)) != len(tasks):
        raise ValueError("participant and task IDs must be unique")
    rng = random.Random(seed)
    shuffled_tasks = tasks[:]
    rng.shuffle(shuffled_tasks)
    rows: list[dict[str, str | int]] = []
    for participant_index, participant in enumerate(participant_ids):
        order = CONDITIONS if participant_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for session_index, task in enumerate(shuffled_tasks):
            rows.append({
                "participant_id": participant,
                "session_number": session_index + 1,
                "condition": order[session_index % 2],
                "task_id": task,
            })
    return rows


def load_pilot_data(path: Path) -> list[PilotSession]:
    sessions: list[PilotSession] = []
    session_ids: set[str] = set()
    participant_tasks: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict) or set(payload) != PILOT_RECORD_FIELDS:
                raise ValueError("pilot record fields do not match protocol")
            for key in ("participant_id", "session_id", "condition", "task_id"):
                if not isinstance(payload[key], str):
                    raise ValueError(f"{key} must be a string")
            for key in ("requested_stop", "scorer_blinded", "consent_acknowledged"):
                if not isinstance(payload[key], bool):
                    raise ValueError(f"{key} must be boolean")
            if not isinstance(payload["resume_latencies_ms"], list):
                raise ValueError("resume_latencies_ms must be an array")
            if payload["condition"] not in CONDITIONS:
                raise ValueError("invalid condition")
            scores = validate_rubric(payload["rubric"])
            session = PilotSession(
                payload["participant_id"], payload["session_id"], payload["condition"],
                payload["task_id"], scores,
                _nonnegative_int(payload["productive_interruptions"], "productive_interruptions"),
                _nonnegative_int(payload["disruptive_interventions"], "disruptive_interventions"),
                _nonnegative_int(payload["intervention_count"], "intervention_count"),
                _number(payload["duration_minutes"], "duration_minutes"), payload["requested_stop"],
                _number(payload["recall_delay_minutes"], "recall_delay_minutes"), payload["scorer_blinded"],
                tuple(_nonnegative_int(value, "resume latency") for value in payload["resume_latencies_ms"]),
                _nonnegative_int(payload["adapter_warning_count"], "adapter_warning_count"),
                _nonnegative_int(payload["analyzer_failure_count"], "analyzer_failure_count"),
                _nonnegative_int(payload["expired_candidates"], "expired_candidates"),
                _nonnegative_int(payload["cancelled_candidates"], "cancelled_candidates"),
                payload["input_mode"], payload["consent_acknowledged"],
            )
            if not session.participant_id.strip() or not session.session_id.strip() or not session.task_id.strip():
                raise ValueError("participant, session, and task IDs must not be empty")
            if session.session_id in session_ids:
                raise ValueError(f"duplicate session_id: {session.session_id}")
            if (session.participant_id, session.task_id) in participant_tasks:
                raise ValueError("a participant cannot repeat the same task")
            if min(session.productive_interruptions, session.disruptive_interventions, session.intervention_count) < 0 or session.duration_minutes <= 0:
                raise ValueError("invalid safety or duration value")
            if max(session.productive_interruptions, session.disruptive_interventions) > session.intervention_count:
                raise ValueError("safety counts cannot exceed intervention_count")
            if session.condition == "CONTROL" and session.intervention_count != 0:
                raise ValueError("CONTROL sessions must not contain interventions")
            if session.recall_delay_minutes < 0:
                raise ValueError("recall_delay_minutes must be non-negative")
            if session.input_mode not in {"LIVE", "REPLAY", "SYNTHETIC"}:
                raise ValueError("input_mode is invalid")
            if min(
                *session.resume_latencies_ms,
                session.adapter_warning_count,
                session.analyzer_failure_count,
                session.expired_candidates,
                session.cancelled_candidates,
            ) < 0:
                raise ValueError("runtime safety metrics must be non-negative")
            sessions.append(session)
            session_ids.add(session.session_id)
            participant_tasks.add((session.participant_id, session.task_id))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid pilot data line {line_number}: {exc}") from exc
    if not sessions:
        raise ValueError("pilot data is empty")
    return sessions


def validate_rubric(rubric: object) -> dict[str, float]:
    if not isinstance(rubric, Mapping) or set(rubric) != set(RUBRIC_FIELDS):
        raise ValueError("rubric fields do not match protocol")
    scores = {key: _number(value, f"rubric {key}") for key, value in rubric.items()}
    if any(not 0 <= value <= 2 for value in scores.values()):
        raise ValueError("rubric scores must be between 0 and 2")
    return scores


def pilot_record_from_session(
    session_directory: Path,
    *,
    participant_id: str,
    task_id: str,
    rubric: object,
    productive_interruptions: int,
    disruptive_interventions: int,
    requested_stop: bool | None = None,
    recall_delay_minutes: float = 10.0,
    scorer_blinded: bool = False,
) -> dict[str, object]:
    if not participant_id.strip() or not task_id.strip():
        raise ValueError("participant and task IDs must not be empty")
    errors = verify_session(session_directory)
    if errors:
        raise ValueError(f"session integrity verification failed: {errors}")
    manifest = json.loads((session_directory / "manifest.json").read_text(encoding="utf-8"))
    condition = manifest.get("condition")
    if condition not in CONDITIONS:
        raise ValueError("session manifest has no valid experiment condition")
    action_counts = manifest.get("action_counts")
    if not isinstance(action_counts, dict):
        raise ValueError("session manifest action counts are invalid")
    intervention_count = sum(
        int(count) for action, count in action_counts.items() if action != "SILENCE"
    )
    duration_minutes = float(manifest.get("session_duration_minutes", 0))
    if duration_minutes <= 0:
        raise ValueError("session manifest duration is invalid")
    if productive_interruptions < 0 or disruptive_interventions < 0:
        raise ValueError("safety counts must be non-negative")
    if max(productive_interruptions, disruptive_interventions) > intervention_count:
        raise ValueError("safety counts cannot exceed the verified intervention count")
    if recall_delay_minutes < 0:
        raise ValueError("recall_delay_minutes must be non-negative")
    verified_requested_stop = manifest.get("participant_requested_stop")
    if not isinstance(verified_requested_stop, bool):
        raise ValueError("session manifest participant stop status is invalid")
    if requested_stop is not None and requested_stop != verified_requested_stop:
        raise ValueError("requested_stop conflicts with the verified session manifest")
    input_mode = manifest.get("input_mode")
    consent_acknowledged = manifest.get("consent_acknowledged")
    if input_mode not in {"LIVE", "REPLAY"} or not isinstance(consent_acknowledged, bool):
        raise ValueError("session manifest input/consent status is invalid")
    runtime_safety = manifest.get("runtime_safety")
    if not isinstance(runtime_safety, dict):
        raise ValueError("session manifest runtime safety metrics are invalid")
    try:
        resume_latencies_ms = [int(value) for value in runtime_safety["resume_latencies_ms"]]
        runtime_counts = {
            key: int(runtime_safety[key])
            for key in (
                "adapter_warning_count", "analyzer_failure_count",
                "expired_candidates", "cancelled_candidates",
            )
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("session manifest runtime safety metrics are invalid") from exc
    if min(*resume_latencies_ms, *runtime_counts.values()) < 0:
        raise ValueError("session manifest runtime safety metrics are negative")
    return {
        "participant_id": participant_id,
        "session_id": manifest["session_id"],
        "condition": condition,
        "task_id": task_id,
        "rubric": validate_rubric(rubric),
        "productive_interruptions": productive_interruptions,
        "disruptive_interventions": disruptive_interventions,
        "intervention_count": intervention_count,
        "duration_minutes": duration_minutes,
        "requested_stop": verified_requested_stop,
        "recall_delay_minutes": recall_delay_minutes,
        "scorer_blinded": scorer_blinded,
        "resume_latencies_ms": resume_latencies_ms,
        **runtime_counts,
        "input_mode": input_mode,
        "consent_acknowledged": consent_acknowledged,
    }


def append_pilot_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if set(record) != PILOT_RECORD_FIELDS:
        raise ValueError("pilot record fields do not match protocol")
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("pilot record session_id is invalid")
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"existing pilot data line {line_number} is invalid") from exc
            if isinstance(existing, dict) and existing.get("session_id") == session_id:
                raise ValueError(f"duplicate session_id: {session_id}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def analyze_pilot(
    sessions: list[PilotSession], source_path: Path, assignment_path: Path | None = None
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[PilotSession]]] = {}
    for session in sessions:
        grouped.setdefault(session.participant_id, {}).setdefault(session.condition, []).append(session)
    complete = {participant: conditions for participant, conditions in grouped.items() if all(conditions.get(c) for c in CONDITIONS)}
    if not complete:
        raise ValueError("no participant has observations in both conditions")
    differences: list[dict[str, float | str]] = []
    for participant, conditions in sorted(complete.items()):
        control = mean(session.delayed_score for session in conditions["CONTROL"])
        treatment = mean(session.delayed_score for session in conditions["CUELOOP"])
        differences.append({"participant_id": participant, "control_mean": control, "cueloop_mean": treatment, "difference": treatment - control})
    difference_values = [float(row["difference"]) for row in differences]
    bootstrap_interval = _bootstrap_mean_interval(difference_values)
    design_complete = len(complete) == len(grouped) and all(
        len(conditions[condition]) >= 3
        for conditions in complete.values()
        for condition in CONDITIONS
    )
    treatment_sessions = [session for session in sessions if session.condition == "CUELOOP"]
    intervention_minutes = sum(session.duration_minutes for session in treatment_sessions)
    resume_latencies = [
        value for session in treatment_sessions for value in session.resume_latencies_ms
    ]
    assignment = _verify_assignment(sessions, assignment_path)
    deviations = [
        {
            "session_id": session.session_id,
            "reasons": [
                *([] if 9.5 <= session.duration_minutes <= 10.5 else ["session duration outside 9.5-10.5 minute window"]),
                *([] if 9 <= session.recall_delay_minutes <= 11 else ["recall delay outside 9-11 minute window"]),
                *([] if session.scorer_blinded else ["scorer was not blinded"]),
                *([] if not session.requested_stop else ["participant requested stop"]),
                *([] if session.adapter_warning_count == 0 else ["ASR adapter warning occurred"]),
                *([] if session.analyzer_failure_count == 0 else ["analyzer failure occurred"]),
                *([] if session.input_mode == "LIVE" else ["session did not use live input"]),
                *([] if session.consent_acknowledged else ["participant consent was not acknowledged"]),
            ],
        }
        for session in sessions
        if not (
            9.5 <= session.duration_minutes <= 10.5
            and 9 <= session.recall_delay_minutes <= 11
            and session.scorer_blinded
            and not session.requested_stop
            and session.adapter_warning_count == 0
            and session.analyzer_failure_count == 0
            and session.input_mode == "LIVE"
            and session.consent_acknowledged
        )
    ]
    return {
        "experiment": "within-person retained-reasoning pilot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "participants_total": len(grouped), "participants_complete": len(complete), "sessions_total": len(sessions),
        "primary_endpoint": {
            "participant_differences": differences,
            "mean_within_person_difference": mean(difference_values),
            "bootstrap_95_interval": bootstrap_interval,
            "paired_randomization_p_value": _paired_randomization_p_value(difference_values),
            "scale": f"sum of {len(RUBRIC_FIELDS)} blinded 0-2 delayed-recall rubric items",
        },
        "safety": {
            "productive_interruptions": sum(s.productive_interruptions for s in treatment_sessions),
            "disruptive_interventions": sum(s.disruptive_interventions for s in treatment_sessions),
            "requested_stops": sum(s.requested_stop for s in sessions),
            "interventions_per_minute": sum(s.intervention_count for s in treatment_sessions) / intervention_minutes if intervention_minutes else 0.0,
            "mean_resume_latency_ms": mean(resume_latencies) if resume_latencies else None,
            "adapter_warning_count": sum(s.adapter_warning_count for s in sessions),
            "analyzer_failure_count": sum(s.analyzer_failure_count for s in sessions),
            "expired_candidates": sum(s.expired_candidates for s in treatment_sessions),
            "cancelled_candidates": sum(s.cancelled_candidates for s in treatment_sessions),
        },
        "protocol_adherence": {
            "compliant_sessions": len(sessions) - len(deviations),
            "deviations": deviations,
            "minimum_sessions_per_condition": 3,
            "design_complete": design_complete,
            "assignment": assignment,
            "primary_endpoint_confirmatory": (
                not deviations and design_complete and assignment["verified"]
            ),
        },
        "interpretation_guardrail": "A personal pilot estimates local effects and failure modes; it does not establish general cognitive enhancement.",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bootstrap_mean_interval(values: list[float], samples: int = 5_000) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires paired differences")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(20260812)
    means = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def _paired_randomization_p_value(values: list[float]) -> float:
    observed = abs(mean(values))
    if len(values) <= 20:
        randomized = [abs(mean(sign * value for sign, value in zip(signs, values))) for signs in product((-1, 1), repeat=len(values))]
    else:
        rng = random.Random(20260812)
        randomized = [
            abs(mean(rng.choice((-1, 1)) * value for value in values)) for _ in range(100_000)
        ]
    return sum(value >= observed - 1e-12 for value in randomized) / len(randomized)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _verify_assignment(
    sessions: list[PilotSession], assignment_path: Path | None
) -> dict[str, object]:
    if assignment_path is None:
        return {"verified": False, "reason": "assignment file was not provided", "sha256": None}
    try:
        rows = json.loads(assignment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"assignment is unreadable: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("assignment must be a non-empty array")
    expected: dict[tuple[str, str], str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "participant_id", "session_number", "condition", "task_id"
        }:
            raise ValueError("assignment row fields are invalid")
        participant = row["participant_id"]
        task = row["task_id"]
        condition = row["condition"]
        number = row["session_number"]
        if (
            not isinstance(participant, str) or not participant
            or not isinstance(task, str) or not task
            or condition not in CONDITIONS
            or isinstance(number, bool) or not isinstance(number, int) or number < 1
        ):
            raise ValueError("assignment row values are invalid")
        key = (participant, task)
        if key in expected:
            raise ValueError("assignment repeats a participant/task pair")
        expected[key] = condition
    mismatches = [
        session.session_id
        for session in sessions
        if expected.get((session.participant_id, session.task_id)) != session.condition
    ]
    return {
        "verified": not mismatches,
        "reason": None if not mismatches else f"assignment mismatch: {mismatches}",
        "sha256": hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
    }
