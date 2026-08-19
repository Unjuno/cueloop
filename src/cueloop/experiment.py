from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .core import Action, ControllerConfig, HeuristicAnalyzer, InterventionController


@dataclass(frozen=True)
class ExpectedTurn:
    text: str
    expected_action: Action


@dataclass(frozen=True)
class EvaluationCase:
    session_id: str
    split: str
    turns: tuple[ExpectedTurn, ...]


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    correct: int
    harmful_interruptions: int
    missed_interventions: int
    wrong_interventions: int
    weighted_loss: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    config: dict[str, float | int] | None
    metrics: EvaluationMetrics
    predictions: tuple[dict[str, str], ...]


def load_dataset(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    session_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict) or set(payload) != {"session_id", "split", "turns"}:
                raise ValueError("case keys must be session_id, split, and turns")
            if payload["split"] not in {"tune", "eval"}:
                raise ValueError("split must be tune or eval")
            session_id = str(payload["session_id"]).strip()
            if not session_id or session_id in session_ids:
                raise ValueError("session_id must be non-empty and unique")
            turns: list[ExpectedTurn] = []
            for turn in payload["turns"]:
                if not isinstance(turn, dict) or set(turn) != {"text", "expected_action"}:
                    raise ValueError("turn keys must be text and expected_action")
                text = turn["text"].strip()
                if not text:
                    raise ValueError("turn text must not be empty")
                turns.append(ExpectedTurn(text, Action(turn["expected_action"])))
            if not turns:
                raise ValueError("case must contain at least one turn")
            cases.append(EvaluationCase(session_id, payload["split"], tuple(turns)))
            session_ids.add(session_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid dataset line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("dataset is empty")
    if not {case.split for case in cases} >= {"tune", "eval"}:
        raise ValueError("dataset must include both tune and eval splits")
    texts_by_split = {
        split: {turn.text for case in cases if case.split == split for turn in case.turns}
        for split in ("tune", "eval")
    }
    if texts_by_split["tune"] & texts_by_split["eval"]:
        raise ValueError("tune and eval splits contain duplicate turn text")
    for split in ("tune", "eval"):
        actions = {
            turn.expected_action for case in cases if case.split == split for turn in case.turns
        }
        if len(texts_by_split[split]) < 8:
            raise ValueError(f"{split} split must contain at least 8 distinct turns")
        if Action.SILENCE not in actions or len(actions - {Action.SILENCE}) < 3:
            raise ValueError(f"{split} split lacks required action coverage")
    return cases


def evaluate(
    cases: Iterable[EvaluationCase],
    config: ControllerConfig | None,
    *,
    name: str,
    proposal_only: bool = False,
) -> EvaluationResult:
    analyzer = HeuristicAnalyzer()
    predictions: list[dict[str, str]] = []
    correct = harmful = missed = wrong = total = 0
    for case in cases:
        controller = InterventionController(config) if not proposal_only else None
        for index, turn in enumerate(case.turns):
            analysis = analyzer.analyze(turn.text)
            actual = analysis.proposed_action if proposal_only else controller.decide(analysis).action
            expected = turn.expected_action
            total += 1
            if actual is expected:
                correct += 1
            elif expected is Action.SILENCE and actual is not Action.SILENCE:
                harmful += 1
            elif expected is not Action.SILENCE and actual is Action.SILENCE:
                missed += 1
            else:
                wrong += 1
            predictions.append(
                {
                    "session_id": case.session_id,
                    "turn": str(index + 1),
                    "expected": expected.value,
                    "actual": actual.value,
                }
            )
    if total == 0:
        raise ValueError("evaluation selection is empty")
    loss = harmful * 3.0 + missed * 1.0 + wrong * 2.0
    metrics = EvaluationMetrics(total, correct, harmful, missed, wrong, loss, correct / total)
    return EvaluationResult(name, None if config is None else config.to_dict(), metrics, tuple(predictions))


def candidate_configs() -> Iterable[ControllerConfig]:
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
        for cooldown in (1, 2, 3):
            yield ControllerConfig(threshold, cooldown, 16)


def tune_policy(
    tune_cases: list[EvaluationCase], current: ControllerConfig
) -> tuple[ControllerConfig, list[EvaluationResult]]:
    current_result = evaluate(tune_cases, current, name="current-tune")
    trials = [evaluate(tune_cases, candidate, name="candidate") for candidate in candidate_configs()]
    safe = [
        trial
        for trial in trials
        if trial.metrics.harmful_interruptions <= current_result.metrics.harmful_interruptions
    ]
    best = min(
        safe,
        key=lambda result: (
            result.metrics.weighted_loss,
            -result.metrics.correct,
            -float(result.config["minimum_confidence"]),
            int(result.config["cooldown_turns"]),
        ),
    )
    assert best.config is not None
    return ControllerConfig(**best.config), trials


def run_experiment(
    dataset_path: Path,
    report_path: Path,
    policy_path: Path,
    current: ControllerConfig | None = None,
) -> dict[str, object]:
    resolved_paths = {
        "dataset": dataset_path.resolve(),
        "report": report_path.resolve(),
        "policy": policy_path.resolve(),
    }
    if len(set(resolved_paths.values())) != 3:
        raise ValueError("dataset, report, and policy paths must be distinct")
    config = current or ControllerConfig()
    cases = load_dataset(dataset_path)
    tune_cases = [case for case in cases if case.split == "tune"]
    eval_cases = [case for case in cases if case.split == "eval"]
    tuned, trials = tune_policy(tune_cases, config)

    baseline = evaluate(eval_cases, None, name="proposal-only", proposal_only=True)
    current_tune = evaluate(tune_cases, config, name="current-tune")
    tuned_tune = evaluate(tune_cases, tuned, name="tuned-tune")
    current_eval = evaluate(eval_cases, config, name="current-eval")
    tuned_eval = evaluate(eval_cases, tuned, name="tuned-eval")
    adopted = (
        tuned_tune.metrics.weighted_loss < current_tune.metrics.weighted_loss
        and tuned_tune.metrics.harmful_interruptions <= current_tune.metrics.harmful_interruptions
        and tuned_eval.metrics.weighted_loss <= current_eval.metrics.weighted_loss
        and tuned_eval.metrics.harmful_interruptions <= current_eval.metrics.harmful_interruptions
    )
    selected = tuned if adopted else config
    report: dict[str, object] = {
        "experiment": "offline deterministic policy replay",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "hypothesis": "deterministic suppression reduces harmful interruptions without excessive misses",
        "loss_weights": {"harmful_interruption": 3.0, "missed_intervention": 1.0, "wrong_intervention": 2.0},
        "baseline_eval": _result_dict(baseline),
        "current_tune": _result_dict(current_tune),
        "tuned_tune": _result_dict(tuned_tune),
        "current_eval": _result_dict(current_eval),
        "tuned_eval": _result_dict(tuned_eval),
        "candidate_count": len(trials),
        "adopted": adopted,
        "selected_policy": selected.to_dict(),
        "epistemic_limit": "This replay validates controller behavior against labels; it does not establish cognitive improvement.",
    }
    _write_json_atomic(report_path, report)
    _write_json_atomic(policy_path, selected.to_dict())
    return report


def _result_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "name": result.name,
        "config": result.config,
        "metrics": asdict(result.metrics),
        "predictions": list(result.predictions),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
