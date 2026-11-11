"""Fast linear policy learned from logged interventions.

The local LLM is a useful semantic prior but too slow for a live boundary.
This module distills the logged state-action-outcome tuples into a small
linear reward model. Inference is a dot product over a fixed feature vector,
so it runs in microseconds and never blocks the event loop.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .models import SyncState
from .vocabulary import CONTEXT_LINKS, OPERATIONS, PRIORITY

MODEL_VERSION = 1
FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    *(f"operation:{operation}" for operation in OPERATIONS),
    "prediction_confidence",
    "prediction_length",
    "text_length",
    "chars_per_second",
    "elapsed_seconds",
    "prediction_match",
    "context_turns",
    "clause_operations",
    "recent_operations",
    "priority",
    "is_first_missing",
    "confidence_x_priority",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scaled(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return _clamp01(float(value) / scale)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_strings(value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item) for item in value if isinstance(item, str))


@dataclass(frozen=True)
class PolicyObservation:
    """The subset of speech state needed by the fast linear policy."""

    text_length: int = 0
    context_turns: int = 0
    chars_per_second: float = 0.0
    elapsed_ms: int = 0
    prediction_confidence: float = 0.0
    prediction_length: int = 0
    prediction_match: int = 0
    prediction_operations: frozenset[str] = frozenset()
    clause_operations: frozenset[str] = frozenset()
    recent_operations: frozenset[str] = frozenset()
    at_boundary: bool = True
    diverged: bool = False

    @classmethod
    def from_state(cls, state: SyncState) -> PolicyObservation:
        prediction = state.prediction
        return cls(
            text_length=len(state.text),
            context_turns=len(state.context),
            chars_per_second=state.chars_per_second,
            elapsed_ms=max(0, state.now_ms - state.started_at_ms),
            prediction_confidence=prediction.confidence if prediction else 0.0,
            prediction_length=len(prediction.span) if prediction else 0,
            prediction_match=state.prediction_match,
            prediction_operations=(
                frozenset(prediction.operations) if prediction else frozenset()
            ),
            clause_operations=state.clause_operations,
            recent_operations=state.recent_operations,
            at_boundary=state.at_boundary,
            diverged=state.diverged,
        )

    @classmethod
    def from_log(cls, payload: Mapping[str, object]) -> PolicyObservation:
        text = payload.get("text", payload.get("base_text", ""))
        prediction_operations = _as_strings(payload.get("prediction_operations"))
        prediction_length = _as_int(payload.get("prediction_length"))
        if prediction_length == 0:
            prediction_length = len(str(payload.get("prediction_span", "")))
        return cls(
            text_length=_as_int(payload.get("text_length"), len(str(text))),
            context_turns=_as_int(payload.get("context_turns")),
            chars_per_second=_as_float(payload.get("chars_per_second")),
            elapsed_ms=_as_int(payload.get("elapsed_ms")),
            prediction_confidence=_as_float(payload.get("prediction_confidence")),
            prediction_length=prediction_length,
            prediction_match=_as_int(payload.get("prediction_match")),
            prediction_operations=prediction_operations,
            clause_operations=_as_strings(payload.get("clause_operations")),
            recent_operations=_as_strings(payload.get("recent_operations")),
            at_boundary=bool(payload.get("at_boundary", True)),
            diverged=bool(payload.get("diverged", False)),
        )

    def covered(self) -> frozenset[str]:
        return frozenset(
            self.prediction_operations | self.clause_operations | self.recent_operations
        )

    def missing_operations(self) -> tuple[str, ...]:
        covered = self.covered()
        return tuple(operation for operation in PRIORITY if operation not in covered)

    def ranked_missing_operations(self) -> tuple[str, ...]:
        """Missing operations ordered by context fit.

        The static PRIORITY list is context-blind: it always prefers `ground`,
        which misfires on clauses that need a different move. When the clause
        (or the prediction, or recent history) carries at least one tagged
        operation, candidates are re-ordered behind the moves that extend it.
        Anchors are unordered tag sets, so scoring is position-independent:
        each anchor contributes a weight equal to its link strength. With no
        tagged signal anywhere this falls back to PRIORITY, identical to
        `missing_operations`.
        """
        missing = self.missing_operations()

        def anchors() -> tuple[tuple[str, float], ...]:
            # the spoken clause is the strongest signal; the prediction is a guess
            values: list[tuple[str, float]] = []
            for source, weight in (
                (self.clause_operations, 1.5),
                (self.recent_operations, 1.0),
                (self.prediction_operations, 1.0),
            ):
                try:
                    values.extend((str(item), weight) for item in source)  # type: ignore[arg-type]
                except TypeError:
                    continue
            deduped: list[tuple[str, float]] = []
            for item, weight in values:
                if item not in {name for name, _ in deduped}:
                    deduped.append((item, weight))
            return tuple(deduped)

        anchors_tuple = anchors()
        if not anchors_tuple:
            return missing

        def score(operation: str) -> tuple[float, ...]:
            total = 0.0
            for anchor, anchor_weight in anchors_tuple:
                links = CONTEXT_LINKS.get(anchor, ())
                if operation in links:
                    # earlier link in the prior = stronger adjacency
                    total += ((len(links) - links.index(operation)) / len(links)) * anchor_weight
            return (-total,)

        return tuple(
            sorted(missing, key=lambda op: (score(op), PRIORITY.index(op)))
        )


def feature_vector(observation: PolicyObservation, operation: str) -> list[float]:
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    missing = observation.missing_operations()
    first_missing = missing[0] if missing else ""
    priority_index = (
        PRIORITY.index(operation) if operation in PRIORITY else len(PRIORITY)
    )
    priority = 1.0 - priority_index / max(1, len(PRIORITY))
    confidence = _clamp01(observation.prediction_confidence)
    values = [
        1.0,
        *(1.0 if operation == candidate else 0.0 for candidate in OPERATIONS),
        confidence,
        _scaled(observation.prediction_length, 20.0),
        _scaled(observation.text_length, 80.0),
        _scaled(observation.chars_per_second, 8.0),
        _scaled(observation.elapsed_ms, 30_000.0),
        _scaled(
            observation.prediction_match,
            max(1.0, float(observation.prediction_length)),
        ),
        _scaled(observation.context_turns, 8.0),
        _scaled(len(observation.clause_operations), 4.0),
        _scaled(len(observation.recent_operations), 4.0),
        priority,
        1.0 if operation == first_missing else 0.0,
        confidence * priority,
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("feature vector does not match FEATURE_NAMES")
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = math.exp(-value)
        return 1.0 / (1.0 + scale)
    scale = math.exp(value)
    return scale / (1.0 + scale)


@dataclass
class LinearPolicyModel:
    """A calibrated probability that choosing an operation yields carryover."""

    weights: list[float]
    threshold: float = 0.5
    metadata: dict[str, object] = field(default_factory=dict)
    label: str = "linear-policy"

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} weights, got {len(self.weights)}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.weights = [float(weight) for weight in self.weights]

    def score(self, observation: PolicyObservation, operation: str) -> float:
        values = feature_vector(observation, operation)
        logit = sum(weight * value for weight, value in zip(self.weights, values))
        return _sigmoid(logit)

    def rank(
        self,
        observation: PolicyObservation,
        operations: Sequence[str] = PRIORITY,
    ) -> list[tuple[float, str]]:
        candidates = [
            (self.score(observation, operation), operation) for operation in operations
        ]
        candidates.sort(key=lambda item: (-item[0], PRIORITY.index(item[1])))
        return candidates

    def choose(
        self, observation: PolicyObservation, operations: Sequence[str] = PRIORITY
    ) -> tuple[str, float] | None:
        ranked = self.rank(observation, operations)
        if not ranked:
            return None
        score, operation = ranked[0]
        if score <= self.threshold:
            return None
        return operation, score

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": MODEL_VERSION,
                    "feature_names": list(FEATURE_NAMES),
                    "weights": self.weights,
                    "threshold": self.threshold,
                    "label": self.label,
                    "metadata": self.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LinearPolicyModel:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read policy model: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"policy model is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise TypeError("policy model must be a JSON object")
        if payload.get("version") != MODEL_VERSION:
            raise ValueError("unsupported policy model version")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("policy model feature schema does not match")
        weights = payload.get("weights")
        if not isinstance(weights, list) or not all(
            isinstance(weight, (int, float)) and not isinstance(weight, bool)
            for weight in weights
        ):
            raise TypeError("policy model weights must be a numeric array")
        threshold = payload.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError("policy model threshold must be numeric")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        label = payload.get("label", "linear-policy")
        if not isinstance(label, str):
            label = "linear-policy"
        return cls(
            weights=[float(weight) for weight in weights],
            threshold=float(threshold),
            metadata=metadata,
            label=label,
        )


@dataclass(frozen=True)
class TrainingSample:
    observation: PolicyObservation
    operation: str
    reward: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.reward <= 1.0:
            raise ValueError("reward must be between 0 and 1")
        if self.weight <= 0:
            raise ValueError("sample weight must be positive")


def _balanced_weights(samples: Sequence[TrainingSample]) -> list[float]:
    positives = sum(sample.weight for sample in samples if sample.reward >= 0.5)
    negatives = sum(sample.weight for sample in samples if sample.reward < 0.5)
    total = positives + negatives
    if positives == 0 or negatives == 0:
        return [1.0 for _ in samples]
    positive_scale = total / (2.0 * positives)
    negative_scale = total / (2.0 * negatives)
    return [
        sample.weight * (positive_scale if sample.reward >= 0.5 else negative_scale)
        for sample in samples
    ]


def train_linear_policy(
    samples: Sequence[TrainingSample],
    *,
    epochs: int = 400,
    learning_rate: float = 0.15,
    l2: float = 0.01,
    threshold: float | None = None,
) -> LinearPolicyModel:
    if len(samples) < 2:
        raise ValueError("at least two training samples are required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    labels = {sample.reward >= 0.5 for sample in samples}
    if len(labels) < 2:
        raise ValueError("training samples require both positive and negative rewards")

    weights = [0.0] * len(FEATURE_NAMES)
    sample_weights = _balanced_weights(samples)
    total_weight = sum(sample_weights)
    for _ in range(epochs):
        gradients = [0.0] * len(FEATURE_NAMES)
        for sample, sample_weight in zip(samples, sample_weights):
            values = feature_vector(sample.observation, sample.operation)
            probability = _sigmoid(
                sum(weight * value for weight, value in zip(weights, values))
            )
            error = (probability - sample.reward) * sample_weight
            for index, value in enumerate(values):
                gradients[index] += error * value
        for index in range(len(weights)):
            gradient = gradients[index] / total_weight
            if index > 0:
                gradient += l2 * weights[index]
            weights[index] -= learning_rate * gradient

    model = LinearPolicyModel(
        weights=weights,
        threshold=0.5,
        metadata={
            "samples": len(samples),
            "positives": sum(1 for sample in samples if sample.reward >= 0.5),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
        },
    )
    if threshold is None:
        predictions = [
            model.score(sample.observation, sample.operation) for sample in samples
        ]
        best_threshold = 0.5
        best_score = -1.0
        for step in range(1, 20):
            candidate = step / 20.0
            true_positive = true_negative = false_positive = false_negative = 0
            for sample, probability in zip(samples, predictions):
                predicted = probability >= candidate
                actual = sample.reward >= 0.5
                if predicted and actual:
                    true_positive += 1
                elif predicted and not actual:
                    false_positive += 1
                elif not predicted and actual:
                    false_negative += 1
                else:
                    true_negative += 1
            sensitivity = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            specificity = (
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive
                else 0.0
            )
            score = (sensitivity + specificity) / 2.0
            if score > best_score:
                best_score = score
                best_threshold = candidate
        threshold = max(0.5, best_threshold)
    model.threshold = threshold
    model.metadata["threshold"] = threshold
    model.metadata["feature_names"] = list(FEATURE_NAMES)
    return model


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / (len(positives) * len(negatives))


def evaluate_policy(
    model: LinearPolicyModel,
    samples: Sequence[TrainingSample],
    *,
    threshold: float | None = None,
) -> dict[str, object]:
    """Score a fitted policy on labeled samples.

    The metrics describe the logged actions only. Because untried operations
    have no outcome, this is not an off-policy estimate of the live policy.
    """
    if not samples:
        raise ValueError("at least one sample is required for evaluation")
    cutoff = model.threshold if threshold is None else threshold
    scores = [model.score(sample.observation, sample.operation) for sample in samples]
    labels = [sample.reward >= 0.5 for sample in samples]
    true_positive = true_negative = false_positive = false_negative = 0
    for score, label in zip(scores, labels):
        predicted = score >= cutoff
        if predicted and label:
            true_positive += 1
        elif predicted and not label:
            false_positive += 1
        elif not predicted and label:
            false_negative += 1
        else:
            true_negative += 1
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    accuracy = (true_positive + true_negative) / len(samples)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = sum(
        (score - (1.0 if label else 0.0)) ** 2 for score, label in zip(scores, labels)
    ) / len(samples)
    return {
        "samples": len(samples),
        "positives": sum(1 for label in labels if label),
        "threshold": cutoff,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(scores, labels),
        "brier": brier,
        "mean_score_positive": (
            sum(score for score, label in zip(scores, labels) if label)
            / max(1, sum(1 for label in labels if label))
        ),
        "mean_score_negative": (
            sum(score for score, label in zip(scores, labels) if not label)
            / max(1, len(labels) - sum(1 for label in labels if label))
        ),
    }


def split_samples(
    samples: Sequence[TrainingSample], *, eval_fraction: float, seed: int = 0
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    """Deterministically split samples into stratified train and eval sets."""
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    if eval_fraction == 0.0 or len(samples) < 4:
        return list(samples), []
    rng = random.Random(seed)
    groups = (
        [sample for sample in samples if sample.reward >= 0.5],
        [sample for sample in samples if sample.reward < 0.5],
    )
    train: list[TrainingSample] = []
    test: list[TrainingSample] = []
    for group in groups:
        rng.shuffle(group)
        take = min(round(len(group) * eval_fraction), max(0, len(group) - 1))
        test.extend(group[:take])
        train.extend(group[take:])
    rng.shuffle(train)
    rng.shuffle(test)
    if len(train) < 2 or not test:
        return list(samples), []
    return train, test


def namespace_intervention_ids(
    records: Iterable[Mapping[str, object]], prefix: str
) -> Iterable[Mapping[str, object]]:
    """Prefix ``intervention_id`` so log files cannot collide.

    Exploration arms digest ``intervention_id`` from utterance, time, text and
    drawn operation only, so a CUE session and its paired SILENCE session draw
    the identical id. Merging such logs without namespacing lets the SILENCE
    delivery overwrite the CUE one and silently drops the sample, so every
    sample is keyed by the file it came from.
    """
    for record in records:
        intervention_id = record.get("intervention_id")
        if isinstance(intervention_id, str) and intervention_id:
            yield {**record, "intervention_id": f"{prefix}:{intervention_id}"}
        else:
            yield record


def samples_from_records(
    records: Iterable[Mapping[str, object]], *, label: str = "carryover"
) -> list[TrainingSample]:
    interventions: dict[str, Mapping[str, object]] = {}
    deliveries: dict[str, str] = {}
    outcomes: dict[str, Mapping[str, object]] = {}
    carryovers: dict[str, Mapping[str, object]] = {}
    for record in records:
        kind = record.get("kind")
        intervention_id = record.get("intervention_id")
        if not isinstance(intervention_id, str):
            continue
        if kind == "intervention":
            interventions[intervention_id] = record
        elif kind == "delivery":
            status = record.get("status")
            if isinstance(status, str):
                deliveries[intervention_id] = status
        elif kind == "outcome":
            outcomes[intervention_id] = record
        elif kind == "carryover":
            carryovers[intervention_id] = record

    samples: list[TrainingSample] = []
    for intervention_id, intervention in interventions.items():
        if deliveries.get(intervention_id) != "DELIVERED":
            continue
        operation = intervention.get("operation")
        if not isinstance(operation, str) or operation not in OPERATIONS:
            continue
        observation = PolicyObservation.from_log(intervention)
        if observation.diverged or not observation.at_boundary:
            continue
        if label == "carryover":
            outcome = carryovers.get(intervention_id)
            if outcome is None:
                continue
            reward = 1.0 if outcome.get("carryover_seen") is True else 0.0
        elif label == "uptake":
            outcome = outcomes.get(intervention_id)
            if outcome is None:
                continue
            reward = 1.0 if outcome.get("target_operation_seen") is True else 0.0
        elif label == "blend":
            outcome = outcomes.get(intervention_id)
            carryover = carryovers.get(intervention_id)
            if outcome is None and carryover is None:
                continue
            uptake = (
                1.0 if outcome and outcome.get("target_operation_seen") is True else 0.0
            )
            carried = (
                1.0 if carryover and carryover.get("carryover_seen") is True else 0.0
            )
            reward = min(1.0, 0.4 * uptake + 0.8 * carried)
        else:
            raise ValueError(f"unknown label: {label}")
        samples.append(
            TrainingSample(
                observation=observation,
                operation=operation,
                reward=reward,
            )
        )
    return samples
