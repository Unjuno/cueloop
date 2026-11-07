"""Aggregate the synchronization outcomes into reviewable numbers."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Outcome
from .vocabulary import tag_operations


def summarize(outcomes: Sequence[Outcome]) -> dict[str, object]:
    """Summarize cue uptake and prediction agreement.

    Uptake means the cued operation appears in the continuation window. It is
    a mechanism check, not evidence that the human's reasoning improved.
    """
    finalized = [outcome for outcome in outcomes if outcome.finalized]
    delivered = len(finalized)
    uptake = sum(1 for outcome in finalized if outcome.target_operation_seen)
    followed = sum(1 for outcome in finalized if outcome.prediction_followed)
    per_operation: dict[str, dict[str, float | int]] = {}
    novel = 0
    novel_uptake = 0
    for outcome in finalized:
        operation = outcome.delivery.intervention.operation
        bucket = per_operation.setdefault(
            operation, {"delivered": 0, "uptake": 0, "rate": 0.0}
        )
        bucket["delivered"] = int(bucket["delivered"]) + 1
        if outcome.target_operation_seen:
            bucket["uptake"] = int(bucket["uptake"]) + 1
        if operation not in tag_operations(outcome.delivery.intervention.base_text):
            novel += 1
            if outcome.target_operation_seen:
                novel_uptake += 1
    for bucket in per_operation.values():
        count = int(bucket["delivered"])
        bucket["rate"] = int(bucket["uptake"]) / count if count else 0.0
    return {
        "delivered_outcomes": delivered,
        "uptake": uptake,
        "uptake_rate": uptake / delivered if delivered else 0.0,
        "prediction_followed": followed,
        "prediction_rate": followed / delivered if delivered else 0.0,
        "novel_operation": novel,
        "novel_uptake": novel_uptake,
        "novel_uptake_rate": novel_uptake / novel if novel else 0.0,
        "per_operation": per_operation,
    }
