#!/usr/bin/env python3
"""Generate synthetic session logs in the real CUELOOP log schema.

The paired-analysis arm (`synthesize_experiment.py`) writes *reports* for the
analyzer. This generator writes the JSONL *logs* the policy trainer consumes:
intervention / delivery / outcome / carryover records with the full
`PolicyObservation.from_log` feature set, and a reward structure the linear
model can demonstrably recover:

    carryover_logit = -1.1
        + 1.3 * (prediction_confidence - 0.75) / 0.2
        + 0.5 * (prediction_length - 20) / 20
        + OP_BIAS[operation]

with OP_BIAS concentrated on a few operations (ground highest). Sessions are
CUE-armed (delivery works) with a small share of CANCELLED/EXPIRED deliveries
and diverged interventions so the trainer's filters are exercised. No humans
are involved; this is training-harness data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.vocabulary import CUE_BY_OPERATION, OPERATIONS, PRIORITY

OP_BIAS = {
    "ground": 1.5,
    "mechanism": 0.8,
    "dependency": 0.5,
    "concretize": 0.3,
    "invert": 0.2,
    "boundary": 0.0,
    "comparison": -0.1,
    "conclusion": -0.7,
}


def _sigmoid(value: float) -> float:
    if value >= 0:
        scale = math.exp(-value)
        return 1.0 / (1.0 + scale)
    scale = math.exp(value)
    return scale / (1.0 + scale)


def _carryover_logit(
    confidence: float, prediction_length: int, operation: str
) -> float:
    return (
        -1.1
        + 1.3 * (confidence - 0.75) / 0.2
        + 0.5 * (prediction_length - 20) / 20
        + OP_BIAS.get(operation, 0.0)
    )


def _session(
    *,
    index: int,
    seed: int,
    assignment: str,
    clauses: int,
) -> list[dict[str, object]]:
    rng = random.Random(f"{seed}|{index}|{assignment}")
    records: list[dict[str, object]] = []
    used: set[str] = set()
    elapsed = 0.0
    cps = round(rng.gauss(4.5, 1.0), 2)
    session_id = f"synthetic-{assignment[:3]}-{index:04d}-s{seed}"
    for boundary in range(clauses):
        missing = [op for op in PRIORITY if op not in used]
        if not missing:
            break
        if assignment == "random-operation":
            requested = missing[0]
            delivered = rng.choice(missing)
            assignment_label = f"random-operation(seed={seed};wording=canonical)"
        else:
            requested = missing[0]
            delivered = requested
            assignment_label = "canonical"
        intervention_id = f"cue-syn-{index:04d}-{boundary:03d}-s{seed}"

        prediction_confidence = round(rng.uniform(0.55, 0.97), 3)
        prediction_length = rng.randint(4, 48)
        prediction_operations = rng.sample(
            list(OPERATIONS), k=rng.choice([0, 0, 1, 1, 2])
        )
        clause_operations = rng.sample(list(OPERATIONS), k=rng.choice([0, 1, 1, 2]))
        recent_operations = rng.sample(list(OPERATIONS), k=rng.choice([0, 0, 1]))
        text_length = rng.randint(20, 140)
        context_turns = rng.randint(0, 6)
        elapsed += rng.randint(900, 2400)
        at_boundary = True
        diverged = rng.random() < 0.05
        boundary_at = int(elapsed)
        span_head = rng.choice(["だから", "つまり", "根拠は", "まず", "前提として"])
        prediction_span = span_head + "あ" * (prediction_length - len(span_head))

        logit = _carryover_logit(prediction_confidence, prediction_length, delivered)
        p_carryover = _sigmoid(logit)
        p_uptake = _sigmoid(logit - 0.5)
        carryover_seen = rng.random() < p_carryover
        target_seen = rng.random() < p_uptake

        records.append(
            {
                "kind": "intervention",
                "at_ms": boundary_at,
                "intervention_id": intervention_id,
                "operation": delivered,
                "requested_operation": requested,
                "cue": CUE_BY_OPERATION[delivered][0],
                "cue_style": "bare",
                "reason": "synthetic log generator",
                "confidence": prediction_confidence,
                "scheduled_at_ms": boundary_at + 100,
                "boundary_at_ms": boundary_at,
                "prediction_latency_ms": rng.randint(0, 50),
                "prediction_penalty_ms": 0,
                "prediction_precomputed": True,
                "decision_latency_ms": rng.randint(0, 20),
                "precomputed": True,
                "decision_source": "precomputed",
                "prediction_span": prediction_span,
                "assignment": assignment_label,
                "text": "あ" * text_length,
                "text_length": text_length,
                "context_turns": context_turns,
                "chars_per_second": cps,
                "elapsed_ms": int(elapsed),
                "prediction_confidence": prediction_confidence,
                "prediction_length": prediction_length,
                "prediction_match": rng.randint(0, max(1, prediction_length // 4)),
                "prediction_operations": prediction_operations,
                "clause_operations": sorted(clause_operations),
                "recent_operations": sorted(recent_operations),
                "at_boundary": at_boundary,
                "diverged": diverged,
                "session_id": session_id,
                "condition": "CUE",
                "synthetic": True,
            }
        )
        # ~8% of deliveries fail or never land; exercises the filters.
        delivery_roll = rng.random()
        if delivery_roll < 0.05:
            delivery_status = "CANCELLED"
        elif delivery_roll < 0.07:
            delivery_status = "EXPIRED"
        else:
            delivery_status = "DELIVERED"
        records.append(
            {
                "kind": "delivery",
                "at_ms": boundary_at + 100,
                "intervention_id": intervention_id,
                "status": delivery_status,
                "operation": delivered,
                "assignment": assignment_label,
                "boundary_to_delivery_ms": 100,
            }
        )
        if delivery_status == "DELIVERED" and not diverged:
            records.append(
                {
                    "kind": "outcome",
                    "at_ms": boundary_at + 800,
                    "intervention_id": intervention_id,
                    "operation": delivered,
                    "requested_operation": requested,
                    "target_operation_seen": target_seen,
                    "prediction_followed": rng.random() < 0.6,
                }
            )
            records.append(
                {
                    "kind": "carryover",
                    "at_ms": boundary_at + 30_000,
                    "intervention_id": intervention_id,
                    "operation": delivered,
                    "carryover": carryover_seen,
                    "carryover_seen": carryover_seen,
                }
            )
        if target_seen or carryover_seen:
            used.add(delivered)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=200)
    parser.add_argument("--clauses", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--assignments",
        default="canonical",
        help="Comma-separated: canonical, random-operation (default: canonical).",
    )
    args = parser.parse_args(argv)
    if args.sessions < 1 or args.clauses < 1:
        print("error: --sessions and --clauses must be positive", file=sys.stderr)
        return 2
    assignments = [item.strip() for item in args.assignments.split(",") if item.strip()]
    for item in assignments:
        if item not in {"canonical", "random-operation"}:
            print(f"error: unknown assignment: {item}", file=sys.stderr)
            return 2
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    for assignment in assignments:
        for index in range(args.sessions):
            records = _session(
                index=index,
                seed=args.seed,
                assignment=assignment,
                clauses=args.clauses,
            )
            path = output / f"{assignment}-{index:04d}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )
            total += len(records)
    print(json.dumps({"records": total, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
