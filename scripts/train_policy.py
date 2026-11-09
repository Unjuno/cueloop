#!/usr/bin/env python3
"""Train and evaluate the fast linear intervention policy from JSONL logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.policy_model import (
    evaluate_policy,
    namespace_intervention_ids,
    samples_from_records,
    split_samples,
    train_linear_policy,
)


def _records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield from namespace_intervention_ids([payload], path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fast linear cue policy from logged sessions."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL logs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "sync-policy.json",
        help="Output model path (default: artifacts/sync-policy.json).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the training and holdout metrics as JSON.",
    )
    parser.add_argument(
        "--label",
        choices=["carryover", "uptake", "blend"],
        default="carryover",
        help="Reward label (default: carryover).",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Held-out fraction for evaluation (default: 0.25; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Refuse to write a model below this many samples (default: 20).",
    )
    return parser


_KNOWN_ASSIGNMENT = re.compile(
    r"^(canonical|alternate|local-llm|unspecified"
    r"|randomized\(\d+\)"
    r"|fixed\(.*\)"
    r"|random-operation\(seed=\d+;wording=[A-Za-z0-9_.-]+\))$"
)


def _assignment_counts(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("kind") == "intervention":
                    counts[str(payload.get("assignment", "unspecified"))] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assignments = _assignment_counts(args.logs)
    # Unknown assignment strings train like any other: the reward attaches to
    # the delivered operation. They are reported, never silently discarded.
    unknown_assignments = sorted(
        name for name in assignments if not _KNOWN_ASSIGNMENT.match(name)
    )
    for name in unknown_assignments:
        print(
            f"warning: unknown assignment {name!r} appears "
            f"{assignments[name]} times; training on delivered operations anyway",
            file=sys.stderr,
        )

    samples = samples_from_records(_records(args.logs), label=args.label)
    if len(samples) < args.min_samples:
        print(
            f"error: {len(samples)} usable samples, need {args.min_samples}",
            file=sys.stderr,
        )
        return 2
    try:
        train_samples, eval_samples = split_samples(
            samples, eval_fraction=args.eval_fraction, seed=args.seed
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        model = train_linear_policy(
            train_samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        evaluated_on = "holdout"
    except ValueError as exc:
        # A small or single-class holdout cannot support fitting. Fall back to
        # all samples and label the metrics as in-sample rather than pretending
        # the split was valid.
        print(f"warning: {exc}; evaluating in-sample", file=sys.stderr)
        try:
            model = train_linear_policy(
                samples,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
        except ValueError as fallback_exc:
            print(f"error: {fallback_exc}", file=sys.stderr)
            return 2
        eval_samples = list(samples)
        evaluated_on = "in-sample"

    train_metrics = evaluate_policy(model, train_samples)
    eval_metrics = evaluate_policy(model, eval_samples) if eval_samples else None
    model.metadata["label"] = args.label
    model.metadata["logs"] = [str(path) for path in args.logs]
    model.metadata["dataset_samples"] = len(samples)
    model.metadata["assignments"] = dict(sorted(assignments.items()))
    model.metadata["evaluated_on"] = evaluated_on
    model.metadata["train_metrics"] = train_metrics
    if eval_metrics is not None:
        model.metadata["eval_metrics"] = eval_metrics
    model.save(args.output)

    summary = {
        "model": str(args.output),
        "samples": len(samples),
        "positives": sum(1 for sample in samples if sample.reward >= 0.5),
        "threshold": model.threshold,
        "label": args.label,
        "assignments": dict(sorted(assignments.items())),
        "unknown_assignments": unknown_assignments,
        "evaluated_on": evaluated_on,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
