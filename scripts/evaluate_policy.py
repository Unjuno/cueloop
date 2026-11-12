#!/usr/bin/env python3
"""Evaluate a trained linear cue policy against recorded session logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.policy_model import (
    LinearPolicyModel,
    evaluate_policy,
    namespace_intervention_ids,
    samples_from_records,
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
        description="Score a linear cue policy on logged interventions."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL logs.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "sync-policy.json",
        help="Policy model to load (default: artifacts/sync-policy.json).",
    )
    parser.add_argument(
        "--label",
        choices=["carryover", "uptake", "blend"],
        default="carryover",
        help="Reward label to score (default: carryover).",
    )
    parser.add_argument("--threshold", type=float, help="Override the model threshold.")
    parser.add_argument("--report", type=Path, help="Write metrics as JSON here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = LinearPolicyModel.load(args.model)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    samples = samples_from_records(_records(args.logs), label=args.label)
    if not samples:
        print("error: no labeled samples in the given logs", file=sys.stderr)
        return 2
    try:
        metrics = evaluate_policy(model, samples, threshold=args.threshold)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "model": str(args.model),
        "label": args.label,
        "log_files": len(args.logs),
        "metrics": metrics,
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
