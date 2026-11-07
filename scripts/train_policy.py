#!/usr/bin/env python3
"""Train the fast linear intervention policy from JSONL session logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.policy_model import samples_from_records, train_linear_policy


def _records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fast linear cue policy from logged sessions."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Session JSONL logs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "sync-policy.json",
        help="Output model path (default: artifacts/sync-policy.json).",
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
        "--min-samples",
        type=int,
        default=20,
        help="Refuse to write a model below this many samples (default: 20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = samples_from_records(_records(args.logs), label=args.label)
    if len(samples) < args.min_samples:
        print(
            f"error: {len(samples)} usable samples, need {args.min_samples}",
            file=sys.stderr,
        )
        return 2
    try:
        model = train_linear_policy(
            samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model.metadata["label"] = args.label
    model.metadata["logs"] = [str(path) for path in args.logs]
    model.save(args.output)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "samples": len(samples),
                "positives": sum(1 for sample in samples if sample.reward >= 0.5),
                "threshold": model.threshold,
                "label": args.label,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
