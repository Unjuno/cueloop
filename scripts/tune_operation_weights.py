#!/usr/bin/env python3
"""Derive per-operation insertion weights from human DELIVERED outcomes.

Low-adoption operations waste insertions (the 79%-not-adopted problem). This
tool reads CUE logs, computes each operation's uptake rate, and emits a
weight file usable via `--operation-weights` / `--min-operation-weight`.

Weights are adoption rates scaled to a 0..2 range. Operations with no data
default to 1.0. A floor (default 0.5) suppresses the worst offenders.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def derive(
    logs: list[Path], *, scale: float = 4.0
) -> tuple[dict[str, float], dict[str, dict]]:
    delivered: dict[str, int] = defaultdict(int)
    adopted: dict[str, int] = defaultdict(int)
    for path in logs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("kind") == "intervention":
                operation = record.get("operation")
                if isinstance(operation, str):
                    delivered[operation] += 1
            elif (
                record.get("kind") == "outcome"
                and record.get("target_operation_seen") is True
            ):
                operation = record.get("operation")
                if isinstance(operation, str):
                    adopted[operation] += 1
    weights: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for operation, n in delivered.items():
        rate = adopted[operation] / n if n else 0.0
        weights[operation] = round(min(2.0, rate * scale), 3)
        detail[operation] = {
            "delivered": n,
            "adopted": adopted[operation],
            "adoption_rate": round(rate, 3),
            "weight": weights[operation],
        }
    return weights, detail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="CUE session JSONL logs.")
    parser.add_argument("--output", type=Path, required=True, help="Weights JSON path.")
    parser.add_argument(
        "--report", type=Path, help="Detailed per-operation report JSON."
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument(
        "--exclude-gated",
        action="store_true",
        help=(
            "Only use baseline (non-gated) sessions. Weights learned from "
            "gated sessions absorb the gate's own adoption lift and "
            "self-amplify; gates should be tuned on natural adoption."
        ),
    )
    return parser


def _is_gated(path: Path) -> bool:
    name = path.name.lower()
    return any(tag in name for tag in ("gated", "v2gate", "v3gate", "floor0"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logs = [p for p in args.logs if not (args.exclude_gated and _is_gated(p))]
    weights, detail = derive(logs, scale=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    print(f"derived {len(weights)} operation weights (human adoption data)")
    for operation in sorted(detail, key=lambda op: detail[op]["adoption_rate"]):
        info = detail[operation]
        print(
            f"  {operation:12s} delivered={info['delivered']:3d} "
            f"adopted={info['adopted']:3d} rate={info['adoption_rate']:.3f} "
            f"weight={weights[operation]:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
