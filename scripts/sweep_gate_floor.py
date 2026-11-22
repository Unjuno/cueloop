#!/usr/bin/env python3
"""Offline sweep of --min-operation-weight using real human logs.

For each gated session, replay the recorded intervention sequence through the
same weighted-fatigue chooser but with a different weight floor. For each
delivered cue we know from the actual outcome whether the *spoken* operation
was adopted; if a floor change would have swapped in a different operation,
we approximate its adoption probability from that operation's historical
adoption rate in the clean human logs. This is an offline estimate (labeled
as such) that lets tuning continue without a live speaker.

Output: per-floor expected adoption rate, expected delivered count, and the
share of low-value (dependency) operations.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load() -> tuple[dict[str, int], dict[str, int], list[list[str]]]:
    delivered: Counter[str] = Counter()
    adopted: Counter[str] = Counter()
    sessions: list[list[str]] = []
    for path in sorted(
        glob.glob(str(ROOT / "logs" / "self-*accuracy*paired-*" / "cue.jsonl"))
    ):
        ops: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "intervention" and isinstance(
                rec.get("operation"), str
            ):
                op = rec["operation"]
                ops.append(op)
                delivered[op] += 1
            elif (
                rec.get("kind") == "outcome"
                and rec.get("target_operation_seen") is True
            ):
                op = rec.get("operation")
                if isinstance(op, str):
                    adopted[op] += 1
        if ops:
            sessions.append(ops)
    return dict(delivered), dict(adopted), sessions


def choose(
    eligible: list[str],
    counts: Counter[str],
    weights: dict[str, float],
    penalty: float = 2.0,
) -> str:
    if len(eligible) <= 1:
        return eligible[0] if eligible else ""
    return max(
        eligible,
        key=lambda op: (
            (len(eligible) - eligible.index(op) - penalty * counts[op])
            * weights.get(op, 1.0)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "artifacts" / "operation-weights.json"
    )
    parser.add_argument("--floors", default="0.0,0.3,0.5,0.7,1.0")
    args = parser.parse_args()

    delivered, adopted, sessions = load()
    weights = json.loads(args.weights.read_text())
    rates = {
        op: adopted.get(op, 0) / delivered[op] if delivered.get(op, 0) else 0.0
        for op in weights
    }
    print(
        f"human sessions replayed: {len(sessions)}, total delivered: {sum(delivered.values())}"
    )
    print("floor | expected adoption | delivered | dependency share")
    print("--- | ---: | ---: | ---:")
    for floor in [float(x) for x in args.floors.split(",")]:
        eligible = [op for op, w in weights.items() if w >= floor]
        if not eligible:
            print(f"{floor:.2f} | (no eligible ops) | 0 | -")
            continue
        total_adopt = 0.0
        total_delivered = 0
        dep = 0
        for session in sessions:
            counts: Counter[str] = Counter()
            for _ in session:
                op = choose(eligible, counts, weights)
                counts[op] += 1
                total_delivered += 1
                total_adopt += rates.get(op, 0.0)
                if op == "dependency":
                    dep += 1
        print(
            f"{floor:.2f} | {total_adopt / total_delivered:.3f} | {total_delivered} | "
            f"{dep / total_delivered:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
