#!/usr/bin/env python3
"""Internal audit of the tuning state using only existing logs (no mic needed).

Checks:
  1. installed weights vs measured per-op adoption correlation (should be >0.7)
  2. adjacent exact / same-family repeat rates (fatigue health)
  3. whether any above-floor op has critically low adoption (<0.10)
  4. floor consistency: which ops each floor permits

This is the 'adjust internally' entry point: run it after any new data or
weight change to see whether further tuning is actually warranted.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAMILY = {
    "ground": "evidence",
    "concretize": "evidence",
    "mechanism": "explain",
    "invert": "contrast",
    "conclusion": "conclude",
    "comparison": "compare",
    "boundary": "scope",
    "dependency": "dep",
}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "artifacts" / "operation-weights.json"
    )
    parser.add_argument("--floor", type=float, default=0.5)
    args = parser.parse_args()

    delivered, adopted, sessions = load()
    weights = json.loads(args.weights.read_text())
    ops = [o for o in weights if delivered.get(o, 0) >= 10]
    rates = [adopted[o] / delivered[o] for o in ops]
    ws = [weights[o] for o in ops]
    mr, mw = statistics.mean(rates), statistics.mean(ws)
    cov = sum((r - mr) * (w - mw) for r, w in zip(rates, ws))
    den = (sum((r - mr) ** 2 for r in rates) * sum((w - mw) ** 2 for w in ws)) ** 0.5
    corr = cov / den if den else 0.0

    exact = fam = total = 0
    for ops_seq in sessions:
        for a, b in itertools.pairwise(ops_seq):
            total += 1
            if a == b:
                exact += 1
            if FAMILY.get(a) == FAMILY.get(b):
                fam += 1

    eligible = sorted([o for o, w in weights.items() if w >= args.floor])
    low_adoption = [
        o for o in eligible if delivered.get(o, 0) and adopted[o] / delivered[o] < 0.10
    ]

    print("=== internal tuning audit ===")
    print(f"sessions={len(sessions)} deliveries={sum(delivered.values())}")
    print(
        f"[1] weight/adoption correlation r={corr:.3f} ({'OK' if corr > 0.7 else 'NEEDS WORK'})"
    )
    print(f"[2] adjacent exact repeats={exact}/{total} ({exact / total:.1%})")
    print(f"    adjacent same-family repeats={fam}/{total} ({fam / total:.1%})")
    print(f"[3] above-floor ops with adoption<0.10: {low_adoption or 'none'}")
    print(f"[4] floor={args.floor} permits {len(eligible)} ops: {eligible}")
    if corr > 0.7 and not low_adoption:
        print("VERDICT: tuning internally consistent; no forced change warranted")
    else:
        print("VERDICT: tuning needs attention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
