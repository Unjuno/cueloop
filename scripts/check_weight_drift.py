#!/usr/bin/env python3
"""Detect drift between the installed operation weights and fresh human data.

The adoption weights are a moving target: early samples over-rate rare
operations (v1 boundary 1.455 -> v2 0.750 as data grew). This tool re-derives
weights from all clean CUE logs and compares them to the installed artifact,
so the tuning loop can decide whether to refresh the gate.

Exit codes: 0 = in sync (max delta <= threshold), 1 = drift detected,
2 = error. Writes the fresh weights to --fresh for convenience.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed", type=Path, default=ROOT / "artifacts" / "operation-weights.json"
    )
    parser.add_argument(
        "--fresh",
        type=Path,
        default=ROOT / "artifacts" / "operation-weights-fresh.json",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    logs = sorted(
        glob.glob(str(ROOT / "logs" / "self-*accuracy*paired-*" / "cue.jsonl"))
    )
    if not logs:
        print("error: no clean CUE logs found", file=sys.stderr)
        return 2
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tune_operation_weights.py"),
            *logs,
            "--exclude-gated",
            "--output",
            str(args.fresh),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return 2
    installed = (
        json.loads(args.installed.read_text()) if args.installed.exists() else {}
    )
    fresh = json.loads(args.fresh.read_text())
    deltas = {
        op: fresh.get(op, 0.0) - installed.get(op, 0.0)
        for op in set(fresh) | set(installed)
    }
    worst_op = max(deltas, key=lambda op: abs(deltas[op]))
    worst = abs(deltas[worst_op])
    print(f"installed={args.installed}  fresh={args.fresh}")
    print(
        f"max weight drift = {worst:.3f} on '{worst_op}' (threshold {args.threshold})"
    )
    for op in sorted(deltas, key=lambda o: abs(deltas[o]), reverse=True)[:5]:
        print(
            f"  {op:12s} {installed.get(op, 0):.3f} -> {fresh.get(op, 0):.3f} ({deltas[op]:+.3f})"
        )
    if worst > args.threshold:
        print("VERDICT: DRIFT DETECTED -> refresh installed weights and re-run the A/B")
        return 1
    print("VERDICT: weights in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
