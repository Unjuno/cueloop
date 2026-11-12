#!/usr/bin/env python3
"""Power analysis for the paired CUE/SILENCE design on synthetic sessions.

For each (effect, number of paired sessions), re-run the event-level
simulator across many seeds and record how often the analyzer's bootstrap
CI excludes zero. This yields the sample size at which a given programmed
effect reliably surfaces — the answer a researcher needs before running the
human study, not after.

Usage:
  scripts/analyze_power.py [--seeds 16] [--sessions-grid 8,16,24,48] \
      [--effects 0,0.05,0.1,0.15,0.2,0.3] [--clauses 8] [--output artifacts/power.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_experiment import analyze, session_metrics
from simulate_sessions import run_session


def _cell(
    *,
    seeds: list[int],
    sessions: int,
    clauses: int,
    uptake_effect: float,
    carryover_effect: float,
) -> dict[str, object]:
    detected = 0
    deltas: list[float] = []
    pairs_insufficient = 0
    for seed in seeds:
        runs = []
        for condition in ("CUE", "SILENCE"):
            for index in range(sessions):
                report = run_session(
                    index=index,
                    seed=seed,
                    condition=condition,
                    uptake_effect=uptake_effect,
                    base_uptake=0.15,
                    carryover_effect=carryover_effect,
                    base_carryover=0.10,
                    clauses=clauses,
                    predictor_name="cycle",
                    cooldown_ms=250,
                    outcome_window_ms=4_000,
                )
                runs.append(
                    (f"{condition.lower()}-{index:03d}.json", session_metrics(report))
                )
        result = analyze(runs, resamples=1_000, seed=0, confidence=0.95)
        comparison = result["comparisons"].get("rule") or {}
        metrics = comparison.get("metrics") or {}
        uptake = metrics.get("uptake_rate") or {}
        point = uptake.get("point")
        low = uptake.get("ci_low")
        if point is not None:
            deltas.append(float(point))
        if low == "insufficient" or low is None:
            pairs_insufficient += 1
        elif float(low) > 0:
            detected += 1
    n = len(seeds)
    mean_delta = (sum(deltas) / len(deltas)) if deltas else None
    return {
        "sessions_per_pair": sessions,
        "detected": detected,
        "insufficient_ci": pairs_insufficient,
        "seeds": n,
        "detection_rate": detected / n if n else None,
        "mean_uptake_delta": mean_delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seeds", type=int, default=16, help="Random seeds per cell (default: 16)."
    )
    parser.add_argument(
        "--sessions-grid",
        type=str,
        default="8,16,24,48",
        help="Comma list of pairs per cell.",
    )
    parser.add_argument(
        "--effects",
        type=str,
        default="0,0.1,0.2,0.3",
        help="Comma list of uptake effects.",
    )
    parser.add_argument("--clauses", type=int, default=8)
    parser.add_argument(
        "--carryover-effect",
        type=float,
        default=None,
        help="Carryover boost added in the CUE arm; defaults to --effects value per cell.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON results here."
    )
    args = parser.parse_args(argv)
    grid = [int(x) for x in args.sessions_grid.split(",") if x.strip()]
    effects = [float(x) for x in args.effects.split(",") if x.strip()]
    seeds = list(range(args.seeds))
    table: list[dict[str, object]] = []
    for effect in effects:
        carry = args.carryover_effect if args.carryover_effect is not None else effect
        for n in grid:
            cell = _cell(
                seeds=seeds,
                sessions=n,
                clauses=args.clauses,
                uptake_effect=effect,
                carryover_effect=carry,
            )
            cell["uptake_effect"] = effect
            table.append(cell)
            rate = (cell["detection_rate"] or 0.0) * 100
            mean_d = cell["mean_uptake_delta"]
            print(
                f"effect={effect:.2f} pairs={n:>3}: {cell['detected']:>2}/{cell['seeds']} detected ({rate:.0f}%), mean delta {mean_d:+.3f}",
                flush=True,
            )
    payload = {
        "synthetic": True,
        "note": "power analysis on simulated Japanese speakers; informs design of the human study, not evidence about humans",
        "clauses": args.clauses,
        "carryover_effect": args.carryover_effect,
        "cells": table,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
