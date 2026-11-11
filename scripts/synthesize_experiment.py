#!/usr/bin/env python3
"""Synthesize paired CUE/SILENCE reports with a KNOWN causal effect.

CUELOOP cannot replay what a human would say next, so a static fixture yields
a guaranteed-zero CUE - SILENCE delta. This generator instead defines a
generative speaker model with a programmed cue effect, and emits report JSON
in exactly the schema `analyze_experiment.py` consumes. The paired sessions
share one skeleton (same missing-operation sets, same requested/delivered
draws, same uniform random numbers per boundary via common random numbers);
only the intervention effect differs between CUE and SILENCE. Any detected
difference is therefore attributable to the programmed effect.

This is calibration of the analysis pipeline, NOT evidence about humans.
The cue effect is a modeling choice, not a measurement.

Values of ``--uptake-effect 0`` produce the null arm used to check that the
analyzer reports a CI covering zero on data with no effect.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.vocabulary import CUE_BY_OPERATION, OPERATIONS

BACKENDS: dict[str, str] = {
    "rule": "prediction-gap(assigner=canonical)",
    "linear": "linear-policy(synthetic)",
    "llm": "local-llm(synthetic)",
}


def _requested(
    backend: str, missing: list[str], usage: dict[str, int], rng: random.Random
) -> str:
    if not missing:
        raise ValueError("missing must be non-empty")
    if backend == "rule":
        return missing[0]
    if backend == "linear":
        return min(missing, key=lambda op: (usage.get(op, 0), OPERATIONS.index(op)))
    # llm: seeded exploratory choice, different from rule's positional rule
    return rng.choice(missing)


def _session(
    *,
    backend: str,
    condition: str,
    index: int,
    seed: int,
    base_uptake: float,
    base_carryover: float,
    uptake_effect: float,
    carryover_effect: float,
    clauses: int,
    assignment: str,
) -> dict[str, object]:
    rng = random.Random(f"{seed}|{backend}|{index}|{assignment}")
    used: set[str] = set()
    usage: dict[str, int] = {}
    outcomes: list[dict[str, object]] = []
    delays: list[float] = []
    growth: list[str] = []
    intervention = 0
    for boundary in range(clauses):
        missing = [op for op in OPERATIONS if op not in used]
        if not missing:
            break
        requested = _requested(backend, missing, usage, rng)
        delivered = (
            rng.choice(missing) if assignment == "random-operation" else requested
        )
        intervention += 1
        cue = CUE_BY_OPERATION[delivered][0]
        # Delay jitter and outcome uniforms come from dedicated per-boundary
        # streams so they stay paired regardless of how the op path diverges.
        jitter = random.Random(
            f"delay|{seed}|{backend}|{index}|{boundary}|{assignment}"
        )
        delays.append(100 + jitter.choice([-10, 0, 10]))
        uniforms = random.Random(f"u|{seed}|{backend}|{index}|{boundary}|{assignment}")
        u_uptake = uniforms.random()
        u_carryover = uniforms.random()
        p_uptake = min(
            1.0, max(0.0, base_uptake + (uptake_effect if condition == "CUE" else 0.0))
        )
        p_carryover = min(
            1.0,
            max(
                0.0, base_carryover + (carryover_effect if condition == "CUE" else 0.0)
            ),
        )
        seen = u_uptake < p_uptake
        carry = u_carryover < p_carryover
        novel = delivered not in used
        outcomes.append(
            {
                "operation": delivered,
                "requested_operation": requested,
                "novel_operation": novel,
                "target_operation_seen": seen,
                "carryover_seen": carry,
                "cue": cue,
            }
        )
        usage[delivered] = usage.get(delivered, 0) + 1
        if seen or carry:
            if novel:
                growth.append(delivered)
            used.add(delivered)
    median_delay = sorted(delays)[len(delays) // 2] if delays else None
    paired_key = f"{backend}-seed{seed}-synthetic-{index:03d}"
    return {
        "predictor": "synthetic(3)",
        "decision_model": (
            f"{BACKENDS[backend]}"
            + (
                f"+random-operation(seed={seed})"
                if assignment == "random-operation"
                else ""
            )
        ),
        "condition": condition,
        "assignment": (
            f"random-operation(seed={seed};wording=canonical)"
            if assignment == "random-operation"
            else "canonical"
        ),
        "seed": seed,
        "session_id": f"{condition.lower()}-{backend}-{index:03d}",
        "pair_key": paired_key,
        "silent": condition == "SILENCE",
        "delivered": intervention if condition == "CUE" else 0,
        "held": intervention if condition == "SILENCE" else 0,
        "interventions": intervention,
        "boundary_to_delivery_ms": delays,
        "median_boundary_to_delivery_ms": median_delay,
        "repertoire_growth": growth,
        "outcomes": outcomes,
        "synthetic": True,
        "note": "programmed cue effect; not human data",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired CUE/SILENCE synthetic reports with a known, "
            "programmed cue effect. Synthetic calibration data, not human evidence."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--clauses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backends", default="rule,linear,llm")
    parser.add_argument(
        "--assignment",
        choices=["random-operation", "requested"],
        default="random-operation",
    )
    parser.add_argument("--base-uptake", type=float, default=0.15)
    parser.add_argument("--base-carryover", type=float, default=0.25)
    parser.add_argument("--uptake-effect", type=float, default=0.30)
    parser.add_argument("--carryover-effect", type=float, default=0.30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sessions < 1 or args.clauses < 1:
        print("error: --sessions and --clauses must be positive", file=sys.stderr)
        return 2
    for name in ("base_uptake", "base_carryover", "uptake_effect", "carryover_effect"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            print(
                f"error: --{name.replace('_', '-')} must be within [0, 1]",
                file=sys.stderr,
            )
            return 2
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = [item for item in backends if item not in BACKENDS]
    if unknown:
        print(f"error: unknown backends: {', '.join(unknown)}", file=sys.stderr)
        return 2
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        print(f"error: output directory is not empty: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for backend in backends:
        for index in range(args.sessions):
            for condition in ("CUE", "SILENCE"):
                report = _session(
                    backend=backend,
                    condition=condition,
                    index=index,
                    seed=args.seed,
                    base_uptake=args.base_uptake,
                    base_carryover=args.base_carryover,
                    uptake_effect=args.uptake_effect,
                    carryover_effect=args.carryover_effect,
                    clauses=args.clauses,
                    assignment=args.assignment,
                )
                path = output / f"{backend}-{condition.lower()}-{index:03d}.json"
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                count += 1
    manifest = {
        "synthetic": True,
        "note": "programmed cue effect; calibration of the analysis pipeline, not human evidence",
        "parameters": {
            "sessions": args.sessions,
            "clauses": args.clauses,
            "seed": args.seed,
            "backends": backends,
            "assignment": args.assignment,
            "base_uptake": args.base_uptake,
            "base_carryover": args.base_carryover,
            "uptake_effect": args.uptake_effect,
            "carryover_effect": args.carryover_effect,
        },
        "reports": count,
    }
    (output / "synthetic-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"reports": count, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
