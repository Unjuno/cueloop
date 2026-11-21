#!/usr/bin/env python3
"""A/B comparison of the anti-over-insertion gate from the feedback log.

Baseline is the pre-gate sessions in artifacts/feedback-loop.jsonl
(gated=false or absent); tuned is gated=true. Reports adoption rate
(uptake/delivered) and the share of low-value deliveries, plus a seeded
bootstrap of the paired per-pair adoption differences where possible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def bootstrap(
    values: list[float], *, resamples: int, seed: int
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = int(0.025 * len(means))
    hi = min(len(means) - 1, int(0.975 * len(means)))
    return means[lo], means[hi]


def summarize(rows: list[dict]) -> dict:
    delivered = sum(r.get("cue_delivered", 0) for r in rows)
    uptake = sum(r.get("cue_uptake", 0) for r in rows)
    return {
        "pairs": len(rows),
        "delivered": delivered,
        "uptake": uptake,
        "adoption_rate": uptake / delivered if delivered else 0.0,
        "per_pair_delivered": delivered / len(rows) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "feedback", type=Path, nargs="?", default=Path("artifacts/feedback-loop.jsonl")
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.feedback.read_text().splitlines() if l.strip()]
    baseline = [r for r in rows if not r.get("gated") and r.get("clean")]
    tuned = [r for r in rows if r.get("gated") and r.get("clean")]
    print("=== anti-over-insertion gate A/B ===")
    for label, group in (("baseline (no gate)", baseline), ("tuned (gate)", tuned)):
        s = summarize(group)
        print(
            f"{label:20s} pairs={s['pairs']:3d} delivered={s['delivered']:4d} "
            f"uptake={s['uptake']:3d} adoption={s['adoption_rate']:.3f} "
            f"delivered/pair={s['per_pair_delivered']:.2f}"
        )
    if not tuned:
        print("tuned group empty -> run a batch with GATE_WEIGHTS set first")
        return 0
    b, t = summarize(baseline), summarize(tuned)
    delta = t["adoption_rate"] - b["adoption_rate"]
    print(f"adoption delta (tuned - baseline) = {delta:+.3f}")
    if baseline and tuned:
        # Cluster-bootstrap: resample pairs within each arm.
        br = random.Random(args.seed)
        diffs = []
        for _ in range(args.resamples):
            bs = [br.choice(baseline) for _ in baseline]
            ts = [br.choice(tuned) for _ in tuned]
            diffs.append(
                summarize(ts)["adoption_rate"] - summarize(bs)["adoption_rate"]
            )
        diffs.sort()
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[int(0.975 * len(diffs))]
        verdict = (
            "clear improvement"
            if lo > 0
            else ("clear regression" if hi < 0 else "inconclusive")
        )
        print(f"95% CI of adoption delta = [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
