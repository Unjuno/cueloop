#!/usr/bin/env python
"""Aggregate per-op-token fire spectra across live paired sessions.

Splits each operation into "speaker-natural" (also fired on the SILENT arm
with no cues) and "cue-divergent" (only ever appeared under CUE delivery).
This does NOT prove causation — it just makes the natural-vs-divergent
vocabulary split explicit so readers don't confuse the two.
"""

import glob
import json
import sys
from collections import defaultdict

OPS = (
    "ground",
    "boundary",
    "dependency",
    "comparison",
    "invert",
    "concretize",
    "mechanism",
    "conclusion",
)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main(roots):
    cue = defaultdict(lambda: [0, 0, 0])  # delivered, uptake, carryover
    sil = defaultdict(
        lambda: [0, 0, 0]
    )  # (no cues possible): outcomes, uptake, carryover
    for arg in roots:
        for root in glob.glob(arg):
            for cue_path in glob.glob(f"{root}/cue.json"):
                m = _load_json(cue_path)
                for op, agg in m.get("per_operation", {}).items():
                    cue[op][0] += agg.get("delivered", 0)
                    cue[op][1] += agg.get("uptake", 0)
                    cue[op][2] += agg.get("carryover", 0)
            for sil_path in glob.glob(f"{root}/silence.json"):
                m = _load_json(sil_path)
                for op, agg in m.get("per_operation", {}).items():
                    sil[op][0] += agg.get("delivered", 0)
                    sil[op][1] += agg.get("uptake", 0)
                    sil[op][2] += agg.get("carryover", 0)
    natural_ops = sorted(op for op in OPS if sil[op][1] > 0)
    divergent_ops = sorted(op for op in OPS if sil[op][1] == 0 and cue[op][1] > 0)

    print("# Live op-token fire spectrum (design-choice pairs)\n")
    print(f"- paired dirs scanned: {sum(len(glob.glob(arg)) for arg in roots)}")
    print(
        f"- speaker-natural ops (fire on SILENT arm too): {', '.join(natural_ops) or '(none)'}"
    )
    print(
        f"- cue-divergent ops (only under CUE): {', '.join(divergent_ops) or '(none)'}\n"
    )
    print("| op | CUE del | CUE upd | CUE carry | SILENT upd | class |")
    print("|---|---|---|---|---|---|")
    cue_d, cue_u, sil_u = 0, 0, 0
    for op in OPS:
        cls = (
            "natural"
            if sil[op][1] > 0
            else ("divergent" if cue[op][1] > 0 else "silent")
        )
        print(
            f"| {op} | {cue[op][0]} | {cue[op][1]} | {cue[op][2]} | {sil[op][1]} | {cls} |"
        )
        cue_d += cue[op][0]
        cue_u += cue[op][1]
        sil_u += sil[op][1]
    print(f"\nCUE: delivered={cue_d} uptake={cue_u} — SILENT: uptake={sil_u}")
    natural_n = sum(cue[op][1] for op in natural_ops)
    divergent_n = sum(cue[op][1] for op in divergent_ops)
    print(
        f"CUE uptake split: speaker-natural op-tokens={natural_n} | cue-divergent={divergent_n}"
    )


if __name__ == "__main__":
    roots = sys.argv[1:] or ["logs/self-design-choice-paired-*"]
    main(roots)
