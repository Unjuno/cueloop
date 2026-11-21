#!/usr/bin/env python3
"""Re-evaluate H1/H2 from clean paired human sessions.

H1: cueing an announced operation raises its probability in the next clause.
H2: repeated cueing improves a delayed, unassisted reconstruction (NOT
    measured here; requires the blinded delayed-reconstruction rubric).

A pair is 'clean' only if it was captured with the fixed runtime, where the
SILENCE control arm scores zero outcomes by design (only audible cues are
scored). We take all such pairs, run the seeded paired bootstrap, and report
whether the uptake CI excludes zero (the protocol's criterion for H1).
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# The fixed runtime (PTY event capture + SILENCE outcome guard) landed on
# 2026-11-21 23:44 UTC. Any accuracy-* pair with a timestamp at/after that
# moment is clean, including pairs recorded after midnight on 2026-11-22.
FIXED_RUNTIME_CUTOFF = (2026, 11, 21, 23, 44)


def _is_fixed_runtime(name: str) -> bool:
    if "accuracy" not in name:
        return False
    # name ends with ...-paired-YYYYMMDD-HHMMSS
    try:
        stamp = name.rsplit("paired-", 1)[-1]  # YYYYMMDD-HHMMSS
        date_part, time_part = stamp.split("-")
        year, month, day = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8])
        hour, minute = int(time_part[:2]), int(time_part[2:4])
    except (ValueError, IndexError):
        return False
    return (year, month, day, hour, minute) >= FIXED_RUNTIME_CUTOFF


def clean_pairs() -> list[Path]:
    result: list[Path] = []
    for raw in sorted(glob.glob(str(ROOT / "logs" / "self-*paired-*"))):
        path = Path(raw)
        # Only pairs created with the fixed runtime (PTY fix at 23:44 UTC).
        if not _is_fixed_runtime(path.name):
            continue
        try:
            cue = json.loads((path / "cue.json").read_text(encoding="utf-8"))
            sil = json.loads((path / "silence.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cue_out = len(cue.get("outcomes", []))
        sil_out = len(sil.get("outcomes", []))
        if cue_out > 0 and sil_out == 0:
            result.append(path)
    return result


def main() -> int:
    pairs = clean_pairs()
    print(f"clean pairs captured with the fixed runtime: {len(pairs)}")
    if len(pairs) < 2:
        print("need at least 2 clean pairs for a paired interval")
        return 0
    with tempfile.TemporaryDirectory() as td:
        for pair in pairs:
            for fname in ("cue.json", "silence.json"):
                src = pair / fname
                (Path(td) / f"{pair.name}-{fname}").write_text(
                    src.read_text(encoding="utf-8")
                )
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_experiment.py"),
                td,
                "--min-outcomes",
                "1",
                "--seed",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
        print(completed.stdout)
        # Protocol criterion for H1: paired uptake CI excludes zero.
        json_path = Path(td) / "analysis.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_experiment.py"),
                td,
                "--min-outcomes",
                "1",
                "--json",
                str(json_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        uptake = (
            payload.get("comparisons", {})
            .get("rule", {})
            .get("metrics", {})
            .get("uptake_rate", {})
        )
        low, high = uptake.get("ci_low"), uptake.get("ci_high")
        point = uptake.get("point")
        n = uptake.get("n_pairs", 0)
        print(f"H1 uptake delta: point={point} CI=[{low}, {high}] n_pairs={n}")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if low > 0 or high < 0:
                print("H1 VERDICT: CI excludes zero -> consistent with a cue effect")
            else:
                print("H1 VERDICT: CI includes zero -> no detectable effect yet")
        else:
            print("H1 VERDICT: insufficient data for an interval")
        print("H2 VERDICT: not measured (needs blinded delayed-reconstruction task)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
