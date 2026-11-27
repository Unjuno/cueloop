#!/usr/bin/env python3
"""Compare real-mic (BlackHole) CUE/SILENCE pairs by cue assignment.

Reports paired uptake/carryover with bootstrap CIs for each assignment
(alternate vs canonical) from logs/agent-live-realbh*paired-* directories.
"""

from __future__ import annotations

import glob
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze(pattern: str, label: str) -> None:
    dirs = sorted(glob.glob(str(ROOT / pattern)))
    if not dirs:
        print(f"{label}: no pairs found ({pattern})")
        return
    with tempfile.TemporaryDirectory() as td:
        for d in dirs:
            for fname in ("cue.json", "silence.json"):
                src = Path(d) / fname
                if src.exists():
                    (Path(td) / f"{Path(d).name}-{fname}").write_text(src.read_text())
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_experiment.py"),
                td,
                "--seed",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"=== {label} (n={len(dirs)}) ===")
        for line in result.stdout.splitlines():
            if "uptake_rate" in line or "carryover_rate" in line:
                print("  " + line.strip())


def main() -> int:
    analyze("logs/agent-live-realbh-alt*paired-*", "ALTERNATE (real mic)")
    analyze("logs/agent-live-realbh-long*paired-*", "ALTERNATE-long (real mic)")
    analyze("logs/agent-live-realbh-canon*paired-*", "CANONICAL (real mic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
