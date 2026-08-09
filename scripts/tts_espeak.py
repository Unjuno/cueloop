#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def intervention_text(payload: dict) -> str | None:
    action = payload.get("action")
    if action == "CUE":
        cue = payload.get("cue")
        return cue if isinstance(cue, str) and cue.strip() else None
    if action == "CONTINUE":
        return os.environ.get("CUELOOP_CONTINUE_TEXT", "go on")
    return None


def main() -> int:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("intervention payload must be a JSON object")

    text = intervention_text(payload)
    if not text:
        return 0

    output_dir = Path(os.environ.get("CUELOOP_TTS_DIR", "logs/tts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    output = output_dir / f"{stamp}.wav"

    voice = os.environ.get("CUELOOP_ESPEAK_VOICE", "en")
    speed = os.environ.get("CUELOOP_ESPEAK_SPEED", "175")
    process = subprocess.run(
        ["espeak", "-v", voice, "-s", speed, "-w", str(output), text],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-500:])
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
