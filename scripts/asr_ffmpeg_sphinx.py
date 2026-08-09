#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

HMM = "/usr/share/pocketsphinx/model/en-us/en-us"
DICT = "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict"
LM = "/usr/share/pocketsphinx/model/en-us/en-us.lm.bin"


def ffprobe_duration_ms(path: Path) -> int:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return max(0, int(round(float(process.stdout.strip()) * 1000)))


def recognize(path: Path) -> str:
    af = (
        "apad=pad_dur=1,"
        f"asr=hmm={HMM}:dict={DICT}:lm={LM},"
        "ametadata=print:file=-"
    )
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            af,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-1000:])

    merged = process.stdout + "\n" + process.stderr
    matches = re.findall(r"lavfi\.asr\.text=(.*)", merged)
    if not matches:
        return ""
    return " ".join(part.strip() for part in matches if part.strip())


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: asr_ffmpeg_sphinx.py AUDIO_FILE")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"audio file not found: {path}")

    duration_ms = ffprobe_duration_ms(path)
    text = recognize(path)

    print(
        json.dumps(
            {
                "type": "transcript",
                "kind": "FINAL",
                "text": text,
                "start_ms": 0,
                "end_ms": duration_ms,
                "confidence": None,
            },
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "type": "boundary",
                "kind": "TURN_CANDIDATE",
                "at_ms": duration_ms,
                "silence_ms": 1000,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
