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


def duration_seconds(path: Path) -> float:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(process.stdout.strip())


def silence_intervals(path: Path) -> list[tuple[float, float]]:
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
            "-af", "silencedetect=noise=-35dB:d=0.45",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-1000:])

    starts = [float(v) for v in re.findall(r"silence_start: ([0-9.]+)", process.stderr)]
    ends = [float(v) for v in re.findall(r"silence_end: ([0-9.]+)", process.stderr)]
    intervals: list[tuple[float, float]] = []
    for start, end in zip(starts, ends):
        if end >= start:
            intervals.append((start, end))
    return intervals


def speech_intervals(path: Path) -> list[tuple[float, float, float]]:
    duration = duration_seconds(path)
    silences = silence_intervals(path)
    result: list[tuple[float, float, float]] = []
    cursor = 0.0

    for silence_start, silence_end in silences:
        if silence_start - cursor >= 0.20:
            result.append((cursor, silence_start, silence_end - silence_start))
        cursor = max(cursor, silence_end)

    if duration - cursor >= 0.20:
        result.append((cursor, duration, 0.0))
    return result


def recognize_segment(path: Path, start: float, end: float) -> str:
    length = max(0.0, end - start)
    if length <= 0:
        return ""

    af = (
        "apad=pad_dur=0.5,"
        f"asr=hmm={HMM}:dict={DICT}:lm={LM},"
        "ametadata=print:file=-"
    )
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "info",
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", str(path), "-ar", "16000", "-ac", "1",
            "-af", af, "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-1000:])

    merged = process.stdout + "\n" + process.stderr
    matches = re.findall(r"lavfi\.asr\.text=(.*)", merged)
    return " ".join(part.strip() for part in matches if part.strip())


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: asr_ffmpeg_sphinx_segmented.py AUDIO_FILE")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"audio file not found: {path}")

    for start, end, trailing_silence in speech_intervals(path):
        start_ms = int(round(start * 1000))
        end_ms = int(round(end * 1000))
        text = recognize_segment(path, start, end)
        print(json.dumps({
            "type": "transcript",
            "kind": "FINAL",
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": None,
        }, ensure_ascii=False))
        print(json.dumps({
            "type": "boundary",
            "kind": "TURN_CANDIDATE",
            "at_ms": end_ms,
            "silence_ms": int(round(trailing_silence * 1000)),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
