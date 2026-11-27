#!/usr/bin/env python3
"""Convert an LLM-generated Japanese monologue into sync events for the agent A/B.

The mic loopback path (run_agent_live_pair.sh) cannot recognize speech on this
Mac: VAD fires but SFSpeechRecognizer returns no transcript. This script feeds
the SAME kind of text the speaker would have said directly as transcript events,
so the autonomous agent path can be A/B tested without a microphone.

Usage: agent_text_events.py TEXT_FILE [--cps 6.0]
Prints JSONL: speech_started, transcript_partial (cumulative, like the real
adapter), tick, transcript_final, speech_ended, session_ended.
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def split_clauses(text: str) -> list[str]:
    # Keep punctuation attached to the clause, like real ASR partials.
    parts = re.split(r"(?<=[。！？])", text.strip())
    return [p for p in parts if p.strip()]


def emit(text: str, cps: float) -> None:
    clauses = split_clauses(text)
    now = 0.0

    def out(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush() -> None:
        sys.stdout.flush()

    out({"type": "speech_started", "at_ms": 0})
    cumulative = ""
    for clause in clauses:
        cumulative += clause
        now += max(120, len(clause) / max(cps, 1.0) * 1000)
        out({"type": "transcript_partial", "at_ms": round(now), "text": cumulative})
        flush()
        # one short pause tick after each clause
        now += 250
        out({"type": "tick", "at_ms": round(now)})
    now += 500
    out({"type": "transcript_final", "at_ms": round(now), "text": cumulative})
    out({"type": "speech_ended", "at_ms": round(now) + 1})
    out({"type": "session_ended", "at_ms": round(now) + 2})
    flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_file")
    parser.add_argument("--cps", type=float, default=6.0)
    args = parser.parse_args()
    with open(args.text_file, encoding="utf-8") as handle:
        emit(handle.read(), args.cps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
