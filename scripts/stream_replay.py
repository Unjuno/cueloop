#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def event_time_ms(payload: dict) -> int:
    if payload.get("type") == "transcript":
        return int(payload.get("end_ms", 0))
    return int(payload.get("at_ms", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay CUELOOP JSONL events at recorded timing.")
    parser.add_argument("events", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    rows: list[dict] = []
    with args.events.open("r", encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise SystemExit("event rows must be JSON objects")
            rows.append(payload)

    previous_ms = 0
    for payload in rows:
        current_ms = event_time_ms(payload)
        delay_ms = max(0, current_ms - previous_ms)
        if delay_ms:
            time.sleep(delay_ms / 1000 / args.speed)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        previous_ms = current_ms
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
