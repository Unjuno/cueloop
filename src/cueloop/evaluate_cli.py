from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evaluate import evaluate_timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cueloop-evaluate",
        description="Summarize CUELOOP intervention behavior from a timeline JSONL file.",
    )
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_timeline(args.timeline)

    if args.json:
        print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
        return 0

    print(f"turns: {metrics.turns}")
    print(f"silence: {metrics.silence} ({metrics.silence_rate:.1%})")
    print(f"continue: {metrics.continue_count}")
    print(f"cue: {metrics.cue_count}")
    print(f"intervention_rate: {metrics.intervention_rate:.1%}")
    print(f"interventions_per_minute: {metrics.interventions_per_minute:.2f}")
    print(f"low_asr_suppression: {metrics.low_asr_suppression_count}")
    print(f"time_cooldown_suppression: {metrics.time_cooldown_suppression_count}")
    print(f"turn_cooldown_suppression: {metrics.turn_cooldown_suppression_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
