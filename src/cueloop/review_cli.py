from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import CommandCompletion
from .review import SessionReviewer


def load_turns(path: Path) -> list[str]:
    turns: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"line {line_number}: invalid timeline JSON") from exc
            text = payload.get("text") if isinstance(payload, dict) else None
            if isinstance(text, str) and text.strip():
                turns.append(text.strip())
    return turns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cueloop-review",
        description="Extract claims and structural gaps from a completed CUELOOP session without rewriting it.",
    )
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--review-command", nargs="+", required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("logs/session-review.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    turns = load_turns(args.timeline)
    reviewer = SessionReviewer(CommandCompletion(args.review_command, timeout_seconds=args.timeout))
    review = reviewer.review(turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(review.to_json() + "\n", encoding="utf-8")
    print(review.to_json())
    if review.error:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
