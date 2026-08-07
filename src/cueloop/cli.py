from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import Action, Analyzer, HeuristicAnalyzer, InterventionController, append_jsonl
from .llm import CommandCompletion, LLMAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cueloop",
        description="Human-first reasoning CLI. Returns SILENCE, CONTINUE, or one minimal CUE.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/session.jsonl"),
        help="JSONL decision log path (default: logs/session.jsonl)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable session logging.",
    )
    parser.add_argument(
        "--analyzer-command",
        nargs="+",
        metavar=("CMD", "ARG"),
        help=(
            "Use an external semantic analyzer command instead of heuristics. "
            "CUELOOP sends prompt JSON to stdin and expects analyzer-result JSON on stdout."
        ),
    )
    parser.add_argument(
        "--analyzer-timeout",
        type=float,
        default=8.0,
        help="External analyzer timeout in seconds (default: 8.0).",
    )
    return parser


def build_analyzer(args: argparse.Namespace) -> Analyzer:
    if not args.analyzer_command:
        return HeuristicAnalyzer()
    completion = CommandCompletion(args.analyzer_command, timeout_seconds=args.analyzer_timeout)
    return LLMAnalyzer(completion)


def render(action: Action, cue: str | None) -> str:
    if action is Action.SILENCE:
        return "[SILENCE]"
    if action is Action.CONTINUE:
        return "[CONTINUE] うん。"
    return f"[CUE] {cue}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyzer = build_analyzer(args)
    controller = InterventionController()

    mode = "external semantic analyzer" if args.analyzer_command else "heuristic baseline"
    print(f"CUELOOP M1 — text prototype ({mode})")
    print("Type a reasoning turn. Use /quit to exit, /help for commands.")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if text == "/quit":
            return 0
        if text == "/help":
            print("/quit  exit\n/help  show commands")
            continue

        analysis = analyzer.analyze(text)
        decision = controller.decide(analysis)
        print(render(decision.action, decision.cue))
        print(
            f"  state={analysis.state.value} confidence={analysis.confidence:.2f} "
            f"reason={decision.reason}"
        )
        if analysis.evidence:
            print(f"  evidence={'; '.join(analysis.evidence)}")

        if not args.no_log:
            try:
                append_jsonl(args.log, text, decision)
            except OSError as exc:
                print(f"warning: failed to write log: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
