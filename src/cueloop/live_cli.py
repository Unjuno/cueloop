from __future__ import annotations

import argparse
from pathlib import Path

from .core import Analyzer, HeuristicAnalyzer, InterventionController, SessionContext
from .live import LiveSessionRunner
from .llm import CommandCompletion, LLMAnalyzer
from .offline import OfflineSessionRunner, write_timeline
from .output import CommandInterventionSink, CommandSinkConfig, ConsoleSink
from .stream_command import StreamingCommandEventSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cueloop-live",
        description="Run CUELOOP against a streaming audio-event command.",
    )
    parser.add_argument(
        "--source-command",
        nargs="+",
        required=True,
        help="Long-running command that writes one CUELOOP event JSON object per stdout line.",
    )
    parser.add_argument(
        "--analyzer-command",
        nargs="+",
        help="Optional semantic analyzer command. Receives prompt JSON on stdin and returns strict analyzer JSON on stdout.",
    )
    parser.add_argument("--analyzer-timeout", type=float, default=8.0)
    parser.add_argument(
        "--output-command",
        nargs="+",
        help="Optional command receiving one intervention JSON object on stdin for playback/TTS.",
    )
    parser.add_argument("--output-timeout", type=float, default=5.0)
    parser.add_argument("--timeline", type=Path, default=Path("logs/live-timeline.jsonl"))
    parser.add_argument("--context-turns", type=int, default=4)
    parser.add_argument("--context-characters", type=int, default=2400)
    return parser


def build_analyzer(args: argparse.Namespace) -> Analyzer:
    if not args.analyzer_command:
        return HeuristicAnalyzer()
    return LLMAnalyzer(CommandCompletion(args.analyzer_command, timeout_seconds=args.analyzer_timeout))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_turns < 1 or args.context_characters < 1:
        raise SystemExit("context limits must be positive")

    context = SessionContext(
        max_turns=args.context_turns,
        max_characters=args.context_characters,
    )
    session = OfflineSessionRunner(
        analyzer=build_analyzer(args),
        controller=InterventionController(),
        context=context,
    )
    if args.output_command:
        sink = CommandInterventionSink(
            CommandSinkConfig(tuple(args.output_command), timeout_seconds=args.output_timeout)
        )
    else:
        sink = ConsoleSink()

    runner = LiveSessionRunner(session, sink)
    with StreamingCommandEventSource(args.source_command) as source:
        timeline = runner.run(source)

    write_timeline(args.timeline, timeline)
    print(f"wrote {len(timeline)} decisions to {args.timeline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
