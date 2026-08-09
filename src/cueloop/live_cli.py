from __future__ import annotations

import argparse
from pathlib import Path

from .core import Analyzer, HeuristicAnalyzer, InterventionController, SessionContext
from .live import LiveSessionRunner
from .llm import CommandCompletion, LLMAnalyzer
from .offline import OfflineRunnerConfig, OfflineSessionRunner, write_timeline
from .output import CommandInterventionSink, CommandSinkConfig, ConsoleSink
from .prefetch import PrefetchConfig, PrefetchingAnalyzer
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
    parser.add_argument("--prefetch-minimum-characters", type=int, default=24)
    parser.add_argument("--boundary-wait-ms", type=int, default=60)
    parser.add_argument("--minimum-asr-confidence", type=float, default=0.45)
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
    semantic = LLMAnalyzer(CommandCompletion(args.analyzer_command, timeout_seconds=args.analyzer_timeout))
    return PrefetchingAnalyzer(
        semantic,
        PrefetchConfig(
            minimum_characters=args.prefetch_minimum_characters,
            boundary_wait_ms=args.boundary_wait_ms,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_turns < 1 or args.context_characters < 1:
        raise SystemExit("context limits must be positive")
    if args.prefetch_minimum_characters < 1 or args.boundary_wait_ms < 0:
        raise SystemExit("prefetch limits are invalid")
    if not 0.0 <= args.minimum_asr_confidence <= 1.0:
        raise SystemExit("--minimum-asr-confidence must be in [0, 1]")

    analyzer = build_analyzer(args)
    context = SessionContext(
        max_turns=args.context_turns,
        max_characters=args.context_characters,
    )
    session = OfflineSessionRunner(
        analyzer=analyzer,
        controller=InterventionController(),
        context=context,
        config=OfflineRunnerConfig(minimum_asr_confidence=args.minimum_asr_confidence),
    )
    if args.output_command:
        sink = CommandInterventionSink(
            CommandSinkConfig(tuple(args.output_command), timeout_seconds=args.output_timeout)
        )
    else:
        sink = ConsoleSink()

    try:
        runner = LiveSessionRunner(session, sink)
        with StreamingCommandEventSource(args.source_command) as source:
            timeline = runner.run(source)
    finally:
        close = getattr(analyzer, "close", None)
        if callable(close):
            close()

    write_timeline(args.timeline, timeline)
    print(f"wrote {len(timeline)} decisions to {args.timeline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
