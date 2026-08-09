from __future__ import annotations

import argparse
from pathlib import Path

from .asr_command import CommandASRConfig, CommandASREventSource
from .core import Analyzer, HeuristicAnalyzer, InterventionController, SessionContext
from .llm import CommandCompletion, LLMAnalyzer
from .offline import OfflineSessionRunner, write_timeline
from .replay import JSONLReplaySource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cueloop-offline",
        description="Run CUELOOP against recorded audio events or an audio file via an external ASR command.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path, help="Replay JSONL file containing transcript/boundary events.")
    source.add_argument("--audio", type=Path, help="Audio file to process through --asr-command.")
    parser.add_argument(
        "--asr-command",
        nargs="+",
        help="External command that receives the audio path as its final argument and writes CUELOOP event JSONL to stdout.",
    )
    parser.add_argument("--asr-timeout", type=float, default=120.0)
    parser.add_argument(
        "--analyzer-command",
        nargs="+",
        help="Optional semantic analyzer command. Receives prompt JSON on stdin and returns strict analyzer JSON on stdout.",
    )
    parser.add_argument("--analyzer-timeout", type=float, default=8.0)
    parser.add_argument("--context-turns", type=int, default=4)
    parser.add_argument("--context-characters", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=Path("logs/offline-timeline.jsonl"))
    return parser


def build_analyzer(args: argparse.Namespace) -> Analyzer:
    if not args.analyzer_command:
        return HeuristicAnalyzer()
    return LLMAnalyzer(CommandCompletion(args.analyzer_command, timeout_seconds=args.analyzer_timeout))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_turns < 1 or args.context_characters < 1:
        raise SystemExit("context limits must be positive")

    if args.audio is not None:
        if not args.asr_command:
            raise SystemExit("--audio requires --asr-command")
        source = CommandASREventSource(
            args.audio,
            CommandASRConfig(tuple(args.asr_command), timeout_seconds=args.asr_timeout),
        )
    else:
        if args.asr_command:
            raise SystemExit("--asr-command is only valid with --audio")
        source = JSONLReplaySource.from_path(args.events)

    runner = OfflineSessionRunner(
        analyzer=build_analyzer(args),
        controller=InterventionController(),
        context=SessionContext(
            max_turns=args.context_turns,
            max_characters=args.context_characters,
        ),
    )
    timeline = runner.run_source(source)
    write_timeline(args.output, timeline)

    for item in timeline:
        cue = f" cue={item.decision.cue}" if item.decision.cue else ""
        print(
            f"{item.at_ms:>7}ms  {item.decision.action.value:<8} "
            f"state={item.decision.analysis.state.value:<7} confidence={item.decision.analysis.confidence:.2f}{cue}"
        )
        print(f"           {item.text}")

    print(f"wrote {len(timeline)} decisions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
