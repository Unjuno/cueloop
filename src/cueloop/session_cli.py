from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import ControllerConfig
from .commands import parse_command_json
from .llm import CommandCompletion, LLMAnalyzer
from .session import SessionCondition, SessionRetention, purge_expired_sessions, run_integrated_session, verify_session
from .voice import CommandSpeaker, command_event_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify an atomic CUELOOP reasoning session.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path)
    source.add_argument("--event-command", nargs="+")
    source.add_argument("--event-command-json")
    run.add_argument("--goal", required=True)
    run.add_argument("--condition", choices=[condition.value for condition in SessionCondition], default="CUELOOP")
    run.add_argument("--output-root", type=Path, default=Path("data/sessions"))
    run.add_argument("--session-id")
    run.add_argument("--sources", nargs="*", type=Path)
    run.add_argument("--policy", type=Path)
    run.add_argument("--analyzer-command", nargs="+")
    run.add_argument("--analyzer-command-json")
    run.add_argument("--analyzer-timeout", type=float, default=8.0)
    run.add_argument("--profile", type=Path, default=Path("data/personalization.json"))
    run.add_argument("--tts-command", nargs="+")
    run.add_argument("--tts-command-json")
    run.add_argument("--retention-days", type=int, default=30)
    run.add_argument("--max-decision-latency-ms", type=int, default=1_500)
    run.add_argument("--cooldown-ms", type=int, default=5_000)
    run.add_argument("--event-timeout-seconds", type=float, default=1_900)
    run.add_argument(
        "--consent-acknowledged",
        action="store_true",
        help="Required for live input after informing the participant about recording and data handling.",
    )
    run.add_argument("--no-transcript-retention", action="store_true")
    verify = subcommands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    purge = subcommands.add_parser("purge")
    purge.add_argument("--output-root", type=Path, default=Path("data/sessions"))
    purge.add_argument("--confirm", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        errors = verify_session(args.directory)
        if errors:
            for error in errors:
                print(f"FAIL {error}", file=sys.stderr)
            return 1
        print(f"PASS session={args.directory}")
        return 0
    if args.command == "purge":
        results = purge_expired_sessions(args.output_root, confirm=args.confirm)
        for result in results:
            if result.reason == "expired" or result.reason.startswith("invalid"):
                action = "removed" if result.removed else "would_remove" if result.reason == "expired" else "skipped"
                print(f"{action}={result.session_id} reason={result.reason}")
        if not args.confirm:
            print("dry-run only; pass --confirm to delete expired session directories")
        return 0
    try:
        policy = ControllerConfig.from_json_file(args.policy) if args.policy else None
        retention = SessionRetention(not args.no_transcript_retention, False, args.retention_days)
        event_command = parse_command_json(args.event_command_json, args.event_command, label="event")
        tts_command = parse_command_json(args.tts_command_json, args.tts_command, label="tts")
        analyzer_command = parse_command_json(
            args.analyzer_command_json, args.analyzer_command, label="analyzer"
        )
        speaker = CommandSpeaker(tts_command) if tts_command else None
        analyzer = (
            LLMAnalyzer(CommandCompletion(analyzer_command, args.analyzer_timeout))
            if analyzer_command
            else None
        )
        if event_command:
            lines = command_event_lines(event_command, args.event_timeout_seconds)
            handle = None
        else:
            handle = args.events.open("r", encoding="utf-8")
            lines = handle
        try:
            outcome = run_integrated_session(
                lines,
                args.output_root,
                goal=args.goal,
                source_paths=args.sources,
                policy=policy,
                profile_path=args.profile,
                retention=retention,
                speaker=speaker,
                analyzer=analyzer,
                condition=SessionCondition(args.condition),
                max_decision_latency_ms=args.max_decision_latency_ms,
                cooldown_ms=args.cooldown_ms,
                live_input=bool(event_command),
                consent_acknowledged=args.consent_acknowledged,
                session_id=args.session_id,
            )
        finally:
            if handle is not None:
                handle.close()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"session={outcome.session_id} directory={outcome.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
