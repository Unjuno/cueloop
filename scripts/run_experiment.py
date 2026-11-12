#!/usr/bin/env python3
"""Record one policy-comparison session with randomized cue assignment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _entrypoint() -> list[str]:
    python = ROOT / ".venv" / "bin" / "python"
    if python.exists():
        return [str(python), "-m", "cueloop.sync.cli"]
    return [sys.executable, "-m", "cueloop.sync.cli"]


def _say_command(voice: str) -> list[str]:
    return ["say", "-v", voice] if voice else ["say"]


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "session"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a live or replay session for rule/linear/LLM comparison. "
            "Cue assignment is randomized by default."
        )
    )
    parser.add_argument("--condition", choices=["CUE", "SILENCE"], default="CUE")
    parser.add_argument(
        "--decision", choices=["rule", "linear", "local", "steer"], default="rule"
    )
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument(
        "--events", type=Path, default=ROOT / "experiments" / "sync-smoke.jsonl"
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pace replay by event timestamps. Off by default so paired replay "
            "timing is deterministic (default: off)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--assignment",
        choices=["random", "random-operation", "canonical", "alternate", "fixed"],
        default="random",
        help=(
            "Cue assignment. 'random' randomizes wording only and must not be "
            "used for operation-level causal comparisons; "
            "'random-operation' is the exploration arm that draws the operation."
        ),
    )
    parser.add_argument(
        "--cue-style",
        choices=["bare", "stem"],
        default="bare",
        help=(
            "Cue style passed to the CLI (default: bare, preserving the "
            "historical experiment behaviour). 'stem' speaks a complete "
            "initial-condition sentence; length-match NEUTRAL to it."
        ),
    )
    parser.add_argument("--cue", help="Cue used with --assignment fixed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "experiment",
        help="Directory for logs, reports, and the manifest.",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=ROOT / "artifacts" / "sync-policy.json",
    )
    parser.add_argument(
        "--outcome-window-ms",
        type=int,
        default=8_000,
        help="Continuation window for uptake and carryover (default: 8000).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / ".build" / "macos-speech-events",
    )
    parser.add_argument("--tts-voice", default="Kyoko")
    parser.add_argument(
        "--adapter-args-json",
        default="[]",
        help="Extra args appended to the speech adapter command (JSON list).",
    )
    parser.add_argument(
        "--tts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Speak cues; defaults to on for live CUE and off for replay/SILENCE.",
    )
    parser.add_argument("--session", help="Optional session label.")
    parser.add_argument(
        "--pair-key",
        help="Key shared by the CUE and SILENCE runs of one pair (default: derived).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration < 1:
        print("error: --duration must be positive", file=sys.stderr)
        return 2
    if args.outcome_window_ms < 1:
        print("error: --outcome-window-ms must be positive", file=sys.stderr)
        return 2
    if args.assignment == "fixed" and not args.cue:
        print("error: --assignment fixed requires --cue", file=sys.stderr)
        return 2
    if args.decision == "linear" and not args.policy_model.exists():
        print(
            f"error: missing linear policy model: {args.policy_model}\n"
            "train one with scripts/train_policy.py",
            file=sys.stderr,
        )
        return 2
    if args.mode == "live" and not args.adapter.exists():
        print(f"error: missing speech adapter: {args.adapter}", file=sys.stderr)
        return 2

    tts = args.tts
    if tts is None:
        tts = args.mode == "live" and args.condition == "CUE"
    if args.condition == "SILENCE":
        tts = False

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{_slug(args.session)}" if args.session else ""
    stem = f"{stamp}-{args.condition}-{args.decision}-seed{args.seed}{suffix}"
    output_dir = args.output_dir
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{stem}.jsonl"
    report_path = reports_dir / f"{stem}.json"
    fixture = args.events.stem if args.mode == "replay" else "live"
    pair_key = args.pair_key or f"{args.decision}-seed{args.seed}-{fixture}"

    command = _entrypoint() + [
        "--decision",
        args.decision,
        "--assignment",
        args.assignment,
        "--cue-style",
        args.cue_style,
        "--seed",
        str(args.seed),
        "--condition-label",
        args.condition,
        "--outcome-window-ms",
        str(args.outcome_window_ms),
        "--session-id",
        stem,
        "--pair-key",
        pair_key,
        "--log",
        str(log_path),
        "--report",
        str(report_path),
    ]
    if args.assignment == "fixed":
        command += ["--cue", str(args.cue)]
    if args.decision == "linear":
        command += ["--policy-model", str(args.policy_model)]
    elif args.decision == "local":
        command += ["--decision-fallback", "rule"]
    if args.condition == "SILENCE":
        command += ["--silent"]
    if tts:
        command += ["--tts-command-json", json.dumps(_say_command(args.tts_voice))]
    if args.mode == "live":
        extra_args = json.loads(args.adapter_args_json)
        if not isinstance(extra_args, list):
            print("error: --adapter-args-json must be a JSON list", file=sys.stderr)
            return 2
        event_command = [
            str(args.adapter),
            "--duration",
            str(args.duration),
            "--locale",
            "ja-JP",
            *[str(item) for item in extra_args],
        ]
        command += ["--event-command-json", json.dumps(event_command)]
    else:
        command += ["--events", str(args.events)]
        if args.realtime:
            command += ["--realtime"]

    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "condition": args.condition,
        "decision": args.decision,
        "cue_style": args.cue_style,
        "mode": args.mode,
        "seed": args.seed,
        "assignment": args.assignment,
        "pair_key": pair_key,
        "realtime": args.realtime,
        "tts": tts,
        "log": str(log_path),
        "report": str(report_path),
        "command": command,
    }
    manifest_path = output_dir / "manifest.jsonl"
    if not args.dry_run:
        completed = subprocess.run(command, cwd=ROOT, check=False)
        status = completed.returncode
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        manifest["returncode"] = status
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        status = 0
    print(f"log={log_path}")
    print(f"report={report_path}")
    print(f"manifest={manifest_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
