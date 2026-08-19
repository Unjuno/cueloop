from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pilot import (
    RUBRIC_FIELDS,
    analyze_pilot,
    append_pilot_record,
    generate_assignment,
    load_pilot_data,
    pilot_record_from_session,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or analyze a counterbalanced retained-reasoning pilot.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    assign = subcommands.add_parser("assign")
    assign.add_argument("--participants", nargs="+", required=True)
    assign.add_argument("--tasks", nargs="+", required=True)
    assign.add_argument("--seed", type=int, default=20260812)
    assign.add_argument("--output", type=Path, default=Path("artifacts/pilot-assignment.json"))
    analyze = subcommands.add_parser("analyze")
    analyze.add_argument("data", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("artifacts/pilot-report.json"))
    analyze.add_argument("--assignment", type=Path)
    record = subcommands.add_parser("record")
    record.add_argument("session_directory", type=Path)
    record.add_argument("--participant", required=True)
    record.add_argument("--task", required=True)
    for field in RUBRIC_FIELDS:
        record.add_argument(f"--{field.replace('_', '-')}", type=float, required=True)
    record.add_argument("--productive-interruptions", type=int, default=0)
    record.add_argument("--disruptive-interventions", type=int, default=0)
    record.add_argument("--recall-delay-minutes", type=float, required=True)
    record.add_argument("--scorer-blinded", action="store_true")
    record.add_argument("--output", type=Path, default=Path("data/pilot-results.jsonl"))
    args = parser.parse_args(argv)
    try:
        if args.command == "assign":
            payload = generate_assignment(args.participants, args.tasks, args.seed)
            write_json(args.output, payload)
        elif args.command == "analyze":
            payload = analyze_pilot(load_pilot_data(args.data), args.data, args.assignment)
            write_json(args.output, payload)
        else:
            rubric = {field: getattr(args, field) for field in RUBRIC_FIELDS}
            payload = pilot_record_from_session(
                args.session_directory,
                participant_id=args.participant,
                task_id=args.task,
                rubric=rubric,
                productive_interruptions=args.productive_interruptions,
                disruptive_interventions=args.disruptive_interventions,
                recall_delay_minutes=args.recall_delay_minutes,
                scorer_blinded=args.scorer_blinded,
            )
            append_pilot_record(args.output, payload)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
