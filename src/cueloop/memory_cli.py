from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .memory import retain_session_structure
from .personalization import PersonalizationProfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly retain, inspect, or forget user-formed reasoning structures."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    retain = subcommands.add_parser("retain")
    retain.add_argument("session_directory", type=Path)
    retain.add_argument("item_id")
    retain.add_argument("--retention-days", type=int, default=365)
    inspect = subcommands.add_parser("list")
    forget = subcommands.add_parser("forget")
    forget.add_argument("structure_id")
    subcommands.add_parser("purge-expired")
    for command in (retain, inspect, forget, subcommands.choices["purge-expired"]):
        command.add_argument("--profile", type=Path, default=Path("data/personalization.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "retain":
            item = retain_session_structure(
                args.session_directory,
                args.item_id,
                args.profile,
                retention_days=args.retention_days,
            )
            print(f"retained={item.id} expires_at={item.expires_at}")
            return 0
        profile = PersonalizationProfile.load(args.profile)
        if args.command == "list":
            for item in profile.retained_structures:
                print(json.dumps(item.__dict__, ensure_ascii=False))
            return 0
        if args.command == "forget":
            if not profile.forget_structure(args.structure_id):
                raise ValueError(f"retained structure not found: {args.structure_id}")
            profile.save(args.profile)
            print(f"forgot={args.structure_id}")
            return 0
        removed = profile.prune_retained_structures()
        profile.save(args.profile)
        print(f"purged={removed}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
