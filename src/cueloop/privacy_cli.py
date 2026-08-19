from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .privacy import delete_session_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or delete CUELOOP's local session-data directories.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete data/, logs/, artifacts/, recordings/, and transcripts/.",
    )
    args = parser.parse_args(argv)
    try:
        results = delete_session_data(args.root, confirm=args.confirm)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    mode = "removed" if args.confirm else "would_remove"
    for result in results:
        if result.existed:
            print(f"{mode}={result.path}")
    if not args.confirm:
        print("dry-run only; pass --confirm to delete these local session-data directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
