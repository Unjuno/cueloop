from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .structure import extract_structure, load_turns, write_structure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a reviewable reasoning structure from a session log.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/session-structure.json"))
    args = parser.parse_args(argv)
    try:
        structure = extract_structure(load_turns(args.input))
        write_structure(args.output, structure)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"claims={len(structure.claims)} questions={len(structure.unresolved_questions)} "
        f"verify={len(structure.verification_queue)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
