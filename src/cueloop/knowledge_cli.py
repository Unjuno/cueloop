from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .knowledge import KnowledgeIndex, ground_structure, load_structure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ground extracted claims against local text sources.")
    parser.add_argument("structure", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grounded-structure.json"))
    args = parser.parse_args(argv)
    try:
        index = KnowledgeIndex.from_paths(args.sources)
        result = ground_structure(load_structure(args.structure), index)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    supported = sum(item["status"] == "SUPPORTED" for item in result["grounding"])
    print(f"grounded={supported}/{len(result['grounding'])} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
