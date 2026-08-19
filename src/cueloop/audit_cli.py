from __future__ import annotations

import argparse
from pathlib import Path

from .audit import run_completion_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit CUELOOP software completion evidence.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    checks = run_completion_audit(args.root.resolve())
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.evidence}")
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
