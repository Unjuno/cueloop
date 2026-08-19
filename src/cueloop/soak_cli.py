from __future__ import annotations

import argparse
from pathlib import Path

from .soak import run_soak, write_soak_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic continuous-session load test.")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--report", type=Path, default=Path("artifacts/voice-soak.json"))
    args = parser.parse_args(argv)
    try:
        report = run_soak(args.minutes)
        write_soak_report(args.report, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"duration_ms={report['simulated_duration_ms']} turns={report['turns']} "
        f"passed={report['passed']} report={args.report}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
