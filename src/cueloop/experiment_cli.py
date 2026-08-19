from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import ControllerConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and safely tune the CUELOOP intervention policy.")
    parser.add_argument("--dataset", type=Path, default=Path("experiments/policy-eval.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/policy-experiment.json"))
    parser.add_argument("--policy-out", type=Path, default=Path("artifacts/policy.json"))
    parser.add_argument("--current-policy", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        current = ControllerConfig.from_json_file(args.current_policy) if args.current_policy else None
        report = run_experiment(args.dataset, args.report, args.policy_out, current)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    metrics = report["tuned_eval"]["metrics"]
    print(
        f"eval accuracy={metrics['accuracy']:.3f} "
        f"harmful_interruptions={metrics['harmful_interruptions']} "
        f"adopted={report['adopted']}"
    )
    print(f"report={args.report} policy={args.policy_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
