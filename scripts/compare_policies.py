#!/usr/bin/env python3
"""Compare intervention backends from cueloop-sync report JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.evaluate import policy_family


def _reports(paths: list[Path]):
    for path in paths:
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "decision_model" in payload:
                yield candidate, payload


def _outcomes(report: dict[str, object]) -> list[dict[str, object]]:
    raw = report.get("outcomes", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def aggregate(reports: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for report in reports:
        family = policy_family(report.get("decision_model"))
        condition = str(report.get("condition", "unspecified"))
        key = (family, condition)
        group = groups.setdefault(
            key,
            {
                "policy": family,
                "condition": condition,
                "sessions": 0,
                "delivered": 0,
                "outcomes": 0,
                "uptake": 0,
                "novel_operation": 0,
                "novel_uptake": 0,
                "carryover": 0,
                "novel_carryover": 0,
                "repertoire_growth": set(),
                "boundary_to_delivery_ms": [],
                "decision_models": set(),
                "predictors": set(),
            },
        )
        group["sessions"] = int(group["sessions"]) + 1
        group["delivered"] = int(group["delivered"]) + int(
            report.get("delivered", 0) or 0
        )
        group["decision_models"].add(str(report.get("decision_model", "")))
        group["predictors"].add(str(report.get("predictor", "")))
        growth = report.get("repertoire_growth", [])
        if isinstance(growth, list):
            group["repertoire_growth"].update(str(item) for item in growth)
        delays = report.get("boundary_to_delivery_ms", [])
        if isinstance(delays, list):
            group["boundary_to_delivery_ms"].extend(
                int(item) for item in delays if isinstance(item, (int, float))
            )
        sessions_outcomes = _outcomes(report)
        group["outcomes"] = int(group["outcomes"]) + len(sessions_outcomes)
        for outcome in sessions_outcomes:
            target = outcome.get("target_operation_seen") is True
            carried = outcome.get("carryover_seen") is True
            novel = outcome.get("novel_operation") is True
            group["uptake"] = int(group["uptake"]) + int(target)
            group["carryover"] = int(group["carryover"]) + int(carried)
            if novel:
                group["novel_operation"] = int(group["novel_operation"]) + 1
                group["novel_uptake"] = int(group["novel_uptake"]) + int(target)
                group["novel_carryover"] = int(group["novel_carryover"]) + int(carried)

    rendered: dict[str, dict[str, object]] = {}
    for (family, condition), group in sorted(groups.items()):
        delivered = int(group["delivered"])
        outcomes = int(group["outcomes"]) or delivered
        novel_operation = int(group["novel_operation"])
        delays = list(group["boundary_to_delivery_ms"])
        rendered[f"{family}|{condition}"] = {
            "policy": family,
            "condition": condition,
            "sessions": group["sessions"],
            "delivered": delivered,
            "outcomes": outcomes,
            "uptake": group["uptake"],
            "uptake_rate": group["uptake"] / outcomes if outcomes else 0.0,
            "novel_operation": novel_operation,
            "novel_uptake": group["novel_uptake"],
            "novel_uptake_rate": (
                group["novel_uptake"] / novel_operation if novel_operation else 0.0
            ),
            "carryover": group["carryover"],
            "carryover_rate": group["carryover"] / outcomes if outcomes else 0.0,
            "novel_carryover": group["novel_carryover"],
            "novel_carryover_rate": (
                group["novel_carryover"] / novel_operation if novel_operation else 0.0
            ),
            "repertoire_growth": sorted(group["repertoire_growth"]),
            "repertoire_growth_mean": (
                len(group["repertoire_growth"]) / int(group["sessions"])
                if group["sessions"]
                else 0.0
            ),
            "median_boundary_to_delivery_ms": (
                statistics.median(delays) if delays else None
            ),
            "decision_models": sorted(group["decision_models"]),
            "predictors": sorted(group["predictors"]),
        }
    return {"generated_at": datetime.now(UTC).isoformat(), "groups": rendered}


def markdown(payload: dict[str, object]) -> str:
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        return "no policy reports\n"
    lines = [
        "# cueloop policy comparison",
        "",
        "| policy | condition | sessions | delivered | uptake | novel uptake | carryover | novel carryover | repertoire growth | median delay ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        delay = group["median_boundary_to_delivery_ms"]
        lines.append(
            "| {policy} | {condition} | {sessions} | {delivered} | {uptake:.2f} | "
            "{novel_uptake:.2f} | {carryover:.2f} | {novel_carryover:.2f} | "
            "{growth:.2f} | {delay} |".format(
                policy=group["policy"],
                condition=group["condition"],
                sessions=group["sessions"],
                delivered=group["delivered"],
                uptake=group["uptake_rate"],
                novel_uptake=group["novel_uptake_rate"],
                carryover=group["carryover_rate"],
                novel_carryover=group["novel_carryover_rate"],
                growth=group["repertoire_growth_mean"],
                delay="-" if delay is None else f"{float(delay):.0f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare rule, linear, and local-LLM reports by carryover."
    )
    parser.add_argument(
        "reports", nargs="+", type=Path, help="Report JSON files or directories."
    )
    parser.add_argument("--json", type=Path, help="Write the aggregate JSON here.")
    parser.add_argument("--markdown", type=Path, help="Write the Markdown table here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [payload for _, payload in _reports(args.reports)]
    if not reports:
        print("error: no cueloop-sync report JSON found", file=sys.stderr)
        return 2
    payload = aggregate(reports)
    table = markdown(payload)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
