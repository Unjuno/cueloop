#!/usr/bin/env python3
"""Paired CUE vs SILENCE analysis of sync experiment reports.

Sessions are the unit of analysis. A CUE run and a SILENCE run with the same
pair_key form one pair; the paired difference is computed first, then a seeded
bootstrap resamples pairs to give an interval. The seed and the number of
resamples are part of the output, so a rerun on the same reports is bit-for-bit
identical.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.evaluate import policy_family

METRICS: tuple[str, ...] = (
    "uptake_rate",
    "carryover_rate",
    "novel_uptake_rate",
    "novel_carryover_rate",
    "repertoire_growth",
    "median_boundary_to_delivery_ms",
)

COUNT_METRICS: tuple[str, ...] = (
    "delivered",
    "outcomes",
    "uptake",
    "carryover",
    "novel_operation",
    "novel_uptake",
    "novel_carryover",
)


def _outcomes(report: dict[str, object]) -> list[dict[str, object]]:
    raw = report.get("outcomes", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _rate(items: list[dict[str, object]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get(key) is True) / len(items)


def _numbers(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]


def session_metrics(report: dict[str, object]) -> dict[str, object]:
    """One session's counts and rates, computed from its report only."""
    outcomes = _outcomes(report)
    novel = [outcome for outcome in outcomes if outcome.get("novel_operation") is True]
    delays = _numbers(report.get("boundary_to_delivery_ms"))
    growth = report.get("repertoire_growth", [])
    return {
        "policy": policy_family(report.get("decision_model")),
        "condition": str(report.get("condition", "unspecified")),
        "pair_key": str(report.get("pair_key") or ""),
        "session_id": str(report.get("session_id") or ""),
        "assignment": str(report.get("assignment", "unspecified")),
        "delivered": int(report.get("delivered") or 0),
        "outcomes": len(outcomes),
        "uptake": sum(
            1 for outcome in outcomes if outcome.get("target_operation_seen") is True
        ),
        "carryover": sum(
            1 for outcome in outcomes if outcome.get("carryover_seen") is True
        ),
        "novel_operation": len(novel),
        "novel_uptake": sum(
            1 for outcome in novel if outcome.get("target_operation_seen") is True
        ),
        "novel_carryover": sum(
            1 for outcome in novel if outcome.get("carryover_seen") is True
        ),
        "uptake_rate": _rate(outcomes, "target_operation_seen"),
        "carryover_rate": _rate(outcomes, "carryover_seen"),
        "novel_uptake_rate": _rate(novel, "target_operation_seen"),
        "novel_carryover_rate": _rate(novel, "carryover_seen"),
        "repertoire_growth": len(growth) if isinstance(growth, list) else 0,
        "median_boundary_to_delivery_ms": (
            float(statistics.median(delays)) if delays else None
        ),
    }


def load_sessions(paths: list[Path]) -> list[tuple[str, dict[str, object]]]:
    sessions: list[tuple[str, dict[str, object]]] = []
    for path in paths:
        if path.is_dir():
            # flat layout (reports/*.json) and two-level arm layout
            # (reports/{cue,silence}/*.json), matching the simulator output.
            candidates = sorted(path.glob("*.json")) + sorted(path.glob("*/*.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "decision_model" in payload:
                sessions.append((candidate.name, session_metrics(payload)))
    return sorted(sessions, key=lambda item: item[0])


def group_summary(
    sessions: list[tuple[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, object]]] = {}
    for _, metrics in sessions:
        key = (str(metrics["policy"]), str(metrics["condition"]))
        buckets.setdefault(key, []).append(metrics)
    groups: dict[str, dict[str, object]] = {}
    for (policy, condition), items in sorted(buckets.items()):
        entry: dict[str, object] = {
            "policy": policy,
            "condition": condition,
            "n_sessions": len(items),
        }
        for metric in COUNT_METRICS:
            entry[metric] = sum(int(item[metric]) for item in items)
        for metric in METRICS:
            values = [float(item[metric]) for item in items if item[metric] is not None]
            entry[metric] = float(statistics.mean(values)) if values else None
        entry["session_ids"] = sorted(str(item["session_id"]) for item in items)
        groups[f"{policy}|{condition}"] = entry
    return groups


def build_pairs(
    sessions: list[tuple[str, dict[str, object]]],
) -> tuple[dict[str, list[tuple[dict[str, object], dict[str, object]]]], list[str]]:
    buckets: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {}
    unpaired: set[str] = set()
    for _, metrics in sessions:
        policy = str(metrics["policy"])
        pair_key = str(metrics["pair_key"])
        condition = str(metrics["condition"])
        if not pair_key or condition not in {"CUE", "SILENCE"}:
            unpaired.add(policy)
            continue
        buckets.setdefault((policy, pair_key), {"CUE": [], "SILENCE": []})
        buckets[(policy, pair_key)][condition].append(metrics)
    pairs: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for (policy, _pair_key), bucket in sorted(buckets.items()):
        count = min(len(bucket["CUE"]), len(bucket["SILENCE"]))
        if count == 0:
            unpaired.add(policy)
            continue
        pairs.setdefault(policy, []).extend(
            zip(bucket["CUE"][:count], bucket["SILENCE"][:count])
        )
    return pairs, sorted(unpaired)


def _bootstrap_interval(
    values: list[float], *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    n = len(values)
    generator = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = [values[generator.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * len(means))
    high_index = min(len(means) - 1, int((1.0 - tail) * len(means)))
    return means[low_index], means[high_index]


def paired_differences(
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for metric in METRICS:
        deltas = [
            float(cue[metric]) - float(silence[metric])
            for cue, silence in pairs
            if cue[metric] is not None and silence[metric] is not None
        ]
        if not deltas:
            metrics[metric] = {
                "point": None,
                "ci_low": "insufficient",
                "ci_high": "insufficient",
                "n_pairs": 0,
            }
            continue
        point = sum(deltas) / len(deltas)
        if len(deltas) < 2:
            metrics[metric] = {
                "point": point,
                "ci_low": "insufficient",
                "ci_high": "insufficient",
                "n_pairs": len(deltas),
            }
            continue
        low, high = _bootstrap_interval(
            deltas,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
        )
        metrics[metric] = {
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "n_pairs": len(deltas),
        }
    return metrics


def analyze(
    sessions: list[tuple[str, dict[str, object]]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, object]:
    groups = group_summary(sessions)
    pairs, unpaired = build_pairs(sessions)
    compared: dict[str, object] = {}
    policies = sorted(
        {str(metrics["policy"]) for _, metrics in sessions} | set(unpaired)
    )
    for policy in policies:
        policy_pairs = pairs.get(policy, [])
        if not policy_pairs:
            compared[policy] = {
                "n_pairs": 0,
                "status": "not_comparable",
                "reason": "no CUE/SILENCE pair with the same pair_key",
                "metrics": {},
            }
            continue
        compared[policy] = {
            "n_pairs": len(policy_pairs),
            "status": "ok",
            "metrics": paired_differences(
                policy_pairs,
                resamples=resamples,
                seed=seed,
                confidence=confidence,
            ),
        }
    return {
        "seed": seed,
        "resamples": resamples,
        "confidence": confidence,
        "reports": sorted(name for name, _ in sessions),
        "groups": groups,
        "comparisons": compared,
    }


def markdown(payload: dict[str, object]) -> str:
    groups = payload.get("groups", {})
    comparisons = payload.get("comparisons", {})
    if not isinstance(groups, dict) or not isinstance(comparisons, dict):
        return "no experiment reports\n"
    lines = [
        "# cueloop paired experiment",
        "",
        (
            f"- bootstrap: {payload['resamples']} resamples, "
            f"{float(payload['confidence']):.0%} CI, seed {payload['seed']}"
        ),
        "",
        (
            "| policy | condition | sessions | delivered | outcomes | uptake | "
            "carryover | novel uptake | novel carryover | growth | median delay ms |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in groups.values():
        if not isinstance(entry, dict):
            continue

        def fmt(metric: str, entry: dict[str, object] = entry) -> str:
            value = entry.get(metric)
            if value is None:
                return "-"
            return f"{float(value):.2f}"

        delay = entry.get("median_boundary_to_delivery_ms")
        lines.append(
            f"| {entry['policy']} | {entry['condition']} | {entry['n_sessions']} | "
            f"{entry['delivered']} | {entry['outcomes']} | {fmt('uptake_rate')} | "
            f"{fmt('carryover_rate')} | {fmt('novel_uptake_rate')} | "
            f"{fmt('novel_carryover_rate')} | {fmt('repertoire_growth')} | "
            f"{'-' if delay is None else f'{float(delay):.0f}'} |"
        )
    lines += [
        "",
        "## Paired differences (CUE - SILENCE)",
        "",
        "| policy | metric | n_pairs | delta | CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for policy, entry in comparisons.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if not metrics:
            lines.append(
                f"| {policy} | - | 0 | - | not comparable: {entry.get('reason', '-')} |"
            )
            continue
        for metric, values in metrics.items():
            if not isinstance(values, dict):
                continue
            point = values.get("point")
            low = values.get("ci_low")
            high = values.get("ci_high")
            if isinstance(low, str) or isinstance(high, str):
                interval = "insufficient"
            else:
                interval = f"[{float(low):+.3f}, {float(high):+.3f}]"
            point_text = "-" if point is None else f"{float(point):+.3f}"
            lines.append(
                f"| {policy} | {metric} | {values.get('n_pairs', 0)} | "
                f"{point_text} | {interval} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired CUE vs SILENCE analysis with a seeded bootstrap CI."
    )
    parser.add_argument(
        "reports", nargs="+", type=Path, help="Report JSON files or directories."
    )
    parser.add_argument("--json", type=Path, help="Write the analysis JSON here.")
    parser.add_argument("--markdown", type=Path, help="Write the Markdown report here.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resamples < 1:
        print("error: --resamples must be positive", file=sys.stderr)
        return 2
    if not 0.0 < args.confidence < 1.0:
        print("error: --confidence must be between 0 and 1", file=sys.stderr)
        return 2
    sessions = load_sessions(args.reports)
    if not sessions:
        print("error: no cueloop-sync report JSON found", file=sys.stderr)
        return 2
    payload = analyze(
        sessions,
        resamples=args.resamples,
        seed=args.seed,
        confidence=args.confidence,
    )
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
