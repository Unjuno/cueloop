#!/usr/bin/env python3
"""H2 (retention): score delayed, unassisted reconstructions, paired CUE-SILENCE.

This implements the *secondary* outcome in docs/experiment-protocol.md:
after a filler task, the participant writes a reconstruction of each task
without notes. Two raters blind to condition score seven rubric items; this
tool takes those scores (or a deterministic proxy for dry runs), pairs them by
``pair_key`` with the same seeded bootstrap as the H1 analyzer, and reports
the CUE - SILENCE contrast for the rubric total and each subscore.

Input JSON per reconstruction:
  {"pair_key": "...", "condition": "CUE"|"SILENCE", "task": "...",
   "text": "delayed reconstruction ...",
   "raters": [{"rater": "r1", "scores": {"claims": 2, ...}}, ...]}

For a deterministic smoke test, omit ``raters``: a lexical proxy scorer is
used. It is NOT human evidence and is labeled as such in the output.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

RUBRIC: tuple[str, ...] = (
    "claims",
    "premises",
    "dependencies",
    "counterexamples",
    "questions",
    "unsupported",
    "coherence",
)


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
    low = int(tail * len(means))
    high = min(len(means) - 1, int((1.0 - tail) * len(means)))
    return means[low], means[high]


def proxy_scores(text: str) -> dict[str, float]:
    """Deterministic lexical proxy for a rubric (DRY RUN ONLY, not evidence).

    Counts operation-marker phrases per 100 characters for the first six
    items; coherence is 1.0 when the text has sentence-final punctuation and
    multiple sentences. Bounded 0..3 to match a typical 0..3 human rubric.
    """
    markers = {
        "claims": ("結論", "つまり", "だから"),
        "premises": ("前提", "根拠", "なぜなら"),
        "dependencies": ("したがって", "そのため", "よって"),
        "counterexamples": ("逆に", "例外", "ただし"),
        "questions": ("?", "？", "疑問", "未解決"),
        "unsupported": ("たぶん", "おそらく", "たまたま"),
    }
    scale = max(1, len(text)) / 100.0
    out: dict[str, float] = {}
    for key, words in markers.items():
        count = sum(text.count(w) for w in words)
        out[key] = min(3.0, round(count / scale, 2))
    sentences = sum(text.count(p) for p in ("。", ".", "！", "?"))
    out["coherence"] = min(3.0, float(sentences >= 2))
    return out


def llm_rater_scores(text: str, *, model: str, base_url: str) -> dict[str, float]:
    """Blinded LLM rater: score a reconstruction on the 7 rubric items 0..3.

    The condition label is NOT supplied, so the rater cannot know whether the
    text came from a CUE or SILENCE arm. Returns the average of two independent
    samples to reduce single-generation variance.
    """
    import urllib.request

    items = "\n".join(
        f"- {name}: "
        + {
            "claims": "明確な主張の多さ",
            "premises": "前提・根拠・証拠の明確さ",
            "dependencies": "主張間の依存関係の明確さ",
            "counterexamples": "反例・代替説明の有無",
            "questions": "未解決質問の明示",
            "unsupported": "根拠のない断定の多さ（高いほど悪い）",
            "coherence": "文章全体の一体感",
        }[name]
        for name in RUBRIC
    )
    prompt = (
        "あなたは blind な評価者です。以下の再構成テキストだけを読み、"
        "各項目を 0（なし）〜3（非常に明確）で採点してください。\n"
        f"採点項目:\n{items}\n\n"
        "JSON オブジェクトのみで答えてください（説明不要）:\n"
        "{" + ", ".join(f'"{k}": 0' for k in RUBRIC) + "}\n\n"
        f"再構成テキスト:\n{text}\n"
    )
    samples: list[dict[str, float]] = []
    for _ in range(2):
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                payload = json.loads(response.read().decode())
            content = payload["message"]["content"]
            # Local models often wrap JSON in a markdown code fence.
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.lstrip().startswith("json"):
                    content = content.lstrip()[4:]
            parsed = json.loads(content.strip())
            samples.append(
                {k: max(0.0, min(3.0, float(parsed[k]))) for k in RUBRIC if k in parsed}
            )
        except (OSError, ValueError, KeyError, TypeError):
            # A bad rater sample is skipped; the other sample still counts.
            continue
    if not samples:
        raise ValueError("LLM rater produced no usable scores")
    return {k: statistics.mean(s[k] for s in samples if k in s) for k in RUBRIC}


def human_scores(record: dict) -> dict[str, float]:
    """Average the blinded raters' scores for each rubric item."""
    raters = record.get("raters") or []
    numeric: list[dict] = []
    for rater in raters:
        if not isinstance(rater, dict):
            continue
        scores = rater.get("scores")
        if isinstance(scores, dict):
            numeric.append(
                {
                    k: float(scores[k])
                    for k in RUBRIC
                    if isinstance(scores.get(k), (int, float))
                }
            )
    if not numeric:
        raise ValueError("record has no usable 'raters' scores")
    return {k: statistics.mean(r[k] for r in numeric if k in r) for k in RUBRIC}


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("pair_key"):
                payload["_source"] = path.name
                records.append(payload)
    return records


def score_records(
    records: list[dict],
    *,
    proxy: bool,
    llm_model: str | None = None,
    llm_base_url: str = "http://127.0.0.1:11434",
) -> list[dict]:
    scored = []
    for record in records:
        if llm_model:
            scores = llm_rater_scores(
                str(record.get("text", "")), model=llm_model, base_url=llm_base_url
            )
            source = f"blinded-llm-rater({llm_model})"
        elif proxy:
            scores = proxy_scores(str(record.get("text", "")))
            source = "proxy"
        else:
            scores = human_scores(record)
            source = "blinded-raters"
        scored.append(
            {
                "pair_key": str(record["pair_key"]),
                "condition": str(record.get("condition", "")).upper(),
                "task": str(record.get("task", "")),
                "scores": scores,
                "total": sum(scores.values()),
                "source": source,
                "_source": record.get("_source", ""),
            }
        )
    return scored


def pair_and_compare(
    scored: list[dict], *, resamples: int, seed: int, confidence: float
) -> dict:
    buckets: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in scored:
        if row["condition"] not in {"CUE", "SILENCE"} or not row["pair_key"]:
            continue
        key = (row["pair_key"], row["task"])
        buckets.setdefault(key, {"CUE": [], "SILENCE": []})[row["condition"]].append(
            row
        )
    pairs: list[tuple[dict, dict]] = []
    for bucket in buckets.values():
        n = min(len(bucket["CUE"]), len(bucket["SILENCE"]))
        pairs.extend(zip(bucket["CUE"][:n], bucket["SILENCE"][:n]))

    metrics = ["total", *RUBRIC]
    result: dict[str, dict] = {}
    for metric in metrics:
        if metric == "total":
            deltas = [float(c["total"]) - float(s["total"]) for c, s in pairs]
        else:
            deltas = [
                float(c["scores"][metric]) - float(s["scores"][metric])
                for c, s in pairs
            ]
        if not deltas:
            result[metric] = {"point": None, "ci": "insufficient", "n_pairs": 0}
            continue
        point = sum(deltas) / len(deltas)
        if len(deltas) < 2:
            result[metric] = {"point": point, "ci": "insufficient", "n_pairs": 1}
            continue
        low, high = _bootstrap_interval(
            deltas, resamples=resamples, seed=seed, confidence=confidence
        )
        result[metric] = {
            "point": point,
            "ci_low": low,
            "ci_high": high,
            "n_pairs": len(deltas),
            "ci_excludes_zero": bool(low > 0 or high < 0),
        }
    return {
        "seed": seed,
        "resamples": resamples,
        "confidence": confidence,
        "n_pairs": len(pairs),
        "scoring_source": scored[0]["source"] if scored else "none",
        "metrics": result,
    }


def markdown(payload: dict) -> str:
    lines = [
        "# H2 delayed reconstruction (CUE - SILENCE)",
        "",
        f"- pairs: {payload['n_pairs']}",
        f"- scoring: {payload['scoring_source']}",
        (
            f"- bootstrap: {payload['resamples']} resamples, "
            f"{float(payload['confidence']):.0%} CI, seed {payload['seed']}"
        ),
        "",
        "| rubric item | delta | 95% CI | n_pairs | excludes zero |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for metric, values in payload["metrics"].items():
        point = values["point"]
        if point is None:
            lines.append(f"| {metric} | - | insufficient | 0 | - |")
            continue
        ci = values.get("ci")
        if ci == "insufficient" or "ci_low" not in values:
            interval = "insufficient"
        else:
            interval = f"[{values['ci_low']:+.2f}, {values['ci_high']:+.2f}]"
        flag = "yes" if values.get("ci_excludes_zero") else "no"
        lines.append(
            f"| {metric} | {point:+.3f} | {interval} | {values['n_pairs']} | {flag} |"
        )
    if payload["scoring_source"] == "proxy":
        lines += [
            "",
            (
                "**WARNING:** proxy scoring is a deterministic dry-run readout, "
                "NOT human evidence. Collect blinded rater scores for H2."
            ),
        ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", nargs="+", type=Path, help="Reconstruction JSONL files."
    )
    parser.add_argument("--json", type=Path, help="Write the analysis JSON here.")
    parser.add_argument("--markdown", type=Path, help="Write the Markdown report here.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use the lexical proxy scorer (dry run only, not human evidence).",
    )
    parser.add_argument(
        "--llm-rater",
        metavar="MODEL",
        help="Use a blinded local-LLM rater (agent-side H2, not human evidence).",
    )
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:11434")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_records(args.records)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error reading records: {exc}", file=sys.stderr)
        return 2
    if not records:
        print("error: no reconstruction records found", file=sys.stderr)
        return 2
    scored = score_records(
        records,
        proxy=args.proxy,
        llm_model=args.llm_rater,
        llm_base_url=args.llm_base_url,
    )
    payload = pair_and_compare(
        scored,
        resamples=args.resamples,
        seed=args.seed,
        confidence=args.confidence,
    )
    table = markdown(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
