#!/usr/bin/env python3
"""LLM predictor agreement with real Japanese speech (Lux corpus).

Same methodology as `scripts/tune_ngram.py`, replacing the n-gram predictor
with the production local-LLM predictor (`LocalLLMPredictor`, which is what
`--predictor local` uses live). The local LLM is the workhorse of CUELOOP;
the n-gram is a fast fallback. Measuring LLM-backed agreement isolates how
much of the problem is *the* boundary vs *local model quality on real
Japanese speech*.

This is a benchmark of the prediction component — it does not measure the
cueing effect (H1/H2), uptake, carryover, or anything about reasoning.

Usage:
  scripts/benchmark_local_llm.py --corpus data/real-speech/lux/metadata.csv \
      --recitation-only --prefix-chars 12 --samples 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.predictors import LocalLLMPredictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--recitation-only", action="store_true")
    parser.add_argument("--prefix-chars", type=int, default=12)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deselect",
        type=int,
        default=0,
        help="Skip the first N eligible lines before sampling (default: 0).",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.corpus.open(encoding="utf-8")))
    if args.recitation_only:
        rows = [r for r in rows if r["cleaned_path"].startswith("data/cleaned/ita_recitation/")]
    lines = [r["transcript"] for r in rows]
    # deterministic selection
    stride = max(1, len(lines) // 200)
    selected = [
        (i, line) for i, line in enumerate(lines)
        if line.strip() and len(line) > args.prefix_chars + 6
    ][args.deselect :][::stride][: args.samples]
    print(f"selected {len(selected)} lines from {len(lines)} total")

    predictor = LocalLLMPredictor(
        base_url=args.base_url,
        model=args.model,
        temperature=0.0,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )

    exact = 0
    prefix = 0
    valid = 0
    failures = 0
    for index, line in selected:
        prefix_text = line[: args.prefix_chars]
        truth = line[args.prefix_chars:]
        prediction = predictor.predict(
            text=prefix_text, context=(), now_ms=0,
        )
        if prediction is None:
            failures += 1
            continue
        span = prediction.span
        if not span:
            failures += 1
            continue
        valid += 1
        if truth.startswith(span):
            exact += 1
        elif truth.startswith(span[: max(1, len(span) // 2)]):
            prefix += 1

    print(json.dumps({
        "model": args.model,
        "corpus": str(args.corpus),
        "lines_scored": valid,
        "exact_prefix_rate": exact / valid if valid else None,
        "half_prefix_rate": (exact + prefix) / valid if valid else None,
        "failures": failures,
        "matched_exact": exact,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
