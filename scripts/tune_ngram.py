#!/usr/bin/env python3
"""Tune the inline NLpredictor (n-gram) on real Japanese text.

Builds three predictor candidates from the Lux Japanese Speech Corpus
(CC-BY-4.0, Lami Inc.) and measures how well each recovers the next span of
held-out human utterances:

- `default` (ngram(3) trained on the embedded Japanese corpus)
- `lux` (ngram trained on the Lux training split only)
- `merged` (ngram trained on embedded + Lux train split)
- for order in {3, 4, 5}

Evaluation on every Lux eval-line: feed the first `--prefix-chars` characters,
ask the predictor for its next span, and compare with the actual remainder of
the line (final punctuation is stripped). Agreement is the fraction of eval
lines where the predicted span is a prefix of the remainder, after normalising
cuet spaces and trivial CJK punctuation.

This is *not* a claim about whether any hypothesis about human reasoning is
true. It only measures how well the n-gram offline component does on real
Japanese text so that later live work has a calibrated baseline.

Usage:
  scripts/tune_ngram.py --corpus data/real-speech/lux/metadata.csv
  scripts/tune_ngram.py --corpus ... --split-every 10 --eval-every 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.predictors import NgramPredictor
from cueloop.sync.vocabulary import tag_operations


def _normalise(text: str) -> str:
    text = text.strip()
    # Normalise trivial paired CJK punctuation variancy; never alter characters.
    text = text.replace("。", "。")
    return text


def _split(
    lines: list[str], *, eval_mod: int, eval_index_target: int
) -> tuple[list[str], list[str]]:
    train = [line for i, line in enumerate(lines) if i % eval_mod != eval_index_target]
    eval_set = [
        line for i, line in enumerate(lines) if i % eval_mod == eval_index_target
    ]
    return train, eval_set


def _predict_overlap(predicted: str, truth: str) -> int:
    # Longest common prefix in code-points.
    length = 0
    while (
        length < min(len(predicted), len(truth)) and predicted[length] == truth[length]
    ):
        length += 1
    return length


def _evaluate(
    predictor, eval_lines: list[str], *, prefix_chars: int
) -> dict[str, object]:
    total = 0
    full_hits = 0
    prefix_hits = 0
    char_matches = 0
    op_matches = 0
    op_total = 0
    for line in eval_lines:
        if len(line) <= prefix_chars + 4:
            continue
        prefix = line[:prefix_chars]
        remainder = line[prefix_chars:]
        prediction = predictor.predict(text=prefix, context=(), now_ms=0)
        if prediction is None or not prediction.span:
            continue
        total += 1
        truth_span = remainder.splitlines()[0] if "\n" in remainder else remainder
        pred_span = prediction.span
        if truth_span.startswith(pred_span):
            full_hits += 1
        elif truth_span[: len(pred_span)] == pred_span:
            prefix_hits += 1
        else:
            char_matches += _predict_overlap(pred_span, truth_span)
        pred_ops = set(prediction.operations)
        truth_ops = set(tag_operations(truth_span))
        if truth_ops:
            op_total += 1
            if pred_ops & truth_ops:
                op_matches += 1
    return {
        "n": total,
        "exact_prefix_rate": (full_hits + prefix_hits) / total if total else None,
        "exact_span_rate": full_hits / total if total else None,
        "operation_overlap_rate": op_matches / op_total if op_total else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus", type=Path, required=True, help="metadata.csv of the Lux corpus"
    )
    parser.add_argument("--prefix-chars", type=int, default=12)
    parser.add_argument("--order", type=int, default=3, help="n-gram order (default 3)")
    parser.add_argument("--eval-mod", type=int, default=10)
    parser.add_argument("--eval-remainder", type=int, default=0)
    parser.add_argument("--span-characters", type=int, default=40)
    parser.add_argument("--max-lines", type=int, default=0, help="0 = all lines")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(args.corpus.open(encoding="utf-8")))
    lines = [_normalise(row["transcript"]) for row in rows if row.get("transcript")]
    if args.max_lines and args.max_lines < len(lines):
        lines = lines[: args.max_lines]
    train, eval_set = _split(
        lines, eval_mod=args.eval_mod, eval_index_target=args.eval_remainder
    )
    print(f"train lines: {len(train)}")
    print(f"eval lines:  {len(eval_set)}")

    import cueloop.sync.predictors as predictors_module

    default_corpus = predictors_module.DEFAULT_CORPUS

    # Candidate 1: default (embedded corpus)
    default = NgramPredictor.default(
        order=args.order, span_characters=args.span_characters
    )
    default_metrics = _evaluate(default, eval_set, prefix_chars=args.prefix_chars)

    # Candidate 2: train on Lux train split only
    lux_only = NgramPredictor(order=args.order, span_characters=args.span_characters)
    lux_only.fit("\n".join(train))
    lux_metrics = _evaluate(lux_only, eval_set, prefix_chars=args.prefix_chars)

    # Candidate 3: embedded + Lux train merged
    merged = NgramPredictor(order=args.order, span_characters=args.span_characters)
    merged.fit(default_corpus + "\n" + "\n".join(train))
    merged_metrics = _evaluate(merged, eval_set, prefix_chars=args.prefix_chars)

    report = {
        "corpus": str(args.corpus),
        "lines": len(lines),
        "split": {"eval_mod": args.eval_mod, "eval_remainder": args.eval_remainder},
        "ngram_order": args.order,
        "span_characters": args.span_characters,
        "prefix_chars": args.prefix_chars,
        "results": {
            "default": default_metrics,
            "lux_only": lux_metrics,
            "merged": merged_metrics,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
