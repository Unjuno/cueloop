#!/usr/bin/env python3
"""Local-LLM next-span agreement on real Japanese speech, one command.

Workflow: split a corpus sentence at a clause-ending boundary, ask each
model to continue it, and count exact-prefix matches and operation-tag
overlap against the actual transcription. Mirrors what
`LocalLLMPredictor.predict` does on the live path.

This is a benchmark of prediction *capacity*, not of whether cues change a
speaker's reasoning. The result tells you which local model (if any) buys
you signal on real Japanese discourse — for CUELOOP workload that is what
determines what mechanisms can be validated offline.

Usage:
  scripts/bench_predictors.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
      --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
      --models gemma3:4b-it-qat qwen2.5:7b-instruct qwen3:8b --samples 24
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
from cueloop.sync.synchronizer import longest_common_prefix
from cueloop.sync.vocabulary import tag_operations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--whisper-cache", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--boundary-chars",
        default="。、，．！？!?",
        help="clause-ending characters to split on",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    cache = json.loads(args.whisper_cache.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.corpus.open(encoding="utf-8")))
    tasks: list[tuple[str, str, str]] = []
    for row in rows:
        segments = cache.get(row["file_name"], {}).get("segments") or []
        text = "".join(s.get("text", "") for s in segments)
        for index, ch in enumerate(text):
            if ch in args.boundary_chars and index >= 3 and len(text) - index - 1 >= 6:
                tasks.append((row["file_name"], text[: index + 1], text[index + 1 :]))
                break
    stride = max(1, len(tasks) // args.samples)
    tasks = tasks[::stride][: args.samples]
    print(f"tasks: {len(tasks)}")

    summary: dict[str, dict[str, float]] = {}
    for model in args.models:
        predictor = LocalLLMPredictor(
            base_url=args.base_url,
            model=model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        predictor.warmup()
        exact = 0
        prefix_total = 0
        op_offered = 0
        op_hits = 0
        failures = 0
        for file_name, prefix, truth in tasks:
            prediction = predictor.predict(text=prefix, context=(), now_ms=0)
            if prediction is None:
                failures += 1
                continue
            if truth.startswith(prediction.span):
                exact += 1
            prefix_total += longest_common_prefix(truth, prediction.span)
            truth_ops = tag_operations(truth)
            pred_ops = tag_operations(prediction.span)
            if pred_ops:
                op_offered += 1
                if any(op in truth_ops for op in pred_ops):
                    op_hits += 1
        valid = len(tasks) - failures
        avg_lcp = prefix_total / valid if valid else 0.0
        summary[model] = {
            "exact": float(exact),
            "avg_lcp": avg_lcp,
            "op_offered": float(op_offered),
            "op_hits": float(op_hits),
            "valid": float(valid),
            "failures": float(failures),
        }
        print(
            f"model={model} exact={exact}/{valid} avg_lcp={avg_lcp:.2f} "
            f"op_offered={op_offered} op_hits={op_hits} failures={failures}"
        )
    out = Path("artifacts/bench-predictors.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
