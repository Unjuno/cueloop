#!/usr/bin/env python3
"""Local-LLM continuation match on real Japanese sentences (Lux corpus).

The CUELOOP local predictor exists because the n-gram backup is useless on
real Japanese. This benchmark measures whether the local LLM actually
proposes a correct *continuation* of Japanese clauses the way the predictor
asks it to when running live.

Method: split each Lux sentence at '、' into (context, continuation).
Ask the model to write only the continuation. Score exact-prefix match
against the ground-truth continuation.

Usage:
  scripts/benchmark_llm_continuation.py --corpus data/real-speech/lux/metadata.csv \
      --model qwen2.5:7b-instruct --samples 20
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as handle:
        rows = [
            r for r in csv.DictReader(handle)
            if r["cleaned_path"].startswith("data/cleaned/ita_recitation/")
        ]
    pairs = []
    for r in rows:
        t = r["transcript"].strip()
        parts = t.split("、")
        if len(parts) < 2:
            continue
        context = parts[0] + "、"
        truth = "".join(parts[1:]).strip()
        if len(truth) >= 6:
            pairs.append((context, truth))
    pairs = pairs[::3][: args.samples]
    print(f"selected {len(pairs)} task pairs from {len(rows)} files")

    exact = 0
    prefix = 0
    valid = 0
    failures = 0
    total_words = 0
    for index, (context, truth) in enumerate(pairs):
        prompt = (
            "次の日本語の文を続けてください。続きだけを出力してください。\n"
            "説明・記号・改行など余分な文字は入れないこと。\n\n"
            f"{context}"
        )
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index}] REQUEST FAILED: {exc}")
            failures += 1
            continue
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        total_words += len(text)
        if not text:
            failures += 1
            continue
        valid += 1
        if truth.startswith(text):
            exact += 1
        else:
            shared = 0
            for a, b in zip(truth, text):
                if a != b:
                    break
                shared += 1
            if shared >= max(2, len(truth) // 3):
                prefix += 1
    print(json.dumps({
        "model": args.model,
        "corpus": args.corpus,
        "tasks": len(pairs),
        "scored": valid,
        "exact_prefix_rate": exact / valid if valid else None,
        "partial_prefix_rate": prefix / valid if valid else None,
        "failures": failures,
        "avg_output_chars": total_words / max(1, valid),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
