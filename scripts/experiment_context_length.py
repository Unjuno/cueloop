#!/usr/bin/env python3
"""Does more context make attention-grounded stem cues more natural?

Verifies the user's claim: "文脈が多ければ多いほどより自然になる".
For each case we vary the number of context turns (0/1/2/4) while keeping the
utterance and predicted span fixed, generate a v3 attention-grounded stem, and
measure:
  - attended-term uptake: does the cue use the term the speaker is circling?
  - op-word leak: forced operation labels (lower = more natural)
  - one-sentence rate
Baseline = 0 context turns. Target = >= +20 points uptake at 4 turns vs 0.
No microphone; uses the local LLM through StemCueGenerator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.stem import (
    StemCueGenerator,
)

CASES = [
    {
        "text": "この設計は前提を分けて考える。でも逆の立場だと",
        "span": "別の結論になるかもしれない。",
        "attended": "前提",
        "context": [
            "まず前提を確認したい",
            "この設計は前提を分けて考える",
            "前提が曖昧だと話が進まない",
            "逆の立場も前提次第だ",
        ],
    },
    {
        "text": "この施策は効果があると思う。なぜなら",
        "span": "コストが下がるからです。",
        "attended": "コスト",
        "context": [
            "コストの話をしていた",
            "この施策は効果があると思う",
            "コスト削減が狙いだ",
            "費用対効果を見ている",
        ],
    },
    {
        "text": "もっと具体的に言うと、例えば",
        "span": "三段階で進める形です。",
        "attended": "三段階",
        "context": [
            "抽象的な議論が続いている",
            "もっと具体的に言うと",
            "手順を整理したい",
            "三段階に分けたい",
        ],
    },
    {
        "text": "この現象の仕組みは、どうして起きるかというと",
        "span": "入力が変わるためです。",
        "attended": "入力",
        "context": [
            "なぜ起きるかを聞かれた",
            "この現象の仕組みは",
            "入力の変化が疑わしい",
            "入力次第で結果が変わる",
        ],
    },
    {
        "text": "ここまでの議論をまとめると、つまり",
        "span": "やはり再設計が必要です。",
        "attended": "再設計",
        "context": [
            "結論を急がれていた",
            "ここまでの議論をまとめると",
            "再設計が有力だ",
            "再設計の費用も検討中",
        ],
    },
]

OP_LEAK_WORDS = (
    "逆に",
    "逆は",
    "根拠は",
    "具体は",
    "例えば",
    "仕組みは",
    "なぜなら",
    "つまり",
    "結論は",
)


def generate(gen: StemCueGenerator, case: dict, n_context: int) -> str | None:
    context = case["context"][:n_context]
    # The attention extractor supplies the term; pass it explicitly so the
    # prompt uses it rather than the most recent phrase in the transcript.
    return gen.generate(
        "invert",
        text=case["text"],
        span=case["span"],
        context=context,
        attended_term=case["attended"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma3:4b-it-qat")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    gen = StemCueGenerator(
        base_url=args.base_url,
        model=args.model,
        timeout=20,
        max_tokens=160,
        max_characters=48,
    )
    lengths = [0, 1, 2, 4]
    summary: dict[int, dict] = {}
    details = []
    for n in lengths:
        uptake = leak = valid = 0
        for case in CASES:
            cue = generate(gen, case, n)
            ok = cue is not None
            valid += ok
            if not ok:
                details.append({"n_context": n, "case": case["attended"], "cue": None})
                continue
            hit = case["attended"] in cue
            uptake += hit
            leak += any(w in cue for w in OP_LEAK_WORDS)
            details.append(
                {"n_context": n, "case": case["attended"], "cue": cue, "uptake": hit}
            )
        total = len(CASES)
        summary[n] = {
            "attended_uptake": uptake / total,
            "op_leak_rate": leak / total if total else 0,
            "valid": valid,
        }

    print("=== context-length sweep (v3 attention-grounded stem) ===")
    for n in lengths:
        s = summary[n]
        print(
            f"  context={n}: attended_uptake={s['attended_uptake']:.0%} "
            f"op_leak={s['op_leak_rate']:.0%} valid={s['valid']}/{len(CASES)}"
        )
    base = summary[0]["attended_uptake"]
    top = summary[4]["attended_uptake"]
    target = 0.20
    verdict = "PASS (more context -> more natural)" if top - base >= target else "FAIL"
    print(f"\nbaseline (0 context) uptake={base:.0%}; at 4 context={top:.0%}")
    print(f"target: >= +{target:.0%} uptake -> {verdict} (actual {top - base:+.0%})")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "details": details,
                    "target_delta": target,
                    "verdict": verdict,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
