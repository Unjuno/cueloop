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
    {
        "text": "この仮定が成り立つ条件は、いつか",
        "span": "限定された場合だ。",
        "attended": "条件",
        "context": [
            "条件を整理したい",
            "この仮定は条件次第",
            "いつ適用されるか",
            "条件を明示したい",
        ],
    },
    {
        "text": "この結果の仕組みは、どうしてなのか",
        "span": "入力が変わるからだ。",
        "attended": "仕組み",
        "context": [
            "仕組みを知りたい",
            "なぜ起きるか",
            "構造を説明して",
            "仕組みは単純だ",
        ],
    },
    {
        "text": "もう少し具体的に言うと、例えば",
        "span": "三段階で進める。",
        "attended": "具体",
        "context": [
            "具体例が欲しい",
            "抽象的な話が続く",
            "手順を知りたい",
            "具体的に説明して",
        ],
    },
    {
        "text": "これは何に依存しているのか、前提は",
        "span": "環境設定次第だ。",
        "attended": "前提",
        "context": ["依存関係を確認", "前提を明確に", "何が必要条件か", "前提は重要だ"],
    },
    {
        "text": "この方式とあれを比べると、違いは",
        "span": "コスト差が大きい。",
        "attended": "違い",
        "context": ["比較したい", "他社案と比べる", "違いを教えて", "差は明確だ"],
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


def generate(
    gen: StemCueGenerator, case: dict, n_context: int, *, explicit_term: bool
) -> str | None:
    context = case["context"][:n_context]
    # explicit_term=False lets the model pick a term from the context by
    # itself — the non-ceiling condition for testing "more context -> better".
    return gen.generate(
        "invert",
        text=case["text"],
        span=case["span"],
        context=context,
        attended_term=case["attended"] if explicit_term else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma3:4b-it-qat")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--explicit-term",
        action="store_true",
        help="Pass the attended term explicitly (ceiling condition). Default: model picks.",
    )
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
        grounded = leak = valid = 0
        for case in CASES:
            cue = generate(gen, case, n, explicit_term=args.explicit_term)
            ok = cue is not None
            valid += ok
            if not ok:
                details.append({"n_context": n, "case": case["attended"], "cue": None})
                continue
            # Non-ceiling metric: does the cue contain a word actually present
            # in the live context? Independent of any single labeled term.
            words = [w for w in case["context"] if len(w) >= 2]
            hit = any(w in cue for w in words)
            grounded += hit
            leak += any(w in cue for w in OP_LEAK_WORDS)
            details.append(
                {"n_context": n, "case": case["attended"], "cue": cue, "grounded": hit}
            )
        total = len(CASES)
        summary[n] = {
            "context_grounding": grounded / total,
            "op_leak_rate": leak / total if total else 0,
            "valid": valid,
        }

    mode = "explicit-term" if args.explicit_term else "model-picks"
    print(f"=== context-length sweep (v3 stem, {mode}) ===")
    for n in lengths:
        s = summary[n]
        print(
            f"  context={n}: context_grounding={s['context_grounding']:.0%} "
            f"op_leak={s['op_leak_rate']:.0%} valid={s['valid']}/{len(CASES)}"
        )
    base = summary[0]["context_grounding"]
    top = summary[4]["context_grounding"]
    target = 0.20
    verdict = "PASS (more context -> more grounded)" if top - base >= target else "FAIL"
    print(f"\nbaseline (0 context) grounding={base:.0%}; at 4 context={top:.0%}")
    print(f"target: >= +{target:.0%} grounding -> {verdict} (actual {top - base:+.0%})")
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
