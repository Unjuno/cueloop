#!/usr/bin/env python3
"""DSPy-based prompt optimization for CUELOOP stem cues.

This applies DSPy (the user's "dspy"/"DS-Pi" reference) to automatically
optimize the stem-cue prompt, instead of hand-tuning wording. A Signature
declares the task, a metric encodes "natural + attention-grounded", and
BootstrapFewShot compiles demos from the labeled cases.

Baseline  : uncompiled DSPy module (zero-shot Signature).
Candidate : compiled module (BootstrapFewShot demos).
Target    : >= +20 points on the metric vs baseline, on a held-out split.

Run with a local Ollama model; no microphone needed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import dspy

ROOT = Path(__file__).resolve().parents[1]

# Labeled cases: context + utterance + predicted span + the term being circled.
CASES = [
    {
        "context": "まず前提を確認したい",
        "text": "この設計は前提を分けて考える。でも逆の立場だと",
        "span": "別の結論になる。",
        "attended_term": "前提",
    },
    {
        "context": "コストの話をしていた",
        "text": "この施策は効果があると思う。なぜなら",
        "span": "コストが下がるから。",
        "attended_term": "コスト",
    },
    {
        "context": "抽象的な議論が続いている",
        "text": "もっと具体的に言うと、例えば",
        "span": "三段階で進める形です。",
        "attended_term": "三段階",
    },
    {
        "context": "なぜ起きるかを聞かれた",
        "text": "この現象の仕組みは、どうして起きるかというと",
        "span": "入力が変わるためです。",
        "attended_term": "入力",
    },
    {
        "context": "結論を急がれていた",
        "text": "ここまでの議論をまとめると、つまり",
        "span": "やはり再設計が必要です。",
        "attended_term": "再設計",
    },
    {
        "context": "前提が曖昧だと話が進まない",
        "text": "まず前提をはっきりさせたい。根拠は",
        "span": "観測データが足りない。",
        "attended_term": "前提",
    },
    {
        "context": "境界線を議論していた",
        "text": "このルールの範囲は、どこまで及ぶのか",
        "span": "限定listedだ。",
        "attended_term": "範囲",
    },
    {
        "context": "比較対象を聞かれた",
        "text": "この方式とあれを比べると、違いは",
        "span": "コスト差が大きい。",
        "attended_term": "違い",
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


class StemCueSignature(dspy.Signature):
    """Write one natural Japanese cue engaging the term the speaker is circling."""

    context: str = dspy.InputField(desc="直近の文脈")
    text: str = dspy.InputField(desc="いまの発話")
    span: str = dspy.InputField(desc="続きの予測（参考。そのまま書かない）")
    attended_term: str = dspy.InputField(desc="必ず使うべき注目語")
    cue: str = dspy.OutputField(
        desc="48文字以内の自然な日本語一文。句点は文末に一つだけ"
    )


class StemCueModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.write = dspy.Predict(StemCueSignature)

    def forward(self, context: str, text: str, span: str, attended_term: str):
        return self.write(
            context=context, text=text, span=span, attended_term=attended_term
        )


def metric(example, pred, trace=None) -> float:
    cue = (pred.cue or "").strip()
    if not cue:
        return 0.0
    score = 0.0
    if example.attended_term in cue:
        score += 0.5
    if not any(w in cue for w in OP_LEAK_WORDS):
        score += 0.25
    if cue.count("。") == 1 and cue.endswith("。"):
        score += 0.25
    return score


def to_example(case: dict) -> dspy.Example:
    return dspy.Example(
        context=case["context"],
        text=case["text"],
        span=case["span"],
        attended_term=case["attended_term"],
        cue=case.get("cue", ""),
    ).with_inputs("context", "text", "span", "attended_term")


def avg_score(module, examples) -> float:
    scores = []
    for ex in examples:
        try:
            pred = module(
                context=ex.context,
                text=ex.text,
                span=ex.span,
                attended_term=ex.attended_term,
            )
        except (ValueError, RuntimeError, TypeError, KeyError):
            scores.append(0.0)
            continue
        scores.append(metric(ex, pred))
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gemma3:4b-it-qat")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--save", type=Path, help="Save the compiled DSPy program.")
    args = parser.parse_args()

    lm = dspy.LM(
        args.model,
        api_base=args.api_base,
        api_key="ollama",
        max_tokens=160,
        temperature=0.0,
        cache=False,
    )
    dspy.configure(lm=lm)

    rng = random.Random(0)
    cases = list(CASES)
    rng.shuffle(cases)
    split = max(2, len(cases) // 4)
    train, test = cases[split:], cases[:split]
    train_ex = [to_example(c) for c in train]
    test_ex = [to_example(c) for c in test]

    student = StemCueModule()
    baseline = avg_score(student, test_ex)

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1,
        max_errors=20,
    )
    compiled = optimizer.compile(student=student, trainset=train_ex)
    optimized = avg_score(compiled, test_ex)

    target_delta = 0.20
    delta = optimized - baseline
    verdict = "PASS" if delta >= target_delta else "FAIL"
    print("=== DSPy stem-cue prompt optimization ===")
    print(f"model={args.model}")
    print(f"train cases={len(train_ex)} held-out cases={len(test_ex)}")
    print(f"baseline (zero-shot) metric : {baseline:.2f}")
    print(f"optimized (compiled) metric: {optimized:.2f}")
    print(f"target >= +{target_delta:.2f}; actual {delta:+.2f} -> {verdict}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(args.save))
        print(f"saved compiled program: {args.save}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "baseline": baseline,
                    "optimized": optimized,
                    "delta": delta,
                    "target_delta": target_delta,
                    "verdict": verdict,
                    "n_train": len(train_ex),
                    "n_test": len(test_ex),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
