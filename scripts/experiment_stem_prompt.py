#!/usr/bin/env python3
"""A/B experiment for stem-cue prompts (no microphone required).

Baseline  : current build_stem_messages (operation-first instruction).
Candidate : attention-grounded 2-step prompt (identify the attended term in
            the live context, then write one natural initial-condition cue
            that engages THAT term, without forcing an operation word).

Runs both prompts over a fixed set of realistic Japanese reasoning contexts
using the local LLM, and reports:
  - operation-word leakage rate (e.g. 逆に/根拠は/具体は forced into the cue)
  - attended-term uptake (does the cue actually use a word from the context?)
  - mean length / validity

The goal is not a subjective score: we measure whether the candidate reduces
forced operation words while keeping (or raising) context-term uptake.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.stem import (
    StemCueGenerator,
    build_stem_messages,
    finalize_stem,
)

# Fixed, realistic reasoning contexts with a known "attended term" — the word
# the speaker is circling and that a good cue should engage.
CASES = [
    {
        "operation": "invert",
        "text": "この設計は前提を分けて考える。でも逆の立場だと",
        "span": "別の結論になるかもしれない。",
        "context": ["まず前提を確認したい", "この設計は前提を分けて考える"],
        "attended": "前提",
    },
    {
        "operation": "ground",
        "text": "この施策は効果があると思う。なぜなら",
        "span": "コストが下がるからです。",
        "context": ["コストの話をしていた", "この施策は効果があると思う"],
        "attended": "コスト",
    },
    {
        "operation": "concretize",
        "text": "もっと具体的に言うと、例えば",
        "span": "三段階で進める形です。",
        "context": ["抽象的な議論が続いている", "もっと具体的に言うと"],
        "attended": "三段階",
    },
    {
        "operation": "mechanism",
        "text": "この現象の仕組みは、どうして起きるかというと",
        "span": "入力が変わるためです。",
        "context": ["なぜ起きるかを聞かれた", "この現象の仕組みは"],
        "attended": "入力",
    },
    {
        "operation": "conclusion",
        "text": "ここまでの議論をまとめると、つまり",
        "span": "やはり再設計が必要です。",
        "context": ["結論を急がれていた", "ここまでの議論をまとめると"],
        "attended": "再設計",
    },
]

# Operation words that the baseline prompt tends to force into the cue.
OP_LEAK_WORDS = ("逆に", "逆は", "根拠は", "具体は", "例えば", "仕組みは", "なぜなら", "つまり", "結論は")


def build_attention_messages(
    operation: str,
    *,
    text: str,
    span: str,
    context: list[str],
    max_characters: int,
    version: int = 1,
) -> tuple[str, str]:
    """Candidate: attention-grounded, operation-agnostic prompt."""
    recent = " / ".join(context[-3:]) or "なし"
    system = (
        "あなたは日本語の話し手を支援するファシリテーターです。"
        "話し手が今まさに考えている「引っかかっている語」を捉え、"
        "その語を自然に次の文へ繋ぐ一文だけを書きます。"
        "出力は一文のみで、説明・引用符・箇条書き・ラベルを含めません。"
    )
    if version == 1:
        user = (
            f"【直前の文脈】\n{recent}\n\n"
            f"【今の発話】\n{text}\n\n"
            f"【予測される続き】\n{span}\n\n"
            "手順:\n"
            "1. 今の発話と文脈で、話し手が最も引っかかっている語（未解決な概念）を"
            "頭の中で特定してください（出力しません）。\n"
            "2. その語を自然に含む、話し手がそのまま続けられる一文を書いてください。\n"
            "3. 「逆に」「根拠は」「具体は」などの定型ラベルは使わず、"
            "文脈の語そのもので始めてください。\n\n"
            f"条件: {max_characters}文字以内、自然な日本語、一文のみ、"
            "事実を断定しすぎず推量・限定を含めてよい。\n"
            "文:\n"
        )
    elif version == 2:
        user = (
            f"【文脈】{recent}\n"
            f"【発話】{text}\n"
            f"【続きの予測（参考情報。そのまま書かないでください）】{span}\n\n"
            "今いちばん引っかかっている語を決め、それを"
            "新しい言い回しで含む一文だけを書いてください。\n"
            "絶対に守ること:\n"
            "- 句点（。）は文末に1つだけ。途中と末尾に句点を打たない。\n"
            "- 予測された続きをそのまま書き写さない。別の言葉に言い換える。\n"
            "- 「逆に」「根拠は」「具体は」などの定型ラベルは使わない。\n"
            "- 文脈の重要語そのものを入れる。\n"
            f"- {max_characters}文字以内。\n"
            "文:\n"
        )
    elif version == 3:
        user = (
            f"【文脈】{recent}\n"
            f"【発話】{text}\n"
            f"【続きのヒント】{span}\n\n"
            "話し手が今いちばん引っかかっている語を、文脈の中から一つ選び、"
            "その語を自然に使って、話し手が続きに言えそうな一文を書いてください。\n"
            "文末に句点を一つだけ付け、それ以外の句点は使いません。"
            f"{max_characters}文字以内の一文です。\n"
            "文:\n"
        )
    else:
        # v4: combine v1's "start from the attended term, no operation label"
        # with v3's "prediction is only a hint" (fixes v1's two-sentence leak)
        # and a single trailing period. No forbidden-word list (v2 failed).
        user = (
            f"【直前の文脈】\n{recent}\n\n"
            f"【今の発話】\n{text}\n\n"
            f"【続きのヒント（参考のみ。書き写さない）】\n{span}\n\n"
            "手順:\n"
            "1. 発話と文脈の中で、話し手が最も引っかかっている語を一つ選んでください。\n"
            "2. その語を文の始め近くに置き、話し手がそのまま続けられる"
            "自然な一文を書いてください。\n"
            "3. ヒントは言い換えの材料にだけ使い、そのまま書き写さないでください。\n\n"
            f"条件: {max_characters}文字以内の一文。句点は文末に一つだけ。"
            "定型ラベルや見出しは使わず、文脈の語そのもので始めます。\n"
            "文:\n"
        )
    return system, user


def complete(gen: StemCueGenerator, system: str, user: str) -> str | None:
    try:
        content = gen._complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
    except RuntimeError:
        return None
    return finalize_stem(content, max_characters=gen.max_characters)


def _multi_sentence(cue: str) -> bool:
    """True when a cue contains an interior sentence boundary."""
    return cue.count("。") > 1 or "。" in cue[:-1]


def evaluate(gen: StemCueGenerator) -> dict:
    results: dict[str, dict] = {
        "baseline": {},
        "attention_v1": {},
        "attention_v2": {},
        "attention_v3": {},
        "attention_v4": {},
    }
    for name in results:
        leak = uptake = valid = multi = 0
        total = 0
        cues: list[str] = []
        for case in CASES:
            if name == "baseline":
                system, user = build_stem_messages(
                    case["operation"],
                    text=case["text"],
                    span=case["span"],
                    context=case["context"],
                    max_characters=gen.max_characters,
                )
            else:
                version = {
                    "attention_v1": 1,
                    "attention_v2": 2,
                    "attention_v3": 3,
                    "attention_v4": 4,
                }[name]
                system, user = build_attention_messages(
                    case["operation"],
                    text=case["text"],
                    span=case["span"],
                    context=case["context"],
                    max_characters=gen.max_characters,
                    version=version,
                )
            cue = complete(gen, system, user)
            total += 1
            if cue is None:
                continue
            valid += 1
            cues.append(cue)
            if any(w in cue for w in OP_LEAK_WORDS):
                leak += 1
            if case["attended"] in cue:
                uptake += 1
            if _multi_sentence(cue):
                multi += 1
        results[name] = {
            "valid": valid,
            "total": total,
            "op_leak_rate": leak / valid if valid else 0.0,
            "attended_uptake": uptake / valid if valid else 0.0,
            "multi_sentence_rate": multi / valid if valid else 0.0,
            "cues": cues,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # StemCueGenerator appends /v1/chat/completions internally, so pass the
    # server root, not the /v1 endpoint.
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    gen = StemCueGenerator(
        base_url=args.base_url,
        model=args.model,
        timeout=20.0,
        max_tokens=160,
        max_characters=48,
    )
    started = time.time()
    results = evaluate(gen)
    results["elapsed_s"] = round(time.time() - started, 1)
    results["model"] = args.model

    print("=== stem prompt experiment (baseline vs attention-grounded) ===")
    for name in (
        "baseline",
        "attention_v1",
        "attention_v2",
        "attention_v3",
        "attention_v4",
    ):
        r = results[name]
        print(
            f"{name:13s}: valid={r['valid']}/{r['total']} "
            f"op_leak={r['op_leak_rate']:.0%} "
            f"attended_uptake={r['attended_uptake']:.0%} "
            f"multi_sent={r['multi_sentence_rate']:.0%}"
        )
        for cue in r["cues"]:
            print(f"    - {cue}")
    b = results["baseline"]
    a = results["attention_v4"]
    print("\n--- verdict (baseline -> attention_v4) ---")
    print(
        f"op_leak: {b['op_leak_rate']:.0%} -> {a['op_leak_rate']:.0%} "
        f"({'better' if a['op_leak_rate'] < b['op_leak_rate'] else 'worse'})"
    )
    print(
        f"attended_uptake: {b['attended_uptake']:.0%} -> {a['attended_uptake']:.0%} "
        f"({'better' if a['attended_uptake'] > b['attended_uptake'] else 'worse'})"
    )
    print(
        f"multi_sentence: {b['multi_sentence_rate']:.0%} -> {a['multi_sentence_rate']:.0%} "
        f"({'better' if a['multi_sentence_rate'] < b['multi_sentence_rate'] else 'worse'})"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
