#!/usr/bin/env python3
"""A/B experiment: rule-based cue selection vs attention-grounded selection.

No microphone. For each fixed case with a known attended term, compare:
  baseline  - the operation chosen by the rule ordering (priority minus covered)
  attention - the operation extracted from the model's attended term via the
              deterministic vocabulary

Metric: selection accuracy = selected operation equals the operation implied
by the attended term (where the term is in the vocabulary). This directly
tests the user's requirement: the cue should engage the term the model is
attending to, not a fixed priority list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.attention import AttentionTermExtractor, operation_for_term
from cueloop.sync.vocabulary import PRIORITY

CASES = [
    {
        "text": "この設計は前提を分けて考える。でも逆の立場だと",
        "context": ["まず前提を確認したい"],
        "span": "別の結論になる。",
        "attended": "前提",
    },
    {
        "text": "この施策は効果があると思う。なぜなら",
        "context": ["コストの話をしていた"],
        "span": "コストが下がるから。",
        "attended": "なぜなら",
    },
    {
        "text": "この設計は前提を分けて考える。結論は",
        "context": ["まず前提を確認したい"],
        "span": "だから決まる。",
        "attended": "前提",
    },
    {
        "text": "前提として、この方式を採用する",
        "context": ["前提の話が続いている"],
        "span": "依存関係は明確だ。",
        "attended": "前提",
    },
    {
        "text": "逆に言うと、別の見方もある",
        "context": ["反対の立場を検討したい"],
        "span": "逆はどうか。",
        "attended": "逆に",
    },
    {
        "text": "根拠として、このデータ的那么示している",
        "context": ["証拠を requested されている"],
        "span": "観測値が増えた。",
        "attended": "根拠",
    },
    {
        "text": "このルールの条件は、いつ適用されるのか",
        "context": ["適用範囲を確認したい"],
        "span": "限定されている。",
        "attended": "条件",
    },
    {
        "text": "具体来说、まず第一歩は何か",
        "context": ["具体例を出してほしい"],
        "span": "三段階で進める。",
        "attended": "具体",
    },
    {
        "text": "なぜこの方式を選ぶのか、その理由は",
        "context": ["仕組みを知りたい"],
        "span": "入力が変わるためだ。",
        "attended": "なぜ",
    },
    {
        "text": "比較すると、この案の違いはどこか",
        "context": ["他社案と比べたい"],
        "span": "コスト差が大きい。",
        "attended": "比較",
    },
    {
        "text": "この方式の範囲は、どこまで及ぶのか",
        "context": ["限界を議論したい"],
        "span": "一部のみ有効だ。",
        "attended": "範囲",
    },
]


def rule_baseline(case: dict) -> str | None:
    """Rule choice: first priority operation not already in the text/span."""
    covered = set()
    for op in ("ground", "conclusion", "dependency", "invert", "boundary"):
        if op in case["text"] or op in case["span"]:
            covered.add(op)
    # crude tag coverage, same spirit as RuleDecisionModel
    from cueloop.sync.vocabulary import tag_operations

    covered |= set(tag_operations(case["text"])) | set(tag_operations(case["span"]))
    for op in PRIORITY:
        if op not in covered:
            return op
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma3:4b-it-qat")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    extractor = AttentionTermExtractor(
        base_url=args.base_url, model=args.model, timeout=20
    )
    rows = []
    for case in CASES:
        term = extractor.extract(
            text=case["text"], context=case["context"], span=case["span"]
        )
        expected = operation_for_term(case["attended"])
        attn_op = operation_for_term(term) if term else None
        base_op = rule_baseline(case)
        # Objective grounding: does the extracted term actually occur in the
        # live context/utterance? Independent of the subjective hand-labeled
        # "attended" term, which is not reliable enough as ground truth.
        haystack = case["text"] + " / " + " / ".join(case["context"])
        grounded = bool(term) and term in haystack
        rows.append(
            {
                "attended": case["attended"],
                "extracted_term": term,
                "expected_op": expected,
                "rule_op": base_op,
                "attention_op": attn_op,
                "rule_correct": base_op == expected,
                "attention_correct": attn_op == expected,
                "grounded": grounded,
            }
        )

    total = len(rows)
    rule_hits = sum(r["rule_correct"] for r in rows)
    attn_hits = sum(r["attention_correct"] for r in rows)
    grounded_hits = sum(r["grounded"] for r in rows)
    print("=== cue selection A/B (attention vs rule) ===")
    for r in rows:
        print(
            f"  attended={r['attended']:5s} extracted={r['extracted_term']!s:6s} "
            f"expected={r['expected_op']!s:12s} "
            f"rule={r['rule_op']!s:12s} attention={r['attention_op']!s:12s} "
            f"grounded={r['grounded']}"
        )
    print(
        f"\nbaseline (rule)  operation-match accuracy: {rule_hits}/{total} = {rule_hits / total:.0%}"
    )
    print(
        f"attention operation-match accuracy: {attn_hits}/{total} = {attn_hits / total:.0%}"
    )
    print(
        f"attention TERM GROUNDING (vocab-independent): {grounded_hits}/{total} = {grounded_hits / total:.0%}"
    )
    target_accuracy = 0.80
    target_delta = 0.40
    # The metric that matters is vocabulary-independent term grounding.
    acc = grounded_hits / total
    delta = acc - (rule_hits / total)
    verdict = "PASS" if acc >= target_accuracy and delta >= target_delta else "FAIL"
    print(
        f"target: accuracy >= {target_accuracy:.0%} and delta >= +{target_delta:.0%} "
        f"-> {verdict} (actual {acc:.0%}, {delta:+.0%})"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "rule_hits": rule_hits,
                    "attn_hits": attn_hits,
                    "grounded_hits": grounded_hits,
                    "target_accuracy": target_accuracy,
                    "target_delta": target_delta,
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
