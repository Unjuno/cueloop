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
        rows.append(
            {
                "attended": case["attended"],
                "extracted_term": term,
                "expected_op": expected,
                "rule_op": base_op,
                "attention_op": attn_op,
                "rule_correct": base_op == expected,
                "attention_correct": attn_op == expected,
            }
        )

    total = len(rows)
    rule_hits = sum(r["rule_correct"] for r in rows)
    attn_hits = sum(r["attention_correct"] for r in rows)
    print("=== cue selection A/B (attention vs rule) ===")
    for r in rows:
        print(
            f"  attended={r['attended']:5s} extracted={r['extracted_term']!s:6s} "
            f"expected={r['expected_op']!s:12s} "
            f"rule={r['rule_op']!s:12s} attention={r['attention_op']!s:12s}"
        )
    print(
        f"\nbaseline (rule)  selection accuracy: {rule_hits}/{total} = {rule_hits / total:.0%}"
    )
    print(
        f"attention-grounded selection accuracy: {attn_hits}/{total} = {attn_hits / total:.0%}"
    )
    print(
        "verdict: "
        + (
            "attention improves selection"
            if attn_hits > rule_hits
            else "attention does not improve selection on these cases"
        )
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"rows": rows, "rule_hits": rule_hits, "attn_hits": attn_hits},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
