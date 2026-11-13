#!/usr/bin/env python3
"""Offline insertion benchmark on human Japanese reasoning transcripts.

Answers, without any live microphone:

* Prediction axis — when CUELOOP's production predictor (ngram(3), the
  default path) predicts a next span at a clause boundary in real Japanese
  text, do the operation tags in that span match the operation the human
  uses next?  (`op_hit_*` fields)
* Selection axis — when the production missing-operation machinery
  (`PolicyObservation.ranked_missing_operations`, used by the rule decision)
  picks the operation to insert at a boundary, does the human continuation
  actually carry it?  (`sel_hit_*` fields). Three selectors are compared:
  the context-fit ranking (production default, `sel_*_ranked`), a
  deterministic adjacency chain that extends the speaker's last move via
  CONTEXT_LINKS (`sel_*_extendfirst`), and context-blind PRIORITY
  (`sel_*_priority`), two signal-gated selectors (`sel_*_clausegated`:
  ranking only when the clause itself carries an op, else PRIORITY;
  `sel_*_weightgated`: ranking only when the weighted anchor signal in
  prefix ≥ 2.0, else PRIORITY), and an oracle that credits any op inside
  the missing set (`sel_oracle`). The selection block additionally
  reports an anchor decomposition (`sel_anchor_*`: boundaries where at
  least one anchor op — clause/recent/predicted — exists in the prefix,
  split ranked vs PRIORITY; `sel_noanchor_*`: the remainder, where
  production already falls back to PRIORITY) so the signal source of
  each selector is identifiable offline.

This is the *insertion-appropriateness* axis: it does not measure whether a
live cue changes a speaker's real-time reasoning (that needs paired live
sessions, see docs/self-experiment.md); it measures whether insertion alone,
in isolation from feedback, lands where humans actually go next — offline,
deterministically, on closed data.

Usage:
  scripts/benchmark_insertion.py \\
      --corpus data/real-speech/lux/metadata.csv \\
      --corpus data/real-speech/reazon/metadata-sample-200.csv \\
      --json artifacts/insertion-benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.policy_model import PolicyObservation
from cueloop.sync.predictors import NgramPredictor
from cueloop.sync.synchronizer import longest_common_prefix
from cueloop.sync.vocabulary import CONTEXT_LINKS, TOKEN_OPERATIONS, tag_operations

CLAUSE_LIMIT = 3  # rolling window of prior clauses used for `recent_operations`


def last_tagged_op(text: str) -> str | None:
    """The operation whose token occurs latest in `text` (the live move)."""
    best_pos, best_op = -1, None
    for token, operation in TOKEN_OPERATIONS:
        pos = text.rfind(token)
        if pos >= 0 and pos >= best_pos:
            best_pos, best_op = pos, operation
    return best_op


def extend_first_pick(
    clause_text: str, recent_clauses: deque[str], missing: tuple[str, ...]
) -> str:
    """Adjacency-chain selection: extend the speaker's last move if possible.

    Alternative to the score-based ranking: take the most recent tagged
    operation in the clause (falling back to recent clauses), then pick its
    first CONTEXT_LINKS neighbour that is still missing; otherwise fall back
    to PRIORITY. No weights, no learned scores.
    """
    anchor = last_tagged_op(clause_text)
    if anchor is None:
        for prior in reversed(tuple(recent_clauses)):
            anchor = last_tagged_op(prior)
            if anchor:
                break
    if anchor:
        for link in CONTEXT_LINKS.get(anchor, ()):  # ordered by strength
            if link in missing:
                return link
    return missing[0]


def truth_ops_next_clause(
    text: str, boundary_index: int, truth_window: int
) -> list[str]:
    """Operations tagged in the clause starting after `boundary_index`.

    Only the text up to the next clause boundary (or `truth_window` chars) is
    considered: that is the exact span CUELOOP would try to influence next.
    """
    rest = text[boundary_index + 1 : boundary_index + 1 + truth_window]
    for cut, ch in enumerate(rest):
        if ch in "。、，．！？!?":
            rest = rest[:cut]
            break
    return tag_operations(rest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        required=True,
        help="CSV with a `transcript` column; repeat for each corpus",
    )
    parser.add_argument(
        "--truth-window",
        type=int,
        default=40,
        help="characters after the boundary used to tag truth operations",
    )
    parser.add_argument(
        "--max-boundaries-per-file",
        type=int,
        default=0,
        help="cap boundaries per transcript (0 = all)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write machine-readable summary"
    )
    args = parser.parse_args()

    predictor = NgramPredictor.default()

    summary: dict[str, dict[str, object]] = {}
    for corpus_path in args.corpus:
        label = (
            f"{corpus_path.parent.name}/{corpus_path.name}"
            if (
                corpus_path.parent.name not in ("", ".")
                and corpus_path.name == "metadata.csv"
            )
            else corpus_path.name
        )
        rows = list(csv.DictReader(corpus_path.open(encoding="utf-8")))
        boundaries = 0
        predicted = 0
        offered = 0
        hits = 0
        truth_ops_present = 0
        lcp_total = 0
        conf_total = 0.0
        op_offers: Counter[str] = Counter()
        op_hits: Counter[str] = Counter()

        sel_boundaries = 0
        sel_truth_present = 0
        sel_hit_ranked = 0
        sel_hit_priority = 0
        sel_hit_extendfirst = 0
        sel_oracle = 0
        sel_missing_total = 0
        sel_anchor_boundaries = 0
        sel_anchor_hit_ranked = 0
        sel_anchor_hit_priority = 0
        sel_noanchor_boundaries = 0
        sel_noanchor_hit_ranked = 0
        sel_hit_clausegated = 0
        sel_hit_weightgated = 0

        for row in rows:
            text = (row.get("transcript") or "").strip()
            if not text:
                continue
            file_boundaries = 0
            clause_start = 0
            recent_clauses: deque[str] = deque(maxlen=CLAUSE_LIMIT)
            for index, ch in enumerate(text):
                if ch not in "。、，．！？!?":
                    continue
                if index < 3 or len(text) - index - 1 < 6:
                    continue
                if (
                    args.max_boundaries_per_file
                    and file_boundaries >= args.max_boundaries_per_file
                ):
                    break
                prefix, truth = (
                    text[: index + 1],
                    text[index + 1 : index + 1 + args.truth_window],
                )
                clause_text = text[clause_start : index + 1]
                clause_start = index + 1
                file_boundaries += 1
                boundaries += 1

                prediction = predictor.predict(text=prefix, context=(), now_ms=0)

                # --- prediction (span-op) axis ---
                if prediction is not None:
                    predicted += 1
                    lcp_total += longest_common_prefix(truth, prediction.span)
                    conf_total += prediction.confidence
                    pred_ops = tuple(dict.fromkeys(prediction.operations))
                    if pred_ops:
                        offered += 1
                        truth_ops = set(tag_operations(truth))
                        if truth_ops:
                            truth_ops_present += 1
                        per_op_hit = False
                        for op in pred_ops:
                            op_offers[op] += 1
                            if op in truth_ops:
                                op_hits[op] += 1
                                per_op_hit = True
                        if per_op_hit:
                            hits += 1

                # --- selection (missing-op) axis ---
                next_clause_ops = frozenset(
                    truth_ops_next_clause(text, index, args.truth_window)
                )
                clause_ops = frozenset(tag_operations(clause_text))
                recent_ops = frozenset(
                    op for prior in recent_clauses for op in tag_operations(prior)
                )
                pred_ops_frozen = (
                    frozenset(prediction.operations) if prediction else frozenset()
                )
                pred_conf = prediction.confidence if prediction else 0.0
                pred_len = len(prediction.span) if prediction else 0
                obs = PolicyObservation(
                    text_length=len(prefix),
                    context_turns=len(recent_clauses),
                    chars_per_second=1.0,
                    prediction_confidence=pred_conf,
                    prediction_length=pred_len,
                    prediction_operations=pred_ops_frozen,
                    clause_operations=clause_ops,
                    recent_operations=recent_ops,
                    at_boundary=True,
                    diverged=False,
                )
                missing = obs.missing_operations()
                if missing:
                    sel_boundaries += 1
                    sel_missing_total += len(missing)
                    ranked = obs.ranked_missing_operations()
                    extend_pick = extend_first_pick(
                        clause_text, recent_clauses, missing
                    )
                    if next_clause_ops:
                        sel_truth_present += 1
                        if ranked[0] in next_clause_ops:
                            sel_hit_ranked += 1
                        if missing[0] in next_clause_ops:
                            sel_hit_priority += 1
                        if extend_pick in next_clause_ops:
                            sel_hit_extendfirst += 1
                        if any(op in next_clause_ops for op in missing):
                            sel_oracle += 1
                        # gated selectors: signal-strength fallback to PRIORITY
                        anchor_weight = (
                            1.5 * len(clause_ops)
                            + 1.0 * len(recent_ops)
                            + 1.0 * len(pred_ops_frozen)
                        )
                        clausegated = ranked[0] if clause_ops else missing[0]
                        weightgated = ranked[0] if anchor_weight >= 2.0 else missing[0]
                        if clausegated in next_clause_ops:
                            sel_hit_clausegated += 1
                        if weightgated in next_clause_ops:
                            sel_hit_weightgated += 1
                        has_anchor = bool(clause_ops or recent_ops or pred_ops_frozen)
                        if has_anchor:
                            sel_anchor_boundaries += 1
                            if ranked[0] in next_clause_ops:
                                sel_anchor_hit_ranked += 1
                            if missing[0] in next_clause_ops:
                                sel_anchor_hit_priority += 1
                        else:
                            sel_noanchor_boundaries += 1
                            if ranked[0] in next_clause_ops:
                                sel_noanchor_hit_ranked += 1

                recent_clauses.append(clause_text)

        denom = offered or 1
        sel_denom = sel_truth_present or 1
        summary[label] = {
            "corpus": str(corpus_path),
            "rows": len(rows),
            "boundaries": boundaries,
            "predictions": predicted,
            "prediction_rate": predicted / boundaries if boundaries else 0.0,
            "offered": offered,
            "offer_rate": offered / predicted if predicted else 0.0,
            "op_hits": hits,
            "op_hit_rate_offered": hits / denom,
            "op_hit_rate_boundaries": hits / boundaries if boundaries else 0.0,
            "truth_op_boundaries": truth_ops_present,
            "op_hit_rate_given_truth_op": hits / truth_ops_present
            if truth_ops_present
            else None,
            "avg_lcp": lcp_total / predicted if predicted else 0.0,
            "avg_confidence": conf_total / predicted if predicted else 0.0,
            "op_offers": dict(op_offers),
            "op_hits_by_op": dict(op_hits),
            "sel_boundaries": sel_boundaries,
            "sel_avg_missing": sel_missing_total / sel_boundaries
            if sel_boundaries
            else 0.0,
            "sel_truth_present": sel_truth_present,
            "sel_hit_ranked": sel_hit_ranked,
            "sel_hit_priority": sel_hit_priority,
            "sel_hit_extendfirst": sel_hit_extendfirst,
            "sel_hit_rate_ranked": sel_hit_ranked / sel_denom,
            "sel_hit_rate_priority": sel_hit_priority / sel_denom,
            "sel_hit_rate_extendfirst": sel_hit_extendfirst / sel_denom,
            "sel_hit_clausegated": sel_hit_clausegated,
            "sel_hit_weightgated": sel_hit_weightgated,
            "sel_hit_rate_clausegated": sel_hit_clausegated / sel_denom,
            "sel_hit_rate_weightgated": sel_hit_weightgated / sel_denom,
            "sel_anchor_boundaries": sel_anchor_boundaries,
            "sel_noanchor_boundaries": sel_noanchor_boundaries,
            "sel_anchor_hit_rate_ranked": (
                sel_anchor_hit_ranked / sel_anchor_boundaries
                if sel_anchor_boundaries
                else None
            ),
            "sel_anchor_hit_rate_priority": (
                sel_anchor_hit_priority / sel_anchor_boundaries
                if sel_anchor_boundaries
                else None
            ),
            "sel_noanchor_hit_rate_ranked": (
                sel_noanchor_hit_ranked / sel_noanchor_boundaries
                if sel_noanchor_boundaries
                else None
            ),
            "sel_oracle": sel_oracle,
            "sel_oracle_rate": sel_oracle / sel_denom,
        }
        stats = summary[label]
        anchor_ranked = stats["sel_anchor_hit_rate_ranked"]
        anchor_priority = stats["sel_anchor_hit_rate_priority"]
        print(
            f"{label}: boundaries={boundaries} "
            f"pred_hit@truth_op={stats['op_hit_rate_given_truth_op']} "
            f"sel_truth_next={sel_truth_present} "
            f"sel_rate_ranked={stats['sel_hit_rate_ranked']:.3f} "
            f"sel_rate_clausegated={stats['sel_hit_rate_clausegated']:.3f} "
            f"sel_rate_weightgated={stats['sel_hit_rate_weightgated']:.3f} "
            f"sel_rate_extend={stats['sel_hit_rate_extendfirst']:.3f} "
            f"sel_rate_priority={stats['sel_hit_rate_priority']:.3f} "
            f"sel_oracle={stats['sel_oracle_rate']:.3f} "
            f"avg_missing={stats['sel_avg_missing']:.2f} "
            f"anchor_ranked={anchor_ranked} anchor_priority={anchor_priority} "
            f"noanchor_ranked={stats['sel_noanchor_hit_rate_ranked']}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
