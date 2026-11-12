#!/usr/bin/env python3
"""Simulate paired CUE/SILENCE sessions through the real runtime.

`synthesize_experiment.py` writes analyzer-ready *reports* from a coin-flip
speaker. This script goes one layer deeper: it generates Japanese clause
sequences, feeds them through the real `SyncRuntime` (the default cycle
predictor rotates one tagged operation per call; rule decision; assignment;
canned speaker), and emits analyzer-ready reports plus
JSONL logs — so the full measurement path (prediction → synchronization →
cue delivery → outcome scoring → carryover) is exercised on data where the
effect is known by construction.

The speaker is stereotyped; clause patterns are canned. This is a harness for
*measuring measurement*, not a model of human reasoning. CUE and SILENCE arms
share the skeleton and all text/timing draws (common random numbers), and the
per-boundary uptake coin is drawn from a dedicated stream keyed by clause
index — so paired differences are attributable to the audible cue only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.models import EventType, Prediction, SyncEvent
from cueloop.sync.policy import CanonicalAssigner, RuleDecisionModel
from cueloop.sync.predictors import NgramPredictor, ScriptedPredictor
from cueloop.sync.runtime import CallableSpeaker, SyncRuntime
from cueloop.sync.stem import StemCueGenerator
from cueloop.sync.vocabulary import tag_operations


class CyclePredictor:
    """Deterministic synthetic predictor: one operation tag per call, cycling.

    The spans are hand-authored to cover cumulative prefixes of PRIORITY, so
    the rule's cue walks through all eight operations in rotation instead of
    always choosing ground.
    """

    _SPANS = (
        "だから結論は決まる。",  # covers conclusion → cue ground
        "根拠ははっきりしない。",  # +ground → boundary
        "根拠と条件はある。",  # +boundary → dependency
        "根拠と条件と前提はある。",  # +dependency → comparison
        "根拠と条件と前提と比較はある。",  # +comparison → invert
        "根拠と条件と前提と比較と逆もある。",  # +invert → concretize
        "根拠と条件と前提と比較と逆と例えばある。",  # +concretize → mechanism
        "根拠と条件と前提と比較と逆と例えば理由はある。",  # +mechanism → conclusion
    )

    def __init__(self) -> None:
        self._index = 0
        self.label = "simulated"

    def predict(self, *, text: str, context, now_ms: int) -> Prediction:
        span = self._SPANS[self._index % len(self._SPANS)]
        self._index += 1
        return Prediction(
            span=span,
            operations=tuple(tag_operations(span)),
            confidence=0.9,
            source=self.label,
            created_at_ms=now_ms,
            base_length=len(text),
        )


TOKEN_BY_OPERATION = {
    "conclusion": ["つまり", "結論は", "だから"],
    "ground": ["根拠は", "なぜなら"],
    "concretize": ["例えば", "具体的には"],
    "dependency": ["前提は", "依存し"],
    "invert": ["逆に", "逆なら"],
    "comparison": ["違いは", "比較は"],
    "boundary": ["条件は", "範囲は"],
    "mechanism": ["理由は", "仕組みは"],
}

SUBJECTS = ["この設計は", "この仕組みは", "この仮説は", "この実験は", "この前提は"]
OBJECTS = ["観測データ", "実験結果", "検証ログ", "三つの試行", "設定条件"]
FALLBACK_CONTINUATIONS = [
    "それで問題ありません。",
    "おそらく成立すると思います。",
    "まあまあ、確認は取れています。",
    "一応、大丈夫と考えています。",
    "やや自信は持てません。",
]
ANNOUNCE_CYCLE = [
    "conclusion",
    "conclusion",
    "conclusion",
    "conclusion",
    "conclusion",
    "conclusion",
]


def _chunk_text(text: str, rng: random.Random):
    index = 0
    while index < len(text):
        step = min(len(text) - index, rng.choice([2, 3, 3, 4, 5]))
        yield text[: index + step]
        index += step


def _new_clause(index: int, rng: random.Random) -> dict:
    announced = ANNOUNCE_CYCLE[index % len(ANNOUNCE_CYCLE)]
    return {
        "subject": rng.choice(SUBJECTS),
        "object": rng.choice(OBJECTS),
        "announced": announced,
        "announce": rng.choice(TOKEN_BY_OPERATION[announced]),
    }


def _continuation(
    clause: dict,
    rng: random.Random,
    *,
    cued_op: str | None,
    uptake_draw: float,
    base_uptake: float,
    uptake_effect: float,
    carryover_op: str | None,
    carryover_draw: float,
    base_carryover: float,
    carryover_effect: float,
    carryover_applies: bool,
) -> tuple[str, bool]:
    p = base_uptake + (uptake_effect if cued_op else 0.0)
    if cued_op and uptake_draw < p:
        token = rng.choice(TOKEN_BY_OPERATION[cued_op])
        piece = f"{clause['subject']}{token}{clause['object']}からだと考えます。"
    else:
        piece = rng.choice(FALLBACK_CONTINUATIONS)
    q = base_carryover + (carryover_effect if carryover_applies else 0.0)
    if carryover_op and carryover_draw < q:
        token = rng.choice(TOKEN_BY_OPERATION[carryover_op])
        piece += f"{clause['subject']}{token}{clause['object']}だと考えます。"
        return piece, True
    return piece, False


def _feed_partial(runtime: SyncRuntime, text_so_far: str, now_ms: int):
    runtime.feed(SyncEvent(EventType.TRANSCRIPT_PARTIAL, now_ms, text_so_far))
    runtime.feed(SyncEvent(EventType.TICK, now_ms + 50))


def run_session(
    *,
    index: int,
    seed: int,
    condition: str,
    uptake_effect: float,
    base_uptake: float,
    carryover_effect: float,
    base_carryover: float,
    clauses: int,
    predictor_name: str,
    cooldown_ms: int,
    outcome_window_ms: int,
    cue_style: str = "bare",
) -> dict:
    silent = condition == "SILENCE"
    rng = random.Random(f"{seed}|{index}|skeleton")
    skeleton = [_new_clause(i, rng) for i in range(clauses)]
    uniforms = [random.Random(f"{seed}|{index}|u|{i}").random() for i in range(clauses)]
    timing_rng = random.Random(f"{seed}|{index}|timing")

    predictor = (
        CyclePredictor()
        if predictor_name == "cycle"
        else ScriptedPredictor(
            {"この": ("だから結論は決まる。", 0.9)},
            fallback=("だから結論は決まる。", 0.9),
        )
        if predictor_name == "scripted"
        else NgramPredictor.default()
    )
    stem_generator = (
        StemCueGenerator(
            base_url="http://127.0.0.1:1",  # unreachable on purpose: template falls back
            model="test-model",
            timeout=0.2,
        )
        if cue_style == "stem"
        else None
    )
    runtime = SyncRuntime(
        predictor=predictor,
        decision_model=RuleDecisionModel(
            CanonicalAssigner(),
            insertion_delay_ms=0,
            context_fit=False,
            cue_style=cue_style,
            stem_generator=stem_generator,
        ),
        speaker=CallableSpeaker(lambda text: None),
        cooldown_ms=cooldown_ms,
        outcome_window_ms=outcome_window_ms,
        silent=silent,
        condition=condition,
        assignment="canonical",
        seed=seed,
        session_id=f"{condition.lower()}-sim-{index:03d}-s{seed}",
        pair_key=f"sim-seed{seed}-{index:03d}",
    )

    now_ms = 0
    utterance_text = ""
    runtime.feed(SyncEvent(EventType.SPEECH_STARTED, now_ms))
    recent_cues: list[tuple[int, str]] = []
    consumed: set[int] = set()

    for clause_index, clause in enumerate(skeleton):
        decisions_before = len(runtime.interventions)
        base_len = len(utterance_text)

        phrase_a = f"{clause['subject']}{clause['object']}です。{clause['announce']}"
        for step_text in _chunk_text(phrase_a, timing_rng):
            utterance_text = utterance_text[:base_len] + step_text
            _feed_partial(runtime, utterance_text, now_ms)
            now_ms += timing_rng.choice([110, 130, 150, 170])

        # Newly decided interventions belong to this boundary.
        new_decisions = runtime.interventions[decisions_before:]
        for decision in new_decisions:
            recent_cues.append((clause_index, decision.operation))
        cued_op = new_decisions[-1].operation if new_decisions else None

        # Carryover candidate: earliest unconsumed cue from 3+ clauses ago.
        consumed_idx: int | None = None
        carryover_op = None
        for j, op in recent_cues:
            if j <= clause_index - 3 and j not in consumed:
                carryover_op = op
                consumed_idx = j
                break
        carryover_draw = (
            random.Random(f"{seed}|{index}|cy|{clause_index}|{consumed_idx}").random()
            if consumed_idx is not None
            else 1.0
        )
        carryover_applies = condition == "CUE"

        piece_b, fired = _continuation(
            clause,
            timing_rng,
            cued_op=cued_op,
            uptake_draw=uniforms[clause_index],
            base_uptake=base_uptake,
            uptake_effect=uptake_effect if condition == "CUE" else 0.0,
            carryover_op=carryover_op,
            carryover_draw=carryover_draw,
            base_carryover=base_carryover,
            carryover_effect=carryover_effect,
            carryover_applies=carryover_applies,
        )
        if fired and consumed_idx is not None:
            consumed.add(consumed_idx)

        for step_text in _chunk_text(piece_b, timing_rng):
            utterance_text = utterance_text[:base_len] + phrase_a + step_text
            _feed_partial(runtime, utterance_text, now_ms)
            now_ms += timing_rng.choice([110, 130, 150, 170])

    runtime.feed(SyncEvent(EventType.TRANSCRIPT_FINAL, now_ms, utterance_text))
    # Advance past the outcome window while text is still arriving; the final
    # outcomes are force-finalized at SESSION_ENDED with the full continuation.
    now_ms += 500
    runtime.feed(SyncEvent(EventType.SPEECH_ENDED, now_ms))
    runtime.feed(SyncEvent(EventType.SESSION_ENDED, now_ms + 1))
    report = runtime.report()
    report["synthetic"] = True
    report["note"] = "simulated Japanese speaker; calibration, not human evidence"
    report["session_index"] = index
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--clauses", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-uptake", type=float, default=0.15)
    parser.add_argument("--uptake-effect", type=float, default=0.4)
    parser.add_argument("--base-carryover", type=float, default=0.10)
    parser.add_argument("--carryover-effect", type=float, default=0.35)
    parser.add_argument("--outcome-window-ms", type=int, default=4_000)
    parser.add_argument(
        "--predictor",
        choices=["cycle", "scripted", "ngram"],
        default="cycle",
        help="""Prediction backend for the synthetic speaker sessions.
'cycle' (default) rotates one tagged operation per call so the cue varies;
'scripted' always predicts a conclusion-only span (cue is then rule-first
missing, i.e. ground); 'ngram' uses the embedded corpus for realism.""",
    )
    parser.add_argument(
        "--cue-style",
        choices=["bare", "stem"],
        default="bare",
        help="""Cue style to speak. 'bare' (default) speaks the operation word;
'stem' speaks a complete initial-condition sentence rendered from the
deterministic span template (the LLM bow-outs on the unreachable test
server on purpose, so the arm is reproducible offline).""",
    )
    parser.add_argument("--cooldown-ms", type=int, default=1_500)
    args = parser.parse_args(argv)
    if args.sessions < 1 or args.clauses < 1:
        print("error: --sessions and --clauses must be positive", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"error: output directory is not empty: {args.output_dir}", file=sys.stderr
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.sessions):
        for condition in ("CUE", "SILENCE"):
            report = run_session(
                index=index,
                seed=args.seed,
                condition=condition,
                uptake_effect=args.uptake_effect,
                base_uptake=args.base_uptake,
                clauses=args.clauses,
                predictor_name=args.predictor,
                cooldown_ms=args.cooldown_ms,
                carryover_effect=args.carryover_effect,
                base_carryover=args.base_carryover,
                outcome_window_ms=args.outcome_window_ms,
                cue_style=args.cue_style,
            )
            path = args.output_dir / f"{condition.lower()}-{index:03d}.json"
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(
        json.dumps(
            {"sessions": args.sessions, "output_dir": str(args.output_dir)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
