"""Command line entry point for the predictive synchronization loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from .evaluate import summarize
from .models import SyncEvent
from .policy import (
    AsyncDecisionModel,
    LinearDecisionModel,
    LocalLLMDecisionModel,
    RandomOperationDecisionModel,
    RuleDecisionModel,
    build_assigner,
)
from .policy_model import LinearPolicyModel
from .predictors import AsyncPredictor, LocalLLMPredictor, NgramPredictor, Predictor
from .runtime import CommandSpeaker, NullSpeaker, Speaker, SyncRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the speaker's next span, track synchronization, and insert "
            "a short cue before the predicted span is spoken."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--events", type=Path, help="Replay a JSONL event file; use - for stdin."
    )
    source.add_argument(
        "--event-command", nargs="+", help="Command that streams JSONL events."
    )
    source.add_argument(
        "--event-command-json",
        help="JSON string array for a streaming command with option-like arguments.",
    )
    parser.add_argument(
        "--decision",
        choices=["rule", "linear", "local"],
        default="rule",
        help="Timing/cue decision backend (default: rule).",
    )
    parser.add_argument(
        "--predictor",
        choices=["ngram", "local"],
        default="ngram",
        help="Prediction backend (default: ngram).",
    )
    parser.add_argument(
        "--async-prediction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a local predictor in the background and use ngram as fallback (default: enabled).",
    )
    parser.add_argument("--prediction-debounce-ms", type=int, default=200)
    parser.add_argument(
        "--async-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-compute the local decision before the boundary (default: enabled).",
    )
    parser.add_argument(
        "--decision-fallback",
        choices=["rule", "none"],
        default="rule",
        help="Instant fallback when a precomputed local decision is not ready.",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=Path("artifacts/sync-policy.json"),
        help="Linear policy model used by --decision linear (default: artifacts/sync-policy.json).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        help="Corpus for the ngram predictor; repeat the flag for multiple files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="OpenAI-compatible server URL.",
    )
    parser.add_argument(
        "--model", default=None, help="Model id; resolved from the server when omitted."
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--assignment",
        choices=["canonical", "alternate", "random", "random-operation", "fixed"],
        default="canonical",
        help=(
            "Cue assignment strategy (default: canonical). 'random' randomizes "
            "wording only and must not be used for operation-level causal "
            "comparisons; 'random-operation' draws the operation from "
            "the policy's candidate set (wording uses canonical)."
        ),
    )
    parser.add_argument("--cue", help="Cue used by --assignment fixed.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for randomized wording and random-operation exploration.",
    )
    parser.add_argument("--max-cue-characters", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument(
        "--pause-ms", type=int, default=350, help="Pause that marks a clause boundary."
    )
    parser.add_argument(
        "--early-lead-ms",
        type=int,
        default=0,
        help="Plan a cue this many ms before the pause threshold (default: 0, off).",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=2_500,
        help="Minimum gap between delivered cues.",
    )
    parser.add_argument(
        "--insertion-delay-ms",
        type=int,
        default=100,
        help="Delay between the boundary and the cue, after the decision returns.",
    )
    parser.add_argument(
        "--delivery-window-ms",
        type=int,
        default=1_200,
        help="How long a planned cue stays valid.",
    )
    parser.add_argument(
        "--outcome-window-ms",
        type=int,
        default=8_000,
        help="Continuation window for uptake.",
    )
    parser.add_argument(
        "--replan-on-divergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-predict and re-plan where the speaker diverges (default: enabled).",
    )
    parser.add_argument("--min-replan-interval-ms", type=int, default=1_000)
    parser.add_argument(
        "--condition-label",
        default="unspecified",
        help="Experimental condition written to every log entry.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help=(
            "Record the decision and delivery timing but never speak; the cue is "
            "logged with status SILENT and no uptake outcome is created."
        ),
    )
    parser.add_argument(
        "--no-context-fit",
        action="store_true",
        help=(
            "Restore the legacy static operation priority. By default the rule "
            "policy prefers moves adjacent to what the speaker just said and "
            "stays silent when the transcript carries no operation signal."
        ),
    )
    parser.add_argument(
        "--pair-key",
        default="",
        help="Key shared by paired CUE and SILENCE runs for paired analysis.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Session identifier written to the report and used by analysis.",
    )
    parser.add_argument("--min-text-characters", type=int, default=4)
    parser.add_argument(
        "--tts-command",
        nargs="+",
        help="TTS command; the cue is appended as one argument.",
    )
    parser.add_argument(
        "--tts-command-json", help="JSON string array for the TTS command."
    )
    parser.add_argument(
        "--tts-async",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speak without blocking the event loop (default: enabled).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace file replay by event timestamps (ignored for live commands).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="JSONL log for predictions, decisions, deliveries, outcomes.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sync-report.json"),
        help="Aggregate report path (default: artifacts/sync-report.json).",
    )
    return parser


def _parse_command_json(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("command JSON is invalid") from exc
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("command JSON must be a non-empty array of strings")
    return value


def _event_lines(args: argparse.Namespace) -> Iterable[str]:
    if args.events is not None:
        if str(args.events) == "-":
            for line in sys.stdin:
                yield line
            return
        with args.events.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line
        return
    if args.event_command_json:
        command = _parse_command_json(args.event_command_json)
    elif args.event_command:
        command = list(args.event_command)
    else:
        raise ValueError("an event source is required")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            yield line
    finally:
        process.stdout.close()
        return_code = process.wait(timeout=10)
        if return_code != 0:
            raise RuntimeError(f"event command exited with status {return_code}")


def _parse_events(lines: Iterable[str]) -> Iterable[SyncEvent]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield SyncEvent.from_json(stripped)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid event line {line_number}: {exc}") from exc


def _build_predictor(args: argparse.Namespace) -> Predictor:
    if args.predictor == "local":
        local = LocalLLMPredictor(
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.request_timeout,
            max_tokens=args.max_tokens,
        )
        if args.async_prediction:
            fallback = (
                NgramPredictor.from_paths(args.corpus)
                if args.corpus
                else NgramPredictor.default()
            )
            return AsyncPredictor(
                local,
                fallback=fallback,
                debounce_ms=args.prediction_debounce_ms,
            )
        return local
    if args.corpus:
        return NgramPredictor.from_paths(args.corpus)
    return NgramPredictor.default()


def _build_speaker(args: argparse.Namespace) -> Speaker:
    if args.tts_command_json:
        return CommandSpeaker(
            _parse_command_json(args.tts_command_json),
            background=args.tts_async,
        )
    if args.tts_command:
        return CommandSpeaker(list(args.tts_command), background=args.tts_async)
    return NullSpeaker()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictor = _build_predictor(args)
        wording_assignment = (
            "canonical" if args.assignment == "random-operation" else args.assignment
        )
        assigner = build_assigner(
            wording_assignment,
            seed=args.seed,
            cue=args.cue,
            max_characters=args.max_cue_characters,
        )
        if args.decision == "linear":
            decision_model = LinearDecisionModel(
                LinearPolicyModel.load(args.policy_model),
                assigner,
                max_characters=args.max_cue_characters,
                delivery_window_ms=args.delivery_window_ms,
                insertion_delay_ms=args.insertion_delay_ms,
            )
        elif args.decision == "local":
            decision_model = LocalLLMDecisionModel(
                base_url=args.base_url,
                model=args.model,
                assigner=assigner,
                timeout=args.request_timeout,
                max_tokens=args.max_tokens,
                max_characters=args.max_cue_characters,
                delivery_window_ms=args.delivery_window_ms,
                insertion_delay_ms=args.insertion_delay_ms,
                min_confidence=args.min_confidence,
            )
        else:
            decision_model = RuleDecisionModel(
                assigner,
                min_confidence=args.min_confidence,
                max_characters=args.max_cue_characters,
                delivery_window_ms=args.delivery_window_ms,
                insertion_delay_ms=args.insertion_delay_ms,
                context_fit=not args.no_context_fit,
            )
        if args.assignment == "random-operation":
            decision_model = RandomOperationDecisionModel(
                decision_model, assigner, seed=args.seed
            )
        if args.decision == "local" and args.async_decision:
            fallback = (
                RuleDecisionModel(
                    assigner,
                    min_confidence=args.min_confidence,
                    max_characters=args.max_cue_characters,
                    delivery_window_ms=args.delivery_window_ms,
                    insertion_delay_ms=args.insertion_delay_ms,
                    context_fit=not args.no_context_fit,
                )
                if args.decision_fallback == "rule"
                else None
            )
            if fallback is not None and args.assignment == "random-operation":
                fallback = RandomOperationDecisionModel(
                    fallback, assigner, seed=args.seed
                )
            decision_model = AsyncDecisionModel(
                decision_model,
                fallback=fallback,
            )
        if args.predictor == "local" and hasattr(predictor, "warmup"):
            predictor.warmup()
        if args.decision == "local" and hasattr(decision_model, "warmup"):
            decision_model.warmup()
        speaker = _build_speaker(args)
        runtime = SyncRuntime(
            predictor=predictor,
            decision_model=decision_model,
            speaker=speaker,
            log_path=args.log,
            pause_ms=args.pause_ms,
            early_lead_ms=args.early_lead_ms,
            cooldown_ms=args.cooldown_ms,
            min_text_characters=args.min_text_characters,
            outcome_window_ms=args.outcome_window_ms,
            replan_on_divergence=args.replan_on_divergence,
            min_replan_interval_ms=args.min_replan_interval_ms,
            condition=args.condition_label,
            assignment=args.assignment,
            seed=args.seed,
            session_id=args.session_id,
            pair_key=args.pair_key,
            silent=args.silent,
        )
        last_at_ms = 0
        replay_started = time.monotonic()
        try:
            for event in _parse_events(_event_lines(args)):
                if args.realtime and args.events is not None:
                    delay = replay_started + event.at_ms / 1_000 - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                last_at_ms = max(last_at_ms, event.at_ms)
                produced = runtime.feed(event)
                for delivery in produced:
                    print(
                        f"[{delivery.delivered_at_ms:>6}ms] {delivery.status:9} "
                        f"cue={delivery.intervention.cue!r} "
                        f"op={delivery.intervention.operation} "
                        f"+{delivery.delivered_at_ms - delivery.intervention.detected_at_ms}ms"
                    )
        except KeyboardInterrupt:
            pass
        finally:
            for component in (speaker, predictor, decision_model):
                close = getattr(component, "close", None)
                if close is not None:
                    close()
        runtime.feed(
            SyncEvent.from_json(
                json.dumps({"type": "session_ended", "at_ms": last_at_ms + 1})
            )
        )
        report = runtime.report()
        report["summary"] = summarize(runtime.outcomes)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"predictor={report['predictor']} decision={report['decision_model']} "
        f"utterances={report['utterances']} predictions={report['predictions']} "
        f"delivered={report['delivered']} cancelled={report['cancelled']} "
        f"expired={report['expired']}"
    )
    for operation, bucket in sorted(report["per_operation"].items()):
        print(
            f"  {operation:12} delivered={bucket['delivered']} "
            f"uptake={bucket['uptake']} rate={bucket['rate']:.2f} "
            f"carryover={bucket['carryover']}"
        )
    summary = report["summary"]
    print(
        f"  carryover={summary['carryover']} rate={summary['carryover_rate']:.2f} "
        f"repertoire_growth={','.join(report['repertoire_growth']) or '-'}"
    )
    if args.log is not None:
        print(f"log={args.log}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
