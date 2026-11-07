"""Runtime that predicts, synchronizes, and inserts cues during speech."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .models import (
    Delivery,
    EventType,
    Intervention,
    Outcome,
    Prediction,
    SyncEvent,
    SyncState,
)
from .policy import DecisionModel
from .predictors import Predictor
from .synchronizer import Synchronizer, longest_common_prefix
from .vocabulary import tag_operations


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class NullSpeaker:
    def speak(self, text: str) -> None:
        del text


class CallableSpeaker:
    def __init__(self, function: Callable[[str], None]) -> None:
        self.function = function
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        self.function(text)


class CommandSpeaker:
    """Runs a TTS command with the cue appended as one argument.

    In background mode the runtime keeps consuming transcript events while the
    cue is being spoken. The runtime uses ``last_duration_ms`` for echo
    suppression; in background mode this is a conservative estimate because
    the actual process has not finished yet.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float = 8.0,
        background: bool = True,
    ) -> None:
        if not command:
            raise ValueError("speaker command must not be empty")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.background = background
        self.last_duration_ms = 0
        self._processes: list[subprocess.Popen[bytes]] = []
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        if self.background:
            self._reap()
            self.last_duration_ms = _estimate_speech_ms(text)
            process = subprocess.Popen(
                [*self.command, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._processes.append(process)
            return
        started = time.perf_counter()
        completed = subprocess.run(
            [*self.command, text],
            check=False,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        self.last_duration_ms = round((time.perf_counter() - started) * 1_000)
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"speaker command failed: {message}")

    def _reap(self) -> None:
        with self._lock:
            self._processes = [
                process for process in self._processes if process.poll() is None
            ]

    def close(self) -> None:
        with self._lock:
            processes = list(self._processes)
            self._processes.clear()
        for process in processes:
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()


def _estimate_speech_ms(text: str) -> int:
    return min(8_000, max(500, 1_000 + 150 * len(text)))


class SyncRuntime:
    """One continuous synchronization loop.

    Events are ordered by ``at_ms``. A clause boundary triggers a prediction
    about the next span; the decision model may schedule a cue for that same
    moment, so the cue is heard before the speaker produces the next clause.
    If the speaker diverges from the prediction before delivery, the cue is
    cancelled rather than delivered into an unrelated train of thought.
    """

    def __init__(
        self,
        *,
        predictor: Predictor,
        decision_model: DecisionModel,
        speaker: Speaker | None = None,
        log_path: Path | None = None,
        pause_ms: int = 350,
        early_lead_ms: int = 0,
        cooldown_ms: int = 2_500,
        min_text_characters: int = 4,
        outcome_window_ms: int = 8_000,
        replan_on_divergence: bool = True,
        min_replan_interval_ms: int = 1_000,
        condition: str = "unspecified",
        max_context_turns: int = 6,
        max_context_characters: int = 1_200,
    ) -> None:
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")
        if early_lead_ms < 0:
            raise ValueError("early_lead_ms must be non-negative")
        if min_text_characters < 1:
            raise ValueError("min_text_characters must be positive")
        if outcome_window_ms < 1:
            raise ValueError("outcome_window_ms must be positive")
        if min_replan_interval_ms < 0:
            raise ValueError("min_replan_interval_ms must be non-negative")
        self.predictor = predictor
        self.decision_model = decision_model
        self.speaker = speaker or NullSpeaker()
        self.log_path = log_path
        self.cooldown_ms = cooldown_ms
        self.early_lead_ms = early_lead_ms
        self.min_text_characters = min_text_characters
        self.outcome_window_ms = outcome_window_ms
        self.replan_on_divergence = replan_on_divergence
        self.min_replan_interval_ms = min_replan_interval_ms
        self.condition = condition
        self.max_context_turns = max_context_turns
        self.max_context_characters = max_context_characters
        self.synchronizer = Synchronizer(pause_ms=pause_ms)
        self.now_ms = 0
        self.speaking = False
        self.paused = False
        self.utterance_count = 0
        self.prediction_count = 0
        self.interventions: list[Intervention] = []
        self.deliveries: list[Delivery] = []
        self.outcomes: list[Outcome] = []
        self._context: list[str] = []
        self._committed_text: list[str] = []
        self._current_text = ""
        self._pending: Intervention | None = None
        self._last_delivery_at_ms: int | None = None
        self._last_plan_at_ms = -(10**9)
        self._suppress_text_until_ms = 0
        self._operation_delivery_at: dict[str, int] = {}

    def feed(self, event: SyncEvent) -> list[Delivery]:
        produced: list[Delivery] = []
        self.now_ms = event.at_ms

        if event.type is EventType.INTERVENTIONS_PAUSED:
            self.paused = True
            self._cancel_pending(event.at_ms, "interventions paused by user control")
            return produced
        if event.type is EventType.INTERVENTIONS_RESUMED:
            self.paused = False
            return produced
        if event.type is EventType.PARTICIPANT_STOP:
            self.paused = True
            self._cancel_pending(event.at_ms, "participant requested stop")
            return produced
        if event.type is EventType.SESSION_ENDED:
            if self.speaking:
                self._commit_current_text()
                self.speaking = False
            self._cancel_pending(event.at_ms, "session ended before delivery")
            self._finalize_outcomes(event.at_ms, force=True)
            return produced
        if event.type is EventType.SPEECH_STARTED:
            self._begin_utterance(event.at_ms)
            self._advance(event.at_ms, produced)
            return produced
        if event.type is EventType.SPEECH_ENDED:
            if self.speaking:
                self._cancel_pending(event.at_ms, "speech ended before delivery")
                self._commit_current_text()
                self.speaking = False
                self._finalize_outcomes(event.at_ms)
            return produced

        if event.type in {EventType.TRANSCRIPT_PARTIAL, EventType.TRANSCRIPT_FINAL}:
            if event.at_ms <= self._suppress_text_until_ms:
                return produced
            if not self.speaking:
                self._begin_utterance(event.at_ms)
            text = event.text.strip()
            if len(text) < self.min_text_characters:
                return produced
            boundary = self.synchronizer.observe(text, now_ms=event.at_ms)
            self._current_text = text
            self._submit_prediction(event.at_ms)
            self._submit_decision(event.at_ms)
            self._update_outcomes(event.at_ms)
            if self.paused:
                pass
            elif boundary:
                self._plan(event.at_ms)
            elif (
                self.replan_on_divergence
                and self.synchronizer.diverged
                and self.synchronizer.prediction is not None
            ):
                cancelled = self._cancel_pending(
                    event.at_ms, "speaker diverged from prediction"
                )
                if cancelled is not None:
                    produced.append(cancelled)
                self._plan(event.at_ms, force=True)
            self._advance(event.at_ms, produced)
            return produced

        if event.type is EventType.TICK:
            if not self.speaking:
                return produced
            if not self.paused and self._current_text:
                if self.synchronizer.check_pause(now_ms=event.at_ms):
                    self._plan(event.at_ms)
                elif (
                    self.early_lead_ms > 0
                    and self._pending is None
                    and self.synchronizer.pause_due_within(
                        event.at_ms, self.early_lead_ms
                    )
                ):
                    self._plan(event.at_ms, force=True)
            self._update_outcomes(event.at_ms)
            self._advance(event.at_ms, produced)
            return produced

        return produced

    def _begin_utterance(self, at_ms: int) -> None:
        if self.speaking:
            self._commit_current_text()
        self.utterance_count += 1
        self._current_text = ""
        self.synchronizer.start(
            utterance_id=f"u{self.utterance_count}",
            at_ms=at_ms,
        )
        reset = getattr(self.predictor, "reset", None)
        if reset is not None:
            reset()
        reset_decision = getattr(self.decision_model, "reset", None)
        if reset_decision is not None:
            reset_decision()
        self.speaking = True

    def _commit_current_text(self) -> None:
        text = self._current_text.strip()
        if not text:
            return
        self._context.append(text)
        self._committed_text.append(text)
        self._trim_context()
        self._current_text = ""

    def _trim_context(self) -> None:
        while len(self._context) > self.max_context_turns:
            self._context.pop(0)
        while sum(len(item) for item in self._context) > self.max_context_characters:
            self._context.pop(0)
            if not self._context:
                break

    def _full_text(self) -> str:
        return "".join(self._committed_text) + self._current_text

    def _submit_prediction(self, now_ms: int) -> None:
        submit = getattr(self.predictor, "submit", None)
        if submit is None:
            return
        submit(
            text=self.synchronizer.text,
            context=tuple(self._context),
            now_ms=now_ms,
        )

    def _prediction_for(
        self, now_ms: int, *, min_base_length: int = 0
    ) -> Prediction | None:
        take = getattr(self.predictor, "take", None)
        if take is not None:
            return take(
                text=self.synchronizer.text,
                context=tuple(self._context),
                now_ms=now_ms,
                min_base_length=min_base_length,
            )
        return self.predictor.predict(
            text=self.synchronizer.text,
            context=tuple(self._context),
            now_ms=now_ms,
        )

    def _build_state(
        self,
        now_ms: int,
        *,
        prediction: Prediction,
        clause_text: str,
        at_boundary: bool = True,
    ) -> SyncState:
        return SyncState(
            utterance_id=self.synchronizer.utterance_id,
            text=self.synchronizer.text,
            context=tuple(self._context),
            now_ms=now_ms,
            started_at_ms=self.synchronizer.started_at_ms,
            chars_per_second=self.synchronizer.chars_per_second,
            at_boundary=at_boundary,
            prediction=prediction,
            prediction_match=self.synchronizer.prediction_match,
            diverged=self.synchronizer.diverged,
            clause_operations=frozenset(tag_operations(clause_text)),
            recent_operations=self._recent_operations(now_ms),
        )

    def _submit_decision(self, now_ms: int) -> None:
        submit = getattr(self.decision_model, "submit", None)
        if submit is None:
            return
        prediction = self._prediction_for(now_ms)
        if prediction is None:
            return
        submit(
            self._build_state(
                now_ms,
                prediction=prediction,
                clause_text=self.synchronizer.clause_text(),
            )
        )

    def _plan(self, now_ms: int, *, force: bool = False) -> None:
        if self.paused or self._pending is not None or self._cooldown_active(now_ms):
            return
        if force and now_ms - self._last_plan_at_ms < self.min_replan_interval_ms:
            return
        self._last_plan_at_ms = now_ms
        if len(self.synchronizer.text) < self.min_text_characters:
            return
        prediction_started = time.perf_counter()
        prediction = self._prediction_for(
            now_ms, min_base_length=self.synchronizer.last_boundary_length
        )
        prediction_latency_ms = int(getattr(self.predictor, "last_latency_ms", 0))
        if prediction_latency_ms == 0:
            prediction_latency_ms = round(
                (time.perf_counter() - prediction_started) * 1_000
            )
        prediction_precomputed = bool(
            getattr(self.predictor, "last_precomputed", False)
        )
        prediction_penalty_ms = 0 if prediction_precomputed else prediction_latency_ms
        self.prediction_count += 1
        if prediction is None:
            self.synchronizer.clear_prediction()
            self._log(
                "prediction",
                {
                    "at_ms": now_ms,
                    "status": "none",
                    "latency_ms": prediction_latency_ms,
                },
            )
            return
        self.synchronizer.set_prediction(prediction)
        clause_text = (
            self.synchronizer.text[self.synchronizer.last_boundary_length :]
            if force
            else self.synchronizer.last_clause
        )
        state = self._build_state(
            now_ms,
            prediction=prediction,
            clause_text=clause_text,
        )
        self._log(
            "prediction",
            {
                "at_ms": now_ms,
                "status": "predicted",
                "span": prediction.span,
                "operations": list(prediction.operations),
                "confidence": prediction.confidence,
                "source": prediction.source,
                "base_length": prediction.base_length,
                "chars_per_second": self.synchronizer.chars_per_second,
                "clause": clause_text,
                "latency_ms": prediction_latency_ms,
            },
        )
        take_decision = getattr(self.decision_model, "take", None)
        precomputed = False
        speculative_at_ms = now_ms
        if take_decision is not None:
            intervention, precomputed, decision_latency_ms, speculative_at_ms = (
                take_decision(state)
            )
        else:
            decision_started = time.perf_counter()
            intervention = self.decision_model.decide(state)
            decision_latency_ms = round(
                (time.perf_counter() - decision_started) * 1_000
            )
        if intervention is None:
            return
        if precomputed:
            shift_ms = now_ms - speculative_at_ms
            intervention = replace(
                intervention,
                scheduled_at_ms=intervention.scheduled_at_ms + shift_ms,
                expires_at_ms=intervention.expires_at_ms + shift_ms,
            )
        else:
            total_latency_ms = prediction_penalty_ms + decision_latency_ms
            if total_latency_ms > 0:
                intervention = replace(
                    intervention,
                    scheduled_at_ms=intervention.scheduled_at_ms + total_latency_ms,
                    expires_at_ms=intervention.expires_at_ms + total_latency_ms,
                )
        self._pending = intervention
        self.interventions.append(intervention)
        self._log(
            "intervention",
            {
                "at_ms": now_ms,
                "intervention_id": intervention.intervention_id,
                "operation": intervention.operation,
                "requested_operation": intervention.requested_operation,
                "cue": intervention.cue,
                "reason": intervention.reason,
                "confidence": intervention.confidence,
                "scheduled_at_ms": intervention.scheduled_at_ms,
                "boundary_at_ms": intervention.detected_at_ms,
                "prediction_latency_ms": prediction_latency_ms,
                "prediction_penalty_ms": prediction_penalty_ms,
                "prediction_precomputed": prediction_precomputed,
                "decision_latency_ms": decision_latency_ms,
                "precomputed": precomputed,
                "decision_source": getattr(
                    self.decision_model, "last_source", "synchronous"
                ),
                "prediction_span": intervention.prediction.span,
                "assignment": intervention.assignment,
            },
        )

    def _advance(self, now_ms: int, produced: list[Delivery]) -> None:
        pending = self._pending
        if pending is None:
            return
        if self.paused:
            cancelled = self._cancel_pending(
                now_ms, "interventions paused by user control"
            )
            if cancelled is not None:
                produced.append(cancelled)
            return
        if self.synchronizer.diverged:
            cancelled = self._cancel_pending(now_ms, "speaker diverged from prediction")
            if cancelled is not None:
                produced.append(cancelled)
            return
        if now_ms < pending.scheduled_at_ms:
            return
        if now_ms > pending.expires_at_ms:
            produced.append(self._expire_pending(now_ms))
            return
        produced.append(self._deliver(now_ms))

    def _deliver(self, now_ms: int) -> Delivery:
        assert self._pending is not None
        intervention = self._pending
        status = "DELIVERED"
        speak_started = time.perf_counter()
        try:
            self.speaker.speak(intervention.cue)
        except Exception as exc:  # noqa: BLE001
            status = "SUPPRESSED"
            self._log(
                "delivery_error",
                {
                    "at_ms": now_ms,
                    "intervention_id": intervention.intervention_id,
                    "error": str(exc),
                },
            )
        measured_ms = round((time.perf_counter() - speak_started) * 1_000)
        speak_ms = int(getattr(self.speaker, "last_duration_ms", 0)) or measured_ms
        if speak_ms >= 20:
            self._suppress_text_until_ms = max(
                self._suppress_text_until_ms, now_ms + speak_ms + 250
            )
        delivery = Delivery(
            intervention=intervention,
            status=status,
            delivered_at_ms=now_ms,
            text_at_delivery=self.synchronizer.text,
        )
        self.deliveries.append(delivery)
        self._last_delivery_at_ms = now_ms
        self._operation_delivery_at[intervention.operation] = now_ms
        self._pending = None
        self._log(
            "delivery",
            {
                "at_ms": now_ms,
                "intervention_id": intervention.intervention_id,
                "status": status,
                "operation": intervention.operation,
                "cue": intervention.cue,
                "assignment": intervention.assignment,
                "text_at_delivery": delivery.text_at_delivery,
                "boundary_at_ms": intervention.detected_at_ms,
                "boundary_to_delivery_ms": (now_ms - intervention.detected_at_ms),
                "speak_ms": speak_ms,
                "echo_suppression_until_ms": self._suppress_text_until_ms
                if speak_ms >= 20
                else None,
            },
        )
        if status == "DELIVERED":
            self.outcomes.append(
                Outcome(
                    delivery=delivery,
                    window_end_ms=now_ms + self.outcome_window_ms,
                    start_index=len(self._full_text()),
                )
            )
        return delivery

    def _cancel_pending(self, now_ms: int, reason: str) -> Delivery | None:
        if self._pending is None:
            return None
        delivery = Delivery(
            intervention=self._pending,
            status="CANCELLED",
            delivered_at_ms=now_ms,
            text_at_delivery=self.synchronizer.text,
        )
        self.deliveries.append(delivery)
        self._pending = None
        self._log(
            "delivery",
            {
                "at_ms": now_ms,
                "intervention_id": delivery.intervention.intervention_id,
                "status": "CANCELLED",
                "reason": reason,
            },
        )
        return delivery

    def _expire_pending(self, now_ms: int) -> Delivery:
        assert self._pending is not None
        delivery = Delivery(
            intervention=self._pending,
            status="EXPIRED",
            delivered_at_ms=now_ms,
            text_at_delivery=self.synchronizer.text,
        )
        self.deliveries.append(delivery)
        self._pending = None
        self._log(
            "delivery",
            {
                "at_ms": now_ms,
                "intervention_id": delivery.intervention.intervention_id,
                "status": "EXPIRED",
            },
        )
        return delivery

    def _update_outcomes(self, now_ms: int) -> None:
        full_text = self._full_text()
        for outcome in self.outcomes:
            if outcome.finalized:
                continue
            if outcome.start_index <= len(full_text):
                outcome.continuation = full_text[outcome.start_index :]
            self._score_outcome(outcome)
            if now_ms >= outcome.window_end_ms:
                self._finalize_outcome(outcome, now_ms)

    def _finalize_outcomes(self, now_ms: int, *, force: bool = False) -> None:
        full_text = self._full_text()
        for outcome in self.outcomes:
            if outcome.finalized:
                continue
            if outcome.start_index <= len(full_text):
                outcome.continuation = full_text[outcome.start_index :]
            self._score_outcome(outcome)
            if force or now_ms >= outcome.window_end_ms:
                self._finalize_outcome(outcome, now_ms)

    def _score_outcome(self, outcome: Outcome) -> None:
        operation = outcome.delivery.intervention.operation
        outcome.target_operation_seen = operation in tag_operations(
            outcome.continuation
        )
        span = outcome.delivery.intervention.prediction.span
        threshold = min(6, len(span))
        outcome.prediction_followed = (
            threshold > 0
            and longest_common_prefix(outcome.continuation, span) >= threshold
        )

    def _finalize_outcome(self, outcome: Outcome, now_ms: int) -> None:
        outcome.finalized = True
        self._log(
            "outcome",
            {
                "at_ms": now_ms,
                "intervention_id": outcome.delivery.intervention.intervention_id,
                "operation": outcome.delivery.intervention.operation,
                "continuation": outcome.continuation,
                "target_operation_seen": outcome.target_operation_seen,
                "prediction_followed": outcome.prediction_followed,
            },
        )

    def _cooldown_active(self, now_ms: int) -> bool:
        return (
            self._last_delivery_at_ms is not None
            and now_ms - self._last_delivery_at_ms < self.cooldown_ms
        )

    def _recent_operations(self, now_ms: int) -> frozenset[str]:
        return frozenset(
            operation
            for operation, delivered_at in self._operation_delivery_at.items()
            if now_ms - delivered_at < self.cooldown_ms
        )

    def _log(self, kind: str, payload: dict[str, object]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"kind": kind, **payload}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def report(self) -> dict[str, object]:
        delivered = [
            delivery for delivery in self.deliveries if delivery.status == "DELIVERED"
        ]
        finalized = [outcome for outcome in self.outcomes if outcome.finalized]
        per_operation: dict[str, dict[str, float | int]] = {}
        for outcome in finalized:
            operation = outcome.delivery.intervention.operation
            bucket = per_operation.setdefault(
                operation, {"delivered": 0, "uptake": 0, "rate": 0.0}
            )
            bucket["delivered"] = int(bucket["delivered"]) + 1
            if outcome.target_operation_seen:
                bucket["uptake"] = int(bucket["uptake"]) + 1
        for bucket in per_operation.values():
            delivered_count = int(bucket["delivered"])
            bucket["rate"] = (
                int(bucket["uptake"]) / delivered_count if delivered_count else 0.0
            )
        boundary_delays = sorted(
            delivery.delivered_at_ms - delivery.intervention.detected_at_ms
            for delivery in delivered
        )
        median_delay = (
            boundary_delays[len(boundary_delays) // 2] if boundary_delays else None
        )
        return {
            "predictor": self.predictor.label,
            "decision_model": self.decision_model.label,
            "condition": self.condition,
            "utterances": self.utterance_count,
            "predictions": self.prediction_count,
            "interventions": len(self.interventions),
            "delivered": len(delivered),
            "cancelled": sum(
                1 for delivery in self.deliveries if delivery.status == "CANCELLED"
            ),
            "expired": sum(
                1 for delivery in self.deliveries if delivery.status == "EXPIRED"
            ),
            "suppressed": sum(
                1 for delivery in self.deliveries if delivery.status == "SUPPRESSED"
            ),
            "per_operation": per_operation,
            "boundary_to_delivery_ms": boundary_delays,
            "median_boundary_to_delivery_ms": median_delay,
            "outcomes": [
                {
                    "intervention_id": outcome.delivery.intervention.intervention_id,
                    "operation": outcome.delivery.intervention.operation,
                    "cue": outcome.delivery.intervention.cue,
                    "assignment": outcome.delivery.intervention.assignment,
                    "continuation": outcome.continuation,
                    "target_operation_seen": outcome.target_operation_seen,
                    "prediction_followed": outcome.prediction_followed,
                }
                for outcome in self.outcomes
            ],
        }
