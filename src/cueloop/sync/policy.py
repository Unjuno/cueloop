"""Intervention policy and cue assignment.

The policy decides *when* a cue is warranted. The assigner decides *what* cue
is spoken. This split is what keeps a live product policy separate from an
experimental manipulation: an experiment can fix or randomize the cue while
the same timing policy runs in both conditions.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from .models import Intervention, SyncState
from .policy_model import LinearPolicyModel, PolicyObservation
from .predictors import PREFERRED_DECISION_MODELS, chat_completion, resolve_model
from .vocabulary import (
    CUE_BY_OPERATION,
    OPERATIONS,
    PRIORITY,
    cue_for,
    operation_for_cue,
)


class DecisionModel(Protocol):
    label: str

    def decide(self, state: SyncState) -> Intervention | None: ...


@dataclass(frozen=True)
class SpeculativeDecision:
    intervention: Intervention | None
    text: str
    utterance_id: str
    created_at_ms: int
    latency_ms: int


class AsyncDecisionModel:
    """Pre-compute the slow local decision before the boundary arrives.

    A boundary can arrive while the local model is still composing a cue. The
    runtime submits the current state on every partial that has a prediction;
    at the boundary it takes the newest result whose text is still a prefix of
    the utterance. A taken result is consumed, so a stale decision is not
    reused at a later boundary. The intervention is rebased to the actual
    boundary by the runtime, so the cue can land inside the pause instead of
    after the model latency.
    """

    def __init__(
        self,
        primary: DecisionModel,
        *,
        fallback: DecisionModel | None = None,
        max_age_ms: int = 2_500,
        max_results: int = 4,
    ) -> None:
        if max_age_ms < 1:
            raise ValueError("max_age_ms must be positive")
        if max_results < 1:
            raise ValueError("max_results must be positive")
        self.primary = primary
        self.fallback = fallback
        self.max_age_ms = max_age_ms
        self.max_results = max_results
        self.label = primary.label
        self.last_latency_ms = 0
        self.last_source = "synchronous"
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._submitted_text = ""
        self._results: list[SpeculativeDecision] = []
        self._generation = 0
        self._closed = False

    def warmup(self) -> None:
        warmup = getattr(self.primary, "warmup", None)
        if warmup is not None:
            warmup()

    def submit(self, state: SyncState) -> None:
        if state.prediction is None:
            return
        with self._lock:
            if self._closed or self._in_flight >= 1:
                return
            if state.text == self._submitted_text:
                return
            self._submitted_text = state.text
            self._in_flight += 1
            generation = self._generation
            started = time.perf_counter()
            self._executor.submit(self._run, state, started, generation)

    def _run(self, state: SyncState, started: float, generation: int) -> None:
        try:
            intervention = self.primary.decide(state)
        except Exception:  # noqa: BLE001
            intervention = None
        latency_ms = round((time.perf_counter() - started) * 1_000)
        with self._lock:
            self._in_flight -= 1
            if generation != self._generation:
                return
            self._results.append(
                SpeculativeDecision(
                    intervention=intervention,
                    text=state.text,
                    utterance_id=state.utterance_id,
                    created_at_ms=state.now_ms,
                    latency_ms=latency_ms,
                )
            )
            self._results = self._results[-self.max_results :]

    def take(self, state: SyncState) -> tuple[Intervention | None, bool, int, int]:
        """Return (intervention, precomputed, latency_ms, state_at_ms)."""
        with self._lock:
            candidates = [
                result
                for result in self._results
                if result.utterance_id == state.utterance_id
                and state.text.startswith(result.text)
                and state.now_ms - result.created_at_ms <= self.max_age_ms
            ]
        if candidates:
            result = max(candidates, key=lambda item: len(item.text))
            self._results = [
                candidate for candidate in self._results if candidate is not result
            ]
            self.last_latency_ms = result.latency_ms
            self.last_source = "precomputed"
            return result.intervention, True, result.latency_ms, result.created_at_ms
        if self.fallback is not None:
            self.last_latency_ms = 0
            self.last_source = "fallback"
            return self.fallback.decide(state), False, 0, state.now_ms
        started = time.perf_counter()
        intervention = self.primary.decide(state)
        latency_ms = round((time.perf_counter() - started) * 1_000)
        self.last_latency_ms = latency_ms
        self.last_source = "synchronous"
        return intervention, False, latency_ms, state.now_ms

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._results.clear()
            self._submitted_text = ""
            self.last_latency_ms = 0
            self.last_source = "synchronous"

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
        self._executor.shutdown(wait=False, cancel_futures=True)


class CueAssigner(Protocol):
    label: str

    def assign(self, operation: str) -> str: ...


class CanonicalAssigner:
    """Always use the first canonical cue for the operation."""

    def __init__(self, *, max_characters: int = 8) -> None:
        self.max_characters = max_characters
        self.label = "canonical"

    def assign(self, operation: str) -> str:
        cue = cue_for(operation, max_characters=self.max_characters)
        if cue is None:
            raise ValueError(f"no cue fits operation {operation}")
        return cue


class FixedAssigner:
    """Deliver one fixed cue. Used for a controlled experimental condition."""

    def __init__(self, cue: str) -> None:
        if not cue.strip():
            raise ValueError("fixed cue must not be empty")
        self.cue = cue
        self.label = f"fixed({cue})"

    def assign(self, operation: str) -> str:
        del operation
        return self.cue


class AlternateAssigner:
    """Rotate through the operation's cues to avoid repetition."""

    def __init__(self, *, max_characters: int = 8) -> None:
        self.max_characters = max_characters
        self._offsets: dict[str, int] = {}
        self.label = "alternate"

    def assign(self, operation: str) -> str:
        candidates = [
            cue
            for cue in CUE_BY_OPERATION.get(operation, ())
            if len(cue) <= self.max_characters
        ]
        if not candidates:
            raise ValueError(f"no cue fits operation {operation}")
        offset = self._offsets.get(operation, 0)
        self._offsets[operation] = offset + 1
        return candidates[offset % len(candidates)]


class RandomAssigner:
    """Randomize the cue for an operation with a recorded seed.

    Randomization is the experimental condition. The sampled cue is written to
    the intervention log so an analysis can recover which wording was heard.
    """

    def __init__(self, *, seed: int = 0, max_characters: int = 8) -> None:
        self._random = random.Random(seed)
        self.max_characters = max_characters
        self.label = f"randomized({seed})"

    def assign(self, operation: str) -> str:
        candidates = [
            cue
            for cue in CUE_BY_OPERATION.get(operation, ())
            if len(cue) <= self.max_characters
        ]
        if not candidates:
            raise ValueError(f"no cue fits operation {operation}")
        return self._random.choice(candidates)


class LocalLLMDecisionModel:
    """Ask a local OpenAI-compatible model whether and what to insert.

    The model sees the current text, the predicted next span, and the
    operations already present. It must answer with JSON only. Any malformed
    answer or transport error fails closed as silence, so a bad model can
    never force an unvalidated cue into the conversation.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str | None = None,
        assigner: CueAssigner | None = None,
        temperature: float = 0.0,
        timeout: float = 5.0,
        max_tokens: int = 64,
        max_characters: int = 8,
        delivery_window_ms: int = 1_200,
        insertion_delay_ms: int = 0,
        min_confidence: float = 0.5,
        require_missing: bool = True,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if delivery_window_ms < 1:
            raise ValueError("delivery_window_ms must be positive")
        if insertion_delay_ms < 0:
            raise ValueError("insertion_delay_ms must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.model = model or resolve_model(
            self.base_url,
            timeout=timeout,
            preferred=PREFERRED_DECISION_MODELS,
        )
        self.assigner = assigner
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_characters = max_characters
        self.delivery_window_ms = delivery_window_ms
        self.insertion_delay_ms = insertion_delay_ms
        self.min_confidence = min_confidence
        self.require_missing = require_missing
        self.label = f"local-llm({self.model})"
        self.last_error: str | None = None

    def warmup(self) -> None:
        """Load the model so the first decision is not slowed by a cold start."""
        try:
            chat_completion(
                self.base_url,
                self.model,
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                timeout=self.timeout,
                max_tokens=1,
            )
        except (RuntimeError, TypeError, ValueError):
            pass

    def _missing(self, state: SyncState) -> tuple[frozenset[str], list[str]]:
        prediction = state.prediction
        assert prediction is not None
        covered_set = (
            set(prediction.operations)
            | set(state.clause_operations)
            | set(state.recent_operations)
        )
        missing = [operation for operation in PRIORITY if operation not in covered_set]
        return frozenset(covered_set), missing

    def _build_messages(self, state: SyncState) -> tuple[str, str] | None:
        prediction = state.prediction
        assert prediction is not None
        covered_set, missing = self._missing(state)
        if not missing:
            return None
        covered = ", ".join(sorted(covered_set)) or "なし"
        recent = ", ".join(sorted(state.recent_operations)) or "なし"
        context = " / ".join(state.context[-4:]) or "なし"
        system = (
            "あなたは話し手の発話中に短い日本語の cue を差し込む意思決定モデルです。"
            "目的は、話し手が次に言おうとしていることの構造的な抜けを、1語だけ補うことです。"
            "割り込むと思考を妨げる場合は SILENCE を選びます。"
            "必ずJSONオブジェクトだけを返し、説明を含めないこと。"
        )
        user = (
            f"【直前までの文脈】\n{context}\n\n"
            f"【現在の発話】\n{state.text}\n\n"
            f"【次に来ると予測される節】\n{prediction.span}\n"
            f"【予測される操作】\n{', '.join(prediction.operations) or 'なし'}\n"
            f"【既に現れた操作】\n{covered}\n"
            f"【直近の操作】\n{recent}\n"
            f"【補う候補】\n{', '.join(missing)}\n\n"
            "【指示】CUE か SILENCE を選んでください。CUE なら【補う候補】から "
            "operation を1つ選び、"
            f"{self.max_characters}文字以内の日本語 cue を返してください。"
            "confidence は 0.0〜1.0 の実数で、CUE の確信度を表します。"
            "既に現れた操作や予測に含まれる操作を選ばず、抜けている操作を1つ選んでください。"
            "予測が確実でない、または割り込む価値が低い場合は SILENCE にしてください。\n\n"
            "【出力形式】\n"
            '{"action":"CUE","operation":"ground","cue":"根拠は？",'
            '"confidence":0.7,"reason":"..."}\n'
            'または {"action":"SILENCE","confidence":0.7,"reason":"..."}'
        )
        return system, user

    def decide(self, state: SyncState) -> Intervention | None:
        if not state.at_boundary or state.prediction is None or state.diverged:
            return None
        if not _span_has_content(state.prediction.span):
            return None
        messages = self._build_messages(state)
        if messages is None:
            return None
        system, user = messages
        try:
            content = chat_completion(
                self.base_url,
                self.model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
            intervention = self._parse(content, state)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            return None
        self.last_error = None
        return intervention

    def _parse(self, content: str, state: SyncState) -> Intervention | None:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("decision response did not contain a JSON object")
        payload = json.loads(content[start : end + 1])
        if not isinstance(payload, dict):
            raise TypeError("decision response must be a JSON object")
        action = payload.get("action")
        if not isinstance(action, str):
            raise TypeError("decision response requires an action")
        action = action.strip().upper()
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        if action == "SILENCE":
            return None
        if action != "CUE":
            raise ValueError(f"unknown decision action: {action}")

        operation = payload.get("operation")
        cue = payload.get("cue")
        if operation not in OPERATIONS and isinstance(cue, str):
            operation = operation_for_cue(cue.strip())
        requested_operation = operation if operation in OPERATIONS else "unmapped"
        if self.assigner is not None:
            cue = self.assigner.assign(requested_operation)
        else:
            if not isinstance(cue, str) or not cue.strip():
                if requested_operation not in OPERATIONS:
                    raise ValueError("CUE requires an operation or a cue")
                cue = cue_for(requested_operation, max_characters=self.max_characters)
            if not cue:
                raise ValueError("no cue fits the requested operation")
            cue = cue.strip()
            if len(cue) > self.max_characters:
                fallback = (
                    cue_for(requested_operation, max_characters=self.max_characters)
                    if requested_operation in OPERATIONS
                    else None
                )
                if fallback is None:
                    raise ValueError("cue is longer than max_characters")
                cue = fallback
        actual_operation = operation_for_cue(cue) or "unmapped"
        if self.require_missing:
            covered = (
                set(state.prediction.operations)
                | set(state.clause_operations)
                | set(state.recent_operations)
            )
            if actual_operation in covered:
                return None
        confidence = payload.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise TypeError("decision confidence must be numeric")
        confidence = min(1.0, max(0.0, float(confidence)))
        if confidence < self.min_confidence:
            return None

        scheduled_at = state.now_ms + self.insertion_delay_ms
        digest = hashlib.sha256(
            f"{state.utterance_id}:{state.now_ms}:{actual_operation}:{cue}:{state.text}".encode()
        ).hexdigest()[:12]
        return Intervention(
            intervention_id=f"cue-{digest}",
            detected_at_ms=state.now_ms,
            operation=actual_operation,
            requested_operation=requested_operation,
            cue=cue,
            reason=f"local model: {reason}" if reason else "local model decision",
            confidence=confidence,
            scheduled_at_ms=scheduled_at,
            expires_at_ms=scheduled_at + self.delivery_window_ms,
            insertion_delay_ms=self.insertion_delay_ms,
            assignment=self.assigner.label if self.assigner else "local-llm",
            base_text=state.text,
            prediction=state.prediction,
        )


class RuleDecisionModel:
    """Prediction-gap policy: cue an operation the next span would omit.

    The model fires only at a clause boundary, so the inserted cue is heard
    while the speaker is still talking but before the next clause is produced.
    It stays silent when the prediction is uncertain, when the speaker has
    already diverged, or when every useful operation is already covered.
    """

    def __init__(
        self,
        assigner: CueAssigner,
        *,
        min_confidence: float = 0.55,
        max_characters: int = 8,
        priority: Sequence[str] = PRIORITY,
        delivery_window_ms: int = 1_200,
        insertion_delay_ms: int = 0,
        context_fit: bool = True,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if delivery_window_ms < 1:
            raise ValueError("delivery_window_ms must be positive")
        if insertion_delay_ms < 0:
            raise ValueError("insertion_delay_ms must be non-negative")
        self.assigner = assigner
        self.min_confidence = min_confidence
        self.max_characters = max_characters
        self.priority = tuple(priority)
        self.delivery_window_ms = delivery_window_ms
        self.insertion_delay_ms = insertion_delay_ms
        self.context_fit = context_fit
        self.label = f"prediction-gap(assigner={assigner.label})"

    def decide(self, state: SyncState) -> Intervention | None:
        prediction = state.prediction
        if not state.at_boundary or prediction is None or state.diverged:
            return None
        if not _span_has_content(prediction.span):
            return None
        if prediction.confidence < self.min_confidence:
            return None
        covered = set(prediction.operations) | set(state.clause_operations)
        missing = [
            operation
            for operation in self.priority
            if operation not in covered and operation not in state.recent_operations
        ]
        if not missing:
            return None
        context_fit = False
        if self.context_fit:
            # No tagged signal anywhere: an operation choice would be noise.
            if (
                not state.clause_operations
                and not state.recent_operations
                and not prediction.operations
            ):
                return None
            ranked = PolicyObservation.from_state(state).ranked_missing_operations()
            if ranked:
                missing = list(ranked)
            context_fit = True
        requested_operation = missing[0]
        cue = self.assigner.assign(requested_operation)
        operation = operation_for_cue(cue) or "unmapped"
        scheduled_at = state.now_ms + self.insertion_delay_ms
        digest = hashlib.sha256(
            f"{state.utterance_id}:{state.now_ms}:{operation}:{cue}:{state.text}".encode()
        ).hexdigest()[:12]
        return Intervention(
            intervention_id=f"cue-{digest}",
            detected_at_ms=state.now_ms,
            operation=operation,
            requested_operation=requested_operation,
            cue=cue,
            reason=(
                f"next span omits {requested_operation}; insert the cue before "
                f"the next clause is produced"
                + (" (context-fit)" if context_fit else "")
            ),
            confidence=prediction.confidence,
            scheduled_at_ms=scheduled_at,
            expires_at_ms=scheduled_at + self.delivery_window_ms,
            insertion_delay_ms=self.insertion_delay_ms,
            assignment=self.assigner.label,
            base_text=state.text,
            prediction=prediction,
        )


class SteeringDecisionModel(LocalLLMDecisionModel):
    """Trajectory-steering backend: change the speaker's next move, not just fill gaps.

    Same fail-closed JSON contract as the local LLM backend, but the prompt
    asks for the one move that best advances or redirects the reasoning
    trajectory — repeating a move, anchoring on one frame, jumping to a rival
    before it is earned — rather than the next unplayed item on a checklist.
    Silence is the default when the current trajectory is already productive.
    This backend exists to be compared, not trusted: whether steering a
    speaker's reasoning helps is a live-session question nobody has answered.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.label = f"steering({self.model})"

    def _build_messages(self, state: SyncState) -> tuple[str, str] | None:
        prediction = state.prediction
        assert prediction is not None
        covered_set, missing = self._missing(state)
        if not missing:
            return None
        covered = ", ".join(sorted(covered_set)) or "なし"
        recent = ", ".join(sorted(state.recent_operations)) or "なし"
        context = " / ".join(state.context[-4:]) or "なし"
        system = (
            "あなたは話し手の発話中に短い日本語の cue を差し込む推論ファシリテーターです。"
            "目的は「抜けを埋める」ではなく「この話し手の推論軌道を、より有用な方向へ1手だけ押し進める」"
            "ことです。同じ操作を繰り返している、1つの枠に固執している、根拠なく飛躍している、"
            "逆を検討していない、といった軌道の癖を見抜いて、その1手で軌道が変わる促しを入れます。"
            "現時点の軌道で十分有用な場合や、割り込むと思考を妨げる場合は SILENCE を選びます。"
            "必ずJSONオブジェクトだけを返し、説明を含めないこと。"
        )
        user = (
            f"【直前までの文脈】\n{context}\n\n"
            f"【現在の発話】\n{state.text}\n\n"
            f"【次に来ると予測される節】\n{prediction.span}\n"
            f"【予測される操作】\n{', '.join(prediction.operations) or 'なし'}\n"
            f"【既に現れた操作】\n{covered}\n"
            f"【直近の操作】\n{recent}\n"
            f"【未登場の操作】\n{', '.join(missing)}\n\n"
            "【指示】この話し手の推論軌道を1手だけ伸ばすか、逆に振る cue を、"
            f"【未登場の操作】の中から選んでください。{self.max_characters}文字以内の日本語の cue を返し、"
            "どの癖に対してどんな狙いの一手なのかを reason に一言で書いてください。"
            "confidence は 0.0〜1.0 の実数です。軌道に問題がないなら SILENCE にしてください。\n\n"
            "【出力形式】\n"
            '{"action":"CUE","operation":"invert","cue":"逆は？",'
            '"confidence":0.7,"reason":"結論を出す前に反対側を試すべき"}\n'
            'または {"action":"SILENCE","confidence":0.7,"reason":"..."}'
        )
        return system, user


class LinearDecisionModel:
    """Fast learned policy: score each missing operation with a linear model.

    The model is trained offline from logged states and carryover outcomes.
    Inference is a dot product over a fixed feature vector, so it can run at
    the boundary without the latency of a local LLM. It deliberately only
    chooses among operations the prediction-gap policy would consider.
    """

    def __init__(
        self,
        model: LinearPolicyModel,
        assigner: CueAssigner,
        *,
        delivery_window_ms: int = 1_200,
        insertion_delay_ms: int = 0,
        max_characters: int = 8,
    ) -> None:
        if delivery_window_ms < 1:
            raise ValueError("delivery_window_ms must be positive")
        if insertion_delay_ms < 0:
            raise ValueError("insertion_delay_ms must be non-negative")
        self.model = model
        self.assigner = assigner
        self.delivery_window_ms = delivery_window_ms
        self.insertion_delay_ms = insertion_delay_ms
        self.max_characters = max_characters
        self.label = f"linear-policy({model.label})"

    def decide(self, state: SyncState) -> Intervention | None:
        prediction = state.prediction
        if not state.at_boundary or prediction is None or state.diverged:
            return None
        if not _span_has_content(prediction.span):
            return None
        observation = PolicyObservation.from_state(state)
        candidates = observation.missing_operations()
        chosen = self.model.choose(observation, candidates)
        if chosen is None:
            return None
        requested_operation, score = chosen
        cue = self.assigner.assign(requested_operation)
        operation = operation_for_cue(cue) or "unmapped"
        scheduled_at = state.now_ms + self.insertion_delay_ms
        digest = hashlib.sha256(
            f"{state.utterance_id}:{state.now_ms}:{operation}:{cue}:{state.text}".encode()
        ).hexdigest()[:12]
        return Intervention(
            intervention_id=f"cue-{digest}",
            detected_at_ms=state.now_ms,
            operation=operation,
            requested_operation=requested_operation,
            cue=cue,
            reason=(
                f"linear policy chose {requested_operation} "
                f"(carryover probability {score:.2f})"
            ),
            confidence=score,
            scheduled_at_ms=scheduled_at,
            expires_at_ms=scheduled_at + self.delivery_window_ms,
            insertion_delay_ms=self.insertion_delay_ms,
            assignment=self.assigner.label,
            base_text=state.text,
            prediction=prediction,
        )


class RandomOperationDecisionModel:
    """Exploration arm: keep the timing policy, randomize the operation.

    The wrapped model still decides *when* a cue is warranted. When it fires,
    one operation is drawn uniformly from the same missing-operation set the
    policy considers, and the cue for that operation is assigned normally. The
    point is honest logging: ``requested_operation`` is what the policy wanted,
    ``operation`` is what the participant actually heard. Without this arm,
    untried operations have no outcome and the comparison stays correlational.
    """

    def __init__(
        self,
        inner: DecisionModel,
        assigner: CueAssigner,
        *,
        seed: int = 0,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.inner = inner
        self.assigner = assigner
        self.seed = seed
        self._random = random.Random(seed)
        self._lock = threading.Lock()
        self.label = f"{inner.label}+random-operation(seed={seed})"

    def warmup(self) -> None:
        warmup = getattr(self.inner, "warmup", None)
        if warmup is not None:
            warmup()

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if reset is not None:
            reset()

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()

    def decide(self, state: SyncState) -> Intervention | None:
        base = self.inner.decide(state)
        if base is None:
            return None
        candidates = PolicyObservation.from_state(state).missing_operations()
        if not candidates:
            return None
        with self._lock:
            operation = self._random.choice(list(candidates))
        cue = self.assigner.assign(operation)
        actual = operation_for_cue(cue) or "unmapped"
        digest = hashlib.sha256(
            f"{state.utterance_id}:{state.now_ms}:{actual}:{cue}:{state.text}".encode()
        ).hexdigest()[:12]
        return replace(
            base,
            intervention_id=f"cue-{digest}",
            operation=actual,
            cue=cue,
            assignment=(
                f"random-operation(seed={self.seed};wording={self.assigner.label})"
            ),
            reason=(
                f"randomized operation {actual} "
                f"(policy requested {base.requested_operation})"
            ),
        )


def build_assigner(
    name: str,
    *,
    seed: int = 0,
    cue: str | None = None,
    max_characters: int = 8,
) -> CueAssigner:
    if name == "canonical":
        return CanonicalAssigner(max_characters=max_characters)
    if name == "alternate":
        return AlternateAssigner(max_characters=max_characters)
    if name == "random":
        return RandomAssigner(seed=seed, max_characters=max_characters)
    if name == "fixed":
        if cue is None:
            raise ValueError("fixed assignment requires a cue")
        return FixedAssigner(cue)
    raise ValueError(f"unknown assigner: {name}")


def _span_has_content(span: str) -> bool:
    return any(character.isalnum() for character in span)


def operation_priority() -> Mapping[str, int]:
    return {operation: index for index, operation in enumerate(PRIORITY)}
