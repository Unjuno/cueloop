from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import subprocess
import threading
from typing import Callable, Sequence

from .core import (
    Action,
    Analysis,
    Evidence,
    EvidenceKind,
    HeuristicAnalyzer,
    SessionContext,
    State,
    safe_analysis,
)


SYSTEM_PROMPT = """You are CUELOOP's semantic analyzer.
You do not answer the user's problem.
You only classify the current reasoning turn and propose at most one minimal intervention.

Allowed states: PRODUCTIVE, RETRIEVAL_FAILURE, STRATEGY_FAILURE, FIXATION, STRUCTURE_GAP, OVERLOAD.
Allowed actions: SILENCE, BACKCHANNEL, QUESTION, MIRROR, PERTURB, CONSTRAIN, RETRIEVE, VERIFY.

Rules:
- Prefer SILENCE when reasoning is productive or uncertainty is high.
- BACKCHANNEL only preserves the user's turn; it is not praise or agreement.
- QUESTION, MIRROR, PERTURB, and CONSTRAIN must contain one short cognitive operation in cue.
- RETRIEVE requires one or more explicit source_refs and will otherwise be suppressed.
- OVERLOAD and PRODUCTIVE may only propose SILENCE.
- Never provide a solution, explanation, recommendation, or completed sentence.
- The cue must be a SHORT cognitive cue of at most 15 Japanese characters. Use the canonical cue set directly where it fits: 何に依存？ 具体的には？ 逆なら？ どこで破れる？ 結論は？ 前提は？ 根拠は？ 条件は？ 具体例は？ 反例は？ それで？ だから？ Examples of VALID short cues: 何に依存？ 逆なら？ 具体的には？ 結論は？ どこで破れる？ Examples of INVALID over-long cues: 性能について何か具体的なことを教えて。
- Evidence must describe observable properties of the supplied text or recent turns, not hidden mental states.
- Recent context is bounded and may be incomplete. Do not infer stable traits from it.

Return JSON only with exactly these keys:
{
  "state": "PRODUCTIVE|RETRIEVAL_FAILURE|STRATEGY_FAILURE|FIXATION|STRUCTURE_GAP|OVERLOAD",
  "confidence": 0.0,
  "evidence": [{"kind":"TRANSCRIPT|TIMING|PROSODY|INTERACTION_HISTORY|USER_SIGNAL","description":"observable reason","source_span":null}],
  "proposed_action": "SILENCE|BACKCHANNEL|QUESTION|MIRROR|PERTURB|CONSTRAIN|RETRIEVE|VERIFY",
  "cue": null,
  "source_refs": []
}
"""


@dataclass(frozen=True)
class PromptRequest:
    system: str
    user: str


CompletionFn = Callable[[PromptRequest], str]


class CommandCompletion:
    """Provider-neutral subprocess adapter.

    Sends {"system": ..., "user": ...} JSON to stdin. The command must emit
    only the analyzer-result JSON to stdout. This deliberately keeps provider
    credentials and SDKs outside the CUELOOP core.
    """

    def __init__(self, command: Sequence[str], timeout_seconds: float = 8.0) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = False

    def __call__(self, request: PromptRequest) -> str:
        process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process
            cancel_requested = self._cancel_requested
        if cancel_requested:
            process.terminate()
        try:
            stdout, stderr = process.communicate(
                json.dumps(asdict(request), ensure_ascii=False), timeout=self._timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise RuntimeError("analyzer command timed out") from exc
        finally:
            with self._lock:
                self._process = None
                self._cancel_requested = False
        if process.returncode != 0:
            stderr = stderr.strip()[:240]
            raise RuntimeError(f"analyzer command failed ({process.returncode}): {stderr}")
        return stdout.strip()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()


class LLMAnalyzer:
    """Provider-neutral semantic analyzer using a strict JSON contract."""

    def __init__(self, complete: CompletionFn) -> None:
        self._complete = complete

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        recent = context.render() if context is not None else "(no previous turns)"
        request = PromptRequest(
            system=SYSTEM_PROMPT,
            user=(
                "Recent reasoning turns (oldest to newest):\n"
                f"{recent}\n\n"
                "Current reasoning turn:\n"
                f"{text}\n\n"
                "Analyze the current turn in light of the bounded recent context."
            ),
        )
        try:
            raw = self._complete(request)
        except Exception as exc:  # provider failures must never force intervention
            return safe_analysis(f"analyzer transport failure: {type(exc).__name__}")
        return parse_analysis_json(raw)

    def cancel(self) -> None:
        cancel = getattr(self._complete, "cancel", None)
        if callable(cancel):
            cancel()


class HybridAnalyzer:
    """Fast-path heuristic with optional slow-path LLM refinement.

    Returns an instant heuristic decision when it is safe and confident. Only
    when the heuristic is uncertain (low confidence or SILENCE with no clear
    state) does it consult the language model. This keeps live latency low
    while preserving semantic depth on ambiguous turns.
    """

    def __init__(self, llm: LLMAnalyzer | None = None) -> None:
        self._heuristic = HeuristicAnalyzer()
        self._llm = llm

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        heuristic = self._heuristic.analyze(text, context)
        if self._llm is None:
            return heuristic
        if heuristic.confidence >= 0.7 and heuristic.proposed_action is not Action.SILENCE:
            return heuristic
        if heuristic.state is State.PRODUCTIVE and heuristic.confidence >= 0.7:
            return heuristic
        refined = self._llm.analyze(text, context)
        # Prefer the safer of the two: never let a refined analysis propose an
        # intervention that the heuristic would have suppressed outright.
        if heuristic.state in {State.PRODUCTIVE, State.OVERLOAD} and refined.state not in {
            State.PRODUCTIVE,
            State.OVERLOAD,
        }:
            return heuristic
        if refined.confidence >= heuristic.confidence:
            return refined
        return heuristic

    def cancel(self) -> None:
        if self._llm is not None:
            self._llm.cancel()


def parse_analysis_json(raw: str) -> Analysis:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return safe_analysis("analyzer returned invalid JSON")

    if not isinstance(payload, dict):
        return safe_analysis("analyzer JSON must be an object")

    allowed_keys = {"state", "confidence", "evidence", "proposed_action", "cue", "source_refs"}
    if set(payload) != allowed_keys:
        return safe_analysis("analyzer JSON keys do not match contract")

    try:
        state = State(payload["state"])
        action = Action(payload["proposed_action"])
        confidence = float(payload["confidence"])
    except (ValueError, TypeError, KeyError):
        return safe_analysis("analyzer returned invalid enum or confidence")

    if not 0.0 <= confidence <= 1.0:
        return safe_analysis("analyzer confidence out of range")

    evidence_raw = payload["evidence"]
    evidence_items: list[Evidence] = []
    if not isinstance(evidence_raw, list):
        return safe_analysis("analyzer evidence must be an array")
    try:
        for item in evidence_raw:
            if not isinstance(item, dict) or set(item) != {"kind", "description", "source_span"}:
                raise ValueError
            description = item["description"]
            source_span = item["source_span"]
            if not isinstance(description, str) or not description.strip():
                raise ValueError
            if source_span is not None and not isinstance(source_span, str):
                raise ValueError
            evidence_items.append(
                Evidence(EvidenceKind(item["kind"]), description.strip(), source_span)
            )
    except (TypeError, ValueError):
        return safe_analysis("analyzer evidence does not match the typed contract")
    evidence = tuple(evidence_items)
    if not evidence:
        return safe_analysis("analyzer evidence is empty")

    cue_raw = payload["cue"]
    if cue_raw is not None and not isinstance(cue_raw, str):
        return safe_analysis("analyzer cue must be a string or null")
    cue = cue_raw.strip() if isinstance(cue_raw, str) else None

    source_refs_raw = payload["source_refs"]
    if not isinstance(source_refs_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in source_refs_raw
    ):
        return safe_analysis("analyzer source_refs must be a list of non-empty strings")
    source_refs = tuple(item.strip() for item in source_refs_raw)

    text_actions = {Action.QUESTION, Action.MIRROR, Action.PERTURB, Action.CONSTRAIN, Action.RETRIEVE}
    if action in text_actions and not cue:
        return safe_analysis(f"{action.value} action requires a cue")
    if action not in text_actions and cue is not None:
        return safe_analysis(f"{action.value} action must not include a cue")
    if state in {State.PRODUCTIVE, State.OVERLOAD} and action is not Action.SILENCE:
        return safe_analysis(f"{state.value} state may only propose SILENCE")
    if action is Action.RETRIEVE and not source_refs:
        return safe_analysis("RETRIEVE action requires source_refs")
    if action is not Action.RETRIEVE and source_refs:
        return safe_analysis("only RETRIEVE may include source_refs")

    return Analysis(
        state=state,
        confidence=confidence,
        evidence=evidence,
        proposed_action=action,
        cue=cue,
        source_refs=source_refs,
    )
