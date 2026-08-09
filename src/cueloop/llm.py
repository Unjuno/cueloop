from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import subprocess
from typing import Callable, Sequence

from .core import Action, Analysis, SessionContext, State, safe_analysis


SYSTEM_PROMPT = """You are CUELOOP's semantic analyzer.
You do not answer the user's problem.
You only classify the current reasoning turn and propose at most one minimal intervention.

Allowed states: FLOWING, PAUSED, STUCK.
Allowed actions: SILENCE, CONTINUE, CUE.

Primary objective:
Preserve human-generated reasoning.  The best intervention is often no intervention.

Rules:
- Prefer SILENCE when reasoning is productive, self-correcting, exploratory, or uncertainty is high.
- Never classify STUCK solely because of filler words, repetitions, transcription artifacts, grammar errors, or disfluency. Those may be ASR noise.
- STUCK requires semantic evidence that the reasoning trajectory is looping, blocked, or missing a next operation across the current turn or recent bounded context.
- PAUSED means the thought is incomplete but not clearly blocked.
- CONTINUE only preserves the user's turn; it is not praise, agreement, explanation, or content assistance.
- CUE must trigger exactly one reasoning operation without supplying its result.
- Good CUE operations include: define a term, state the conclusion, identify a dependency, name a condition, produce a counterexample, narrow scope, identify evidence, compare an alternative, or surface an assumption.
- Never complete the user's sentence, provide a missing fact, propose the final answer, recommend a solution, summarize the whole discussion, or explain a concept.
- Prefer a cue of 2-8 words. Keep it under 16 Japanese characters when practical.
- Evidence must describe observable properties of supplied text/recent turns, never hidden mental states or stable traits.
- Recent context is bounded and may be incomplete. Do not infer personality, intelligence, diagnosis, or enduring ability.
- If the current turn already performs the operation you would cue, choose SILENCE.

Examples of acceptable cues:
- "何に依存？"
- "反例は？"
- "前提は？"
- "何が証拠？"
- "範囲は？"
- "逆なら？"

Examples of unacceptable cues:
- "それは○○だからです"
- "次は市場規模を考えましょう"
- "つまりあなたの結論は○○です"
- any completed answer or factual explanation

Return JSON only with exactly these keys:
{
  "state": "FLOWING|PAUSED|STUCK",
  "confidence": 0.0,
  "evidence": ["observable reason"],
  "proposed_action": "SILENCE|CONTINUE|CUE",
  "cue": null
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
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._command = tuple(command)
        self._timeout_seconds = timeout_seconds

    def __call__(self, request: PromptRequest) -> str:
        process = subprocess.run(
            self._command,
            input=json.dumps(asdict(request), ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()[:240]
            raise RuntimeError(f"analyzer command failed ({process.returncode}): {stderr}")
        return process.stdout.strip()


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
                "Analyze only whether a minimal intervention is justified now. "
                "If the user can plausibly continue productively without help, choose SILENCE."
            ),
        )
        try:
            raw = self._complete(request)
        except Exception as exc:  # provider failures must never force intervention
            return safe_analysis(f"analyzer transport failure: {type(exc).__name__}")
        return parse_analysis_json(raw)


def parse_analysis_json(raw: str) -> Analysis:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return safe_analysis("analyzer returned invalid JSON")

    if not isinstance(payload, dict):
        return safe_analysis("analyzer JSON must be an object")

    allowed_keys = {"state", "confidence", "evidence", "proposed_action", "cue"}
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
    if not isinstance(evidence_raw, list) or not all(isinstance(item, str) for item in evidence_raw):
        return safe_analysis("analyzer evidence must be a string list")
    evidence = tuple(item.strip() for item in evidence_raw if item.strip())
    if not evidence:
        return safe_analysis("analyzer evidence is empty")

    cue_raw = payload["cue"]
    if cue_raw is not None and not isinstance(cue_raw, str):
        return safe_analysis("analyzer cue must be a string or null")
    cue = cue_raw.strip() if isinstance(cue_raw, str) else None

    if action is Action.CUE and not cue:
        return safe_analysis("CUE action requires a cue")
    if action is not Action.CUE and cue is not None:
        return safe_analysis("non-CUE action must not include a cue")
    if state is State.FLOWING and action is not Action.SILENCE:
        return safe_analysis("FLOWING state may only propose SILENCE")

    return Analysis(
        state=state,
        confidence=confidence,
        evidence=evidence,
        proposed_action=action,
        cue=cue,
    )
