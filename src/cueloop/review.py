from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from typing import Iterable

from .llm import CompletionFn, PromptRequest


class EpistemicStatus(StrEnum):
    FACT_CLAIM = "FACT_CLAIM"
    INFERENCE = "INFERENCE"
    HYPOTHESIS = "HYPOTHESIS"


@dataclass(frozen=True)
class ClaimReview:
    text: str
    status: EpistemicStatus
    premises: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]
    counterexamples: tuple[str, ...]
    verification_needed: bool


@dataclass(frozen=True)
class SessionReview:
    claims: tuple[ClaimReview, ...]
    unresolved_questions: tuple[str, ...]
    structural_gaps: tuple[str, ...]
    error: str | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        for claim in payload["claims"]:
            claim["status"] = claim["status"].value if hasattr(claim["status"], "value") else claim["status"]
        return json.dumps(payload, ensure_ascii=False, indent=2)


REVIEW_SYSTEM_PROMPT = """You review a spoken reasoning session after it ends.
Do not rewrite it into a better answer and do not solve the user's problem.
Extract the user's reasoning structure so the user can inspect and repair it.

For each substantive claim classify epistemic status as:
- FACT_CLAIM: externally checkable assertion about the world.
- INFERENCE: conclusion drawn from premises in the session.
- HYPOTHESIS: tentative explanatory or predictive proposition.

For every claim identify only information actually present or logically required:
- premises
- assumptions
- evidence mentioned by the user
- counterexamples mentioned by the user
- whether external verification is needed

Also list unresolved questions and structural gaps such as undefined comparison metrics, missing causal links, scope ambiguity, or unsupported generalization.
Never invent sources or evidence. Never silently correct the user.
Return JSON only with exactly these top-level keys:
{
  "claims": [
    {
      "text": "...",
      "status": "FACT_CLAIM|INFERENCE|HYPOTHESIS",
      "premises": [],
      "assumptions": [],
      "evidence": [],
      "counterexamples": [],
      "verification_needed": true
    }
  ],
  "unresolved_questions": [],
  "structural_gaps": []
}
"""


class SessionReviewer:
    def __init__(self, complete: CompletionFn) -> None:
        self._complete = complete

    def review(self, turns: Iterable[str]) -> SessionReview:
        clean = [" ".join(turn.strip().split()) for turn in turns if turn.strip()]
        if not clean:
            return SessionReview((), (), (), error="session contains no reasoning turns")
        numbered = "\n".join(f"{i + 1}. {turn}" for i, turn in enumerate(clean))
        request = PromptRequest(
            system=REVIEW_SYSTEM_PROMPT,
            user=f"Reasoning session:\n{numbered}\n\nExtract structure only.",
        )
        try:
            raw = self._complete(request)
        except Exception as exc:
            return SessionReview((), (), (), error=f"review transport failure: {type(exc).__name__}")
        return parse_review_json(raw)


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(item.strip() for item in value if item.strip())


def parse_review_json(raw: str) -> SessionReview:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return SessionReview((), (), (), error="review returned invalid JSON")
    if not isinstance(payload, dict) or set(payload) != {"claims", "unresolved_questions", "structural_gaps"}:
        return SessionReview((), (), (), error="review JSON keys do not match contract")

    claims_raw = payload["claims"]
    if not isinstance(claims_raw, list):
        return SessionReview((), (), (), error="claims must be a list")

    claims: list[ClaimReview] = []
    try:
        for item in claims_raw:
            required = {
                "text", "status", "premises", "assumptions", "evidence",
                "counterexamples", "verification_needed",
            }
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("claim keys do not match contract")
            text = item["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("claim text must be non-empty")
            verify = item["verification_needed"]
            if not isinstance(verify, bool):
                raise ValueError("verification_needed must be boolean")
            claims.append(
                ClaimReview(
                    text=text.strip(),
                    status=EpistemicStatus(item["status"]),
                    premises=_strings(item["premises"], "premises"),
                    assumptions=_strings(item["assumptions"], "assumptions"),
                    evidence=_strings(item["evidence"], "evidence"),
                    counterexamples=_strings(item["counterexamples"], "counterexamples"),
                    verification_needed=verify,
                )
            )
        unresolved = _strings(payload["unresolved_questions"], "unresolved_questions")
        gaps = _strings(payload["structural_gaps"], "structural_gaps")
    except (ValueError, KeyError, TypeError) as exc:
        return SessionReview((), (), (), error=f"invalid review contract: {exc}")

    return SessionReview(tuple(claims), unresolved, gaps)
