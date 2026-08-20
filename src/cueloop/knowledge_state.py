"""Knowledge acquisition state tracking for CUELOOP.

This module implements Priority C: Human Knowledge State. Rather than tracking
what the user "knows" as a stable capability score, it tracks *how* a concept
was acquired and whether the user can later produce it independently.

Design principles (from instruction #1, #9, #10):
- Never store stable ability scores or personality traits.
- Evidence, confidence, timestamps, and expiry are mandatory.
- Spontaneous reuse is detected, not assumed.
- The state is a review candidate, not ground truth.
- scientific claims about cognitive enhancement are not made.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
from typing import Sequence


class KnowledgeState(StrEnum):
    """Lifecycle of a concept the user was exposed to.

    The progression is: EXTERNALLY_SUPPLIED -> CUE_ASSISTED ->
    SPONTANEOUSLY_REUSED -> EXPLICITLY_RETRAINED -> EXPIRED.

    USER_RECONSTRUCTED is a possible intermediate state that is not
    auto-detected; it requires explicit user feedback.
    """

    EXTERNALLY_SUPPLIED = "externally_supplied"
    CUE_ASSISTED = "cue_assisted"
    USER_RECONSTRUCTED = "user_reconstructed"
    SPONTANEOUSLY_REUSED = "spontaneously_reused"
    EXPLICITLY_RETRAINED = "explicitly_retained"
    EXPIRED = "expired"


# Valid state transitions.  An invalid transition is silently ignored.
_VALID_TRANSITIONS: dict[KnowledgeState, frozenset[KnowledgeState]] = {
    KnowledgeState.EXTERNALLY_SUPPLIED: frozenset({
        KnowledgeState.CUE_ASSISTED,
        KnowledgeState.SPONTANEOUSLY_REUSED,
        KnowledgeState.EXPLICITLY_RETRAINED,
        KnowledgeState.EXPIRED,
    }),
    KnowledgeState.CUE_ASSISTED: frozenset({
        KnowledgeState.USER_RECONSTRUCTED,
        KnowledgeState.SPONTANEOUSLY_REUSED,
        KnowledgeState.EXPLICITLY_RETRAINED,
        KnowledgeState.EXPIRED,
    }),
    KnowledgeState.USER_RECONSTRUCTED: frozenset({
        KnowledgeState.SPONTANEOUSLY_REUSED,
        KnowledgeState.EXPLICITLY_RETRAINED,
        KnowledgeState.EXPIRED,
    }),
    KnowledgeState.SPONTANEOUSLY_REUSED: frozenset({
        KnowledgeState.EXPLICITLY_RETRAINED,
        KnowledgeState.EXPIRED,
    }),
    KnowledgeState.EXPLICITLY_RETRAINED: frozenset({
        KnowledgeState.EXPIRED,
    }),
    KnowledgeState.EXPIRED: frozenset(),
}


def _concept_key(text: str) -> str:
    """Derive a stable key from a concept description.

    The key is a SHA-256 hash of the normalized text, truncated to 16 hex
    characters.  Normalization lowercases and collapses whitespace.
    """
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def extract_keywords(text: str, max_keywords: int = 3) -> tuple[str, ...]:
    """Extract candidate keywords from a short cue or concept text.

    Particles and common stop words are removed.  The result is at most
    *max_keywords* items, each at most 8 characters long.
    """
    stop = {
        "は", "が", "を", "に", "で", "と", "も", "の", "から", "まで",
        "より", "など", "って", "という", "ということ", "というか",
        "し", "て", "だ", "な", "ます", "です", "した", "して",
        "いる", "ある", "なる", "できる", "おる",
        "する", "される",
        "この", "その", "あの", "どの",
        "こと", "もの", "ため", "よう",
        "に", "な", "ら",
    }
    # Split on particles that commonly separate words in Japanese
    import re as _re
    split_tokens = _re.split(r"[はがをにでともからまでなど]|する|される", text)
    tokens: list[str] = []
    for chunk in split_tokens:
        tokens.extend(_re.findall(r"[\w\u3040-\u9FFF]{2,8}", chunk))
    keywords: list[str] = []
    for token in tokens:
        if token in stop:
            continue
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return tuple(keywords)


def speech_contains_keywords(speech_text: str, keywords: Sequence[str]) -> bool:
    """Check whether user speech contains evidence of a keyword concept.

    This is a deliberately conservative check: all keywords must appear
    (AND logic) to reduce false positives from ASR noise.
    """
    if not keywords:
        return False
    normalized = speech_text.strip().lower()
    return all(kw.lower() in normalized for kw in keywords)


@dataclass
class KnowledgeAcquisition:
    """A single concept tracking record.

    This is a review candidate, not a measurement of stable ability.
    The ``confidence`` field reflects detection reliability, not the
    user's competence.
    """

    concept_key: str
    concept_text: str
    state: KnowledgeState
    confidence: float
    keywords: tuple[str, ...]
    source_session_id: str
    observed_at: str
    expires_at: str
    last_state_change_at: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.concept_text.strip():
            raise ValueError("concept_text must not be empty")
        if not self.keywords:
            raise ValueError("keywords must not be empty")
        observed = datetime.fromisoformat(self.observed_at)
        expires = datetime.fromisoformat(self.expires_at)
        if observed.tzinfo is None or expires.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if expires <= observed:
            raise ValueError("expires_at must be after observed_at")

    def is_expired(self, now: datetime | None = None) -> bool:
        reference = now or datetime.now(timezone.utc)
        return datetime.fromisoformat(self.expires_at) <= reference

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": self.concept_key,
            "concept_text": self.concept_text,
            "state": self.state.value,
            "confidence": self.confidence,
            "keywords": list(self.keywords),
            "source_session_id": self.source_session_id,
            "observed_at": self.observed_at,
            "expires_at": self.expires_at,
            "last_state_change_at": self.last_state_change_at,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: object) -> "KnowledgeAcquisition":
        if not isinstance(payload, dict):
            raise ValueError("knowledge acquisition must be a dict")
        required = {
            "concept_key", "concept_text", "state", "confidence",
            "keywords", "source_session_id", "observed_at", "expires_at",
            "last_state_change_at",
        }
        if not required <= set(payload):
            raise ValueError(f"missing fields: {required - set(payload)}")
        return cls(
            concept_key=str(payload["concept_key"]),
            concept_text=str(payload["concept_text"]),
            state=KnowledgeState(payload["state"]),
            confidence=float(payload["confidence"]),
            keywords=tuple(str(k) for k in payload["keywords"]),
            source_session_id=str(payload["source_session_id"]),
            observed_at=str(payload["observed_at"]),
            expires_at=str(payload["expires_at"]),
            last_state_change_at=str(payload["last_state_change_at"]),
            evidence=tuple(str(e) for e in payload.get("evidence", [])),
        )


class KnowledgeStore:
    """Tracks knowledge acquisitions across sessions.

    The store never stores raw transcripts.  It stores concept keys,
    keyword lists, state transitions, and metadata.
    """

    def __init__(
        self,
        acquisitions: list[KnowledgeAcquisition] | None = None,
        retention_days: int = 90,
    ) -> None:
        self.acquisitions = acquisitions or []
        self.retention_days = retention_days
        self._index: dict[str, KnowledgeAcquisition] = {
            ac.concept_key: ac for ac in self.acquisitions
        }

    def record_cue_delivery(
        self,
        cue_text: str,
        *,
        source_session_id: str,
        now: datetime | None = None,
        confidence: float = 0.7,
        retention_days: int | None = None,
    ) -> KnowledgeAcquisition:
        """Record that a cue was delivered to the user.

        If the concept already exists, the state is not downgraded.
        """
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        days = retention_days or self.retention_days
        key = _concept_key(cue_text)
        existing = self._index.get(key)
        if existing is not None:
            return existing
        keywords = extract_keywords(cue_text)
        if not keywords:
            keywords = (cue_text.strip()[:8],)
        acquisition = KnowledgeAcquisition(
            concept_key=key,
            concept_text=cue_text.strip(),
            state=KnowledgeState.CUE_ASSISTED,
            confidence=confidence,
            keywords=keywords,
            source_session_id=source_session_id,
            observed_at=current.isoformat(),
            expires_at=(current + timedelta(days=days)).isoformat(),
            last_state_change_at=current.isoformat(),
            evidence=(f"cue delivered: {cue_text.strip()}",),
        )
        self.acquisitions.append(acquisition)
        self._index[key] = acquisition
        return acquisition

    def detect_spontaneous_reuse(
        self,
        speech_text: str,
        *,
        source_session_id: str = "",
        now: datetime | None = None,
    ) -> list[KnowledgeAcquisition]:
        """Check user speech for spontaneous reuse of previously cued concepts.

        Returns a list of acquisitions whose state was upgraded.  Each
        returned acquisition has ``evidence`` updated with the speech text
        that triggered detection.
        """
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        upgraded: list[KnowledgeAcquisition] = []
        for acquisition in list(self.acquisitions):
            if acquisition.is_expired(current):
                self._transition(acquisition, KnowledgeState.EXPIRED, current)
                continue
            if acquisition.state in {
                KnowledgeState.EXPLICITLY_RETRAINED,
                KnowledgeState.EXPIRED,
            }:
                continue
            if not speech_contains_keywords(speech_text, acquisition.keywords):
                continue
            if acquisition.state is KnowledgeState.SPONTANEOUSLY_REUSED:
                continue
            evidence = acquisition.evidence + (f"spontaneous reuse in: {speech_text.strip()[:120]}",)
            new = KnowledgeAcquisition(
                concept_key=acquisition.concept_key,
                concept_text=acquisition.concept_text,
                state=KnowledgeState.SPONTANEOUSLY_REUSED,
                confidence=min(1.0, acquisition.confidence + 0.1),
                keywords=acquisition.keywords,
                source_session_id=acquisition.source_session_id or source_session_id,
                observed_at=acquisition.observed_at,
                expires_at=acquisition.expires_at,
                last_state_change_at=current.isoformat(),
                evidence=evidence,
            )
            idx = self.acquisitions.index(acquisition)
            self.acquisitions[idx] = new
            self._index[new.concept_key] = new
            upgraded.append(new)
        return upgraded

    def get_fading_level(self, cue_text: str) -> int:
        """Return the fading level for a cue based on its acquisition state.

        Returns:
            0 - no fading (full cue)
            1 - keyword-only cue
            2 - SILENCE (suppress intervention)
        """
        key = _concept_key(cue_text)
        acquisition = self._index.get(key)
        if acquisition is None:
            return 0
        if acquisition.is_expired():
            return 2
        if acquisition.state is KnowledgeState.SPONTANEOUSLY_REUSED:
            return 1
        if acquisition.state is KnowledgeState.EXPLICITLY_RETRAINED:
            return 2
        return 0

    def get_acquisition(self, cue_text: str) -> KnowledgeAcquisition | None:
        key = _concept_key(cue_text)
        return self._index.get(key)

    def prune_expired(self, now: datetime | None = None) -> int:
        """Remove expired acquisitions and return the count removed."""
        reference = now or datetime.now(timezone.utc)
        before = len(self.acquisitions)
        self.acquisitions = [
            ac for ac in self.acquisitions if not ac.is_expired(reference)
        ]
        self._index = {ac.concept_key: ac for ac in self.acquisitions}
        return before - len(self.acquisitions)

    def to_list(self) -> list[dict[str, object]]:
        return [ac.to_dict() for ac in self.acquisitions]

    @classmethod
    def from_list(cls, items: list[object], retention_days: int = 90) -> "KnowledgeStore":
        return cls(
            acquisitions=[KnowledgeAcquisition.from_dict(item) for item in items],
            retention_days=retention_days,
        )

    def _transition(
        self,
        acquisition: KnowledgeAcquisition,
        new_state: KnowledgeState,
        now: datetime,
    ) -> None:
        """Attempt a state transition.  Invalid transitions are silently ignored."""
        allowed = _VALID_TRANSITIONS.get(acquisition.state, frozenset())
        if new_state not in allowed:
            return
        updated = KnowledgeAcquisition(
            concept_key=acquisition.concept_key,
            concept_text=acquisition.concept_text,
            state=new_state,
            confidence=acquisition.confidence,
            keywords=acquisition.keywords,
            source_session_id=acquisition.source_session_id,
            observed_at=acquisition.observed_at,
            expires_at=acquisition.expires_at,
            last_state_change_at=now.isoformat(),
            evidence=acquisition.evidence,
        )
        idx = self.acquisitions.index(acquisition)
        self.acquisitions[idx] = updated
        self._index[updated.concept_key] = updated
