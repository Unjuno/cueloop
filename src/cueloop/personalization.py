from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .core import Action, Analysis, Decision, InterventionController, State


class FeedbackRating(StrEnum):
    HELPFUL = "helpful"
    NEUTRAL = "neutral"
    DISRUPTIVE = "disruptive"


@dataclass
class InterventionStats:
    delivered: int = 0
    opportunities: int = 0
    helpful: int = 0
    neutral: int = 0
    disruptive: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> "InterventionStats":
        if not isinstance(payload, dict) or set(payload) != {
            "delivered", "opportunities", "helpful", "neutral", "disruptive"
        }:
            raise ValueError("invalid intervention statistics")
        values = {key: int(value) for key, value in payload.items()}
        if any(value < 0 for value in values.values()):
            raise ValueError("intervention statistics must be non-negative")
        if values["delivered"] > values["opportunities"]:
            raise ValueError("delivered interventions cannot exceed opportunities")
        if values["helpful"] + values["neutral"] + values["disruptive"] > values["delivered"]:
            raise ValueError("reviewed interventions cannot exceed delivered interventions")
        return cls(**values)


@dataclass(frozen=True)
class StateObservation:
    state: State
    confidence: float
    evidence_hash: str
    observed_at: str
    expires_at: str

    @classmethod
    def from_dict(cls, payload: object) -> "StateObservation":
        if not isinstance(payload, dict) or set(payload) != {
            "state", "confidence", "evidence_hash", "observed_at", "expires_at"
        }:
            raise ValueError("invalid state observation")
        observation = cls(
            State(payload["state"]),
            float(payload["confidence"]),
            str(payload["evidence_hash"]),
            str(payload["observed_at"]),
            str(payload["expires_at"]),
        )
        if not 0 <= observation.confidence <= 1:
            raise ValueError("state confidence out of range")
        observed = datetime.fromisoformat(observation.observed_at)
        expires = datetime.fromisoformat(observation.expires_at)
        if observed.tzinfo is None or expires.tzinfo is None or expires <= observed:
            raise ValueError("state observation timestamps are invalid")
        if len(observation.evidence_hash) != 64 or any(
            character not in "0123456789abcdef" for character in observation.evidence_hash
        ):
            raise ValueError("state evidence hash is invalid")
        return observation


@dataclass(frozen=True)
class RetainedStructure:
    id: str
    kind: str
    text: str
    source_session_id: str
    retained_at: str
    expires_at: str

    @classmethod
    def from_dict(cls, payload: object) -> "RetainedStructure":
        if not isinstance(payload, dict) or set(payload) != {
            "id", "kind", "text", "source_session_id", "retained_at", "expires_at"
        }:
            raise ValueError("invalid retained structure")
        if not all(isinstance(value, str) for value in payload.values()):
            raise ValueError("retained structure fields must be strings")
        item = cls(**payload)
        if not all((item.id, item.kind, item.text, item.source_session_id)):
            raise ValueError("retained structure fields must not be empty")
        retained = datetime.fromisoformat(item.retained_at)
        expires = datetime.fromisoformat(item.expires_at)
        if retained.tzinfo is None or expires.tzinfo is None or expires <= retained:
            raise ValueError("retained structure timestamps are invalid")
        return item


class PersonalizationProfile:
    """Stores explicit intervention feedback only; it never models stable user traits."""

    schema_version = 3

    def __init__(
        self,
        stats: dict[str, InterventionStats] | None = None,
        state_observations: list[StateObservation] | None = None,
        retained_structures: list[RetainedStructure] | None = None,
    ) -> None:
        self.stats = stats or {}
        self.state_observations = state_observations or []
        self.retained_structures = retained_structures or []

    @staticmethod
    def key(action: Action, cue: str | None) -> str:
        return f"{action.value}:{cue or ''}"

    def for_intervention(self, action: Action, cue: str | None) -> InterventionStats:
        return self.stats.setdefault(self.key(action, cue), InterventionStats())

    def record_feedback(self, action: Action, cue: str | None, rating: FeedbackRating) -> None:
        stats = self.for_intervention(action, cue)
        if stats.delivered <= stats.helpful + stats.neutral + stats.disruptive:
            raise ValueError("feedback requires an unreviewed delivered intervention")
        setattr(stats, rating.value, getattr(stats, rating.value) + 1)

    def observe_state(
        self,
        analysis: Analysis,
        *,
        observed_at: datetime | None = None,
        retention_days: int = 30,
    ) -> None:
        if not 1 <= retention_days <= 365:
            raise ValueError("state retention_days must be between 1 and 365")
        now = observed_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("state observation time must be timezone-aware")
        evidence_payload = [
            {
                "kind": item.kind.value,
                "description": item.description,
                "source_span": item.source_span,
            }
            for item in analysis.evidence
        ]
        digest = hashlib.sha256(
            json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.state_observations.append(
            StateObservation(
                analysis.state,
                analysis.confidence,
                digest,
                now.isoformat(),
                (now + timedelta(days=retention_days)).isoformat(),
            )
        )
        self.prune_state_observations()

    def prune_state_observations(self, now: datetime | None = None) -> int:
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            raise ValueError("state pruning time must be timezone-aware")
        before = len(self.state_observations)
        self.state_observations = [
            item for item in self.state_observations if datetime.fromisoformat(item.expires_at) > reference
        ]
        return before - len(self.state_observations)

    def recurring_states(self, minimum_count: int = 2) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.state_observations:
            if item.state in {State.PRODUCTIVE, State.OVERLOAD}:
                continue
            counts[item.state.value] = counts.get(item.state.value, 0) + 1
        return {key: value for key, value in sorted(counts.items()) if value >= minimum_count}

    def retain_structure(
        self,
        *,
        kind: str,
        text: str,
        source_session_id: str,
        retained_at: datetime | None = None,
        retention_days: int = 365,
    ) -> RetainedStructure:
        normalized_kind = kind.strip().upper()
        normalized_text = " ".join(text.strip().split())
        normalized_session = source_session_id.strip()
        if not normalized_kind or not normalized_text or not normalized_session:
            raise ValueError("retained structure fields must not be empty")
        if len(normalized_text) > 2_000:
            raise ValueError("retained structure text is too long")
        if not 1 <= retention_days <= 365:
            raise ValueError("structure retention_days must be between 1 and 365")
        now = retained_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("structure retention time must be timezone-aware")
        identifier = "structure-" + hashlib.sha256(
            f"{normalized_session}:{normalized_kind}:{normalized_text}".encode("utf-8")
        ).hexdigest()[:16]
        existing = next((item for item in self.retained_structures if item.id == identifier), None)
        if existing is not None:
            return existing
        item = RetainedStructure(
            identifier,
            normalized_kind,
            normalized_text,
            normalized_session,
            now.isoformat(),
            (now + timedelta(days=retention_days)).isoformat(),
        )
        self.retained_structures.append(item)
        return item

    def forget_structure(self, structure_id: str) -> bool:
        before = len(self.retained_structures)
        self.retained_structures = [
            item for item in self.retained_structures if item.id != structure_id
        ]
        return len(self.retained_structures) < before

    def prune_retained_structures(self, now: datetime | None = None) -> int:
        reference = now or datetime.now(timezone.utc)
        if reference.tzinfo is None:
            raise ValueError("structure pruning time must be timezone-aware")
        before = len(self.retained_structures)
        self.retained_structures = [
            item for item in self.retained_structures
            if datetime.fromisoformat(item.expires_at) > reference
        ]
        return before - len(self.retained_structures)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stats": {
                key: {
                    "delivered": value.delivered,
                    "opportunities": value.opportunities,
                    "helpful": value.helpful,
                    "neutral": value.neutral,
                    "disruptive": value.disruptive,
                }
                for key, value in sorted(self.stats.items())
            },
            "state_observations": [
                {
                    "state": item.state.value,
                    "confidence": item.confidence,
                    "evidence_hash": item.evidence_hash,
                    "observed_at": item.observed_at,
                    "expires_at": item.expires_at,
                }
                for item in self.state_observations
            ],
            "retained_structures": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "text": item.text,
                    "source_session_id": item.source_session_id,
                    "retained_at": item.retained_at,
                    "expires_at": item.expires_at,
                }
                for item in self.retained_structures
            ],
        }

    @classmethod
    def load(cls, path: Path) -> "PersonalizationProfile":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid personalization profile")
        if payload.get("schema_version") == 1 and set(payload) == {"schema_version", "stats"}:
            if not isinstance(payload["stats"], dict):
                raise ValueError("invalid personalization profile stats")
            return cls({key: InterventionStats.from_dict(value) for key, value in payload["stats"].items()})
        if payload.get("schema_version") == 2 and set(payload) == {
            "schema_version", "stats", "state_observations"
        }:
            profile = cls(
                {key: InterventionStats.from_dict(value) for key, value in payload["stats"].items()},
                [StateObservation.from_dict(value) for value in payload["state_observations"]],
            )
            profile.prune_state_observations()
            return profile
        if set(payload) != {"schema_version", "stats", "state_observations", "retained_structures"}:
            raise ValueError("invalid personalization profile")
        if payload["schema_version"] != cls.schema_version or not isinstance(payload["stats"], dict) or not isinstance(payload["state_observations"], list) or not isinstance(payload["retained_structures"], list):
            raise ValueError("unsupported personalization profile")
        profile = cls(
            {key: InterventionStats.from_dict(value) for key, value in payload["stats"].items()},
            [StateObservation.from_dict(value) for value in payload["state_observations"]],
            [RetainedStructure.from_dict(value) for value in payload["retained_structures"]],
        )
        profile.prune_state_observations()
        profile.prune_retained_structures()
        return profile

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class AdaptiveController:
    """Applies bounded feedback adaptation on top of deterministic safety control.

    Adaptation can only suppress or fade an intervention. It cannot bypass a
    confidence threshold, cooldown, cue-length limit, or productive-flow rule.
    """

    def __init__(self, base: InterventionController, profile: PersonalizationProfile) -> None:
        self.base = base
        self.profile = profile

    def decide(self, analysis: Analysis) -> Decision:
        self.profile.observe_state(analysis)
        decision = self.base.decide(analysis)
        if decision.action is Action.SILENCE:
            return decision
        stats = self.profile.for_intervention(decision.action, decision.cue)
        stats.opportunities += 1
        reviewed = stats.helpful + stats.neutral + stats.disruptive
        if reviewed >= 2 and stats.disruptive / reviewed >= 0.5:
            self.base.cancel_delivery(decision)
            return Decision(Action.SILENCE, None, "explicit feedback suppressed disruptive intervention", analysis)

        # After repeated helpful feedback, fade the cue deterministically. This
        # preserves occasional availability without increasing dependence.
        if stats.helpful >= 3:
            interval = min(4, 1 + stats.helpful // 3)
            if stats.opportunities % interval != 0:
                self.base.cancel_delivery(decision)
                return Decision(Action.SILENCE, None, "successful intervention is fading", analysis)

        stats.delivered += 1
        return decision

    def cancel_delivery(self, decision: Decision) -> None:
        """Prevent cancelled TTS from being learned as a delivered cue."""
        if decision.action is not Action.SILENCE:
            stats = self.profile.for_intervention(decision.action, decision.cue)
            stats.delivered = max(0, stats.delivered - 1)
        self.base.cancel_delivery(decision)
