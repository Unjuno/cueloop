from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Iterable, Protocol


class State(StrEnum):
    PRODUCTIVE = "PRODUCTIVE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    STRATEGY_FAILURE = "STRATEGY_FAILURE"
    FIXATION = "FIXATION"
    STRUCTURE_GAP = "STRUCTURE_GAP"
    OVERLOAD = "OVERLOAD"


class Action(StrEnum):
    SILENCE = "SILENCE"
    BACKCHANNEL = "BACKCHANNEL"
    QUESTION = "QUESTION"
    MIRROR = "MIRROR"
    PERTURB = "PERTURB"
    CONSTRAIN = "CONSTRAIN"
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"


class EvidenceKind(StrEnum):
    TRANSCRIPT = "TRANSCRIPT"
    TIMING = "TIMING"
    PROSODY = "PROSODY"
    INTERACTION_HISTORY = "INTERACTION_HISTORY"
    USER_SIGNAL = "USER_SIGNAL"


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    description: str
    source_span: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("evidence description must not be empty")

    def __str__(self) -> str:
        return self.description


CUES: tuple[str, ...] = (
    "何に依存？",
    "具体的には？",
    "逆なら？",
    "どこで破れる？",
    "結論は？",
)

ALLOWED_STATE_ACTIONS: dict[State, frozenset[Action]] = {
    State.PRODUCTIVE: frozenset({Action.SILENCE}),
    State.RETRIEVAL_FAILURE: frozenset(
        {Action.SILENCE, Action.BACKCHANNEL, Action.QUESTION, Action.RETRIEVE, Action.VERIFY}
    ),
    State.STRATEGY_FAILURE: frozenset(
        {Action.SILENCE, Action.QUESTION, Action.CONSTRAIN, Action.VERIFY}
    ),
    State.FIXATION: frozenset({Action.SILENCE, Action.PERTURB, Action.MIRROR, Action.VERIFY}),
    State.STRUCTURE_GAP: frozenset({Action.SILENCE, Action.QUESTION, Action.MIRROR, Action.VERIFY}),
    State.OVERLOAD: frozenset({Action.SILENCE}),
}


@dataclass(frozen=True)
class Analysis:
    state: State
    confidence: float
    evidence: tuple[Evidence, ...]
    proposed_action: Action
    cue: str | None = None
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Keep the Python API ergonomic while ensuring all runtime records use
        # the same typed evidence shape as human-state.schema.json.
        normalized = tuple(
            item if isinstance(item, Evidence) else Evidence(EvidenceKind.TRANSCRIPT, str(item))
            for item in self.evidence
        )
        object.__setattr__(self, "evidence", normalized)


@dataclass
class SessionContext:
    """Bounded recent-turn context; never an unbounded transcript."""

    max_turns: int = 4
    max_characters: int = 2400
    _turns: list[str] = field(default_factory=list, repr=False)

    def add(self, text: str) -> None:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return
        self._turns.append(normalized)
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        while sum(len(turn) for turn in self._turns) > self.max_characters and len(self._turns) > 1:
            self._turns.pop(0)
        if self._turns and len(self._turns[-1]) > self.max_characters:
            self._turns[-1] = self._turns[-1][-self.max_characters :]

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._turns)

    def render(self) -> str:
        if not self._turns:
            return "(no previous turns)"
        return "\n".join(f"{index + 1}. {turn}" for index, turn in enumerate(self._turns))


class Analyzer(Protocol):
    """Produces a non-binding state hypothesis and intervention proposal."""

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis: ...


@dataclass(frozen=True)
class Decision:
    action: Action
    cue: str | None
    reason: str
    analysis: Analysis


@dataclass
class ControllerConfig:
    minimum_confidence: float = 0.65
    cooldown_turns: int = 2
    max_cue_characters: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if self.cooldown_turns < 0:
            raise ValueError("cooldown_turns must be non-negative")
        if self.max_cue_characters < 1:
            raise ValueError("max_cue_characters must be positive")

    @classmethod
    def from_json_file(cls, path: Path) -> "ControllerConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("policy config must be a JSON object")
        allowed = {"minimum_confidence", "cooldown_turns", "max_cue_characters"}
        if not set(payload) <= allowed:
            raise ValueError("policy config contains unknown keys")
        return cls(**payload)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_confidence": self.minimum_confidence,
            "cooldown_turns": self.cooldown_turns,
            "max_cue_characters": self.max_cue_characters,
        }


class HeuristicAnalyzer:
    """Deterministic baseline used when no semantic analyzer is configured."""

    _filler_pattern = re.compile(r"(?:えーと|えっと|その|なんだろう|うーん)")
    _repeat_pattern = re.compile(r"(.{2,12})\1{1,}")
    _structure_terms = ("なぜ", "前提", "根拠", "依存", "条件", "反例", "具体")
    _overload_terms = ("疲れた", "混乱", "情報が多すぎ", "わからなくなった", "頭がいっぱい")
    _strategy_terms = ("どう進め", "何から", "方法がわから", "どう考え", "手順がわから")

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        del context
        normalized = " ".join(text.strip().split())
        if not normalized:
            return Analysis(
                state=State.RETRIEVAL_FAILURE,
                confidence=0.55,
                evidence=("input is empty",),
                proposed_action=Action.SILENCE,
            )

        evidence: list[str] = []
        fillers = len(self._filler_pattern.findall(normalized))
        repeated = bool(self._repeat_pattern.search(normalized))
        unfinished = normalized.endswith(("けど", "ので", "から", "というか", "…", "..."))
        has_structure = any(term in normalized for term in self._structure_terms)
        sentence_like = normalized.endswith(("。", "？", "?", "！", "!"))

        if any(term in normalized for term in self._overload_terms):
            return Analysis(
                state=State.OVERLOAD,
                confidence=0.82,
                evidence=("explicit overload or fatigue language",),
                proposed_action=Action.SILENCE,
            )

        if any(term in normalized for term in self._strategy_terms):
            return Analysis(
                state=State.STRATEGY_FAILURE,
                confidence=0.76,
                evidence=("explicit uncertainty about the next reasoning operation",),
                proposed_action=Action.CONSTRAIN,
                cue="結論は？",
            )

        if repeated or fillers >= 3:
            if repeated:
                evidence.append("repeated phrase pattern")
            if fillers >= 3:
                evidence.append(f"high filler count: {fillers}")
            if any(token in normalized for token in ("効率", "性能", "良い", "悪い", "抽象", "構造", "モデル")):
                state = State.STRUCTURE_GAP
                action = Action.QUESTION
            else:
                state = State.FIXATION
                action = Action.PERTURB
            return Analysis(
                state=state,
                confidence=min(0.95, 0.68 + 0.05 * fillers),
                evidence=tuple(evidence),
                proposed_action=action,
                cue=self._select_cue(normalized),
            )

        if unfinished and len(normalized) < 140:
            return Analysis(
                state=State.RETRIEVAL_FAILURE,
                confidence=0.68,
                evidence=("utterance appears unfinished",),
                proposed_action=Action.BACKCHANNEL,
            )

        if len(normalized) >= 80 or has_structure or sentence_like:
            evidence.append("reasoning is continuing with sufficient content")
            if has_structure:
                evidence.append("structural reasoning terms detected")
            return Analysis(
                state=State.PRODUCTIVE,
                confidence=0.78 if has_structure else 0.7,
                evidence=tuple(evidence),
                proposed_action=Action.SILENCE,
            )

        return Analysis(
            state=State.RETRIEVAL_FAILURE,
            confidence=0.6,
            evidence=("short turn without clear completion",),
            proposed_action=Action.BACKCHANNEL,
        )

    @staticmethod
    def _select_cue(text: str) -> str:
        if any(token in text for token in ("絶対", "常に", "全部", "必ず")):
            return "どこで破れる？"
        if any(token in text for token in ("効率", "性能", "良い", "悪い")):
            return "何に依存？"
        if any(token in text for token in ("抽象", "構造", "モデル")):
            return "具体的には？"
        return "逆なら？"


class InterventionController:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        trusted_source_refs: frozenset[str] | None = None,
    ) -> None:
        self.config = config or ControllerConfig()
        self.trusted_source_refs = trusted_source_refs or frozenset()
        self._turns_since_intervention = self.config.cooldown_turns

    def decide(self, analysis: Analysis) -> Decision:
        self._turns_since_intervention += 1

        if analysis.state is State.PRODUCTIVE:
            return Decision(Action.SILENCE, None, "productive flow must not be interrupted", analysis)

        if analysis.state is State.OVERLOAD:
            return Decision(Action.SILENCE, None, "overload must not receive new input", analysis)

        if analysis.confidence < self.config.minimum_confidence:
            return Decision(Action.SILENCE, None, "confidence below delivery threshold", analysis)

        if analysis.proposed_action not in ALLOWED_STATE_ACTIONS[analysis.state]:
            return Decision(Action.SILENCE, None, "state-action combination is prohibited", analysis)

        if self._turns_since_intervention < self.config.cooldown_turns:
            return Decision(Action.SILENCE, None, "intervention cooldown active", analysis)

        if analysis.proposed_action is Action.RETRIEVE:
            if not analysis.source_refs or not set(analysis.source_refs) <= self.trusted_source_refs:
                return Decision(Action.SILENCE, None, "real-time retrieval requires grounded source context", analysis)
            cue = analysis.cue
            if not cue or len(cue) > self.config.max_cue_characters:
                return Decision(Action.SILENCE, None, "grounded retrieval is missing or too long", analysis)
            self._turns_since_intervention = 0
            return Decision(Action.RETRIEVE, cue, "grounded minimal retrieval has explicit source references", analysis)

        if analysis.proposed_action in {
            Action.QUESTION,
            Action.MIRROR,
            Action.PERTURB,
            Action.CONSTRAIN,
        }:
            cue = analysis.cue
            if not cue or len(cue) > self.config.max_cue_characters:
                return Decision(Action.SILENCE, None, "cue is missing or too long", analysis)
            self._turns_since_intervention = 0
            return Decision(analysis.proposed_action, cue, "state has one valid minimal operation", analysis)

        if analysis.proposed_action is Action.BACKCHANNEL:
            self._turns_since_intervention = 0
            return Decision(Action.BACKCHANNEL, None, "unfinished thought may benefit from turn preservation", analysis)

        if analysis.proposed_action is Action.VERIFY:
            self._turns_since_intervention = 0
            return Decision(Action.VERIFY, None, "claim marked for delayed verification", analysis)

        return Decision(Action.SILENCE, None, "no deliverable intervention", analysis)

    def cancel_delivery(self, decision: Decision) -> None:
        """Undo delivery-only cooldown when a live cue is cancelled."""
        if decision.action is not Action.SILENCE:
            self._turns_since_intervention = self.config.cooldown_turns


def safe_analysis(reason: str) -> Analysis:
    """Fail closed: uncertain analyzer failures must not create an intervention."""
    return Analysis(
        state=State.RETRIEVAL_FAILURE,
        confidence=0.0,
        evidence=(Evidence(EvidenceKind.INTERACTION_HISTORY, reason),),
        proposed_action=Action.SILENCE,
    )


def append_jsonl(path: Path, text: str, decision: Decision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "decision": {
            "action": decision.action.value,
            "cue": decision.cue,
            "reason": decision.reason,
            "analysis": {
                **asdict(decision.analysis),
                "state": decision.analysis.state.value,
                "proposed_action": decision.analysis.proposed_action.value,
            },
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_turns(
    texts: Iterable[str],
    log_path: Path | None = None,
    analyzer: Analyzer | None = None,
    controller: InterventionController | None = None,
) -> list[Decision]:
    selected_analyzer = analyzer or HeuristicAnalyzer()
    selected_controller = controller or InterventionController()
    context = SessionContext()
    decisions: list[Decision] = []
    for text in texts:
        decision = selected_controller.decide(selected_analyzer.analyze(text, context))
        decisions.append(decision)
        if log_path is not None:
            append_jsonl(log_path, text, decision)
        context.add(text)
    return decisions
