from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Iterable, Protocol


class State(StrEnum):
    FLOWING = "FLOWING"
    PAUSED = "PAUSED"
    STUCK = "STUCK"


class Action(StrEnum):
    SILENCE = "SILENCE"
    CONTINUE = "CONTINUE"
    CUE = "CUE"


CUES: tuple[str, ...] = (
    "何に依存？",
    "具体的には？",
    "逆なら？",
    "どこで破れる？",
    "結論は？",
)


@dataclass(frozen=True)
class Analysis:
    state: State
    confidence: float
    evidence: tuple[str, ...]
    proposed_action: Action
    cue: str | None = None


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
    minimum_interval_ms: int = 1800
    max_cue_characters: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.cooldown_turns < 0:
            raise ValueError("cooldown_turns must be non-negative")
        if self.minimum_interval_ms < 0:
            raise ValueError("minimum_interval_ms must be non-negative")
        if self.max_cue_characters < 1:
            raise ValueError("max_cue_characters must be positive")


class HeuristicAnalyzer:
    """Deterministic baseline used when no semantic analyzer is configured."""

    _filler_pattern = re.compile(r"(?:えーと|えっと|その|なんだろう|うーん)")
    _repeat_pattern = re.compile(r"(.{2,12})\1{1,}")
    _structure_terms = ("なぜ", "前提", "根拠", "依存", "条件", "反例", "具体")

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        del context
        normalized = " ".join(text.strip().split())
        if not normalized:
            return Analysis(
                state=State.PAUSED,
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

        if repeated or fillers >= 3:
            if repeated:
                evidence.append("repeated phrase pattern")
            if fillers >= 3:
                evidence.append(f"high filler count: {fillers}")
            return Analysis(
                state=State.STUCK,
                confidence=min(0.95, 0.68 + 0.05 * fillers),
                evidence=tuple(evidence),
                proposed_action=Action.CUE,
                cue=self._select_cue(normalized),
            )

        if unfinished and len(normalized) < 140:
            return Analysis(
                state=State.PAUSED,
                confidence=0.68,
                evidence=("utterance appears unfinished",),
                proposed_action=Action.CONTINUE,
            )

        if len(normalized) >= 80 or has_structure or sentence_like:
            evidence.append("reasoning is continuing with sufficient content")
            if has_structure:
                evidence.append("structural reasoning terms detected")
            return Analysis(
                state=State.FLOWING,
                confidence=0.78 if has_structure else 0.7,
                evidence=tuple(evidence),
                proposed_action=Action.SILENCE,
            )

        return Analysis(
            state=State.PAUSED,
            confidence=0.6,
            evidence=("short turn without clear completion",),
            proposed_action=Action.CONTINUE,
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
    """Final delivery authority.

    Analyzer output is advisory.  This controller enforces confidence, turn
    cooldown, wall-clock spacing when timestamps are available, and cue length.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._turns_since_intervention = self.config.cooldown_turns
        self._last_intervention_ms: int | None = None

    def decide(self, analysis: Analysis, *, now_ms: int | None = None) -> Decision:
        if now_ms is not None and now_ms < 0:
            raise ValueError("now_ms must be non-negative")

        self._turns_since_intervention += 1

        if analysis.state is State.FLOWING:
            return Decision(Action.SILENCE, None, "productive flow must not be interrupted", analysis)

        if analysis.confidence < self.config.minimum_confidence:
            return Decision(Action.SILENCE, None, "confidence below delivery threshold", analysis)

        if self._turns_since_intervention < self.config.cooldown_turns:
            return Decision(Action.SILENCE, None, "intervention turn cooldown active", analysis)

        if (
            now_ms is not None
            and self._last_intervention_ms is not None
            and now_ms - self._last_intervention_ms < self.config.minimum_interval_ms
        ):
            return Decision(Action.SILENCE, None, "intervention time cooldown active", analysis)

        if analysis.proposed_action is Action.CUE:
            cue = analysis.cue
            if not cue or len(cue) > self.config.max_cue_characters:
                return Decision(Action.SILENCE, None, "cue is missing or too long", analysis)
            self._record_delivery(now_ms)
            return Decision(Action.CUE, cue, "stuck state with valid minimal cue", analysis)

        if analysis.proposed_action is Action.CONTINUE:
            self._record_delivery(now_ms)
            return Decision(Action.CONTINUE, None, "unfinished thought may benefit from turn preservation", analysis)

        return Decision(Action.SILENCE, None, "no deliverable intervention", analysis)

    def _record_delivery(self, now_ms: int | None) -> None:
        self._turns_since_intervention = 0
        if now_ms is not None:
            self._last_intervention_ms = now_ms


def safe_analysis(reason: str) -> Analysis:
    """Fail closed: uncertain analyzer failures must not create an intervention."""
    return Analysis(
        state=State.PAUSED,
        confidence=0.0,
        evidence=(reason,),
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
