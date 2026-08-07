from __future__ import annotations

from dataclasses import asdict, dataclass
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


class Analyzer(Protocol):
    """Produces a non-binding state hypothesis and intervention proposal."""

    def analyze(self, text: str) -> Analysis: ...


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


class HeuristicAnalyzer:
    """Deterministic baseline used when no semantic analyzer is configured."""

    _filler_pattern = re.compile(r"(?:えーと|えっと|その|なんだろう|うーん)")
    _repeat_pattern = re.compile(r"(.{2,12})\1{1,}")
    _structure_terms = ("なぜ", "前提", "根拠", "依存", "条件", "反例", "具体")

    def analyze(self, text: str) -> Analysis:
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
    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self._turns_since_intervention = self.config.cooldown_turns

    def decide(self, analysis: Analysis) -> Decision:
        self._turns_since_intervention += 1

        if analysis.state is State.FLOWING:
            return Decision(Action.SILENCE, None, "productive flow must not be interrupted", analysis)

        if analysis.confidence < self.config.minimum_confidence:
            return Decision(Action.SILENCE, None, "confidence below delivery threshold", analysis)

        if self._turns_since_intervention < self.config.cooldown_turns:
            return Decision(Action.SILENCE, None, "intervention cooldown active", analysis)

        if analysis.proposed_action is Action.CUE:
            cue = analysis.cue
            if not cue or len(cue) > self.config.max_cue_characters:
                return Decision(Action.SILENCE, None, "cue is missing or too long", analysis)
            self._turns_since_intervention = 0
            return Decision(Action.CUE, cue, "stuck state with valid minimal cue", analysis)

        if analysis.proposed_action is Action.CONTINUE:
            self._turns_since_intervention = 0
            return Decision(Action.CONTINUE, None, "unfinished thought may benefit from turn preservation", analysis)

        return Decision(Action.SILENCE, None, "no deliverable intervention", analysis)


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
    decisions: list[Decision] = []
    for text in texts:
        decision = selected_controller.decide(selected_analyzer.analyze(text))
        decisions.append(decision)
        if log_path is not None:
            append_jsonl(log_path, text, decision)
    return decisions
