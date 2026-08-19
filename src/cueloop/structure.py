from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class StructuredItem:
    id: str
    text: str
    turn_index: int
    epistemic_type: str


@dataclass(frozen=True)
class Dependency:
    source_id: str
    target_id: str
    relation: str


@dataclass(frozen=True)
class SessionStructure:
    claims: tuple[StructuredItem, ...]
    premises: tuple[StructuredItem, ...]
    evidence: tuple[StructuredItem, ...]
    dependencies: tuple[Dependency, ...]
    counterexamples: tuple[StructuredItem, ...]
    unresolved_questions: tuple[StructuredItem, ...]
    verification_queue: tuple[StructuredItem, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*|\n+")
_PREMISE_MARKERS = ("前提", "なぜなら", "という条件", "仮定", "assuming", "because")
_EVIDENCE_MARKERS = ("根拠", "データ", "観測", "実験", "によると", "evidence", "measured")
_COUNTER_MARKERS = ("反例", "しかし", "一方", "逆なら", "例外", "counterexample", "however")
_QUESTION_MARKERS = ("？", "?", "未解決", "わからない", "要確認", "unknown")
_VERIFY_MARKERS = ("必ず", "常に", "全て", "すべて", "研究", "統計", "報告", "according to")
_CAUSAL_MARKERS = ("ので", "から", "ため", "ゆえ", "therefore", "because")


def extract_structure(turns: Iterable[str]) -> SessionStructure:
    claims: list[StructuredItem] = []
    premises: list[StructuredItem] = []
    evidence: list[StructuredItem] = []
    counterexamples: list[StructuredItem] = []
    questions: list[StructuredItem] = []
    verification: list[StructuredItem] = []
    dependencies: list[Dependency] = []
    counters = {"C": 0, "P": 0, "E": 0, "X": 0, "Q": 0, "V": 0}

    for turn_index, turn in enumerate(turns, start=1):
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(turn) if part.strip()]
        for sentence in sentences:
            categories: list[tuple[str, list[StructuredItem]]] = []
            if any(marker in sentence for marker in _QUESTION_MARKERS):
                categories.append(("Q", questions))
            if any(marker in sentence.lower() for marker in _PREMISE_MARKERS):
                categories.append(("P", premises))
            if any(marker in sentence.lower() for marker in _EVIDENCE_MARKERS):
                categories.append(("E", evidence))
            if any(marker in sentence.lower() for marker in _COUNTER_MARKERS):
                categories.append(("X", counterexamples))
            if _needs_verification(sentence):
                categories.append(("V", verification))
            if not categories or not any(prefix in {"Q", "X"} for prefix, _ in categories):
                categories.append(("C", claims))

            sentence_items: list[StructuredItem] = []
            for prefix, destination in categories:
                counters[prefix] += 1
                item = StructuredItem(
                    f"{prefix}{counters[prefix]}",
                    sentence,
                    turn_index,
                    _epistemic_type(prefix, sentence),
                )
                destination.append(item)
                sentence_items.append(item)

            claim = next((item for item in sentence_items if item.id.startswith("C")), None)
            support = next(
                (item for item in sentence_items if item.id.startswith(("P", "E"))), None
            )
            if claim and support:
                dependencies.append(Dependency(support.id, claim.id, "supports"))
            elif claim and any(marker in sentence.lower() for marker in _CAUSAL_MARKERS) and len(claims) >= 2:
                dependencies.append(Dependency(claims[-2].id, claim.id, "precedes_causally"))

    return SessionStructure(
        tuple(claims),
        tuple(premises),
        tuple(evidence),
        tuple(dependencies),
        tuple(counterexamples),
        tuple(questions),
        tuple(verification),
    )


def _needs_verification(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(re.search(r"\d", sentence)) or any(marker in lowered for marker in _VERIFY_MARKERS)


def _epistemic_type(prefix: str, sentence: str) -> str:
    lowered = sentence.lower()
    if prefix == "E":
        return "OBSERVATION"
    if prefix == "P":
        return "PREMISE"
    if prefix == "Q":
        return "QUESTION"
    if prefix == "X":
        return "COUNTEREXAMPLE"
    if prefix == "V":
        return "UNVERIFIED_FACT"
    if any(marker in lowered for marker in ("仮説", "可能性", "かもしれ", "hypothesis", "might")):
        return "HYPOTHESIS"
    if any(marker in lowered for marker in ("と思う", "推論", "したがって", "therefore", "infer")):
        return "INFERENCE"
    return "CLAIM"


def load_turns(path: Path) -> list[str]:
    turns: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            turns.append(stripped)
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} must be text or a JSON object")
        text = payload.get("text")
        if payload.get("kind") == "voice_turn":
            turn = payload.get("turn")
            text = turn.get("text") if isinstance(turn, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"line {line_number} does not contain transcript text")
        turns.append(text.strip())
    if not turns:
        raise ValueError("transcript is empty")
    return turns


def write_structure(path: Path, structure: SessionStructure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(structure.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
