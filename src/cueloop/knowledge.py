from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .core import Action, Analysis, Analyzer, SessionContext


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    source: str
    source_sha256: str
    text: str
    terms: frozenset[str]


@dataclass(frozen=True)
class GroundingMatch:
    source_ref: str
    source: str
    source_sha256: str
    score: float
    passage: str


class KnowledgeIndex:
    def __init__(self, chunks: Iterable[KnowledgeChunk]) -> None:
        self.chunks = tuple(chunks)

    @classmethod
    def from_paths(cls, paths: Iterable[Path], chunk_characters: int = 800) -> "KnowledgeIndex":
        if chunk_characters < 100:
            raise ValueError("chunk_characters must be at least 100")
        chunks: list[KnowledgeChunk] = []
        for path in paths:
            source_bytes = path.read_bytes()
            content = source_bytes.decode("utf-8")
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
            for paragraph in paragraphs:
                for offset in range(0, len(paragraph), chunk_characters):
                    text = paragraph[offset : offset + chunk_characters]
                    digest = hashlib.sha256(f"{path}:{offset}:{text}".encode()).hexdigest()[:12]
                    chunks.append(
                        KnowledgeChunk(
                            digest, str(path), source_sha256, text, frozenset(_terms(text))
                        )
                    )
        if not chunks:
            raise ValueError("knowledge sources contain no text")
        return cls(chunks)

    def search(self, query: str, limit: int = 3, minimum_score: float = 0.08) -> list[GroundingMatch]:
        query_terms = _terms(query)
        if not query_terms:
            return []
        ranked: list[GroundingMatch] = []
        for chunk in self.chunks:
            overlap = len(query_terms & chunk.terms)
            score = overlap / max(1, len(query_terms))
            if score >= minimum_score:
                ranked.append(
                    GroundingMatch(
                        chunk.id, chunk.source, chunk.source_sha256, score, chunk.text
                    )
                )
        ranked.sort(key=lambda match: (-match.score, match.source_ref))
        return ranked[:limit]

    @property
    def source_refs(self) -> frozenset[str]:
        return frozenset(chunk.id for chunk in self.chunks)


class GroundedRetrievalAnalyzer:
    """Replaces proposed retrieval text and refs with a local indexed passage."""

    def __init__(self, base: Analyzer, index: KnowledgeIndex, max_characters: int = 16) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self.base = base
        self.index = index
        self.max_characters = max_characters

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        analysis = self.base.analyze(text, context)
        if analysis.proposed_action is not Action.RETRIEVE:
            return analysis
        matches = self.index.search(text, limit=1)
        if not matches:
            return replace(analysis, source_refs=())
        match = matches[0]
        normalized = " ".join(match.passage.split())
        cue = normalized[: self.max_characters].rstrip()
        if not cue:
            return replace(analysis, source_refs=())
        return replace(analysis, cue=cue, source_refs=(match.source_ref,))


def ground_structure(structure: dict[str, object], index: KnowledgeIndex) -> dict[str, object]:
    grounded = dict(structure)
    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for collection in ("claims", "verification_queue"):
        raw_items = structure.get(collection, [])
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError(f"structure field {collection} must be an array")
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"structure field {collection} contains an invalid item")
            key = (str(item.get("id")), item["text"])
            if key in seen:
                continue
            seen.add(key)
            matches = index.search(item["text"])
            results.append(
                {
                    "item_id": item.get("id"),
                    "text": item["text"],
                    "epistemic_type": item.get("epistemic_type", "CLAIM"),
                    "status": "SUPPORTED" if matches else "UNSUPPORTED",
                    "matches": [asdict(match) for match in matches],
                }
            )
    grounded["grounding"] = results
    grounded["grounding_notice"] = (
        "Lexical matches are candidate evidence for human review, not proof that a claim is true."
    )
    return grounded


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    japanese_runs = re.findall(r"[ぁ-んァ-ヶ一-龠]{2,}", lowered)
    bigrams = {run[index : index + 2] for run in japanese_runs for index in range(len(run) - 1)}
    return words | bigrams


def load_structure(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("structure must be a JSON object")
    return payload
