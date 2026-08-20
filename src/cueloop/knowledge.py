from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .core import Action, Analysis, Analyzer, Evidence, EvidenceKind, SessionContext, State


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
    """Replaces proposed retrieval text and refs with a local indexed passage.

    Also converts explicit knowledge-seeking questions into a grounded RETRIEVE
    intervention. Without this conversion, a loaded source has no effect on the
    decision path: the heuristic analyzer never emits Action.RETRIEVE, so a user
    asking a direct question that the source already answers ("how is X
    guaranteed?") would receive silence even though grounded knowledge is
    available. This analyzer closes that gap while staying minimal: it surfaces
    only a tiny, non-answer pointer to a source fragment (never the resolution),
    so the user remains the primary reasoner.
    """

    # Surface triggers for an explicit knowledge-seeking question. A question
    # ending in ？/? is treated as knowledge-seeking only when it also carries a
    # question-seeking surface, to avoid misfiring on rhetorical or strategic
    # questions that the heuristic layer already classifies as STRATEGY_FAILURE.
    _QUESTION_TOKENS = (
        "なに", "何", "なにで", "なんで", "どうして", "どうやって", "なぜ",
        "どこ", "いつ", "だれ", "誰", "どれ", "どの", "なん",
        "って", "て何", "ものか", "のか", "かな", "かしら", "ですか", "ますか",
    )
    _NON_RETRIEVAL_STATES = {State.OVERLOAD, State.STRATEGY_FAILURE, State.FIXATION}

    def __init__(
        self,
        base: Analyzer,
        index: KnowledgeIndex,
        max_characters: int = 16,
        minimum_match_score: float = 0.2,
        minimum_deliverable_confidence: float = 0.6,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        if not 0.0 <= minimum_match_score <= 1.0:
            raise ValueError("minimum_match_score must be between 0 and 1")
        if not 0.0 <= minimum_deliverable_confidence <= 1.0:
            raise ValueError("minimum_deliverable_confidence must be between 0 and 1")
        self.base = base
        self.index = index
        self.max_characters = max_characters
        self.minimum_match_score = minimum_match_score
        self.minimum_deliverable_confidence = minimum_deliverable_confidence

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        base_analysis = self.base.analyze(text, context)

        # If the base layer already proposed RETRIEVE, ground its cue to the
        # local index so we never surface model-authored content.
        if base_analysis.proposed_action is Action.RETRIEVE:
            return self._ground_existing_retrieval(text, base_analysis)

        # If the user asked an explicit knowledge-seeking question, upgrade it
        # to a grounded RETRIEVE. This is the high-value path: an otherwise
        # silent PRODUCTIVE turn whose source already addresses the question.
        if self._is_knowledge_seeking(text, base_analysis):
            grounded = self._grounded_retrieval(text)
            if grounded is not None:
                cue, source_ref, score = grounded
                return Analysis(
                    state=State.RETRIEVAL_FAILURE,
                    confidence=self._confidence_for_question(score),
                    evidence=(
                        Evidence(
                            EvidenceKind.TRANSCRIPT,
                            "knowledge-seeking question matched a trusted source",
                            source_span=cue,
                        ),
                        Evidence(
                            EvidenceKind.INTERACTION_HISTORY,
                            "retrieval is a minimal source fragment, not the resolution",
                        ),
                    ),
                    proposed_action=Action.RETRIEVE,
                    cue=cue,
                    source_refs=(source_ref,),
                )

        return base_analysis

    def _ground_existing_retrieval(self, text: str, analysis: Analysis) -> Analysis:
        matches = self.index.search(text, limit=1)
        if not matches:
            return replace(analysis, source_refs=())
        return self._with_grounded_cue(analysis, matches[0], text)

    def _grounded_retrieval(self, text: str) -> tuple[str, str, float] | None:
        matches = self.index.search(text, limit=1)
        if not matches:
            return None
        match = matches[0]
        if match.score < self.minimum_match_score:
            return None
        cue = _grounded_cue(text, match.passage, self.max_characters)
        if not cue:
            return None
        return cue, match.source_ref, match.score

    def _with_grounded_cue(
        self, analysis: Analysis, match: "GroundingMatch", query: str
    ) -> Analysis:
        return replace(
            analysis,
            confidence=self._confidence_for(match.score),
            proposed_action=Action.RETRIEVE,
            cue=_grounded_cue(query, match.passage, self.max_characters),
            source_refs=(match.source_ref,),
        )

    def _is_knowledge_seeking(self, text: str, base_analysis: Analysis) -> bool:
        if base_analysis.proposed_action is Action.RETRIEVE:
            return False
        if base_analysis.state in self._NON_RETRIEVAL_STATES:
            return False
        return _looks_knowledge_seeking(text)

    def _confidence_for(self, match_score: float) -> float:
        return max(
            self.minimum_deliverable_confidence,
            min(0.88, 0.5 + 0.5 * match_score),
        )

    def _confidence_for_question(self, match_score: float) -> float:
        """Confidence for upgrading an explicit knowledge-seeking question.

        The question grammar itself is strong, independent evidence that
        retrieval is warranted; we lift the floor (above the generic
        threshold) whenever a real, if lexically weak, source match exists, so
        that synonym-bridged questions ("整合性" vs "一貫性") still surface a
        source pointer. Off-topic questions with no match are still suppressed
        upstream by ``minimum_match_score``.
        """
        return max(
            0.72,
            min(0.88, 0.5 + 0.5 * match_score),
        )

    def cancel(self) -> None:
        cancel = getattr(self.base, "cancel", None)
        if callable(cancel):
            cancel()


def _grounded_cue(query: str, passage: str, max_characters: int) -> str:
    """A bounded, non-answer fragment drawn from the *matched region* of a source.

    The whole chunk (up to 800 chars) may match while its opening is unrelated
    to the query, so surfacing the chunk head would point the user at the wrong
    topic. Instead we locate the cluster of query terms with the densest
    coverage in the passage and take a fragment starting there.
    """
    terms = _terms(query)
    positions = sorted(
        {
            start
            for term in terms
            for start in _term_offsets(passage, term)
        }
    )
    if not positions:
        normalized = " ".join(passage.strip().split())
        return normalized[:max_characters].rstrip()
    window = max_characters
    best_start, best_count = positions[0], -1
    for start in positions:
        captured = sum(
            passage.find(term, start, start + window) >= 0 for term in terms
        )
        if captured > best_count or (captured == best_count and start < best_start):
            best_start, best_count = start, captured
    fragment = passage[best_start : best_start + max_characters]
    return " ".join(fragment.strip().split()).rstrip()


def _term_offsets(passage: str, term: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        found = passage.find(term, start)
        if found < 0:
            return offsets
        offsets.append(found)
        start = found + 1


def _looks_knowledge_seeking(text: str) -> bool:
    """A direct question seeking information the source may provide.

    Detected by the presence of a question-seeking surface (question words,
    って/て何/ものか/のか/かな/かしら, or ですか/ますか) or a terminal bare "は？"
    ("Xは？"). Any such surface inside a turn is treated as knowledge-seeking,
    regardless of terminal punctuation: a turn phrased as "Xってどうやって...だろう。"
    is a request for grounding, not a productive declaration.
    """
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(
        token in lowered for token in GroundedRetrievalAnalyzer._QUESTION_TOKENS
    ):
        return True
    return len(normalized) >= 5 and normalized.endswith(("は？", "は?"))


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
