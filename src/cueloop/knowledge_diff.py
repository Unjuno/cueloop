from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Iterable

from .core import Action, Analysis, Analyzer, Evidence, EvidenceKind, SessionContext, State
from .knowledge import KnowledgeIndex, _terms
from .structure import SessionStructure, StructuredItem, extract_structure


class StructuralDiffKind(StrEnum):
    """A structurally observable gap between the user's reasoning and a source."""

    MISSING_PREMISE = "MISSING_PREMISE"
    MISSING_VARIABLE = "MISSING_VARIABLE"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    CONTRADICTION = "CONTRADICTION"
    UNSUPPORTED_INFERENCE = "UNSUPPORTED_INFERENCE"


@dataclass(frozen=True)
class StructuralDiff:
    """A candidate structural difference.

    A diff is always a review candidate, never a proof. It describes a gap
    between two structure representations so a human or the controller can
    decide whether a single minimal cue is warranted.
    """

    kind: StructuralDiffKind
    source_item: StructuredItem
    cue: str
    confidence: float
    evidence: str
    related_user_item: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "source_item": {
                "id": self.source_item.id,
                "text": self.source_item.text,
                "epistemic_type": self.source_item.epistemic_type,
            },
            "cue": self.cue,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "related_user_item": self.related_user_item,
        }


_CUE_BY_KIND: dict[StructuralDiffKind, str] = {
    StructuralDiffKind.MISSING_PREMISE: "前提は？",
    StructuralDiffKind.MISSING_VARIABLE: "変数は？",
    StructuralDiffKind.MISSING_DEPENDENCY: "依存は？",
    StructuralDiffKind.CONTRADICTION: "逆は？",
    StructuralDiffKind.UNSUPPORTED_INFERENCE: "根拠は？",
}

_NEGATION_MARKERS = (
    "逆",
    "しない",
    "ではない",
    "違う",
    "未必",
    "しかし",
    "ただし",
    "例外",
    "unless",
    "however",
    "not",
)


def build_source_structure(index: KnowledgeIndex) -> SessionStructure:
    """Represent a trusted source with the same structure model as a session.

    Chunks are fed through the same extractor used for user turns so that the
    two representations stay directly comparable. Chunk boundaries are only a
    reading unit; they do not imply that a chunk's structure is a fact.
    """
    return extract_structure(chunk.text for chunk in index.chunks)


def structure_from_turns(turns: Iterable[str]) -> SessionStructure:
    return extract_structure(turns)


def compare_structures(
    user_structure: SessionStructure,
    source_structure: SessionStructure,
    *,
    minimum_confidence: float = 0.3,
    max_cue_characters: int = 16,
) -> tuple[StructuralDiff, ...]:
    """Compare user reasoning structure with a source and emit candidate diffs.

    The comparison is lexical-structural, deliberately not a claim about the
    truth of the source or the user. A diff names a *possible* gap that the
    human should resolve; it never supplies the resolution.

    Priority order: contradiction, missing dependency, missing premise,
    missing variable, unsupported inference. Only one diff is returned per
    kind, selected by confidence.
    """
    if minimum_confidence < 0.0 or minimum_confidence > 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if max_cue_characters < 1:
        raise ValueError("max_cue_characters must be positive")

    user_claims = _structural_texts(user_structure.claims)
    user_all = _structural_texts(
        tuple(
            item
            for collection in (
                user_structure.claims,
                user_structure.premises,
                user_structure.evidence,
                user_structure.counterexamples,
            )
            for item in collection
        )
    )
    if not user_claims:
        return ()
    if not _same_topic(user_all, source_structure):
        return ()

    diffs: list[StructuralDiff] = []

    for item in source_structure.claims + source_structure.premises:
        source_terms = _terms(item.text)
        if not source_terms:
            continue
        best_overlap = 0.0
        best_user: str | None = None
        for candidate in user_all:
            overlap = _jaccard(source_terms, _terms(candidate))
            if overlap > best_overlap:
                best_overlap = overlap
                best_user = candidate
        if best_user is None:
            continue
        if _matches(best_overlap, 0.25):
            if _looks_contradictory(user_structure, item.text, best_user):
                diffs.append(
                    _make_diff(
                        StructuralDiffKind.CONTRADICTION,
                        item,
                        best_overlap,
                        f"user statement contradicts source statement comparable to {item.text[:40]}",
                        best_user,
                    )
                )
            else:
                user_support = _user_supports(
                    user_structure, best_user, source_supporters(source_structure, item)
                )
                if not user_support:
                    diffs.append(
                        _make_diff(
                            StructuralDiffKind.MISSING_DEPENDENCY,
                            item,
                            best_overlap,
                            f"user statement resembles source statement but lacks its support: {item.text[:40]}",
                            best_user,
                        )
                    )
        else:
            diffs.append(
                _make_diff(
                    StructuralDiffKind.MISSING_PREMISE,
                    item,
                    best_overlap,
                    f"source premise absent from user reasoning: {item.text[:40]}",
                    best_user,
                )
            )

    variables = _missing_variables(source_structure, user_all)
    for variable, confidence in variables:
        diffs.append(
            StructuralDiff(
                StructuralDiffKind.MISSING_VARIABLE,
                _source_anchor(source_structure),
                _CUE_BY_KIND[StructuralDiffKind.MISSING_VARIABLE],
                confidence,
                f"frequent source term absent from user reasoning: {variable}",
            )
        )

    for claim in user_structure.claims:
        source_terms_all = _all_source_terms(source_structure)
        if not _matches(_jaccard(source_terms_all, _terms(claim.text)), 0.2) and (
            claim.epistemic_type in {"INFERENCE", "HYPOTHESIS"}
            or _is_inference_language(claim.text)
        ):
            diffs.append(
                StructuralDiff(
                    StructuralDiffKind.UNSUPPORTED_INFERENCE,
                    _source_anchor(source_structure),
                    _CUE_BY_KIND[StructuralDiffKind.UNSUPPORTED_INFERENCE],
                    0.55,
                    f"user inference has no lexical support in the source: {claim.text[:40]}",
                    claim.text,
                )
            )

    ordered = sorted(diffs, key=lambda diff: (diff.kind != StructuralDiffKind.CONTRADICTION, -diff.confidence))
    seen: set[StructuralDiffKind] = set()
    selected: list[StructuralDiff] = []
    for diff in ordered:
        if diff.kind in seen:
            continue
        if len(diff.cue) > max_cue_characters:
            continue
        if diff.confidence < minimum_confidence:
            continue
        seen.add(diff.kind)
        selected.append(diff)
    return tuple(selected)


class StructuralDiffAnalyzer:
    """Proposes a minimal cue from a structural diff, authority stays upstream.

    Wraps a base analyzer. It never interrupts a PRODUCTIVE user and only
    ever *proposes*; the deterministic controller remains the delivery
    authority. Relative to the base analyzer this adds the source-vs-user
    structure gap as an additional evidence signal on non-productive turns.
    """

    def __init__(
        self,
        base: Analyzer,
        source_structure: SessionStructure,
        minimum_confidence: float = 0.5,
    ) -> None:
        if minimum_confidence < 0.0 or minimum_confidence > 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.base = base
        self.source_structure = source_structure
        self.minimum_confidence = minimum_confidence

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        base_analysis = self.base.analyze(text, context)
        if base_analysis.state in {State.PRODUCTIVE, State.OVERLOAD}:
            return base_analysis
        turns = list(context.snapshot()) if context is not None else []
        if text.strip():
            turns.append(text)
        user_structure = structure_from_turns(turns)
        diffs = compare_structures(
            user_structure,
            self.source_structure,
            minimum_confidence=self.minimum_confidence,
        )
        if not diffs:
            return base_analysis
        diff = diffs[0]
        diff_action = _diff_action(diff.kind)
        if base_analysis.proposed_action in {
            Action.PERTURB,
            Action.MIRROR,
            Action.CONSTRAIN,
        } and diff_action is not Action.PERTURB:
            return base_analysis
        return Analysis(
            state=State.STRUCTURE_GAP,
            confidence=min(0.9, base_analysis.confidence, diff.confidence + 0.2),
            evidence=(
                Evidence(EvidenceKind.TRANSCRIPT, diff.evidence),
                Evidence(EvidenceKind.TRANSCRIPT, "structural difference proposed, not proven"),
            ),
            proposed_action=diff_action,
            cue=diff.cue,
        )


def _make_diff(
    kind: StructuralDiffKind,
    item: StructuredItem,
    overlap: float,
    evidence: str,
    related_user: str,
    *,
    floor: float = 0.3,
) -> StructuralDiff:
    confidence = min(0.9, max(floor, overlap + 0.15))
    return StructuralDiff(kind, item, _CUE_BY_KIND[kind], confidence, evidence, related_user)


def _diff_action(kind: StructuralDiffKind) -> Action:
    if kind is StructuralDiffKind.CONTRADICTION:
        return Action.PERTURB
    return Action.QUESTION


def _structural_texts(items: Iterable[StructuredItem]) -> list[str]:
    return [item.text for item in items]


def _jaccard(terms_a: set[str], terms_b: set[str]) -> float:
    if not terms_a or not terms_b:
        return 0.0
    return len(terms_a & terms_b) / len(terms_a | terms_b)


def _same_topic(user_all: list[str], source: SessionStructure) -> bool:
    """Gate: only reason about a source when the user is on a similar topic.

    Without this gate a user reasoning about anything would create pretend
    structure gaps against every loaded source, which would turn the system
    into a noisy reminder instead of a knowledge channel.

    The gate uses absolute shared-term recall rather than Jaccard because
    Japanese bigram tokens make the denominator large and dilute real topic
    overlap. A low absolute overlap is enough to confirm the same material is
    in play; the structure comparison then decides whether a diff is warranted.
    """
    user_terms: set[str] = set()
    for text in user_all:
        user_terms.update(_terms(text))
    source_terms = _all_source_terms(source)
    if not user_terms or not source_terms:
        return False
    shared = user_terms & source_terms
    if len(shared) >= 2:
        return True
    if len(user_terms) <= 4:
        return False
    long_source = {term for term in source_terms if len(term) >= 3}
    return bool(shared & long_source) and len(shared) >= 1


def _matches(overlap: float, threshold: float) -> bool:
    return overlap >= threshold


def _resolve_supporters(source: SessionStructure, item: StructuredItem) -> tuple[StructuredItem, ...]:
    by_id: dict[str, StructuredItem] = {
        evidence.id: evidence
        for collection in (source.premises, source.evidence)
        for evidence in collection
    }
    resolved = []
    for dependency in source.dependencies:
        if dependency.target_id != item.id:
            continue
        supporter = by_id.get(dependency.source_id)
        if supporter is not None:
            resolved.append(supporter)
    return tuple(resolved)


def source_supporters(source: SessionStructure, item: StructuredItem) -> tuple[StructuredItem, ...]:
    """Directly stated supporters (premises and evidence) of a source item."""
    return _resolve_supporters(source, item)


def _user_supports(
    user: SessionStructure, user_claim: str, source_supporters: tuple[StructuredItem, ...]
) -> bool:
    """Whether the user already stated a premise/evidence resembling the source support."""
    user_support_texts = [item.text for item in user.premises] + [item.text for item in user.evidence]
    for supporter in source_supporters:
        supporter_terms = _terms(supporter.text)
        if not supporter_terms:
            continue
        if any(_matches(_jaccard(supporter_terms, _terms(candidate)), 0.4) for candidate in user_support_texts):
            return True
    return False


def _looks_contradictory(user: SessionStructure, source_text: str, user_text: str) -> bool:
    lowered_user = user_text.lower()
    lowered_source = source_text.lower()
    if any(marker in lowered_user for marker in _NEGATION_MARKERS) and any(
        marker in lowered_source for marker in _NEGATION_MARKERS
    ):
        return False
    return any(marker in lowered_user for marker in _NEGATION_MARKERS) or any(
        marker in lowered_source for marker in _NEGATION_MARKERS
    )


_INFERENCE_LANGUAGE = (
    "と考える",
    "考える",
    "思う",
    "推測",
    "推定",
    "べきだ",
    "べき",
    "だろう",
    "はず",
    "infer",
    "assume",
    "suppose",
    "should",
)


def _is_inference_language(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _INFERENCE_LANGUAGE)


def _missing_variables(source: SessionStructure, user_all: list[str]) -> list[tuple[str, float]]:
    term_counts: dict[str, int] = {}
    for item in source.claims + source.premises:
        for term in _variable_terms(item.text):
            term_counts[term] = term_counts.get(term, 0) + 1
    user_terms: set[str] = set()
    for text in user_all:
        user_terms.update(_terms(text))
    missing = sorted(
        ((term, count) for term, count in term_counts.items() if term not in user_terms),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [(term, min(0.8, 0.4 + 0.05 * count)) for term, count in missing[:3]]


_LONG_RUN = re.compile(r"[a-zA-Z]{3,}|[ァ-ヶ]{2,}|[0-9]+%?")
_KATAKANA_RUN = re.compile(r"[ァ-ヶ]{2,}")


def _variable_terms(text: str) -> set[str]:
    """Terms likely to be a meaningful variable of the model under discussion.

    A variable is an observable parameter of the source material. Latin terms,
    katakana loanwords, and numbers qualify. Plain Japanese grammatical units
    do not: they are connective tissue, and flagging them would fire on every
    source. Latin words are split into lowercase tokens so structure-relevant
    terms (for example a named factor) surface independently.
    """
    variables: set[str] = set()
    for match in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}", text):
        variables.add(match.lower())
    variables.update(_KATAKANA_RUN.findall(text))
    variables.update(re.findall(r"[0-9]+", text))
    return variables


_ALL_TERMS_CACHE: dict[tuple[str, ...], set[str]] = {}


def _all_source_terms(source: SessionStructure) -> set[str]:
    key = tuple(
        item.text for collection in (source.claims, source.premises, source.evidence)
        for item in collection
    )
    if key not in _ALL_TERMS_CACHE:
        terms: set[str] = set()
        for text in key:
            terms.update(_terms(text))
        _ALL_TERMS_CACHE[key] = terms
    return _ALL_TERMS_CACHE[key]


_SOURCE_ANCHOR_CACHE: dict[str, StructuredItem] = {}


def _source_anchor(source: SessionStructure) -> StructuredItem:
    key = source.claims[0].id if source.claims else "empty"
    cached = _SOURCE_ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    if source.claims:
        anchor = source.claims[0]
    elif source.premises:
        anchor = source.premises[0]
    else:
        anchor = StructuredItem("S0", "(source has no comparable structure)", 0, "CLAIM")
    _SOURCE_ANCHOR_CACHE[key] = anchor
    return anchor