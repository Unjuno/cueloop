"""Unified analyzer for real-time intervention decisions.

Combines heuristic analysis, structural diff awareness, and knowledge state
tracking into a single analyzer usable for both mid-speech and turn-boundary
intervention decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core import (
    Action,
    Analysis,
    Analyzer,
    Evidence,
    EvidenceKind,
    HeuristicAnalyzer,
    SessionContext,
    State,
)
from .knowledge_diff import (
    compare_structures,
    structure_from_turns,
)
from .knowledge_state import KnowledgeStore
from .structure import SessionStructure


@dataclass
class UnifiedAnalyzerConfig:
    """Configuration for the unified analyzer."""

    # Heuristic analyzer config
    minimum_confidence: float = 0.6
    # Structural diff config
    source_structure: Optional[SessionStructure] = None
    structural_minimum_confidence: float = 0.3
    # Knowledge state config
    knowledge_store: Optional[KnowledgeStore] = None
    # Cue length limits
    max_cue_characters: int = 16
    max_mid_speech_cue_characters: int = 8


class MidSpeechUnifiedAnalyzer:
    """Lightweight unified analyzer for mid-speech injection.

    Uses heuristic analysis + structural diff (if source available) +
    vocabulary-based operation detection + knowledge state tracking.
    Optimized for low latency during ongoing speech.

    Cue selection is gap-driven, not echo-driven:

    - A structural diff against the source is the highest-value signal: it names
      a relation the user's own reasoning has not produced, so it should be
      delivered even while the user is reasoning fluently.
    - A vocabulary operation is only delivered when the user has *not* already
      performed that operation in the current utterance. Re-asking an operation
      the user is visibly performing (for example cueing "具体は？" right after
      "具体的には") is a stale echo and is suppressed. When the performed
      operation implies a complementary one, the complement may be delivered
      instead (for example after concluding, ask for support).
    """

    # Vocabulary-to-operation mapping (from original MidSpeechInjector)
    _VOCAB_OPERATIONS: tuple[tuple[str, str], ...] = (
        ("なぜなら", "mechanism"),
        ("仕組み", "mechanism"),
        ("理由", "mechanism"),
        ("なぜ", "mechanism"),
        ("どうして", "mechanism"),
        ("だから", "conclusion"),
        ("つまり", "conclusion"),
        ("なので", "conclusion"),
        ("結論", "conclusion"),
        ("たぶん", "ground"),
        ("だと思う", "ground"),
        ("ような気がする", "ground"),
        ("思う", "ground"),
        ("感じ", "ground"),
        ("例えば", "concretize"),
        ("具体的", "concretize"),
        ("まず", "concretize"),
        ("依存", "dependency"),
        ("前提", "dependency"),
        ("必要", "dependency"),
        ("条件", "boundary"),
        ("限界", "boundary"),
        ("範囲", "boundary"),
        ("いつも", "boundary"),
        ("必ず", "boundary"),
        ("逆に", "invert"),
        ("逆なら", "invert"),
        ("逆は", "invert"),
        ("逆", "invert"),
        ("比較", "comparison"),
        ("違い", "comparison"),
        ("比べる", "comparison"),
        ("違う", "comparison"),
    )

    # Operation-to-cue candidates (from original MidSpeechInjector)
    CUE_BY_OPERATION: dict[str, tuple[str, ...]] = {
        "dependency": ("何に？", "依存は？", "前提は？"),
        "concretize": ("具体は？", "例は？", "場合、？"),
        "invert": ("逆は？", "逆なら？", "逆で？"),
        "boundary": ("範囲は？", "条件は？", "どこまで？"),
        "conclusion": ("結論は？", "つまり？", "それで？"),
        "ground": ("根拠は？", "根拠", "答えは？"),
        "mechanism": ("仕組みは？", "なぜ？", "どうして？"),
        "comparison": ("何と比べる？", "違いは？", "基準は？"),
    }

    # Performative closures: when one of these connectives appears in the
    # current utterance, the operation has already been performed by the user's
    # own rhetoric, so cueing the same operation would be a stale echo. The
    # "ground" family is intentionally empty: uncertainty tokens (たぶん/思う/
    # 気がする) assert a conclusion without support, so asking for grounding
    # remains the value-added move after such an utterance.
    STALE_MARKERS: dict[str, tuple[str, ...]] = {
        "mechanism": ("なぜなら", "仕組みは", "理由は", "どうして"),
        "conclusion": ("つまり", "だから", "なので", "結論として", "結論は"),
        "concretize": ("例えば", "具体的には", "具体的な", "という例"),
        "invert": ("逆に", "逆なら", "逆は", "逆で"),
        "boundary": ("条件として", "条件は", "限界は", "範囲は", "どこまで"),
        "dependency": ("前提として", "前提は", "依存は"),
        "comparison": ("比較すると", "比較して", "比べる", "違いは"),
        "ground": (),
    }

    # A performed operation implies a complementary one that may still add
    # value. After concluding or concretizing, what grounds it? After naming a
    # dependency, where does it stop holding? After inverting, what is the
    # boundary of the flipped view?
    COMPLEMENT_OPS: dict[str, str] = {
        "conclusion": "ground",
        "concretize": "ground",
        "mechanism": "ground",
        "dependency": "boundary",
        "boundary": "dependency",
        "invert": "boundary",
        "comparison": "boundary",
        "ground": "boundary",
    }

    def __init__(
        self,
        config: UnifiedAnalyzerConfig,
        base_analyzer: Optional[Analyzer] = None,
    ) -> None:
        self.config = config
        self.base_analyzer = base_analyzer or HeuristicAnalyzer()
        self.source_structure = config.source_structure
        self.knowledge_store = config.knowledge_store
        self._candidate_index_by_operation: dict[str, int] = {}

    def analyze(self, text: str, context: Optional[SessionContext] = None) -> Analysis:
        # Get base heuristic analysis
        base_analysis = self.base_analyzer.analyze(text, context)

        # Mid-speech only suppresses OVERLOAD (not PRODUCTIVE - we want to nudge even productive speech)
        if base_analysis.state is State.OVERLOAD:
            return base_analysis

        # Check structural diff if source available
        if self.source_structure is not None and context is not None:
            turns = list(context.snapshot())
            if text.strip():
                turns.append(text)
            from .knowledge_diff import structure_from_turns, compare_structures, _diff_action
            user_structure = structure_from_turns(turns)
            diffs = compare_structures(
                user_structure,
                self.source_structure,
                minimum_confidence=self.config.structural_minimum_confidence,
                max_cue_characters=self.config.max_mid_speech_cue_characters,
            )
            if diffs:
                diff = diffs[0]
                diff_action = _diff_action(diff.kind)
                # Only override if base action is compatible
                if base_analysis.proposed_action in {
                    Action.QUESTION,
                    Action.MIRROR,
                    Action.PERTURB,
                    Action.CONSTRAIN,
                } and diff_action is not Action.PERTURB:
                    pass  # Keep base analysis
                else:
                    base_analysis = Analysis(
                        state=State.STRUCTURE_GAP,
                        confidence=min(0.9, base_analysis.confidence, diff.confidence + 0.2),
                        evidence=(
                            Evidence(EvidenceKind.TRANSCRIPT, diff.evidence),
                            Evidence(EvidenceKind.TRANSCRIPT, "structural difference proposed, not proven"),
                        ),
                        proposed_action=diff_action,
                        cue=diff.cue,
                    )

        # If no structural diff cue, fall back to vocabulary-based operation detection
        # (Mid-speech uses vocabulary matching independently of heuristic's proposed_action)
        if base_analysis.cue is None and base_analysis.state is not State.OVERLOAD:
            operation = self._select_operation(text)
            if operation is not None and base_analysis.confidence >= self.config.minimum_confidence:
                # Stale-echo suppression: if the user's own rhetoric already
                # performed this operation, do not repeat it. Move to the
                # complementary operation when one fits, otherwise stay silent.
                if self._is_performative_close(operation, text):
                    operation = self.COMPLEMENT_OPS.get(operation)
                cue = None
                if operation is not None and not self._is_performative_close(operation, text):
                    cue = self._pick_cue(operation, text)
                if cue:
                    base_analysis = Analysis(
                        state=base_analysis.state,
                        confidence=base_analysis.confidence,
                        evidence=base_analysis.evidence,
                        proposed_action=base_analysis.proposed_action,
                        cue=cue,
                    )

        # Apply knowledge state fading
        if base_analysis.cue and self.knowledge_store is not None:
            faded_cue = self._apply_fading(base_analysis.cue)
            if faded_cue is None:
                return Analysis(
                    state=base_analysis.state,
                    confidence=base_analysis.confidence,
                    evidence=base_analysis.evidence + (
                        Evidence(EvidenceKind.INTERACTION_HISTORY, "cue suppressed by fading"),
                    ),
                    proposed_action=Action.SILENCE,
                    cue=None,
                )
            elif faded_cue != base_analysis.cue:
                base_analysis = Analysis(
                    state=base_analysis.state,
                    confidence=base_analysis.confidence,
                    evidence=base_analysis.evidence,
                    proposed_action=base_analysis.proposed_action,
                    cue=faded_cue,
                )

        return base_analysis

    def _select_operation(self, text: str) -> str | None:
        for token, operation in self._VOCAB_OPERATIONS:
            if token in text:
                return operation
        return None

    def _is_performative_close(self, operation: str, text: str) -> bool:
        """Whether the current utterance already performed this operation."""
        markers = self.STALE_MARKERS.get(operation, ())
        return any(marker in text for marker in markers)

    def _pick_cue(self, operation: str, text: str) -> str | None:
        del text
        candidates = self.CUE_BY_OPERATION[operation]
        start = self._candidate_index_by_operation.get(operation, 0)
        for offset in range(len(candidates)):
            index = (start + offset) % len(candidates)
            candidate = candidates[index]
            if len(candidate) <= self.config.max_mid_speech_cue_characters:
                self._candidate_index_by_operation[operation] = (index + 1) % len(candidates)
                return candidate
        return None

    def _apply_fading(self, cue: str) -> Optional[str]:
        if self.knowledge_store is None:
            return cue
        from .cue_fading import CueFadingPolicy
        policy = CueFadingPolicy(self.knowledge_store)
        decision = policy.apply(cue, max_cue_characters=self.config.max_mid_speech_cue_characters)
        if not decision.should_deliver:
            return None
        return decision.faded_cue

    def reset_session(self) -> None:
        """Reset per-session state (called on SPEECH_STARTED)."""
        self._candidate_index_by_operation.clear()


def record_cue_delivery(
    knowledge_store: Optional[KnowledgeStore],
    cue: str,
    *,
    source_session_id: str = "",
    confidence: float = 0.7,
) -> None:
    """Record that a cue was delivered to the user."""
    if knowledge_store is None or not cue:
        return
    knowledge_store.record_cue_delivery(
        cue,
        source_session_id=source_session_id,
        confidence=confidence,
    )


def detect_spontaneous_reuse(
    knowledge_store: Optional[KnowledgeStore],
    speech_text: str,
    *,
    source_session_id: str = "",
) -> list:
    """Check user speech for spontaneous reuse of previously cued concepts."""
    if knowledge_store is None:
        return []
    return knowledge_store.detect_spontaneous_reuse(
        speech_text,
        source_session_id=source_session_id,
    )


def get_cue_fading_level(
    knowledge_store: Optional[KnowledgeStore],
    cue: str,
) -> int:
    """Get the fading level for a cue (0=full, 1=keyword, 2=silence)."""
    if knowledge_store is None:
        return 0
    return knowledge_store.get_fading_level(cue)


class UnifiedSessionAnalyzer:
    """Turn-boundary analyzer combining heuristic, structural diff, and knowledge state.

    This analyzer wraps a base analyzer and adds:

    1. Structural diff awareness (source vs user reasoning gaps)
    2. Knowledge state fading (suppress cues for internalized concepts)
    3. Spontaneous reuse detection on completed turn text

    Unlike MidSpeechUnifiedAnalyzer (which optimizes for low latency on partial
    transcripts), this analyzer has the full turn text and recent context
    available. It is designed for the turn-boundary decision path where
    accuracy matters more than speed.

    The structural diff is used as a fallback when the base heuristic does not
    produce a cue. This mirrors the mid-speech path's conservative approach:
    the heuristic leads, structural diff fills gaps.
    """

    def __init__(
        self,
        base_analyzer: Analyzer,
        *,
        source_structure: Optional[SessionStructure] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        structural_minimum_confidence: float = 0.3,
        max_cue_characters: int = 16,
    ) -> None:
        self.base_analyzer = base_analyzer
        self.source_structure = source_structure
        self.knowledge_store = knowledge_store
        self.structural_minimum_confidence = structural_minimum_confidence
        self.max_cue_characters = max_cue_characters

    def analyze(self, text: str, context: Optional[SessionContext] = None) -> Analysis:
        # Detect spontaneous reuse on completed turn text BEFORE fading check,
        # so the knowledge state is up-to-date when fading is applied.
        if self.knowledge_store is not None and text.strip():
            self.detect_spontaneous_reuse(text)

        base_analysis = self.base_analyzer.analyze(text, context)

        if base_analysis.state in {State.PRODUCTIVE, State.OVERLOAD}:
            return base_analysis

        if base_analysis.cue is None and self.source_structure is not None:
            turns = list(context.snapshot()) if context is not None else []
            if text.strip():
                turns.append(text)
            if turns:
                user_structure = structure_from_turns(turns)
                diffs = compare_structures(
                    user_structure,
                    self.source_structure,
                    minimum_confidence=self.structural_minimum_confidence,
                    max_cue_characters=self.max_cue_characters,
                )
                if diffs:
                    diff = diffs[0]
                    from .knowledge_diff import _diff_action
                    diff_action = _diff_action(diff.kind)
                    base_analysis = Analysis(
                        state=State.STRUCTURE_GAP,
                        confidence=min(0.9, base_analysis.confidence, diff.confidence + 0.2),
                        evidence=(
                            Evidence(EvidenceKind.TRANSCRIPT, diff.evidence),
                            Evidence(EvidenceKind.TRANSCRIPT, "structural difference proposed, not proven"),
                        ),
                        proposed_action=diff_action,
                        cue=diff.cue,
                    )

        if base_analysis.cue and self.knowledge_store is not None:
            faded_cue = self._apply_fading(base_analysis.cue)
            if faded_cue is None:
                return Analysis(
                    state=base_analysis.state,
                    confidence=base_analysis.confidence,
                    evidence=base_analysis.evidence + (
                        Evidence(EvidenceKind.INTERACTION_HISTORY, "cue suppressed by fading"),
                    ),
                    proposed_action=Action.SILENCE,
                    cue=None,
                )
            elif faded_cue != base_analysis.cue:
                base_analysis = Analysis(
                    state=base_analysis.state,
                    confidence=base_analysis.confidence,
                    evidence=base_analysis.evidence,
                    proposed_action=base_analysis.proposed_action,
                    cue=faded_cue,
                    source_refs=base_analysis.source_refs,
                )

        return base_analysis

    def detect_spontaneous_reuse(self, turn_text: str) -> list:
        """Detect spontaneous reuse of previously cued concepts in completed turn text.

        Should be called on each completed turn at the turn boundary.
        This catches concept reuse that may have appeared after the last
        mid-speech injection, using the full completed turn text.
        """
        if self.knowledge_store is None:
            return []
        return self.knowledge_store.detect_spontaneous_reuse(turn_text)

    def _apply_fading(self, cue: str) -> Optional[str]:
        if self.knowledge_store is None:
            return cue
        from .cue_fading import CueFadingPolicy
        policy = CueFadingPolicy(self.knowledge_store)
        decision = policy.apply(cue, max_cue_characters=self.max_cue_characters)
        if not decision.should_deliver:
            return None
        return decision.faded_cue