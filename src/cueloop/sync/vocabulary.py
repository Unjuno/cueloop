"""Self-contained cognitive-operation vocabulary for the sync core.

The vocabulary is intentionally small. A cue names one operation the speaker
could perform next; it is not an answer and not an evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

OPERATIONS: tuple[str, ...] = (
    "boundary",
    "comparison",
    "conclusion",
    "concretize",
    "dependency",
    "ground",
    "invert",
    "mechanism",
)

TOKEN_OPERATIONS: tuple[tuple[str, str], ...] = (
    ("なぜなら", "mechanism"),
    ("仕組み", "mechanism"),
    ("理由", "mechanism"),
    ("なぜ", "mechanism"),
    ("どうして", "mechanism"),
    ("だから", "conclusion"),
    ("つまり", "conclusion"),
    ("なので", "conclusion"),
    ("結論", "conclusion"),
    ("根拠", "ground"),
    ("証拠", "ground"),
    ("データ", "ground"),
    ("事実", "ground"),
    ("出典", "ground"),
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

# Adjacency prior: operations that most naturally extend the reasoning move
# the speaker just made. Used to prefer a *context-fitting* missing operation
# over a purely global one. Each tuple is biased toward moves that elaborate,
# qualify, or test the current move before jumping to an unrelated one.
CONTEXT_LINKS: dict[str, tuple[str, ...]] = {
    "dependency": ("concretize", "conclusion", "ground"),
    "concretize": ("dependency", "conclusion", "comparison"),
    "ground": ("mechanism", "conclusion", "concretize"),
    "mechanism": ("ground", "invert", "conclusion"),
    "conclusion": ("invert", "comparison", "mechanism"),
    "invert": ("concretize", "comparison", "conclusion"),
    "comparison": ("invert", "conclusion", "ground"),
    "boundary": ("concretize", "dependency", "conclusion"),
}


OPERATION_BY_CUE: dict[str, str] = {
    cue: operation for operation, cues in CUE_BY_OPERATION.items() for cue in cues
}

PRIORITY: tuple[str, ...] = (
    "ground",
    "boundary",
    "dependency",
    "comparison",
    "invert",
    "concretize",
    "mechanism",
    "conclusion",
)


def tag_operations(text: str) -> list[str]:
    """Return operations in the order their first occurrence appears."""
    occurrences: list[tuple[int, str]] = []
    for token, operation in TOKEN_OPERATIONS:
        start = text.find(token)
        if start >= 0:
            occurrences.append((start, operation))
    occurrences.sort(key=lambda item: item[0])
    ordered: list[str] = []
    seen: set[str] = set()
    for _, operation in occurrences:
        if operation not in seen:
            ordered.append(operation)
            seen.add(operation)
    return ordered


def cue_for(
    operation: str,
    *,
    max_characters: int = 8,
    used_cues: Sequence[str] = (),
) -> str | None:
    used = set(used_cues)
    for candidate in CUE_BY_OPERATION.get(operation, ()):
        if len(candidate) <= max_characters and candidate not in used:
            return candidate
    return None


def operation_for_cue(cue: str) -> str | None:
    return OPERATION_BY_CUE.get(cue)
