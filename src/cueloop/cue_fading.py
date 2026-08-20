"""Cue fading policy for CUELOOP.

This module implements Priority D: Cue Fading.  The progression is:

    explicit cue (full text)
    -> keyword-only cue (first keyword or operation label)
    -> SILENCE (suppress intervention entirely)

The fading is driven by the knowledge acquisition state in
``KnowledgeStore``.  A cue that has been spontaneously reused by the
user is faded.  A cue that has been explicitly retained is suppressed.

Design principles:
- Fading only proceeds when there is *evidence* of user internalization
  (spontaneous reuse).  We never assume internalization from repeated
  exposure alone.
- The faded cue is still a valid cue, just shorter.  It is never a
  completed answer.
- The controller retains final delivery authority; fading is a
  suggestion that the controller may override.
"""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge_state import KnowledgeStore, extract_keywords


@dataclass(frozen=True)
class FadingDecision:
    """The result of applying a fading policy to a cue."""

    original_cue: str
    faded_cue: str | None
    fading_level: int
    reason: str

    @property
    def should_deliver(self) -> bool:
        return self.faded_cue is not None and len(self.faded_cue) > 0


class CueFadingPolicy:
    """Determines whether and how a cue should be faded.

    The policy is pure and stateless: it reads from the ``KnowledgeStore``
    and returns a ``FadingDecision``.  It never modifies the store.
    """

    def __init__(self, knowledge_store: KnowledgeStore | None = None) -> None:
        self.knowledge_store = knowledge_store

    def apply(self, cue: str, *, max_cue_characters: int = 16) -> FadingDecision:
        """Apply fading to a cue based on knowledge acquisition state.

        Parameters
        ----------
        cue:
            The original cue text proposed by the analyzer.
        max_cue_characters:
            The maximum allowed length for a faded cue.

        Returns
        -------
        FadingDecision
            Contains the (possibly faded) cue and the reasoning.
        """
        if not cue or not cue.strip():
            return FadingDecision(cue, None, 0, "empty cue")

        if self.knowledge_store is None:
            return FadingDecision(cue, cue, 0, "no knowledge store")

        level = self.knowledge_store.get_fading_level(cue)

        if level == 0:
            return FadingDecision(cue, cue, 0, "no fading evidence")

        if level == 2:
            return FadingDecision(
                cue, None, 2, "concept explicitly retained or expired"
            )

        # level == 1: keyword-only fading
        keywords = extract_keywords(cue, max_keywords=1)
        if not keywords:
            return FadingDecision(cue, None, 1, "no keywords extracted")

        keyword = keywords[0]
        if len(keyword) > max_cue_characters:
            keyword = keyword[:max_cue_characters]

        return FadingDecision(
            cue, keyword, 1, "concept spontaneously reused, fading to keyword"
        )


def keyword_cue(cue: str, max_characters: int = 8) -> str | None:
    """Extract a single keyword from a cue for faded delivery.

    Returns None if no suitable keyword can be extracted.
    """
    keywords = extract_keywords(cue, max_keywords=1)
    if not keywords:
        return None
    kw = keywords[0]
    if len(kw) > max_characters:
        kw = kw[:max_characters]
    return kw if kw else None
