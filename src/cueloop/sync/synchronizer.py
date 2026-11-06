"""Live synchronization between the speaker's partial transcript and a prediction."""

from __future__ import annotations

from .models import Prediction

CLAUSE_ENDINGS: frozenset[str] = frozenset("。．.!?！？、,，;；")

# When the speaker announces an operation, the cue should land before the
# content that follows the announcement. These suffixes are the audible
# "I am about to ..." moments the original design targeted.
TRIGGER_PHRASES: tuple[str, ...] = (
    "結論は",
    "結論として",
    "つまり",
    "だから",
    "なので",
    "前提は",
    "依存は",
    "逆に",
    "例えば",
    "具体的には",
    "なぜ",
    "どうして",
    "条件は",
    "範囲は",
)


def longest_common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def estimate_chars_per_second(
    characters: int,
    elapsed_ms: int,
    *,
    prior: float = 4.5,
) -> float:
    """Estimate speech rate, clamped to a plausible Japanese speaking range."""
    if characters <= 0 or elapsed_ms <= 0:
        return prior
    estimate = characters / (elapsed_ms / 1_000)
    return min(12.0, max(1.0, estimate))


class Synchronizer:
    """Tracks one utterance and reports clause boundaries and divergence.

    A prediction is anchored at the transcript length where it was made. Once
    the speaker produces characters, those characters are compared with the
    predicted span. A mismatch means the speaker took a different path; the
    runtime then cancels any cue that was scheduled for the old path.
    """

    def __init__(self, *, pause_ms: int = 350) -> None:
        if pause_ms < 0:
            raise ValueError("pause_ms must be non-negative")
        self.pause_ms = pause_ms
        self.utterance_id = ""
        self.text = ""
        self.started_at_ms = 0
        self.last_change_at_ms = 0
        self.last_boundary_length = 0
        self.chars_per_second = 4.5
        self.prediction: Prediction | None = None
        self.prediction_match = 0
        self.diverged = False
        self.last_clause = ""
        self._prediction_base_text = ""
        self._changed = False

    def start(self, *, utterance_id: str, at_ms: int) -> None:
        self.utterance_id = utterance_id
        self.text = ""
        self.started_at_ms = at_ms
        self.last_change_at_ms = at_ms
        self.last_boundary_length = 0
        self.chars_per_second = 4.5
        self.prediction = None
        self.prediction_match = 0
        self.diverged = False
        self.last_clause = ""
        self._prediction_base_text = ""
        self._changed = False

    def observe(self, text: str, *, now_ms: int) -> bool:
        """Update the transcript and return whether a clause boundary is due."""
        cleaned = text.strip()
        if cleaned != self.text:
            self.text = cleaned
            self.last_change_at_ms = now_ms
            self.chars_per_second = estimate_chars_per_second(
                len(cleaned),
                max(1, now_ms - self.started_at_ms),
            )
            self._update_match()
            self._changed = True
        else:
            self._changed = False
        return self._boundary_due(now_ms)

    def check_pause(self, *, now_ms: int) -> bool:
        """Return whether a pause boundary is due without new transcript text."""
        self._changed = False
        return self._boundary_due(now_ms)

    def _boundary_due(self, now_ms: int) -> bool:
        if not self.text or len(self.text) <= self.last_boundary_length:
            return False
        punctuated = self.text[-1] in CLAUSE_ENDINGS
        paused = now_ms - self.last_change_at_ms >= self.pause_ms
        triggered = any(self.text.endswith(phrase) for phrase in TRIGGER_PHRASES)
        if not (punctuated or paused or triggered):
            return False
        if not self._changed and not paused:
            return False
        self.last_clause = self.text[self.last_boundary_length :]
        self.last_boundary_length = len(self.text)
        return True

    def clause_text(self) -> str:
        return self.text[self.last_boundary_length :]

    def set_prediction(self, prediction: Prediction | None) -> None:
        self.prediction = prediction
        self.prediction_match = 0
        self.diverged = False
        self._prediction_base_text = (
            self.text[: prediction.base_length] if prediction else ""
        )
        if prediction is not None:
            self._update_match()

    def clear_prediction(self) -> None:
        self.prediction = None
        self.prediction_match = 0
        self.diverged = False
        self._prediction_base_text = ""

    def _update_match(self) -> None:
        prediction = self.prediction
        if prediction is None:
            self.prediction_match = 0
            self.diverged = False
            return
        if not self.text.startswith(self._prediction_base_text):
            self.prediction_match = 0
            self.diverged = True
            return
        spoken_since = self.text[prediction.base_length :]
        self.prediction_match = longest_common_prefix(spoken_since, prediction.span)
        self.diverged = bool(spoken_since) and self.prediction_match < len(spoken_since)

    def pause_due_within(self, now_ms: int, lead_ms: int) -> bool:
        """Whether the pause threshold will be reached within ``lead_ms``.

        This lets the runtime insert a cue *before* the pause is confirmed.
        It is a prediction, not a boundary: the speaker may resume first, so
        the caller should use a small lead and keep cancellation enabled.
        """
        if lead_ms < 0:
            raise ValueError("lead_ms must be non-negative")
        if not self.text or len(self.text) <= self.last_boundary_length:
            return False
        return now_ms - self.last_change_at_ms >= self.pause_ms - lead_ms

    def eta_ms(self, characters: int) -> int:
        """Estimated time until ``characters`` more characters are spoken."""
        if characters <= 0:
            return 0
        return round(characters / self.chars_per_second * 1_000)
