import time
import unittest

from cueloop.core import Action, Analysis, SessionContext, State
from cueloop.prefetch import PrefetchConfig, PrefetchingAnalyzer


class ImmediateAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        self.calls += 1
        return Analysis(
            state=State.STUCK,
            confidence=0.9,
            evidence=("semantic loop detected",),
            proposed_action=Action.CUE,
            cue="反例は？",
        )


class SlowAnalyzer:
    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        time.sleep(0.15)
        return Analysis(
            state=State.STUCK,
            confidence=0.9,
            evidence=("semantic loop detected",),
            proposed_action=Action.CUE,
            cue="反例は？",
        )


class PrefetchTests(unittest.TestCase):
    def test_ready_exact_text_result_is_reused(self) -> None:
        delegate = ImmediateAnalyzer()
        analyzer = PrefetchingAnalyzer(delegate, PrefetchConfig(minimum_characters=1, boundary_wait_ms=50))
        try:
            analyzer.prefetch("same reasoning turn")
            time.sleep(0.02)
            result = analyzer.analyze("same reasoning turn")
            self.assertEqual(result.proposed_action, Action.CUE)
            self.assertEqual(delegate.calls, 1)
        finally:
            analyzer.close()

    def test_not_ready_at_boundary_fails_closed(self) -> None:
        analyzer = PrefetchingAnalyzer(SlowAnalyzer(), PrefetchConfig(minimum_characters=1, boundary_wait_ms=5))
        try:
            analyzer.prefetch("slow reasoning turn")
            result = analyzer.analyze("slow reasoning turn")
            self.assertEqual(result.proposed_action, Action.SILENCE)
            self.assertEqual(result.confidence, 0.0)
            self.assertIn("not ready", result.evidence[0])
        finally:
            analyzer.close()

    def test_different_final_text_does_not_reuse_stale_candidate(self) -> None:
        delegate = ImmediateAnalyzer()
        analyzer = PrefetchingAnalyzer(delegate, PrefetchConfig(minimum_characters=1, boundary_wait_ms=50))
        try:
            analyzer.prefetch("partial version")
            time.sleep(0.02)
            result = analyzer.analyze("final version")
            self.assertEqual(result.proposed_action, Action.CUE)
            self.assertEqual(delegate.calls, 2)
        finally:
            analyzer.close()


if __name__ == "__main__":
    unittest.main()
