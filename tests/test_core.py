import unittest

from cuelloop.core import Action, HeuristicAnalyzer, InterventionController, State


class HeuristicAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = HeuristicAnalyzer()

    def test_flowing_reasoning_stays_silent(self) -> None:
        analysis = self.analyzer.analyze(
            "この設計は前提と依存関係を分けて考える必要がある。反例も確認する。"
        )
        self.assertEqual(analysis.state, State.FLOWING)
        self.assertEqual(analysis.proposed_action, Action.SILENCE)

    def test_unfinished_turn_preserves_turn(self) -> None:
        analysis = self.analyzer.analyze("まだ比較軸を決めていないので")
        self.assertEqual(analysis.state, State.PAUSED)
        self.assertEqual(analysis.proposed_action, Action.CONTINUE)

    def test_repeated_fillers_trigger_minimal_cue(self) -> None:
        analysis = self.analyzer.analyze("えっと、その、なんだろう、性能が良いと思う")
        self.assertEqual(analysis.state, State.STUCK)
        self.assertEqual(analysis.proposed_action, Action.CUE)
        self.assertEqual(analysis.cue, "何に依存？")


class InterventionControllerTests(unittest.TestCase):
    def test_low_confidence_is_suppressed(self) -> None:
        analyzer = HeuristicAnalyzer()
        controller = InterventionController()
        decision = controller.decide(analyzer.analyze("短い"))
        self.assertEqual(decision.action, Action.SILENCE)

    def test_flowing_always_stays_silent(self) -> None:
        analyzer = HeuristicAnalyzer()
        controller = InterventionController()
        decision = controller.decide(
            analyzer.analyze("前提、根拠、反例を分けて考える必要がある。")
        )
        self.assertEqual(decision.action, Action.SILENCE)

    def test_cooldown_suppresses_repeated_intervention(self) -> None:
        analyzer = HeuristicAnalyzer()
        controller = InterventionController()
        first = controller.decide(analyzer.analyze("えっと、その、なんだろう、性能が良い"))
        second = controller.decide(analyzer.analyze("えっと、その、なんだろう、性能が良い"))
        self.assertEqual(first.action, Action.CUE)
        self.assertEqual(second.action, Action.SILENCE)


if __name__ == "__main__":
    unittest.main()
