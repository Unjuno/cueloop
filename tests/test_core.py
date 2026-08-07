import json
import unittest

from cuelloop.core import Action, HeuristicAnalyzer, InterventionController, State, run_turns
from cuelloop.llm import LLMAnalyzer, PromptRequest, parse_analysis_json


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


class LLMAnalyzerContractTests(unittest.TestCase):
    def test_valid_json_maps_to_analysis(self) -> None:
        raw = json.dumps(
            {
                "state": "STUCK",
                "confidence": 0.82,
                "evidence": ["同じ比較表現が反復している"],
                "proposed_action": "CUE",
                "cue": "何に依存？",
            },
            ensure_ascii=False,
        )
        analysis = parse_analysis_json(raw)
        self.assertEqual(analysis.state, State.STUCK)
        self.assertEqual(analysis.proposed_action, Action.CUE)
        self.assertEqual(analysis.cue, "何に依存？")

    def test_invalid_json_fails_closed(self) -> None:
        analysis = parse_analysis_json("not-json")
        self.assertEqual(analysis.proposed_action, Action.SILENCE)
        self.assertEqual(analysis.confidence, 0.0)

    def test_flowing_cannot_propose_intervention(self) -> None:
        raw = json.dumps(
            {
                "state": "FLOWING",
                "confidence": 0.91,
                "evidence": ["論点が連続して展開している"],
                "proposed_action": "CUE",
                "cue": "逆なら？",
            },
            ensure_ascii=False,
        )
        analysis = parse_analysis_json(raw)
        self.assertEqual(analysis.proposed_action, Action.SILENCE)
        self.assertEqual(analysis.confidence, 0.0)

    def test_provider_failure_fails_closed(self) -> None:
        def failing_completion(_: PromptRequest) -> str:
            raise RuntimeError("provider unavailable")

        analysis = LLMAnalyzer(failing_completion).analyze("考え中")
        self.assertEqual(analysis.proposed_action, Action.SILENCE)
        self.assertEqual(analysis.confidence, 0.0)

    def test_run_turns_accepts_injected_analyzer(self) -> None:
        def completion(_: PromptRequest) -> str:
            return json.dumps(
                {
                    "state": "FLOWING",
                    "confidence": 0.9,
                    "evidence": ["因果関係を連続して展開している"],
                    "proposed_action": "SILENCE",
                    "cue": None,
                },
                ensure_ascii=False,
            )

        decisions = run_turns(["入力"], analyzer=LLMAnalyzer(completion))
        self.assertEqual(decisions[0].action, Action.SILENCE)


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
