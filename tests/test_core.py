import json
import unittest

from cueloop.cli import build_parser
from cueloop.core import (
    Action,
    HeuristicAnalyzer,
    InterventionController,
    SessionContext,
    State,
    run_turns,
)
from cueloop.llm import LLMAnalyzer, PromptRequest, parse_analysis_json


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


class SessionContextTests(unittest.TestCase):
    def test_context_keeps_only_recent_turns(self) -> None:
        context = SessionContext(max_turns=2, max_characters=100)
        context.add("first")
        context.add("second")
        context.add("third")
        self.assertEqual(context.snapshot(), ("second", "third"))

    def test_context_respects_character_bound(self) -> None:
        context = SessionContext(max_turns=4, max_characters=8)
        context.add("1234")
        context.add("5678")
        context.add("90")
        self.assertEqual(sum(len(turn) for turn in context.snapshot()), 6)
        self.assertEqual(context.snapshot(), ("5678", "90"))


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

    def test_bounded_context_is_included_in_prompt(self) -> None:
        captured: list[PromptRequest] = []

        def completion(request: PromptRequest) -> str:
            captured.append(request)
            return json.dumps(
                {
                    "state": "FLOWING",
                    "confidence": 0.9,
                    "evidence": ["論点が継続している"],
                    "proposed_action": "SILENCE",
                    "cue": None,
                },
                ensure_ascii=False,
            )

        context = SessionContext(max_turns=2, max_characters=100)
        context.add("前の論点")
        LLMAnalyzer(completion).analyze("現在の論点", context)
        self.assertIn("前の論点", captured[0].user)
        self.assertIn("現在の論点", captured[0].user)

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


class CLITests(unittest.TestCase):
    def test_analyzer_command_accepts_multiple_tokens(self) -> None:
        args = build_parser().parse_args(["--analyzer-command", "python", "adapter.py"])
        self.assertEqual(args.analyzer_command, ["python", "adapter.py"])


if __name__ == "__main__":
    unittest.main()
