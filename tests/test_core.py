import json
from pathlib import Path
import unittest

from cueloop.cli import build_parser
from cueloop.core import (
    Action,
    ALLOWED_STATE_ACTIONS,
    Analysis,
    EvidenceKind,
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
        self.assertEqual(analysis.state, State.PRODUCTIVE)
        self.assertEqual(analysis.proposed_action, Action.SILENCE)

    def test_unfinished_turn_preserves_turn(self) -> None:
        analysis = self.analyzer.analyze("まだ比較軸を決めていないので")
        self.assertEqual(analysis.state, State.RETRIEVAL_FAILURE)
        self.assertEqual(analysis.proposed_action, Action.BACKCHANNEL)

    def test_repeated_fillers_trigger_minimal_cue(self) -> None:
        analysis = self.analyzer.analyze("えっと、その、なんだろう、性能が良いと思う")
        self.assertEqual(analysis.state, State.STRUCTURE_GAP)
        self.assertEqual(analysis.proposed_action, Action.QUESTION)
        self.assertEqual(analysis.cue, "何に依存？")

    def test_explicit_overload_stays_silent(self) -> None:
        analysis = self.analyzer.analyze("情報が多すぎて混乱した")
        self.assertEqual(analysis.state, State.OVERLOAD)
        self.assertEqual(analysis.proposed_action, Action.SILENCE)

    def test_strategy_failure_gets_one_constraint(self) -> None:
        analysis = self.analyzer.analyze("何からどう進めればよいかわからない")
        self.assertEqual(analysis.state, State.STRATEGY_FAILURE)
        self.assertEqual(analysis.proposed_action, Action.CONSTRAIN)
        self.assertEqual(analysis.cue, "結論は？")

    def test_repetition_without_structure_terms_is_fixation(self) -> None:
        analysis = self.analyzer.analyze("同じ案同じ案")
        self.assertEqual(analysis.state, State.FIXATION)
        self.assertEqual(analysis.proposed_action, Action.PERTURB)


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
                "state": "STRUCTURE_GAP",
                "confidence": 0.82,
                "evidence": [{"kind":"TRANSCRIPT","description":"同じ比較表現が反復している","source_span":None}],
                "proposed_action": "QUESTION",
                "cue": "何に依存？",
                "source_refs": [],
            },
            ensure_ascii=False,
        )
        analysis = parse_analysis_json(raw)
        self.assertEqual(analysis.state, State.STRUCTURE_GAP)
        self.assertEqual(analysis.proposed_action, Action.QUESTION)
        self.assertEqual(analysis.cue, "何に依存？")
        self.assertEqual(analysis.evidence[0].kind, EvidenceKind.TRANSCRIPT)

    def test_invalid_json_fails_closed(self) -> None:
        analysis = parse_analysis_json("not-json")
        self.assertEqual(analysis.proposed_action, Action.SILENCE)
        self.assertEqual(analysis.confidence, 0.0)

    def test_flowing_cannot_propose_intervention(self) -> None:
        raw = json.dumps(
            {
                "state": "PRODUCTIVE",
                "confidence": 0.91,
                "evidence": [{"kind":"TRANSCRIPT","description":"論点が連続して展開している","source_span":None}],
                "proposed_action": "QUESTION",
                "cue": "逆なら？",
                "source_refs": [],
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

    def test_retrieve_without_source_refs_fails_closed(self) -> None:
        raw = json.dumps(
            {
                "state": "RETRIEVAL_FAILURE",
                "confidence": 0.9,
                "evidence": [{"kind":"USER_SIGNAL","description":"用語を思い出せないと明示している","source_span":None}],
                "proposed_action": "RETRIEVE",
                "cue": "用語は制御理論",
                "source_refs": [],
            },
            ensure_ascii=False,
        )
        analysis = parse_analysis_json(raw)
        self.assertEqual(analysis.proposed_action, Action.SILENCE)
        self.assertEqual(analysis.confidence, 0.0)

    def test_bounded_context_is_included_in_prompt(self) -> None:
        captured: list[PromptRequest] = []

        def completion(request: PromptRequest) -> str:
            captured.append(request)
            return json.dumps(
                {
                    "state": "PRODUCTIVE",
                    "confidence": 0.9,
                    "evidence": [{"kind":"TRANSCRIPT","description":"論点が継続している","source_span":None}],
                    "proposed_action": "SILENCE",
                    "cue": None,
                    "source_refs": [],
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
                    "state": "PRODUCTIVE",
                    "confidence": 0.9,
                    "evidence": [{"kind":"TRANSCRIPT","description":"因果関係を連続して展開している","source_span":None}],
                    "proposed_action": "SILENCE",
                    "cue": None,
                    "source_refs": [],
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
        self.assertEqual(first.action, Action.QUESTION)
        self.assertEqual(second.action, Action.SILENCE)

    def test_ungrounded_retrieval_is_always_suppressed(self) -> None:
        analysis = Analysis(
            State.RETRIEVAL_FAILURE,
            0.95,
            ("explicit request for an external fact",),
            Action.RETRIEVE,
            "用語を検索",
        )
        decision = InterventionController().decide(analysis)
        self.assertEqual(decision.action, Action.SILENCE)
        self.assertIn("grounded", decision.reason)

    def test_grounded_retrieval_can_deliver_minimal_source_aware_text(self) -> None:
        analysis = Analysis(
            State.RETRIEVAL_FAILURE,
            0.95,
            ("explicit request for a known term",),
            Action.RETRIEVE,
            "用語は制御理論",
            ("notes/control.md#L8",),
        )
        decision = InterventionController(
            trusted_source_refs=frozenset({"notes/control.md#L8"})
        ).decide(analysis)
        self.assertEqual(decision.action, Action.RETRIEVE)
        self.assertEqual(decision.analysis.source_refs, ("notes/control.md#L8",))

    def test_formal_state_and_action_spaces_match_schemas(self) -> None:
        root = Path(__file__).resolve().parents[1]
        state_schema = json.loads((root / "schemas/human-state.schema.json").read_text(encoding="utf-8"))
        action_schema = json.loads((root / "schemas/intervention-candidate.schema.json").read_text(encoding="utf-8"))
        self.assertEqual({state.value for state in State}, set(state_schema["properties"]["state"]["enum"]))
        self.assertEqual({action.value for action in Action}, set(action_schema["properties"]["action"]["enum"]))
        self.assertEqual(
            {kind.value for kind in EvidenceKind},
            set(state_schema["properties"]["evidence"]["items"]["properties"]["kind"]["enum"]),
        )

    def test_prohibited_state_action_pair_fails_closed(self) -> None:
        analysis = Analysis(
            State.FIXATION,
            0.95,
            ("repeated local path",),
            Action.BACKCHANNEL,
        )
        decision = InterventionController().decide(analysis)
        self.assertEqual(decision.action, Action.SILENCE)
        self.assertIn("prohibited", decision.reason)

    def test_every_prohibited_formal_state_action_pair_fails_closed(self) -> None:
        for state in State:
            for action in Action:
                if action in ALLOWED_STATE_ACTIONS[state]:
                    continue
                cue = "一つ確認？" if action in {
                    Action.QUESTION, Action.MIRROR, Action.PERTURB, Action.CONSTRAIN, Action.RETRIEVE
                } else None
                refs = ("trusted",) if action is Action.RETRIEVE else ()
                analysis = Analysis(state, 0.99, ("observable",), action, cue, refs)
                decision = InterventionController(
                    trusted_source_refs=frozenset({"trusted"})
                ).decide(analysis)
                self.assertEqual(
                    decision.action,
                    Action.SILENCE,
                    f"{state.value}/{action.value} should be prohibited",
                )


class CLITests(unittest.TestCase):
    def test_analyzer_command_accepts_multiple_tokens(self) -> None:
        args = build_parser().parse_args(["--analyzer-command", "python", "adapter.py"])
        self.assertEqual(args.analyzer_command, ["python", "adapter.py"])


if __name__ == "__main__":
    unittest.main()
