"""Stem cues (initial-condition sentences) and the --cue-style switch."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cueloop.sync import cli as sync_cli
from cueloop.sync.models import Prediction, SyncState
from cueloop.sync.policy import (
    CanonicalAssigner,
    LocalLLMDecisionModel,
    RandomOperationDecisionModel,
    RuleDecisionModel,
)
from cueloop.sync.stem import (
    StemCueGenerator,
    finalize_stem,
    render_stem_template,
)
from cueloop.sync.vocabulary import tag_operations


def _prediction(span, operations=(), confidence=0.9):
    return Prediction(
        span=span,
        operations=tuple(operations) or tuple(tag_operations(span)),
        confidence=confidence,
        source="test",
        created_at_ms=0,
        base_length=0,
    )


def _state(**overrides):
    values = {
        "utterance_id": "u1",
        "text": "この設計は前提を分けて考える。",
        "context": (),
        "now_ms": 1_000,
        "started_at_ms": 0,
        "chars_per_second": 4.5,
        "at_boundary": True,
        "prediction": _prediction("だから結論は決まる。"),
        "prediction_match": 0,
        "diverged": False,
        "clause_operations": frozenset(),
        "recent_operations": frozenset(),
    }
    values.update(overrides)
    return SyncState(**values)


class FinalizeStemTests(unittest.TestCase):
    def test_accepts_sentence_and_appends_terminator(self) -> None:
        sentence = finalize_stem("根拠は、観測データでは弱い、からです")
        self.assertEqual(sentence, "根拠は、観測データでは弱い、からです。")

    def test_strips_wrapping_quotes_and_collapses_whitespace(self) -> None:
        sentence = finalize_stem("「根拠は、\n実験で弱いと示されている。」")
        self.assertEqual(sentence, "根拠は、 実験で弱いと示されている。")

    def test_rejects_overlong_sentence(self) -> None:
        self.assertIsNone(finalize_stem("あ" * 49, max_characters=48))

    def test_rejects_json_leakage(self) -> None:
        self.assertIsNone(finalize_stem('{"action":"CUE"}'))
        self.assertIsNone(finalize_stem("根拠は```です。"))

    def test_rejects_empty_input(self) -> None:
        self.assertIsNone(finalize_stem(""))
        self.assertIsNone(finalize_stem("   "))


class StemTemplateTests(unittest.TestCase):
    def test_renders_ground_template(self) -> None:
        self.assertEqual(
            render_stem_template("ground", "観測データでは弱い。"),
            "根拠は、観測データでは弱いからです。",
        )

    def test_renders_invert_template(self) -> None:
        self.assertEqual(
            render_stem_template("invert", "実装コストは許容できる"),
            "逆に言うと、実装コストは許容できるです。",
        )

    def test_empty_span_or_unknown_operation_returns_none(self) -> None:
        self.assertIsNone(render_stem_template("ground", ""))
        self.assertIsNone(render_stem_template("not-an-operation", "span"))

    def test_overlong_span_is_trimmed(self) -> None:
        template = render_stem_template("ground", "あ" * 100)
        self.assertIsNotNone(template)
        assert template is not None
        self.assertLessEqual(len(template), 48)


class RuleStemModeTests(unittest.TestCase):
    class _StubGenerator:
        def __init__(self, sentence):
            self.sentence = sentence
            self.calls = []

        def generate(self, operation, *, text, span, context=()):
            self.calls.append(operation)
            return self.sentence

    def test_stem_cue_is_spoken_and_operation_is_preserved(self) -> None:
        generator = self._StubGenerator("根拠は、実験で弱いと示されているからです。")
        model = RuleDecisionModel(
            CanonicalAssigner(),
            context_fit=False,
            cue_style="stem",
            stem_generator=generator,
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(
            intervention.cue, "根拠は、実験で弱いと示されているからです。"
        )
        # The whole point: the operation tag survives a non-vocabulary sentence.
        self.assertEqual(intervention.operation, "ground")
        self.assertEqual(intervention.requested_operation, "ground")
        self.assertEqual(intervention.cue_style, "stem")
        self.assertEqual(generator.calls, ["ground"])

    def test_generator_failure_falls_back_to_span_template(self) -> None:
        model = RuleDecisionModel(
            CanonicalAssigner(),
            context_fit=False,
            cue_style="stem",
            stem_generator=self._StubGenerator(None),
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(
            intervention.cue, "根拠は、だから結論は決まるからです。"
        )

    def test_bare_mode_is_unchanged(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.cue, "根拠は？")
        self.assertEqual(intervention.cue_style, "bare")


class RandomOperationStemModeTests(unittest.TestCase):
    def test_drawn_operation_gets_its_own_stem(self) -> None:
        class _StubGenerator:
            def __init__(self) -> None:
                self.drawn = []

            def generate(self, operation, *, text, span, context=()):
                self.drawn.append(operation)
                return f"{operation} の初期条件です。"

        inner = RuleDecisionModel(
            CanonicalAssigner(), min_confidence=0.0, context_fit=False
        )
        stub = _StubGenerator()
        model = RandomOperationDecisionModel(
            inner,
            CanonicalAssigner(),
            seed=1,
            cue_style="stem",
            stem_generator=stub,
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.cue_style, "stem")
        # The stem is rendered for the drawn operation, not the requested one.
        self.assertEqual(intervention.cue, f"{stub.drawn[0]} の初期条件です。")


class StemGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                user = request["messages"][-1]["content"]
                if "json-please" in user:
                    content = '{"action": "CUE", "operation": "ground", "cue": "x"}'
                else:
                    content = "「根拠は、実験で弱いと示されているからです。」"
                body = json.dumps(
                    {"choices": [{"message": {"content": content}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_generates_finalized_sentence(self) -> None:
        generator = StemCueGenerator(
            base_url=self.base_url,
            model="test-model",
            timeout=2,
        )
        sentence = generator.generate(
            "ground",
            text="この設計は前提を分けて考える。",
            span="観測データでは弱い",
        )
        # Wrapping quotes are stripped and the terminator is normalized away.
        self.assertEqual(sentence, "根拠は、実験で弱いと示されているからです。")

    def test_json_leakage_falls_back_to_template(self) -> None:
        generator = StemCueGenerator(
            base_url=self.base_url,
            model="test-model",
            timeout=2,
        )
        sentence = generator.generate(
            "ground",
            text="json-please here",
            span="観測データでは弱い",
        )
        self.assertEqual(sentence, "根拠は、観測データでは弱いからです。")

    def test_unreachable_server_falls_back_to_template(self) -> None:
        generator = StemCueGenerator(
            base_url="http://127.0.0.1:1",
            model="test-model",
            timeout=0.5,
        )
        sentence = generator.generate(
            "ground",
            text="この設計は前提を分けて考える。",
            span="観測データでは弱い",
        )
        self.assertEqual(sentence, "根拠は、観測データでは弱いからです。")


class LocalDecisionStemModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                system = request["messages"][0]["content"]
                if request.get("max_tokens") == 1:
                    content = "ping"
                elif "初期条件の一文" in system:
                    content = (
                        '{"action": "CUE", "operation": "ground", '
                        '"cue": "根拠は、実験で弱いと示されているからです。", '
                        '"confidence": 0.8, "reason": "test"}'
                    )
                elif "意思決定モデル" in system:
                    content = (
                        '{"action": "CUE", "operation": "ground", '
                        '"cue": "根拠は？", "confidence": 0.8, "reason": "test"}'
                    )
                else:
                    content = '{"span": "x"}'
                body = json.dumps(
                    {"choices": [{"message": {"content": content}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_stem_mode_keeps_operation_for_full_sentence(self) -> None:
        model = LocalLLMDecisionModel(
            base_url=self.base_url,
            model="test-model",
            assigner=CanonicalAssigner(),
            timeout=2,
            cue_style="stem",
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(
            intervention.cue, "根拠は、実験で弱いと示されているからです。"
        )
        self.assertEqual(intervention.operation, "ground")
        self.assertEqual(intervention.cue_style, "stem")

    def test_bare_mode_is_unchanged(self) -> None:
        model = LocalLLMDecisionModel(
            base_url=self.base_url,
            model="test-model",
            assigner=CanonicalAssigner(),
            timeout=2,
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.cue, "根拠は？")
        self.assertEqual(intervention.cue_style, "bare")


class StemCliTests(unittest.TestCase):
    def test_stem_is_the_default_cue_style(self) -> None:
        args = sync_cli.build_parser().parse_args(["--events", "x.jsonl"])
        self.assertEqual(args.cue_style, "stem")

    def test_bare_can_be_selected(self) -> None:
        args = sync_cli.build_parser().parse_args(
            ["--events", "x.jsonl", "--cue-style", "bare"]
        )
        self.assertEqual(args.cue_style, "bare")


if __name__ == "__main__":
    unittest.main()
