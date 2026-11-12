import json
import random
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cueloop.sync.evaluate import summarize
from cueloop.sync.models import Delivery, Intervention, Outcome, Prediction, SyncState
from cueloop.sync.policy import (
    AlternateAssigner,
    AsyncDecisionModel,
    CanonicalAssigner,
    FixedAssigner,
    LocalLLMDecisionModel,
    RandomAssigner,
    RandomOperationDecisionModel,
    RuleDecisionModel,
    build_assigner,
)
from cueloop.sync.predictors import (
    AsyncPredictor,
    LocalLLMPredictor,
    NgramPredictor,
    ScriptedPredictor,
    choose_model,
    parse_prediction_payload,
    resolve_model,
)
from cueloop.sync.synchronizer import Synchronizer, longest_common_prefix
from cueloop.sync.vocabulary import OPERATIONS, operation_for_cue, tag_operations


def _prediction(span, operations=(), confidence=0.9, base_length=0):
    return Prediction(
        span=span,
        operations=tuple(operations) or tuple(tag_operations(span)),
        confidence=confidence,
        source="test",
        created_at_ms=0,
        base_length=base_length,
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
        "prediction": _prediction("だから結論は決まる。", base_length=15),
        "prediction_match": 0,
        "diverged": False,
        "clause_operations": frozenset(),
        "recent_operations": frozenset(),
    }
    values.update(overrides)
    return SyncState(**values)


def _outcome(
    *,
    operation: str = "ground",
    base_text: str = "この設計は前提を分けて考える。",
    continuation: str = "根拠は観測データだ。",
) -> Outcome:
    prediction = _prediction("だから結論は決まる。", base_length=len(base_text))
    intervention = Intervention(
        intervention_id="cue-test",
        detected_at_ms=900,
        operation=operation,
        requested_operation=operation,
        cue="根拠は？",
        reason="test",
        confidence=0.9,
        scheduled_at_ms=1_000,
        expires_at_ms=2_000,
        insertion_delay_ms=100,
        assignment="canonical",
        base_text=base_text,
        prediction=prediction,
    )
    delivery = Delivery(
        intervention=intervention,
        status="DELIVERED",
        delivered_at_ms=1_000,
        text_at_delivery=base_text,
    )
    return Outcome(
        delivery=delivery,
        window_end_ms=3_000,
        continuation=continuation,
        target_operation_seen=True,
        finalized=True,
    )


class SummaryTests(unittest.TestCase):
    def test_counts_novel_operation_uptake(self) -> None:
        summary = summarize([_outcome()])
        self.assertEqual(summary["novel_operation"], 1)
        self.assertEqual(summary["novel_uptake"], 1)
        self.assertEqual(summary["novel_uptake_rate"], 1.0)

    def test_operation_already_in_base_text_is_not_novel(self) -> None:
        summary = summarize(
            [_outcome(base_text="根拠はある。この設計は前提を分ける。")]
        )
        self.assertEqual(summary["novel_operation"], 0)

    def test_counts_carryover(self) -> None:
        outcome = _outcome()
        outcome.carryover_seen = True
        summary = summarize([outcome])
        self.assertEqual(summary["carryover"], 1)
        self.assertEqual(summary["carryover_rate"], 1.0)
        self.assertEqual(summary["novel_carryover"], 1)
        self.assertEqual(summary["novel_carryover_rate"], 1.0)


class VocabularyTests(unittest.TestCase):
    def test_orders_by_first_appearance_and_dedupes(self) -> None:
        self.assertEqual(
            tag_operations("前提を確認し、つまり結論として決まる。"),
            ["dependency", "conclusion"],
        )

    def test_empty_text(self) -> None:
        self.assertEqual(tag_operations("今日はいい天気だ。"), [])


class SynchronizerTests(unittest.TestCase):
    def test_punctuation_boundary_fires_once(self) -> None:
        sync = Synchronizer()
        sync.start(utterance_id="u1", at_ms=0)
        self.assertTrue(sync.observe("この設計は前提を分ける。", now_ms=800))
        self.assertFalse(sync.observe("この設計は前提を分ける。", now_ms=900))

    def test_announced_operation_is_a_boundary(self) -> None:
        sync = Synchronizer()
        sync.start(utterance_id="u1", at_ms=0)
        self.assertTrue(sync.observe("この設計の結論は", now_ms=500))

    def test_pause_boundary(self) -> None:
        sync = Synchronizer(pause_ms=300)
        sync.start(utterance_id="u1", at_ms=0)
        self.assertFalse(sync.observe("この設計は前提を分ける", now_ms=500))
        self.assertTrue(sync.check_pause(now_ms=900))

    def test_detects_divergence_from_prediction(self) -> None:
        sync = Synchronizer()
        sync.start(utterance_id="u1", at_ms=0)
        text = "この設計は前提を分ける。"
        sync.observe(text, now_ms=800)
        sync.set_prediction(_prediction("だから結論は決まる。", base_length=len(text)))
        sync.observe(text + "逆に考える", now_ms=1_000)
        self.assertTrue(sync.diverged)
        self.assertEqual(sync.prediction_match, 0)

    def test_matching_prefix_reduces_divergence(self) -> None:
        sync = Synchronizer()
        sync.start(utterance_id="u1", at_ms=0)
        text = "この設計は"
        sync.observe(text, now_ms=500)
        sync.set_prediction(_prediction("前提を分ける。", base_length=len(text)))
        sync.observe(text + "前提を", now_ms=1_000)
        self.assertFalse(sync.diverged)
        self.assertEqual(sync.prediction_match, 3)

    def test_longest_common_prefix(self) -> None:
        self.assertEqual(longest_common_prefix("abcdef", "abcxyz"), 3)
        self.assertEqual(longest_common_prefix("", "abc"), 0)


class PredictorTests(unittest.TestCase):
    def test_ngram_predictor_produces_a_span(self) -> None:
        predictor = NgramPredictor.default()
        prediction = predictor.predict(
            text="この設計は前提を分けて考える。",
            context=(),
            now_ms=0,
        )
        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertTrue(prediction.span)
        self.assertGreater(prediction.confidence, 0.0)

    def test_parse_prediction_payload(self) -> None:
        prediction = parse_prediction_payload(
            '{"span": "だから結論は決まる。", "operations": ["conclusion"], "confidence": 0.8}',
            text="この設計は",
            now_ms=12,
            source="test",
        )
        self.assertEqual(prediction.span, "だから結論は決まる。")
        self.assertEqual(prediction.operations, ("conclusion",))
        self.assertEqual(prediction.base_length, 5)

    def test_parse_prediction_deduplicates_operations(self) -> None:
        prediction = parse_prediction_payload(
            '{"span": "根拠を示す。", "operations": ["ground", "ground", "conclusion"]}',
            text="この設計は",
            now_ms=0,
            source="test",
        )
        self.assertEqual(prediction.operations, ("ground", "conclusion"))

    def test_parse_prediction_rejects_missing_span(self) -> None:
        with self.assertRaises(ValueError):
            parse_prediction_payload(
                '{"confidence": 0.8}', text="x", now_ms=0, source="t"
            )


class _SlowPredictor:
    label = "slow"

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay

    def predict(self, *, text, context, now_ms):
        time.sleep(self.delay)
        return _prediction("次の結論", ("conclusion",), base_length=len(text))


class AsyncPredictorTests(unittest.TestCase):
    def test_uses_fallback_until_background_result_is_ready(self) -> None:
        predictor = AsyncPredictor(
            _SlowPredictor(),
            fallback=ScriptedPredictor({"設計": ("速い", 0.9)}),
            debounce_ms=0,
        )
        try:
            predictor.submit(text="設計", context=(), now_ms=0)
            immediate = predictor.take(text="設計", context=(), now_ms=0)
            self.assertEqual(immediate.source, "scripted")
            time.sleep(0.1)
            completed = predictor.take(text="設計は", context=(), now_ms=100)
            self.assertEqual(completed.source, "test")
            self.assertGreater(predictor.last_latency_ms, 0)
            self.assertTrue(predictor.last_precomputed)
            predictor.reset()
            after_reset = predictor.take(text="設計は", context=(), now_ms=200)
            self.assertEqual(after_reset.source, "scripted")
        finally:
            predictor.close()

    def test_debounces_rapid_submissions(self) -> None:
        predictor = AsyncPredictor(_SlowPredictor(delay=0.01), debounce_ms=200)
        try:
            predictor.submit(text="設", context=(), now_ms=0)
            predictor.submit(text="設計", context=(), now_ms=50)
            self.assertEqual(predictor._submitted_text, "設")
        finally:
            predictor.close()


class _SlowDecisionModel:
    label = "slow-decision"

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.inner = RuleDecisionModel(CanonicalAssigner(), context_fit=False)

    def decide(self, state):
        time.sleep(self.delay)
        return self.inner.decide(state)


class AsyncDecisionModelTests(unittest.TestCase):
    def test_uses_fallback_until_background_decision_is_ready(self) -> None:
        model = AsyncDecisionModel(
            _SlowDecisionModel(),
            fallback=RuleDecisionModel(CanonicalAssigner(), context_fit=False),
        )
        try:
            model.submit(_state(now_ms=400))
            intervention, precomputed, latency_ms, at_ms = model.take(
                _state(now_ms=500, text="この設計は前提を分けて考える。次に")
            )
            self.assertIsNotNone(intervention)
            self.assertFalse(precomputed)
            self.assertEqual(latency_ms, 0)
            self.assertEqual(at_ms, 500)
            self.assertEqual(model.last_source, "fallback")
            time.sleep(0.1)
            intervention, precomputed, latency_ms, at_ms = model.take(
                _state(now_ms=700, text="この設計は前提を分けて考える。次に")
            )
            self.assertIsNotNone(intervention)
            self.assertTrue(precomputed)
            self.assertEqual(at_ms, 400)
            self.assertGreater(latency_ms, 0)
            self.assertEqual(model.last_source, "precomputed")
            _, reused, _, _ = model.take(
                _state(now_ms=800, text="この設計は前提を分けて考える。次に")
            )
            self.assertFalse(reused)
        finally:
            model.close()

    def test_reset_discards_precomputed_decisions(self) -> None:
        model = AsyncDecisionModel(_SlowDecisionModel(delay=0.01))
        try:
            model.submit(_state(now_ms=400))
            time.sleep(0.05)
            model.reset()
            _, precomputed, _, _ = model.take(
                _state(now_ms=500, text="この設計は前提を分けて考える。次に")
            )
            self.assertFalse(precomputed)
        finally:
            model.close()

    def test_ignores_results_from_another_utterance(self) -> None:
        model = AsyncDecisionModel(_SlowDecisionModel(delay=0.01))
        try:
            model.submit(_state(now_ms=400))
            time.sleep(0.05)
            _, precomputed, _, _ = model.take(_state(now_ms=500, utterance_id="u2"))
            self.assertFalse(precomputed)
        finally:
            model.close()

    def test_ignores_stale_results(self) -> None:
        model = AsyncDecisionModel(_SlowDecisionModel(delay=0.01), max_age_ms=100)
        try:
            model.submit(_state(now_ms=400))
            time.sleep(0.05)
            _, precomputed, _, _ = model.take(
                _state(now_ms=900, text="この設計は前提を分けて考える。次に")
            )
            self.assertFalse(precomputed)
        finally:
            model.close()


class LocalLLMPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = json.dumps({"data": [{"id": "test-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                assert request["model"] == "test-model"
                system = request["messages"][0]["content"]
                if "意思決定モデル" in system:
                    content = '{"action": "CUE", "operation": "ground", "cue": "根拠は？", "confidence": 0.8, "reason": "test"}'
                else:
                    content = '{"span": "根拠を示す。", "operations": ["ground"], "confidence": 0.8}'
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

    def test_resolves_model(self) -> None:
        self.assertEqual(resolve_model(self.base_url, timeout=2), "test-model")

    def test_choose_model_normalizes_separators(self) -> None:
        identifiers = ["qwen2.5-7b-instruct", "gemma3:4b-it-qat"]
        self.assertEqual(
            choose_model(identifiers, ["gemma3-4b-it-qat"]), "gemma3:4b-it-qat"
        )
        self.assertEqual(choose_model(identifiers, []), "qwen2.5-7b-instruct")

    def test_decides_with_local_model(self) -> None:
        model = LocalLLMDecisionModel(
            base_url=self.base_url,
            model="test-model",
            assigner=CanonicalAssigner(),
            timeout=2,
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.operation, "ground")
        self.assertEqual(intervention.requested_operation, "ground")
        self.assertEqual(intervention.cue, "根拠は？")

    def test_local_decision_rejects_covered_operation(self) -> None:
        model = LocalLLMDecisionModel(model="test-model")
        self.assertIsNone(
            model._parse(
                '{"action": "CUE", "operation": "conclusion", "cue": "結論は？", "confidence": 0.9}',
                _state(
                    prediction=_prediction(
                        "だから結論は決まる。", operations=("conclusion",)
                    ),
                    clause_operations=frozenset({"conclusion"}),
                ),
            )
        )

    def test_low_confidence_local_decision_is_silent(self) -> None:
        model = LocalLLMDecisionModel(model="test-model", min_confidence=0.5)
        self.assertIsNone(
            model._parse(
                '{"action": "CUE", "operation": "ground", "cue": "根拠は？", "confidence": 0.1}',
                _state(),
            )
        )

    def test_local_decision_without_assigner_uses_model_cue(self) -> None:
        model = LocalLLMDecisionModel(
            base_url=self.base_url, model="test-model", timeout=2
        )
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.cue, "根拠は？")
        self.assertEqual(intervention.assignment, "local-llm")

    def test_predicts_with_local_model(self) -> None:
        predictor = LocalLLMPredictor(
            base_url=self.base_url, model="test-model", timeout=2
        )
        prediction = predictor.predict(
            text="この設計は", context=("前の発話。",), now_ms=12
        )
        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.span, "根拠を示す。")
        self.assertEqual(prediction.operations, ("ground",))
        self.assertEqual(prediction.base_length, 5)


class DecisionModelTests(unittest.TestCase):
    def test_cues_missing_operation_before_next_clause(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.operation, "ground")
        self.assertEqual(intervention.cue, "根拠は？")
        self.assertEqual(intervention.scheduled_at_ms, 1_000)

    def test_fatigue_rotates_away_from_repeats(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=True)

        def _state() -> SyncState:
            prediction = Prediction(
                span="だから結論は決まる。",
                operations=("conclusion",),
                confidence=0.9,
                source="test",
                created_at_ms=800,
                base_length=len("この設計は前提を分けて考える。"),
            )
            return SyncState(
                utterance_id="u1",
                text="この設計は前提を分けて考える。",
                context=(),
                now_ms=800,
                started_at_ms=0,
                chars_per_second=4.0,
                at_boundary=True,
                prediction=prediction,
                prediction_match=6,
                diverged=False,
                clause_operations=frozenset({"dependency"}),
                recent_operations=frozenset(),
            )

        picks = [model.decide(_state()).requested_operation for _ in range(6)]
        self.assertEqual(picks[0], "concretize")
        self.assertNotEqual(picks[1], "concretize")
        self.assertLessEqual(picks.count("concretize"), 2)
        self.assertGreaterEqual(len(set(picks)), 3)

    def test_no_fatigue_when_disabled(self) -> None:
        model = RuleDecisionModel(
            CanonicalAssigner(), context_fit=True, fatigue_penalty=0
        )
        state = _state(
            prediction=_prediction(
                "だから結論は決まる。", operations=("conclusion",)
            ),
            clause_operations=frozenset({"dependency"}),
        )
        first = model.decide(state)
        second = model.decide(state)
        assert first is not None
        assert second is not None
        self.assertEqual(first.requested_operation, second.requested_operation)

    def test_rejects_low_confidence(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), min_confidence=0.95)
        state = _state(prediction=_prediction("だから結論は決まる。", confidence=0.5))
        self.assertIsNone(model.decide(state))

    def test_silent_when_all_operations_are_covered(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        state = _state(
            prediction=_prediction(
                "根拠と範囲と依存を確認する。",
                operations=("ground", "boundary", "dependency"),
            ),
            clause_operations=frozenset(
                {"comparison", "invert", "concretize", "mechanism", "conclusion"}
            ),
        )
        self.assertIsNone(model.decide(state))

    def test_skips_recently_delivered_operation(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        state = _state(recent_operations=frozenset({"ground"}))
        intervention = model.decide(state)
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.operation, "boundary")

    def test_silent_when_not_at_boundary(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        self.assertIsNone(model.decide(_state(at_boundary=False)))

    def test_contentless_prediction_is_silent(self) -> None:
        self.assertIsNone(
            RuleDecisionModel(CanonicalAssigner(), context_fit=False).decide(
                _state(prediction=_prediction("。", base_length=15))
            )
        )
        self.assertIsNone(
            LocalLLMDecisionModel(model="test-model").decide(
                _state(prediction=_prediction("。", base_length=15))
            )
        )

    def test_insertion_delay_schedules_later(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), insertion_delay_ms=400)
        intervention = model.decide(_state())
        assert intervention is not None
        self.assertEqual(intervention.scheduled_at_ms, 1_400)
        self.assertEqual(intervention.insertion_delay_ms, 400)

    def test_alternate_assigner_rotates(self) -> None:
        assigner = AlternateAssigner()
        first = assigner.assign("ground")
        second = assigner.assign("ground")
        self.assertEqual(first, "根拠は？")
        self.assertEqual(second, "根拠")

    def test_fixed_and_random_assigners(self) -> None:
        self.assertEqual(FixedAssigner("逆は？").assign("ground"), "逆は？")
        first = RandomAssigner(seed=3).assign("ground")
        second = RandomAssigner(seed=3).assign("ground")
        self.assertEqual(first, second)
        self.assertIn(first, ("根拠は？", "根拠", "答えは？"))
        self.assertTrue(isinstance(random.Random(3).choice([1]), int))

    def test_unmapped_cue_is_logged_as_unmapped(self) -> None:
        model = RuleDecisionModel(FixedAssigner("そうですね"), context_fit=False)
        intervention = model.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.operation, "unmapped")
        self.assertEqual(intervention.requested_operation, "ground")


if __name__ == "__main__":
    unittest.main()


class RandomOperationTests(unittest.TestCase):
    """The exploration arm must randomize *what* without changing *when*."""

    def _wrapped(self, seed: int = 3) -> RandomOperationDecisionModel:
        return RandomOperationDecisionModel(
            RuleDecisionModel(CanonicalAssigner(), context_fit=False),
            CanonicalAssigner(),
            seed=seed,
        )

    def test_draws_multiple_operations_and_replays_with_seed(self) -> None:
        model = self._wrapped()
        first = [model.decide(_state()) for _ in range(100)]
        replay = self._wrapped()
        second = [replay.decide(_state()) for _ in range(100)]
        first_ids = [item.intervention_id for item in first if item is not None]
        second_ids = [item.intervention_id for item in second if item is not None]
        self.assertEqual(len(first_ids), 100)
        self.assertGreater(len(set(first_ids)), 1)
        self.assertEqual(first_ids, second_ids)

    def test_keeps_policy_request_and_changes_operation(self) -> None:
        base = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        expected = base.decide(_state())
        assert expected is not None
        draws = [self._wrapped(seed=seed).decide(_state()) for seed in range(12)]
        for draw in draws:
            assert draw is not None
            self.assertEqual(draw.requested_operation, expected.requested_operation)
            self.assertIn("random-operation(seed=", draw.assignment)
            self.assertEqual(draw.operation, operation_for_cue(draw.cue) or "unmapped")
        self.assertGreater(len({draw.operation for draw in draws}), 1)

    def test_stays_silent_when_the_policy_is_silent(self) -> None:
        model = RandomOperationDecisionModel(
            RuleDecisionModel(CanonicalAssigner(), min_confidence=0.99),
            CanonicalAssigner(),
            seed=0,
        )
        self.assertIsNone(
            model.decide(_state(prediction=_prediction("次は", confidence=0.4)))
        )

    def test_stays_silent_without_candidates(self) -> None:
        covered = _state(
            prediction=_prediction("すべて確認する。", operations=tuple(OPERATIONS)),
            clause_operations=frozenset(OPERATIONS),
        )
        base = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        sample = base.decide(_state())
        assert sample is not None

        class AlwaysCue:
            label = "always"

            def decide(self, state: SyncState) -> Intervention:
                del state
                return sample

        model = RandomOperationDecisionModel(AlwaysCue(), CanonicalAssigner(), seed=0)
        self.assertIsNone(model.decide(covered))

    def test_backward_compatible_assigners_still_build(self) -> None:
        self.assertEqual(build_assigner("canonical").label, "canonical")
        self.assertEqual(build_assigner("alternate").label, "alternate")
        self.assertEqual(build_assigner("random", seed=5).label, "randomized(5)")
        self.assertEqual(build_assigner("fixed", cue="なぜ？").label, "fixed(なぜ？)")
        with self.assertRaises(ValueError):
            build_assigner("unknown")


class ContextFitTests(unittest.TestCase):
    """The rule policy should cue the move the clause needs, not a global one."""

    def test_ranks_candidates_toward_the_clause_move(self) -> None:
        state = _state(
            prediction=_prediction(
                "だから結論は決まる。",
                operations=("conclusion",),
            ),
            clause_operations=frozenset({"dependency"}),
        )
        model = RuleDecisionModel(CanonicalAssigner())
        intervention = model.decide(state)
        assert intervention is not None
        # clause dependency(1.5 anchor) beats prediction conclusion(1.0): concretize
        self.assertEqual(intervention.requested_operation, "concretize")
        self.assertIn("context-fit", intervention.reason)

    def test_context_fit_prefers_prediction_adjacency_when_clause_silent(self) -> None:
        state = _state(
            prediction=_prediction(
                "だから結論は決まる。",
                operations=("conclusion",),
            ),
        )
        model = RuleDecisionModel(CanonicalAssigner())
        intervention = model.decide(state)
        assert intervention is not None
        # conclusion's adjacency prior starts with invert
        self.assertEqual(intervention.requested_operation, "invert")

    def test_context_fit_can_be_disabled(self) -> None:
        model = RuleDecisionModel(CanonicalAssigner(), context_fit=False)
        intervention = model.decide(_state())
        assert intervention is not None
        self.assertEqual(intervention.requested_operation, "ground")

    def test_no_tagged_signal_silences_the_rule_policy(self) -> None:
        state = _state(
            prediction=_prediction("さらに", operations=()),
            clause_operations=frozenset(),
            recent_operations=frozenset(),
        )
        model = RuleDecisionModel(CanonicalAssigner())
        self.assertIsNone(model.decide(state))

    def test_ranking_falls_back_to_priority_without_signal(self) -> None:
        from cueloop.sync.policy_model import PolicyObservation
        from cueloop.sync.vocabulary import PRIORITY

        state = _state(
            prediction=_prediction("次は", operations=()),
            clause_operations=frozenset(),
            recent_operations=frozenset(),
        )
        observation = PolicyObservation.from_state(state)
        self.assertEqual(observation.ranked_missing_operations(), PRIORITY)
