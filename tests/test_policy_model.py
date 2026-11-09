from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cueloop.sync.models import Prediction, SyncState
from cueloop.sync.policy import CanonicalAssigner, LinearDecisionModel
from cueloop.sync.policy_model import (
    FEATURE_NAMES,
    LinearPolicyModel,
    PolicyObservation,
    TrainingSample,
    evaluate_policy,
    feature_vector,
    namespace_intervention_ids,
    samples_from_records,
    split_samples,
    train_linear_policy,
)


def _state(text: str = "この設計は前提を分けて考える。") -> SyncState:
    prediction = Prediction(
        span="だから結論は決まる。",
        operations=("conclusion",),
        confidence=0.9,
        source="test",
        created_at_ms=800,
        base_length=len(text),
    )
    return SyncState(
        utterance_id="u1",
        text=text,
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


def _intervention(identifier: str, operation: str) -> dict[str, object]:
    return {
        "kind": "intervention",
        "intervention_id": identifier,
        "operation": operation,
        "text": "この設計は前提を分けて考える。",
        "text_length": 15,
        "context_turns": 1,
        "chars_per_second": 4.0,
        "elapsed_ms": 800,
        "prediction_confidence": 0.9,
        "prediction_length": 10,
        "prediction_match": 6,
        "prediction_operations": ["conclusion"],
        "clause_operations": ["dependency"],
        "recent_operations": [],
        "at_boundary": True,
        "diverged": False,
    }


class PolicyFeatureTests(unittest.TestCase):
    def test_vector_matches_schema_and_marks_operation(self) -> None:
        observation = PolicyObservation.from_state(_state())
        values = feature_vector(observation, "ground")
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(values[FEATURE_NAMES.index("operation:ground")], 1.0)
        self.assertIn("ground", observation.missing_operations())

    def test_log_observation_round_trip(self) -> None:
        observation = PolicyObservation.from_log(_intervention("a", "ground"))
        self.assertEqual(observation.prediction_operations, {"conclusion"})
        self.assertEqual(observation.clause_operations, {"dependency"})
        self.assertEqual(observation.missing_operations()[0], "ground")

    def test_model_save_and_load(self) -> None:
        model = LinearPolicyModel(weights=[0.0] * len(FEATURE_NAMES), threshold=0.4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            model.save(path)
            loaded = LinearPolicyModel.load(path)
        self.assertEqual(loaded.threshold, 0.4)
        self.assertEqual(len(loaded.weights), len(FEATURE_NAMES))


class LinearDecisionTests(unittest.TestCase):
    def _model(self, ground_weight: float) -> LinearPolicyModel:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("operation:ground")] = ground_weight
        return LinearPolicyModel(weights=weights, threshold=0.5)

    def test_chooses_highest_scoring_missing_operation(self) -> None:
        model = self._model(6.0)
        decision = LinearDecisionModel(model, CanonicalAssigner())
        intervention = decision.decide(_state())
        self.assertIsNotNone(intervention)
        assert intervention is not None
        self.assertEqual(intervention.operation, "ground")

    def test_stays_silent_below_threshold(self) -> None:
        model = LinearPolicyModel(weights=[-6.0] * len(FEATURE_NAMES), threshold=0.5)
        decision = LinearDecisionModel(model, CanonicalAssigner())
        self.assertIsNone(decision.decide(_state()))

    def test_neutral_score_is_not_evidence(self) -> None:
        model = LinearPolicyModel(weights=[0.0] * len(FEATURE_NAMES), threshold=0.5)
        decision = LinearDecisionModel(model, CanonicalAssigner())
        self.assertIsNone(decision.decide(_state()))


class PolicyTrainingTests(unittest.TestCase):
    def test_training_learns_operation_preference(self) -> None:
        samples: list[TrainingSample] = []
        for index in range(4):
            samples.append(
                TrainingSample(
                    observation=PolicyObservation.from_state(_state()),
                    operation="ground",
                    reward=1.0 if index < 3 else 0.0,
                )
            )
            samples.append(
                TrainingSample(
                    observation=PolicyObservation.from_state(_state()),
                    operation="dependency",
                    reward=1.0 if index == 0 else 0.0,
                )
            )
        model = train_linear_policy(samples, epochs=300)
        self.assertGreaterEqual(model.threshold, 0.5)
        observation = PolicyObservation.from_state(_state())
        self.assertGreater(
            model.score(observation, "ground"),
            model.score(observation, "dependency"),
        )

    def test_exploration_records_train_on_the_delivered_operation(self) -> None:
        intervention = _intervention("a", "boundary")
        intervention["requested_operation"] = "ground"
        intervention["assignment"] = "random-operation(seed=0;wording=canonical)"
        records = [
            intervention,
            {"kind": "delivery", "intervention_id": "a", "status": "DELIVERED"},
            {"kind": "carryover", "intervention_id": "a", "carryover_seen": True},
        ]
        samples = samples_from_records(records)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].operation, "boundary")
        self.assertEqual(samples[0].reward, 1.0)

    def test_samples_join_delivery_and_carryover(self) -> None:
        records = [
            _intervention("a", "ground"),
            {"kind": "delivery", "intervention_id": "a", "status": "DELIVERED"},
            {
                "kind": "outcome",
                "intervention_id": "a",
                "target_operation_seen": True,
            },
            {
                "kind": "carryover",
                "intervention_id": "a",
                "carryover_seen": True,
            },
            _intervention("b", "dependency"),
            {"kind": "delivery", "intervention_id": "b", "status": "CANCELLED"},
        ]
        samples = samples_from_records(records)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].operation, "ground")
        self.assertEqual(samples[0].reward, 1.0)


class PolicyEvaluationTests(unittest.TestCase):
    def _samples(self) -> list[TrainingSample]:
        observation = PolicyObservation.from_state(_state())
        samples: list[TrainingSample] = []
        for index in range(8):
            samples.append(
                TrainingSample(
                    observation=observation,
                    operation="ground" if index % 2 == 0 else "dependency",
                    reward=1.0 if index % 2 == 0 else 0.0,
                )
            )
        return samples

    def test_evaluate_policy_reports_confusion_metrics(self) -> None:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("operation:ground")] = 6.0
        weights[FEATURE_NAMES.index("operation:dependency")] = -6.0
        model = LinearPolicyModel(weights=weights, threshold=0.5)
        metrics = evaluate_policy(model, self._samples())
        self.assertEqual(metrics["samples"], 8)
        self.assertEqual(metrics["positives"], 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertLess(metrics["brier"], 0.1)

    def test_split_samples_is_stratified_and_deterministic(self) -> None:
        samples = self._samples()
        train, test = split_samples(samples, eval_fraction=0.25, seed=3)
        self.assertEqual(len(train) + len(test), len(samples))
        self.assertTrue(test)
        self.assertEqual(train, split_samples(samples, eval_fraction=0.25, seed=3)[0])
        self.assertEqual(
            sum(1 for sample in test if sample.reward >= 0.5),
            sum(1 for sample in samples if sample.reward >= 0.5)
            * len(test)
            // len(samples),
        )

    def test_split_samples_disabled_keeps_everything(self) -> None:
        samples = self._samples()
        train, test = split_samples(samples, eval_fraction=0.0)
        self.assertEqual(len(train), len(samples))
        self.assertEqual(test, [])


if __name__ == "__main__":
    unittest.main()


class NamespacedLogsTests(unittest.TestCase):
    def test_paired_cue_and_silence_logs_do_not_clobber_each_other(self) -> None:
        shared = "cue-abc123"
        records = [
            {
                "kind": "intervention",
                "intervention_id": shared,
                "operation": "ground",
                "requested_operation": "invert",
                "assignment": "random-operation(seed=0;wording=canonical)",
                "at_ms": 900,
                "boundary_at_ms": 900,
                "at_boundary": True,
                "diverged": False,
                "prediction_confidence": 0.8,
                "confidence": 0.8,
                "text_length": 10,
                "context_turns": 0,
                "chars_per_second": 12.0,
                "elapsed_ms": 900,
                "prediction_length": 5,
                "prediction_match": 0,
                "prediction_operations": [],
                "clause_operations": ["ground"],
                "recent_operations": [],
                "prediction_span": "次は",
                "text": "本文",
            },
            {"kind": "delivery", "intervention_id": shared, "status": "DELIVERED"},
            {"kind": "carryover", "intervention_id": shared, "carryover_seen": True},
        ]
        silence = [
            dict(records[0]),
            {"kind": "delivery", "intervention_id": shared, "status": "SILENT"},
            {"kind": "carryover", "intervention_id": shared, "carryover_seen": False},
        ]
        merged = list(namespace_intervention_ids(records, "cue.jsonl")) + list(
            namespace_intervention_ids(silence, "silence.jsonl")
        )
        samples = samples_from_records(iter(merged))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].reward, 1.0)


class TrainPolicyExplorationTests(unittest.TestCase):
    def test_exploration_logs_train_and_unknown_assignment_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[1]

        def intervention(
            intervention_id: str, assignment: str, at_ms: int = 900
        ) -> dict:
            return {
                "kind": "intervention",
                "condition": "CUE",
                "decision_model": "prediction-gap(assigner=canonical)+random-operation(seed=0)",
                "predictor": "ngram(3)",
                "at_ms": at_ms,
                "intervention_id": intervention_id,
                "operation": "ground",
                "requested_operation": "invert",
                "assignment": assignment,
                "boundary_at_ms": at_ms,
                "at_boundary": True,
                "diverged": False,
                "prediction_confidence": 0.8,
                "confidence": 0.8,
                "text_length": 10,
                "context_turns": 0,
                "chars_per_second": 12.0,
                "elapsed_ms": at_ms,
                "prediction_length": 5,
                "prediction_match": 0,
                "prediction_operations": [],
                "clause_operations": ["ground"],
                "recent_operations": [],
                "prediction_span": "次は",
                "text": "本文",
            }

        records = []
        for index, carryover in enumerate([True, False, True, False]):
            intervention_id = f"cue-explore-{index}"
            assignment = (
                "random-operation(seed=0;wording=canonical)"
                if index % 2 == 0
                else "mystery-assignment(v9)"
            )
            records.append(intervention(intervention_id, assignment, 900 + index))
            records.append(
                {
                    "kind": "delivery",
                    "intervention_id": intervention_id,
                    "status": "DELIVERED",
                }
            )
            records.append(
                {
                    "kind": "carryover",
                    "intervention_id": intervention_id,
                    "carryover_seen": carryover,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "explore.jsonl"
            log.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output = Path(directory) / "policy.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_policy.py",
                    str(log),
                    "--output",
                    str(output),
                    "--min-samples",
                    "4",
                    "--epochs",
                    "20",
                    "--eval-fraction",
                    "0",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("mystery-assignment(v9)", completed.stderr)
            summary = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(summary["samples"], 4)
            self.assertIn("mystery-assignment(v9)", summary["unknown_assignments"])
            self.assertEqual(
                summary["assignments"]["random-operation(seed=0;wording=canonical)"], 2
            )
