from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile
import unittest

from cueloop.core import Action, Analysis, ControllerConfig, HeuristicAnalyzer, InterventionController, State
from cueloop.personalization import AdaptiveController, FeedbackRating, PersonalizationProfile


class PersonalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = HeuristicAnalyzer().analyze("えっと、その、なんだろう、性能が良い")

    def test_disruptive_feedback_can_only_suppress(self) -> None:
        profile = PersonalizationProfile()
        controller = AdaptiveController(InterventionController(ControllerConfig(cooldown_turns=0)), profile)
        for _ in range(2):
            decision = controller.decide(self.analysis)
            self.assertEqual(decision.action, Action.QUESTION)
            profile.record_feedback(decision.action, decision.cue, FeedbackRating.DISRUPTIVE)
        suppressed = controller.decide(self.analysis)
        self.assertEqual(suppressed.action, Action.SILENCE)
        self.assertIn("feedback", suppressed.reason)

    def test_helpful_intervention_fades_after_repetition(self) -> None:
        profile = PersonalizationProfile()
        controller = AdaptiveController(InterventionController(ControllerConfig(cooldown_turns=0)), profile)
        for _ in range(3):
            decision = controller.decide(self.analysis)
            self.assertEqual(decision.action, Action.QUESTION)
            profile.record_feedback(decision.action, decision.cue, FeedbackRating.HELPFUL)
        self.assertEqual(controller.decide(self.analysis).action, Action.QUESTION)
        faded = controller.decide(self.analysis)
        self.assertEqual(faded.action, Action.SILENCE)
        self.assertIn("fading", faded.reason)

    def test_profile_round_trip_contains_no_transcript(self) -> None:
        profile = PersonalizationProfile()
        stats = profile.for_intervention(Action.BACKCHANNEL, None)
        stats.delivered = 1
        stats.opportunities = 1
        profile.record_feedback(Action.BACKCHANNEL, None, FeedbackRating.NEUTRAL)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            loaded = PersonalizationProfile.load(path)
            self.assertEqual(loaded.for_intervention(Action.BACKCHANNEL, None).neutral, 1)
            self.assertNotIn("transcript", path.read_text(encoding="utf-8"))

    def test_recurring_states_store_hashes_not_evidence_text(self) -> None:
        profile = PersonalizationProfile()
        controller = AdaptiveController(InterventionController(ControllerConfig(cooldown_turns=0)), profile)
        controller.decide(self.analysis)
        controller.decide(self.analysis)
        self.assertEqual(profile.recurring_states(), {"STRUCTURE_GAP": 2})
        serialized = str(profile.to_dict())
        self.assertNotIn(self.analysis.evidence[0].description, serialized)
        self.assertEqual(len(profile.state_observations[0].evidence_hash), 64)

    def test_expired_state_observations_are_pruned(self) -> None:
        profile = PersonalizationProfile()
        old = datetime.now(timezone.utc) - timedelta(days=2)
        profile.observe_state(self.analysis, observed_at=old, retention_days=1)
        self.assertEqual(profile.state_observations, [])

    def test_corrupt_review_counts_are_rejected(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps({
                    "schema_version": 3,
                    "stats": {"QUESTION:x": {"delivered": 1, "opportunities": 1, "helpful": 2, "neutral": 0, "disruptive": 0}},
                    "state_observations": [],
                    "retained_structures": [],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "reviewed interventions"):
                PersonalizationProfile.load(path)

    def test_adaptive_suppression_does_not_consume_base_cooldown(self) -> None:
        profile = PersonalizationProfile()
        rejected = profile.for_intervention(Action.QUESTION, "何に依存？")
        rejected.opportunities = rejected.delivered = rejected.disruptive = 2
        controller = AdaptiveController(
            InterventionController(ControllerConfig(cooldown_turns=2)), profile
        )
        first = controller.decide(self.analysis)
        self.assertEqual(first.action, Action.SILENCE)
        alternative = Analysis(
            State.STRUCTURE_GAP, 0.9, ("gap",), Action.QUESTION, "比較軸は？"
        )
        second = controller.decide(alternative)
        self.assertEqual(second.action, Action.QUESTION)


if __name__ == "__main__":
    unittest.main()
