import unittest

from cueloop.core import Action, Analysis, ControllerConfig, InterventionController, State


def stuck(cue: str = "反例は？") -> Analysis:
    return Analysis(
        state=State.STUCK,
        confidence=0.9,
        evidence=("semantic loop detected",),
        proposed_action=Action.CUE,
        cue=cue,
    )


class ControllerTimingTests(unittest.TestCase):
    def test_time_cooldown_suppresses_close_intervention(self) -> None:
        controller = InterventionController(
            ControllerConfig(cooldown_turns=0, minimum_interval_ms=1800)
        )
        first = controller.decide(stuck(), now_ms=1000)
        second = controller.decide(stuck(), now_ms=2200)
        third = controller.decide(stuck(), now_ms=3000)

        self.assertEqual(first.action, Action.CUE)
        self.assertEqual(second.action, Action.SILENCE)
        self.assertIn("time cooldown", second.reason)
        self.assertEqual(third.action, Action.CUE)

    def test_timestamp_free_text_mode_keeps_turn_based_behavior(self) -> None:
        controller = InterventionController(
            ControllerConfig(cooldown_turns=0, minimum_interval_ms=10_000)
        )
        self.assertEqual(controller.decide(stuck()).action, Action.CUE)
        self.assertEqual(controller.decide(stuck()).action, Action.CUE)

    def test_negative_timestamp_is_rejected(self) -> None:
        controller = InterventionController()
        with self.assertRaises(ValueError):
            controller.decide(stuck(), now_ms=-1)


if __name__ == "__main__":
    unittest.main()
