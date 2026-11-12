"""Event-level simulator: analyzer sees programmed effects end to end."""

import unittest

from scripts.simulate_sessions import run_session


class SimulateSessionsTests(unittest.TestCase):
    def test_cue_and_silence_arms_report_pair_key(self) -> None:
        cue = run_session(
            index=0,
            seed=0,
            condition="CUE",
            uptake_effect=0.4,
            base_uptake=0.15,
            carryover_effect=0.35,
            base_carryover=0.10,
            clauses=4,
            predictor_name="cycle",
            cooldown_ms=250,
            outcome_window_ms=4_000,
        )
        silence = run_session(
            index=0,
            seed=0,
            condition="SILENCE",
            uptake_effect=0.4,
            base_uptake=0.15,
            carryover_effect=0.35,
            base_carryover=0.10,
            clauses=4,
            predictor_name="cycle",
            cooldown_ms=250,
            outcome_window_ms=4_000,
        )
        self.assertEqual(cue["pair_key"], silence["pair_key"])
        self.assertTrue(cue["silent"] is False or cue["silent"] == "false")
        self.assertEqual(cue["condition"], "CUE")
        self.assertEqual(silence["condition"], "SILENCE")

    def test_silence_never_speaks(self) -> None:
        silence = run_session(
            index=0,
            seed=0,
            condition="SILENCE",
            uptake_effect=0.4,
            base_uptake=0.15,
            carryover_effect=0.35,
            base_carryover=0.10,
            clauses=4,
            predictor_name="cycle",
            cooldown_ms=250,
            outcome_window_ms=4_000,
        )
        self.assertEqual(silence["delivered"], 0)
        self.assertEqual(silence["interventions"], len(silence["outcomes"]))

    def test_effect_zero_is_exact_zero(self) -> None:
        cue = run_session(
            index=0,
            seed=0,
            condition="CUE",
            uptake_effect=0.0,
            base_uptake=0.15,
            carryover_effect=0.0,
            base_carryover=0.10,
            clauses=4,
            predictor_name="cycle",
            cooldown_ms=250,
            outcome_window_ms=4_000,
        )
        silence = run_session(
            index=0,
            seed=0,
            condition="SILENCE",
            uptake_effect=0.0,
            base_uptake=0.15,
            carryover_effect=0.0,
            base_carryover=0.10,
            clauses=4,
            predictor_name="cycle",
            cooldown_ms=250,
            outcome_window_ms=4_000,
        )
        acc = lambda report: [
            (o["operation"], o["target_operation_seen"], o["carryover_seen"])
            for o in report["outcomes"]
        ]
        self.assertEqual(acc(cue), acc(silence))


if __name__ == "__main__":
    unittest.main()
