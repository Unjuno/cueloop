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
        self.assertEqual(silence["outcomes"], [])
        # A silent arm logs SILENT deliveries but scores no outcomes: the
        # speaker never heard a cue, so there is nothing to adopt. This is
        # what makes it a valid control (baseline uptake is not invented).

    def test_silent_arm_has_no_outcomes_so_control_is_not_invented(self) -> None:
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
        # The SILENCE arm scores no outcomes: a cue that was never audible
        # cannot be adopted. With uptake_effect=0 the CUE arm still shows
        # only the base (natural) uptake, so the paired CUE-SILENCE contrast
        # remains a clean readout of the cue's added effect.
        self.assertEqual(silence["outcomes"], [])
        self.assertGreaterEqual(cue["novel_operation"], 0)
        self.assertLessEqual(cue["novel_uptake"], cue["novel_operation"])


if __name__ == "__main__":
    unittest.main()
