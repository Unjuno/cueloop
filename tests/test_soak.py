import unittest

from cueloop.soak import run_soak


class SoakTests(unittest.TestCase):
    def test_one_minute_soak_preserves_all_invariants(self) -> None:
        report = run_soak(duration_minutes=1)
        self.assertTrue(report["passed"])
        self.assertEqual(report["turns"], 12)
        self.assertTrue(all(report["invariants"].values()))
        self.assertGreater(report["action_counts"].get("QUESTION", 0), 0)
        self.assertGreater(report["action_counts"].get("BACKCHANNEL", 0), 0)


if __name__ == "__main__":
    unittest.main()
