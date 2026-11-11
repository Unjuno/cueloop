import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DemoReplayTests(unittest.TestCase):
    def test_demo_replay_completes_and_delivers(self) -> None:
        import os

        env = {**os.environ, "CUELOOP_TTS": "0"}
        completed = subprocess.run(
            ["scripts/demo.sh", "replay"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DELIVERED", completed.stdout)
        self.assertIn("report=", completed.stdout)


if __name__ == "__main__":
    unittest.main()
