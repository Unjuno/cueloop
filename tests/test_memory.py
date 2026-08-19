from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from cueloop.memory import retain_session_structure
from cueloop.personalization import PersonalizationProfile
from cueloop.session import run_integrated_session


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_verified_structure_can_be_explicitly_retained_and_forgotten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (self.root / "experiments/voice-smoke.jsonl").open(encoding="utf-8") as lines:
                outcome = run_integrated_session(lines, root / "sessions", goal="memory", session_id="s1")
            profile_path = root / "profile.json"
            retained = retain_session_structure(outcome.directory, "C1", profile_path)
            profile = PersonalizationProfile.load(profile_path)
            self.assertEqual(profile.retained_structures[0].id, retained.id)
            self.assertEqual(profile.retained_structures[0].source_session_id, "s1")
            self.assertTrue(profile.forget_structure(retained.id))
            self.assertEqual(profile.retained_structures, [])

    def test_retained_structure_expires(self) -> None:
        profile = PersonalizationProfile()
        now = datetime.now(timezone.utc)
        profile.retain_structure(
            kind="claims", text="利用者が選んだ主張", source_session_id="s1",
            retained_at=now, retention_days=1,
        )
        self.assertEqual(profile.prune_retained_structures(now + timedelta(days=2)), 1)
        self.assertEqual(profile.retained_structures, [])


if __name__ == "__main__":
    unittest.main()
