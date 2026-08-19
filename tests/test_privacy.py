from pathlib import Path
import tempfile
import unittest

from cueloop.privacy import delete_session_data


class PrivacyTests(unittest.TestCase):
    def test_dry_run_does_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "session.txt").write_text("private", encoding="utf-8")
            result = delete_session_data(root)
            self.assertTrue(data.exists())
            self.assertFalse(result[0].removed)

    def test_confirm_deletes_only_named_direct_children(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data", "logs", "artifacts", "recordings", "transcripts"):
                path = root / name
                path.mkdir()
                (path / "session.txt").write_text("private", encoding="utf-8")
            keep = root / "keep.txt"
            keep.write_text("keep", encoding="utf-8")
            results = delete_session_data(root, confirm=True)
            self.assertTrue(all(result.removed for result in results))
            self.assertTrue(keep.exists())

    def test_rejects_escape_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                delete_session_data(Path(directory), ("../outside",), confirm=True)

    def test_validation_happens_before_any_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            with self.assertRaises(ValueError):
                delete_session_data(root, ("data", "../outside"), confirm=True)
            self.assertTrue(data.exists())

    def test_symlink_target_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = root / "keep"
            keep.mkdir()
            (keep / "important.txt").write_text("keep", encoding="utf-8")
            (root / "data").symlink_to(keep, target_is_directory=True)
            delete_session_data(root, ("data",), confirm=True)
            self.assertFalse((root / "data").exists())
            self.assertTrue((keep / "important.txt").exists())


if __name__ == "__main__":
    unittest.main()
