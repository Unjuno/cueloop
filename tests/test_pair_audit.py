import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = {"condition": "CUE", "utterances": 1, "delivered": 2, "intervention_id": "x"}


def write_pair(
    root: Path,
    name: str,
    *,
    cue_text: int,
    sil_text: int,
    cue_utt: int = 1,
    sil_utt: int = 1,
) -> Path:
    pair = root / name
    pair.mkdir(parents=True)
    (pair / "cue.json").write_text(
        json.dumps({**REPORT, "utterances": cue_utt}), encoding="utf-8"
    )
    (pair / "silence.json").write_text(
        json.dumps({**REPORT, "condition": "SILENCE", "utterances": sil_utt}),
        encoding="utf-8",
    )

    def clauses(count: int) -> str:
        return "\n".join(
            json.dumps({"clause": f"話{i}", "kind": "prediction"}) for i in range(count)
        ) + ("\n" if count else "")

    (pair / "cue.jsonl").write_text(clauses(cue_text), encoding="utf-8")
    (pair / "silence.jsonl").write_text(clauses(sil_text), encoding="utf-8")
    return pair


class PairAuditTests(unittest.TestCase):
    def _run(self, *roots: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/pair_audit.py", "--quiet", *roots],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_classifies_ok_empty_silent_light_and_nonlive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pair(
                root, "20261101-ok", cue_text=2, sil_text=2, cue_utt=2, sil_utt=2
            )
            write_pair(root, "20261102-empty", cue_text=0, sil_text=0)
            write_pair(root, "20261103-silent", cue_text=0, sil_text=2)
            write_pair(
                root, "20261104-light", cue_text=1, sil_text=1, cue_utt=1, sil_utt=1
            )
            broken = root / "20261105-nonlive"
            broken.mkdir()
            (broken / "cue.json").write_text("not json", encoding="utf-8")
            completed = self._run(str(root / "202611*"))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("EMPTY", completed.stdout)
            self.assertIn("SILENT", completed.stdout)
            self.assertIn("LIGHT", completed.stdout)
            self.assertIn("NON-LIVE", completed.stdout)
            self.assertIn("OK", completed.stdout)
            self.assertIn("scanned=5", completed.stdout)
