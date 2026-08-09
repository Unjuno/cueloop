import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from cueloop.asr_command import CommandASRConfig, CommandASREventSource
from cueloop.audio import BoundaryEvent, TranscriptEvent


class CommandASREventSourceTests(unittest.TestCase):
    def test_external_command_can_turn_audio_path_into_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFFfake")
            script = root / "fake_asr.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys
                    from pathlib import Path

                    audio = Path(sys.argv[-1])
                    assert audio.exists()
                    print(json.dumps({
                        "type": "transcript",
                        "kind": "FINAL",
                        "text": "えっと えっと えっと 性能が",
                        "start_ms": 0,
                        "end_ms": 1000,
                    }, ensure_ascii=False))
                    print(json.dumps({
                        "type": "boundary",
                        "kind": "TURN_CANDIDATE",
                        "at_ms": 1500,
                        "silence_ms": 500,
                    }, ensure_ascii=False))
                    """
                ),
                encoding="utf-8",
            )
            source = CommandASREventSource(
                audio,
                CommandASRConfig((sys.executable, str(script))),
            )
            self.assertIsInstance(source.next_event(), TranscriptEvent)
            self.assertIsInstance(source.next_event(), BoundaryEvent)
            self.assertIsNone(source.next_event())

    def test_missing_audio_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            CommandASREventSource(
                Path("definitely-missing.wav"),
                CommandASRConfig((sys.executable, "noop.py")),
            )


if __name__ == "__main__":
    unittest.main()
