from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence

from .replay import JSONLReplaySource, ReplayFormatError


@dataclass(frozen=True)
class CommandASRConfig:
    command: tuple[str, ...]
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("ASR command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class CommandASREventSource:
    """Turns an audio file into CUELOOP events through an external command.

    Contract:
    - CUELOOP appends the audio path as the final CLI argument.
    - The command writes CUELOOP replay JSONL to stdout.
    - stderr is diagnostic only.
    - Non-zero exit, timeout, or malformed JSONL is a hard offline-run error.

    Keeping ASR outside the core avoids coupling the reasoning system to a
    specific vendor or local model implementation.
    """

    def __init__(self, audio_path: Path, config: CommandASRConfig) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        process = subprocess.run(
            [*config.command, str(audio_path)],
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()[:500]
            raise RuntimeError(f"ASR command failed ({process.returncode}): {stderr}")
        try:
            self._source = JSONLReplaySource.from_lines(process.stdout.splitlines())
        except ReplayFormatError as exc:
            raise RuntimeError(f"ASR command returned invalid event JSONL: {exc}") from exc

    def next_event(self):
        return self._source.next_event()
