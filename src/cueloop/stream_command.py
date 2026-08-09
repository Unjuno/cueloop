from __future__ import annotations

import json
import subprocess
from typing import Sequence

from .audio import AudioEvent
from .replay import ReplayFormatError, event_from_mapping


class StreamingCommandEventSource:
    """Reads one CUELOOP event JSON object per stdout line from a long-running process."""

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("stream command must not be empty")
        self._process = subprocess.Popen(
            tuple(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdout is None:
            raise RuntimeError("stream command stdout unavailable")

    def next_event(self) -> AudioEvent | None:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if line == "":
                return_code = self._process.poll()
                if return_code is None:
                    continue
                if return_code != 0:
                    stderr = ""
                    if self._process.stderr is not None:
                        stderr = self._process.stderr.read().strip()[:500]
                    raise RuntimeError(f"stream command failed ({return_code}): {stderr}")
                return None
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError("stream command returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise RuntimeError("stream command event must be a JSON object")
            try:
                return event_from_mapping(payload)
            except ReplayFormatError as exc:
                raise RuntimeError(f"stream command returned invalid event: {exc}") from exc

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

    def __enter__(self) -> "StreamingCommandEventSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
