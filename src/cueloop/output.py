from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Protocol, Sequence

from .core import Action, Decision


class InterventionSink(Protocol):
    def emit(self, decision: Decision) -> None: ...


class ConsoleSink:
    def emit(self, decision: Decision) -> None:
        if decision.action is Action.SILENCE:
            return
        if decision.action is Action.CONTINUE:
            print("[CONTINUE] うん。", flush=True)
            return
        print(f"[CUE] {decision.cue}", flush=True)


@dataclass(frozen=True)
class CommandSinkConfig:
    command: tuple[str, ...]
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("output command must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class CommandInterventionSink:
    """Sends one intervention JSON object to an external playback/TTS command."""

    def __init__(self, config: CommandSinkConfig) -> None:
        self._config = config

    def emit(self, decision: Decision) -> None:
        if decision.action is Action.SILENCE:
            return
        text = "うん。" if decision.action is Action.CONTINUE else (decision.cue or "")
        payload = {
            "action": decision.action.value,
            "text": text,
            "cue": decision.cue,
            "reason": decision.reason,
        }
        process = subprocess.run(
            self._config.command,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=self._config.timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip()[:500]
            raise RuntimeError(f"output command failed ({process.returncode}): {stderr}")
