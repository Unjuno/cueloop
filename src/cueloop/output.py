from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import subprocess
from threading import Lock
from typing import Protocol

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

    def cancel(self) -> None:
        return


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
    """Sends one intervention JSON object to an external playback/TTS command.

    The active subprocess is cancellable so user speech can immediately win a
    barge-in race against an already-playing intervention.
    """

    def __init__(self, config: CommandSinkConfig) -> None:
        self._config = config
        self._lock = Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancelled_pids: set[int] = set()

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

        process = subprocess.Popen(
            self._config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process

        try:
            _, stderr = process.communicate(
                input=json.dumps(payload, ensure_ascii=False),
                timeout=self._config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("output command timed out")
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None

        with self._lock:
            cancelled = process.pid in self._cancelled_pids
            self._cancelled_pids.discard(process.pid)
        if cancelled:
            return
        if process.returncode != 0:
            raise RuntimeError(f"output command failed ({process.returncode}): {stderr.strip()[:500]}")

    def cancel(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._cancelled_pids.add(process.pid)
            process.terminate()


class AsyncInterventionSink:
    """Keeps TTS/playback latency off the streaming event loop.

    A single worker preserves intervention order. `cancel()` terminates current
    playback when supported and drops queued interventions that have not begun.
    """

    def __init__(self, delegate: InterventionSink) -> None:
        self._delegate = delegate
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cueloop-output")
        self._lock = Lock()
        self._futures: list[Future[None]] = []

    def emit(self, decision: Decision) -> None:
        if decision.action is Action.SILENCE:
            return
        future = self._executor.submit(self._delegate.emit, decision)
        with self._lock:
            self._futures.append(future)

    def cancel(self) -> None:
        with self._lock:
            futures = list(self._futures)
            self._futures = [future for future in futures if future.running()]
        for future in futures:
            if not future.running():
                future.cancel()
        cancel = getattr(self._delegate, "cancel", None)
        if callable(cancel):
            cancel()

    def close(self) -> None:
        with self._lock:
            futures = list(self._futures)
            self._futures.clear()
        self._executor.shutdown(wait=True)
        for future in futures:
            if future.cancelled():
                continue
            future.result()
