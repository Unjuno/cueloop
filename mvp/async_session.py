"""Non-blocking decision-model adapter for live transcript revisions.

Inference runs off the audio event thread. Late results never become valid for a
newer revision. Delivery is one-shot and fail-closed.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable

from .core import Decision, Gate, SYSTEM, parse


class AsyncSession:
    def __init__(self, complete: Callable[[str, str], str], gate: Gate | None = None):
        self.complete = complete
        self.gate = gate or Gate()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cueloop-decision")
        self._lock = Lock()
        self._revision = 0
        self._transcript = ""
        self._pending: tuple[Future[str], int, int] | None = None
        self._closed = False

    def update(self, transcript: str, now_ms: int) -> None:
        if now_ms < 0:
            raise ValueError("negative timestamp")
        with self._lock:
            if self._closed:
                raise RuntimeError("session closed")
            text = transcript[-2400:]
            if text == self._transcript:
                return
            self._transcript = text
            self._revision += 1
            previous = self._pending
            self._pending = None
            if previous is not None:
                previous[0].cancel()
            # Inference is submitted while holding the lock but executes outside it.
            self._pending = (
                self._pool.submit(self.complete, SYSTEM, text),
                self._revision,
                now_ms,
            )

    def emit(self, now_ms: int, *, speaking: bool = False) -> Decision:
        with self._lock:
            if self._closed or self._pending is None:
                return Decision("NONE", reason="no candidate")
            future, revision, created_ms = self._pending
            # Boundary processing must not block waiting for model inference.
            if not future.done():
                return Decision("NONE", reason="model still running")
            self._pending = None
            transcript = self._transcript
            current_revision = self._revision
        try:
            decision = parse(future.result())
        except Exception:
            decision = Decision("NONE", reason="model unavailable")
        with self._lock:
            if self._closed or self._revision != current_revision:
                return Decision("NONE", reason="superseded during decision")
            return self.gate.choose(
                decision,
                candidate_revision=revision,
                current_revision=current_revision,
                created_ms=created_ms,
                now_ms=now_ms,
                transcript=transcript,
                speaking=speaking,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._pending is not None:
                self._pending[0].cancel()
                self._pending = None
        self._pool.shutdown(wait=False, cancel_futures=True)
