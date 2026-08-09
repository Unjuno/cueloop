from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Lock

from .core import Analysis, Analyzer, SessionContext, safe_analysis


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def _copy_context(context: SessionContext | None) -> SessionContext | None:
    if context is None:
        return None
    copy = SessionContext(
        max_turns=context.max_turns,
        max_characters=context.max_characters,
    )
    for turn in context.snapshot():
        copy.add(turn)
    return copy


@dataclass(frozen=True)
class PrefetchConfig:
    minimum_characters: int = 24
    boundary_wait_ms: int = 60

    def __post_init__(self) -> None:
        if self.minimum_characters < 1:
            raise ValueError("minimum_characters must be positive")
        if self.boundary_wait_ms < 0:
            raise ValueError("boundary_wait_ms must be non-negative")


class PrefetchingAnalyzer:
    """Speculatively analyzes partial speech while the user is still talking.

    The cache is exact-text keyed: a candidate generated for one partial is
    never reused for a different final transcript. At the turn boundary we wait
    only a very small bounded interval. If the candidate is not ready, CUELOOP
    fails closed to SILENCE instead of making the user wait for a late model.
    """

    def __init__(self, delegate: Analyzer, config: PrefetchConfig | None = None) -> None:
        self._delegate = delegate
        self.config = config or PrefetchConfig()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cueloop-prefetch")
        self._lock = Lock()
        self._key: str | None = None
        self._future: Future[Analysis] | None = None

    def prefetch(self, text: str, context: SessionContext | None = None) -> None:
        key = _normalize(text)
        if len(key) < self.config.minimum_characters:
            return

        with self._lock:
            if key == self._key and self._future is not None:
                return
            if self._future is not None and not self._future.running():
                self._future.cancel()
            self._key = key
            context_copy = _copy_context(context)
            self._future = self._executor.submit(self._delegate.analyze, key, context_copy)

    def analyze(self, text: str, context: SessionContext | None = None) -> Analysis:
        key = _normalize(text)
        with self._lock:
            future = self._future if key == self._key else None

        if future is None:
            # Offline/text paths are allowed to use the delegate normally.
            return self._delegate.analyze(key, context)

        try:
            return future.result(timeout=self.config.boundary_wait_ms / 1000)
        except TimeoutError:
            return safe_analysis("speculative analyzer not ready at turn boundary")
        except Exception as exc:
            return safe_analysis(f"speculative analyzer failure: {type(exc).__name__}")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
