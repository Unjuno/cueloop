"""Predictors that estimate the speaker's next span.

The predictor is the "decision model" input: it says what the speaker is about
to say and how confident it is. It does not decide whether to intervene.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from .models import Prediction
from .vocabulary import OPERATIONS, tag_operations

PREFERRED_PREDICTION_MODELS = (
    "gemma3:4b-it-qat",
    "qwen2.5:7b-instruct",
    "qwen2.5:1.5b-instruct",
    "lfm2.5",
    "gemma3:1b-it-qat",
)

PREFERRED_DECISION_MODELS = (
    "qwen2.5:1.5b-instruct",
    "lfm2.5-1.2b",
    "gemma3:4b-it-qat",
    "qwen2.5:7b-instruct",
    "gemma3:1b-it-qat",
)


class Predictor(Protocol):
    label: str

    def predict(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
    ) -> Prediction | None: ...


class ScriptedPredictor:
    """Deterministic predictor for replay tests and dry runs."""

    def __init__(
        self,
        rules: Mapping[str, tuple[str, float]],
        *,
        fallback: tuple[str, float] | None = None,
    ) -> None:
        if not rules:
            raise ValueError("scripted predictor requires at least one rule")
        self._rules = dict(rules)
        self._fallback = fallback
        self.label = "scripted"

    def predict(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
    ) -> Prediction | None:
        del context
        matched: tuple[str, float] | None = None
        matched_key = ""
        for key, value in self._rules.items():
            if key in text and len(key) >= len(matched_key):
                matched = value
                matched_key = key
        if matched is None:
            matched = self._fallback
        if matched is None:
            return None
        span, confidence = matched
        return Prediction(
            span=span,
            operations=tuple(tag_operations(span)),
            confidence=confidence,
            source=self.label,
            created_at_ms=now_ms,
            base_length=len(text),
        )

    def take(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
        min_base_length: int = 0,
    ) -> Prediction | None:
        del min_base_length
        return self.predict(text=text, context=context, now_ms=now_ms)


DEFAULT_CORPUS = """\
この設計は前提と依存関係を分けて考える必要がある。つまり、全体の一貫性を保つ方法が要る。
この方式は性能が良いと思う。たぶん、スループットは上がる。でも根拠はまだ弱い。
モノリスとして最初に作るのは簡単だ。しかし、障害の影響範囲はどうだろう。
分散データベースでは、二相コミットを使って原子性を維持する。ただ、ネットワーク分断の扱いが難しい。
この要件変更にはまだ確信が持てない。たぶん、影響範囲が広いような気がする。まず、前提を整理する必要がある。
普通はハードコードしない。逆に、この段階では設定ファイルに寄せる方法もあると思う。
マイクロサービス化が前提なので、各チームの依存を切り離す必要がある。範囲はどこまでか決める。
遅延が発生するのはなぜだろう。たぶん共有バッファの仕組みが原因だと思う。根拠を測る。
この機能は必ずしも全ユーザーに配布すべきではない。適用範囲は条件を絞るべきだ。
方針を比較したい。例えば、キャッシュ層を挟む案と挟まない案の違いを具体的に見る。
データが揃っている。つまり結論を出してもいい。なぜなら理由が明確だから。
結論を出す前に、根拠と条件を確認する。逆に考えると別の選択肢が見える。
"""


class AsyncPredictor:
    """Run a slow predictor while the speaker keeps talking.

    The runtime submits the current text on every partial. At a boundary it
    takes the most recent completed prediction if it is still a prefix of the
    current text, otherwise it falls back to a cheap predictor. This keeps a
    local language model off the critical path.
    """

    def __init__(
        self,
        primary: Predictor,
        *,
        fallback: Predictor | None = None,
        debounce_ms: int = 200,
    ) -> None:
        if debounce_ms < 0:
            raise ValueError("debounce_ms must be non-negative")
        self.primary = primary
        self.fallback = fallback
        self.debounce_ms = debounce_ms
        self.label = f"async({primary.label})"
        self.last_latency_ms = 0
        self.last_precomputed = False
        self.last_error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._submitted_text = ""
        self._last_submit_ms = -(10**9)
        self._results: list[tuple[str, Prediction, int]] = []
        self._generation = 0
        self._closed = False

    def warmup(self) -> None:
        warmup = getattr(self.primary, "warmup", None)
        if warmup is not None:
            warmup()

    def submit(self, *, text: str, context: Sequence[str], now_ms: int) -> None:
        with self._lock:
            if self._closed:
                return
            if self._in_flight >= 2:
                return
            if text == self._submitted_text:
                return
            if now_ms - self._last_submit_ms < self.debounce_ms:
                return
            self._last_submit_ms = now_ms
            self._submitted_text = text
            self._in_flight += 1
            base_text = text
            context_tuple = tuple(context)
            generation = self._generation
            started = time.perf_counter()
            self._executor.submit(
                self._run, base_text, context_tuple, now_ms, started, generation
            )

    def _run(
        self,
        text: str,
        context: tuple[str, ...],
        now_ms: int,
        started: float,
        generation: int,
    ) -> None:
        prediction: Prediction | None
        error: str | None = None
        try:
            prediction = self.primary.predict(text=text, context=context, now_ms=now_ms)
        except Exception as exc:  # noqa: BLE001
            prediction = None
            error = str(exc)
        latency_ms = round((time.perf_counter() - started) * 1_000)
        with self._lock:
            self._in_flight -= 1
            if generation != self._generation:
                return
            self.last_error = error
            if prediction is not None:
                self._results.append((text, prediction, latency_ms))
                self._results = self._results[-8:]

    def take(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
        min_base_length: int = 0,
    ) -> Prediction | None:
        with self._lock:
            candidates = [
                (base_text, prediction, latency_ms)
                for base_text, prediction, latency_ms in self._results
                if prediction.base_length >= min_base_length
                and text.startswith(base_text)
            ]
        if candidates:
            _, prediction, latency_ms = max(
                candidates, key=lambda item: item[1].base_length
            )
            self.last_latency_ms = latency_ms
            self.last_precomputed = True
            return prediction
        self.last_latency_ms = 0
        self.last_precomputed = False
        if self.fallback is not None:
            return self.fallback.predict(text=text, context=context, now_ms=now_ms)
        return None

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._results.clear()
            self._submitted_text = ""
            self._last_submit_ms = -(10**9)
            self.last_latency_ms = 0
            self.last_precomputed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
        self._executor.shutdown(wait=False, cancel_futures=True)


class NgramPredictor:
    """Character n-gram predictor trained on a local reasoning corpus."""

    def __init__(self, order: int = 3, span_characters: int = 24) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if span_characters < 2:
            raise ValueError("span_characters must be at least 2")
        self.order = order
        self.span_characters = span_characters
        self._counts: dict[str, Counter[str]] = {}
        self.label = f"ngram({order})"

    @classmethod
    def default(cls, *, order: int = 3, span_characters: int = 24) -> NgramPredictor:
        model = cls(order=order, span_characters=span_characters)
        model.fit(DEFAULT_CORPUS)
        return model

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Path],
        *,
        order: int = 3,
        span_characters: int = 24,
    ) -> NgramPredictor:
        model = cls(order=order, span_characters=span_characters)
        for path in paths:
            model.fit(path.read_text(encoding="utf-8"))
        if not model._counts:
            raise ValueError("corpus did not contain enough characters")
        return model

    def fit(self, text: str) -> None:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for index in range(1, len(stripped)):
                for size in range(1, min(self.order, index) + 1):
                    context = stripped[index - size : index]
                    self._counts.setdefault(context, Counter())[stripped[index]] += 1

    def predict(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
    ) -> Prediction | None:
        del context
        seed = text
        probabilities: list[float] = []
        generated: list[str] = []
        while len(generated) < self.span_characters:
            suffix = (seed + "".join(generated))[-self.order :]
            counts = self._next_counts(suffix)
            if not counts:
                break
            total = sum(counts.values()) + len(counts)
            character, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            probabilities.append((count + 1) / total)
            generated.append(character)
            if character in "。．.!?！？":
                break
        if not generated:
            return None
        confidence = sum(probabilities) / len(probabilities)
        span = "".join(generated)
        return Prediction(
            span=span,
            operations=tuple(tag_operations(span)),
            confidence=min(1.0, max(0.0, confidence)),
            source=self.label,
            created_at_ms=now_ms,
            base_length=len(text),
        )

    def take(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
        min_base_length: int = 0,
    ) -> Prediction | None:
        del min_base_length
        return self.predict(text=text, context=context, now_ms=now_ms)

    def _next_counts(self, suffix: str) -> Counter[str]:
        for size in range(len(suffix), 0, -1):
            counts = self._counts.get(suffix[-size:])
            if counts:
                return counts
        return Counter()


def parse_prediction_payload(
    content: str,
    *,
    text: str,
    now_ms: int,
    source: str,
) -> Prediction:
    """Parse a JSON prediction returned by a language model."""
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("prediction response did not contain a JSON object")
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("prediction response JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("prediction response must be a JSON object")
    span = payload.get("span")
    if not isinstance(span, str) or not span.strip():
        raise ValueError("prediction response requires a non-empty span")
    raw_operations = payload.get("operations", [])
    if not isinstance(raw_operations, list) or not all(
        isinstance(operation, str) for operation in raw_operations
    ):
        raise ValueError("prediction operations must be a list of strings")
    operations = tuple(
        dict.fromkeys(
            operation for operation in raw_operations if operation in OPERATIONS
        )
    ) or tuple(tag_operations(span))
    confidence = payload.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError("prediction confidence must be numeric")
    return Prediction(
        span=span.strip(),
        operations=operations,
        confidence=min(1.0, max(0.0, float(confidence))),
        source=source,
        created_at_ms=now_ms,
        base_length=len(text),
    )


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    timeout: float,
    max_tokens: int = 64,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    request_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"prediction request failed: {exc}") from exc
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected prediction payload: {exc}") from exc


def _normalize_model_id(value: str) -> str:
    return value.lower().replace(":", "-").replace("_", "-")


def choose_model(identifiers: Sequence[str], preferred: Sequence[str] = ()) -> str:
    if not identifiers:
        raise ValueError("at least one model identifier is required")
    normalized = [
        (identifier, _normalize_model_id(identifier)) for identifier in identifiers
    ]
    for preference in preferred:
        needle = _normalize_model_id(preference)
        for identifier, candidate in normalized:
            if needle in candidate:
                return identifier
    return identifiers[0]


def resolve_model(
    base_url: str,
    timeout: float = 10.0,
    *,
    preferred: Sequence[str] = (),
) -> str:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not resolve a model from {url}: {exc}") from exc
    candidates = body.get("data") or body.get("models") or []
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"no models reported by {url}")
    identifiers = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("id", "model", "name"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                identifiers.append(value)
                break
    if not identifiers:
        raise RuntimeError(f"model entry from {url} has no identifier")
    return choose_model(identifiers, preferred)


class LocalLLMPredictor:
    """Predicts the next span with an OpenAI-compatible local model.

    The model is asked for the next short clause, the operations it performs,
    and a confidence. Any failure returns ``None`` so the runtime fails closed
    instead of inserting a cue based on a malformed prediction.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str | None = None,
        temperature: float = 0.4,
        timeout: float = 5.0,
        max_tokens: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.model = model or resolve_model(
            self.base_url,
            timeout=timeout,
            preferred=PREFERRED_PREDICTION_MODELS,
        )
        self.label = f"local-llm({self.model})"
        self.last_error: str | None = None

    def warmup(self) -> None:
        """Load the model without asking for a real prediction."""
        try:
            chat_completion(
                self.base_url,
                self.model,
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                timeout=self.timeout,
                max_tokens=1,
            )
        except (RuntimeError, TypeError, ValueError):
            pass

    def predict(
        self,
        *,
        text: str,
        context: Sequence[str],
        now_ms: int,
    ) -> Prediction | None:
        system = (
            "あなたは話し手の思考の続きを予測するモデルです。"
            "話し手が既に言った内容を繰り返さず、その後に続く1節だけを予測します。"
            "必ずJSONオブジェクトだけを返し、説明を含めないこと。"
        )
        history = "\n".join(context[-4:]) if context else "(なし)"
        user = (
            f"【直前までの発話】\n{history}\n\n"
            f"【現在の発話（途中）】\n{text}\n\n"
            "【指示】現在の発話の後に続く1節を40文字以内で予測してください。"
            "現在の発話をそのまま繰り返さないでください。\n\n"
            "【出力形式】\n"
            '{"span": "次に続く1節（40文字以内）", '
            '"operations": ["ground" などの操作名], '
            '"confidence": 0.0から1.0}'
        )
        try:
            content = chat_completion(
                self.base_url,
                self.model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
            prediction = parse_prediction_payload(
                content,
                text=text,
                now_ms=now_ms,
                source=self.label,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.last_error = str(exc)
            return None
        self.last_error = None
        return prediction
