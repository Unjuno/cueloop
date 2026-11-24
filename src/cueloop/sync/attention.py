"""Attention-grounded cue *selection*.

The rule backend previously chose the cue operation from a fixed priority
list, so the cue could be unrelated to the term the speaker was actually
circling. This module asks the local model only for the *attended term* in
the live context; the mapping from term to operation is then done by the
existing deterministic vocabulary (``tag_operations``), so the LLM never
chooses an operation on its own and the result stays reproducible.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .stem import StemCueGenerator
from .vocabulary import tag_operations

SYSTEM = (
    "あなたは日本語の会話を読み、話し手がいま最も引っかかっている語を一つだけ"
    "抜き出すアシスタントです。説明や理由は書かず、語だけを出力します。"
)


class AttentionTermExtractor:
    """Extract the single attended term from the live context via local LLM."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str | None = None,
        timeout: float = 1.5,
        max_tokens: int = 32,
    ) -> None:
        # StemCueGenerator owns the OpenAI-compatible chat call; reuse it so
        # the attention extractor and the stem writer share one transport.
        self._client = StemCueGenerator(
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
        )
        self.last_error: str | None = None
        self.last_term: str | None = None

    def extract(
        self, *, text: str, context: Sequence[str] = (), span: str = ""
    ) -> str | None:
        """Return one attended term present in the context, or None."""
        recent = " / ".join(context[-4:]) or "なし"
        user = (
            f"【直前の文脈】{recent}\n"
            f"【今の発話】{text}\n"
            f"【続きのヒント】{span}\n\n"
            "今の発話と文脈の中で、話し手が最も引っかかっている語（未解決な概念）を"
            "一つだけ抜き出してください。文脈に出てくる語をそのまま使い、"
            "余計な言葉・記号・説明は付けないでください。\n"
            "語:\n"
        )
        try:
            raw = self._client._complete(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=32,
            )
        except RuntimeError as exc:
            self.last_error = str(exc)
            return None
        term = self._clean(raw)
        if not term:
            self.last_error = "empty attention term"
            return None
        if not self._in_context(term, text, context):
            self.last_error = f"term not grounded in context: {term}"
            return None
        self.last_error = None
        self.last_term = term
        return term

    @staticmethod
    def _clean(raw: str) -> str | None:
        text = raw.strip()
        # Tolerate JSON {"term": "..."} or a bare word.
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return None
            value = payload.get("term") if isinstance(payload, dict) else None
            text = value.strip() if isinstance(value, str) else ""
        text = re.sub(r"^[「\"']+|[」\"']+$", "", text).strip()
        text = text.split("\n")[0].strip()
        if not text or len(text) > 16 or "。" in text:
            return None
        return text

    @staticmethod
    def _in_context(term: str, text: str, context: Sequence[str]) -> bool:
        haystack = f"{text} / {' / '.join(context)}"
        return term in haystack


def operation_for_term(term: str) -> str | None:
    """Map an attended term to an operation using the deterministic vocabulary."""
    operations = tag_operations(term)
    return operations[0] if operations else None
