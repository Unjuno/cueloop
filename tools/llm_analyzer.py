#!/usr/bin/env python3
"""Read a {"system": ..., "user": ...} JSON prompt from stdin, call the
OpenAI-compatible endpoint, and print only the analyzer-result JSON to stdout.

The CUELOOP contract demands a single JSON object with exact keys:
state, confidence, evidence, proposed_action, cue, source_refs.
This adapter is a fallback to a local model, so it also sanitizes output.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

DEFAULT_URL = os.environ.get("CUELOOP_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("CUELOOP_LLM_MODEL", "qwen3-coder:30b")
DEFAULT_TIMEOUT = float(os.environ.get("CUELOOP_LLM_TIMEOUT", "8"))
DEFAULT_TEMPERATURE = float(os.environ.get("CUELOOP_LLM_TEMPERATURE", "0.2"))

KNOWN_STATES = (
    "PRODUCTIVE",
    "RETRIEVAL_FAILURE",
    "STRATEGY_FAILURE",
    "FIXATION",
    "STRUCTURE_GAP",
    "OVERLOAD",
)
KNOWN_ACTIONS = (
    "SILENCE",
    "BACKCHANNEL",
    "QUESTION",
    "MIRROR",
    "PERTURB",
    "CONSTRAIN",
    "RETRIEVE",
    "VERIFY",
)


def fail_closed(reason: str) -> dict:
    return {
        "state": "RETRIEVAL_FAILURE",
        "confidence": 0.0,
        "evidence": [
            {
                "kind": "INTERACTION_HISTORY",
                "description": reason,
                "source_span": None,
            }
        ],
        "proposed_action": "SILENCE",
        "cue": None,
        "source_refs": [],
    }


def sanitize(payload: object) -> dict:
    if not isinstance(payload, dict):
        return fail_closed("analyzer output is not an object")
    allowed = {"state", "confidence", "evidence", "proposed_action", "cue", "source_refs"}
    if set(payload) != allowed:
        return fail_closed(
            f"analyzer keys do not match contract: {sorted(set(payload))}"
        )
    state = payload.get("state")
    action = payload.get("proposed_action")
    if state not in KNOWN_STATES or action not in KNOWN_ACTIONS:
        return fail_closed("invalid state or proposed_action")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return fail_closed("confidence is not numeric")
    if not 0.0 <= confidence <= 1.0:
        return fail_closed("confidence out of range")
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return fail_closed("evidence is empty or not a list")
    normalized_evidence = []
    for item in evidence:
        if not isinstance(item, dict):
            return fail_closed("evidence item is not an object")
        kind = item.get("kind")
        description = item.get("description")
        if kind not in ("TRANSCRIPT", "TIMING", "PROSODY", "INTERACTION_HISTORY", "USER_SIGNAL"):
            return fail_closed("invalid evidence kind")
        if not isinstance(description, str) or not description.strip():
            return fail_closed("invalid evidence description")
        normalized_evidence.append(
            {
                "kind": kind,
                "description": description.strip(),
                "source_span": item.get("source_span"),
            }
        )
    cue = payload.get("cue")
    if cue is not None and not isinstance(cue, str):
        return fail_closed("cue is not a string or null")
    source_refs = payload.get("source_refs")
    if not isinstance(source_refs, list) or not all(
        isinstance(item, str) and item.strip() for item in source_refs
    ):
        return fail_closed("source_refs is not a list of non-empty strings")
    text_actions = {"QUESTION", "MIRROR", "PERTURB", "CONSTRAIN", "RETRIEVE"}
    if action in text_actions and not (cue or "").strip():
        return fail_closed("text action requires a cue")
    if action not in text_actions and cue is not None:
        return fail_closed("non-text action must not include a cue")
    if state in ("PRODUCTIVE", "OVERLOAD") and action != "SILENCE":
        return fail_closed("productive/overload may only propose SILENCE")
    if action == "RETRIEVE" and not source_refs:
        return fail_closed("RETRIEVE requires source_refs")
    if action != "RETRIEVE" and source_refs:
        return fail_closed("only RETRIEVE may include source_refs")
    if action in text_actions:
        cue = shorten_cue(cue or "", action)
    return {
        "state": state,
        "confidence": confidence,
        "evidence": normalized_evidence,
        "proposed_action": action,
        "cue": (cue or "").strip() if cue is not None else None,
        "source_refs": [item.strip() for item in source_refs],
    }


CANONICAL_CUES = {
    "QUESTION": ("何に依存？", "具体的には？", "条件は？", "根拠は？"),
    "MIRROR": ("つまり、こういう仮定？", "前提はこれ？", "依存はこれ？"),
    "PERTURB": ("逆なら？", "どこで破れる？", "反例は？", "別の前提なら？"),
    "CONSTRAIN": ("結論は？", "絞ると？", "どれを捨てる？"),
    "RETRIEVE": ("根拠を探す。", "文献を引く。"),
}


def shorten_cue(cue: str, action: str) -> str:
    """Force a deliverable cue: keep it if short enough, otherwise fall back
    to a canonical 16-char-or-shorter cue for that action."""
    candidate = cue.strip()
    if len(candidate) <= 16:
        return candidate
    fallbacks = CANONICAL_CUES.get(action, CANONICAL_CUES["QUESTION"])
    return fallbacks[0]


def call_completion(prompt: dict) -> dict:
    request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": prompt.get("system", "")},
            {"role": "user", "content": prompt.get("user", "")},
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": 256,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(request).encode("utf-8")
    req = urllib.request.Request(
        os.environ.get("CUELOOP_LLM_URL", DEFAULT_URL),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(strip_json_fences(content))


def strip_json_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        first_line_break = stripped.find("\n")
        stripped = stripped[first_line_break + 1 :] if first_line_break != -1 else stripped
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def main() -> int:
    raw = sys.stdin.read()
    try:
        prompt = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps(fail_closed("adapter prompt is invalid JSON"), ensure_ascii=False))
        return 0
    try:
        payload = call_completion(prompt)
    except Exception as exc:
        print(
            json.dumps(
                fail_closed(f"adapter transport failure: {type(exc).__name__}"),
                ensure_ascii=False,
            )
        )
        return 0
    print(json.dumps(sanitize(payload), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())