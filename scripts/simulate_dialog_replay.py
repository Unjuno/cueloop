#!/usr/bin/env python3
"""Replay MagicData Japanese spontaneous two-person dialogue through SyncRuntime.

Sibling to `scripts/simulate_lux_replay.py`, except instead of whisper-mlx
+Lux-style per-file recitation, this uses the publisher-provided TXT
timeline: `[start_sec, end_sec] \\t speaker_id \\t gender \\t transcript` with
utterances as the natural "utterance" atoms for CUELOOP.

The corpus is conversational Japanese (two spontaneous speakers per file),
much closer to the register CUELOOP is built for than Lux recited prose.

Works at utterance level rather than word level: word timestamps do not
exist in the corpus, so each utterance is split into ~3-6 char chunks
weighed by character count, and each chunk is emitted as a
`transcript_partial` event spread evenly over `[start_ms, end_ms]`.

This is CANNED dialogue: the speakers do not react to cues, so uptake via
this arm is structurally 0. What this verifies is that the CUELOOP pipeline
fires, decides, and delivers on real Japanese two-person conversation audio.

License: MagicData card tags the dataset Apache-2.0. It is used here for
non-redistributive evaluation; the WAVs are not committed (data/ is
git-ignored).

Usage:
  scripts/simulate_dialog_replay.py --output-dir logs/dialog-replay --xct files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueloop.sync.models import EventType, SyncEvent
from cueloop.sync.policy import (
    CanonicalAssigner,
    LocalLLMDecisionModel,
    RuleDecisionModel,
)
from cueloop.sync.predictors import LocalLLMPredictor, NgramPredictor
from cueloop.sync.runtime import NullSpeaker, SyncRuntime

CORPUS_ROOT = (
    ROOT
    / "data"
    / "real-speech"
    / "magicdata"
    / "data"
    / "wavs"
    / "Japanese_Spontaneous_Conversation_Training_Dataset"
)


def parse_corpus(corpus_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Return dict mapping file stem -> list of utterance dicts in timeline order."""
    txt_dir = corpus_dir / "TXT"
    parsed: dict[str, list[dict[str, object]]] = {}
    for path in sorted(txt_dir.glob("*.txt")):
        stem = path.stem
        utterances: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            time_range, speaker, _gender, text = (
                parts[0],
                parts[1],
                parts[2],
                "\t".join(parts[3:]),
            )
            try:
                start_s, end_s = (
                    time_range.split(",")[0].lstrip("["),
                    time_range.split(",")[-1].rstrip("]"),
                )
                start_ms = int(float(start_s) * 1000)
                end_ms = int(float(end_s) * 1000)
            except (ValueError, IndexError):
                continue
            if start_ms >= end_ms:
                continue
            clean = text.strip()
            if not clean:
                continue
            if not any(
                0x3040 <= ord(ch) <= 0x309F
                or 0x30A0 <= ord(ch) <= 0x30FF
                or 0x4E00 <= ord(ch) <= 0x9FFF
                or (ch.isascii() and ch.isalpha())
                for ch in clean
            ):
                # utterance carries only marker text (no spoken content)
                continue
            utterances.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": speaker,
                    "text": clean,
                }
            )
        parsed[stem] = utterances
    return parsed


def _chunk_text(text: str, start_ms: int, end_ms: int) -> list[tuple[int, str]]:
    """Yield (at_ms, cumulative_text) events for an utterance."""
    if len(text) <= 4:
        return [(start_ms, text)]
    # aim for ~4-6 char chunks
    n = max(1, min(12, len(text) // 5 + 1))
    step = max(1, len(text) // n)
    span = end_ms - start_ms
    events = []
    for i in range(0, len(text), step):
        pos = min(i + step, len(text))
        at_ms = start_ms + int(span * (pos / len(text)))
        events.append((at_ms, text[:pos]))
    return events


def events_for(stem: str, utterances: list[dict[str, object]]) -> list[SyncEvent]:
    events: list[SyncEvent] = []
    started = False
    for utt in utterances:
        start_ms = int(utt["start_ms"])
        end_ms = int(utt["end_ms"])
        text = str(utt["text"])
        if not started:
            events.append(SyncEvent(EventType.SPEECH_STARTED, 0))
            started = True
        for chunk_ms, chunk_text in _chunk_text(text, start_ms, end_ms):
            events.append(SyncEvent(EventType.TRANSCRIPT_PARTIAL, chunk_ms, chunk_text))
        events.append(SyncEvent(EventType.TRANSCRIPT_FINAL, end_ms, text))
        events.append(SyncEvent(EventType.SPEECH_ENDED, end_ms))
    if not events:
        return []
    last_ms = events[-1].at_ms
    events.append(SyncEvent(EventType.SESSION_ENDED, last_ms + 1))
    return events


def run_session(
    *,
    file_name: str,
    utterances: list[dict[str, object]],
    condition: str,
    pause_ms: int,
    cooldown_ms: int,
    min_text_characters: int,
    predictor_name: str = "ngram",
    decision_name: str = "rule",
    local_model: str = "gemma3:4b-it-qat",
    local_base_url: str = "http://127.0.0.1:11434",
) -> dict[str, object]:
    silent = condition == "SILENCE"
    if predictor_name == "ngram":
        predictor = NgramPredictor.default()
    elif predictor_name == "local":
        predictor = LocalLLMPredictor(
            base_url=local_base_url,
            model=local_model,
            timeout=15.0,
            max_tokens=64,
        )
    else:
        msg = f"unknown predictor: {predictor_name}"
        raise ValueError(msg)
    if decision_name == "rule":
        decision_model = RuleDecisionModel(CanonicalAssigner(), context_fit=True)
    elif decision_name == "local":
        decision_model = LocalLLMDecisionModel(
            base_url=local_base_url,
            model=local_model,
            assigner=CanonicalAssigner(),
            timeout=30.0,
            max_tokens=192,
            cue_style="bare",
        )
    else:
        msg = f"unknown decision model: {decision_name}"
        raise ValueError(msg)
    runtime = SyncRuntime(
        predictor=predictor,
        decision_model=decision_model,
        speaker=NullSpeaker(),
        pause_ms=pause_ms,
        cooldown_ms=cooldown_ms,
        min_text_characters=min_text_characters,
        outcome_window_ms=8_000,
        condition=condition,
        assignment="canonical",
        seed=0,
        session_id=f"{condition.lower()}-{file_name}",
        pair_key=file_name,
        silent=silent,
        log_path=None,
    )
    events = events_for(file_name, utterances)
    for event in events:
        runtime.feed(event)
    return runtime.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--pause-ms", type=int, default=350)
    parser.add_argument("--cooldown-ms", type=int, default=1_500)
    parser.add_argument("--min-text-characters", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--predictor", choices=["ngram", "local"], default="ngram")
    parser.add_argument("--decision", choices=["rule", "local"], default="rule")
    parser.add_argument("--local-model", default="gemma3:4b-it-qat")
    parser.add_argument("--local-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--source-name", default="dialog")
    args = parser.parse_args()

    parsed = parse_corpus(args.corpus_dir)
    stems = sorted(parsed)
    if args.limit:
        stems = stems[: args.limit]
    print(
        f"dialog files: {len(stems)} ({sum(len(v) for v in parsed.values())} utterances parsed)"
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"error: output directory is not empty: {args.output_dir}", file=sys.stderr
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("CUE", "SILENCE"):
        out = args.output_dir / condition.lower()
        out.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            report = run_session(
                file_name=stem,
                utterances=parsed[stem],
                condition=condition,
                pause_ms=args.pause_ms,
                cooldown_ms=args.cooldown_ms,
                min_text_characters=args.min_text_characters,
                predictor_name=args.predictor,
                decision_name=args.decision,
                local_model=args.local_model,
                local_base_url=args.local_base_url,
            )
            report["dialog_file"] = stem
            report["source_name"] = args.source_name
            report["utterance_count"] = len(parsed[stem])
            slug = stem.replace("/", "-")
            (out / f"{slug}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {"condition": condition, "files": len(stems)}, ensure_ascii=False
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
