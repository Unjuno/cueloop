#!/usr/bin/env python3
"""Replay real Japanese speech through the real CUELOOP runtime.

`scripts/simulate_sessions.py` proves the analyzer recovers a *known* effect
on a stereotyped speaker; this script does the same thing on real recorded
human speech. Two corpora are wired up:

- **Lux Japanese Speech Corpus** (Lami Inc., CC-BY-4.0) — 324 clean read
  sentences (`data/real-speech/lux/metadata.csv`). Read prose, no reasoning.
- **ReazonSpeech** (Reazon Holdings, CDLA-Sharing-1.0; YouTube-derived
  spontaneous/lecture-style Japanese) — sample selected at
  `data/real-speech/reazon/metadata-sample-200.csv`. Closer to the discourse
  register CUELOOP was designed for; still not live software reasoning.

How: whisper-base-mlx is run on each WAV (already cached the first time) to
get per-word timestamps. Each word boundary becomes a `transcript_partial`
event, bookended by `speech_started`/`speech_ended`, and the whole session is
fed through `SyncRuntime` in an unpaced replay. Each file is one "session";
the same file is replayed under both `CUE` and `SILENCE` to form a paired
comparison, and each replay writes an analyzer-ready report.

Note on live vs replayed speech: real speakers react to cues; canned audio
cannot. Uptake in this arm is structurally 0 — what this arm verifies is
that the pipeline composes end-to-end on real Japanese audio, whether the
content contains CUELOOP's operation vocabulary or not. For the
vocabulary-containing domain (Reazon sample: 30/200 lines tag at least one
operation), predictions have a real signal to act on.

License boundary: the Lux corpus is CC-BY-4.0; the ReazonSpeech mirror
(`osanpo/reazonspeech`) is CDLA-Sharing-1.0 (permissive sharing with
attribution); its metadata index (`efwkjn/reazonspeech_metadata`) is likewise
derived from YouTube audio. **Do not commit audio files** — `data/` is
git-ignored. Regenerate everything locally with the commands in
`docs/real-speech.md`.

Usage:
  scripts/simulate_lux_replay.py --output-dir logs/replay
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
      --whisper-cache data/real-speech/reazon/reazon-whisper-results.json \
      --source-name reazon \
      --output-dir artifacts/reazon-rule-200 --predictor ngram --decision rule
"""

from __future__ import annotations

import argparse
import csv
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
    SteeringDecisionModel,
)
from cueloop.sync.predictors import LocalLLMPredictor, NgramPredictor
from cueloop.sync.runtime import NullSpeaker, SyncRuntime

LUX_METADATA = ROOT / "data" / "real-speech" / "lux" / "metadata.csv"


def load_or_transcribe(
    metadata_path: Path,
    whisper_cache: Path,
    whisper_hf_repo: str = "/tmp/whisper-base",
    limit: int = 0,
) -> dict[str, dict[str, object]]:
    """Load per-file transcripts with word timestamps; cache to disk.

    With ``limit > 0``, only transcribe the first ``limit`` files.
    Cache written as a whole file; if a limit run is needed, the first
    ``limit`` entries are still read from the cache if present.
    """
    metadata = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    rows: list = metadata[:limit] if limit else metadata

    whisper_cache.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, object]] = {}
    if whisper_cache.exists():
        cache = json.loads(whisper_cache.read_text(encoding="utf-8"))

    missing = [
        row
        for row in rows
        if row["file_name"] not in cache
        and (metadata_path.parent / row["cleaned_path"]).exists()
    ]
    if not missing:
        subset = {
            row["file_name"]: cache[row["file_name"]]
            for row in rows
            if row["file_name"] in cache
        }
        if limit:
            return dict(list(subset.items())[:limit])
        return subset

    import mlx_whisper  # noqa: WPS433

    for row in missing:
        wav = metadata_path.parent / row["cleaned_path"]
        print(f"transcribing {row['file_name']}", flush=True)
        result = mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=whisper_hf_repo,
            language="ja",
            word_timestamps=True,
            verbose=False,
        )
        cache[row["file_name"]] = result
        whisper_cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    subset = {
        row["file_name"]: cache[row["file_name"]]
        for row in rows
        if row["file_name"] in cache
    }
    if limit:
        return dict(list(subset.items())[:limit])
    return subset


def events_for(file_name: str, segments: list[dict[str, object]]) -> list[SyncEvent]:
    events: list[SyncEvent] = []
    ms = 0
    seen_text = False
    for segment in segments:
        words = segment.get("words") or []
        if not isinstance(words, list) or not words:
            continue
        if not seen_text:
            events.append(SyncEvent(EventType.SPEECH_STARTED, ms))
            seen_text = True
        accumulated = ""
        for index, word in enumerate(words):
            text = str(word.get("word", ""))
            start_ms = int(float(word.get("start", 0.0)) * 1000)
            accumulated += text
            if index == len(words) - 1:
                events.append(
                    SyncEvent(EventType.TRANSCRIPT_FINAL, start_ms, accumulated)
                )
                events.append(SyncEvent(EventType.SPEECH_ENDED, start_ms))
            else:
                events.append(
                    SyncEvent(EventType.TRANSCRIPT_PARTIAL, start_ms, accumulated)
                )
    if not events:
        return []
    events.append(SyncEvent(EventType.SESSION_ENDED, ms + 1))
    return events


def run_session(
    *,
    file_name: str,
    segments: list[dict[str, object]],
    condition: str,
    pause_ms: int,
    cooldown_ms: int,
    min_text_characters: int,
    predictor_name: str = "cycle",
    decision_name: str = "rule",
    local_model: str = "qwen2.5:7b-instruct",
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
    elif predictor_name == "cycle":
        from cueloop.sync.predictors import ScriptedPredictor

        # A minimal synthetic cycle predictor, useful as a stable test target.
        # It rotates one operation-spanning span per call; CUELOOP's rule
        # policy will therefore fire a deterministic sequence of cues.
        predictor = ScriptedPredictor(
            {
                "結論は": ("だから結論は決まる。", 0.9),
                "前提と": ("依存関係を分けて考える必要がある。", 0.9),
                "この設計": ("前提と依存関係を分けて考える必要がある。", 0.9),
            },
            fallback=("だから結論は決まる。", 0.9),
        )
    else:
        msg = f"unknown predictor: {predictor_name}"
        raise ValueError(msg)
    if decision_name == "rule":
        decision_model = RuleDecisionModel(
            CanonicalAssigner(),
            context_fit=True,
        )
    elif decision_name == "local":
        decision_model = LocalLLMDecisionModel(
            base_url=local_base_url,
            model=local_model,
            assigner=CanonicalAssigner(),
            timeout=30.0,
            max_tokens=192,
            cue_style="bare",
        )
    elif decision_name == "steer":
        decision_model = SteeringDecisionModel(
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
        session_id=f"{condition.lower()}-{file_name.removesuffix('.wav').replace('/', '-')}",
        pair_key=file_name.removesuffix(".wav").replace("/", "-"),
        silent=silent,
        log_path=None,
    )
    events = events_for(file_name, segments)
    if not events:
        return {
            "predictor": "none",
            "decision_model": "none",
            "condition": condition,
            "assignment": "canonical",
            "seed": 0,
            "session_id": f"{condition.lower()}-{file_name.removesuffix('.wav').replace('/', '-')}",
            "pair_key": file_name.removesuffix(".wav").replace("/", "-"),
            "lux_file": file_name,
            "synthetic": True,
            "note": "lux file had no recognizable words; skipped",
            "utterances": 0,
            "interventions": 0,
            "delivered": 0,
            "outcomes": [],
        }
    for event in events:
        runtime.feed(event)
    return runtime.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=LUX_METADATA)
    parser.add_argument(
        "--whisper-cache",
        type=Path,
        default=None,
        help="default: alongside the corpus (lux-whisper-results.json for Lux)",
    )
    parser.add_argument(
        "--source-name",
        default="lux",
        help="label embedded in the report note (e.g. 'lux', 'reazon')",
    )
    parser.add_argument("--pause-ms", type=int, default=350)
    parser.add_argument("--cooldown-ms", type=int, default=1_500)
    parser.add_argument("--min-text-characters", type=int, default=4)
    parser.add_argument("--whisper-model", default="/tmp/whisper-base")
    parser.add_argument("--limit", type=int, default=0, help="only N files (0 = all)")
    parser.add_argument(
        "--predictor",
        choices=["local", "ngram"],
        default="ngram",
    )
    parser.add_argument(
        "--decision",
        choices=["local", "rule", "steer"],
        default="rule",
    )
    parser.add_argument("--local-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--local-base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    whisper_cache = args.whisper_cache
    if whisper_cache is None:
        if args.corpus == LUX_METADATA:
            whisper_cache = LUX_METADATA.with_name("lux-whisper-results.json")
        else:
            whisper_cache = args.corpus.with_name(
                args.corpus.stem + "-whisper-results.json"
            )
    cache = load_or_transcribe(args.corpus, whisper_cache, args.whisper_model, limit=args.limit)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(
            f"error: output directory is not empty: {args.output_dir}", file=sys.stderr
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("CUE", "SILENCE"):
        out = args.output_dir / condition.lower()
        out.mkdir(parents=True, exist_ok=True)
        manifest = []
        for file_name, payload in sorted(cache.items()):
            report = run_session(
                file_name=file_name,
                segments=payload.get("segments", []),  # type: ignore[arg-type]
                condition=condition,
                pause_ms=args.pause_ms,
                cooldown_ms=args.cooldown_ms,
                min_text_characters=args.min_text_characters,
                predictor_name=args.predictor,
                decision_name=args.decision,
                local_model=args.local_model,
                local_base_url=args.local_base_url,
            )
            report["lux_file"] = file_name
            report["synthetic"] = True
            report["note"] = (
                f"Recorded Japanese speech ({args.source_name}) replayed offline; "
                "no human uptake is possible — this tests measurement, not effect."
            )
            slug = file_name.removesuffix(".wav").replace("/", "-")
            path = out / f"{slug}.json"
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest.append(path.name)
        print(
            json.dumps(
                {"condition": condition, "sessions": len(manifest)}, ensure_ascii=False
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
