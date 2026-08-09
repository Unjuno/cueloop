from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit CUELOOP replay JSONL from an audio file using faster-whisper."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--turn-gap-ms", type=int, default=450)
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Install faster-whisper to use this adapter") from exc

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments, _info = model.transcribe(
        str(args.audio),
        language=args.language,
        vad_filter=True,
        word_timestamps=False,
    )

    last_end_ms: int | None = None
    for segment in segments:
        start_ms = max(0, int(round(segment.start * 1000)))
        end_ms = max(start_ms, int(round(segment.end * 1000)))
        text = segment.text.strip()
        if not text:
            continue

        if last_end_ms is not None:
            gap_ms = max(0, start_ms - last_end_ms)
            if gap_ms >= args.turn_gap_ms:
                print(
                    json.dumps(
                        {
                            "type": "boundary",
                            "kind": "TURN_CANDIDATE",
                            "at_ms": start_ms,
                            "silence_ms": gap_ms,
                        },
                        ensure_ascii=False,
                    )
                )

        print(
            json.dumps(
                {
                    "type": "transcript",
                    "kind": "FINAL",
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                },
                ensure_ascii=False,
            )
        )
        last_end_ms = end_ms

    if last_end_ms is not None:
        print(
            json.dumps(
                {
                    "type": "boundary",
                    "kind": "TURN_CANDIDATE",
                    "at_ms": last_end_ms + args.turn_gap_ms,
                    "silence_ms": args.turn_gap_ms,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
