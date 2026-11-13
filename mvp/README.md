# Decision Sync MVP

This is the minimal, independent decision-model path. It is intentionally separate from the older multi-stage cognition pipeline.

## Run

Python 3.11+; standard library only.

```bash
python -m unittest discover -s tests -v
printf '%s\n' '{"type":"transcript","text":"モデルの効率を比較したい","at_ms":0}' '{"type":"boundary","at_ms":400}' | python -m mvp
```

Default replay is synchronous and deterministic. Add `--async-model` to run inference on background worker threads; in this mode, a boundary returns NONE immediately when inference is unfinished. A later boundary may deliver the candidate only if its transcript revision is still current and it has not expired.

To connect an LLM:

```bash
python -m mvp --async-model --model-command python path/to/adapter.py
```

The adapter reads JSON `{"system":"...","user":"..."}` on stdin and writes exactly one JSON object `{"action":"NONE"}` or `{"action":"CUE","text":"比較軸"}` to stdout. Without a model command, the safe default is NONE.

## Guarantees and limitations

- Only NONE or a cue of up to 16 characters can be delivered.
- Delivery checks transcript revision, candidate age, duplicate text, cooldown and speaking state.
- Model failure, malformed JSON and unfinished inference produce NONE.
- Inference is asynchronous in `--async-model`, but the external model command still runs per transcript revision; model backpressure and cancellation are not fully optimized.
- This is **not yet** a microphone-to-speaker MVP. There is no real ASR/VAD/TTS binding in this minimal path.
- No test here establishes that cues improve human reasoning or learning.

Next: connect the existing macOS audio-event source and cancellable TTS sink through a small adapter; verify actual audio end-to-end in a container where hardware is available, then remove the legacy default path in a separate change.
