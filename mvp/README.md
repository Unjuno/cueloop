# Decision Sync MVP

This is the intentionally small, independent experimental path. The legacy `src/cueloop` implementation remains untouched in this branch so the rewrite is reversible.

## Run

Python 3.11+; no third-party dependencies.

```bash
python -m unittest tests.test_decision_sync_mvp -v
printf '%s\n' '{"type":"transcript","text":"モデルの効率を比較したい","at_ms":0}' '{"type":"boundary","at_ms":400}' | python -m mvp
```

Without a model command the safe default is NONE. Supply `--model-command python path/to/your_adapter.py` to obtain live decisions from an LLM. The adapter must accept JSON with `system` and `user` keys on stdin and return `{"action":"NONE"}` or `{"action":"CUE","text":"比較軸"}` on stdout.

## Scope and limits

This is a text-event replay MVP, **not yet a microphone-to-speaker product**. The model invocation is synchronous, so it does not yet prefetch during speech. The boundary gate prevents stale, repeated, or excessive cues. Integrate with the existing ASR/TTS adapters after replay behavior is satisfactory. No cognitive improvement claim is made.
