# M1 Python CLI

This milestone validates the control loop before adding real-time audio.

## Scope

The prototype deliberately uses three operational states:

- `FLOWING`
- `PAUSED`
- `STUCK`

and three actions:

- `SILENCE`
- `CONTINUE`
- `CUE`

The default analyzer is deterministic and heuristic-based. It is a replaceable baseline, not a validated cognitive-state model.

An optional provider-neutral semantic analyzer can be connected through an external command. The core still owns validation and delivery suppression.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run

Heuristic baseline:

```bash
cueloop
```

Disable logging:

```bash
cueloop --no-log
```

Use a custom JSONL log path:

```bash
cueloop --log logs/manual-session.jsonl
```

## External semantic analyzer

CUELOOP can delegate semantic analysis to any executable without taking a dependency on a specific LLM SDK:

```bash
cueloop --analyzer-command python tools/my_llm_adapter.py
```

The command receives one JSON object on stdin:

```json
{
  "system": "...CUELOOP analyzer instructions...",
  "user": "Analyze this reasoning turn only: ..."
}
```

It must emit exactly one JSON object on stdout:

```json
{
  "state": "STUCK",
  "confidence": 0.82,
  "evidence": ["同じ比較表現が反復している"],
  "proposed_action": "CUE",
  "cue": "何に依存？"
}
```

Invalid JSON, provider failures, schema violations, invalid state/action combinations, or timeouts fail closed to `SILENCE`.

The external analyzer proposes. `InterventionController` decides delivery.

## Test

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the suite on Python 3.11 and 3.12.

## Design constraints

- Productive reasoning must result in `SILENCE`.
- Low-confidence classifications must be suppressed.
- Analyzer failure must fail closed to `SILENCE`.
- Only one intervention may be delivered per turn.
- Repeated interventions are subject to cooldown.
- Cues have a strict length limit.
- Logs record observable evidence and controller reasons.
- Semantic analyzers must describe observable text evidence, not claim access to hidden cognitive state.

## Current limitations

- No provider-specific LLM adapter is included.
- No microphone, ASR, VAD, or TTS exists yet.
- The heuristic baseline is intentionally narrow.
- The semantic analyzer sees one reasoning turn rather than full session context.
- This implementation validates software boundaries, not cognitive effectiveness.

## Next implementation step

Add session context as a bounded, structured object so a semantic analyzer can distinguish local hesitation from repeated fixation without receiving an unbounded transcript.
