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

The initial analyzer is deterministic and heuristic-based. It is a replaceable baseline, not a validated cognitive-state model.

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

## Test

```bash
python -m unittest discover -s tests -v
```

## Design constraints

- Productive reasoning must result in `SILENCE`.
- Low-confidence classifications must be suppressed.
- Only one intervention may be delivered per turn.
- Repeated interventions are subject to cooldown.
- Cues have a strict length limit.
- Logs record observable evidence and controller reasons.

## Current limitations

- No LLM provider is connected.
- No microphone, ASR, VAD, or TTS exists yet.
- State labels are inferred from text heuristics only.
- The Japanese filler and completion rules are intentionally narrow.
- This implementation validates software boundaries, not cognitive effectiveness.

## Next implementation step

Introduce an analyzer protocol and an optional LLM-backed implementation while keeping the deterministic controller unchanged.
