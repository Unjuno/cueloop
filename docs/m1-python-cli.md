# M1 Python CLI

This milestone validates the control loop before adding real-time audio.

## Scope

The original M1 prototype used three coarse operational states. The runnable core now uses the formal state model:

- `PRODUCTIVE`
- `RETRIEVAL_FAILURE`
- `STRATEGY_FAILURE`
- `FIXATION`
- `STRUCTURE_GAP`
- `OVERLOAD`

and the formal action space:

- `SILENCE`
- `BACKCHANNEL`
- `QUESTION`
- `MIRROR`
- `PERTURB`
- `CONSTRAIN`
- `RETRIEVE`
- `VERIFY`

The default analyzer is deterministic and heuristic-based. It is a replaceable baseline, not a validated cognitive-state model.

An optional provider-neutral semantic analyzer can be connected through an external command. The core still owns validation and delivery suppression.

A bounded `SessionContext` exposes only a small recent-turn window to semantic analysis. It is intentionally not an unbounded transcript or a long-term user model.

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

Bound recent context explicitly:

```bash
cueloop --context-turns 4 --context-characters 2400
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
  "user": "...bounded recent turns and current reasoning turn..."
}
```

It must emit exactly one JSON object on stdout:

```json
{
  "state": "STRUCTURE_GAP",
  "confidence": 0.82,
  "evidence": [
    {
      "kind": "TRANSCRIPT",
      "description": "同じ比較表現が反復している",
      "source_span": null
    }
  ],
  "proposed_action": "QUESTION",
  "cue": "何に依存？",
  "source_refs": []
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

- `PRODUCTIVE` and `OVERLOAD` must result in `SILENCE`.
- Low-confidence classifications must be suppressed.
- Analyzer failure must fail closed to `SILENCE`.
- Only one intervention may be delivered per turn.
- Repeated interventions are subject to cooldown.
- Cues have a strict length limit.
- Recent context is bounded by turn count and total characters.
- Logs record observable evidence and controller reasons.
- Semantic analyzers must describe observable text evidence, not claim access to hidden cognitive state.

## Current limitations

- No provider-specific LLM adapter is included.
- The text core remains provider-neutral; the repository now also includes the event-driven voice path and a macOS adapter described in `voice-mvp.md`.
- The heuristic baseline is intentionally narrow.
- Real-time context remains deliberately bounded; structured claims and questions are produced after the session rather than injected unbounded into the fast path.
- This implementation validates software boundaries, not cognitive effectiveness.

## Integration status

M1 is validated in CI and integrated with the M2–M5 runtime. Voice operation, completion evidence, and experiment procedures are documented in the repository README, `voice-mvp.md`, `completion-evidence.md`, and `experiment-protocol.md`.
