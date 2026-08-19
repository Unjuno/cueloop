# Software Completion Evidence

This matrix defines what “complete” means for the current CUELOOP software scope. It does not redefine scientific efficacy as a software deliverable.

| Requirement | Authoritative artifact | Executable evidence |
|---|---|---|
| M0 formal states, actions, evidence, prohibitions | `schemas/`, `core.py`, `llm.py` | schema-space and fail-closed contract tests |
| M1 bounded text control loop | `core.py`, `cli.py` | unit tests and JSONL decision logs |
| M2 10–30 minute voice runtime | `voice.py`, macOS adapter | live command E2E, candidate deadlines, resumed-speech/TTS cancellation, 10-minute soak |
| M3 post-session reasoning structure | `structure.py` | integrated session structure artifact and category tests |
| M4 reviewable grounding | `knowledge.py` | indexed-source replacement and unsupported-claim tests |
| M5 bounded personalization | `personalization.py`, `memory.py` | explicit feedback, expiry, opt-in structure retention/forgetting, privacy, suppression-only tests |
| Privacy and retention | `session.py`, `privacy.py` | atomic manifests, semantic/hash verification, no-transcript mode, dry-run and applied expiry deletion |
| Safe autonomous policy improvement | `experiment.py`, `policy-eval.jsonl` | disjoint tune/eval validation, weighted loss, no-regression adoption gate |
| Human experiment operations | `pilot.py`, `experiment-protocol.md` | counterbalancing, verified session ingestion, delay/blinding deviations, paired uncertainty statistics |
| Distribution | `pyproject.toml`, CI workflow | clean wheel build, Python 3.11/3.12 jobs, macOS Swift compile |

Run the complete local gate:

```bash
python -m unittest discover -s tests -v
cueloop-experiment --report /tmp/policy-report.json --policy-out /tmp/policy.json
cueloop-soak --minutes 10 --report /tmp/soak.json
cueloop-audit
```

The gate may establish software completeness and experimental readiness. Only consented participant data satisfying the documented design can provide evidence about retained reasoning. Synthetic data and replay labels are never efficacy evidence.
