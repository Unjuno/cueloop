# Contributing to CUELOOP

CUELOOP is a research vehicle, so the bar for changing the experiment is
different from the bar for changing the plumbing.

## Ground rules

- **Do not weaken the experiment.** The predictor, boundary detector, timing,
  assignment policy, and fallback behavior are held constant across conditions
  by design. Changing them changes what is being measured. If you believe one
  is wrong, open an issue with evidence rather than patching around it in a PR
  that also claims an effect.
- **Synthetic data proves the pipeline, not the hypothesis.** Numbers from
  `synthesize_experiment.py` / `synthesize_logs.py` calibrate the analyzer and
  the trainer. They say nothing about human speech and must never appear as
  "evidence for the cue effect" in docs or release notes.
- **No human data leaves this repo.** Session logs contain transcripts. They
  belong in `data/` (git-ignored). Never commit logs, audio, or `artifacts/`.
- **License:** MIT. See `LICENSE`. Attribution is appreciated; require it in
  your fork only if you must.

## Setup and checks

```bash
python3.11 -m venv .venv          # or 3.12
.venv/bin/pip install -e . pytest ruff
bash scripts/check.sh             # ruff + pytest + unittest + demo replay
```

CI additionally enforces `ruff format --check`. Please run `ruff format` on
changed files before pushing. Commit messages follow Conventional Commits
(`feat:`, `fix:`, `docs:`, `chore:`).

## What a change usually needs

- A failing test first, when feasible.
- `scripts/check.sh` green.
- If you change logging, update `docs/sync-core.md` and the log schema tables.
- If you change an outcome definition (uptake, carryover, repertoire), update
  `docs/experiment-protocol.md` and `docs/benchmark.md` in the same PR —
  the analyzer and docs must agree.

## Reproducing the calibration benchmark

```bash
scripts/run_synthetic_benchmark.sh   # writes artifacts/benchmark/ (~1 min)
```

It is seeded; results should match `docs/benchmark.md` exactly. If yours
differ, either the environment changed or a metric changed — investigate
before claiming an improvement.

## Reporting results

Claims about uptake/carryover/reconstruction require paired live sessions
(CUE/SILENCE) recorded with `scripts/run_experiment.py`, and analysis with
`scripts/analyze_experiment.py`. Report effect sizes with intervals, not just
p-values, and the number of eligible boundaries/deliveries/cancellations.
