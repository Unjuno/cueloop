# Changelog

## 0.3.0 (2026-11-12)

### Added
- Synthetic session-log generator (`scripts/synthesize_logs.py`):
  real-schema logs with a planted, recoverable reward structure for training
  the linear policy at scale (600-session corpus → ROC-AUC ≈ 0.70, close to
  the ≈ 0.73 Bayes ceiling of the generative model).
- One-command calibration benchmark (`scripts/run_synthetic_benchmark.sh`,
  documented in `docs/benchmark.md`): analyzer grid over effects × seeds ×
  backends (null cells exactly zero, 27/27 effect cells CI-excludes-zero) plus
  policy training/evaluation at scale.
- Operation-level exploration arm (`--assignment random-operation`): the timing
  policy still decides *when* to fire; the delivered operation is drawn
  uniformly from the missing-operation set so `requested_operation` and
  `operation` diverge. Seed-reproducible.
- Stem cues (`--cue-style stem`, the CLI default): instead of a bare
  interrogative word ("根拠は？"), the intervention is a complete,
  context-grounded "initial-condition" sentence that embodies the operation
  ("根拠は、観測データでは弱いからです。"). For `local`/`steer` the stem is
  written inside the decision call; for `rule`/`linear` a bounded synchronous
  LLM call writes it (`--stem-timeout`, default 1.5 s) with a predicted-span
  template and then bare wording as fallbacks. `--max-stem-characters` (default
  48) caps the sentence; stems take seconds to speak, so echo suppression
  windows are longer. Under `--async-decision` (default) the stem writer runs
  speculatively in the background for all backends, so boundaries do not block
  on the stem LLM; an unready precompute falls back to a bare cue on time.
  `--cue-style bare` restores the previous behaviour; experiment runs should
  use bare or length-match NEUTRAL to stem.
- SILENCE control (`--silent`): decisions and timing are logged, no cue is
  spoken, delivery is recorded as `SILENT`. Both conditions are comparable
  under the paired analysis.
- Paired CUE/SILENCE analysis (`scripts/analyze_experiment.py`): sessions are
  clusters, percentile bootstrap with fixed seed, `insufficient` below two
  complete pairs, `not_comparable` when no pair exists.
- Context-fit operation ranking (default): candidates are ordered by a
  hand-authored adjacency prior anchored to the spoken clause. The rule stays
  silent when the transcript carries no operation signal. `--no-context-fit`
  restores the legacy static priority order.
- Steering backend (`--decision steer`, experimental): a local LLM proposes
  the one move that best redirects the reasoning trajectory instead of filling
  the next missing vocabulary item. Logged under the `steering` family.
- `--adapter-args-json` on the experiment runner (e.g. a lower `--speech-db`
  for weak-signal acoustic tests).
- Synthetic generative calibration arm (`scripts/synthesize_experiment.py`):
  paired reports with a programmed cue effect so the analyzer's detection
  power can be checked without human data.
- CI: ruff and pytest run on every push/PR.

### Changed
- Default rule backend now ranks candidates by context fit rather than global
  PRIORITY. Fixture replays now produce `具体は？`/`逆は？`-style cues instead
  of `根拠は？` depending on the anchored move.
- Boundary-to-delivery delay is computed over delivered *and* silent
  decisions.

### Fixed
- Paired CUE/SILENCE exploration sessions digested identical
  `intervention_id`s; log files are namespaced before joins so SILENCE
  deliveries no longer drop CUE training samples.
- Unknown `assignment` strings are reported by `train_policy.py` instead of
  being silently ignored.

### Notes
- All numbers measured on fixtures are synthetic. No human session exists
  yet; claims about human reasoning outcomes are unsupported until paired
  live recordings exist.
