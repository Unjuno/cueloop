# Changelog

## Unreleased

### Fixed
- **Runtime delivery clamp**: synchronous backends (fresh prediction created
  in the same event) no longer have their intervention pushed out by
  prediction+decision wall latency. Before, the shift landed the cue into an
  already-diverged window and the synchronizer cancelled it — so every
  local-LLM intervention on the ReazonSpeech sample and in live replays was
  cancelled (0 delivered). With the fix the local-LLM backend delivers at
  the boundary event, like the rule backend: 668/668 on the full 200-file
  ReazonSpeech sample (gemma3:4b-it-qat), 138/138 on the
  operation-tagged 30-file subset, 324/324 on the full Lux 328-file corpus
  (gemma3:4b-it-qat), and 327/327 on the Reazon qwen2.5:1.5b-instruct full
  sample; synthetic calibration unchanged (null cells +0.000, 27/27 effect
  cells CI-excluding-zero).

### Added
- Per-session repetition fatigue in `RuleDecisionModel` (`fatigue_penalty`,
  default 2.0): each chosen operation accrues fatigue so the rule backend
  stops delivering a near-monotone stream of one cue on discourse Japanese.
  Cue diversity on the ReazonSpeech arm rose from 94% `concretize` to a
  7-operation mix (concretize 46%); delivery counts, prediction counts and all paired analyzer deltas
  (+0.000) are unchanged; synthetic null/effect cells are unchanged
  (27/27 effect cells CI-excludes-zero).
- `scripts/simulate_lux_replay.py` accepts `--corpus`/`--whisper-cache`/
  `--source-name` and replays any compatible corpus (Lux or ReazonSpeech).
  Fixes a whisper-cache subset bug (warm cache used to return the whole
  file regardless of the requested corpus) and a relative-path corpus
  resolution bug that created duplicate caches; both covered by tests.
- Real-speech replay completed end-to-end on the Lux and Reazon corpora for
  the local-LLM decision path, pre-fix and post-fix; post-fix measurements
  now dominate the docs (pre-fix numbers retained for transparency, clearly
  labeled "pre-runtime-fix"): local-LLM delivers at every decision boundary
  under the fixed runtime — 668/668 interventions on the full 200-file
  ReazonSpeech sample (gemma3:4b-it-qat), 138/138 on the operation-tagged
  30-file subset, 327/327 on the Reazon qwen2.5:1.5b-instruct full sample,
  and 324/324 on the full Lux 328-file corpus (gemma3:4b-it-qat).

## 0.3.0 (2026-11-12)

### Added
- Synthetic session-log generator (`scripts/synthesize_logs.py`):
  real-schema logs with a planted, recoverable reward structure for training
  the linear policy at scale (600-session corpus → ROC-AUC ≈ 0.70, close to
  the ≈ 0.73 Bayes ceiling of the generative model).
- End-to-end runtime simulator (`scripts/simulate_sessions.py`): generates
  Japanese clause sequences and feeds them through the real `SyncRuntime`
  (cycle predictor, rule decision, assignment, canned speaker), emitting
  analyzer-ready reports and JSONL logs. Its null arm is exactly +0.000;
  programmed effects of 0.2/0.4 are recovered with CIs excluding zero.
- Real-speech replay (`scripts/simulate_lux_replay.py`): plays the local Lux
  corpus back through the same `SyncRuntime`, emitting 656 analyzer-ready
  reports (328 CUE + 328 SILENCE) with zero uptake as expected for read
  speech. Confirms the live measurement path runs on real Japanese audio.
- One-command calibration benchmark (`scripts/run_synthetic_benchmark.sh`,
  documented in `docs/benchmark.md`): analyzer grid over effects × seeds ×
  backends (null cells exactly zero, 27/27 effect cells CI-excludes-zero), the
  end-to-end runtime simulation (null exactly zero; 0.2/0.4 effects recovered),
  plus policy training/evaluation at scale. `scripts/analyze_power.py`
  computes the paired design's power curve (16 seeds/cell; type-I stays at 0%;
  effect 0.1 detection hits 88% at 24 pairs and 100% at 48).
- `scripts/tune_ngram.py`: n-gram evaluation harness for any `metadata.csv`-shaped
  speech corpus. Used for calibration against the Lux Speech Corpus.
- `tools/file_asr.swift`: one-shot voicing recognizer using the same
  SFSpeechRecognizer the live adapter uses; benchmarks CUELOOP's production
  ASR engine per file when run on a license-accepted macOS machine.
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
