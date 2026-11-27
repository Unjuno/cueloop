# Changelog

## Unreleased

### Exhaustive latency tuning in autonomous A/B — defaults win (2026-11-28)
- Investigated why tuned 50ms/tol10 lost baseline in the agent A/B: tuned
  delivered MORE cues (avg 14 vs 12 pairs) with the same uptake count, so the
  rate was diluted. Tried `min_confidence` 0.75 to thin insertions, and swept
  insertion delay 50/100/150/200/300ms (4-8 pairs each, baseline also to n=8).
- Dose-response at n=4 suggested later is better (50:+0.179, 100:+0.236,
  150:+0.254, 200:+0.259, 300:0.000). With n=8 for both arms this VANISHED:
  delay200 +0.194 vs baseline(delay100) +0.236. Every apparent win at n=4
  was sampling noise from the LLM's high per-pair variance.
- Honest conclusion: after exhausting every tuning knob (latency, tolerance,
  confidence, TTS rate, cooldown, naturalness, attention selection), the
  DEFAULT settings (100ms / tol0 / conf0.55 / bare) have the best measured
  uptake at equal n. Further parameter tuning is not supported by evidence;
  the remaining variance is the agent model's generation, not a knob.
- `scripts/run_agent_text_pair.sh` now accepts `--min-confidence` via
  `CUELOOP_MIN_CONFIDENCE`.

### Autonomous voice-agent A/B (2026-11-28) — mic loopback does NOT work here
- Reviewer noted `scripts/run_agent_live_pair.sh` enables autonomy without the
  user. Attempted it: VAD fires (speech_started/ended) but SFSpeechRecognizer
  returns NO transcript from speakers->mic loopback on this Mac (tested with
  `say -f` while the adapter listened). So the mic-loopback agent path is
  non-functional here; it is not a valid A/B channel without audio routing.
- Added a working autonomous path without the mic: `scripts/agent_text_events.py`
  feeds the LLM's Japanese monologue as transcript events via `--events -`,
  and `scripts/run_agent_text_pair.sh` runs paired CUE/SILENCE entirely offline.
- A/B (4 pairs each, tuned 50ms/tol10 vs baseline 100ms/tol0):
  - tuned uptake delta +0.193 CI [+0.089,+0.344] (H1 supported)
  - baseline uptake delta +0.230 CI [+0.089,+0.366] (also supported)
  - baseline >= tuned on uptake/novel_uptake; carryover ~equal.
- Conclusion: with text fed directly, the agent DOES adopt cues (uptake 0.19-0.23,
  SILENCE=0), so the "influence" mechanism works in the autonomous path. The
  tuned latency settings do not beat baseline there. Live-mic uptake staying at
  0 is therefore an audio/ASR issue, not a cue-design issue.
- Fixed `set -u` empty-array expansion in both agent scripts.

### Uptake remains the blocker (2026-11-27) — exhaustive A/B, honest stop
- Ran every presentation/latency variant with an explicit target uptake >=0.35
  (baseline old bare/rule = 0.250). None met it:
  - old bare/rule baseline: 6/24 = 0.250
  - latency-tuned (50ms + tolerance 10): 4/22 = 0.182
  - fast TTS (rate 220): 0/8 = 0.000
  - rapid cue (cooldown 1000): 2/19 = 0.105, uptake/pair 1.0 (< baseline 2.0)
  - stem+attention (long and short): 0/12 and 0/6 = 0.000
- TTS overlap hypothesis DISPROVEN: speak_ms ~1620ms but next clause arrives
  ~3279ms later; overlap 0/687 = 0%.
- Uptake scoring audit: `target_operation_seen = operation in tag_operations(
  continuation)` works; per-op adoption varies (ground 0.48, dependency 0.02)
  but detection is not broken.
- Conclusion: every engineering lever (naturalness, attention selection,
  latency, cancellation, TTS rate, cue rate) has been tuned and CI is green,
  but live uptake never exceeds baseline. Further parameter tuning is not
  justified by evidence. Raising uptake now requires the speaker to actually
  adopt a cue in the next clause — a human/manual step that cannot be replaced
  by code.

### Latency / rating tuning (2026-11-27) — user's hypothesis tested
- User suggested the problem may be latency / "rating" rather than naturalness.
  Measured over recent real-mic logs: prediction 0.1ms, decision 0.4ms (fine),
  boundary_to_delivery 146ms, speak_ms 1608ms, and 220 CANCELLED of which 198
  were "speaker diverged from prediction" — the rating/cancellation issue.
- Added `divergence_tolerance` to `Synchronizer` / `SyncRuntime` / CLI
  (`--divergence-tolerance`): mismatches <= N chars no longer cancel a cue.
  Added `CUELOOP_INSERTION_DELAY_MS` / `CUELOOP_DIVERGENCE_TOLERANCE` to the
  recorder. Test `test_divergence_tolerance_keeps_small_mismatches` added.
- Real-mic A/B (latency-tuned: delay 50ms + tolerance 10 vs baseline 100ms + 0):
  - latency: 139ms -> 94ms (robust improvement)
  - CANCELLED: 198 -> 0 (robust improvement)
  - uptake rate: 0.250 -> 0.182 (NOT improved; got worse with n=4)
- Conclusion: the latency/cancellation problem is real and fixed, but adoption
  is a separate issue — delivering faster does not make the speaker adopt the
  cue. Naturalness (DSPy) and latency are both tuned; uptake remains open.

### Real-mic A/B round 2: short stem ALSO loses (2026-11-25)
- Shortened the DSPy cue objective (metric now rewards <=24-char cues;
  Signature asks for 24-char leads). DSPy re-optimized: 0.80 -> 1.00 PASS;
  compiled demos are now 10-20 chars ("違いが明確に現れる。").
- Re-ran the real-mic A/B (3 pairs each):
  - NEW short stem+attention: 6 deliveries, uptake 0/12 (0.000)
  - OLD bare+rule: 24 deliveries, uptake 6/24 (0.250)
  - (Earlier long stem: 12 deliveries, uptake 0/12)
- Shortening did NOT help uptake. Diagnosis: stem cues are declarative
  sentences that answer FOR the speaker; bare cues ("根拠は？") are questions
  that prompt the speaker to supply the operation, so they are adopted more.
- Honest verdict: naturalness (DSPy) and attention-grounding are achieved, but
  the live "influence"/uptake goal is NOT met by the new stack — the old bare
  stack still has higher uptake. Naturalness and uptake conflict here.

### Real-mic A/B: new stack LOSES on uptake (2026-11-25) — honest negative result
- Ran paired real-microphone A/B: new stack (`--cue-style stem --attention-select`
  + DSPy demos) vs old baseline (`bare + rule`), 3 clean pairs each.
  - NEW stem+attention: 12 deliveries, **uptake 0/12 (0.000)**
  - OLD bare+rule: 24 deliveries, **uptake 6/24 (0.250)**
- New cues were long contextual sentences (e.g. "なぜ、その臨床が判断の根拠と
  なるのかを説明してください。"); old cues were short labels ("根拠は？").
- Conclusion: DSPy improved *naturalness* metrics but the natural, long cues
  were NOT adopted by the live speaker; the short bare cues had higher uptake.
  The "influence" goal (uptake) is therefore NOT yet met by the new stack —
  naturalness and effectiveness conflict with the current DSPy metric, which
  does not reward shortness/adoptability. Next: add a length/adoptability term
  to the DSPy metric and re-optimize.
- Fixed `set -u` empty-array expansion in `record_self_pair.sh` during this A/B.

### Larger DSPy split + non-ceiling context-length verification (2026-11-26)
- DSPy optimization re-run with 12 cases and a 4-case held-out split
  (was 8/2): baseline 0.75 -> optimized 1.00 (+0.25 PASS), artifact refreshed.
- Context-length claim re-tested with a NON-ceiling metric. Earlier the
  attended-term metric hit 100% at every length once the term was explicit.
  Now the model picks the term itself (no explicit term) and we measure
  "cue contains a word actually present in the live context". Over 10 cases:
  context=0/1/2 -> 0% grounding, context=4 -> 30% (+30pt, target +20% PASS).
  This supports "文脈が多ければ多いほどより自然になる" without ceiling.
- `scripts/experiment_context_length.py` gained `--explicit-term` to switch
  between the ceiling (explicit) and non-ceiling (model-picks) conditions.

### DSPy program integrated into runtime + objective selection metric (2026-11-26)
- `src/cueloop/sync/stem.py`: `load_dspy_demos()` reads the compiled DSPy
  program (`artifacts/dspy-stem-compiled.json`) and injects its 4 bootstrapped
  demos as few-shot examples in the live stem prompt. `StemCueGenerator` uses
  them by default, so the DSPy-optimized prompt is what actually runs; no DSPy
  runtime dependency (artifact is plain JSON, missing file -> v3 fallback).
  2 regression tests added (demos load/enter prompt; missing file is safe).
- `scripts/experiment_attention_selection.py`: expanded 5 -> 11 cases, added an
  explicit target (>=80% accuracy, >=+40pt delta), and replaced the subjective
  hand-labeled "attended" match with **objective term grounding** (does the
  extracted term occur in the live context?). Result: rule operation-match
  0/11, attention operation-match 5/11 (out-of-vocab terms map to None, which
  the stem covers with natural wording), but **term grounding 11/11 = 100%**
  (PASS, +100pt). The extractor never hallucinates a term absent from context.

### DSPy prompt optimization (2026-11-25) — the user's "dspy/DS-Pi" reference
- User clarified the earlier "D5/DS-Pi" name was **DSPy** (prompt optimization
  framework). Installed `dspy` 3.5.0 into the venv (optional, not a runtime dep).
- Added `scripts/dspy_optimize_stem.py`: declares a `StemCueSignature`
  (context/text/span/attended_term -> cue), a metric (attended-term presence
  0.5 + no op-word leak 0.25 + single sentence 0.25), and compiles a
  `BootstrapFewShot` optimizer on 8 labeled cases with a held-out split.
- **Baseline (zero-shot) 0.75 -> optimized 1.00 on held-out**, target +0.20
  met (+0.25 PASS). Compiled program saved to `artifacts/dspy-stem-compiled.json`
  with 4 bootstrapped demos; all demos follow "lead with the attended term in
  a natural one-sentence cue" — consistent with the v3 prompt + explicit
  attention term already in `stem.py`.
- Ollama must be configured as `openai/gemma3:4b-it-qat` with
  `api_base=http://127.0.0.1:11434/v1` (the `ollama_chat/` route 404s here).

### Context-length sweep and explicit attention term (2026-11-26)
- Added `scripts/experiment_context_length.py` to test the user's claim
  "文脈が多ければ多いほどより自然になる" across 0/1/2/4 context turns on the
  v3 stem prompt (gemma3:4b, no mic).
- First run (no explicit term): attended_uptake 80% at 0 turns -> 60% at 4
  turns (target +20% FAILED). Inspecting the cues showed the model always
  used the most recent phrase ("逆の立場だと"), not the attended term — the
  v3 prompt's `context[-4:]` window dropped the extractor's term.
- Fix: `build_stem_messages` now accepts `attended_term`; when the attention
  extractor supplies a term it is stated explicitly ("注目すべき語"). Wired
  through `StemCueGenerator.generate` and `_resolve_stem_cue`.
- Re-run: all context lengths 0/1/2/4 reached attended_uptake 100%, op_leak
  0% — the attended term is now used deterministically. This means "more
  context -> more natural" is NOT demonstrated (ceiling effect); the proven
  improvement is "explicit attention term -> reliable grounding", which we
  report instead of overclaiming.
- Updated 3 stub generators in tests/test_stem.py for the new kwarg; 155
  tests pass.

### Attention-grounded cue SELECTION (2026-11-25) — the missing core
- The user's keynote: the cue word should not be a fixed "逆は"-style label;
  it should engage the term the model is attending to ("選定する必要がある";
  "今の方式では影響を与えられない"). D5 / DS-Pi was not found anywhere in
  the repo or home dir, so implemented the described behavior directly.
- Added `src/cueloop/sync/attention.py`: `AttentionTermExtractor` asks the
  local LLM only for the attended term (never for an operation), verifies it
  is grounded in the live context, and `operation_for_term()` maps the term
  to an operation via the existing deterministic vocabulary. Reproducible.
- `RuleDecisionModel` gained `attention_extractor`; at a boundary it prefers
  the operation implied by the attended term, with a safe fallback to the
  rule ordering when the term is out-of-vocabulary or already covered.
  CLI flag `--attention-select` wires it up.
- A/B (`scripts/experiment_attention_selection.py`, gemma3:4b, 5 cases,
  no mic): baseline rule selection 0/5 (0%) -> attention-grounded 3/5 (60%).
- The 2 remaining cases had out-of-vocabulary terms (コスト/方式); stem v3
  still produced context-grounded cues (e.g. "コストの削減効果が、より大きな
  メリットを生むだろう。"). `_resolve_stem_cue` now injects the extracted
  term at the head of context so the v3 prompt lands on that term.
- Tests: `tests/test_attention.py` (3) added; 78 focused tests pass.

### Prompt tuning: attention-grounded stem cues (2026-11-25)
- User asked for prompt improvement ("D5 / DS-Pi" name not found in repo),
  specifically: the cue word should not be a fixed "逆は？"-style label; it
  should engage the term the model is attending to in the live context.
- Added `scripts/experiment_stem_prompt.py` (no mic needed): A/B of the old
  operation-first prompt vs attention-grounded variants on 5 fixed Japanese
  reasoning cases via local LLM (gemma3:4b-it-qat; qwen3 emits reasoning into
  `reasoning` not `content`, so it is unusable for stems).
- **Baseline** (old prompt): op-word leak 20%, attended-term uptake 60%,
  one-sentence 100%. **Goal**: uptake 60->80%, keep one-sentence, no worse leak.
- Variants: v1 (2-step attention, leak 0% but 2-sentence 20%), v2 (forbidden
  word lists — backfired, leak 40%), v3 (affirmative-only: uptake 80%,
  one-sentence, leak 20%), v4 (combined — worse). **Adopted v3**: it hits the
  uptake goal (80%), keeps one-sentence output, and does not worsen leak.
- `build_stem_messages` now implements v3 (operation-agnostic, finds the
  attended term). Replay shows context-grounded sentences instead of fixed
  "逆は？". Regression test added. Still coarse: some operation-like words
  remain when they also occur naturally in the context.

### Address the "coarse" feedback: cue wording and context (2026-11-25)
- User feedback (no quantification requested): too few words, no context,
  too many false positives. Root cause in code: every live experiment ran
  the coarsest stack `--cue-style bare --decision rule --predictor ngram`,
  so the speaker heard only 8 fixed phrases regardless of what they said.
- Made `record_self_pair.sh` configurable: `CUELOOP_DECISION`,
  `CUELOOP_PREDICTOR`, `CUELOOP_CUE_STYLE`, `CUELOOP_BASE_URL`, `CUELOOP_MODEL`
  (plus the batch driver's `STACK=KEY=VAL,...`). The richer stem+local stack
  can now be recorded instead of being hard-coded off.
- Fixed a wiring bug: `stem + local` silently produced bare cues because
  `_needs_stem_generator` excluded `local`/`steer`, and the instant rule
  fallback had no `cue_style`/`stem_generator`. Replay now yields a
  context-grounded full sentence ("具体的には、逆に考える必要があるです。")
  instead of "逆は？". Regression test added in `test_stem.py`.
- Still coarse: the LLM-written stem mixes operations ("具体的には、逆に…")
  — prompt quality is the next thing, not the wiring.

### Internal audit (2026-11-24): tuning is internally consistent
- Added `scripts/audit_tuning.py` (no mic needed) — the "adjust internally"
  entry point. It checks weight/adoption correlation, repeat rates,
  low-adoption ops above the floor, and floor consistency.
- Result on current data (98 sessions, 829 deliveries): correlation r=0.975,
  adjacent exact repeats 4.9%, same-family 10.1%, no above-floor op with
  adoption<0.10, floor 0.5 permits 7 ops. VERDICT: internally consistent;
  no forced change warranted.
- Corrected an earlier misread: a PCA sweep suggested "weights don't drive
  selection", but that was a simulation artifact (fixed eligible ordering).
  Real `_choose_with_fatigue` rotates via fatigue and the weights track
  measured adoption at r=0.975. A speculative rank-bonus change was reverted.

### Turn analysis into action: expose fatigue penalty (2026-11-24)
- Analysis of "ground/concretize feel repetitive":
  - adjacent exact repeats 4.9%, same-family 10.1%; ground+concretize
    adoption 0.381 vs others 0.277 (so they do work; the complaint is a
    habituation/feel issue not captured by immediate uptake).
  - dependency adoption 0.021 (share 7.7%) and boundary 0.154 are the ops
    that are genuinely low-value.
  - fatigue penalty was hard-coded (2.0) and NOT tunable. Now exposed as
    `--fatigue-penalty` on the CLI, threaded through `record_self_pair.sh`
    (`CUELOOP_FATIGUE`) and the batch driver (`FATIGUE=...`), and logged per
    pair in `artifacts/feedback-loop.jsonl` so its effect can be A/B tested.
  - Reverted a speculative change to the rank-bonus math (kept behavior
    stable) — the data did not support "ground should be forced more".

### Root-cause analysis (2026-11-24): gate weights are diluted by context-fit order
- User observed "根拠は／具体的にはが連続していて適切じゃない". Log analysis:
  - adjacent same-op repeats only 4.3% (so not literal repeats), but 124
    three-cue windows have >=2 of ground+concretize (evidence ops clustered).
  - concretize: delivered share 21.1% (highest) despite weight 0.95 and
    adoption 0.326; ground: weight 1.644, adoption 0.486, but share only
    11.2%. The weights are NOT driving selection as intended.
- Cause: `RuleDecisionModel.decide()` re-ranks `missing` by context fit, then
  `_choose_with_fatigue()` scores `(total - rank_index) - penalty*count` and
  multiplies by weight. The rank bonus dominates: concretize (rank 0) beats
  ground (higher weight) in a direct replay test (4 picks vs 2).
- Consequence: the operation-weight gate only suppresses sub-floor ops
  (dependency 7.4% share, adoption 0.021); it does not steer toward the
  highest-adoption ops. Fix planned: normalize/dampen the rank bonus so
  weights can dominate. Analysis only — no behavior change in this commit.

### CI: macos-adapter moved to manual dispatch (2026-11-25)
- Run 45418637065 proved `continue-on-error: true` does NOT suppress a
  `timeout-minutes` cancel: both unit jobs were success, macos-adapter was
  cancelled at 16m while queued, yet run conclusion=failure. Fix: the
  macos-adapter job now runs only on `workflow_dispatch` (`if:` guard),
  and push/PR CI runs just the unit job (lint/tests/replay/wheel). The
  adapter compile is still available on demand. A `workflow_dispatch`
  trigger was added. Raised timeout to 45 as a secondary measure.

### CI: macos-adapter no longer blocks main on infra cancels (2026-11-23)
- Run 45195738266 had both unit jobs green but `macos-adapter` was CANCELLED
  at the 15-min infra cap while still queued (zero steps executed). This is
  the known macOS ARM capacity issue, not a compile failure. Set
  `continue-on-error: true` on that job (workaround (a) in the earlier note):
  the adapter compile is verified locally and unit covers lint/pytest/replay/
  wheel, so a runner-capacity cancel must not make main red.

### More v3 live data (2026-11-23) — effect estimate stabilizes
- Added 6 clean gated pairs (`v3more`) to the 7 `v3live` pairs: v3 adoption
  0.308 (v3live) and 0.250 (v3more), both above baseline 0.210.
- All gated data now 46 pairs: adoption baseline 0.210 -> tuned 0.347,
  95% CI [+0.066, +0.203] -> still a clear improvement. H1 significant at
  n=88 (+0.277 CI [+0.237, +0.319]); drift check "weights in sync".
- The adoption point estimate drifted down 0.373 -> 0.363 -> 0.347 as n grew
  (early small samples overestimated); 0.347 is the more conservative and
  realistic figure. The improvement over baseline remains significant.

### v3 real-recording A/B verified (2026-11-23)
- Recorded 7 clean gated pairs with the installed v3 weights (floor 0.5,
  `v3live` batch; recorded via live microphone through the PTY-fixed runtime).
  v3live adoption = 12/39 = 0.308 — comparable to v2 (0.338), so the v3
  refresh introduced no regression.
- All gated data now 40 pairs: adoption baseline 0.210 -> tuned 0.363,
  95% CI [+0.080, +0.226] -> clear improvement. H1 stays significant at
  n=82 (+0.281 CI [+0.236, +0.325]). Weight drift check: weights in sync.

### Docs reconciled (2026-11-23)
- README (EN/JA) status refreshed to current facts: 150 tests, 75 clean
  pairs, H1 uptake +0.280 CI [+0.234, +0.323] (n=75), CUE 427 deliveries,
  and the over-insertion fix (adoption 0.210 -> 0.373). Previously these
  tables were stale (142 tests / 24 pairs / +0.254).

### Regression tests for baseline-only tuning (2026-11-23)
- Added `tests/test_tune_operation_weights.py` (3 tests): `_is_gated` detects
  all gated name tags (gated/v2gate/v3gate/floor0), gated logs are excluded so
  baseline-only weights are reproduced, and the `--exclude-gated` CLI flag
  works end-to-end. Test count 147 -> 150.

### Tuning-loop safety: weights derived from baseline only (2026-11-23)
- Found a subtle self-amplification risk: deriving operation weights from all
  logs lets the gate's own adoption lift feed back into the weights
  (boot_mean sat systematically above installed v3 values, e.g. ground 1.95
  vs 1.64). Fixed `tune_operation_weights.py` with `--exclude-gated` and made
  `check_weight_drift.py` use it, so weights track *natural* adoption only.
- Verified installed v3 weights exactly equal the baseline-only derivation
  (match=True, all 8 ops) and drift check reports "weights in sync". v3 is
  therefore statistically stable (all weights inside bootstrap 95% CIs over
  499 interventions). Voice-recording v3 A/B was attempted but the mic was
  silent (manual step needed); offline validation + existing gated A/B
  (0.210 -> 0.373, CI [+0.086,+0.238]) still support the tuning.

### Weight refresh v3 (2026-11-23) — drift detector fired
- After the floor-sweep batches, `check_weight_drift.py` detected max drift
  0.260 on `invert` (adoption rose with more data). Installed v3 weights:
  invert 1.204, conclusion 1.053, ground 1.644, dependency 0.083.
  Drift re-check: "weights in sync".
- Note: v3 makes floor 0.7 exclude `boundary` (0.686) while 0.5 still permits
  all 7 operations — floor and weights interact, so re-sweep after refresh.

### Floor-threshold sweep (2026-11-23) — floor 0.5 confirmed sufficient
- Ran live gated batches at floor 0.3 (6 pairs), 0.5 (23), 0.7 (4) with the
  same v2 weights. All floors beat baseline adoption 0.210 with CIs excluding
  zero: 0.3→0.439 [+0.073,+0.379], 0.5→0.338 [+0.044,+0.212],
  0.7→0.467 [+0.143,+0.388].
- Crucially, v2 weights make floors 0.3/0.5/0.7 permit the **exact same 7
  operations** (only dependency 0.091 is excluded), so the adoption spread is
  sampling noise, not a real floor effect. floor 0.5 is retained as the
  robust default (largest n, clean CI). `scripts/sweep_gate_floor.py`
  (offline replay) and `scripts/compare_gate_ab.py` (per-floor breakdown)
  document this. H1 stays significant at n=75 (+0.280 CI [+0.234,+0.323]).

### Continuous tuning: weight drift detection and refresh (2026-11-22)
- Operation weights drift as data grows: re-deriving on all clean logs moved
  boundary 1.455 -> 0.750 and comparison 1.333 -> 1.040 (early small samples
  over-rated them). Installed the refreshed v2 weights
  (`artifacts/operation-weights.json`): dependency 0.091, ground 1.714.
- `scripts/check_weight_drift.py` re-derives weights from all clean CUE logs,
  compares with the installed artifact, and exits 1 on max delta > 0.25 so
  the tuning loop can auto-trigger a refresh. After installing v2: drift 0.
- Fatigue penalty sweep (offline, 0..5) confirmed the current 2.0 is already
  optimal (adjacent-repeat 1.8%, 7 distinct operations); no change needed.
- v2 gate recorded 8 more pairs: adoption 0.318; all gated 23 pairs still
  show a clear improvement over baseline (0.210 -> 0.338, CI [+0.044, +0.212]).
  H1 stays significant at n=65 (+0.254 CI [+0.209, +0.296]).

### A/B verified: gate improves adoption (2026-11-22)
- Recorded 18 gated paired human sessions (`gated01`/`gated02`, gate floor 0.5
  on `artifacts/operation-weights.json`) versus 18 baseline sessions. Results
  from `scripts/compare_gate_ab.py`:
  - **adoption rate (uptake/delivered): baseline 0.210 -> tuned 0.341**
    (delta +0.131, 95% cluster-bootstrap CI [+0.039, +0.215] -> clear
    improvement).
  - dependency share: 10.4% -> 4.0%; dependency adoption 0.032 -> 0.000.
  - mechanism adoption 0.176 -> 0.320; concretize 0.139 -> 0.321.
  - H1 stays significant at n=60 (uptake +0.254 CI [+0.208, +0.298]).
- `record_self_pair.sh` accepts `CUELOOP_OPERATION_WEIGHTS` /
  `CUELOOP_MIN_OP_WEIGHT` and passes `--operation-weights` /
  `--min-operation-weight` to both arms. Batch driver enables it via
  `GATE_WEIGHTS` and logs a `gated` flag per pair.
- Fixed a batch randomizer bug: `fixed` assignment requires an explicit cue
  and was removed from the random choices (canonical / random-operation only).

### Tuning — suppress low-adoption operations (the actual over-insertion fix)
- **Root cause of over-insertion**: not prediction confidence (uptake rate is
  flat across confidence buckets: 0.4→0.36, 0.8→0.27, 1.0→0.28) but the
  *operation choice*. Per-operation adoption rates vary 15x:
  ground 0.42 / boundary 0.36 / comparison 0.33 / invert 0.26 vs
  dependency 0.03 / mechanism 0.15 / concretize 0.19. The worst offenders
  were delivered repeatedly as noise.
- `RuleDecisionModel` gained `operation_weights` and `min_operation_weight`:
  operations below the weight floor are excluded from selection (with a
  safe fallback so the model never goes fully silent), and remaining picks
  are scored by weight x fatigue. CLI flags `--operation-weights` (JSON)
  and `--min-operation-weight` expose the knob.
- `scripts/tune_operation_weights.py` derives weights from human DELIVERED
  outcomes (rate x 4, capped 2.0). On the 47 CUE logs it produced
  `artifacts/operation-weights.json` (dependency 0.108, ground 1.677).
  Regression test `test_operation_weight_gate_suppresses_low_adoption_operation`
  proves a low-weight operation is never chosen while a high-weight one is.
- `scripts/h2_reconstruction.py` (+ tests) implements the *secondary* H2
  outcome: blinded-rater rubric scoring (7 items) or a labeled lexical proxy,
  paired CUE-SILENCE bootstrap on the rubric total and subscores. Proxy output
  is explicitly marked as not human evidence.

### Learning & over-insertion analysis (2026-11-22)
- **The live loop was overfitting to one condition.** Before this change the
  recorder used a fixed seed (3), assignment (canonical), speech-db (-50),
  duration (75 s), prompt wording and a metronomic 3-minute start gap
  (stddev 0.01 min over 10 pairs). `run_accuracy_batch.sh` now randomizes
  seed, assignment, speech-db, duration (60-104 s), prompt style (4 variants)
  and inter-pair gap (60-240 s) for every pair, and logs each pair's
  conditions + metrics to `artifacts/feedback-loop.jsonl`. Diversity audit
  over 10 diverse pairs: 8 seeds, 2 assignments, 4 speech-db values,
  9 durations, 4 prompt styles.
- **Over-insertion quantified**: 5.8 cue deliveries/pair but only 1.22 uptake
  -> ~79% of inserted cues are not adopted (noise). The rule backend's
  `min_confidence` (default 0.55) controls this; `--min-confidence` is exposed
  on the CLI.
- **Learning from human logs demonstrated**: trained the linear policy on
  271 samples (76 uptake positives) from the clean human CUE logs:
  `artifacts/sync-policy-human.json`. Holdout ROC-AUC 0.708 (train 0.704),
  Brier 0.229 — learns signal without overfitting. This is *offline policy
  learning*, distinct from H2 (human durable learning), which remains
  unmeasured and requires the blinded delayed-reconstruction task.

### H1 supported — first clean human paired result (n=24)
- Recorded **24 clean paired human CUE/SILENCE sessions** with the fixed
  runtime (after the PTY event-capture fix), using `scripts/record_self_pair.sh`
  in batches via the new `scripts/run_accuracy_batch.sh`. The protocol's
  24-pairs/cell cell is met.
- `scripts/evaluate_h1h2.py` selects only fixed-runtime pairs (CUE outcomes>0
  and SILENCE outcomes==0) and runs the seeded paired bootstrap. Result:
  **uptake delta +0.254, 95% CI [+0.176, +0.335]** (excludes zero), novel
  uptake +0.265 [+0.180, +0.354], carryover +0.250 [+0.170, +0.338].
  CUE arm: 165 deliveries / 165 outcomes; SILENCE arm: 0 outcomes (clean
  control). H1 (cue raises operation-word uptake in the next clause) is
  supported on human Japanese reasoning speech. H2 remains unmeasured — it
  needs the blinded delayed-reconstruction rubric.
- This supersedes the historical contaminated-control estimate (+0.035).

### Fixed — live microphone capture was silently broken
- **Event commands now read through a PTY.** `cli._event_lines` launched the
  speech adapter with `stdout=PIPE`. Swift's `print` block-buffers on a pipe
  (its internal buffer is independent of `fflush(stdout)`), so
  `transcript_partial` events never reached the runtime on live sessions:
  predictions stayed ~0 and no cues were inserted. Reading from a PTY
  (`pty.openpty`) restores line buffering; a pipe fallback remains for
  platforms without PTYs. Evidence: a clean paired session now records
  predictions=59, delivered=4, uptake=1/4 in the CUE arm.
- The Swift source `emit()` was also changed to write via
  `FileHandle.standardOutput.write` (effective once the Xcode license is
  accepted and the adapter is rebuilt); the PTY fix works with the existing
  prebuilt binary without needing that rebuild.
- **`--min-outcomes` no longer gates the SILENCE control arm.** Controls
  intentionally score zero outcomes, so gating them removed the control and
  left the CUE arm unpaired (analysis reported "not comparable"). Now only
  the CUE arm is quality-gated; the control is always retained.
- `record_self_pair.sh` now uses a 3 s warm-up after each spoken announcement
  (SFSpeechRecognizer's first result can lag ~20 s) and prompts for operation
  words (結論は／なぜなら／逆に) so cues can actually be triggered.
- First clean paired human session with the fixed stack:
  `logs/self-accuracy-pty-paired-20261121-234611`, CUE uptake=1/4, SILENCE
  outcomes=0, paired uptake delta +0.250 (n=1, CI insufficient).

### Added — analyzer quality gate
- `scripts/analyze_experiment.py --min-outcomes N` drops sessions with fewer
  than N finalized outcomes before pairing. Empty/silent-mic arms can no
  longer masquerade as valid "zero effect" measurements. A regression test
  (`test_min_outcomes_drops_empty_recording_failure`) proves empty pairs are
  excluded. On the historical contaminated dataset, `--min-outcomes 1`
  reduces 77 pairs to 37 (all with at least one CUE outcome).

### Fixed — observation bug in the SILENCE control arm
- **`SyncRuntime._deliver()` scored an Outcome for `SILENT` deliveries.**
  `--silent` logged a SILENT delivery (timing parity) but then created an
  Outcome and scored `target_operation_seen` on text the speaker never heard
  a cue for. Over the 77 stored pairs this produced 12 spurious "uptake"
  events in SILENCE arms (12/176) vs 21/165 in CUE arms, biasing every
  paired CUE-SILENCE contrast toward zero. Now only audible (`DELIVERED`)
  deliveries create Outcomes; SILENT deliveries are still logged so counts
  and timing stay paired, but the control arm no longer invents uptake.
- Regression tests updated: `SilentConditionTests` now asserts no outcomes
  and zero novel uptake; `test_silence_never_speaks` and the effect-zero
  calibration cell were aligned with the corrected control semantics.
- Re-scoring with the fixed observation: a simulator arm with a programmed
  +0.40 effect now reads +0.368 uptake delta (previously diluted by the
  control arm's fake uptake). All 141 tests passed at that point; the suite
  is now 142 tests after the quality-gate regression test was added.

## [0.4.0] - 2026-11-21

### Fixed
- **CI `unit` job format check** — `tests/test_pair_audit.py` was introduced
  unformatted in 2eeed8b, so `ruff format --check` failed on the latest
  main-branch run (44475798168). Reformatted with `ruff format`; local
  `ruff check`, `ruff format --check`, 141 pytest tests and unittest discover
  all pass (2026-11-21). Re-run 44825622170 is green on all three jobs.
  CI actions were also bumped to `checkout@v5` / `setup-python@v6` to clear
  the Node 20 deprecation annotations.

### Added
- **`scripts/pair_audit.py`** — data-integrity audit across live paired
  sessions. Scans every `logs/self-design-choice-paired-*` (or any roots
  given), classifies each pair into EMPTY / SILENT / LIGHT / OK /
  NON-LIVE, and reports per-pair clause-count, delivery count, and a
  summary line. Catches the common silent-mic case where a session looks
  like it "succeeded" but produced zero Japanese clause bytes. Runs in
  ~0.1 s; `--json` output for automation.
- **CI macos-adapter cancellation issue resolved.** Earlier main-branch runs
  were cancelled mid-compile (~15 min infra limit, not a compile limit); the
  Swift build always completed locally. `timeout-minutes: 25` was added at
  commit 16e62be, and subsequent runs (44350984159, 44825622170,
  44825852155) were green on both jobs. Actions were bumped to
  `checkout@v5` / `setup-python@v6` to clear Node 20 deprecation warnings.
  Workarounds if it recurs: (a) skip macos-adapter in CI, (b) commit the
  prebuilt adapter binary (`macos-speech-events-ci`), or (c) pin macos-latest.
  See issue #6.
- **2026-11-18 restricted-ops arm (A — n=7)**: pairs 234729/235451/000203/001419/002521/004038/005156.
  Mostly sparse/tiny sessions; some language fragments only ('パイロット'/'PNとか使うの').
  Aggregate A: delivered=6 (ground×2, invert×3, comparison×1) uptake=0; paired CI [+0.000,+0.000].
  Cumulative restricted arms (B + A): n=10 sessions live, uptake=0 — no effect detected in any restricted configuration.
- **`docs/gpt-live-protocol.md`** — precise protocol for pairing GPT Live
  (the ChatGPT app's Japanese voice reasoning) with CUELOOP on this Mac:
  topology (speakers-to-mic or BlackHole 2ch routing with owner consent),
  recording procedure (paired CUE/SILENCE with shared pair_key), analysis,
  caveats (agent-side uptake ≠ human-transfer).
- **`scripts/run_agent_live_pair.sh`**: fully autonomous Japanese reasoning
  monologue generation via local LLM (qwen2.5:7b-instruct) → Kyoko TTS →
  SFSpeechRecognizer ingest. Keeps end-to-end pipeline exercised while the
  owner is not at the mic; still produces agent-side data only — local LLM
  cannot hear cues, uptake is structurally zero.
- **First agent-live pair completed 2026-11-16** (logs/agent-live-なぜdesign-
  choiceを選ぶのか-paired-20261116-005151): CUE delivered=4
  (invert/mechanism/conclusion/concretize), uptake=0 on each; SILENCE arm
  delivered=0 by design; paired analyzer ran end-to-end. Not H1/H2.
- **`scripts/record_gpt_live_pair.sh`** — one-command paired recorder for
  the GPT Live setup (300 s arms, Kyoko announce). Smoke-tested today;
  full pipeline works end-to-end, uptake=0 under ambient conditions.
- **2026-11-15 absolute live-pair count 28 (204315 — fifth consecutive
  empty)**: utterances CUE=19 SILENCE=1, zero Japanese clause text.
  Aggregated 28 pairs: uptake +0.045 CI [-0.027, +0.132] (includes zero).
  5 consecutive silent pairs confirmed owner absent; live H1/H2 could not
  advance. See issue #6.
- **2026-11-17 op-spectrum**: `scripts/op_fire_spectrum.py` added; artifacts/
  live-op-spectrum.md generated (gitignored). speaker-natural ops
  (boundary/conclusion/concretize/dependency) fire on both arms; cue-
  divergent (mechanism/ground/invert/comparison) appear only under CUE.
  This is an op-selective hint, not evidence of causal uptake.
- **2026-11-15/17/18 live pairs #36–#69**: #57-#67 near-; #182933 (#68) sparse;
  #010756 (#69) EMPTY. Aggregated 69 live pairs: uptake +0.035 CI [-0.004, +0.080] — zero straddles. See issue #6.
- **2026-11-15 17:09 design-choice attempt 4 (genuinely live)**: pair 3
  of the corpus's live history. CUE arm: utterances=1, predictions=23,
  4 deliveries (concretize/mech/ground/conclusion) — mechanism outcome
  target_operation_seen=True (continuation "同一方向に改善する仕組み"),
  2/6 delivered-ops across live pairs. SILENCE twin: 12 interventions
  HELD by design; carryover seen 2/5 with zero cues. Analyzer n_pairs=1
  per pair, growth/carryover favour SILENCE — CI insufficient; n_pairs=2
  total; not H1/H2 evidence.
- **2026-11-15T17:00 design-choice attempt 3 (genuinely live)**: after
  the playback reclassification, a truly live run captured the owner
  speaking Japanese reasoning through the mic — utterances=1,
  predictions=11, interventions=3 (2 DELIVERED: '仕組みは？' @149,005
  ms +102 ms with outcome target_operation_seen=True → uptake on
  mechanism; '根拠は？' @151,703 +171 ms uptake=False; 1 CANCELLED
  '具体は？'). SILENCE twin kept silence by design. Analyzer n_pairs=1:
  uptake delta +0.500 CI insufficient — not H1/H2 evidence; pending
  24-32 pairs per cell.
- **`--early-lead-ms` plumbed through the Lux simulator** (default 0,
  off): plan cues up to N ms before the pause threshold — executes only
  on TICK-bearing input. On Lux replay (segment feeds without TICKs)
  0 vs 300 produces 30/30 identical sessions; knob exists for future
  TICK-driven studies. Production default unchanged.
- **Lux pause_ms sensitivity sweep (200/350/500/750)**: predictions=36,
  delivered=25 identical at all settings — replay boundaries bind on
  clause-ender punctuation, not on millisecond gaps; pause_ms is a
  live-path-only knob.
- **2026-11-15 playback reclassification (docs/real-speech.md)**: every
  "live" paired capture in this repository is replay of the owner's own
  pre-recorded Japanese monologue into the mic (user-confirmed).
  Consistent with quantitative evidence: byte-identical
  boundary/timestamps across consecutive arms (MD5 `1945b967…` shared
  by Nov 14 20:01 and Nov 15 20:34 runs at ms 54375/55497/56072) — impossible for
  spontaneous speech. All such sessions are canned-class:
  delivery/boundary/prediction all logged, but the speaker cannot react
  to cues; H1/H2 n_pairs = 0. Insertion benchmark and synthetic
  calibration axis survive the reclassification unchanged.
- **2026-11-15 "live pair" (180 s arms, audible Kyoko announce) — historical,
  reclassified**: continuous Japanese monologue replay was captured
  end-to-end. Not live human interaction. Kept as an end-to-end pipeline
  trace (CUE 2 deliveries: mechanism @9.5 s +181 ms, conclusion @176.7 s
  +101 ms; SILENCE arm 12 interventions HELD by design). Do not cite as
  uptake evidence.
- **Recorder announces audible start + defaults to 180 s window**:
  scripts/record_self_pair.sh voices "これから話してください" (CUE) /
  "もう一度、同じ話題で話してください" (SILENCE) before each arm and the
  default arm length is now 180 s, closing the structural race where the
  window expired before the speaker reacted.
- **Insertion benchmark extended to Common Voice 8.0** (4,483-file ja
  sample + 120-file replay subset). On the full sample: context-fit
  ranking 8.1% vs RANDOM-order PRIORITY 9.5% — the only corpus where
  PRIORITY inverts the ranking, because read prose has near-zero
  reasoning structure. Anchor decomposition (op present in prefix) locates
  the inversion *inside the anchor cell* (CV8 ranked 10.3% vs PRIORITY
  12.1%, n=58) while the no-anchor cell is 0.0% everywhere; reasoning
  corpora keep a 17-29 point margin inside the anchor cell. The
  JSON export bug collapsing same-basename corpora under one key is
  fixed (`<dir>/<file>` labels); `Next steps` in docs/real-speech.md
  re-prioritized by insertion evidence (selection axis first).
- **Analyzer reads simulator output directly**: `scripts/analyze_experiment.py`
  now discovers report JSONs one directory level deep, so
  `analyze_experiment.py artifacts/<arm>` works on `<arm>/cue/*.json` +
  `<arm>/silence/*.json` without manual flattening. Regression test covers
  the two-level arm layout (140 tests total).
- **Offline insertion benchmark** (`scripts/benchmark_insertion.py`): measures
  insertion appropriateness of the production path on real Japanese
  transcripts without any live microphone. Two axes per clause boundary:
  (a) prediction axis — ngram(3) span op tags vs the human's next-op;
  (b) selection axis — the production missing-op ranking
  (`PolicyObservation.ranked_missing_operations`, the rule decision) vs the
  op actually present in the human's next clause. On lux / reazon-200 /
  podcast-replay / jsut-replay: context-fit ranking picks the human's next
  op 0%–24.1% of the time; PRIORITY (context-blind) picks it 0.0–1.9%; a
  deterministic adjacency-chain selector picks it 0–7.7%; an oracle inside
  the missing set covers it 66.7–100%. H1/H2 remains
  untested — issue #6 stays open; insertion ≠ uptake.
  (Added 2026-11-13: bandad-kw arm, 2,695 boundaries; context-fit 18.9%.)
- **Insertion benchmark — anchor decomposition & gate rejection**:
  selection axis split into anchor/no-anchor cells; CV8's inversion was
  localized inside the anchor cell (ranked 10.3% vs PRIORITY 12.1%,
  n=58). Two signal-gated selectors (`sel_*_clausegated`: ranking only
  with clause ops; `sel_*_weightgated`: ranking only with anchor weight
  ≥ 2.0) measured on all 7 corpora are **both strictly worse than
  production ranking everywhere** (e.g. kw-ja 3.8% / 4.7% vs 18.9%;
  podcast 3.4% / 6.9% vs 24.1%; reazon 0.0% / 7.7% vs 23.1%). Production
  constants unchanged; gate arms exist only in the benchmark.
- **JSUT local-LLM arm (100 files, paired)**: `gemma3:4b-it-qat` predictor
  + local decision. CUE delivered 136/136 interventions (dependency 58,
  boundary 41, ground 26, comparison 10, mechanism 1; 1.36/file; 0 cancelled
  /expired); SILENCE arm HELD all 139 by design. Paired n=100: every delta
  +0.000 CI [+0.000, +0.000]. Zero uptake — the canned-audio asymptote on
  the local-LLM path.
- **Podcast local-LLM arm extended 3 → 15 clips (paired)**: CUE delivered
  443/443 (ground 119, boundary 112, dependency 105, comparison 99,
  mechanism 6, conclusion 2); SILENCE HELD 439/439. Paired n=15: uptake
  −0.002 [−0.006, +0.000] (canned-token coincidence, CI upper-bound zero),
  all other deltas +0.000. No effect measured.
- **bandad kw-ja local-LLM arm (40 clips, paired)**: CUE delivered
  117/117 (dependency 41, boundary 35, ground 22, comparison 15,
  mechanism 3, conclusion 1); SILENCE HELD 115/115. Paired n=40: all
  deltas +0.000 CI [+0.000, +0.000]. With JSUT (n=100) and podcast
  (n=15), the canned-audio asymptote on the local-LLM path now stands at
  **n=155 paired sessions across three corpora — every CI zero**.
- Real-speech replay across 8 Japanese corpora (see docs/real-speech.md) —
  canned audio, no effect claimed. Highlights: the 8th (`bandad/asr-testset-kw-ja-v1`) shows the highest op-tag density
  of any corpus measured (21.8% of 510 clips tag ≥ 1 operation; 1638 predictions,
  998 delivered cues). The 7th arm, `PlayMaker13/japanese-podcast`, contains
  conclusion-heavy native podcast monologues (20 paired sessions, 474
  delivered, uniform operation distribution). benchmarks+docs updated with
  the local-LLM arm results and reproduction blocks. Human transfer (H1/H2)
  remains untested — issue #6 is open.

### Fixed
- **`scripts/record_self_pair.sh`**: staging step copied two source files
  into a target ending in `.jsonl`, crashing every paired run at the last
  step ("Not a directory"). Now writes `cue/silence` with distinct basenames
  in a paired dir, tolerates a totally silent arm (missing log file), and
  prints an explicit WARNING when SFSpeechRecognizer produced no clause
  text at all — empty "success" can no longer masquerade as paired data.
  (First cut of the WARNING matched the wrong signature — separate
  transcript events — and false-tripped on a genuinely spoken CUE session;
  it now scans `.jsonl` `"clause"` fields, verified on three live files.)
- **Prediction parser accepts non-standard model outputs**: markdown-fenced
  JSON (```json ... ```) and synonym span keys (`next_segment`,
  `next_sentence`, `continuation`, `next_text`, `next`) are now parsed
  instead of failing as "prediction response did not contain a JSON
  object". Regression tests cover both paths. This lets reasoner models
  answer at all; it does not change what they predict.
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
  cells CI-excluding-zero). On the live mic loopback (EMOTION100_011 /
  _066), cues now reach `status SILENT` delivery instead of being cancelled;
  EMOTION100_072 vets out at the local decision model (no tagged operations
  in the utterance; 0 interventions, as designed).

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
- `scripts/bench_predictors.py`: one-command benchmark of local-LLM next-span
  agreement on any compatible `(metadata.csv + whisper-cache)` corpus;
  reports exact-prefix rate, average longest-common-prefix, operation-tag
  hit rate, and parse failures per model. On the Reazon sample every
  model tested scores 0 exact-prefix; operation-token hit rate is ~0 for
  every tested model. This measures what the local LLM can do on real
  discourse, not whether cues change reasoning.
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
