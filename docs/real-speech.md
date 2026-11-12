# Real Speech Corpus Evaluation

CUELOOP's live path depends on (a) the macOS microphone adapter for ASR and
(b) the n-gram predictor for low-latency prediction with the local LLM as
semantic prior. This document records what we found when we put publicly
available Japanese speech corpora through that pipeline.

**Corpora.**

- *Lux Japanese Speech Corpus* (Lami Inc., CC-BY-4.0) — the `ita_recitation`
  clean read subset: 324 files, ~230 MB, 96 kHz/16-bit WAV. Read prose;
  no CUELOOP operation-vocabulary overlap. Used for the first real-speech arm.
- *ReazonSpeech* (Reazon Holdings, CDLA-Sharing-1.0; mirror
  `osanpo/reazonspeech`) — YouTube-derived Japanese speech (news, sports,
  talk, cooking — real speaker discourse). `metadata-sample-200.csv` draws the
  200 longest/most-content-dense utterances with at least 6 s and 15
  characters from chunk `000` (matches `efwkjn/reazonspeech_metadata` for
  transcriptions; YouTube-derived; audio is not redistributed in this repo —
  `data/` is git-ignored). Average content density ~4.5× the Lux set; 30/200
  lines tag at least one CUELOOP operation. Used for the vocabulary-domain
  arm.

These corpora are real recorded Japanese speech but they are not CUELOOP's
live target either: neither contains human-against-cue interaction (canned
audio cannot respond). They remain honest calibration targets, just ordered by
proximity to the operational discourse register.

## What was measured

### 1. ASR quality on the corpus (whisper-base-mlx, not the production path)

`mlx-community/whisper-base-mlx` (multilingual base checkpoint, untuned)
transcribed all 324 files. Character accuracy vs the ground truth:

```text
file count:        324
total characters:  ~
character accuracy: 73.8%
```

Errors concentrate in two clusters:

- **Unknown names / proper-reading mis-transcriptions** (e.g. `ツァツォ`
  becomes `サツ`, `チューティー` variants), which is expected for a
  96 kHz professional recording read at pace with rich vocabulary.
- **Approximately 22 files under 50%**, which are either whisper-blended
  noise or content of overwhelmingly foreign proper-noun density.

This is not a criticism of whisper-base. A larger model would close most of
the gap. The point is the **CUELOOP pipeline is only as good as its ASR
front-end**, and the macOS SFSpeechRecognizer on-device model (the one we
actually use) is a different model with different failure modes. Both share
the property that they will mis-transcribe rare Japanese names and loanwords;
they will not share the errors on common words. We have not yet measured the
on-device recognizer on this corpus (it requires a live microphone or a
Swift-file recognizer build, both of which are platform-limited).

### 2. N-gram predictor on the corpus

The n-gram predictor (order 3/4/5, span 4/8/16/24/40) achieves **zero**
exact-prefix matches on held-out lines from this corpus, under all
configurations, in both train-on-Lux-only and train-on-embedded+Lux. This
does not mean training on Lux is useless; it means the embedded corpus
does not transfer to human Japanese, and Lux-only training does not
generalize across genres of Japanese prose.

The implication:

- The CUELOOP n-gram predictor is **not useful on real Japanese speech**.
- The local LLM (`--decision local`) is the only model that performs
  prediction at all on this corpus.
- The fallback n-gram is a *very* weak signal and should not be treated
  as meaningful input — `min_confidence=0.55` is a low bar.

**We have not fixed this yet.** Options include (a) expand the embedded
corpus on Japanese-software-reasoning prose, (b) always launch with
`--predictor local` for live sessions, (c) retire the n-gram fallback
entirely.

### 3. Local-LLM predictor on the corpus

The local model is the workhorse (`--predictor local`, the intended live
path). Using the exact `LocalLLMPredictor` code CUELOOP ships with, the
same prefix task on the same corpus:

```text
model: qwen2.5:7b-instruct (Ollama)
corpus: Lux ita_recitation, 12-char prefix
exact_prefix_rate: 0/30
```

The local model *never replicates* the human-spoken continuation. This is
unsurprising: the Lux ita_recitation is translations of European literature
and philosophy — far from own-domain Japanese software-engineering speech,
which is what CUELOOP was built around. Topicality is much better (it
writes `笑顔で手を振っている` for `女の子がキッキッ嬉しそう`), but no model
knows which specific token a writer picked.

The signal we can draw here is narrower than it first appears:

- Out-of-domain Japanese: no predictor helps. This does not indict CUELOOP's
  mechanism; it means we have no proof either way for the target domain until
  live sessions exist.
- The decision-model latency to consider: local LLM is still the only model
  with any signal; n-gram does nothing at all beyond substring recall.

## What was not measured

- **The CUELOOP arbitration pipeline as a whole**: we cannot yet test
  uptake/carryover from real Lux audio because:
    - To trigger an intervention you need a cue *during* the speaker's turn.
      Lux reads are single-shot utterances; there is no live turn to react to.
    - To test uptake/carryover on real humans you need headsets/microphones.
- **The macOS SFSpeechRecognizer on Lux files**: not run locally (Xcode
  license not accepted; the Swift file-asr tool in `tools/file_asr.swift`
  waits on CI or this machine accepting the license).  The microphone path
  has been driven once against a clean recording of Lux files played back
  from a speaker, which produced recognizable transcript events: on
  `EMOTION100_011.wav`, the on-device recognizer recorded `イタリア旅行で`,
  `例えばナポリやフィレンツェ`, etc.; on `EMOTION100_066.wav`, the trigger
  phrase `だから` was detected and a boundary fired at 4505 ms, but the rule
  policy stayed silent because the embedded corpus predicted the next span
  was `。` (only punctuation, no operations to rank). With
  `--decision local --predictor local` on the same file, one intervention
  (`仕組みは？`) was planned but then cancelled because the speaker resumed
  within the insertion window — that is the intended cancellation behaviour,
  observed live on real audio.
- **A Japanese-software-reasoning speech corpus**: none publicly available
  short of paying for it. The synthetic CUELOOP corpus in
  `experiments/`/`docs/benchmark.md` is the closest substitute.

## What was added to CUELOOP from this evaluation

- `scripts/tune_ngram.py` — offline n-gram evaluation/tuning harness on any
  `metadata.csv`. Useful once a proper corpus appears for the target domain.
- `tools/file_asr.swift` — Swift one-shot recognizer per Lux-style file,
  for direct SFSpeechRecognizer comparison when the license is accepted.
- `scripts/simulate_lux_replay.py` — replays Lux *or any compatible corpus*
  (CSV with file_name/cleaned_path/transcript) through the actual
  `SyncRuntime`, emitting analyzer-ready paired reports (CUE and SILENCE).
  On ReazonSpeech it delivered 455 cues with the rule backend and zero with
  the local-LLM backend, and confirmed the live measurement path runs on
  real Japanese discourse without changing the runtime. Also fixes a
  cache-subset bug where a warm whisper cache returned extra files not in
  the requested corpus (regression test included).

Note: the Lux corpus is literary Japanese (read speech) with essentially no
operation-tag vocabulary overlap — both the embedded n-gram and the local
LLM produce predictions whose operation tags never coincide with the real
operations in the next sentence. This is intrinsic to the corpus (it is not
a reasoning dataset) rather than something that can be fixed by tuning the
predictor.

See `docs/benchmark.md` for the synthetic evaluation and
`docs/experiment-protocol.md` for the human study design.

## Rule arm on full Lux replay (328 files)

The `rule` (default n-gram prediction-gap) pipeline was fired over every
Lux file. Counts (from the analyzer,`artifacts/lux-rule-328-analysis.md`):

```text
files                  328
predictions fired      405
interventions delivered 276 (held on SILENCE: 276)
cancellations          0
outcomes               276
uptake / carryover     0.00 / 0.00 (both arms identical; read speech)
repertoire_growth      0.04
```

Distribution of chosen operations (after fatigue tuning): `concretize 194`,
`dependency 50`, `mechanism 13`, `conclusion 8`, `invert 6`, `ground 4`,
`comparison 1`. Cues used: `具体は？` 194×, `何に？` 50×, `仕組みは？` 13×,
`結論は？` 8×, `逆は？` 6×, `根拠は？` 4×, `何と比べる？` 1×. The rule backend
now applies per-session repetition fatigue (`fatigue_penalty=2.0`, see
`RuleDecisionModel`): before it, concretize took 264/276 (96%); after it the
mix spans 7 distinct operations while the delivery count (276), prediction
count (405) and every paired delta (+0.000) are unchanged.

The rule fires on the embedded n-gram's predictions at the small pauses in
the TTS read-through — both arms receive an identical count of
interventions so the paired delta is `+0.000` on every metric. The
trigger is *not* meaningful announcement detection — Lux is a TTS
proofreader script. Before fatigue tuning, most cues fired on `concretize`
because the embedded corpus is dominated by example-language (`例えば`,
`具体的`); the fatigue penalty now rotates the mix (see above).

## Local-LLM end-to-end on the Lux corpus

A complete local-LLM run was executed over all 328 local Lux files under both
conditions (command below; reports at `artifacts/lux-llm-328-full/`):

```text
file                              328 CUE + 328 SILENCE
predictions                      480 (identical in both arms)
interventions                    457 CUE / 466 SILENCE
delivered                        0  (both arms; all interventions cancelled/held)
files with >= 1 prediction       201 / 328
predictions not intervened        23 CUE / 14 SILENCE (decision model vetoed
                                 its own prediction)
uptake / carryover               0.00 / 0.00 (identical; read speech)
paired delta                     +0.000 on every metric
n_pairs                          328
median boundary->delivery delay   insufficient (no deliveries)
repertoire_growth                0.06 (identical both arms; reflects operation
                                 tags present in the transcript, not an effect
                                 — nothing was delivered)
```

CUE and SILENCE predictions are identical because the predictor and the seed
are deterministic; interventions differ by 9 because the local-LLM decision
is not bit-deterministic across processes (Ollama batching/scheduling).

Idempotent `--limit` runs: the earlier `--limit 50` run produced
`files=50, predictions=60, interventions=60, delivered=0, cancelled=60`,
fully consistent with the complete 328-file run.

Every intervention is cancelled (CUE) or held (SILENCE) because the actually
spoken text diverges from the prediction before the speaker pauses long
enough for a cue to land. This is the honest state: **the pipeline decides
when to speak, but never manages to say it during read speech** — the corpus
contains single-shot literary sentences, not an unfolding reasoning stream.
What the run *does* verify is that the whole path — per-word ASR events →
prediction → prediction-gap decision → SILENCE hold / CUE cancel — composes
on real Japanese audio in both arms.

The most frequent predictor engagements are on `EMOTION100_072` (13
predictions) and `EMOTION100_011` (7) — driven by surface trigger vocabulary
(`例えば`, `理由`), not by announced reasoning operations.

To improve:

- (a) a Japanese software-reasoning speech corpus in CUELOOP's operation
  vocabulary — not available publicly;
- (b) real software-reasoning recordings — not available publicly;
- (c) accept that the reliable arm on this corpus is the `rule` backend with
  the bare cue (276 delivered, all `具体は？`-heavy), not the local LLM.

## ReazonSpeech sample: vocabulary-domain arm (YouTube Japanese)

YouTube-derived ReazonSpeech audio is the closest publicly available corpus
to the register CUELOOP was built for, and it does contain CUELOOP's
operation vocabulary:

```text
sample:         200 files, ≥6 s and ≥15 chars, total ≈ 52 minutes
ops tagged in transcripts:  30 / 200 files (conclusion 10, concretize 7,
                            dependency 6, mechanism 5, boundary 3,
                            invert 1, ground 1, comparison 1)
whisper-base-mlx vs dataset whisper-turbo pseudo-labels: 78.0% char accuracy
                            (0/200 exact; both are machine transcriptions,
                            so this is an agreement bound, not an accuracy bound)
```

### Rule backend, 200-file sample

```text
files                    200 (CUE and SILENCE paired)
predictions fired        894  (vs 405 on the Lux read corpus; 2.2×)
interventions delivered  455  (CUE)
held (SILENCE)          455
cancellations            0
files delivering ≥ 1 cue 133 / 200
max cues per file         9
operations delivered     concretize 207, dependency 134, conclusion 60,
                            mechanism 23, ground 18, invert 13
                            (fatigue tuning; before: concretize 427, dependency 2)
paired uptake delta     +0.000 (both arms identical: the continuation text
                            is identical in both arms, so target_operation_seen
                            fired 8/455 times in both)
```

### Local-LLM backend, 200-file full sample (qwen2.5:1.5b-instruct)

A full-sample local-LLM run was executed with the fast local model
(`qwen2.5:1.5b-instruct`, ~25 min per arm on this machine):

```text
files                    200 CUE + 200 SILENCE
predictions              986 CUE / 973 SILENCE (sampled with the model's
                          default temperature; not bit-deterministic)
files with ≥ 1 prediction 142 / 200 (both arms)
interventions            434 CUE / 425 SILENCE
delivered                2 CUE / 0 SILENCE  <-- first non-zero LLM deliveries
cancelled                432 CUE / 424 SILENCE
delivery rate            0.46%
analyzer n_pairs         200, every metric +0.000
median delay ms          1570 (2 deliveries)
uptake / carryover       0.00 / 0.00 in both arms (the 2 delivered cues —
                          boundary `範囲は？` on `ミリカ!`, ground `根拠は？`
                          on `おととい、` — were not picked up; read speech,
                          expected)
```

The 2 delivered cues are the first ever produced by the local-LLM path on
real Japanese audio — they prove the cancellation is not a structural
impossibility, just rarity. The local-LLM arm still fires essentially never
enough to affect anything: ~0.5% delivery rate vs 455 delivered (100% of
interventions) under the rule backend on the same files.

A `gemma3:4b-it-qat` run over the same sample started but was stopped at
52/200 CUE files for compute (partial CUE-only run: 255 predictions, 249
interventions, 0 delivered — consistent with the 30-tagged-file gemma3:4b
run). `qwen2.5:1.5b-instruct` is not recommended as a predictor for this
domain; it was used here because each session costs ~4s of client time.

### Local-LLM backend, 30 operation-tagged files (gemma3:4b-it-qat)

The 30 files whose transcripts tag at least one CUELOOP operation — the
domain where a cue has a real target — were run with gemma3:4b-it-qat
(~1 min of LLM time per 15 s utterance per arm):

```text
files                    30 (CUE and SILENCE paired)
predictions fired        254 (identical in both arms)
interventions            250 CUE / 249 SILENCE (not bit-deterministic)
delivered                0 (all 250/249 cancelled/held)
files with ≥ 1 delivered 0 / 30
analyzer n_pairs         30, every metric +0.000
median delay             insufficient (no deliveries)
repertoire_growth        1.03 both arms (transcript tags, not effect —
                            nothing was delivered)
```

The local-LLM predictor produces plausible continuations but never the exact
author's wording — so after each boundary the synchronizer marks the speech
diverged and every intervention is cancelled. The rule backend survives the
same data because its echo-like n-gram predictions partially match the
continuation often enough to keep the synchronizer un-diverged long enough
for the delivery window to elapse; that is not foresight, it is substring
echo, and it is why the rule arm can deliver while the local-LLM arm cannot.

**Honest framing.** This corpus changes the *base rate*, not the mechanism:
CUELOOP's operation vocabulary is genuinely present in real Japanese
discourse, the rule backend fires and delivers cues against it (455 times,
concentrated on `具体は？`), and the local-LLM backend fires often but never
completes a delivery. Neither result says anything about whether a human
hearing a cue would shift their reasoning — that remains the open H1/H2
question for live sessions.

### Artifacts and reproduction

```bash
# whisper transcription (already cached in data/real-speech/reazon/)
# full 200-file sample, rule backend (~4 min)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
  --source-name reazon --output-dir artifacts/reazon-rule-200 --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/reazon-rule-200/cue \
  artifacts/reazon-rule-200/silence --markdown artifacts/reazon-rule-200-analysis.md

# 30 operation-tagged files, local-LLM backend (~60 min; needs Ollama + gemma3)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-tagged.csv \
  --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
  --source-name reazon-tagged --output-dir artifacts/reazon-llm-tagged \
  --predictor local --decision local --local-model gemma3:4b-it-qat

# full 200-file sample, fast local model (~30 min; needs Ollama + qwen2.5:1.5b)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
  --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
  --source-name reazon-llm15b --output-dir artifacts/reazon-llm15b-200 \
  --predictor local --decision local --local-model qwen2.5:1.5b-instruct

# corpus acquisition (audio is not in git):
#   tar of chunk 000 from huggingface.co/datasets/osanpo/reazonspeech (CDLA-Sharing-1.0)
#   transcriptions from huggingface.co/datasets/efwkjn/reazonspeech_metadata
```

## How to reproduce these numbers

```bash
# ASR evaluation (whisper-base-mlx) — see docs/real-speech.md results above.
.venv/bin/python - <<'PY'
import csv, mlx_whisper, time
t0 = time.time()
rows = list(csv.DictReader(open('data/real-speech/lux/metadata.csv', encoding='utf-8')))
total = 0; match = 0
for row in rows:
    if not row['cleaned_path'].startswith('data/cleaned/ita_recitation/'):
        continue
    wav = 'data/real-speech/lux/' + row['cleaned_path']
    result = mlx_whisper.transcribe(
        wav, path_or_hf_repo='/tmp/whisper-base',
        language='ja', task='transcribe', fp16=False, verbose=None,
    )
    pred = result['text'].strip()
    truth = row['transcript'].strip()
    sm = __import__('difflib').SequenceMatcher(None, pred, truth)
    match += sum(m.size for m in sm.get_matching_blocks())
    total += len(truth)
print(f"{total} chars, character accuracy: {match / total * 100:.1f}%, time {time.time() - t0:.1f}s")
PY

# n-gram evaluation
.venv/bin/python scripts/tune_ngram.py --corpus data/real-speech/lux/metadata.csv \
  --order 3 --prefix-chars 8 --span-characters 8 \
  --eval-mod 10 --eval-remainder 0

# full Lux replay, rule backend (~2 minutes)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 scripts/simulate_lux_replay.py \
  --output-dir artifacts/lux-rule-328 --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/lux-rule-328/cue artifacts/lux-rule-328/silence \
  --markdown artifacts/lux-rule-328-analysis.md

# full Lux replay, local-LLM backend (~30 minutes; needs Ollama + a model)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 scripts/simulate_lux_replay.py \
  --output-dir artifacts/lux-llm-328-full --predictor local --decision local \
  --local-model gemma3:4b-it-qat
.venv/bin/python scripts/analyze_experiment.py artifacts/lux-llm-328-full/cue artifacts/lux-llm-328-full/silence \
  --markdown artifacts/lux-llm-328-full-analysis.md
```

Whisper characterization needs `mlx-whisper` installed; the pip experience on
PyPI for MLX is slow (~20 minutes per wheel). The Lux corpus is deliberately
git-ignored — don't commit the WAVs.

## Next steps

1. Accept Xcode license on the developer laptop and run the macOS on-device
   recognizer on Lux/Reazon files (tools/file_asr.swift + a tiny harness),
   comparing error profile vs whisper-base-mlx.
2. **Human sessions remain the only gate to H1/H2**: Japanese reasoning
   speech recorded live with a speaker who can hear and react to cues.
   The Reazon arm shows real content with the vocabulary present, but
   everything remains held/cancelled — nothing has been measured about
   effect on a human. Issue #6.
3. With a Reazon-style discourse corpus in hand, the most valuable next
   technical step is *domain-adapted prediction*: measure whether the local
   LLM (or n-gram) can anticipate operations in the 30-tagged-file subset
   with enough fidelity to keep the synchronizer un-diverged past the
   delivery window — that is the only way local-LLM cues can complete a
   delivery on recorded speech.
4. Expand the embedded corpus on Japanese software engineering (code review,
   design discussion) texts; re-evaluate `exact_prefix_rate`. The embedded
   corpus is the single largest source of false-positive predictions.
