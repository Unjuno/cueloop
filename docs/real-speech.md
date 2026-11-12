# Real Speech Corpus Evaluation

CUELOOP's live path depends on (a) the macOS microphone adapter for ASR and
(b) the n-gram predictor for low-latency prediction with the local LLM as
semantic prior. This document records what we found when we put a publicly
available Japanese speech corpus (real human audio of Japanese) through that
pipeline.

**Corpus.** *Lux Japanese Speech Corpus* (Lami Inc., CC-BY-4.0) — the
`ita_recitation` clean read subset: 324 files, ~230 MB, 96 kHz/16-bit WAV,
transcriptions in `metadata.csv`. It contains one Japanese voice (the Lux
character) reading diverse prose (biographical, historical, culinary,
epistemic). License is attribution-only for CC-BY-4.0.

This is a **non-conversational, purposeful read speech** corpus. It is not a
representative sample of spontaneous Japanese reasoning, and it is not the
kind of speech CUELOOP operates on live. It is, however, real recorded speech,
and it is broadly Japanese, so it is a far more honest calibration target
than a synthetic replay.

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
- `scripts/simulate_lux_replay.py` — replays the 328 local Lux files through
  the actual `SyncRuntime`, emitting analyzer-ready paired reports (CUE and
  SILENCE) for every file. Produces zero uptake in both arms, as expected
  for read speech; confirms the live measurement path runs on real Japanese
  audio without changing the runtime.

Note: the Lux corpus is literary Japanese (read speech) with essentially no
operation-tag vocabulary overlap — both the embedded n-gram and the local
LLM produce predictions whose operation tags never coincide with the real
operations in the next sentence. This is intrinsic to the corpus (it is not
a reasoning dataset) rather than something that can be fixed by tuning the
predictor.

See `docs/benchmark.md` for the synthetic evaluation, `docs/experiment-protocol.md`
for the human study design, and `docs/real-speech.md` (this file) for the
Limited Corpus evaluation.

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

Distribution of chosen operations: `concretize 264`, `invert 6`, `ground 4`,
`mechanism 1`, `conclusion 1`. Cues used: `具体は？`, `逆は？`, `根拠は？`,
`仕組みは？`, `結論は？` in the same order of frequency.

The rule fires on the embedded n-gram's predictions at the small pauses in
the TTS read-through — both arms receive an identical count of
interventions so the paired delta is `+0.000` on every metric. The
trigger is *not* meaningful announcement detection — Lux is a TTS
proofreader script. Most cues (264/276) fire on `concretize` because our
embedded corpus is dominated by example-language (`例えば`, `具体的`).

## Local-LLM end-to-end on the Lux corpus

`scripts/simulate_lux_replay.py --predictor local --decision local` was run
on the 38 files that finished first (the rest are queued as a background job).
The output is re
produced in `artifacts/lux-llm-328/`:

```text
files with predictions:    26 / 38 (every prediction produced an intervention)
delivered:                  0 / 38 (all 71 interventions cancelled — speaker kept talking)
held on SILENCE:            71 (next-best, latency-matched arm receives 0)
perceived novel wording:    71 interventions matched the symbolic parser

A 50-file rerun (`--limit 50`) produced:

```text
files                 50
predictions          60
interventions        60
delivered            0
cancelled            60 (all)
SILENCE interventions 60 (all)
```

Every local-LLM prediction fires the symbolic parser (every predicted span
tags at least one operation token, or the context-fit constraint is
satisfied), and every intervention is cancelled because the actually spoken
text diverges from the prediction before the speaker pauses long enough for
a cue to land. This is the honest state: **the pipeline knows when to speak,
but not what to say**, because the corpus is literary Japanese and the local
model cannot match the author's choices. To improve:

- (a) train a Japanese software-reasoning speech corpus targeted at the
  vocabulary CUELOOP uses — not available publicly;
- (b) evaluate on real software-reasoning recordings — not available publicly;
- (c) accept that the reliable arm is the `rule` backend with the bare cue,
  not the local LLM.
```

**The key mismatch is also visible in the emotions files most frequently
engaging the predictor**: `EMOTION100_011` (a participant speech containing
例えば) yields 7 predictions; `EMOTION100_072` (理由 gives 13); the written
recitation corpus (e.g. `RECITATION324_002` etc.) produces rarely any
predictions (1 or 0). The predictor is engaging on the *trigger words*
its own regex detects, not on the actual reasoning/shape of the sentence.

The local LLM cannot detect the announcement and the reason for a mutation;
it merely fires on the presence of vocabulary (前提/結論/依存 etc.) absent
of any genuine reason to intervene. This means the live Japanese behaviour
of CUELOOP's local-LLM path will be triggered predominantly by `trigger_words`
rarely by reasoned operation shifts.

## How to reproduce these numbers

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
```

Whisper characterization needs `mlx-whisper` installed; the pip experience on
PyPI for MLX is slow (~20 minutes per wheel). The Lux corpus is deliberately
git-ignored — don't commit the WAVs.

## Next steps

1. Accept Xcode license on the developer laptop and run the macOS on-device
   recognizer on Lux files (tools/file_asr.swift + a tiny harness),
   comparing error profile vs whisper-base-mlx.
2. Record live CUELOOP sessions on actual laptop microphones against a
   controlling condition (politeness-matched recording of Lux reading the
   same transcripts), with per-condition diarization scored offline.
3. Expand the embedded corpus on Japanese software engineering (code review,
   design discussion) texts; re-evaluate `exact_prefix_rate`. The embedded
   corpus is the single largest source of false-positive predictions.
