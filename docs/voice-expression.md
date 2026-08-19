# Voice Expression Architecture

The audible delivery layer of CUELOOP is not a fixed TTS call. It is an
expression channel whose properties shape how the human receives a cognitive
cue. This document records the design direction discussed during live
experiments and the implementation split between what exists and what is next.

## Why voice matters

CUELOOP's core hypothesis is that a pretrained model's broad knowledge can be
converted into minimal, context-sensitive feedback that helps a human update
their own reasoning policies and internal representations (see
`origin-and-thesis.md`).

The audible cue is the only contact point where model-derived knowledge enters
the human's live reasoning. If that contact feels like a machine, the human
treats it as an interruption and the knowledge is rejected. If the cue is
indistinguishable from natural speech, the human can absorb the missing
operation without noticing the transfer.

Two requirements follow:

1. **Minimal perceptible disruption.** The cue should be spoken so that the
   listener does not stop generating their own reasoning. This is not about
   volume; it is about prosody, phrasing, and timing that fit the ongoing
   utterance instead of clashing with it.
2. **Transfer, not answer.** The cue must trigger one reasoning operation. It
   must never carry the model's completed answer, because a delivered answer
   would replace the human's reconstruction instead of strengthening it.

## Design vocabulary

- **Cue**: the minimal linguistic token delivered during speech, bounded to a
  few characters (for example `何に？`).
- **Voice profile**: a named set of expression parameters (TTS command, voice
  name, rate, volume, pitch) plus the delivery intent it serves.
- **Expression intent**: why this cue is being spoken right now. Example
  intents: `backchannel` (continue), `question` (one operation), `mirror`
  (reflect structure), `constrain` (narrow search), `verify` (flag a claim).
- **Delivery channel**: the physical route the cue takes to the human. The
  first channel is local TTS. A future channel is a human facilitator, where
  the cue is handed to a person in the conversation instead of synthesized.
- **Discomfort budget**: the total amount of perceptual mismatch the system is
  allowed to introduce per turn before the cue starts to read as noise.

## Implemented (M2 voice core)

- Streaming VAD/ASR events through the Apple adapter.
- Turn-boundary detection and endpoint silence.
- Expiring, cancellable candidate decisions.
- Interruptible command-based TTS (`CommandSpeaker`).
- Mid-speech cue injection (`MidSpeechInjector`) using context-aware short
  cues derived from the user's own vocabulary.
- Suppression on overload, silence by default, cooldown, and
  repeat-form avoidance.
- Optional `--tts-voice` so the delivery voice can be selected.

## Next: voice profiles

Replace the single fixed TTS command with a small set of voice profiles so the
system can select the least-disruptive expression for the current intent.

### Profile model

```text
VoiceProfile
  name            e.g. "whisper", "neutral", "assert"
  command         TTS command arguments (provider-neutral)
  voice           voice name for providers that take one
  rate            words per minute, if the provider supports it
  volume          0.0 .. 1.0
  pitch           relative shift, if supported
  intent          the delivery intent this profile serves
```

### Selection rule

1. The intent is decided before delivery (already part of `Analysis`).
2. The profile with that intent is used. If none matches, the neutral profile
   is used.
3. Profiles are declarative, not learned from interaction, in the initial
   scope. M5 personalization may later fade profiles, not re-train them.
4. All delivery still goes through the existing controller suppression and
   cooldown rules. Profiles only change the expression, never the decision.

### Expected effect

- `backchannel` and `question` use a soft, neutral, low-volume voice so the
  cue blends into the speaker's rhythm.
- `constrain` and `verify` may use a slightly clearer voice without becoming
  alarm-like.
- The same cue text can be expressed differently per intent, which also helps
  the global "avoid repeating the same intervention form" rule
  (`intervention-atlas.md`).

## Next: natural delivery through expression parameters

Beyond voice choice, the same TTS provider can reduce discomfort through
parameters:

- Slower rate for longer cues, faster for single-word backchannels.
- Lower volume for mid-speech cues, restored for post-silence cues.
- Phrase-initial silence so the cue starts at a natural breath boundary.
- A ceiling on cue length (already enforced) so the cue never exceeds the
  discomfort budget.

## Future: human delivery channel

A cue does not have to be synthesized speech. When a session includes another
person (a facilitator, coach, or partner), the system can emit the cue as a
short written prompt to that person, who then says it naturally. This:

- removes the remaining machine quality of the voice,
- preserves the determinism of the intervention decision,
- makes the discomfort budget effectively zero for the listener.

The delivery-channel boundary must stay the same: the controller decides
*whether* and *what*, the channel decides *how*. The human facilitator is a
channel, not a decision-maker.

## Non-goals for this layer

- Do not generate a model persona or a persistent voice identity.
- Do not clone any real person's voice.
- Do not store or transmit raw audio from the delivery side.
- Do not let expression parameters override a `SILENCE` decision.
