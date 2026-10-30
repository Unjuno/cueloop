# Decision-model-first CUELOOP: minimal runtime contract

Status: proposed architecture, not a claim of cognitive effectiveness.

## Product objective
Synchronize a person's evolving spoken reasoning with a model's evolving candidate directions. Deliver **zero or one** tiny, timely audio cue only when it plausibly adds a useful direction without taking over the person's reasoning.

## Default runtime
```text
microphone -> streaming transcript -> rolling context
                                      |
                                      v
                             decision model
                                      |
                             NONE or micro-cue
                                      |
                   freshness / overlap / cooldown gate
                                      |
                              cancellable TTS
```

The model is allowed to construct context-specific candidates and choose NONE. The external gate owns final delivery. A candidate becomes invalid when its supporting transcript revision is superseded, its deadline passes, or the user resumes speaking.

## Minimal interface
Input: recent finalized speech, current partial revision, timestamp, recently spoken cues, optional grounded context.

Output: `{ "action": "NONE" | "CUE", "text": "...", "operation": "...", "based_on_revision": 0, "expires_at_ms": 0 }`.

`text` is a single concept, relation, contrast, or short question; no completed answer. `operation` is descriptive telemetry, not a mandatory state taxonomy. Confidence scores are not assumed calibrated.

## What stays optional
Legacy state classifiers, structural-diff extraction, source grounding, long-term memory, personalization, fading, pilots, and post-session review are optional plugins, **not prerequisites** for the live cue path. Preserve existing APIs until an alternative passes the same regression checks. Avoid deleting working functionality in the first migration.

## Implementation sequence
1. Trace the actual current live path and identify the smallest adapter for a model-driven NONE/CUE decision.
2. Introduce a feature-flagged minimal path without changing the legacy default.
3. Run identical recorded/replay inputs through both paths and record cue timing, repetition, cancellation, and output length.
4. Verify no stale candidate is delivered and that user speech cancels pending/output cues.
5. If the minimal path is at least as robust and simpler, make it the default in a separate reviewed change; otherwise document failure and retain the original.

## Evidence and scope
Unit tests prove contract behavior only. Replay compares engineering behavior, not creativity or long-term learning. Do not equate model prediction with observed human thought, or lexical novelty with cognitive benefit.

## Three connected disciplines
- Streaming systems: revision consistency, expiry, cancellation.
- Decision theory: abstention and asymmetric intervention cost.
- Cognitive science/HCI: attention, interruption cost, human agency.

## Completion gate
A reproducible CLI/replay run demonstrates NONE and CUE, stale-revision rejection, duplicate suppression, cooldown, barge-in cancellation, and provider failure falling back to NONE. Record measured environment and latency distributions when benchmarking.
