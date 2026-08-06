# Deterministic Intervention Policy

The language model proposes hypotheses. The controller decides actions.

## Priority order

1. Safety and privacy constraints
2. Suppression rules
3. Turn and timing validity
4. Cognitive-load constraints
5. State confidence
6. Intervention selection
7. Cooldown and repetition control

## Suppression rules

Always choose `SILENCE` when any of the following applies:

- the user is productively generating
- the user is currently speaking and the candidate is linguistic
- state confidence is below the configured threshold
- the candidate has expired
- the same intervention was recently delivered
- the user has requested no interruption
- overload is likely
- required source grounding is unavailable for a knowledge claim

## Initial state-to-action mapping

| State | Primary action | Secondary action | Forbidden default |
|---|---|---|---|
| `PRODUCTIVE` | `SILENCE` | — | question generation |
| `RETRIEVAL_FAILURE` | `BACKCHANNEL` | minimal `QUESTION` | long explanation |
| `STRATEGY_FAILURE` | `QUESTION` | `CONSTRAIN` | completed answer |
| `FIXATION` | `PERTURB` | `MIRROR` | more content in the same direction |
| `STRUCTURE_GAP` | `QUESTION` | `MIRROR` | ungrounded factual insertion |
| `OVERLOAD` | `SILENCE` | end or defer | new knowledge |

## Candidate acceptance

A candidate may be delivered only if:

- its intended effect is explicit
- evidence for the state is recorded
- confidence exceeds the action threshold
- it contains one cognitive operation
- it respects the maximum length
- its delivery window is still open
- no stronger suppression rule applies

## Candidate lifecycle

```text
PROPOSED
→ ELIGIBLE
→ DELIVERED | CANCELLED | EXPIRED
```

Candidates should be cancelled when the user resumes productive speech or the relevant state changes.

## Initial policy pseudocode

```text
if privacy_or_safety_block:
    SILENCE
elif user_is_productive:
    SILENCE
elif linguistic_candidate_and_user_is_speaking:
    SILENCE
elif candidate_is_stale or confidence_is_low:
    SILENCE
elif overload_likely:
    SILENCE
elif cooldown_active or intervention_is_repetitive:
    SILENCE
else:
    deliver(highest_priority_eligible_candidate)
```

## Logging requirements

For every proposed candidate, record:

- timestamp
- observed evidence
- state hypothesis
- confidence
- proposed action
- delivery status
- cancellation reason
- user response window

The log is for debugging the system, not for claiming cognitive improvement.
