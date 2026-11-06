"""Predictive synchronization core for CUELOOP.

The original CUELOOP loop was turn-based: wait for silence, classify, then
respond. The synchronization loop is different. It models what the speaker is
about to say, tracks how the live partial transcript aligns with that
prediction, and inserts a very short cue *before* the predicted span is spoken.

Two concerns are deliberately separate:

- the decision model decides *when* an intervention is warranted;
- the cue assignment decides *what* cue is delivered.

Keeping them separate makes it possible to measure the linguistic-framing
effect of a cue (a weak Sapir-Whorf style hypothesis) without confounding it
with a model's preference for cues that look effective.
"""

from .evaluate import summarize
from .models import (
    Delivery,
    EventType,
    Intervention,
    Outcome,
    Prediction,
    SyncEvent,
    SyncState,
)
from .policy import (
    AlternateAssigner,
    AsyncDecisionModel,
    CanonicalAssigner,
    CueAssigner,
    DecisionModel,
    FixedAssigner,
    LocalLLMDecisionModel,
    RandomAssigner,
    RuleDecisionModel,
)
from .predictors import (
    AsyncPredictor,
    LocalLLMPredictor,
    NgramPredictor,
    Predictor,
    ScriptedPredictor,
)
from .runtime import SyncRuntime
from .synchronizer import Synchronizer
from .vocabulary import CUE_BY_OPERATION, OPERATIONS, tag_operations

__all__ = [
    "CUE_BY_OPERATION",
    "OPERATIONS",
    "AlternateAssigner",
    "AsyncDecisionModel",
    "AsyncPredictor",
    "CanonicalAssigner",
    "CueAssigner",
    "DecisionModel",
    "Delivery",
    "EventType",
    "FixedAssigner",
    "Intervention",
    "LocalLLMDecisionModel",
    "LocalLLMPredictor",
    "NgramPredictor",
    "Outcome",
    "Prediction",
    "Predictor",
    "RandomAssigner",
    "RuleDecisionModel",
    "ScriptedPredictor",
    "SyncEvent",
    "SyncRuntime",
    "SyncState",
    "Synchronizer",
    "summarize",
    "tag_operations",
]
