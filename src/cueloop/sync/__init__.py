"""Predictive synchronization core for CUELOOP.

The original CUELOOP loop was turn-based: wait for silence, classify, then
respond. The synchronization loop is different. It models what the speaker is
about to say, tracks how the live partial transcript aligns with that
prediction, and inserts a cue *before* the predicted span is spoken — by
default a complete "initial-condition" sentence (stem) that embodies the
operation, rather than a bare interrogative word.

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
    LinearDecisionModel,
    LocalLLMDecisionModel,
    RandomAssigner,
    RuleDecisionModel,
)
from .policy_model import (
    LinearPolicyModel,
    PolicyObservation,
    samples_from_records,
    train_linear_policy,
)
from .predictors import (
    AsyncPredictor,
    LocalLLMPredictor,
    NgramPredictor,
    Predictor,
    ScriptedPredictor,
)
from .runtime import SyncRuntime
from .stem import StemCueGenerator
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
    "LinearDecisionModel",
    "LinearPolicyModel",
    "LocalLLMDecisionModel",
    "LocalLLMPredictor",
    "NgramPredictor",
    "Outcome",
    "PolicyObservation",
    "Prediction",
    "Predictor",
    "RandomAssigner",
    "RuleDecisionModel",
    "ScriptedPredictor",
    "StemCueGenerator",
    "SyncEvent",
    "SyncRuntime",
    "SyncState",
    "Synchronizer",
    "samples_from_records",
    "summarize",
    "tag_operations",
    "train_linear_policy",
]
