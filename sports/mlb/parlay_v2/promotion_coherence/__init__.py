"""Shadow-only promotion coherence layer for MLB parlays.

This subpackage is additive and non-invasive. It never mutates a payload
or a selector; it reads a `daily_predictions.json` payload (or any
persisted parlay decision record with the same shape) and returns a
parallel `CoherentPromotionDecision` describing what would have been
published if the conservative quality overlay were the final publication
authority AND a `promotion_confidence` margin cleared its required buffer.

Why this exists:
    On 2026-09-02 the normal parlay's `public_quality_overlay` returned
    ABSTAIN (three blocking reasons: both leg probabilities below 70% and
    the joint below 50%), but the payload's `parlays.action` remained
    "ACT" and the parlay was published as a promoted pick. The overlay
    already exists inside the payload as an "authoritative shadow"; this
    module treats it as authoritative in a strictly parallel decision
    stream so its behavior can be measured against real slates before any
    live publication path is changed.

Scope guarantees:
    * No import from and no write into any live selector, run script,
      frontend payload builder, or workflow. All coherence logic operates
      on already-persisted payload dictionaries.
    * Deterministic. Given the same payload, the same decision.
    * All deductions default to 0.0 so `promotion_margin ==
      calibrated_joint - break_even` unless the caller (or a tuned
      profile) supplies concrete penalties. Introducing a penalty is a
      deliberate, explicit act, never a silent default.
"""

from .promotion_confidence import (
    CoherentPromotionDecision,
    PromotionConfidenceComponents,
    PromotionPenalties,
    PromotionThresholds,
    decide_coherent_promotion,
    default_thresholds,
    promotion_confidence_components,
)
from .pair_schema_v2 import (
    PAIR_OBSERVATION_SCHEMA_VERSION_V2,
    MarketDisagreementProfile,
    PairObservationV2,
    market_disagreement_deduction,
)
from .same_game_penalty import (
    SameGamePenaltyProfile,
    apply_same_game_penalty,
    same_game_shared_failure_deduction,
)

__all__ = [
    "CoherentPromotionDecision",
    "MarketDisagreementProfile",
    "PAIR_OBSERVATION_SCHEMA_VERSION_V2",
    "PairObservationV2",
    "PromotionConfidenceComponents",
    "PromotionPenalties",
    "PromotionThresholds",
    "SameGamePenaltyProfile",
    "apply_same_game_penalty",
    "decide_coherent_promotion",
    "default_thresholds",
    "market_disagreement_deduction",
    "promotion_confidence_components",
    "same_game_shared_failure_deduction",
]
