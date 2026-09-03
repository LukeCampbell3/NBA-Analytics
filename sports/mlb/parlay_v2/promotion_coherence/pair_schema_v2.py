"""Additive v2 pair-observation schema for per-leg model and no-vig
market probabilities, and the market-disagreement deduction that
consumes them.

This module DOES NOT modify the existing
`sports/mlb/parlay_v2/calibration/pair_schema.py`. It defines an
additive v2 shape that a future capture path can populate, and
provides a `PairObservationV2` reader that returns a typed view of
either a real v1 row (with per-leg fields absent -> None) or a v2 row
(with per-leg fields present).

Why an additive schema, not a modification:

    * The live pair-ingest path writes v1 rows today. The promotion-
      coherence shadow's whole point is to be additive. Modifying the
      live schema would need an ingest change and a data migration --
      both live-codebase touches this branch is not asked to make.
    * The synthetic pair ledger (`synthesize_pairs.py`) already
      populates the per-leg model probabilities, so v2-reader-compatible
      rows already exist in this branch. The remaining data-plumbing
      task is to capture no-vig market probability at decision time on
      the live path.
    * The v2 fields are ALL OPTIONAL. `PairObservationV2.from_row`
      accepts a v1-only row and returns a valid object with the v2
      fields as None. Downstream code that reads v2-only fields is
      responsible for None-handling explicitly.

Once the live capture writes no-vig market probability into the pair
ledger, `market_disagreement_deduction` returns a real, testable
number, and the coherence gate has one more explainable signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


PAIR_OBSERVATION_SCHEMA_VERSION_V2 = "PAIR_OBSERVATION_V2"


@dataclass(frozen=True)
class PairObservationV2:
    """Typed view over a pair-observation row, v2-aware.

    Every field the existing v1 schema exposes is included, so this
    class is a strict superset. The v2-added fields are all Optional
    -- a v1-only row roundtrips through this reader with those fields
    as None, no exception, no default lie.

    Callers that need a v2 field must check for None explicitly. Static
    types make that visible.
    """

    # v1 fields (present on every real pair observation)
    predicted_joint_probability: Optional[float]
    quoted_pair_price: Optional[float]
    same_game: bool
    same_team: bool
    market_pair_type: Optional[str]
    slate_id: Optional[str]
    pair_id: Optional[str]

    # v2-added fields (Optional -- None on v1 rows)
    leg_1_model_probability: Optional[float] = None
    leg_2_model_probability: Optional[float] = None
    leg_1_no_vig_market_probability: Optional[float] = None
    leg_2_no_vig_market_probability: Optional[float] = None
    leg_1_price: Optional[float] = None
    leg_2_price: Optional[float] = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PairObservationV2":
        return cls(
            predicted_joint_probability=_finite(row.get("predicted_joint_probability")),
            quoted_pair_price=_finite(row.get("quoted_pair_price")),
            same_game=bool(row.get("same_game")),
            same_team=bool(row.get("same_team")),
            market_pair_type=(str(row["market_pair_type"]) if row.get("market_pair_type") else None),
            slate_id=(str(row["slate_id"]) if row.get("slate_id") else None),
            pair_id=(str(row["pair_id"]) if row.get("pair_id") else None),
            leg_1_model_probability=_finite(row.get("leg_1_model_probability")),
            leg_2_model_probability=_finite(row.get("leg_2_model_probability")),
            leg_1_no_vig_market_probability=_finite(row.get("leg_1_no_vig_market_probability")),
            leg_2_no_vig_market_probability=_finite(row.get("leg_2_no_vig_market_probability")),
            leg_1_price=_finite(row.get("leg_1_price")),
            leg_2_price=_finite(row.get("leg_2_price")),
        )

    # --- convenience predicates -----------------------------------------

    def has_v2_model_probabilities(self) -> bool:
        return (
            self.leg_1_model_probability is not None
            and self.leg_2_model_probability is not None
        )

    def has_v2_market_probabilities(self) -> bool:
        return (
            self.leg_1_no_vig_market_probability is not None
            and self.leg_2_no_vig_market_probability is not None
        )


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


# --- market-disagreement deduction --------------------------------------

@dataclass(frozen=True)
class MarketDisagreementProfile:
    """Configuration for the market-disagreement deduction.

    The deduction is proportional to the total per-leg gap between the
    model probability and the no-vig market probability, in absolute
    terms. `disagreement_coefficient` scales that gap into the
    deduction. `disagreement_threshold` is a small floor below which
    the deduction is treated as noise and returned as zero -- the model
    and market rarely agree exactly, and small gaps should not steal
    signal from the calibrated joint. `max_deduction` caps the output
    so no single row can dominate.
    """

    disagreement_coefficient: float = 0.30
    disagreement_threshold: float = 0.03
    max_deduction: float = 0.12


def _abs_disagreement(model_p: float, market_p: float) -> float:
    return abs(model_p - market_p)


def market_disagreement_deduction(
    row: Mapping[str, Any],
    *,
    profile: MarketDisagreementProfile | None = None,
) -> float:
    """Compute the market-disagreement deduction for a pair-observation row.

    Returns 0.0 whenever either leg is missing the model probability
    or the no-vig market probability (v1 rows, or v2 rows that captured
    only one signal). The absence is the honest answer: with no signal
    we deduct nothing rather than invent a fabricated penalty. Once
    the live pair-ingest captures no-vig market probability, this
    function starts returning real, testable numbers.
    """
    profile = profile or MarketDisagreementProfile()
    v = PairObservationV2.from_row(row)
    if not (v.has_v2_model_probabilities() and v.has_v2_market_probabilities()):
        return 0.0

    gap_1 = _abs_disagreement(v.leg_1_model_probability, v.leg_1_no_vig_market_probability)
    gap_2 = _abs_disagreement(v.leg_2_model_probability, v.leg_2_no_vig_market_probability)
    if gap_1 < profile.disagreement_threshold:
        gap_1 = 0.0
    if gap_2 < profile.disagreement_threshold:
        gap_2 = 0.0

    total_gap = gap_1 + gap_2
    deduction = profile.disagreement_coefficient * total_gap
    if deduction > profile.max_deduction:
        return profile.max_deduction
    return deduction
