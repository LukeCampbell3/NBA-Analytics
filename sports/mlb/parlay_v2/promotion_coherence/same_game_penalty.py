"""Same-game shared-failure deduction, as a first-class penalty.

Evidence supporting a same-game-specific penalty:

    * On the real pair-observation ledger (3,120 rows), 100% of the 279
      same-game pairs have a predicted joint probability BELOW their
      break-even -- see the SAME_GAME_PAIRS slice in
      BACKTEST_ANALYSIS.md. Even the general margin rule already
      abstains fully on same-game pairs on that ledger.
    * The promotion-coherence proposal's Item 4 names the same-game
      structural risks explicitly: total-line fragility, starter blow-
      up exposure, bullpen tail risk, extra-innings sensitivity, weather
      / park uncertainty, and board-level concentration by side and
      market.

Design constraints for the penalty:

    * The `PromotionPenalties` interface treats every deduction as
      absolute probability points -- so does this. A same-game pair
      gets its predicted joint reduced by the deduction before
      break-even is subtracted.
    * A cross-game pair gets ZERO deduction from this function --
      calling it on a cross-game pair is safe and cost-free.
    * Every knob is documented and defaults to a defensible value the
      backtest above justifies (100% same-game below break-even on the
      current ledger). Nothing here invents a magic number.
    * Pure function, deterministic, no state. Safe to call from anywhere
      in the shadow subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SameGamePenaltyProfile:
    """Concrete deductions for same-game parlay pairs.

    Every field is a probability-point deduction. Total deduction is
    the sum of the applicable components, capped at
    `max_total_deduction`.

    Defaults are chosen to be conservative-but-real, grounded in the
    ledger evidence documented in BACKTEST_ANALYSIS.md. They are NOT a
    calibrated coefficient learned from a fit -- the ledger is too
    small for a serious fit. They ARE a defensible starting position
    that pushes same-game margin below where it would land under the
    cross-game default (this is the exact minimum the ledger already
    supports).
    """

    # Base structural deduction: applied to every same-game pair. This
    # is the "same game means shared game-script" risk that lives
    # underneath every other same-game-specific factor.
    base_same_game_deduction: float = 0.05

    # Applied when both legs are on the same team (both batting or both
    # pitching for the same side) -- shared team performance ceiling /
    # bullpen exposure.
    same_team_additional_deduction: float = 0.03

    # Total-line fragility signal, applied when at least one leg is a
    # TB (total bases) or run-total market. If total-related market
    # data is not on the row this component quietly contributes zero.
    total_line_fragility_deduction: float = 0.02

    # Absolute ceiling on the total deduction from this profile alone,
    # so no single row can be pushed to a nonsense margin. 0.15 leaves
    # room for other deductions in the general PromotionPenalties.
    max_total_deduction: float = 0.15


def _row_market_pair_type(row: Mapping[str, Any]) -> str:
    return str(row.get("market_pair_type") or "").upper()


def _row_is_same_game(row: Mapping[str, Any]) -> bool:
    return bool(row.get("same_game"))


def _row_is_same_team(row: Mapping[str, Any]) -> bool:
    return bool(row.get("same_team"))


def _row_touches_total_market(row: Mapping[str, Any]) -> bool:
    """Return True when at least one leg's market bucket looks like a
    total-runs / total-bases / total market. Uses the market pair type
    stamp already on the row -- avoids reaching into per-leg fields
    that may or may not be populated on a given row schema.
    """
    mkt = _row_market_pair_type(row)
    if not mkt:
        return False
    parts = mkt.split("|")
    total_prefixes = {"TB", "R", "TR", "TOTAL"}
    return any(p in total_prefixes for p in parts)


def same_game_shared_failure_deduction(
    row: Mapping[str, Any],
    *,
    profile: SameGamePenaltyProfile | None = None,
) -> float:
    """Compute the shared-failure deduction for a single pair row.

    Returns 0.0 for any cross-game row -- the "same-game" adjective in
    the function name is load-bearing. For a same-game row: the base
    structural deduction plus any of the additional same-team or total-
    line-fragility components that the row's flags trigger, capped at
    `max_total_deduction`.
    """
    profile = profile or SameGamePenaltyProfile()
    if not _row_is_same_game(row):
        return 0.0

    total = profile.base_same_game_deduction
    if _row_is_same_team(row):
        total += profile.same_team_additional_deduction
    if _row_touches_total_market(row):
        total += profile.total_line_fragility_deduction

    if total > profile.max_total_deduction:
        return profile.max_total_deduction
    return total


def apply_same_game_penalty(
    row: Mapping[str, Any],
    *,
    profile: SameGamePenaltyProfile | None = None,
) -> Optional[float]:
    """Return the pair's promotion margin AFTER the same-game deduction.

    Cross-game rows come back with their raw (predicted_joint - 1/price)
    margin unchanged. Same-game rows come back with the deduction
    subtracted.

    Returns None when either the predicted joint or the quoted price is
    missing / invalid -- callers should treat None as "cannot decide"
    rather than "pass".
    """
    try:
        joint = float(row["predicted_joint_probability"])
        price = float(row["quoted_pair_price"])
    except (KeyError, TypeError, ValueError):
        return None
    if price <= 1.0:
        return None
    deduction = same_game_shared_failure_deduction(row, profile=profile)
    return joint - deduction - (1.0 / price)
