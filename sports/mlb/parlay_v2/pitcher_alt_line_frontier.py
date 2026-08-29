from __future__ import annotations

"""ROI-aware pitcher strikeout alt-line frontier.

The FanDuel provider already exposes both standard pitcher-K totals and
one-sided alternate strikeout thresholds. The legacy pitcher-parlay selector
collapsed those real prices to one consensus line per pitcher and then chose
the two highest-probability pitchers before looking at price. That can create
an apparently "safe" parlay made from brutally expensive legs such as -2500
or -2200, even when harder real lines retain strong model probability and pay
materially better.

This additive selector keeps the predictive model and calibration evidence
unchanged, enumerates every real line, and solves the constrained problem we
actually want for the ROI-oriented pitcher board:

    maximize quoted-price model EV
    subject to each leg probability >= 60%
               each leg model EV >= 0%
               each leg decimal price >= 1.20  (no worse than -500)
               combined probability >= 50%
               combined decimal price >= 2.00  (at least +100)
               combined model EV >= 5%
               two distinct games/pitchers

If nothing clears those constraints, the board abstains. It does not fall
back to an ultra-short negative-EV pair merely to manufacture a selection.
"""

import math
from typing import Any, Callable, Optional

from calibration.store import CalibrationStore
from calibration.support import evaluate_support
from select_mlb_pitcher_parlay import (
    MIN_COMBO_ABS_EDGE,
    MIN_COMBO_JOINT_PROBABILITY,
    MIN_REAL_BOOKS,
    STATE_BUCKET,
    PitcherKLeg,
    PitcherParlayCandidate,
    _normalize_name,
    american_to_decimal,
    no_vig_two_sided_probabilities,
)
import pitcher_strikeout_model as k_model


# ROI-oriented SHADOW selection gates. These are intentionally separate from
# the frozen/certification gates used elsewhere in the repo.
MIN_LEG_PROBABILITY = 0.60
MIN_LEG_EXPECTED_VALUE = 0.0
MIN_LEG_DECIMAL_PRICE = 1.20  # equivalent to -500 American odds
MIN_COMBO_DECIMAL_PRICE = 2.00  # at least +100 combined payout
MIN_COMBO_EXPECTED_VALUE = 0.05


def _best_price(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    priced = [row for row in rows if american_to_decimal(row.get("price_american")) is not None]
    return max(priced, key=lambda row: american_to_decimal(row.get("price_american")) or 0.0, default=None)


def build_pitcher_k_alt_line_legs(
    starters: list[dict[str, Any]],
    odds_rows: list[dict[str, Any]],
    *,
    season: int,
    calibration_store: Optional[CalibrationStore] = None,
    calibration_as_of: Optional[str] = None,
    min_real_books: int = MIN_REAL_BOOKS,
    fetch_season_stats: Callable[..., Any] = k_model.fetch_pitcher_season_stats,
) -> list[PitcherKLeg]:
    """Build one leg for every real priced pitcher/line/side combination.

    Unlike the legacy constructor, no consensus-line collapse is performed.
    A one-sided FanDuel alternate threshold is valid real pricing; its no-vig
    probability is deliberately ``None`` unless the opposite side is also
    quoted at that exact line.
    """

    k_rows = [row for row in odds_rows if row.get("market_type") == "pitcher_strikeouts"]
    rows_by_player: dict[str, list[dict[str, Any]]] = {}
    for row in k_rows:
        rows_by_player.setdefault(_normalize_name(row.get("player_name")), []).append(row)

    snapshot_rows = (
        calibration_store.observations_as_of(calibration_as_of)
        if calibration_store is not None and calibration_as_of is not None
        else []
    )
    independent_slates = len({row.get("slate_id") for row in snapshot_rows})

    legs: list[PitcherKLeg] = []
    for starter in starters:
        pitcher_id = starter.get("pitcher_id")
        pitcher_name = str(starter.get("pitcher_name") or "").strip()
        if not pitcher_id or not pitcher_name:
            continue

        projection = fetch_season_stats(int(pitcher_id), season, name=pitcher_name)
        projected_mean = projection.projected_mean_strikeouts if projection else None
        if projected_mean is None:
            continue

        rows = rows_by_player.get(_normalize_name(pitcher_name), [])
        if not rows:
            continue

        by_line: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            try:
                line = float(row.get("line"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(line) or line < 0:
                continue
            by_line.setdefault(line, []).append(row)

        for line in sorted(by_line):
            line_rows = by_line[line]
            over_rows = [row for row in line_rows if str(row.get("side") or "").lower() == "over"]
            under_rows = [row for row in line_rows if str(row.get("side") or "").lower() == "under"]
            best_over = _best_price(over_rows)
            best_under = _best_price(under_rows)
            over_price = best_over.get("price_american") if best_over else None
            under_price = best_under.get("price_american") if best_under else None
            no_vig_over, no_vig_under = no_vig_two_sided_probabilities(over_price, under_price)

            model_over = k_model.poisson_over_probability(line, projected_mean)
            if model_over is None:
                continue

            books_for_market = len({str(row.get("sportsbook") or "") for row in line_rows if row.get("sportsbook")})
            specs = [
                ("over", best_over, model_over, no_vig_over),
                ("under", best_under, 1.0 - model_over, no_vig_under),
            ]
            for side, best_row, model_probability, no_vig_probability in specs:
                if best_row is None:
                    continue
                price = best_row.get("price_american")
                price_confirmed = bool(
                    american_to_decimal(price) is not None and books_for_market >= min_real_books
                )

                support_passed = False
                blocking: list[str] = []
                if snapshot_rows:
                    support = evaluate_support(
                        snapshot_rows,
                        market_bucket="pitcher_strikeouts",
                        # Preserve the legacy evidence bucket so this additive
                        # line frontier does not silently create a new support
                        # universe just because a different real threshold is
                        # chosen for the same pitcher/side.
                        line_bucket=f"pitcher_strikeouts|{pitcher_id}|{side}",
                        state_bucket=STATE_BUCKET,
                        independent_slate_count=independent_slates,
                    )
                    support_passed = support.in_support
                    blocking = list(support.blocking_dimensions)

                legs.append(
                    PitcherKLeg(
                        pitcher_id=int(pitcher_id),
                        pitcher_name=pitcher_name,
                        team=str(starter.get("team") or ""),
                        opponent=str(starter.get("opponent") or ""),
                        game_id=str(starter.get("game_id") or ""),
                        line=line,
                        side=side,
                        model_probability=float(model_probability),
                        no_vig_market_probability=no_vig_probability,
                        price_american=price,
                        sportsbook=str(best_row.get("sportsbook") or ""),
                        market_books=books_for_market,
                        price_confirmed=price_confirmed,
                        leg_authorized=bool(price_confirmed and support_passed),
                        support_blocking_dimensions=blocking,
                        sportsbook_deeplink=best_row.get("sportsbook_deeplink"),
                    )
                )

    return legs


def _pair_candidate(
    leg_a: PitcherKLeg,
    leg_b: PitcherKLeg,
    *,
    min_combo_abs_edge: float,
    min_combo_expected_value: float,
    min_combo_joint_probability: float,
) -> PitcherParlayCandidate:
    joint_probability = leg_a.model_probability * leg_b.model_probability
    naive_no_vig = (
        leg_a.no_vig_market_probability * leg_b.no_vig_market_probability
        if leg_a.no_vig_market_probability is not None and leg_b.no_vig_market_probability is not None
        else None
    )
    decimal_a, decimal_b = leg_a.decimal_price, leg_b.decimal_price
    combo_decimal_price = decimal_a * decimal_b if decimal_a is not None and decimal_b is not None else None
    naive_market_joint_raw = None if combo_decimal_price in (None, 0) else 1.0 / combo_decimal_price
    probability_edge = joint_probability - naive_no_vig if naive_no_vig is not None else None
    expected_value = joint_probability * combo_decimal_price - 1.0 if combo_decimal_price is not None else None

    # Authorization still obeys the original calibration/support discipline.
    # One-sided alternate thresholds may have no no-vig counterpart, so they
    # remain shadow-only until their own evidence path is mature.
    edge_passed = probability_edge is not None and probability_edge >= min_combo_abs_edge
    ev_passed = expected_value is not None and expected_value > min_combo_expected_value
    authorized = bool(
        leg_a.leg_authorized
        and leg_b.leg_authorized
        and joint_probability >= min_combo_joint_probability
        and edge_passed
        and ev_passed
    )
    return PitcherParlayCandidate(
        leg_a=leg_a,
        leg_b=leg_b,
        naive_independence_probability=joint_probability,
        naive_no_vig_combo_probability=naive_no_vig,
        naive_market_joint_raw_probability=naive_market_joint_raw,
        combo_decimal_price=combo_decimal_price,
        probability_edge=probability_edge,
        expected_value_per_unit=expected_value,
        candidate_authorized=authorized,
    )


def build_pitcher_parlay_frontier(
    legs: list[PitcherKLeg],
    *,
    min_leg_probability: float = MIN_LEG_PROBABILITY,
    min_leg_expected_value: float = MIN_LEG_EXPECTED_VALUE,
    min_leg_decimal_price: float = MIN_LEG_DECIMAL_PRICE,
    min_combo_joint_probability: float = MIN_COMBO_JOINT_PROBABILITY,
    min_combo_decimal_price: float = MIN_COMBO_DECIMAL_PRICE,
    min_combo_abs_edge: float = MIN_COMBO_ABS_EDGE,
    min_combo_expected_value: float = MIN_COMBO_EXPECTED_VALUE,
) -> Optional[PitcherParlayCandidate]:
    """Maximize EV among all real ROI-eligible, probability-safe pairs.

    This intentionally rejects an ultra-short favorite even if it raises raw
    hit probability. A leg must contribute both probability and price value.
    """

    eligible = []
    for leg in legs:
        decimal = leg.decimal_price
        leg_ev = leg.expected_value_per_unit
        if not leg.price_confirmed or decimal is None or leg_ev is None:
            continue
        if leg.model_probability < min_leg_probability:
            continue
        if leg_ev < min_leg_expected_value:
            continue
        if decimal < min_leg_decimal_price:
            continue
        eligible.append(leg)

    if len({leg.pitcher_id for leg in eligible}) < 2:
        return None

    pairs: list[PitcherParlayCandidate] = []
    for index, leg_a in enumerate(eligible):
        for leg_b in eligible[index + 1 :]:
            if leg_a.pitcher_id == leg_b.pitcher_id:
                continue
            # Cross-game product only: never claim independence for two
            # pitchers participating in the same real game.
            if leg_a.game_id and leg_b.game_id and leg_a.game_id == leg_b.game_id:
                continue
            joint_probability = leg_a.model_probability * leg_b.model_probability
            if joint_probability < min_combo_joint_probability:
                continue
            decimal_a, decimal_b = leg_a.decimal_price, leg_b.decimal_price
            if decimal_a is None or decimal_b is None:
                continue
            combo_decimal = decimal_a * decimal_b
            if combo_decimal < min_combo_decimal_price:
                continue

            candidate = _pair_candidate(
                leg_a,
                leg_b,
                min_combo_abs_edge=min_combo_abs_edge,
                min_combo_expected_value=min_combo_expected_value,
                min_combo_joint_probability=min_combo_joint_probability,
            )
            if candidate.expected_value_per_unit is None:
                continue
            if candidate.expected_value_per_unit < min_combo_expected_value:
                continue
            pairs.append(candidate)

    if not pairs:
        return None

    # ROI is the primary objective once every probability and payout gate has
    # passed. If EV ties, prefer the bigger actual payout, then the higher hit
    # probability. This is intentionally different from the old max-hit rule.
    pairs.sort(
        key=lambda candidate: (
            float(candidate.expected_value_per_unit),
            float(candidate.combo_decimal_price or 0.0),
            float(candidate.naive_independence_probability),
        ),
        reverse=True,
    )
    return pairs[0]
