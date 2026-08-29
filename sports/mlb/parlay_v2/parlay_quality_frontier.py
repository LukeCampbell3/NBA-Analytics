from __future__ import annotations

"""Additive quality-frontier selection for MLB parlay research products.

This module does not modify the frozen PARLAY_POLICY_V2 path.

It fixes two narrower selection failures in the non-frozen research boards:

1. Pitcher K parlays previously collapsed every pitcher to one consensus
   strikeout line and then ranked the surviving legs only by hit probability.
   That can select extremely short alt-over prices even when a higher real
   FanDuel K rung still clears the same probability floor and offers much
   better expected return.  Here we retain every real quoted K line and solve
   the intended constrained problem: maximize model EV *inside* a minimum
   leg/joint-probability region.

2. Same-game candidates already had a 50% joint-probability authorization
   floor, but low-probability/high-synthetic-EV candidates still remained in
   ``combo_candidates`` and could therefore be featured by the shadow UI.
   ``apply_same_game_probability_frontier`` moves those rows to a retained
   research-only list and leaves only probability-safe, positive-value rows in
   the primary candidate list.  If no combo clears the existing floor, the
   correct primary answer is abstention rather than showcasing a 20-30% SGP.

All thresholds default to constants already used by the existing selectors;
no result from a settled game is used to fit a new cutoff here.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import select_mlb_pitcher_parlay as pitcher_select
import select_mlb_same_game_bets as same_game_select
from calibration.support import evaluate_support


PITCHER_MIN_LEG_PROBABILITY = 0.70
PITCHER_MIN_JOINT_PROBABILITY = pitcher_select.MIN_COMBO_JOINT_PROBABILITY
SAME_GAME_MIN_JOINT_PROBABILITY = same_game_select.MIN_COMBO_JOINT_PROBABILITY


@dataclass(frozen=True)
class PitcherFrontierSelection:
    candidate: Optional[pitcher_select.PitcherParlayCandidate]
    selection_mode: str
    priced_pair_count: int
    probability_safe_pair_count: int
    positive_ev_pair_count: int
    min_leg_probability: float
    min_joint_probability: float

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy": "pitcher_alt_line_value_frontier_v1",
            "selection_mode": self.selection_mode,
            "priced_pair_count": self.priced_pair_count,
            "probability_safe_pair_count": self.probability_safe_pair_count,
            "positive_ev_pair_count": self.positive_ev_pair_count,
            "min_leg_probability": self.min_leg_probability,
            "min_joint_probability": self.min_joint_probability,
            "objective": "maximize_model_ev_subject_to_leg_and_joint_probability_floors",
        }


def _best_side_row(rows: list[dict[str, Any]], side: str) -> Optional[dict[str, Any]]:
    side_rows = [row for row in rows if str(row.get("side") or "").lower() == side]
    if not side_rows:
        return None
    return max(
        side_rows,
        key=lambda row: pitcher_select.american_to_decimal(row.get("price_american")) or 0.0,
    )


def build_pitcher_alt_line_legs(
    starters: list[dict[str, Any]],
    odds_rows: list[dict[str, Any]],
    *,
    season: int,
    calibration_store=None,
    calibration_as_of: Optional[str] = None,
    min_real_books: int = pitcher_select.MIN_REAL_BOOKS,
    fetch_season_stats: Callable[..., Any] = pitcher_select.k_model.fetch_pitcher_season_stats,
) -> list[pitcher_select.PitcherKLeg]:
    """Build one real model leg for every quoted pitcher-K line/side.

    The legacy builder intentionally chooses ``_consensus_line(rows)`` first.
    That is sensible for one canonical market view, but it destroys the alt
    line ladder before the parlay selector gets a chance to trade a small
    amount of hit probability for materially better price.  This builder keeps
    every distinct real line and otherwise mirrors the existing projection,
    no-vig, price-confirmation and calibration-support logic.
    """

    k_rows = [row for row in odds_rows if row.get("market_type") == "pitcher_strikeouts"]
    rows_by_player: dict[str, list[dict[str, Any]]] = {}
    for row in k_rows:
        rows_by_player.setdefault(pitcher_select._normalize_name(row.get("player_name")), []).append(row)

    snapshot_rows = None
    independent_slate_count = 0
    if calibration_store is not None and calibration_as_of is not None:
        snapshot_rows = calibration_store.observations_as_of(calibration_as_of)
        independent_slate_count = len({row.get("slate_id") for row in snapshot_rows})

    legs: list[pitcher_select.PitcherKLeg] = []
    for starter in starters:
        pitcher_id = starter.get("pitcher_id")
        pitcher_name = str(starter.get("pitcher_name") or "").strip()
        if not pitcher_id or not pitcher_name:
            continue

        projection = fetch_season_stats(int(pitcher_id), season, name=pitcher_name)
        projected_mean = projection.projected_mean_strikeouts if projection else None
        if projected_mean is None:
            continue

        player_rows = rows_by_player.get(pitcher_select._normalize_name(pitcher_name), [])
        real_lines = sorted(
            {
                float(row["line"])
                for row in player_rows
                if row.get("line") is not None
            }
        )
        for line in real_lines:
            at_line = [row for row in player_rows if row.get("line") is not None and float(row["line"]) == line]
            if not at_line:
                continue

            best_over = _best_side_row(at_line, "over")
            best_under = _best_side_row(at_line, "under")
            over_price = best_over.get("price_american") if best_over else None
            under_price = best_under.get("price_american") if best_under else None
            no_vig_over, no_vig_under = pitcher_select.no_vig_two_sided_probabilities(over_price, under_price)
            model_over = pitcher_select.k_model.poisson_over_probability(line, projected_mean)
            if model_over is None:
                continue

            distinct_books = {str(row.get("sportsbook") or "") for row in at_line if row.get("sportsbook")}
            books_for_market = len(distinct_books)
            specs = (
                ("over", over_price, no_vig_over, model_over, best_over),
                ("under", under_price, no_vig_under, 1.0 - model_over, best_under),
            )
            for side, price, no_vig_probability, model_probability, price_row in specs:
                if price is None or price_row is None:
                    continue

                price_confirmed = bool(
                    pitcher_select.american_to_decimal(price) is not None
                    and books_for_market >= min_real_books
                )
                support_passed = False
                blocking: list[str] = []
                if snapshot_rows is not None:
                    support = evaluate_support(
                        snapshot_rows,
                        market_bucket="pitcher_strikeouts",
                        line_bucket=f"pitcher_strikeouts|{pitcher_id}|{side}",
                        state_bucket=pitcher_select.STATE_BUCKET,
                        independent_slate_count=independent_slate_count,
                    )
                    support_passed = support.in_support
                    blocking = list(support.blocking_dimensions)

                legs.append(
                    pitcher_select.PitcherKLeg(
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
                        sportsbook=str(price_row.get("sportsbook") or ""),
                        market_books=books_for_market,
                        price_confirmed=price_confirmed,
                        leg_authorized=bool(price_confirmed and support_passed),
                        support_blocking_dimensions=blocking,
                        sportsbook_deeplink=price_row.get("sportsbook_deeplink"),
                    )
                )

    return legs


def _pitcher_combo(
    leg_a: pitcher_select.PitcherKLeg,
    leg_b: pitcher_select.PitcherKLeg,
    *,
    min_joint_probability: float,
) -> pitcher_select.PitcherParlayCandidate:
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

    combo_edge_passed = probability_edge is not None and probability_edge >= pitcher_select.MIN_COMBO_ABS_EDGE
    combo_ev_passed = expected_value is not None and expected_value > pitcher_select.MIN_COMBO_EXPECTED_VALUE
    candidate_authorized = bool(
        leg_a.leg_authorized
        and leg_b.leg_authorized
        and joint_probability >= min_joint_probability
        and combo_edge_passed
        and combo_ev_passed
    )

    return pitcher_select.PitcherParlayCandidate(
        leg_a=leg_a,
        leg_b=leg_b,
        naive_independence_probability=joint_probability,
        naive_no_vig_combo_probability=naive_no_vig,
        naive_market_joint_raw_probability=naive_market_joint_raw,
        combo_decimal_price=combo_decimal_price,
        probability_edge=probability_edge,
        expected_value_per_unit=expected_value,
        candidate_authorized=candidate_authorized,
    )


def select_pitcher_value_frontier(
    legs: list[pitcher_select.PitcherKLeg],
    *,
    min_leg_probability: float = PITCHER_MIN_LEG_PROBABILITY,
    min_joint_probability: float = PITCHER_MIN_JOINT_PROBABILITY,
) -> PitcherFrontierSelection:
    """Choose the best-priced probability-safe pitcher pair.

    Primary selection is **not** highest probability and **not** raw EV.
    It is maximum model EV among pairs whose two real legs each clear the
    leg floor and whose joint hit probability clears the combo floor.

    When that probability-safe region exists but every real price is
    negative-EV, the function deliberately falls back to the highest-hit
    probability pair with ``selection_mode=high_hit_price_fail``.  This is
    useful calibration research, but it makes the economic failure explicit
    instead of presenting a -2500/-2200 pair as an efficient wager.
    """

    priced = [
        leg
        for leg in legs
        if leg.price_confirmed
        and leg.decimal_price is not None
        and leg.model_probability >= min_leg_probability
    ]

    all_pairs: list[pitcher_select.PitcherParlayCandidate] = []
    for i, leg_a in enumerate(priced):
        for leg_b in priced[i + 1 :]:
            if leg_a.pitcher_id == leg_b.pitcher_id:
                continue
            if leg_a.game_id and leg_b.game_id and leg_a.game_id == leg_b.game_id:
                continue
            all_pairs.append(_pitcher_combo(leg_a, leg_b, min_joint_probability=min_joint_probability))

    probability_safe = [
        combo
        for combo in all_pairs
        if combo.naive_independence_probability >= min_joint_probability
    ]
    positive_ev = [
        combo
        for combo in probability_safe
        if combo.expected_value_per_unit is not None and combo.expected_value_per_unit > 0.0
    ]

    if positive_ev:
        chosen = max(
            positive_ev,
            key=lambda combo: (
                float(combo.expected_value_per_unit or float("-inf")),
                combo.naive_independence_probability,
            ),
        )
        mode = "frontier_value"
    elif probability_safe:
        chosen = max(
            probability_safe,
            key=lambda combo: (
                combo.naive_independence_probability,
                float(combo.expected_value_per_unit or float("-inf")),
            ),
        )
        mode = "high_hit_price_fail"
    else:
        chosen = None
        mode = "abstain_no_probability_safe_pair"

    return PitcherFrontierSelection(
        candidate=chosen,
        selection_mode=mode,
        priced_pair_count=len(all_pairs),
        probability_safe_pair_count=len(probability_safe),
        positive_ev_pair_count=len(positive_ev),
        min_leg_probability=min_leg_probability,
        min_joint_probability=min_joint_probability,
    )


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def apply_same_game_probability_frontier(
    payload: dict[str, Any],
    *,
    min_joint_probability: float = SAME_GAME_MIN_JOINT_PROBABILITY,
    min_combo_expected_value: float = same_game_select.MIN_COMBO_EXPECTED_VALUE,
    min_combo_abs_edge: float = same_game_select.MIN_COMBO_ABS_EDGE,
) -> dict[str, Any]:
    """Move low-hit same-game rows out of the primary shadow candidate set.

    The original rows remain available under ``ev_research_combo_candidates``
    for analysis.  The existing frontend consumes ``combo_candidates``;
    therefore a day with no >=50% joint, positive-value combo now correctly
    renders as an abstention instead of promoting a 24.7% high-EV research
    position to the main SGP card.
    """

    games = payload.get("games") if isinstance(payload, dict) else None
    if not isinstance(games, list):
        return payload

    total_original = 0
    total_primary = 0
    total_research = 0
    for game in games:
        if not isinstance(game, dict):
            continue
        original = game.get("combo_candidates")
        if not isinstance(original, list):
            continue

        original = [row for row in original if isinstance(row, dict)]
        total_original += len(original)
        primary: list[dict[str, Any]] = []
        research: list[dict[str, Any]] = []

        for row in original:
            joint_probability = _finite_number(row.get("real_joint_model_probability"))
            expected_value = _finite_number(row.get("expected_value_per_unit"))
            probability_edge = _finite_number(row.get("probability_edge"))
            passes = bool(
                joint_probability is not None
                and joint_probability >= min_joint_probability
                and expected_value is not None
                and expected_value > min_combo_expected_value
                and probability_edge is not None
                and probability_edge >= min_combo_abs_edge
            )
            (primary if passes else research).append(row)

        primary.sort(
            key=lambda row: (
                _finite_number(row.get("expected_value_per_unit")) or float("-inf"),
                _finite_number(row.get("real_joint_model_probability")) or float("-inf"),
            ),
            reverse=True,
        )
        research.sort(
            key=lambda row: (
                _finite_number(row.get("expected_value_per_unit")) or float("-inf"),
                _finite_number(row.get("real_joint_model_probability")) or float("-inf"),
            ),
            reverse=True,
        )

        # Keep every original row for audit/research, but only probability-safe
        # rows remain eligible for primary shadow rendering.
        game["combo_candidates"] = primary
        game["ev_research_combo_candidates"] = research
        game["quality_frontier"] = {
            "policy": "same_game_probability_value_frontier_v1",
            "min_joint_probability": min_joint_probability,
            "min_combo_expected_value": min_combo_expected_value,
            "min_combo_abs_edge": min_combo_abs_edge,
            "primary_candidate_count": len(primary),
            "research_only_candidate_count": len(research),
            "decision": "candidate" if primary else "abstain",
        }
        # This count is the count the primary board should reason over now.
        game["candidate_authorized_count"] = sum(1 for row in primary if row.get("candidate_authorized"))
        total_primary += len(primary)
        total_research += len(research)

    payload["quality_frontier"] = {
        "policy": "same_game_probability_value_frontier_v1",
        "min_joint_probability": min_joint_probability,
        "min_combo_expected_value": min_combo_expected_value,
        "min_combo_abs_edge": min_combo_abs_edge,
        "original_candidate_count": total_original,
        "primary_candidate_count": total_primary,
        "research_only_candidate_count": total_research,
        "decision": "candidate" if total_primary else "abstain",
        "objective": "maximize_synthetic_ev_only_after_joint_probability_edge_and_value_gates",
    }
    payload["candidate_authorized_count"] = sum(
        int(game.get("candidate_authorized_count") or 0)
        for game in games
        if isinstance(game, dict)
    )
    return payload
