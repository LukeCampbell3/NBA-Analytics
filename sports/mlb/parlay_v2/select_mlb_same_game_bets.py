from __future__ import annotations

"""Real same-game MLB combo selection: pairs two real, cross-market legs
FROM THE SAME REAL GAME (moneyline + full-game total, moneyline + F5
total, full-game total + F5 total), priced with the real joint
probability read directly off game_simulation_model.py's Monte Carlo
trials -- never the naive independence product of each leg's own
marginal probability.

This is the deliverable for the standing decision on same-game parlays:
"real dependence modeling before going live." Two separate, deliberately
distinct gates apply here, mirroring how support.py already separates
REQUIRED dimensions from OBSERVE_ONLY research dimensions:

  1. EACH LEG must individually clear this repo's existing single-leg
     calibration-ledger support gate (calibration/support.py, reused
     verbatim -- the same real >= 20-prior-settled-observations bar
     every other board in this repo starts from). A combo is never
     authorized on the strength of a leg that has no real track record
     of its own.
  2. The COMBO's real joint-vs-naive-market edge/EV must also clear a
     combo-level threshold (MIN_COMBO_ABS_EDGE / MIN_COMBO_EXPECTED_VALUE
     below).

Both gates start unmet (the calibration ledger is empty for this brand-
new same-game policy), so every candidate here starts `candidate_
authorized=False` -- a real, priced, EV-ranked SHADOW_ONLY suggestion,
identical honest posture to every other board this session has built.

REAL, DISCLOSED PRICING LIMITATION: there is no real book-quoted
same-game-parlay price to compare against (The Odds API prices each
market separately; it does not expose a book's own correlation-adjusted
SGP price). The "market" baseline used for the combo edge here is the
NAIVE PRODUCT of the two legs' own real no-vig single-market
probabilities -- i.e., what an unsophisticated, independence-assuming
combination of the real single-leg market would imply. Comparing our
real JOINT (correlated) model probability against that naive baseline is
exactly the right test of whether real dependence modeling earns an
edge over ignoring correlation -- it is NOT a claim that we've beaten a
real market's own SGP price, because no such real price exists to beat
yet. This matches calibration/pair_schema.py's own documented
convention (`quoted_pair_price: None for same-game -- no real SGP
quote`); observations recorded here reuse that schema.
"""

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "predictions"))
import game_simulation_model as sim  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration.pair_schema import build_pair_observation  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from calibration.support import evaluate_support  # noqa: E402
from fanduel_betslip import FANDUEL_SPORTSBOOK_KEY, build_fanduel_betslip_url  # noqa: E402

# A same-game combo's joint probability used to carry its own 50% floor
# here too (borrowed from HIGH_HIT_PARLAY_V1's JOINT_PROBABILITY_FLOOR,
# which gates two independent PLAYER-PROP legs -- each individually much
# more likely to hit on its own). Removed 2026-08-29 after checking every
# real day of production data since this product's odds source went
# live (2026-08-25 through 2026-08-29, 253 real priced combos): the real
# joint probability for a moneyline x game-total/F5-total combo -- two
# TEAM/GAME-level legs, a structurally lower-ceiling combo shape -- never
# once reached even 36%, let alone 50%. The borrowed floor wasn't
# conservative, it was unsatisfiable: candidate_authorized had never
# once been True for a same-game combo since launch. Edge and EV remain
# the real bar for this product; a real same-game-specific probability
# floor can be added later if a settled track record ever justifies one.
MIN_ABS_EDGE = 0.05
MIN_EXPECTED_VALUE = 0.0
MIN_REAL_BOOKS = 1
MIN_COMBO_ABS_EDGE = 0.05
MIN_COMBO_EXPECTED_VALUE = 0.0

# (market, side) -> which the real leg needs from a market_odds row and how
# to read its probability/mask off a GameSimulationResult.
_MARKETS = ("moneyline", "game_total", "first_5_innings_total")


def american_to_decimal(price: Optional[float]) -> Optional[float]:
    if price is None or not math.isfinite(price) or abs(price) < 100.0:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def implied_probability(price: Optional[float]) -> Optional[float]:
    decimal = american_to_decimal(price)
    return None if decimal is None else 1.0 / decimal


def no_vig_two_sided_probabilities(price_a: Optional[float], price_b: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Real no-vig probabilities for a real two-sided market (moneyline
    home/away, or over/under) -- normalizes each side's raw implied
    probability by their real combined sum, the standard way to strip
    vig from a two-outcome market."""
    prob_a = implied_probability(price_a)
    prob_b = implied_probability(price_b)
    if prob_a is None or prob_b is None or (prob_a + prob_b) <= 0:
        return None, None
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def _consensus_line(rows: list[dict]) -> Optional[float]:
    """The real line quoted by the most distinct real books -- a real,
    data-derived consensus, never an arbitrarily chosen one, for markets
    (totals) where different real books can quote different real lines."""
    by_line: dict[float, set[str]] = {}
    for row in rows:
        line = row.get("line")
        if line is None:
            continue
        by_line.setdefault(float(line), set()).add(row.get("sportsbook", ""))
    if not by_line:
        return None
    return max(by_line.keys(), key=lambda line: len(by_line[line]))


@dataclass
class SameGameLeg:
    market: str  # "moneyline" | "game_total" | "first_5_innings_total"
    side: str    # "home" | "away" | "over" | "under"
    line: Optional[float]
    model_probability: float
    no_vig_market_probability: Optional[float]
    price_american: Optional[float]
    sportsbook: str
    market_books: int
    price_confirmed: bool
    leg_authorized: bool
    support_blocking_dimensions: list[str] = field(default_factory=list)
    sportsbook_deeplink: Optional[str] = None

    @property
    def decimal_price(self) -> Optional[float]:
        return american_to_decimal(self.price_american)

    @property
    def event_id(self) -> str:
        line_part = "" if self.line is None else f"|{self.line}"
        return f"{self.market}|{self.side}{line_part}|{self.sportsbook}"

    def mask(self, result: sim.GameSimulationResult) -> np.ndarray:
        if self.market == "moneyline":
            return result.home_win if self.side == "home" else ~result.home_win
        if self.market == "game_total":
            over_mask = result.full_total_over_mask(self.line)
        elif self.market == "first_5_innings_total":
            over_mask = result.f5_total_over_mask(self.line)
        else:
            raise ValueError(f"unknown market: {self.market}")
        return over_mask if self.side == "over" else ~over_mask

    def as_dict(self) -> dict[str, Any]:
        return {
            "market": self.market, "side": self.side, "line": self.line,
            "model_probability": self.model_probability,
            "no_vig_market_probability": self.no_vig_market_probability,
            "price_american": self.price_american, "sportsbook": self.sportsbook,
            "market_books": self.market_books, "price_confirmed": self.price_confirmed,
            "leg_authorized": self.leg_authorized,
            "support_blocking_dimensions": self.support_blocking_dimensions,
            "sportsbook_deeplink": self.sportsbook_deeplink,
        }


def _build_legs_for_market(
    market: str,
    market_odds: list[dict],
    result: sim.GameSimulationResult,
    *,
    calibration_store: Optional[CalibrationStore],
    calibration_as_of: Optional[str],
    min_real_books: int,
) -> list[SameGameLeg]:
    rows = [r for r in market_odds if r.get("target") == market]
    if not rows:
        return []

    if market == "moneyline":
        by_book: dict[str, dict] = {}
        for row in rows:
            book = row.get("sportsbook", "")
            home_ml, away_ml = row.get("home_moneyline"), row.get("away_moneyline")
            if home_ml is None or away_ml is None:
                continue
            by_book[book] = row
        if not by_book:
            return []
        best_book = max(by_book, key=lambda b: american_to_decimal(by_book[b]["home_moneyline"]) or 0.0)
        best_row = by_book[best_book]
        home_price, away_price = best_row["home_moneyline"], best_row["away_moneyline"]
        no_vig_home, no_vig_away = no_vig_two_sided_probabilities(home_price, away_price)
        specs = [
            ("home", home_price, no_vig_home, result.home_win_probability, best_row.get("home_moneyline_deeplink")),
            ("away", away_price, no_vig_away, 1.0 - result.home_win_probability, best_row.get("away_moneyline_deeplink")),
        ]
        line = None
        books_for_market = len(by_book)
    else:
        consensus_line = _consensus_line(rows)
        if consensus_line is None:
            return []
        at_line = [r for r in rows if r.get("line") == consensus_line]
        by_book = {r.get("sportsbook", ""): r for r in at_line}
        best_book = max(by_book, key=lambda b: american_to_decimal(by_book[b].get("over_price")) or 0.0)
        best_row = by_book[best_book]
        over_price, under_price = best_row.get("over_price"), best_row.get("under_price")
        no_vig_over, no_vig_under = no_vig_two_sided_probabilities(over_price, under_price)
        model_over = result.full_total_over_probability(consensus_line) if market == "game_total" else result.f5_total_over_probability(consensus_line)
        specs = [
            ("over", over_price, no_vig_over, model_over, best_row.get("over_deeplink")),
            ("under", under_price, no_vig_under, 1.0 - model_over, best_row.get("under_deeplink")),
        ]
        line = consensus_line
        books_for_market = len(by_book)

    legs: list[SameGameLeg] = []
    for side, price, no_vig_probability, model_probability, deeplink in specs:
        price_confirmed = bool(price is not None and american_to_decimal(price) is not None and books_for_market >= min_real_books)
        # Mirrors golf's select_pga_bets.py: support starts UNMET (never
        # a default-True fallback) -- a leg is authorized only once a
        # real calibration_store is actually consulted AND clears it, so
        # omitting the store never silently authorizes on price alone.
        support_passed = False
        blocking: list[str] = []
        if calibration_store is not None and calibration_as_of is not None:
            snapshot_rows = calibration_store.observations_as_of(calibration_as_of)
            support = evaluate_support(
                snapshot_rows,
                market_bucket=market,
                line_bucket=f"{market}|{side}",
                state_bucket="mlb_same_game_joint_sim_v1",
                independent_slate_count=len({row.get("slate_id") for row in snapshot_rows}),
            )
            support_passed = support.in_support
            blocking = list(support.blocking_dimensions)
        authorized = bool(price_confirmed and support_passed)
        legs.append(
            SameGameLeg(
                market=market, side=side, line=line,
                model_probability=model_probability, no_vig_market_probability=no_vig_probability,
                price_american=price, sportsbook=best_book, market_books=books_for_market,
                price_confirmed=price_confirmed, leg_authorized=authorized,
                support_blocking_dimensions=blocking,
                sportsbook_deeplink=deeplink,
            )
        )
    return legs


@dataclass
class SameGameComboCandidate:
    game_id: str
    home_team: str
    away_team: str
    event_date: str
    leg_a: SameGameLeg
    leg_b: SameGameLeg
    real_joint_model_probability: float
    naive_independence_probability: float
    naive_no_vig_combo_probability: Optional[float]
    naive_market_joint_raw_probability: Optional[float]
    combo_decimal_price: Optional[float]
    probability_edge: Optional[float]
    expected_value_per_unit: Optional[float]
    candidate_authorized: bool
    support_blocking_dimensions: list[str]

    @property
    def joint_residual(self) -> float:
        """How much real correlation modeling actually changed the
        estimate vs. assuming independence -- the direct, observable
        payoff of building the joint simulator instead of multiplying
        marginals."""
        return self.real_joint_model_probability - self.naive_independence_probability

    @property
    def betslip(self) -> dict[str, Any]:
        """Real "Add to Betslip" deep link for this same-game combo --
        this product IS a two-leg parlay (one real correlated wager
        placed as two FanDuel selections in one slip), so it gets the
        exact same real multi-leg deep link construction PARLAY_POLICY_V2
        and the legacy ticket already use (fanduel_betslip.py). Only
        ready when both real legs actually priced at FanDuel -- a combo
        priced off a different book (or off two different books) has no
        single real multi-leg link to offer, and is never faked into one."""
        legs = [
            {"selected_sportsbook_key": self.leg_a.sportsbook, "sportsbook_deeplink": self.leg_a.sportsbook_deeplink},
            {"selected_sportsbook_key": self.leg_b.sportsbook, "sportsbook_deeplink": self.leg_b.sportsbook_deeplink},
        ]
        url = build_fanduel_betslip_url(legs)
        if url is None:
            return {
                "sportsbook_key": FANDUEL_SPORTSBOOK_KEY, "sportsbook": "FanDuel", "status": "unavailable",
                "reason": "one_or_more_legs_have_no_live_fanduel_selection",
            }
        return {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY, "sportsbook": "FanDuel", "status": "ready",
            "leg_count": 2, "url": url, "source": "direct_fanduel_public_market_ids",
        }

    def as_dict(self) -> dict[str, Any]:
        betslip = self.betslip
        return {
            "game_id": self.game_id, "home_team": self.home_team, "away_team": self.away_team,
            "event_date": self.event_date,
            "leg_a": self.leg_a.as_dict(), "leg_b": self.leg_b.as_dict(),
            "real_joint_model_probability": self.real_joint_model_probability,
            "naive_independence_probability": self.naive_independence_probability,
            "joint_residual": self.joint_residual,
            "naive_no_vig_combo_probability": self.naive_no_vig_combo_probability,
            "naive_market_joint_raw_probability": self.naive_market_joint_raw_probability,
            "combo_decimal_price": self.combo_decimal_price,
            "probability_edge": self.probability_edge,
            "expected_value_per_unit": self.expected_value_per_unit,
            "candidate_authorized": self.candidate_authorized,
            "support_blocking_dimensions": self.support_blocking_dimensions,
            "betslip": betslip,
            "betslip_url": betslip.get("url"),
        }


def build_single_leg_team_market_candidates(
    game: dict[str, Any],
    result: sim.GameSimulationResult,
    market_odds: list[dict[str, Any]],
    *,
    calibration_store: Optional[CalibrationStore] = None,
    calibration_as_of: Optional[str] = None,
    min_real_books: int = MIN_REAL_BOOKS,
) -> list[SameGameLeg]:
    """Every real single-market leg (moneyline / game_total /
    first_5_innings_total) this real game has a priced side for -- the
    exact same real legs (same model, same real odds, same
    calibration/support.py REQUIRED gate) `build_same_game_candidates`
    below combines into cross-market pairs, exposed standalone so a
    single-leg consumer (the main single-leg board) can use them without
    needing a same-game partner. `leg.leg_authorized` is the only field
    that should ever gate whether a leg is shown as a live pick -- it is
    computed identically here and inside a combo, so a market/line/state
    bucket's real evidence means the same thing everywhere it's used."""
    legs: list[SameGameLeg] = []
    for market in _MARKETS:
        legs.extend(
            _build_legs_for_market(
                market, market_odds, result,
                calibration_store=calibration_store, calibration_as_of=calibration_as_of, min_real_books=min_real_books,
            )
        )
    return legs


def build_same_game_candidates(
    game: dict[str, Any],
    result: sim.GameSimulationResult,
    market_odds: list[dict[str, Any]],
    *,
    calibration_store: Optional[CalibrationStore] = None,
    calibration_as_of: Optional[str] = None,
    min_real_books: int = MIN_REAL_BOOKS,
    min_abs_edge: float = MIN_ABS_EDGE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    min_combo_abs_edge: float = MIN_COMBO_ABS_EDGE,
    min_combo_expected_value: float = MIN_COMBO_EXPECTED_VALUE,
) -> list[SameGameComboCandidate]:
    """`game`: {"game_id", "date", "home_team", "away_team"}. `market_odds`:
    real rows from TheOddsApiMlbTeamMarketProvider for this one game
    (target in {"moneyline", "game_total", "first_5_innings_total"}).
    Builds every real cross-market 2-leg combo (never same-market
    opposite sides, e.g. never "over total + under total") this real
    game has priced legs for, with the real joint probability read
    directly off `result`'s shared Monte Carlo trials."""
    legs_by_market: dict[str, list[SameGameLeg]] = {
        market: _build_legs_for_market(
            market, market_odds, result,
            calibration_store=calibration_store, calibration_as_of=calibration_as_of, min_real_books=min_real_books,
        )
        for market in _MARKETS
    }

    combos: list[SameGameComboCandidate] = []
    market_pairs = [(_MARKETS[i], _MARKETS[j]) for i in range(len(_MARKETS)) for j in range(i + 1, len(_MARKETS))]
    for market_a, market_b in market_pairs:
        for leg_a in legs_by_market[market_a]:
            for leg_b in legs_by_market[market_b]:
                mask_a, mask_b = leg_a.mask(result), leg_b.mask(result)
                real_joint = result.joint_probability(mask_a, mask_b)
                naive_independence = leg_a.model_probability * leg_b.model_probability

                naive_market = None
                if leg_a.no_vig_market_probability is not None and leg_b.no_vig_market_probability is not None:
                    naive_market = leg_a.no_vig_market_probability * leg_b.no_vig_market_probability

                combo_decimal = None
                if leg_a.decimal_price is not None and leg_b.decimal_price is not None:
                    combo_decimal = leg_a.decimal_price * leg_b.decimal_price

                # Raw (vig-included) market joint -- the naive product of
                # each leg's OWN quoted price, with no de-vigging at all.
                # Kept alongside naive_market (de-vigged) rather than in
                # place of it: probability_edge below is computed against
                # the de-vigged figure (the fairer, preferred baseline),
                # but a viewer reproducing "edge" from the raw displayed
                # odds needs this real intermediate too, not just the
                # final number.
                naive_market_joint_raw = None if combo_decimal is None or combo_decimal == 0 else (1.0 / combo_decimal)

                edge = None if naive_market is None else (real_joint - naive_market)
                expected_value = None if combo_decimal is None else (real_joint * combo_decimal - 1.0)
                price_confirmed = bool(leg_a.price_confirmed and leg_b.price_confirmed)

                authorized = bool(
                    price_confirmed
                    and leg_a.leg_authorized
                    and leg_b.leg_authorized
                    and edge is not None and edge >= min_combo_abs_edge
                    and expected_value is not None and expected_value > min_combo_expected_value
                )
                blocking = sorted(set(leg_a.support_blocking_dimensions) | set(leg_b.support_blocking_dimensions))

                combos.append(
                    SameGameComboCandidate(
                        game_id=str(game.get("game_id", "")), home_team=game.get("home_team", ""),
                        away_team=game.get("away_team", ""), event_date=game.get("date", ""),
                        leg_a=leg_a, leg_b=leg_b,
                        real_joint_model_probability=real_joint, naive_independence_probability=naive_independence,
                        naive_no_vig_combo_probability=naive_market,
                        naive_market_joint_raw_probability=naive_market_joint_raw, combo_decimal_price=combo_decimal,
                        probability_edge=edge, expected_value_per_unit=expected_value,
                        candidate_authorized=authorized, support_blocking_dimensions=blocking,
                    )
                )
    return combos


def top_combo_candidates(candidates: list[SameGameComboCandidate], *, max_candidates: int = 10) -> list[SameGameComboCandidate]:
    """Ranks real, priced combos by expected value -- display/shadow
    ranking only, never a substitute for candidate_authorized."""
    priced = [c for c in candidates if c.expected_value_per_unit is not None]
    priced.sort(key=lambda c: c.expected_value_per_unit, reverse=True)
    return priced[:max_candidates]


def build_pair_observation_for_combo(
    combo: SameGameComboCandidate,
    *,
    slate_id: str,
    predictive_version: str,
    policy_version: str,
    decision_timestamp: str,
    leg_a_result: Optional[int] = None,
    leg_b_result: Optional[int] = None,
    settlement_status: str = "ungraded",
    settlement_timestamp: str = "",
):
    """Records this real combo into the pair-level RESEARCH ledger
    (calibration/pair_schema.py, reused verbatim -- its own documented
    convention already covers a same-game pair with no real SGP quote:
    `quoted_pair_price=None`). OBSERVE_ONLY: this never gates
    `candidate_authorized` (see support.py's joint_support dimension) --
    it exists purely to accumulate real evidence on whether real joint
    modeling continues to diverge from the naive independence baseline
    over time."""
    market_pair_type = "|".join(sorted([combo.leg_a.market, combo.leg_b.market]))
    line_pair_type = "__".join(
        sorted([f"{combo.leg_a.market}|{combo.leg_a.side}|{combo.leg_a.line}", f"{combo.leg_b.market}|{combo.leg_b.side}|{combo.leg_b.line}"])
    )
    return build_pair_observation(
        slate_id=slate_id,
        leg_1_event_id=f"{combo.game_id}|{combo.leg_a.event_id}",
        leg_2_event_id=f"{combo.game_id}|{combo.leg_b.event_id}",
        same_game=True,
        same_team=False,  # legs here are always TEAM-market legs on the whole game, not one specific team
        market_pair_type=market_pair_type,
        line_pair_type=line_pair_type,
        state_bucket_pair="mlb_same_game_joint_sim_v1",
        price_bucket=f"{round(combo.combo_decimal_price, 1) if combo.combo_decimal_price else 'NA'}",
        quoted_pair_price=None,  # no real book-quoted SGP price exists -- see module docstring
        predicted_joint_probability=combo.real_joint_model_probability,
        predicted_independence_probability=combo.naive_independence_probability,
        counterexample_count=0,
        counterexample_mass=0.0,
        retained_world_count=0,
        retained_probability_mass=0.0,
        calibration_snapshot_id=None,
        predictive_version=predictive_version,
        policy_version=policy_version,
        decision_timestamp=decision_timestamp,
        leg_1_result=leg_a_result,
        leg_2_result=leg_b_result,
        settlement_status=settlement_status,
        settlement_timestamp=settlement_timestamp,
    )
