from __future__ import annotations

"""Real single-leg PGA bet selection: joins the score-projection model's
real, field-relative outcome probabilities against real market prices,
computes real expected value, and gates candidates through this repo's
established shadow-until-earned calibration discipline (reusing
sports.golf.parlay_v2.calibration verbatim from MLB/NFL).

No candidate here is ever `candidate_authorized=True` until the real
calibration ledger clears this policy's support thresholds -- until then
every candidate is a real, priced, EV-ranked SHADOW_ONLY suggestion, the
same honest posture MLB and NFL's boards started from.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from odds_provider import OddsRow
from score_model import FieldOutcomeProbabilities

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration.store import CalibrationStore  # noqa: E402
from calibration.support import evaluate_support  # noqa: E402

# Frozen selection policy -- deliberately conservative given this is a
# brand-new, unproven pipeline. Mirrors the spirit of MLB's
# production_action_board gates (min edge, min EV, min real books) but
# starts stricter since there is no real historical track record yet to
# justify looser thresholds.
MIN_ABS_EDGE = 0.05  # model probability vs. no-vig market probability
MIN_EXPECTED_VALUE = 0.0
MIN_REAL_BOOKS = 1
MAX_CANDIDATES_PER_MARKET = 5


def american_to_decimal(price: float) -> Optional[float]:
    if price is None or not math.isfinite(price) or abs(price) < 100.0:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def implied_probability(price: float) -> Optional[float]:
    decimal = american_to_decimal(price)
    return None if decimal is None else 1.0 / decimal


def no_vig_field_probabilities(rows: list[OddsRow], *, market: str, sportsbook_key: str) -> dict[str, float]:
    """Real no-vig probability for every player in one (market, book)'s
    field, normalizing each player's raw implied probability by the sum
    across the WHOLE priced field -- the correct way to remove vig from an
    outright/field market, where vig is spread across many outcomes
    rather than two sides of one line."""
    same_market = [row for row in rows if row.market == market and row.sportsbook_key == sportsbook_key]
    raw: dict[str, float] = {}
    for row in same_market:
        prob = implied_probability(row.price_american)
        if prob is None:
            continue
        raw[row.player_name] = prob
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {name: prob / total for name, prob in raw.items()}


def _best_price_for(rows: list[OddsRow], *, player_name: str, market: str) -> Optional[OddsRow]:
    candidates = [row for row in rows if row.player_name == player_name and row.market == market]
    priced = [row for row in candidates if american_to_decimal(row.price_american) is not None]
    if not priced:
        return None
    return max(priced, key=lambda row: row.price_american)


@dataclass
class PgaCandidate:
    player_id: str
    player_name: str
    market: str
    model_probability: float
    no_vig_market_probability: Optional[float]
    probability_edge: Optional[float]
    selected_side_price: Optional[float]
    selected_sportsbook_key: str
    expected_value_per_unit: Optional[float]
    market_books: int
    price_confirmed: bool
    candidate_authorized: bool
    support_blocking_dimensions: list[str]
    player_headshot_url: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "player_headshot_url": self.player_headshot_url,
            "market": self.market,
            "model_probability": self.model_probability,
            "no_vig_market_probability": self.no_vig_market_probability,
            "probability_edge": self.probability_edge,
            "selected_side_price": self.selected_side_price,
            "selected_sportsbook_key": self.selected_sportsbook_key,
            "expected_value_per_unit": self.expected_value_per_unit,
            "market_books": self.market_books,
            "price_confirmed": self.price_confirmed,
            "candidate_authorized": self.candidate_authorized,
            "support_blocking_dimensions": self.support_blocking_dimensions,
        }


_TARGET_TO_PROBABILITY_FIELD = {
    "WINNER": "win_probability",
    "TOP_5": "top5_probability",
    "TOP_10": "top10_probability",
    "TOP_20": "top20_probability",
    "MAKE_CUT": "make_cut_probability",
}


def build_candidates(
    outcome_probabilities: list[FieldOutcomeProbabilities],
    odds_rows: list[OddsRow],
    *,
    event_id: str,
    calibration_store: Optional[CalibrationStore] = None,
    calibration_as_of: Optional[str] = None,
    min_abs_edge: float = MIN_ABS_EDGE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    min_real_books: int = MIN_REAL_BOOKS,
    player_headshots: Optional[dict[str, str]] = None,
) -> list[PgaCandidate]:
    """Builds one real candidate per (player, market) the model has a
    probability for AND the real market has a real price for. A market
    with no real price anywhere (a real, common state for golf -- see
    odds_provider's module docstring) simply produces no candidates for
    that market, never a fabricated one."""
    candidates: list[PgaCandidate] = []
    no_vig_cache: dict[tuple[str, str], dict[str, float]] = {}
    headshots = player_headshots or {}

    for outcome in outcome_probabilities:
        for market, field_name in _TARGET_TO_PROBABILITY_FIELD.items():
            model_probability = getattr(outcome, field_name)
            if model_probability is None:
                continue  # e.g. MAKE_CUT on a real no-cut event
            best_row = _best_price_for(odds_rows, player_name=outcome.player_name, market=market)
            books_for_market = len({row.sportsbook_key for row in odds_rows if row.player_name == outcome.player_name and row.market == market})

            no_vig_probability: Optional[float] = None
            if best_row is not None:
                cache_key = (market, best_row.sportsbook_key)
                if cache_key not in no_vig_cache:
                    no_vig_cache[cache_key] = no_vig_field_probabilities(odds_rows, market=market, sportsbook_key=best_row.sportsbook_key)
                no_vig_probability = no_vig_cache[cache_key].get(outcome.player_name)

            edge = None if no_vig_probability is None else (model_probability - no_vig_probability)
            decimal_price = None if best_row is None else american_to_decimal(best_row.price_american)
            expected_value = None if decimal_price is None else (model_probability * decimal_price - 1.0)
            price_confirmed = bool(best_row is not None and decimal_price is not None and books_for_market >= min_real_books)

            support_blocking: list[str] = []
            support_passed = False
            if calibration_store is not None and calibration_as_of is not None:
                snapshot_rows = calibration_store.observations_as_of(calibration_as_of)
                support = evaluate_support(
                    snapshot_rows,
                    market_bucket=market,
                    line_bucket=f"{market}|{outcome.player_id}",
                    state_bucket="pga_field_relative_form_v1",
                    independent_slate_count=len({row.get("slate_id") for row in snapshot_rows}),
                )
                support_passed = support.in_support
                support_blocking = list(support.blocking_dimensions)

            authorized = bool(
                price_confirmed
                and edge is not None
                and edge >= min_abs_edge
                and expected_value is not None
                and expected_value > min_expected_value
                and support_passed
            )

            candidates.append(
                PgaCandidate(
                    player_id=outcome.player_id,
                    player_name=outcome.player_name,
                    market=market,
                    model_probability=model_probability,
                    no_vig_market_probability=no_vig_probability,
                    probability_edge=edge,
                    selected_side_price=(best_row.price_american if best_row else None),
                    selected_sportsbook_key=(best_row.sportsbook_key if best_row else ""),
                    expected_value_per_unit=expected_value,
                    market_books=books_for_market,
                    price_confirmed=price_confirmed,
                    candidate_authorized=authorized,
                    support_blocking_dimensions=support_blocking,
                    player_headshot_url=headshots.get(outcome.player_id, ""),
                )
            )

    return candidates


def top_candidates_per_market(candidates: list[PgaCandidate], *, max_per_market: int = MAX_CANDIDATES_PER_MARKET) -> list[PgaCandidate]:
    """Ranks real, priced candidates by expected value within each market
    -- display/shadow ranking only, never a substitute for
    candidate_authorized when deciding what is actually stakeable."""
    by_market: dict[str, list[PgaCandidate]] = {}
    for candidate in candidates:
        if candidate.expected_value_per_unit is None:
            continue
        by_market.setdefault(candidate.market, []).append(candidate)
    ranked: list[PgaCandidate] = []
    for market_candidates in by_market.values():
        market_candidates.sort(key=lambda c: c.expected_value_per_unit, reverse=True)
        ranked.extend(market_candidates[:max_per_market])
    return ranked
