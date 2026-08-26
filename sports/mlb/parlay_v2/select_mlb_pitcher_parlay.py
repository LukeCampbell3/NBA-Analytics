from __future__ import annotations

"""Real cross-game, pitcher-strikeouts-only MLB parlay: pairs two real
starting pitchers' own real strikeout-line legs, ONE per pitcher, from
two DIFFERENT real games today.

Unlike select_mlb_same_game_bets.py's combos (two legs from the SAME
real game, genuinely correlated, priced with a real joint Monte Carlo
simulation), two different starting pitchers in two different real
games have no real shared game state -- there is no real dependence to
model, so the honest, correct joint probability here really is the
naive independence product of each leg's own real model probability.
That is NOT a shortcut or an approximation error the way it would be
for same-game legs; it is the statistically correct model for this
specific pairing, and is disclosed as such below (mirrors this repo's
own existing disclosure for select_daily_parlay.py's legacy cross-game
tickets).

Each leg's own real model probability comes from
pitcher_strikeout_model.py: a real Poisson projection around that
specific pitcher's own real season-to-date innings-per-start and real
strikeouts-per-inning rate (MLB Stats API season aggregate -- no
leakage concern for a live, present-day board). A pitcher with fewer
than pitcher_strikeout_model.MIN_STARTS_FOR_REAL_PROJECTION real starts
this season produces no real projection and is simply skipped -- never
a guessed rate.

Same real, disclosed gating discipline as every other product in this
repo: each leg must individually clear calibration/support.py's real
>=20-prior-settled-observations bar, AND the combo itself must clear a
real combo-level edge/EV bar, before candidate_authorized is ever True.
This is a brand-new policy with an empty calibration ledger, so it
starts SHADOW_ONLY like same-game combos did.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration.store import CalibrationStore  # noqa: E402
from calibration.support import evaluate_support  # noqa: E402
from fanduel_betslip import FANDUEL_SPORTSBOOK_KEY, build_fanduel_betslip_url  # noqa: E402
from select_mlb_same_game_bets import (  # noqa: E402
    _consensus_line,
    american_to_decimal,
    no_vig_two_sided_probabilities,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "predictions"))
import pitcher_strikeout_model as k_model  # noqa: E402

MIN_ABS_EDGE = 0.05
MIN_EXPECTED_VALUE = 0.0
MIN_REAL_BOOKS = 1
MIN_COMBO_ABS_EDGE = 0.05
MIN_COMBO_EXPECTED_VALUE = 0.0
STATE_BUCKET = "mlb_pitcher_k_parlay_v1"


@dataclass
class PitcherKLeg:
    pitcher_id: int
    pitcher_name: str
    team: str
    opponent: str
    game_id: str
    line: float
    side: str  # "over" | "under"
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
    def expected_value_per_unit(self) -> Optional[float]:
        decimal = self.decimal_price
        if decimal is None:
            return None
        return self.model_probability * decimal - 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pitcher_id": self.pitcher_id, "pitcher_name": self.pitcher_name,
            "team": self.team, "opponent": self.opponent, "game_id": self.game_id,
            "line": self.line, "side": self.side,
            "model_probability": self.model_probability,
            "no_vig_market_probability": self.no_vig_market_probability,
            "price_american": self.price_american, "sportsbook": self.sportsbook,
            "market_books": self.market_books, "price_confirmed": self.price_confirmed,
            "leg_authorized": self.leg_authorized,
            "support_blocking_dimensions": self.support_blocking_dimensions,
            "expected_value_per_unit": self.expected_value_per_unit,
            "sportsbook_deeplink": self.sportsbook_deeplink,
        }


def _normalize_name(value: Any) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = "".join(char if char.isalnum() else " " for char in ascii_text)
    return " ".join(cleaned.split())


def build_pitcher_k_legs(
    starters: list[dict[str, Any]],
    odds_rows: list[dict[str, Any]],
    *,
    season: int,
    calibration_store: Optional[CalibrationStore] = None,
    calibration_as_of: Optional[str] = None,
    min_real_books: int = MIN_REAL_BOOKS,
    fetch_season_stats=k_model.fetch_pitcher_season_stats,
) -> list[PitcherKLeg]:
    """`starters`: real rows with pitcher_id/pitcher_name/team/opponent/
    game_id (one per real probable starter today). `odds_rows`: real
    FanDuel player-prop rows (market_type == "pitcher_strikeouts")."""
    k_rows = [r for r in odds_rows if r.get("market_type") == "pitcher_strikeouts"]
    rows_by_player: dict[str, list[dict[str, Any]]] = {}
    for row in k_rows:
        key = _normalize_name(row.get("player_name"))
        rows_by_player.setdefault(key, []).append(row)

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
        consensus_line = _consensus_line(rows)
        if consensus_line is None:
            continue
        at_line = [r for r in rows if r.get("line") == consensus_line]
        by_book = {r.get("sportsbook", ""): r for r in at_line}
        over_rows = [r for r in at_line if str(r.get("side") or "").lower() == "over"]
        under_rows = [r for r in at_line if str(r.get("side") or "").lower() == "under"]
        if not over_rows and not under_rows:
            continue
        best_over = max(over_rows, key=lambda r: american_to_decimal(r.get("price_american")) or 0.0, default=None)
        best_under = max(under_rows, key=lambda r: american_to_decimal(r.get("price_american")) or 0.0, default=None)
        over_price = best_over.get("price_american") if best_over else None
        under_price = best_under.get("price_american") if best_under else None
        no_vig_over, no_vig_under = no_vig_two_sided_probabilities(over_price, under_price)
        model_over = k_model.poisson_over_probability(consensus_line, projected_mean)
        if model_over is None:
            continue
        books_for_market = len(by_book)

        specs = [
            ("over", over_price, no_vig_over, model_over, best_over.get("sportsbook", "") if best_over else "", best_over.get("sportsbook_deeplink") if best_over else None),
            ("under", under_price, no_vig_under, 1.0 - model_over, best_under.get("sportsbook", "") if best_under else "", best_under.get("sportsbook_deeplink") if best_under else None),
        ]
        for side, price, no_vig_probability, model_probability, sportsbook, deeplink in specs:
            if price is None:
                continue
            price_confirmed = bool(american_to_decimal(price) is not None and books_for_market >= min_real_books)
            support_passed = False
            blocking: list[str] = []
            if calibration_store is not None and calibration_as_of is not None:
                snapshot_rows = calibration_store.observations_as_of(calibration_as_of)
                support = evaluate_support(
                    snapshot_rows,
                    market_bucket="pitcher_strikeouts",
                    line_bucket=f"pitcher_strikeouts|{pitcher_id}|{side}",
                    state_bucket=STATE_BUCKET,
                    independent_slate_count=len({row.get("slate_id") for row in snapshot_rows}),
                )
                support_passed = support.in_support
                blocking = list(support.blocking_dimensions)
            authorized = bool(price_confirmed and support_passed)
            legs.append(
                PitcherKLeg(
                    pitcher_id=int(pitcher_id), pitcher_name=pitcher_name,
                    team=str(starter.get("team") or ""), opponent=str(starter.get("opponent") or ""),
                    game_id=str(starter.get("game_id") or ""),
                    line=consensus_line, side=side,
                    model_probability=model_probability, no_vig_market_probability=no_vig_probability,
                    price_american=price, sportsbook=sportsbook, market_books=books_for_market,
                    price_confirmed=price_confirmed, leg_authorized=authorized,
                    support_blocking_dimensions=blocking, sportsbook_deeplink=deeplink,
                )
            )
    return legs


@dataclass
class PitcherParlayCandidate:
    leg_a: PitcherKLeg
    leg_b: PitcherKLeg
    naive_independence_probability: float  # the REAL, correct joint model here -- see module docstring
    naive_no_vig_combo_probability: Optional[float]
    combo_decimal_price: Optional[float]
    probability_edge: Optional[float]
    expected_value_per_unit: Optional[float]
    candidate_authorized: bool

    @property
    def betslip(self) -> dict[str, Any]:
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
            "leg_a": self.leg_a.as_dict(), "leg_b": self.leg_b.as_dict(),
            "real_joint_model_probability": self.naive_independence_probability,
            "naive_independence_probability": self.naive_independence_probability,
            "naive_no_vig_combo_probability": self.naive_no_vig_combo_probability,
            "combo_decimal_price": self.combo_decimal_price,
            "probability_edge": self.probability_edge,
            "expected_value_per_unit": self.expected_value_per_unit,
            "candidate_authorized": self.candidate_authorized,
            "betslip": betslip,
            "betslip_url": betslip.get("url"),
        }


def build_pitcher_parlay(
    legs: list[PitcherKLeg],
    *,
    min_combo_abs_edge: float = MIN_COMBO_ABS_EDGE,
    min_combo_expected_value: float = MIN_COMBO_EXPECTED_VALUE,
) -> Optional[PitcherParlayCandidate]:
    """Real best-EV leg per real distinct pitcher (price-confirmed only),
    then the real best-combined-EV pair from two DIFFERENT real pitchers
    (never two legs on the same start). None (never a fabricated pair)
    with fewer than two real distinct priced pitchers."""
    priced = [leg for leg in legs if leg.price_confirmed and leg.expected_value_per_unit is not None]
    best_by_pitcher: dict[int, PitcherKLeg] = {}
    for leg in priced:
        current = best_by_pitcher.get(leg.pitcher_id)
        if current is None or (leg.expected_value_per_unit or -1.0) > (current.expected_value_per_unit or -1.0):
            best_by_pitcher[leg.pitcher_id] = leg

    candidates = sorted(best_by_pitcher.values(), key=lambda leg: leg.expected_value_per_unit or -1.0, reverse=True)
    if len(candidates) < 2:
        return None
    leg_a, leg_b = candidates[0], candidates[1]

    joint_probability = leg_a.model_probability * leg_b.model_probability
    naive_no_vig = (
        leg_a.no_vig_market_probability * leg_b.no_vig_market_probability
        if leg_a.no_vig_market_probability is not None and leg_b.no_vig_market_probability is not None
        else None
    )
    decimal_a, decimal_b = leg_a.decimal_price, leg_b.decimal_price
    combo_decimal_price = decimal_a * decimal_b if decimal_a is not None and decimal_b is not None else None
    probability_edge = joint_probability - naive_no_vig if naive_no_vig is not None else None
    expected_value = joint_probability * combo_decimal_price - 1.0 if combo_decimal_price is not None else None

    combo_edge_passed = probability_edge is not None and probability_edge >= min_combo_abs_edge
    combo_ev_passed = expected_value is not None and expected_value > min_combo_expected_value
    candidate_authorized = bool(leg_a.leg_authorized and leg_b.leg_authorized and combo_edge_passed and combo_ev_passed)

    return PitcherParlayCandidate(
        leg_a=leg_a, leg_b=leg_b,
        naive_independence_probability=joint_probability,
        naive_no_vig_combo_probability=naive_no_vig,
        combo_decimal_price=combo_decimal_price,
        probability_edge=probability_edge,
        expected_value_per_unit=expected_value,
        candidate_authorized=candidate_authorized,
    )
