from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "parlay_v2"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import pitcher_strikeout_model as k_model  # noqa: E402
import select_mlb_pitcher_parlay as legacy_pitcher  # noqa: E402
from pitcher_alt_line_frontier import (  # noqa: E402
    MIN_COMBO_DECIMAL_PRICE,
    MIN_LEG_DECIMAL_PRICE,
    MIN_LEG_PROBABILITY,
    build_pitcher_k_alt_line_legs,
    build_pitcher_parlay_frontier,
)
from same_game_quality_selector import exploratory_ev_candidates, quality_safe_candidates  # noqa: E402


def test_same_game_headline_rejects_low_joint_high_ev_candidate() -> None:
    low_joint_high_ev = SimpleNamespace(
        real_joint_model_probability=0.247,
        probability_edge=0.081,
        expected_value_per_unit=0.389,
    )
    safe_lower_ev = SimpleNamespace(
        real_joint_model_probability=0.56,
        probability_edge=0.06,
        expected_value_per_unit=0.12,
    )

    headline = quality_safe_candidates([low_joint_high_ev, safe_lower_ev])
    exploratory = exploratory_ev_candidates([low_joint_high_ev, safe_lower_ev])

    assert headline == [safe_lower_ev]
    assert exploratory == [low_joint_high_ev]


def test_same_game_headline_abstains_when_no_combo_clears_joint_floor() -> None:
    candidates = [
        SimpleNamespace(real_joint_model_probability=0.49, probability_edge=0.20, expected_value_per_unit=0.80),
        SimpleNamespace(real_joint_model_probability=0.31, probability_edge=0.10, expected_value_per_unit=0.50),
    ]
    assert quality_safe_candidates(candidates) == []


def _starter(pitcher_id: int = 1, name: str = "Pitcher One", game_id: str = "g1") -> dict:
    return {
        "pitcher_id": pitcher_id,
        "pitcher_name": name,
        "team": f"T{pitcher_id}",
        "opponent": f"O{pitcher_id}",
        "game_id": game_id,
    }


def _odds(player: str, line: float, side: str, price: int) -> dict:
    return {
        "market_type": "pitcher_strikeouts",
        "player_name": player,
        "line": line,
        "side": side,
        "price_american": price,
        "sportsbook": "fanduel",
        "sportsbook_deeplink": f"https://sportsbook.fanduel.com/addToBetslip?marketId={line}&selectionId={abs(price)}",
    }


def _fake_projection(games_started: int = 15, outs: int = 270, strikeouts: int = 90):
    def fetch(pitcher_id, season, name=""):
        return k_model.PitcherStrikeoutSeasonStats(
            pitcher_id=pitcher_id,
            name=name,
            games_started=games_started,
            games_pitched=games_started,
            outs=outs,
            strikeouts=strikeouts,
        )

    return fetch


def test_pitcher_alt_line_builder_preserves_every_real_threshold() -> None:
    legs = build_pitcher_k_alt_line_legs(
        [_starter()],
        [
            _odds("Pitcher One", 4.5, "over", -250),
            _odds("Pitcher One", 5.5, "over", -120),
            _odds("Pitcher One", 6.5, "over", 145),
        ],
        season=2026,
        fetch_season_stats=_fake_projection(),
    )

    assert {(leg.line, leg.side) for leg in legs} == {(4.5, "over"), (5.5, "over"), (6.5, "over")}
    assert all(leg.no_vig_market_probability is None for leg in legs)


def _leg(
    pitcher_id: int,
    *,
    probability: float,
    price: int,
    game_id: str | None = None,
    line: float = 4.5,
) -> legacy_pitcher.PitcherKLeg:
    return legacy_pitcher.PitcherKLeg(
        pitcher_id=pitcher_id,
        pitcher_name=f"Pitcher {pitcher_id}",
        team=f"T{pitcher_id}",
        opponent=f"O{pitcher_id}",
        game_id=game_id or f"g{pitcher_id}",
        line=line,
        side="over",
        model_probability=probability,
        no_vig_market_probability=None,
        price_american=price,
        sportsbook="fanduel",
        market_books=1,
        price_confirmed=True,
        leg_authorized=False,
    )


def test_pitcher_frontier_uses_harder_roi_line_instead_of_minus_2500_undercut() -> None:
    # Same pitcher, two real thresholds: legacy max-hit logic wants the
    # 93.4%-probability -2500 undercut. The ROI frontier must instead use the
    # harder 76.5%-probability line because it clears the probability floor,
    # has positive EV, and contributes a real payout.
    cease_undercut = _leg(1, probability=0.9343621783, price=-2500, line=4.5)
    cease_roi_line = _leg(1, probability=0.7653472459, price=-150, line=6.5)
    brown_undercut = _leg(2, probability=0.9071067708, price=-2200, line=2.5)
    second_roi_pitcher = _leg(3, probability=0.78, price=-150, line=4.5)

    legacy = legacy_pitcher.build_pitcher_parlay(
        [cease_undercut, cease_roi_line, brown_undercut, second_roi_pitcher]
    )
    frontier = build_pitcher_parlay_frontier(
        [cease_undercut, cease_roi_line, brown_undercut, second_roi_pitcher]
    )

    assert legacy is not None and frontier is not None
    assert {legacy.leg_a.pitcher_id, legacy.leg_b.pitcher_id} == {1, 2}
    assert legacy.leg_a.line == pytest.approx(4.5)

    assert {frontier.leg_a.pitcher_id, frontier.leg_b.pitcher_id} == {1, 3}
    cease_selected = frontier.leg_a if frontier.leg_a.pitcher_id == 1 else frontier.leg_b
    assert cease_selected.line == pytest.approx(6.5)
    assert cease_selected.price_american == -150
    assert cease_selected.decimal_price >= MIN_LEG_DECIMAL_PRICE
    assert frontier.combo_decimal_price >= MIN_COMBO_DECIMAL_PRICE
    assert frontier.naive_independence_probability >= 0.50
    assert frontier.expected_value_per_unit is not None and frontier.expected_value_per_unit >= 0.05


def test_pitcher_frontier_abstains_instead_of_falling_back_to_ultra_short_negative_ev_pair() -> None:
    cease = _leg(1, probability=0.9343621783, price=-2500)
    brown = _leg(2, probability=0.9071067708, price=-2200)

    legacy = legacy_pitcher.build_pitcher_parlay([cease, brown])
    frontier = build_pitcher_parlay_frontier([cease, brown])

    assert legacy is not None
    assert legacy.expected_value_per_unit is not None and legacy.expected_value_per_unit < 0.0
    assert frontier is None


def test_pitcher_frontier_never_trades_below_probability_floor_for_roi() -> None:
    safe_a = _leg(1, probability=0.82, price=-180)
    safe_b = _leg(2, probability=0.80, price=-170)
    tempting_longshot = _leg(3, probability=0.55, price=300)

    frontier = build_pitcher_parlay_frontier([safe_a, safe_b, tempting_longshot])

    assert frontier is not None
    assert {frontier.leg_a.pitcher_id, frontier.leg_b.pitcher_id} == {1, 2}
    assert frontier.leg_a.model_probability >= MIN_LEG_PROBABILITY
    assert frontier.leg_b.model_probability >= MIN_LEG_PROBABILITY
    assert frontier.naive_independence_probability >= 0.50


def test_pitcher_frontier_requires_meaningful_combined_payout() -> None:
    # Individually acceptable -500 favorites still do not make a useful ROI
    # parlay together: 1.20 * 1.20 = 1.44 decimal, below the +100 floor.
    short_a = _leg(1, probability=0.85, price=-500)
    short_b = _leg(2, probability=0.85, price=-500)

    assert short_a.expected_value_per_unit > 0.0
    assert short_b.expected_value_per_unit > 0.0
    assert build_pitcher_parlay_frontier([short_a, short_b]) is None


def test_pitcher_frontier_requires_different_games_for_independence_claim() -> None:
    same_game_a = _leg(1, probability=0.90, price=-200, game_id="same")
    same_game_b = _leg(2, probability=0.90, price=-200, game_id="same")
    other_game = _leg(3, probability=0.75, price=100, game_id="other")

    frontier = build_pitcher_parlay_frontier([same_game_a, same_game_b, other_game])

    assert frontier is not None
    assert frontier.leg_a.game_id != frontier.leg_b.game_id
