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


def test_pitcher_frontier_prefers_positive_ev_pair_over_max_probability_negative_ev_pair() -> None:
    # Mirrors the structural shape of the 2026-08-28 board: Cease and
    # Brown had the two highest hit probabilities but extremely short
    # prices; Hancock was only slightly lower probability at a much more
    # efficient price.  This test uses only pre-event model/price fields.
    cease = _leg(1, probability=0.9343621783, price=-2500)
    brown = _leg(2, probability=0.9071067708, price=-2200)
    hancock = _leg(3, probability=0.8918332911, price=-400)

    legacy = legacy_pitcher.build_pitcher_parlay([cease, brown, hancock])
    frontier = build_pitcher_parlay_frontier([cease, brown, hancock])

    assert legacy is not None and frontier is not None
    assert {legacy.leg_a.pitcher_id, legacy.leg_b.pitcher_id} == {1, 2}
    assert legacy.expected_value_per_unit == pytest.approx(-0.078, abs=0.002)

    assert {frontier.leg_a.pitcher_id, frontier.leg_b.pitcher_id} == {1, 3}
    assert frontier.naive_independence_probability > 0.80
    assert frontier.expected_value_per_unit is not None
    assert frontier.expected_value_per_unit > 0.08


def test_pitcher_frontier_never_trades_below_probability_floors_for_roi() -> None:
    safe_a = _leg(1, probability=0.82, price=-180)
    safe_b = _leg(2, probability=0.80, price=-170)
    tempting_longshot = _leg(3, probability=0.55, price=300)

    frontier = build_pitcher_parlay_frontier([safe_a, safe_b, tempting_longshot])

    assert frontier is not None
    assert {frontier.leg_a.pitcher_id, frontier.leg_b.pitcher_id} == {1, 2}
    assert frontier.leg_a.model_probability >= 0.70
    assert frontier.leg_b.model_probability >= 0.70
    assert frontier.naive_independence_probability >= 0.50


def test_pitcher_frontier_requires_different_games_for_independence_claim() -> None:
    same_game_a = _leg(1, probability=0.90, price=-200, game_id="same")
    same_game_b = _leg(2, probability=0.90, price=-200, game_id="same")
    other_game = _leg(3, probability=0.75, price=100, game_id="other")

    frontier = build_pitcher_parlay_frontier([same_game_a, same_game_b, other_game])

    assert frontier is not None
    assert frontier.leg_a.game_id != frontier.leg_b.game_id
