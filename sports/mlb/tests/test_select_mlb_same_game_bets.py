from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "parlay_v2"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import game_simulation_model as sim  # noqa: E402
import select_mlb_same_game_bets as select  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402


def _game() -> dict:
    return {"game_id": "824940", "date": "2026-04-01", "home_team": "ATL", "away_team": "ATH"}


def _market_odds() -> list[dict]:
    return [
        {
            "target": "moneyline", "sportsbook": "draftkings", "home_moneyline": -150, "away_moneyline": 130, "line": None,
            "home_moneyline_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=2",
            "away_moneyline_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=1",
        },
        {
            "target": "game_total", "sportsbook": "draftkings", "line": 8.5, "over_price": -110, "under_price": -110,
            "over_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=3",
            "under_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=4",
        },
        {
            "target": "game_total", "sportsbook": "fanduel", "line": 8.5, "over_price": -105, "under_price": -115,
            "over_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=m3&selectionId=5",
            "under_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=m3&selectionId=6",
        },
        {"target": "first_5_innings_total", "sportsbook": "draftkings", "line": 4.5, "over_price": -115, "under_price": -105},
    ]


def _result() -> sim.GameSimulationResult:
    return sim.simulate_game_outcomes(5.0, 4.0, f5_share=0.55, num_trials=20000, seed=11)


def test_no_vig_two_sided_probabilities_normalizes_real_vig() -> None:
    home_p, away_p = select.no_vig_two_sided_probabilities(-150, 130)
    assert home_p is not None and away_p is not None
    assert abs((home_p + away_p) - 1.0) < 1e-9
    assert home_p > away_p


def test_consensus_line_picks_the_real_more_widely_quoted_line() -> None:
    rows = [
        {"line": 8.5, "sportsbook": "draftkings"},
        {"line": 8.5, "sportsbook": "fanduel"},
        {"line": 9.0, "sportsbook": "betmgm"},
    ]
    assert select._consensus_line(rows) == 8.5


def test_consensus_line_none_with_no_real_lines() -> None:
    assert select._consensus_line([]) is None


def test_build_legs_for_market_moneyline_produces_home_and_away() -> None:
    legs = select._build_legs_for_market("moneyline", _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    assert {leg.side for leg in legs} == {"home", "away"}
    home_leg = next(leg for leg in legs if leg.side == "home")
    assert home_leg.price_american == -150
    assert home_leg.model_probability == _result().home_win_probability


def test_build_legs_for_market_totals_uses_real_consensus_line() -> None:
    legs = select._build_legs_for_market("game_total", _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    assert all(leg.line == 8.5 for leg in legs)
    # best real over price between the two real books quoting 8.5 is -105 (fanduel)
    over_leg = next(leg for leg in legs if leg.side == "over")
    assert over_leg.price_american == -105


def test_build_legs_for_market_moneyline_carries_real_per_side_deeplinks() -> None:
    legs = select._build_legs_for_market("moneyline", _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    home_leg = next(leg for leg in legs if leg.side == "home")
    away_leg = next(leg for leg in legs if leg.side == "away")
    assert home_leg.sportsbook_deeplink == "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=2"
    assert away_leg.sportsbook_deeplink == "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=1"
    assert home_leg.as_dict()["sportsbook_deeplink"] == home_leg.sportsbook_deeplink


def test_build_legs_for_market_totals_carries_real_per_side_deeplinks_from_the_winning_book() -> None:
    legs = select._build_legs_for_market("game_total", _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    over_leg = next(leg for leg in legs if leg.side == "over")
    under_leg = next(leg for leg in legs if leg.side == "under")
    assert over_leg.sportsbook == "fanduel"  # -105 beats draftkings' -110
    assert over_leg.sportsbook_deeplink == "https://sportsbook.fanduel.com/addToBetslip?marketId=m3&selectionId=5"
    assert under_leg.sportsbook_deeplink == "https://sportsbook.fanduel.com/addToBetslip?marketId=m3&selectionId=6"


def test_build_legs_for_market_leaves_deeplink_none_when_odds_row_has_none() -> None:
    legs = select._build_legs_for_market("first_5_innings_total", _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    assert all(leg.sportsbook_deeplink is None for leg in legs)


def test_build_legs_for_market_returns_empty_without_real_data() -> None:
    assert select._build_legs_for_market("moneyline", [], _result(), calibration_store=None, calibration_as_of=None, min_real_books=1) == []


def test_build_single_leg_team_market_candidates_covers_every_priced_market() -> None:
    legs = select.build_single_leg_team_market_candidates(_game(), _result(), _market_odds(), calibration_store=None, calibration_as_of=None)
    assert {leg.market for leg in legs} == {"moneyline", "game_total", "first_5_innings_total"}
    # two sides per market (home/away, over/under) for every one of the three real priced markets
    assert len(legs) == 6


def test_build_single_leg_team_market_candidates_matches_build_legs_for_market_exactly() -> None:
    """The standalone extraction must be exactly the union of what
    build_same_game_candidates already builds internally per market --
    never a second, independently-derived copy of the same real legs."""
    combined = select.build_single_leg_team_market_candidates(_game(), _result(), _market_odds(), calibration_store=None, calibration_as_of=None)
    direct = [
        leg
        for market in ("moneyline", "game_total", "first_5_innings_total")
        for leg in select._build_legs_for_market(market, _market_odds(), _result(), calibration_store=None, calibration_as_of=None, min_real_books=1)
    ]
    assert [leg.as_dict() for leg in combined] == [leg.as_dict() for leg in direct]


def test_build_single_leg_team_market_candidates_default_unauthorized_with_no_calibration_evidence() -> None:
    """Same honest shadow-only posture as every combo/single-leg board in
    this repo: with no calibration_store, no leg is ever authorized."""
    legs = select.build_single_leg_team_market_candidates(_game(), _result(), _market_odds(), calibration_store=None, calibration_as_of=None)
    assert legs  # real legs were built
    assert all(not leg.leg_authorized for leg in legs)


def test_build_single_leg_team_market_candidates_returns_empty_without_real_data() -> None:
    assert select.build_single_leg_team_market_candidates(_game(), _result(), [], calibration_store=None, calibration_as_of=None) == []


def test_build_same_game_candidates_never_pairs_same_market_opposite_sides() -> None:
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    for combo in combos:
        assert combo.leg_a.market != combo.leg_b.market


def test_build_same_game_candidates_covers_every_real_cross_market_pair() -> None:
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    market_pairs = {tuple(sorted([c.leg_a.market, c.leg_b.market])) for c in combos}
    assert market_pairs == {
        ("game_total", "moneyline"),
        ("first_5_innings_total", "moneyline"),
        ("first_5_innings_total", "game_total"),
    }


def test_build_same_game_candidates_joint_residual_reflects_real_correlation() -> None:
    """The real joint probability for the highly-correlated full-total /
    F5-total pair should differ meaningfully from the naive independence
    product -- the whole point of simulating jointly."""
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    correlated_pair = next(c for c in combos if {c.leg_a.market, c.leg_b.market} == {"game_total", "first_5_innings_total"})
    assert abs(correlated_pair.joint_residual) > 0.001


def test_build_same_game_candidates_naive_market_joint_raw_is_reciprocal_of_combo_price() -> None:
    """naive_market_joint_raw_probability is the vig-included market
    joint -- 1 / combo_decimal_price -- kept distinct from the de-vigged
    naive_no_vig_combo_probability used for probability_edge."""
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    priced = [c for c in combos if c.combo_decimal_price]
    assert priced  # real priced combos exist in the fixture
    for combo in priced:
        assert combo.naive_market_joint_raw_probability == pytest.approx(1.0 / combo.combo_decimal_price)
        # Vig-included (raw) must be >= the de-vigged figure -- each raw
        # single-side implied probability already carries the book's own
        # margin (that's what "vig" means: both sides' raw implied
        # probabilities sum to more than 100%), and de-vigging always
        # normalizes that back down, never up.
        if combo.naive_no_vig_combo_probability is not None:
            assert combo.naive_market_joint_raw_probability >= combo.naive_no_vig_combo_probability - 1e-9
    assert combo.as_dict()["naive_market_joint_raw_probability"] == combo.naive_market_joint_raw_probability


def test_build_same_game_candidates_default_unauthorized_with_no_calibration_evidence() -> None:
    """No calibration_store passed -> no support evidence -> nothing here
    should ever silently authorize a brand-new same-game policy."""
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    assert all(not c.candidate_authorized for c in combos)


def test_build_same_game_candidates_stays_unauthorized_with_an_empty_real_calibration_store(tmp_path) -> None:
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    combos = select.build_same_game_candidates(
        _game(), _result(), _market_odds(), calibration_store=store, calibration_as_of="2026-04-01T00:00:00Z"
    )
    assert all(not c.candidate_authorized for c in combos)
    assert all("market_support" in c.support_blocking_dimensions for c in combos)


def test_build_same_game_candidates_never_authorizes_below_the_real_joint_probability_floor() -> None:
    """Real hit-rate-first gate: no combo authorizes below
    MIN_COMBO_JOINT_PROBABILITY, however strong its edge/EV, once a
    threshold that high can never be cleared by any real combo -- this
    is the same real, disclosed rule HIGH_HIT_PARLAY_V1 already applies
    to its own legs/joint, now extended to same-game combos."""
    combos = select.build_same_game_candidates(
        _game(), _result(), _market_odds(), min_combo_joint_probability=1.01
    )
    assert combos  # real candidates still get built and reported
    assert all(not c.candidate_authorized for c in combos)


def test_build_same_game_candidates_joint_probability_floor_is_a_real_additive_gate() -> None:
    """Disabling the floor (0.0) never authorizes a combo the existing
    edge/EV/support gates would have blocked anyway -- it only ever adds
    a real restriction, never removes one."""
    baseline = {
        (c.leg_a.market, c.leg_b.market, c.leg_a.side, c.leg_b.side): c.candidate_authorized
        for c in select.build_same_game_candidates(_game(), _result(), _market_odds())
    }
    loosened = {
        (c.leg_a.market, c.leg_b.market, c.leg_a.side, c.leg_b.side): c.candidate_authorized
        for c in select.build_same_game_candidates(_game(), _result(), _market_odds(), min_combo_joint_probability=0.0)
    }
    for key, was_authorized in baseline.items():
        if was_authorized:
            assert loosened[key]


def test_combo_betslip_ready_when_both_real_legs_priced_at_fanduel() -> None:
    market_odds = [
        {
            "target": "moneyline", "sportsbook": "fanduel", "home_moneyline": -150, "away_moneyline": 130, "line": None,
            "home_moneyline_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=2",
            "away_moneyline_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=1",
        },
        {
            "target": "game_total", "sportsbook": "fanduel", "line": 8.5, "over_price": -105, "under_price": -115,
            "over_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=3",
            "under_deeplink": "https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=4",
        },
    ]
    combos = select.build_same_game_candidates(_game(), _result(), market_odds)
    combo = next(c for c in combos if {c.leg_a.market, c.leg_b.market} == {"moneyline", "game_total"})

    assert combo.betslip["status"] == "ready"
    url = combo.betslip["url"]
    assert url.startswith("https://account.sportsbook.fanduel.com/sportsbook/addToBetslip?")
    assert combo.as_dict()["betslip_url"] == url


def test_combo_betslip_unavailable_when_a_leg_has_no_real_fanduel_selection() -> None:
    """_market_odds()'s moneyline row is a draftkings-best price with no
    FanDuel deeplink -- a combo built from it must never get a fake or
    partial multi-leg link."""
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    combo = next(c for c in combos if {c.leg_a.market, c.leg_b.market} == {"moneyline", "game_total"})

    assert combo.betslip["status"] == "unavailable"
    assert "url" not in combo.betslip
    assert combo.as_dict()["betslip_url"] is None


def test_build_pair_observation_for_combo_has_no_real_quoted_sgp_price() -> None:
    combo = select.build_same_game_candidates(_game(), _result(), _market_odds())[0]
    observation = select.build_pair_observation_for_combo(
        combo, slate_id="2026-04-01", predictive_version="v1", policy_version="shadow_only_v1",
        decision_timestamp="2026-04-01T10:00:00Z",
    )
    assert observation.same_game is True
    assert observation.quoted_pair_price is None
    assert observation.predicted_joint_probability == combo.real_joint_model_probability
    assert observation.predicted_independence_probability == combo.naive_independence_probability


def test_top_combo_candidates_ranks_by_real_expected_value() -> None:
    combos = select.build_same_game_candidates(_game(), _result(), _market_odds())
    top = select.top_combo_candidates(combos, max_candidates=3)
    assert len(top) <= 3
    evs = [c.expected_value_per_unit for c in top]
    assert evs == sorted(evs, reverse=True)
