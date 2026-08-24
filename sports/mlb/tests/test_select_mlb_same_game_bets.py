from __future__ import annotations

import sys
from pathlib import Path

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
        {"target": "moneyline", "sportsbook": "draftkings", "home_moneyline": -150, "away_moneyline": 130, "line": None},
        {"target": "game_total", "sportsbook": "draftkings", "line": 8.5, "over_price": -110, "under_price": -110},
        {"target": "game_total", "sportsbook": "fanduel", "line": 8.5, "over_price": -105, "under_price": -115},
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


def test_build_legs_for_market_returns_empty_without_real_data() -> None:
    assert select._build_legs_for_market("moneyline", [], _result(), calibration_store=None, calibration_as_of=None, min_real_books=1) == []


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
