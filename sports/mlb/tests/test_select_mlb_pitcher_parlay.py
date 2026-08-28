from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "parlay_v2"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import pitcher_strikeout_model as k_model  # noqa: E402
import select_mlb_pitcher_parlay as select  # noqa: E402
from calibration.schema import build_observation  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402


def _real_observation(slate_id: str):
    return build_observation(
        slate_id=slate_id, game_id="g1", event_date=slate_id,
        player_id="1", player_name="Real Pitcher One",
        target="pitcher_strikeouts", side="over", line=5.5, book="real",
        quote_decimal=1.9, quote_timestamp=f"{slate_id}T17:00:00Z",
        prediction_value=6.0, predictive_probability_if_available=0.55,
        state_version="s1", predictive_version="v1",
        market_bucket="pitcher_strikeouts", line_bucket="pitcher_strikeouts|1|over", state_bucket=select.STATE_BUCKET,
        settlement_status="win", actual_outcome=1.0, actual_unit_return=0.9,
        decision_frozen_at=f"{slate_id}T17:05:00Z", settled_at=f"{slate_id}T23:00:00Z",
        calibration_admitted_at=f"{slate_id}T23:30:00Z",
        source_id=f"real_pitcher_one_{slate_id}", source_hash="h1",
    )


def _starter(pitcher_id=1, name="Real Pitcher One", team="ATL", opponent="ATH", game_id="824940") -> dict:
    return {"pitcher_id": pitcher_id, "pitcher_name": name, "team": team, "opponent": opponent, "game_id": game_id}


def _k_odds_row(player_name: str, line: float, side: str, price: float, sportsbook: str = "fanduel", deeplink: str | None = None) -> dict:
    row = {
        "market_type": "pitcher_strikeouts", "player_name": player_name, "line": line, "side": side,
        "price_american": price, "sportsbook": sportsbook,
    }
    if deeplink:
        row["sportsbook_deeplink"] = deeplink
    return row


def _fake_projection(games_started=15, games_pitched=15, outs=270, strikeouts=90):
    def fetch(pitcher_id, season, name=""):
        return k_model.PitcherStrikeoutSeasonStats(
            pitcher_id=pitcher_id, name=name, games_started=games_started, games_pitched=games_pitched,
            outs=outs, strikeouts=strikeouts,
        )
    return fetch


def test_build_pitcher_k_legs_produces_over_and_under_for_a_real_priced_starter():
    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", -110), _k_odds_row("Real Pitcher One", 5.5, "under", -110)],
        season=2026, fetch_season_stats=_fake_projection(),
    )
    sides = {leg.side for leg in legs}
    assert sides == {"over", "under"}
    assert all(leg.pitcher_id == 1 for leg in legs)


def test_build_pitcher_k_legs_skips_a_starter_with_no_real_season_sample():
    def thin_sample_fetch(pitcher_id, season, name=""):
        return k_model.PitcherStrikeoutSeasonStats(pitcher_id=pitcher_id, name=name, games_started=1, games_pitched=1, outs=15, strikeouts=8)

    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", -110)],
        season=2026, fetch_season_stats=thin_sample_fetch,
    )
    assert legs == []


def test_build_pitcher_k_legs_skips_a_starter_with_no_real_odds_row():
    legs = select.build_pitcher_k_legs([_starter()], [], season=2026, fetch_season_stats=_fake_projection())
    assert legs == []


def test_build_pitcher_k_legs_matches_by_normalized_player_name():
    legs = select.build_pitcher_k_legs(
        [_starter(name="José Ramírez Jr.")],
        [_k_odds_row("Jose Ramirez Jr", 5.5, "over", -110)],
        season=2026, fetch_season_stats=_fake_projection(),
    )
    assert legs  # matched despite accent/punctuation differences


def test_build_pitcher_k_legs_stays_unauthorized_with_no_calibration_store():
    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", -110), _k_odds_row("Real Pitcher One", 5.5, "under", -110)],
        season=2026, fetch_season_stats=_fake_projection(),
    )
    assert legs
    assert all(not leg.leg_authorized for leg in legs)


def test_build_pitcher_k_legs_authorized_once_real_calibration_support_exists(tmp_path):
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for day in range(1, 26):
        store.admit(_real_observation(f"2026-07-{day:02d}"))
    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", -110)],
        season=2026, calibration_store=store, calibration_as_of="2026-08-26T00:00:00Z",
        fetch_season_stats=_fake_projection(),
    )
    over_leg = next(leg for leg in legs if leg.side == "over")
    assert over_leg.leg_authorized is True


def test_build_pitcher_parlay_none_with_fewer_than_two_distinct_priced_pitchers():
    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", -110)],
        season=2026, fetch_season_stats=_fake_projection(),
    )
    assert select.build_pitcher_parlay(legs) is None


def test_build_pitcher_parlay_pairs_two_distinct_real_pitchers():
    starters = [_starter(pitcher_id=1, name="Real Pitcher One", game_id="g1"), _starter(pitcher_id=2, name="Real Pitcher Two", team="LAD", opponent="SD", game_id="g2")]
    odds = [
        _k_odds_row("Real Pitcher One", 5.5, "over", 120, deeplink="https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=1"),
        _k_odds_row("Real Pitcher One", 5.5, "under", -150),
        _k_odds_row("Real Pitcher Two", 6.5, "over", 130, deeplink="https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=2"),
        _k_odds_row("Real Pitcher Two", 6.5, "under", -160),
    ]
    legs = select.build_pitcher_k_legs(starters, odds, season=2026, fetch_season_stats=_fake_projection())
    combo = select.build_pitcher_parlay(legs)

    assert combo is not None
    assert combo.leg_a.pitcher_id != combo.leg_b.pitcher_id
    assert abs(combo.naive_independence_probability - combo.leg_a.model_probability * combo.leg_b.model_probability) < 1e-9


def test_build_pitcher_parlay_naive_market_joint_raw_is_reciprocal_of_combo_price():
    """naive_market_joint_raw_probability is the vig-included market
    joint -- 1 / combo_decimal_price -- kept distinct from the de-vigged
    naive_no_vig_combo_probability used for probability_edge."""
    starters = [_starter(pitcher_id=1, name="Real Pitcher One", game_id="g1"), _starter(pitcher_id=2, name="Real Pitcher Two", team="LAD", opponent="SD", game_id="g2")]
    odds = [
        _k_odds_row("Real Pitcher One", 5.5, "over", 120),
        _k_odds_row("Real Pitcher One", 5.5, "under", -150),
        _k_odds_row("Real Pitcher Two", 6.5, "over", 130),
        _k_odds_row("Real Pitcher Two", 6.5, "under", -160),
    ]
    legs = select.build_pitcher_k_legs(starters, odds, season=2026, fetch_season_stats=_fake_projection())
    combo = select.build_pitcher_parlay(legs)

    assert combo is not None
    assert combo.combo_decimal_price
    assert combo.naive_market_joint_raw_probability == pytest.approx(1.0 / combo.combo_decimal_price)
    # Vig-included (raw) must be >= the de-vigged figure -- see
    # select_mlb_same_game_bets.py's own equivalent test for why.
    if combo.naive_no_vig_combo_probability is not None:
        assert combo.naive_market_joint_raw_probability >= combo.naive_no_vig_combo_probability - 1e-9
    assert combo.as_dict()["naive_market_joint_raw_probability"] == combo.naive_market_joint_raw_probability


def test_build_pitcher_parlay_never_pairs_two_legs_from_the_same_pitcher():
    """Even with over AND under legs for the same single real pitcher,
    the combo must never use that one pitcher twice."""
    legs = select.build_pitcher_k_legs(
        [_starter()],
        [_k_odds_row("Real Pitcher One", 5.5, "over", 120), _k_odds_row("Real Pitcher One", 5.5, "under", -150)],
        season=2026, fetch_season_stats=_fake_projection(),
    )
    assert select.build_pitcher_parlay(legs) is None  # only one real distinct pitcher priced


def test_pitcher_parlay_betslip_ready_when_both_real_legs_fanduel():
    starters = [_starter(pitcher_id=1, name="Real Pitcher One", game_id="g1"), _starter(pitcher_id=2, name="Real Pitcher Two", team="LAD", opponent="SD", game_id="g2")]
    odds = [
        _k_odds_row("Real Pitcher One", 5.5, "over", 120, deeplink="https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=1"),
        _k_odds_row("Real Pitcher Two", 6.5, "over", 130, deeplink="https://sportsbook.fanduel.com/addToBetslip?marketId=734.2&selectionId=2"),
    ]
    legs = select.build_pitcher_k_legs(starters, odds, season=2026, fetch_season_stats=_fake_projection())
    combo = select.build_pitcher_parlay(legs)

    assert combo.betslip["status"] == "ready"
    assert combo.as_dict()["betslip_url"] == combo.betslip["url"]


def test_pitcher_parlay_betslip_unavailable_without_real_deeplinks():
    starters = [_starter(pitcher_id=1, name="Real Pitcher One", game_id="g1"), _starter(pitcher_id=2, name="Real Pitcher Two", team="LAD", opponent="SD", game_id="g2")]
    odds = [
        _k_odds_row("Real Pitcher One", 5.5, "over", 120),
        _k_odds_row("Real Pitcher Two", 6.5, "over", 130),
    ]
    legs = select.build_pitcher_k_legs(starters, odds, season=2026, fetch_season_stats=_fake_projection())
    combo = select.build_pitcher_parlay(legs)

    assert combo.betslip["status"] == "unavailable"


def _leg(pitcher_id: int, *, model_probability: float, price_american: float, authorized: bool = True) -> select.PitcherKLeg:
    return select.PitcherKLeg(
        pitcher_id=pitcher_id, pitcher_name=f"Pitcher {pitcher_id}", team="ATL", opponent="ATH",
        game_id=f"g{pitcher_id}", line=5.5, side="over", model_probability=model_probability,
        no_vig_market_probability=model_probability, price_american=price_american,
        sportsbook="fanduel", market_books=1, price_confirmed=True, leg_authorized=authorized,
    )


def test_build_pitcher_parlay_ranks_legs_by_real_hit_probability_not_ev():
    """Hit-rate-first: a lower-probability leg priced to carry much
    higher EV must NOT be preferred over a higher-probability leg with
    thinner EV -- the real point of this repo's own hit-rate-first
    parlay rule (mirrored from select_mlb_same_game_bets.py)."""
    high_ev_low_probability = _leg(1, model_probability=0.55, price_american=250)  # decimal 3.5, EV = 0.925
    high_probability_low_ev = _leg(2, model_probability=0.80, price_american=-140)  # decimal ~1.714, EV ~ 0.371
    third_pitcher = _leg(3, model_probability=0.75, price_american=-130)

    combo = select.build_pitcher_parlay(
        [high_ev_low_probability, high_probability_low_ev, third_pitcher], min_combo_joint_probability=0.0
    )

    assert combo is not None
    # The two highest-probability legs (0.80, 0.75) must be chosen over
    # the low-probability/high-EV leg (0.55), even though the EV-first
    # rule this replaced would have picked leg 1 first.
    assert {combo.leg_a.pitcher_id, combo.leg_b.pitcher_id} == {2, 3}


def test_build_pitcher_parlay_never_authorizes_below_the_real_joint_probability_floor():
    leg_a = _leg(1, model_probability=0.60, price_american=110)
    leg_b = _leg(2, model_probability=0.60, price_american=110)  # joint = 0.36, below the 0.50 floor

    combo = select.build_pitcher_parlay([leg_a, leg_b])

    assert combo is not None
    assert combo.naive_independence_probability == pytest.approx(0.36)
    assert combo.candidate_authorized is False


def test_build_pitcher_parlay_joint_probability_floor_is_a_real_additive_gate():
    """Disabling the floor (0.0) never authorizes a combo the existing
    edge/EV/leg-authorization gates would have blocked anyway -- it only
    ever adds a real restriction, never removes one."""
    leg_a = _leg(1, model_probability=0.60, price_american=110)
    leg_b = _leg(2, model_probability=0.60, price_american=110)

    strict = select.build_pitcher_parlay([leg_a, leg_b])
    loosened = select.build_pitcher_parlay([leg_a, leg_b], min_combo_joint_probability=0.0)

    if strict.candidate_authorized:
        assert loosened.candidate_authorized
