from __future__ import annotations

import sys
from pathlib import Path

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
