from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))

import run_mlb_pitcher_parlay_daily as orchestrator  # noqa: E402
import pitcher_strikeout_model as k_model  # noqa: E402


def _schedule_payload(*, preview=True) -> dict:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 999001, "officialDate": "2026-08-26",
                        "status": {"abstractGameState": "Preview" if preview else "Final"},
                        "teams": {
                            "home": {"team": {"abbreviation": "ATL", "name": "Atlanta Braves"}, "probablePitcher": {"id": 1, "fullName": "Real Pitcher One"}},
                            "away": {"team": {"abbreviation": "ATH", "name": "Athletics"}, "probablePitcher": {"id": 2, "fullName": "Real Pitcher Two"}},
                        },
                    },
                    {
                        "gamePk": 999002, "officialDate": "2026-08-26",
                        "status": {"abstractGameState": "Preview" if preview else "Final"},
                        "teams": {
                            "home": {"team": {"abbreviation": "LAD", "name": "Los Angeles Dodgers"}, "probablePitcher": {"id": 3, "fullName": "Real Pitcher Three"}},
                            "away": {"team": {"abbreviation": "SD", "name": "San Diego Padres"}, "probablePitcher": {}},
                        },
                    },
                ]
            }
        ]
    }


class _FakeProvider:
    def __init__(self, odds_rows):
        self._odds_rows = odds_rows

    def collect_player_props(self):
        return {"status": "success", "odds": self._odds_rows}


def _k_row(player_name, line, side, price):
    return {"market_type": "pitcher_strikeouts", "player_name": player_name, "line": line, "side": side, "price_american": price, "sportsbook": "fanduel"}


def test_build_starters_covers_both_sides_of_every_real_preview_game():
    games = orchestrator.extract_scheduled_games(_schedule_payload())
    starters = orchestrator.build_starters(games)
    names = {s["pitcher_name"] for s in starters}
    assert names == {"Real Pitcher One", "Real Pitcher Two", "Real Pitcher Three"}  # 4th side had no real probable pitcher


def test_build_daily_payload_no_real_games_scheduled():
    payload = orchestrator.build_daily_payload(run_date=date(2026, 8, 26), schedule_payload=_schedule_payload(preview=False))
    assert payload["status"] == "no_real_games_scheduled_today"


def test_build_daily_payload_no_real_probable_starters_posted():
    payload_source = _schedule_payload()
    del payload_source["dates"][0]["games"][0]["teams"]["home"]["probablePitcher"]
    del payload_source["dates"][0]["games"][0]["teams"]["away"]["probablePitcher"]
    del payload_source["dates"][0]["games"][1]["teams"]["home"]["probablePitcher"]
    payload = orchestrator.build_daily_payload(run_date=date(2026, 8, 26), schedule_payload=payload_source)
    assert payload["status"] == "no_real_probable_starters_posted_yet"


def test_build_daily_payload_produces_a_real_parlay_when_two_starters_are_priced():
    def fake_fetch(pitcher_id, season, name=""):
        return k_model.PitcherStrikeoutSeasonStats(pitcher_id=pitcher_id, name=name, games_started=15, games_pitched=15, outs=270, strikeouts=90)

    odds = [
        _k_row("Real Pitcher One", 5.5, "over", 120),
        _k_row("Real Pitcher Two", 6.5, "over", 130),
        _k_row("Real Pitcher Three", 4.5, "over", 110),
    ]
    payload = orchestrator.build_daily_payload(
        run_date=date(2026, 8, 26), schedule_payload=_schedule_payload(), fanduel_provider=_FakeProvider(odds),
        calibration_ledger=None, fetch_season_stats=fake_fetch,
    )

    assert payload["status"] == "ok"
    assert payload["real_starters_posted"] == 3
    assert payload["parlay_status"] == "ready"
    assert payload["parlay"]["leg_a"]["pitcher_id"] != payload["parlay"]["leg_b"]["pitcher_id"]


def test_build_daily_payload_no_parlay_when_fewer_than_two_starters_have_real_projections():
    def only_first_pitcher_has_sample(pitcher_id, season, name=""):
        if pitcher_id == 1:
            return k_model.PitcherStrikeoutSeasonStats(pitcher_id=pitcher_id, name=name, games_started=15, games_pitched=15, outs=270, strikeouts=90)
        return None

    odds = [_k_row("Real Pitcher One", 5.5, "over", 120), _k_row("Real Pitcher Two", 6.5, "over", 130)]
    payload = orchestrator.build_daily_payload(
        run_date=date(2026, 8, 26), schedule_payload=_schedule_payload(), fanduel_provider=_FakeProvider(odds),
        calibration_ledger=None, fetch_season_stats=only_first_pitcher_has_sample,
    )

    assert payload["parlay_status"] == "no_real_pair_from_two_distinct_priced_starters"
    assert payload["parlay"] is None


def test_write_web_payload_round_trips_through_disk(tmp_path):
    payload = {"status": "ok", "run_date": "2026-08-26"}
    out_path = orchestrator.write_web_payload(payload, web_data_root=tmp_path)
    assert out_path.name == "pitcher_parlay_predictions.json"
    import json

    assert json.loads(out_path.read_text(encoding="utf-8")) == payload
