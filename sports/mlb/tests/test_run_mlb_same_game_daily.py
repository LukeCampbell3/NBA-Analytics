from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"))

import run_mlb_same_game_daily as orchestrator  # noqa: E402
from the_odds_api_mlb_team_market_provider import TheOddsApiMlbTeamMarketProvider  # noqa: E402


def _schedule_payload(*, home="Atlanta Braves", away="Athletics", home_abbr="ATL", away_abbr="ATH", preview=True) -> dict:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 999001,
                        "officialDate": "2026-06-20",
                        "status": {"abstractGameState": "Preview" if preview else "Final"},
                        "teams": {
                            "home": {"team": {"abbreviation": home_abbr, "name": home}, "probablePitcher": {"id": 519242, "fullName": "Chris Sale"}},
                            "away": {"team": {"abbreviation": away_abbr, "name": away}, "probablePitcher": {"id": 622663, "fullName": "Luis Severino"}},
                        },
                    }
                ]
            }
        ]
    }


def test_extract_scheduled_games_keeps_only_real_preview_games() -> None:
    games = orchestrator.extract_scheduled_games(_schedule_payload(preview=False))
    assert games == []
    games = orchestrator.extract_scheduled_games(_schedule_payload(preview=True))
    assert len(games) == 1
    assert games[0]["home_team"] == "ATL"
    assert games[0]["away_team"] == "ATH"
    assert games[0]["home_starter_id"] == 519242


def test_extract_scheduled_games_normalizes_real_abbreviation_mismatches() -> None:
    games = orchestrator.extract_scheduled_games(_schedule_payload(home="Arizona Diamondbacks", home_abbr="AZ"))
    assert games[0]["home_team"] == "ARI"


def test_extract_scheduled_games_handles_no_real_probable_pitcher_posted_yet() -> None:
    payload = _schedule_payload()
    del payload["dates"][0]["games"][0]["teams"]["home"]["probablePitcher"]
    games = orchestrator.extract_scheduled_games(payload)
    assert games[0]["home_starter_id"] is None
    assert games[0]["home_starter_name"] == ""


def _fake_odds_provider() -> TheOddsApiMlbTeamMarketProvider:
    fixture = {
        "id": "evt1", "commence_time": "2026-06-20T22:00:00Z",
        "home_team": "Atlanta Braves", "away_team": "Athletics",
        "bookmakers": [
            {
                "key": "draftkings", "title": "DraftKings", "last_update": "2026-06-20T20:00:00Z",
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Atlanta Braves", "price": -150}, {"name": "Athletics", "price": 130}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 8.5, "price": -110}, {"name": "Under", "point": 8.5, "price": -110}]},
                    {"key": "totals_1st_5_innings", "outcomes": [{"name": "Over", "point": 4.5, "price": -115}, {"name": "Under", "point": 4.5, "price": -105}]},
                ],
            }
        ],
    }
    return TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[fixture])


def _write_historical_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    team_rows = []
    pitcher_rows = []
    teams = ["ATL", "ATH"]
    for i in range(40):
        home = teams[i % 2]
        away = teams[(i + 1) % 2]
        team_rows.append(
            {
                "date": f"2026-04-{(i % 28) + 1:02d}", "game_id": f"g{i}", "home_team": home, "away_team": away,
                "home_score": "5", "away_score": "3", "home_innings_1_5": "3", "away_innings_1_5": "2",
                "market_home_moneyline": "-150", "market_away_moneyline": "130", "market_run_total": "7.5",
            }
        )
        pitcher_rows.append(
            {
                "date": f"2026-04-{(i % 28) + 1:02d}", "home_team": home, "away_team": away,
                "home_starter_id": "519242", "home_starter_name": "Chris Sale", "home_starter_outs": "18", "home_starter_earned_runs": "2",
                "home_bullpen_outs": "9", "home_bullpen_earned_runs": "1",
                "away_starter_id": "622663", "away_starter_name": "Luis Severino", "away_starter_outs": "15", "away_starter_earned_runs": "3",
                "away_bullpen_outs": "12", "away_bullpen_earned_runs": "2",
            }
        )
    team_csv = tmp_path / "teams.csv"
    pitcher_csv = tmp_path / "pitchers.csv"
    with open(team_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(team_rows[0].keys()))
        writer.writeheader()
        writer.writerows(team_rows)
    with open(pitcher_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pitcher_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pitcher_rows)
    return team_csv, pitcher_csv


def test_build_daily_payload_reports_no_real_games_scheduled(tmp_path) -> None:
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    payload = orchestrator.build_daily_payload(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=500,
        schedule_payload={"dates": []}, odds_provider=_fake_odds_provider(),
    )
    assert payload["status"] == "no_real_games_scheduled_today"


def test_build_daily_payload_produces_a_real_game_entry_with_combo_candidates(tmp_path) -> None:
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    payload = orchestrator.build_daily_payload(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=2000,
        schedule_payload=_schedule_payload(), odds_provider=_fake_odds_provider(),
    )
    assert payload["status"] == "ok"
    assert len(payload["games"]) == 1
    game_entry = payload["games"][0]
    assert game_entry["status"] == "ok"
    assert game_entry["home_team"] == "ATL"
    assert 0.0 < game_entry["home_win_probability"] < 1.0
    assert len(game_entry["combo_candidates"]) > 0
    # brand-new policy, empty calibration ledger by construction (None passed) -> nothing authorized yet
    assert game_entry["candidate_authorized_count"] == 0


def test_build_daily_payload_reports_no_real_odds_priced_yet(tmp_path) -> None:
    team_csv, pitcher_csv = _write_historical_fixtures(tmp_path)
    empty_odds_provider = TheOddsApiMlbTeamMarketProvider(api_key="fixture", fixture_payloads=[])
    payload = orchestrator.build_daily_payload(
        run_date=date(2026, 6, 20), team_universe_csv=team_csv, pitcher_universe_csv=pitcher_csv,
        calibration_ledger=None, num_trials=500,
        schedule_payload=_schedule_payload(), odds_provider=empty_odds_provider,
    )
    assert payload["games"][0]["status"] == "no_real_odds_priced_yet"


def test_write_web_payload_writes_real_json(tmp_path) -> None:
    payload = {"status": "ok", "games": []}
    out_path = orchestrator.write_web_payload(payload, web_data_root=tmp_path)
    assert out_path.exists()
    assert out_path.name == "same_game_predictions.json"
