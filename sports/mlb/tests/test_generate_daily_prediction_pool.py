from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import generate_daily_prediction_pool as generator


def test_team_context_keys_use_canonical_player_ids() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2026-08-04"],
            "_game_date": [pd.Timestamp("2026-08-04")],
            "Game_Index": [1],
            "Player": ["Hunter_Brown"],
            "Player_Type": ["pitcher"],
            "Team": ["HOU"],
            "Was_Starter": [1],
            "IP": [6.0],
        }
    )

    _, _, recent_pitchers, latest_rows = generator.build_team_contexts(
        [frame],
        pd.Timestamp("2026-08-05"),
    )

    assert "hunter_brown" in latest_rows
    assert recent_pitchers["HOU"] == ["hunter_brown"]


def test_resolve_scheduled_player_team_prefers_market_identity() -> None:
    market_row = pd.Series({"Market_Player_Team": "CWS"})
    stale_history = pd.Series({"Team": "NYY", "Team_ID": 147})

    team = generator.resolve_scheduled_player_team(
        market_row,
        stale_history,
        home_team="DET",
        home_team_id="116",
        away_team="CWS",
        away_team_id="145",
    )

    assert team == "CWS"


def test_resolve_scheduled_player_team_uses_current_schedule_ids_for_legacy_snapshot() -> None:
    legacy_market_row = pd.Series({"Market_Home_Team": "DET", "Market_Away_Team": "CWS"})
    latest_history = pd.Series({"Team": "CWS", "Team_ID": 145, "Opponent_ID": 147})

    team = generator.resolve_scheduled_player_team(
        legacy_market_row,
        latest_history,
        home_team="DET",
        home_team_id="116",
        away_team="CWS",
        away_team_id="145",
    )

    assert team == "CWS"


def test_projection_context_regresses_short_term_total_base_spike() -> None:
    history = pd.DataFrame(
        {
            "TB": ([1.5] * 25) + [7.0, 2.0, 10.0, 4.0, 3.0],
            "Team_PA_share": [0.10] * 30,
            "wOBA": [0.340] * 30,
            "ISO": [0.180] * 30,
            "Barrel%": [9.0] * 30,
            "Batting_Order": [4.0] * 30,
        }
    )
    spec = next(item for item in generator.TARGET_SPECS if item.target == "TB")
    context = generator.build_player_projection_context(history, spec)
    latest = pd.Series(
        {
            "TB_rolling_avg": 5.4,
            "TB_lag1": 4.0,
            "Team_PA_share": 0.10,
            "wOBA": 0.920,
            "ISO": 1.0,
            "Barrel%": 30.0,
            "Park_Factor": 1.0,
            "Temp_F": 70.0,
        }
    )

    prediction, _ = generator.project_from_latest_row(
        latest,
        spec,
        opponent_context={},
        player_context=context,
    )

    assert prediction < 3.0


def test_pitcher_projection_uses_recent_starter_workload() -> None:
    history = pd.DataFrame(
        {
            "K": [2.0, 6.0, 7.0, 5.0, 8.0, 6.0],
            "IP": [1.0, 5.2, 6.0, 5.1, 6.1, 5.2],
            "Pitches": [24.0, 84.0, 96.0, 88.0, 101.0, 92.0],
            "Was_Starter": [0, 1, 1, 1, 1, 1],
        }
    )
    spec = next(item for item in generator.TARGET_SPECS if item.target == "K")
    context = generator.build_player_projection_context(history, spec)
    latest = pd.Series(
        {
            "K_rolling_avg": 3.0,
            "K_lag1": 3.0,
            "Park_Factor": 1.0,
            "Opp_Lineup_K_rate_3": 0.225,
        }
    )

    prediction, _ = generator.project_from_latest_row(
        latest,
        spec,
        opponent_context={"opp_lineup_k_rate": 0.225},
        player_context=context,
    )

    assert context["starter_history_rows"] == 5
    assert context["projected_ip"] >= 5.0
    assert context["projected_pitches"] >= 85.0
    assert prediction > 4.0


def test_market_snapshot_prefers_major_book_standard_line_and_best_price(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "Market_Date": "2026-07-29",
                "Player": "Example_Player",
                "Market_R": 1.5,
                "Market_R_books": 3,
                "Market_R_over_price": 136.418,
                "Market_R_under_price": -181.779,
                "Market_Source_R": "real",
            }
        ]
    ).to_csv(tmp_path / "latest_player_props_wide.csv", index=False)
    rows = [
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "draftkings",
            "line": 0.5,
            "over_price": 125,
            "under_price": -155,
        },
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "fanduel",
            "line": 0.5,
            "over_price": 130,
            "under_price": -160,
        },
        {
            "event_date_et": "2026-07-29",
            "player_name_norm": "Example_Player",
            "market_key": "batter_runs_scored",
            "bookmaker_key": "fanduel",
            "line": 1.5,
            "over_price": 450,
            "under_price": -700,
        },
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "latest_player_props_long.csv", index=False)

    snapshot = generator.load_market_snapshot(tmp_path, pd.Timestamp("2026-07-29"))
    market = snapshot.iloc[0]

    assert market["Market_R"] == 0.5
    assert market["Market_R_over_price"] != 136.418
    assert market["Market_R_common_books"] == 2
    assert market["Market_R_over_price"] == 130
    assert market["Market_R_over_book_key"] == "fanduel"
    assert market["Market_R_under_price"] == -155
    assert market["Market_R_under_book_key"] == "draftkings"


def test_market_snapshot_uses_requested_history_date_and_latest_capture(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "event_date_et": "2026-07-30",
                "player_name_norm": "Newer_Player",
                "market_key": "batter_total_bases",
                "bookmaker_key": "draftkings",
                "line": 1.5,
                "over_price": -110,
                "under_price": -110,
            }
        ]
    ).to_csv(tmp_path / "latest_player_props_long.csv", index=False)
    pd.DataFrame(
        [
            {
                "fetched_at_utc": "2026-07-29T12:00:00Z",
                "event_date_et": "2026-07-29",
                "player_name_norm": "History_Player",
                "market_key": "batter_total_bases",
                "bookmaker_key": "draftkings",
                "line": 2.5,
                "over_price": 140,
                "under_price": -170,
            },
            {
                "fetched_at_utc": "2026-07-29T13:00:00Z",
                "event_date_et": "2026-07-29",
                "player_name_norm": "History_Player",
                "market_key": "batter_total_bases",
                "bookmaker_key": "draftkings",
                "line": 1.5,
                "over_price": 115,
                "under_price": -135,
            },
        ]
    ).to_csv(tmp_path / "history_player_props_long.csv", index=False)

    snapshot = generator.load_market_snapshot(tmp_path, pd.Timestamp("2026-07-29"))

    assert snapshot.iloc[0]["Player"] == "History_Player"
    assert snapshot.iloc[0]["Market_TB"] == 1.5


def test_standard_hitter_market_is_omitted_when_only_alternate_line_is_offered(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "event_date_et": "2026-07-29",
                "player_name_norm": "Alternate_Only",
                "market_key": "batter_runs_scored",
                "bookmaker_key": "fanduel",
                "line": 1.5,
                "over_price": 350,
                "under_price": -500,
            }
        ]
    ).to_csv(tmp_path / "latest_player_props_long.csv", index=False)

    snapshot = generator.load_market_snapshot(tmp_path, pd.Timestamp("2026-07-29"))

    assert "Market_R" not in snapshot.columns
