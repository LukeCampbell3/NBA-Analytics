from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import generate_daily_prediction_pool as generator


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
