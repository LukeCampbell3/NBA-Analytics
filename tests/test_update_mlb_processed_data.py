from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "Player-Predictor" / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

import update_mlb_processed_data as updater


def test_deduplicate_player_games_keeps_one_row_per_role_and_game() -> None:
    frame = pd.DataFrame(
        [
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824912", "Date": "2026-06-16", "H": 1},
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824912", "Date": "2026-06-16", "H": 1},
            {"Player": "Austin_Riley", "Player_Type": "hitter", "Game_ID": "824913", "Date": "2026-06-17", "H": 3},
        ]
    )

    result = updater.deduplicate_player_games(frame)

    assert len(result) == 2
    assert result["Game_ID"].tolist() == ["824912", "824913"]


def test_matchup_network_is_walk_forward_and_uses_stable_starter_id() -> None:
    rows = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-04-01"), "Game_Index": 0,
                "Player": "Test_Pitcher", "Player_MLBAM_ID": 42, "Player_Type": "pitcher",
                "Was_Starter": 1, "IP": 5.0, "BF": 22, "Pitches": 85, "K": 3,
                "ERA": 5.4, "FIP": 5.1, "H_allowed": 7, "HR_allowed": 1, "BB_allowed": 3,
            },
            {
                "Date": pd.Timestamp("2026-04-02"), "Game_Index": 0,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 0, "wOBA": 0.4, "ISO": 0.2, "HardHit%": 50,
                "Barrel%": 12, "Batting_Order": 3, "H": 2, "TB": 3, "R": 1, "HR": 0, "RBI": 1,
            },
            {
                "Date": pd.Timestamp("2026-04-08"), "Game_Index": 1,
                "Player": "Test_Hitter", "Player_MLBAM_ID": 7, "Player_Type": "hitter",
                "Opp_Starter_ID": 42, "Opp_Starter_Player": "Test_Pitcher",
                "PA": 4, "SO": 1, "wOBA": 0.3, "ISO": 0.1, "HardHit%": 35,
                "Barrel%": 6, "Batting_Order": 4, "H": 1, "TB": 1, "R": 0, "HR": 0, "RBI": 0,
            },
        ]
    )

    first = updater.attach_walk_forward_matchup_network(rows)
    changed = rows.copy()
    changed.loc[changed["Date"].eq(pd.Timestamp("2026-04-08")), "H"] = 10
    second = updater.attach_walk_forward_matchup_network(changed)

    later = first.loc[first["Date"].eq(pd.Timestamp("2026-04-08"))].iloc[0]
    later_changed = second.loc[second["Date"].eq(pd.Timestamp("2026-04-08"))].iloc[0]
    assert later["Batter_Vs_Starter_Games"] == 1
    assert later["Matchup_Network_H_Adjustment"] == later_changed["Matchup_Network_H_Adjustment"]
