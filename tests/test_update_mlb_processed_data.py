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
