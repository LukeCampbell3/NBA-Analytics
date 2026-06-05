from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from create_cards import generate_cards


@pytest.fixture()
def cards_dir(tmp_path: Path) -> Path:
    input_file = tmp_path / "players.csv"
    output_dir = tmp_path / "cards"
    pd.DataFrame(
        [
            {
                "player_name": "Fixture Player",
                "team": "LAL",
                "season": 2025,
                "position": "SG",
                "age": 25.0,
                "points_per_game": 18.0,
                "assists_per_game": 4.0,
                "rebounds_per_game": 5.0,
                "steals_per_game": 1.0,
                "blocks_per_game": 0.5,
                "turnovers_per_game": 2.0,
                "field_goal_attempts_per_game": 13.0,
                "three_point_attempts_per_game": 6.0,
                "minutes_per_game": 31.0,
                "games_played": 70,
                "usage_rate": 0.24,
                "plus_minus": 2.0,
            }
        ]
    ).to_csv(input_file, index=False)
    generate_cards(input_file, output_dir)
    return output_dir
