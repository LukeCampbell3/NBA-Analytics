from __future__ import annotations

import pandas as pd

from sports.nfl.predictions.latent import build_sequence_table
from sports.nfl.predictions.pbp_stats import aggregate_player_stats_from_pbp
from sports.nfl.tests.test_pipeline import make_stats


def test_latent_sequence_never_reads_current_game_outcome() -> None:
    original = make_stats(seasons=[2024], players=1, weeks=6)
    modified = original.copy()
    outcome_columns = [
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "attempts",
        "carries",
        "targets",
    ]
    modified.loc[modified["week"].eq(6), outcome_columns] = 9999.0

    original_table, features, _ = build_sequence_table(original, sequence_length=5)
    modified_table, _, _ = build_sequence_table(modified, sequence_length=5)

    original_row = original_table.loc[original_table["week"].eq(6), features].reset_index(drop=True)
    modified_row = modified_table.loc[modified_table["week"].eq(6), features].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_row, modified_row)


def test_play_by_play_aggregation_builds_weekly_contract() -> None:
    plays = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "posteam": ["BUF", "BUF"],
            "defteam": ["NYJ", "NYJ"],
            "passer_player_id": ["qb", None],
            "passer_player_name": ["Q.Back", None],
            "rusher_player_id": [None, "rb"],
            "rusher_player_name": [None, "R.Back"],
            "receiver_player_id": ["wr", None],
            "receiver_player_name": ["W.Receiver", None],
            "complete_pass": [1, 0],
            "incomplete_pass": [0, 0],
            "interception": [0, 0],
            "pass_touchdown": [1, 0],
            "rush_attempt": [0, 1],
            "rush_touchdown": [0, 0],
            "passing_yards": [25, 0],
            "rushing_yards": [0, 7],
            "receiving_yards": [25, 0],
            "air_yards": [15, 0],
            "epa": [1.2, 0.3],
        }
    )
    roster = pd.DataFrame(
        {
            "gsis_id": ["qb", "rb", "wr"],
            "season": [2025] * 3,
            "week": [1] * 3,
            "team": ["BUF"] * 3,
            "full_name": ["Quarter Back", "Running Back", "Wide Receiver"],
            "position": ["QB", "RB", "WR"],
        }
    )

    output = aggregate_player_stats_from_pbp(plays, roster).set_index("player_id")

    assert output.loc["qb", "attempts"] == 1
    assert output.loc["qb", "passing_yards"] == 25
    assert output.loc["wr", "targets"] == 1
    assert output.loc["wr", "receiving_yards"] == 25
    assert output.loc["rb", "carries"] == 1
    assert output.loc["rb", "rushing_yards"] == 7
