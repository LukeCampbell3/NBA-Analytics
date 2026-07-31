from __future__ import annotations

import numpy as np
import pandas as pd

from sports.nfl.predictions.pipeline import TARGET_SPECS, build_features, train_target


def make_stats(*, seasons=range(2018, 2025), players: int = 8, weeks: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for season in seasons:
        for player_index in range(players):
            player_strength = 180 + (player_index * 9)
            for week in range(1, weeks + 1):
                attempts = 25 + (player_index % 5) + rng.normal(0, 2)
                passing_yards = max(0, player_strength + week * 2 + rng.normal(0, 22))
                rows.append(
                    {
                        "player_id": f"qb-{player_index}",
                        "player_display_name": f"Quarterback {player_index}",
                        "position": "QB",
                        "recent_team": f"T{player_index:02d}",
                        "opponent_team": f"T{(player_index + week) % players:02d}",
                        "season": season,
                        "week": week,
                        "passing_yards": passing_yards,
                        "rushing_yards": max(0, rng.normal(15, 7)),
                        "receiving_yards": 0.0,
                        "attempts": attempts,
                        "completions": attempts * 0.65,
                        "carries": max(0, rng.normal(3, 1)),
                        "targets": 0.0,
                        "receptions": 0.0,
                        "passing_tds": max(0, rng.normal(1.5, 0.8)),
                        "rushing_tds": 0.0,
                        "receiving_tds": 0.0,
                        "interceptions": max(0, rng.normal(0.7, 0.5)),
                        "passing_epa": rng.normal(4, 2),
                        "rushing_epa": rng.normal(0, 1),
                        "receiving_epa": 0.0,
                        "target_share": 0.0,
                        "air_yards_share": 0.0,
                        "wopr": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_features_are_shifted_before_current_game() -> None:
    stats = make_stats(seasons=[2024], players=1, weeks=5)
    spec = TARGET_SPECS[0]
    original, features = build_features(stats, spec)

    changed = stats.copy()
    changed.loc[changed["week"].eq(5), "passing_yards"] = 9999.0
    modified, _ = build_features(changed, spec)

    original_week = original.loc[original["week"].eq(5), features].reset_index(drop=True)
    modified_week = modified.loc[modified["week"].eq(5), features].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_week, modified_week)


def test_holdout_training_uses_only_prior_seasons() -> None:
    stats = make_stats()
    report, artifact, scored = train_target(
        stats,
        TARGET_SPECS[0],
        holdout_season=2024,
        meta_seasons=(2022, 2023),
        random_state=11,
    )

    assert report["holdout_season"] == 2024
    assert report["metrics"]["rows"] == len(scored)
    assert set(scored["season"]) == {2024}
    assert artifact["trained_through_season"] == 2024
    assert np.isfinite(scored["prediction"]).all()
    assert (scored["prediction"] >= 0).all()
