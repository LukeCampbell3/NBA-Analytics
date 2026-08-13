from __future__ import annotations

import pandas as pd

from sports.nfl.fantasy.model import (
    FantasyConfig,
    build_draft_rankings,
    fantasy_points,
)
from sports.nfl.fantasy.accuracy import build_accuracy_features


def test_full_ppr_scoring_contract() -> None:
    stats = pd.DataFrame(
        [
            {
                "passing_yards": 250,
                "passing_tds": 2,
                "interceptions": 1,
                "rushing_yards": 20,
                "rushing_tds": 1,
                "receptions": 3,
                "receiving_yards": 40,
                "receiving_tds": 1,
                "rushing_fumbles_lost": 1,
                "receiving_fumbles_lost": 0,
            }
        ]
    )

    assert fantasy_points(stats).iloc[0] == 35.0


def _history() -> pd.DataFrame:
    rows = []
    for week in range(1, 9):
        for player_id, name, position, team, receiving, rushing, passing in (
            ("wr-1", "Alpha Receiver", "WR", "AAA", 80 + week, 0, 0),
            ("rb-1", "Beta Runner", "RB", "BBB", 25, 65 + week, 0),
            ("qb-1", "Gamma Quarterback", "QB", "AAA", 0, 20, 250 + week),
            ("te-1", "Delta Tight End", "TE", "BBB", 45 + week, 0, 0),
        ):
            rows.append(
                {
                    "player_id": player_id,
                    "player_display_name": name,
                    "position": position,
                    "recent_team": team,
                    "opponent_team": "BBB" if team == "AAA" else "AAA",
                    "season": 2025,
                    "week": week,
                    "season_type": "REG",
                    "passing_yards": passing,
                    "passing_tds": 2 if position == "QB" else 0,
                    "interceptions": 0.5 if position == "QB" else 0,
                    "rushing_yards": rushing,
                    "rushing_tds": 0.4 if position in {"QB", "RB"} else 0,
                    "receptions": 6 if position in {"WR", "TE"} else 2 if position == "RB" else 0,
                    "receiving_yards": receiving,
                    "receiving_tds": 0.5 if position in {"WR", "TE"} else 0,
                }
            )
    return pd.DataFrame(rows)


def test_draft_rankings_are_deterministic_and_expose_distributions() -> None:
    roster = pd.DataFrame(
        [
            {"season": 2026, "gsis_id": "wr-1", "full_name": "Alpha Receiver", "team": "AAA", "position": "WR", "status": "ACT"},
            {"season": 2026, "gsis_id": "rb-1", "full_name": "Beta Runner", "team": "BBB", "position": "RB", "status": "ACT"},
            {"season": 2026, "gsis_id": "qb-1", "full_name": "Gamma Quarterback", "team": "AAA", "position": "QB", "status": "ACT"},
            {"season": 2026, "gsis_id": "te-1", "full_name": "Delta Tight End", "team": "BBB", "position": "TE", "status": "ACT"},
        ]
    )
    schedule = pd.DataFrame(
        [
            {"season": 2026, "week": week, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
            for week in range(1, 5)
        ]
    )
    config = FantasyConfig(
        season=2026,
        simulations=100,
        published_players=10,
        random_seed=7,
        replacement_ranks=(("QB", 1), ("RB", 1), ("WR", 1), ("TE", 1)),
    )

    first = build_draft_rankings(_history(), roster, schedule, config=config)
    second = build_draft_rankings(_history(), roster, schedule, config=config)

    assert [row["player_id"] for row in first["rankings"]] == [
        row["player_id"] for row in second["rankings"]
    ]
    assert first["players_simulated"] == 4
    assert {row["position"] for row in first["rankings"]} == {"QB", "RB", "WR", "TE"}
    assert all(row["fantasy_points"]["season_p10"] <= row["fantasy_points"]["season_p90"] for row in first["rankings"])
    assert all("per_game" in row["projected_stats"] and "season_total" in row["projected_stats"] for row in first["rankings"])


def test_below_replacement_players_keep_negative_vorp() -> None:
    history = _history()
    weak_qb = history.loc[history["player_id"].eq("qb-1")].copy()
    weak_qb["player_id"] = "qb-2"
    weak_qb["player_display_name"] = "Backup Quarterback"
    for column in ("passing_yards", "passing_tds", "rushing_yards", "rushing_tds"):
        weak_qb[column] = 0.0
    history = pd.concat([history, weak_qb], ignore_index=True)
    roster = pd.DataFrame(
        [
            {"season": 2026, "gsis_id": "qb-1", "full_name": "Gamma Quarterback", "team": "AAA", "position": "QB", "status": "ACT"},
            {"season": 2026, "gsis_id": "qb-2", "full_name": "Backup Quarterback", "team": "AAA", "position": "QB", "status": "ACT"},
        ]
    )
    schedule = pd.DataFrame(
        [{"season": 2026, "week": 1, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}]
    )
    payload = build_draft_rankings(
        history,
        roster,
        schedule,
        config=FantasyConfig(
            season=2026,
            simulations=50,
            published_players=10,
            random_seed=9,
            replacement_ranks=(("QB", 1), ("RB", 1), ("WR", 1), ("TE", 1)),
        ),
    )

    backup = next(row for row in payload["rankings"] if row["player_id"] == "qb-2")
    assert backup["value_over_replacement"] < 0


def test_accuracy_features_are_shifted_before_current_game() -> None:
    history = _history()
    history["fantasy_points_ppr_model"] = fantasy_points(history)
    features, _ = build_accuracy_features(history)
    receiver = features.loc[features["player_id"].eq("wr-1")].sort_values("week")

    assert pd.isna(receiver.iloc[0]["fantasy_points_ppr_model_mean3"])
    assert receiver.iloc[1]["fantasy_points_ppr_model_mean3"] == history.loc[
        history["player_id"].eq("wr-1") & history["week"].eq(1)
    ].pipe(fantasy_points).iloc[0]
    assert receiver.iloc[4]["history_games"] == 4
