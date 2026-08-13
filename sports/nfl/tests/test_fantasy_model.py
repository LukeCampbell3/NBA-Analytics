from __future__ import annotations

import pandas as pd

from sports.nfl.fantasy.model import (
    FantasyConfig,
    build_draft_rankings,
    fantasy_points,
)
from sports.nfl.fantasy.accuracy import build_accuracy_features
from sports.nfl.fantasy.lineup import build_lineup_contexts, merge_roster_with_depth_chart


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


def test_depth_chart_is_authoritative_for_team_and_adds_missing_players() -> None:
    roster = pd.DataFrame(
        [
            {
                "season": 2026,
                "gsis_id": "wr-1",
                "full_name": "Moved Receiver",
                "team": "OLD",
                "position": "WR",
                "status": "ACT",
            }
        ]
    )
    depth = pd.DataFrame(
        [
            {"dt": "2026-08-13", "team": "NEW", "player_name": "Moved Receiver", "gsis_id": "wr-1", "pos_abb": "WR", "pos_rank": 1},
            {"dt": "2026-08-13", "team": "NEW", "player_name": "Camp Receiver", "gsis_id": "wr-2", "pos_abb": "WR", "pos_rank": 2},
        ]
    )

    merged = merge_roster_with_depth_chart(roster, depth, season=2026).set_index("player_id")

    assert merged.loc["wr-1", "recent_team"] == "NEW"
    assert merged.loc["wr-1", "depth_rank"] == 1
    assert merged.loc["wr-2", "player_display_name"] == "Camp Receiver"


def test_team_opportunities_are_finite_and_added_receiver_reduces_share() -> None:
    history = _history()
    history["attempts"] = history["passing_yards"].gt(0).astype(float) * 30
    history["targets"] = history["receptions"] * 1.5
    history["carries"] = history["rushing_yards"].gt(0).astype(float) * 12
    base = pd.DataFrame(
        [
            {"player_id": "qb-1", "player_display_name": "Gamma Quarterback", "recent_team": "AAA", "position": "QB", "years_exp": 4, "depth_rank": 1},
            {"player_id": "wr-1", "player_display_name": "Alpha Receiver", "recent_team": "AAA", "position": "WR", "years_exp": 4, "depth_rank": 1},
        ]
    )
    expanded = pd.concat(
        [
            base,
            pd.DataFrame(
                [
                    {"player_id": "wr-2", "player_display_name": "New Receiver", "recent_team": "AAA", "position": "WR", "years_exp": 4, "depth_rank": 2}
                ]
            ),
        ],
        ignore_index=True,
    )

    base_context, _ = build_lineup_contexts(history, base)
    expanded_context, audits = build_lineup_contexts(history, expanded)
    audit = next(item for item in audits if item["team"] == "AAA")

    assert expanded_context["wr-1"]["conditional_opportunities"]["targets"] < base_context["wr-1"]["conditional_opportunities"]["targets"]
    assert audit["allocated_per_game"] == audit["budgets_per_game"]
    assert abs(sum(item["start_probability"] for item in audit["qb_scenarios"]) - 1.0) < 0.001


def test_deep_depth_chart_players_do_not_enter_draft_pool() -> None:
    history = _history()
    roster = pd.DataFrame(
        [
            {"season": 2026, "gsis_id": "wr-1", "full_name": "Alpha Receiver", "team": "AAA", "position": "WR", "status": "ACT"},
            {"season": 2026, "gsis_id": "wr-deep", "full_name": "Deep Receiver", "team": "AAA", "position": "WR", "status": "ACT"},
        ]
    )
    deep_history = history.loc[history["player_id"].eq("wr-1")].copy()
    deep_history["player_id"] = "wr-deep"
    deep_history["player_display_name"] = "Deep Receiver"
    history = pd.concat([history, deep_history], ignore_index=True)
    depth = pd.DataFrame(
        [
            {"dt": "2026-08-13", "team": "AAA", "player_name": "Alpha Receiver", "gsis_id": "wr-1", "pos_abb": "WR", "pos_rank": 1},
            {"dt": "2026-08-13", "team": "AAA", "player_name": "Deep Receiver", "gsis_id": "wr-deep", "pos_abb": "WR", "pos_rank": 14},
        ]
    )
    schedule = pd.DataFrame(
        [{"season": 2026, "week": 1, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}]
    )

    payload = build_draft_rankings(
        history,
        roster,
        schedule,
        depth_chart=depth,
        config=FantasyConfig(season=2026, simulations=20, published_players=10),
    )

    assert "wr-1" in {item["player_id"] for item in payload["rankings"]}
    assert "wr-deep" not in {item["player_id"] for item in payload["rankings"]}
    assert payload["players_excluded_by_lineup"] == 1
