from __future__ import annotations

import pandas as pd

from sports.mlb.decision_engine.matchup_network import (
    NETWORK_VERSION,
    build_matchup_network_signal,
)


def hitter_history(*, strong: bool = True, opposing_pitcher_id: int = 0) -> pd.DataFrame:
    games = 30
    return pd.DataFrame(
        {
            "PA": [4.2] * games,
            "SO": [0.45 if strong else 1.35] * games,
            "wOBA": [0.390 if strong else 0.260] * games,
            "ISO": [0.240 if strong else 0.080] * games,
            "HardHit%": [50.0 if strong else 27.0] * games,
            "Barrel%": [13.0 if strong else 3.0] * games,
            "Batting_Order": [3.0 if strong else 8.0] * games,
            "H": [1.4 if strong else 0.5] * games,
            "TB": [2.4 if strong else 0.7] * games,
            "R": [0.8 if strong else 0.2] * games,
            "HR": [0.3 if strong else 0.03] * games,
            "RBI": [0.8 if strong else 0.2] * games,
            "Opp_Starter_ID": ([opposing_pitcher_id] * 3) + ([0] * (games - 3)),
        }
    )


def pitcher_history(*, vulnerable: bool = True, starts: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Was_Starter": [1] * starts,
            "IP": [5.4 if vulnerable else 6.2] * starts,
            "BF": [24.0] * starts,
            "Pitches": [91.0] * starts,
            "K": [3.0 if vulnerable else 8.0] * starts,
            "ERA": [5.7 if vulnerable else 2.7] * starts,
            "FIP": [5.3 if vulnerable else 2.9] * starts,
            "H_allowed": [7.0 if vulnerable else 4.0] * starts,
            "HR_allowed": [1.4 if vulnerable else 0.4] * starts,
            "BB_allowed": [3.0 if vulnerable else 1.0] * starts,
        }
    )


def test_network_promotes_strong_batter_against_vulnerable_pitcher() -> None:
    signal = build_matchup_network_signal(
        hitter_history(strong=True),
        pitcher_history(vulnerable=True),
    )

    assert signal.version == NETWORK_VERSION
    assert signal.pitcher_uncertainty < 0.25
    assert signal.network_score["H"] > 0.5
    assert signal.adjustment["H"] > 0.0
    assert signal.adjustment["TB"] > signal.adjustment["H"]


def test_uncertain_pitcher_does_not_create_edge_without_direct_evidence() -> None:
    strong = build_matchup_network_signal(hitter_history(strong=True), pd.DataFrame())
    weak = build_matchup_network_signal(hitter_history(strong=False), pd.DataFrame())

    assert strong.pitcher_uncertainty == 1.0
    assert strong.adjustment["H"] == 0.0
    assert weak.adjustment["H"] == 0.0


def test_direct_success_can_create_bounded_edge_against_uncertain_pitcher() -> None:
    history = hitter_history(strong=True, opposing_pitcher_id=42)
    history.loc[history["Opp_Starter_ID"].eq(42), "H"] = 3.0
    signal = build_matchup_network_signal(
        history,
        pd.DataFrame(),
        opposing_pitcher_id=42,
    )

    assert signal.pitcher_uncertainty == 1.0
    assert signal.direct_matchup_games == 3
    assert 0.0 < signal.adjustment["H"] <= 0.10


def test_direct_starter_history_is_shrunk_and_identified_by_stable_id() -> None:
    history = hitter_history(strong=True, opposing_pitcher_id=42)
    history.loc[history["Opp_Starter_ID"].eq(42), "H"] = 3.0
    signal = build_matchup_network_signal(
        history,
        pitcher_history(vulnerable=False),
        opposing_pitcher_id=42,
    )

    assert signal.direct_matchup_games == 3
    assert 0.0 < signal.direct_matchup_lift["H"] < 1.0
    assert signal.adjustment["H"] <= 0.10


def test_explicit_starter_flag_excludes_long_relief_outings() -> None:
    history = pitcher_history(vulnerable=True, starts=5)
    history.loc[len(history)] = {
        "Was_Starter": 0,
        "IP": 4.0,
        "BF": 18.0,
        "Pitches": 65.0,
        "K": 8.0,
        "ERA": 0.0,
        "FIP": 0.0,
        "H_allowed": 0.0,
        "HR_allowed": 0.0,
        "BB_allowed": 0.0,
    }
    signal = build_matchup_network_signal(hitter_history(), history)

    assert signal.pitcher_support == 0.25


def test_archetype_neighbors_borrow_shrunk_batter_results() -> None:
    history = hitter_history(strong=True)
    pitcher = pitcher_history(vulnerable=True)
    current = build_matchup_network_signal(history, pitcher)
    for target in ("H", "TB", "R", "HR", "RBI"):
        column = f"Pitcher_Profile_{target}_Vulnerability"
        history[column] = current.pitcher_vulnerability[target] - 1.5
        history.loc[:9, column] = current.pitcher_vulnerability[target]
    history["Matchup_Network_Pitcher_Support"] = 1.0
    history["Pitcher_Profile_Uncertainty"] = 0.2
    history.loc[:9, "TB"] = 4.0
    history.loc[10:, "TB"] = 0.5

    signal = build_matchup_network_signal(
        history,
        pitcher,
        opposing_pitcher_id=999,
    )

    assert signal.archetype_neighbor_games["TB"] == 30
    assert signal.archetype_neighbor_support["TB"] > 0.0
    assert signal.archetype_neighbor_lift["TB"] > 0.0


def test_archetype_neighbors_exclude_exact_starter_history() -> None:
    history = hitter_history(strong=True, opposing_pitcher_id=42)
    current = build_matchup_network_signal(history, pitcher_history(vulnerable=True))
    for target in ("H", "TB", "R", "HR", "RBI"):
        history[f"Pitcher_Profile_{target}_Vulnerability"] = current.pitcher_vulnerability[target]
    history["Matchup_Network_Pitcher_Support"] = 1.0
    history["Pitcher_Profile_Uncertainty"] = 0.2
    history["Opp_Starter_ID"] = 42

    signal = build_matchup_network_signal(
        history,
        pitcher_history(vulnerable=True),
        opposing_pitcher_id=42,
    )

    assert signal.direct_matchup_games == 30
    assert signal.archetype_neighbor_games["TB"] == 0
    assert signal.archetype_neighbor_lift["TB"] == 0.0
