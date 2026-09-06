from __future__ import annotations

import csv
from pathlib import Path

from sports.mlb.scripts.collect_historical_game_conditioned_outcomes import build_outcome_record, write_ledger
from sports.mlb.scripts.fit_game_conditioned_hitter_moe_outcome_joined import _collect_joined_examples


def _write_player(path: Path, *, current_h: int, current_tb: int, current_hr: int, current_gap: float, current_roll: float) -> None:
    fields = [
        "Date", "Player", "Player_MLBAM_ID", "Player_Type", "Team", "Opponent", "Season", "Game_ID",
        "Game_Index", "Team_ID", "Opponent_ID", "Is_Home", "Opp_Starter_ID", "Opp_Starter_Player",
        "H", "TB", "R", "HR", "RBI", "PA", "AB", "BB", "SO", "Batting_Order", "wOBA", "xwOBA",
        "Barrel%", "HardHit%", "Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3", "Park_Factor", "Temp_F",
        "H_market_gap", "TB_market_gap", "HR_market_gap", "H_rolling_avg", "TB_rolling_avg", "HR_rolling_avg",
    ]
    rows = [
        ["2026-04-01", "Test_Hitter", 123, "hitter", "AAA", "BBB", 2026, "g1", 0, 1, 2, 1, 9001, "Pitcher A", 1, 1, 0, 0, 0, 4, 4, 0, 1, 2, .320, .330, 8, 40, 4.0, 8.0, 1.0, 70, .1, .1, .1, 1.0, 1.0, 0.0],
        ["2026-04-02", "Test_Hitter", 123, "hitter", "AAA", "CCC", 2026, "g2", 1, 1, 3, 0, 9002, "Pitcher B", 2, 4, 1, 1, 2, 4, 3, 1, 0, 2, .500, .520, 20, 60, 3.5, 7.0, 1.0, 75, .2, .2, .2, 1.5, 2.5, .5],
        ["2026-04-03", "Test_Hitter", 123, "hitter", "AAA", "DDD", 2026, "g3", 2, 1, 4, 1, 9003, "Pitcher C", current_h, current_tb, 0, current_hr, 0, 4, 4, 0, 1, 2, .250, .280, 5, 35, 3.8, 9.0, 1.0, 72, current_gap, current_gap, current_gap, current_roll, current_roll, current_roll],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _write_ledger(path: Path) -> None:
    raw = {
        "Date": "2026-04-03", "Player": "Test_Hitter", "Player_MLBAM_ID": "123", "Player_Type": "hitter",
        "Team": "AAA", "Opponent": "DDD", "Game_ID": "g3", "H": "1", "TB": "1", "HR": "0", "PA": "4", "AB": "4",
    }
    record, reason = build_outcome_record(raw, source_file="fixture.csv", source_row=2, season=2026)
    assert reason is None and record is not None
    write_ledger([record], path)


def _run(root: Path, ledger: Path):
    rows, join = _collect_joined_examples(
        root,
        outcome_ledger_path=ledger,
        season=2026,
        max_games=1,
        trials=30,
        min_history=2,
    )
    return rows, join


def test_current_game_outcome_and_derived_fields_cannot_change_features_or_prior(tmp_path):
    ledger = tmp_path / "outcomes.jsonl.gz"
    _write_ledger(ledger)

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_player(root_a / "Test_Hitter" / "2026_processed_processed.csv", current_h=1, current_tb=1, current_hr=0, current_gap=.2, current_roll=1.0)
    _write_player(root_b / "Test_Hitter" / "2026_processed_processed.csv", current_h=4, current_tb=12, current_hr=2, current_gap=99.0, current_roll=99.0)

    rows_a, join_a = _run(root_a, ledger)
    rows_b, join_b = _run(root_b, ledger)

    assert len(rows_a) == len(rows_b) == 3
    for left, right in zip(rows_a, rows_b):
        assert left["target"] == right["target"]
        assert left["outcome"] == right["outcome"]
        assert left["actual"] == right["actual"]
        assert left["outcome_sha256"] == right["outcome_sha256"]
        assert left["prior_probability"] == right["prior_probability"]
        assert left["features"] == right["features"]

    assert join_a["outcomes_read_from_feature_row"] is False
    assert join_a["current_game_realized_columns_masked"] is True
    assert join_a["current_game_target_derived_columns_masked"] is True
    assert join_a["current_row_market_gap_or_rolling_used_in_prior"] is False
    assert join_b["joined_hitter_games"] == 1
