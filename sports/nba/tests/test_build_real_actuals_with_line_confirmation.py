from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.build_real_actuals_with_line_confirmation import (  # noqa: E402
    build_real_actuals,
)


def _candidate_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"candidate_id": "candidate::2026-05-28|test_over_win|PTS|OVER|20.5000", "player": "Test_Over_Win", "market_date": "2026-05-28", "target": "PTS", "direction": "OVER", "market_line": 20.5},
            {"candidate_id": "candidate::2026-05-28|test_under_win|PTS|UNDER|20.5000", "player": "Test_Under_Win", "market_date": "2026-05-28", "target": "PTS", "direction": "UNDER", "market_line": 20.5},
            {"candidate_id": "candidate::2026-05-28|test_push|TRB|OVER|10.0000", "player": "Test_Push", "market_date": "2026-05-28", "target": "TRB", "direction": "OVER", "market_line": 10.0},
            {"candidate_id": "candidate::2026-05-28|test_no_actual|AST|OVER|5.0000", "player": "Test_No_Actual", "market_date": "2026-05-28", "target": "AST", "direction": "OVER", "market_line": 5.0},
            {"candidate_id": "candidate::2026-05-28|test_line_mismatch|PTS|OVER|15.5000", "player": "Test_Line_Mismatch", "market_date": "2026-05-28", "target": "PTS", "direction": "OVER", "market_line": 15.5},
        ]
    )


def _write_data_proc(tmp_path: Path, player: str, date: str, pts=None, trb=None, ast=None) -> None:
    player_dir = tmp_path / player
    player_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Date": date, "PTS": pts, "TRB": trb, "AST": ast}]).to_csv(
        player_dir / "2026_processed_processed.csv", index=False
    )


def _write_historical_props(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "history_player_props_long.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_grades_over_and_under_wins_correctly(tmp_path) -> None:
    data_proc = tmp_path / "Data-Proc"
    _write_data_proc(data_proc, "Test_Over_Win", "2026-05-28", pts=25.0)
    _write_data_proc(data_proc, "Test_Under_Win", "2026-05-28", pts=15.0)
    _write_data_proc(data_proc, "Test_Push", "2026-05-28", trb=10.0)
    _write_data_proc(data_proc, "Test_Line_Mismatch", "2026-05-28", pts=20.0)
    historical_csv = _write_historical_props(tmp_path, [])

    actuals = build_real_actuals(_candidate_pool(), data_proc_root=data_proc, historical_props_csv=historical_csv)

    def _row(candidate_id_suffix: str) -> pd.Series:
        return actuals.loc[actuals["candidate_id"].str.contains(candidate_id_suffix)].iloc[0]

    assert _row("test_over_win")["actual_result"] == "win"
    assert _row("test_over_win")["actual_stat"] == 25.0
    assert _row("test_under_win")["actual_result"] == "win"
    assert _row("test_push")["actual_result"] == "push"


def test_missing_actual_stat_stays_pending_not_guessed(tmp_path) -> None:
    data_proc = tmp_path / "Data-Proc"
    # Test_No_Actual has no Data-Proc file at all
    historical_csv = _write_historical_props(tmp_path, [])
    actuals = build_real_actuals(_candidate_pool(), data_proc_root=data_proc, historical_props_csv=historical_csv)
    row = actuals.loc[actuals["candidate_id"].str.contains("test_no_actual")].iloc[0]
    assert row["actual_result"] is None or pd.isna(row["actual_result"])
    assert row["settlement_status"] == "PENDING_NO_ACTUAL_STAT"


def test_line_confirmed_when_real_books_agree_with_candidate(tmp_path) -> None:
    data_proc = tmp_path / "Data-Proc"
    _write_data_proc(data_proc, "Test_Over_Win", "2026-05-28", pts=25.0)
    historical_csv = _write_historical_props(
        tmp_path,
        [
            {"player_name_norm": "Test_Over_Win", "event_date_et": "2026-05-28", "market_key": "player_points", "bookmaker_key": "draftkings", "line": 20.5},
            {"player_name_norm": "Test_Over_Win", "event_date_et": "2026-05-28", "market_key": "player_points", "bookmaker_key": "fanduel", "line": 20.5},
        ],
    )
    actuals = build_real_actuals(_candidate_pool(), data_proc_root=data_proc, historical_props_csv=historical_csv)
    row = actuals.loc[actuals["candidate_id"].str.contains("test_over_win")].iloc[0]
    assert row["line_confirmed"] is True or row["line_confirmed"] == True  # noqa: E712
    assert row["historical_books_found"] == 2
    assert row["historical_consensus_line"] == 20.5
    assert row["line_discrepancy"] == 0.0


def test_line_discrepancy_flagged_never_silently_trusted(tmp_path) -> None:
    data_proc = tmp_path / "Data-Proc"
    _write_data_proc(data_proc, "Test_Line_Mismatch", "2026-05-28", pts=20.0)
    historical_csv = _write_historical_props(
        tmp_path,
        [
            {"player_name_norm": "Test_Line_Mismatch", "event_date_et": "2026-05-28", "market_key": "player_points", "bookmaker_key": "draftkings", "line": 16.5},
        ],
    )
    actuals = build_real_actuals(_candidate_pool(), data_proc_root=data_proc, historical_props_csv=historical_csv)
    row = actuals.loc[actuals["candidate_id"].str.contains("test_line_mismatch")].iloc[0]
    assert row["line_confirmed"] is False or row["line_confirmed"] == False  # noqa: E712
    assert row["historical_consensus_line"] == 16.5
    assert abs(row["line_discrepancy"] - 1.0) < 1e-9  # 16.5 - 15.5
    # grading still uses the candidate's own line -- 20.0 > 15.5 -> win --
    # but the discrepancy is visible, never hidden.
    assert row["actual_result"] == "win"


def test_no_historical_line_coverage_leaves_line_unconfirmed_not_guessed(tmp_path) -> None:
    data_proc = tmp_path / "Data-Proc"
    _write_data_proc(data_proc, "Test_Over_Win", "2026-05-28", pts=25.0)
    historical_csv = _write_historical_props(tmp_path, [])
    actuals = build_real_actuals(_candidate_pool(), data_proc_root=data_proc, historical_props_csv=historical_csv)
    row = actuals.loc[actuals["candidate_id"].str.contains("test_over_win")].iloc[0]
    assert row["line_confirmed"] is False or row["line_confirmed"] == False  # noqa: E712
    assert row["historical_books_found"] == 0
    assert pd.isna(row["historical_consensus_line"])
    # grading is unaffected -- still uses the real actual stat vs. the candidate's own line
    assert row["actual_result"] == "win"


def test_empty_candidate_pool_returns_empty_frame() -> None:
    assert build_real_actuals(pd.DataFrame()).empty
