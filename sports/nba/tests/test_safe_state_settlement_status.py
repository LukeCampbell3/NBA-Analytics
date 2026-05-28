from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.safe_state.evaluate_safe_state_shadow_results import evaluate_safe_state_shadow_results


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::test",
        "game_date": "2026-05-26",
        "market_date": "2026-05-26",
        "player": "Test Player",
        "target": "PTS",
        "market_type": "PTS_OVER",
        "side": "OVER",
        "direction": "OVER",
        "line": 10.0,
        "market_side_decimal_odds": 1.91,
        "stress_probability": 0.60,
    }
    row.update(overrides)
    return row


def _write_boards(tmp_path: Path, frame: pd.DataFrame) -> None:
    for name in [
        "production_board_as_is",
        "price_defense_only_board",
        "safe_state_core_board",
        "safe_state_near_core_board",
        "true_unstable_shadow_rejections",
        "needs_more_sample_queue",
    ]:
        frame.to_csv(tmp_path / f"{name}.csv", index=False)


def test_unresolved_rows_become_pending_not_push(tmp_path: Path) -> None:
    _write_boards(tmp_path, pd.DataFrame([_row()]))
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    production = next(row for row in report["variant_metrics"] if row["variant"] == "production_board_as_is")

    assert production["pending_rows"] == 1
    assert production["resolved_rows"] == 0
    assert production["pushes"] == 0


def test_actual_stat_equal_to_line_becomes_settled_push(tmp_path: Path) -> None:
    _write_boards(tmp_path, pd.DataFrame([_row(actual_stat=10.0)]))
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    production = next(row for row in report["variant_metrics"] if row["variant"] == "production_board_as_is")

    assert production["resolved_rows"] == 1
    assert production["pushes"] == 1
    assert production["pending_rows"] == 0


def test_win_loss_rows_resolve_correctly(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row(candidate_id="win", actual_stat=11.0), _row(candidate_id="loss", actual_stat=9.0)])
    _write_boards(tmp_path, frame)
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    production = next(row for row in report["variant_metrics"] if row["variant"] == "production_board_as_is")

    assert production["resolved_rows"] == 2
    assert production["wins"] == 1
    assert production["losses"] == 1
    assert production["pushes"] == 0


def test_hit_rate_and_roi_ignore_pending_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame([_row(candidate_id="win", actual_stat=11.0), _row(candidate_id="pending")])
    _write_boards(tmp_path, frame)
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    production = next(row for row in report["variant_metrics"] if row["variant"] == "production_board_as_is")

    assert production["pending_rows"] == 1
    assert production["resolved_rows"] == 1
    assert production["hit_rate"] == 1.0
    assert round(production["profit_units"], 2) == 0.91


def test_resolved_rows_equals_wins_losses_pushes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            _row(candidate_id="win", actual_stat=11.0),
            _row(candidate_id="loss", actual_stat=9.0),
            _row(candidate_id="push", actual_stat=10.0),
            _row(candidate_id="pending"),
        ]
    )
    _write_boards(tmp_path, frame)
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    production = next(row for row in report["variant_metrics"] if row["variant"] == "production_board_as_is")

    assert production["resolved_rows"] == production["wins"] + production["losses"] + production["pushes"]


def test_single_slate_pending_report_cannot_make_promotion_claim(tmp_path: Path) -> None:
    _write_boards(tmp_path, pd.DataFrame([_row()]))
    report = evaluate_safe_state_shadow_results(board_dir=tmp_path)
    audit = pd.read_csv(tmp_path / "safe_state_settlement_status_audit.csv")

    assert report["promotion_ready"] is False
    assert report["promotion_claim"] is False
    assert report["critical_questions"]["does_price_defense_alone_help"] == "requires_settlement"
    assert audit.iloc[0]["settlement_status"] == "PENDING"
