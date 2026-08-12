from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import train_selected_board_calibrator as trainer


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        run_date_col="run_date",
        target_col="target",
        direction_col="direction",
        prob_col="expected_win_rate",
        result_col="result",
    )


def test_resolve_input_columns_falls_back_to_alternate_column_names() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.58,
                "result": "win",
            }
        ]
    )

    resolved = trainer.resolve_input_columns(frame, _args())

    assert resolved.run_date_col == "market_date"
    assert resolved.prob_col == "estimated_win_rate"


def test_discover_rows_csv_prefers_newer_wider_validation_selector_file(tmp_path: Path) -> None:
    stale = pd.DataFrame(
        [
            {
                "run_date": "2026-04-23",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.58,
                "result": "win",
            },
            {
                "run_date": "2026-04-24",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.57,
                "result": "loss",
            },
        ]
    )
    stale.to_csv(tmp_path / "validation_recent_pool_selector_20260423_20260425_rows.csv", index=False)

    newer_validation = pd.DataFrame(
        [
            {
                "run_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.58,
                "result": "win",
            },
            {
                "run_date": "2026-04-25",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.57,
                "result": "loss",
            },
            {
                "run_date": "2026-04-26",
                "target": "TRB",
                "direction": "OVER",
                "expected_win_rate": 0.56,
                "result": "win",
            },
        ]
    )
    newer_validation.to_csv(tmp_path / "validation_recent_pool_selector_20260423_20260430_rows.csv", index=False)

    replay = pd.DataFrame(
        [
            {
                "run_date": "2026-04-24",
                "market_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.58,
                "result": "win",
            },
            {
                "run_date": "2026-04-25",
                "market_date": "2026-04-25",
                "target": "AST",
                "direction": "UNDER",
                "expected_win_rate": 0.57,
                "result": "loss",
            },
            {
                "run_date": "2026-04-26",
                "market_date": "2026-04-26",
                "target": "TRB",
                "direction": "OVER",
                "expected_win_rate": 0.56,
                "result": "win",
            },
            {
                "run_date": "2026-04-27",
                "market_date": "2026-04-27",
                "target": "PTS",
                "direction": "UNDER",
                "expected_win_rate": 0.59,
                "result": "win",
            },
        ]
    )
    replay.to_csv(tmp_path / "selector_replay_rows_rebuilt_20260423_20260430.csv", index=False)

    picked = trainer.discover_rows_csv(tmp_path, _args())

    assert picked.name == "validation_recent_pool_selector_20260423_20260430_rows.csv"


def test_discover_rows_csv_uses_shared_ledger_without_ignored_model_dir(
    tmp_path: Path, monkeypatch
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    pd.DataFrame(
        [
            {
                "market_date": "2026-04-24",
                "target": "PTS",
                "direction": "OVER",
                "estimated_win_rate": 0.58,
                "result": "win",
            }
        ]
    ).to_csv(shared / "validation_recent_pool_selector_rows.csv", index=False)
    missing_analysis = tmp_path / "model" / "analysis"
    monkeypatch.setattr(trainer, "ANALYSIS_ROOT", missing_analysis)
    monkeypatch.setattr(trainer, "SHARED_VALIDATION_ROOT", shared)

    picked = trainer.discover_rows_csv(missing_analysis, _args())

    assert picked == shared / "validation_recent_pool_selector_rows.csv"


def test_locked_calibration_selects_truthful_segment_policy() -> None:
    rows_path = REPO_ROOT / "sports/validation/validation_recent_pool_selector_20260406_20260430_rows.csv"
    raw = pd.read_csv(rows_path)
    rows = trainer._prepare_rows(raw, trainer.resolve_input_columns(raw, _args()))
    config = trainer.CalibratorFitConfig()

    evidence, payload = trainer.locked_calibration_evidence(
        rows,
        config,
        locked_days=5,
        rolling_min_train_days=7,
    )

    assert evidence["status"] == "passed"
    assert evidence["selected_method"] == "segment_monotonic_safety"
    assert evidence["locked_period"]["rows"] == 1067
    locked = {row["method"]: row for row in evidence["locked_comparison"]}
    assert abs(locked["segment_monotonic_safety"]["gap"]) < abs(locked["identity"]["gap"])
    assert locked["segment_monotonic_safety"]["brier"] < locked["identity"]["brier"]
    assert locked["segment_monotonic_safety"]["log_loss"] < locked["identity"]["log_loss"]
    assert all(not month["recent_segments"] for month in payload["months"].values())
