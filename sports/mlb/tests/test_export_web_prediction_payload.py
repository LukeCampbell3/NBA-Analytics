from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import export_web_prediction_payload as exporter


def _row(*, rank: int, game_id: str, selection_score: float) -> dict[str, str]:
    return {
        "Rank": str(rank),
        "Prediction_Run_Date": "2026-06-17",
        "Game_Date": "2026-06-17",
        "Game_ID": game_id,
        "Player": "Duplicate Player",
        "Player_ID": "duplicate_player",
        "Team": "ATL",
        "Opponent": "SF",
        "Target": "TB",
        "Direction": "UNDER",
        "Market_Line": "1.0",
        "Selection_Score": str(selection_score),
        "Precision_Score": "0.9",
        "Abs_Edge": "0.7",
    }


def test_suppress_duplicate_props_ignores_game_id() -> None:
    rows = [
        _row(rank=1, game_id="resumed_game", selection_score=0.8),
        _row(rank=2, game_id="nightcap", selection_score=0.9),
    ]

    deduped, suppressed = exporter.suppress_duplicate_props(rows)

    assert len(deduped) == 1
    assert deduped[0]["Game_ID"] == "nightcap"
    assert len(suppressed) == 1
    assert suppressed[0]["reason"] == "duplicate prop identity on same slate"


def test_suppress_duplicate_props_prefers_matching_official_game_context() -> None:
    rows = [
        _row(rank=1, game_id="resumed_game", selection_score=0.9),
        _row(rank=2, game_id="nightcap", selection_score=0.8),
    ]
    game_context = {
        "resumed_game": {
            "official_date": "2026-06-16",
            "players": {
                "duplicate player": {"team": "SF"},
            },
        },
        "nightcap": {
            "official_date": "2026-06-17",
            "players": {
                "duplicate player": {"team": "ATL"},
            },
        },
    }

    deduped, _ = exporter.suppress_duplicate_props(rows, game_context)

    assert len(deduped) == 1
    assert deduped[0]["Game_ID"] == "nightcap"


def test_build_data_quality_marks_stale_board_for_review() -> None:
    quality = exporter.build_data_quality("2026-06-17", "2026-05-01", 2)

    assert quality["status"] == "review"
    assert quality["lag_days"] == 47
    assert len(quality["reasons"]) == 2


def test_build_data_quality_withholds_empty_board() -> None:
    quality = exporter.build_data_quality("2026-06-19", "2026-05-01", 0, play_count=0)

    assert quality["status"] == "withheld"
    assert quality["lag_days"] == 49
    assert "no plays passed publication filters" in quality["reasons"]


def test_infer_through_date_reads_raw_pool_for_empty_selection(tmp_path: Path) -> None:
    pool_csv = tmp_path / "daily_prediction_pool_20260619.csv"
    pool_csv.write_text("Last_History_Date\n2026-04-30\n2026-05-01\n", encoding="utf-8")

    assert exporter.infer_through_date({"pool_csv": str(pool_csv)}, []) == "2026-05-01"


def test_assert_no_date_regression_rejects_older_run(tmp_path: Path) -> None:
    output = tmp_path / "daily_predictions.json"
    output.write_text(json.dumps({"run_date": "2026-06-17"}), encoding="utf-8")

    try:
        exporter.assert_no_date_regression("2026-05-01", [output], allow_regression=False)
    except RuntimeError as exc:
        assert "Refusing to overwrite newer payload" in str(exc)
    else:
        raise AssertionError("Expected date regression to be rejected")
