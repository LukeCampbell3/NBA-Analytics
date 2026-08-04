from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import export_web_prediction_payload as exporter


def test_valid_american_price_rejects_non_american_consensus_values() -> None:
    assert exporter.valid_american_price(-110.0) == -110.0
    assert exporter.valid_american_price(125.0) == 125.0
    assert exporter.valid_american_price(-1.67) is None
    assert exporter.valid_american_price(-168.686524) is None
    assert exporter.valid_american_price(99.0) is None


def test_parlay_leg_profiles_require_price_value_and_calibration_support() -> None:
    assert exporter.is_mlb_parlay_leg_eligible(
        graded_hit_rate=0.76,
        leg_quality=0.81,
        historical_bucket_support=9000,
        expected_value_per_unit=0.10,
        price_confirmed=True,
        selected_side_price=-205,
        selected_sportsbook_key="caesars",
    )
    assert exporter.is_mlb_parlay_leg_eligible(
        graded_hit_rate=0.535,
        leg_quality=0.65,
        historical_bucket_support=6000,
        expected_value_per_unit=0.15,
        selection_profile="r_tb_over_moderate_edge_v1",
        price_confirmed=True,
        selected_side_price=117,
        selected_sportsbook_key="caesars",
        live_confidence_calibration_support=6,
    )
    assert not exporter.is_mlb_parlay_leg_eligible(
        graded_hit_rate=0.535,
        leg_quality=0.65,
        historical_bucket_support=6000,
        expected_value_per_unit=0.15,
        selection_profile="r_tb_over_moderate_edge_v1",
        price_confirmed=True,
        selected_side_price=117,
        selected_sportsbook_key="caesars",
        live_confidence_calibration_support=0,
    )


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


def test_suppress_closed_games_uses_live_official_state() -> None:
    scheduled = _row(rank=1, game_id="scheduled_game", selection_score=0.9)
    started = _row(rank=2, game_id="started_game", selection_score=0.8)
    final = _row(rank=3, game_id="final_game", selection_score=0.7)
    game_context = {
        "scheduled_game": {"abstract_state": "Preview", "status": "Pre-Game"},
        "started_game": {"abstract_state": "Live", "status": "In Progress"},
        "final_game": {"abstract_state": "Final", "status": "Final"},
    }

    kept, suppressed = exporter.suppress_closed_games(
        [scheduled, started, final],
        game_context,
    )

    assert [row["Game_ID"] for row in kept] == ["scheduled_game"]
    assert [row["game_id"] for row in suppressed] == ["started_game", "final_game"]
    assert all(item["reason"] == "game is no longer open for pregame predictions" for item in suppressed)


def test_suppress_closed_games_retains_rows_without_live_context() -> None:
    row = _row(rank=1, game_id="unknown_game", selection_score=0.9)

    kept, suppressed = exporter.suppress_closed_games([row], {})

    assert kept == [row]
    assert suppressed == []


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
