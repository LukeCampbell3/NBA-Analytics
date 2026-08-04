from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import live_board_confidence as confidence


def test_iter_main_board_paths_excludes_rescue_variants(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260801"
    run_dir.mkdir()
    main = run_dir / "daily_prediction_pool_20260801_high_precision_predictions.csv"
    rescue = run_dir / "daily_prediction_pool_20260801_real_market_rescue_high_precision_predictions.csv"
    wrong_parent = tmp_path / "20260802" / "daily_prediction_pool_20260801_high_precision_predictions.csv"
    wrong_parent.parent.mkdir()
    for path in (main, rescue, wrong_parent):
        path.write_text("Player\n", encoding="utf-8")

    assert confidence.iter_main_board_paths(tmp_path) == [main]


def test_build_live_board_calibration_is_cutoff_safe_and_shrunk(tmp_path: Path) -> None:
    daily_root = tmp_path / "daily"
    processed_root = tmp_path / "processed"
    player_dir = processed_root / "Example_Player"
    player_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Date": "2026-08-01", "Player": "Example Player", "Game_ID": "1", "TB": 0},
            {"Date": "2026-08-02", "Player": "Example Player", "Game_ID": "2", "TB": 3},
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)

    for stamp, game_id, probability in (("20260801", "1", 0.70), ("20260802", "2", 0.60)):
        run_dir = daily_root / stamp
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "Selection_Profile": "current",
                    "Game_Date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
                    "Game_ID": game_id,
                    "Player": "Example Player",
                    "Player_ID": "example_player",
                    "Target": "TB",
                    "Direction": "OVER",
                    "Market_Line": 1.5,
                    "Estimated_Graded_Hit_Rate": probability,
                    "Price_Confirmed": 1,
                    "Selected_Side_Price": 110,
                }
            ]
        ).to_csv(run_dir / f"daily_prediction_pool_{stamp}_high_precision_predictions.csv", index=False)

    payload = confidence.build_live_board_calibration(
        daily_runs_root=daily_root,
        processed_root=processed_root,
        season=2026,
        before_date=date(2026, 8, 2),
        prior_strength=4.0,
        max_abs_adjustment=0.05,
        min_segment_rows=1,
    )

    segment = payload["segments"]["TB|OVER"]
    assert payload["graded_rows"] == 1
    assert segment["wins"] == 0
    assert segment["mean_probability"] == 0.70
    assert segment["adjustment"] == -0.05
    assert payload["walk_forward_validation"]["rows"] == 1
    assert payload["walk_forward_validation"]["adjusted_rows"] == 0


def test_apply_live_board_calibration_uses_active_segment_only() -> None:
    payload = {
        "segments": {
            "TB|OVER": {"active": True, "graded_rows": 6, "adjustment": -0.01},
            "TB|UNDER": {"active": False, "graded_rows": 2, "adjustment": 0.04},
        }
    }

    calibrated, key, support, adjustment = confidence.apply_live_board_calibration(
        0.55, payload, target="TB", direction="OVER"
    )
    assert calibrated == 0.54
    assert key == "TB|OVER"
    assert support == 6
    assert adjustment == -0.01

    unchanged, source, support, adjustment = confidence.apply_live_board_calibration(
        0.75, payload, target="TB", direction="UNDER"
    )
    assert unchanged == 0.75
    assert source == "insufficient_support"
    assert support == 2
    assert adjustment == 0.0
