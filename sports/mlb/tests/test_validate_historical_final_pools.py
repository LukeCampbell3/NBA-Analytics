from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import validate_historical_final_pools as validator


def test_actual_lookup_covers_every_supported_player_prop(tmp_path: Path) -> None:
    player_dir = tmp_path / "Example_Player"
    player_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2026-07-01",
                "Player": "Example Player",
                "Game_ID": "game_1",
                "H": 2,
                "TB": 4,
                "R": 1,
                "HR": 1,
                "RBI": 3,
                "K": 0,
                "ER": 0,
            }
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)

    lookup = validator.build_actual_lookup(tmp_path)

    assert set(validator.TARGET_TO_ACTUAL_COL) == {"H", "TB", "R", "HR", "RBI", "K", "ER"}
    for target, expected in {"H": 2, "TB": 4, "R": 1, "HR": 1, "RBI": 3, "K": 0, "ER": 0}.items():
        assert lookup[("2026-07-01", "example_player", target, "game_1")] == expected


def test_price_validation_requires_integer_american_odds() -> None:
    assert validator.is_valid_american_price(-110)
    assert validator.is_valid_american_price(125.0)
    assert not validator.is_valid_american_price(-1.67)
    assert not validator.is_valid_american_price(-168.686524)
    assert not validator.is_valid_american_price(99)


def test_summary_includes_uncertainty_interval() -> None:
    rows = [
        {
            "run_date": "2026-07-01",
            "result": result,
            "units": 1.0 if result == "win" else -1.0,
            "line_placeable": True,
            "price_confirmed": True,
        }
        for result in ["win", "win", "win", "loss"]
    ]

    summary = validator.summarize_rows(rows)

    assert summary["hit_rate"] == 0.75
    assert summary["hit_rate_wilson_95_low"] < summary["hit_rate"]
    assert summary["hit_rate_wilson_95_high"] > summary["hit_rate"]


def test_profile_promotion_stays_in_probation_until_sample_and_uncertainty_clear() -> None:
    assessment = validator.assess_profile_promotion(
        {
            "play_count": 22,
            "graded_play_count": 22,
            "hit_rate_wilson_95_low": 0.4295,
            "priced_roi": 0.3986,
            "price_confirmed_count": 22,
        }
    )

    assert assessment["status"] == "probation"
    assert assessment["eligible_for_review"] is False
    assert "fewer than 50 graded plays" in assessment["reasons"]


def test_over_maturity_route_keeps_core_and_only_mature_over_rows() -> None:
    rows = [
        {
            "run_date": "2026-05-01",
            "selection_profile": "core_market_v1",
            "history_rows": 40,
            "selected_side_price": -210,
            "result": "win",
            "units": 0.9,
            "line_placeable": True,
            "price_confirmed": True,
        },
        {
            "run_date": "2026-05-01",
            "selection_profile": validator.OPTIMIZED_OVER_SELECTION_PROFILE,
            "history_rows": 40,
            "selected_side_price": 120,
            "result": "loss",
            "units": -1.0,
            "line_placeable": True,
            "price_confirmed": True,
        },
        {
            "run_date": "2026-06-10",
            "selection_profile": validator.OPTIMIZED_OVER_SELECTION_PROFILE,
            "history_rows": 70,
            "selected_side_price": 110,
            "result": "win",
            "units": 1.1,
            "line_placeable": True,
            "price_confirmed": True,
        },
    ]

    route = validator.summarize_over_maturity_route(rows, min_history_rows=55)

    assert route["all_optimized_over"]["play_count"] == 2
    assert route["holdout_mature_optimized_over"]["play_count"] == 1
    assert route["combined_maturity_gated_policy"]["play_count"] == 2
    assert route["combined_maturity_gated_policy"]["hit_rate"] == 1.0
    assert route["premium_price_defended_policy"]["play_count"] == 2
