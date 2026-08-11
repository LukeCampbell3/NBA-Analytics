from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pick_survival_model as survival


def test_historical_candidates_require_real_playable_pregame_market(tmp_path: Path) -> None:
    player_dir = tmp_path / "Example_Player"
    player_dir.mkdir()
    base = {
        "Player": "Example Player",
        "Player_Type": "hitter",
        "Game_ID": "1",
        "R": 1,
        "Market_R": 0.5,
        "R_market_gap": 0.2,
        "Market_R_books": 5,
        "Market_R_over_price": 120,
        "Market_R_under_price": -140,
        "Market_R_line_std": 0.0,
    }
    pd.DataFrame(
        [
            {**base, "Date": "2026-04-01", "Market_Source_R": "real"},
            {**base, "Date": "2026-04-02", "Market_Source_R": "synthetic"},
            {**base, "Date": "2026-04-03", "Market_Source_R": "real", "Market_R_over_price": -1.2},
            {**base, "Date": "2026-05-01", "Market_Source_R": "real"},
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)

    rows = survival.build_historical_candidates(tmp_path, 2026, date(2026, 4, 15))

    assert len(rows) == 3
    assert set(rows["date"]) == {"2026-04-01", "2026-04-03"}
    assert set(rows["direction"]) == {"OVER", "UNDER"}
    assert len(rows.loc[rows["direction"].eq("OVER")]) == 1
    assert rows.loc[rows["direction"].eq("OVER"), "win"].tolist() == [1]
    assert rows.loc[rows["direction"].eq("UNDER"), "win"].tolist() == [0, 0]


def _training_rows() -> pd.DataFrame:
    records = []
    start = date(2026, 4, 1)
    for day_index in range(12):
        day = (start + timedelta(days=day_index)).isoformat()
        for row_index in range(24):
            probability = 0.35 + 0.025 * (row_index % 12)
            win = int((row_index + day_index) % 10 < int(probability * 10))
            records.append(
                {
                    "date": day,
                    "player": f"Player {row_index}",
                    "game_id": f"{day_index}-{row_index}",
                    "target": "R",
                    "direction": "OVER" if row_index % 2 else "UNDER",
                    "player_type": "hitter",
                    "projection": 0.7 if row_index % 2 else 0.3,
                    "market_line": 0.5,
                    "directional_edge": 0.2,
                    "edge_ratio": 0.4,
                    "model_hit_probability": probability,
                    "model_graded_hit_rate": probability,
                    "push_probability": 0.0,
                    "history_rows": 30.0 + row_index,
                    "market_books": 5.0,
                    "market_line_std": 0.0,
                    "market_implied_probability": 0.5,
                    "profit_per_unit": 1.0,
                    "side_price": 100.0,
                    "win": win,
                }
            )
    return pd.DataFrame(records)


def test_training_exports_untouched_holdout_and_shadow_contract() -> None:
    payload = survival.train_survival_model(_training_rows(), top_k=3)

    assert payload["status"] == "shadow"
    assert payload["shadow_only"] is True
    assert payload["split"]["holdout_start_date"] > payload["split"]["expanding_validation_end_date"]
    assert payload["split"]["expanding_validation_dates"] >= 2
    assert payload["expanding_oof_validation"]["survival_top_k"]["rows"] > 0
    assert payload["holdout"]["probability_metrics"]["rows"] > 0
    assert payload["promotion_gate"]["decision"] == "remain_shadow"
    assert set(payload["feature_contract"]["coefficients"]) == set(
        survival.NUMERIC_FEATURES + survival.CATEGORICAL_FEATURES
    )


def test_application_fails_closed_on_cutoff_and_low_segment_support() -> None:
    rows = _training_rows()
    payload = survival.train_survival_model(rows, top_k=3)
    candidate = SimpleNamespace(
        raw={"Player_Type": "hitter"},
        target="R",
        direction="OVER",
        prediction=0.7,
        market_line=0.5,
        history_rows=40,
        market_books=5,
        market_line_std=0.0,
        selected_side_price=110.0,
        opposite_side_price=-130.0,
        run_date=date(2026, 5, 1),
    )

    probability, expected_value, status, support, rank_active = survival.apply_pick_survival_model(candidate, payload)
    assert 0.0 < probability < 1.0
    assert expected_value is not None
    assert status == survival.MODEL_VERSION
    assert support == payload["segment_support"]["R|OVER"]
    assert rank_active is False

    payload["status"] = "active"
    payload["shadow_only"] = False
    payload["deployment_gate"]["authority"] = "rank_tiebreaker"
    assert survival.apply_pick_survival_model(candidate, payload)[4] is True

    candidate.run_date = date(2026, 4, 12)
    assert survival.apply_pick_survival_model(candidate, payload)[2] == "cutoff_violation"

    candidate.run_date = date(2026, 5, 1)
    candidate.target = "K"
    assert survival.apply_pick_survival_model(candidate, payload)[2] == "insufficient_segment_support"


def test_ranked_evaluation_reports_game_diversified_parlay() -> None:
    rows = pd.DataFrame(
        [
            {"date": "2026-04-01", "player": "A", "game_id": "1", "side_price": 100, "win": 1},
            {"date": "2026-04-01", "player": "B", "game_id": "1", "side_price": 100, "win": 1},
            {"date": "2026-04-01", "player": "C", "game_id": "2", "side_price": 100, "win": 1},
        ]
    )

    metrics = survival.evaluate_ranked(rows, [0.9, 0.8, 0.7], top_k=2, roi_weight=0.0)

    assert metrics["diversified_parlay_constraint"] == "maximum_one_leg_per_game"
    assert metrics["diversified_parlay_days"] == 1
    assert metrics["diversified_parlay_roi"] == 3.0
