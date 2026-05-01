from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import select_high_precision_predictions as selector


def _row(
    *,
    player: str,
    team: str,
    game_id: str,
    target: str,
    prediction: float,
    line: float,
    edge: float,
) -> dict[str, str]:
    return {
        "Prediction_Run_Date": "2026-04-27",
        "Game_Date": "2026-04-27",
        "Commence_Time_UTC": "2026-04-27T23:00:00Z",
        "Game_ID": game_id,
        "Game_Status_Code": "P",
        "Game_Status_Detail": "Scheduled",
        "Player": player,
        "Player_ID": player.lower().replace(" ", "_"),
        "Player_Type": "hitter",
        "Team": team,
        "Opponent": "OPP",
        "Is_Home": "1",
        "Target": target,
        "Prediction": str(prediction),
        "Market_Line": str(line),
        "Market_Source": "real",
        "Edge": str(edge),
        "History_Rows": "30",
        "Last_History_Date": "2026-04-26",
        "Model_Selected": "et",
        "Model_Members": "et",
        "Model_Val_MAE": "0.75",
        "Model_Val_RMSE": "1.0",
    }


def test_lookup_historical_bucket_prior_prefers_line_bucket() -> None:
    calibration = {
        "target_direction": {
            "H|OVER": {"graded_rows": 500, "win_rate": 0.61},
        },
        "line_buckets": {
            "H|OVER|0.5": {"graded_rows": 200, "win_rate": 0.55},
        },
    }

    key, win_rate, support, source = selector.lookup_historical_bucket_prior(
        calibration,
        target="H",
        direction="OVER",
        market_line=0.5,
        min_line_rows=50,
    )

    assert key == "H|OVER|0.5"
    assert win_rate == 0.55
    assert support == 200
    assert source == "line_bucket"


def test_build_candidate_blends_model_probability_with_historical_prior() -> None:
    calibration = {
        "target_direction": {
            "H|OVER": {"graded_rows": 8000, "win_rate": 0.526},
        },
        "line_buckets": {
            "H|OVER|0.5": {"graded_rows": 7000, "win_rate": 0.551},
        },
    }

    candidate = selector.build_candidate(
        _row(
            player="Example Over",
            team="AAA",
            game_id="game_1",
            target="H",
            prediction=1.65,
            line=0.5,
            edge=1.15,
        ),
        calibration=calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert candidate is not None
    assert candidate.historical_prior_source == "line_bucket"
    assert candidate.historical_bucket_win_rate == 0.551
    assert candidate.calibrated_hit_probability < candidate.model_hit_probability
    assert candidate.historical_prior_weight > 0.0


def test_build_candidate_computes_price_aware_expected_value() -> None:
    calibration = {
        "target_direction": {
            "TB|UNDER": {"graded_rows": 4000, "win_rate": 0.84},
        },
        "line_buckets": {
            "TB|UNDER|1.5": {"graded_rows": 2500, "win_rate": 0.888},
        },
    }

    row = _row(
        player="Value Under",
        team="AAA",
        game_id="game_2",
        target="TB",
        prediction=0.35,
        line=1.5,
        edge=-1.15,
    )
    row["Market_Books"] = "4"
    row["Market_Line_Std"] = "0.1"
    row["Market_Over_Price"] = "130"
    row["Market_Under_Price"] = "-180"

    candidate = selector.build_candidate(
        row,
        calibration=calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert candidate is not None
    assert candidate.selected_side_price == -180.0
    assert candidate.market_books == 4
    assert candidate.market_implied_probability is not None
    assert candidate.expected_value_per_unit is not None


def test_select_top_candidates_respects_market_bucket_cap() -> None:
    calibration = {
        "target_direction": {
            "H|OVER": {"graded_rows": 8000, "win_rate": 0.526},
            "TB|UNDER": {"graded_rows": 2900, "win_rate": 0.888},
        },
        "line_buckets": {
            "H|OVER|0.5": {"graded_rows": 7000, "win_rate": 0.551},
            "TB|UNDER|1.5": {"graded_rows": 2500, "win_rate": 0.888},
        },
    }

    candidates = [
        selector.build_candidate(
            _row(
                player=f"Over Bat {idx}",
                team=f"T{idx}",
                game_id=f"g{idx}",
                target="H",
                prediction=1.55 - (idx * 0.03),
                line=0.5,
                edge=1.05 - (idx * 0.03),
            ),
            calibration=calibration,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        for idx in range(4)
    ] + [
        selector.build_candidate(
            _row(
                player=f"Under TB {idx}",
                team=f"U{idx}",
                game_id=f"u{idx}",
                target="TB",
                prediction=0.40 + (idx * 0.05),
                line=1.5,
                edge=-1.10 + (idx * 0.05),
            ),
            calibration=calibration,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        for idx in range(2)
    ]

    candidates = [candidate for candidate in candidates if candidate is not None]
    args = selector.argparse.Namespace(
        top_n=4,
        max_per_player=1,
        max_per_game=2,
        max_per_team=3,
        max_per_market_bucket=2,
    )

    selected = selector.select_top_candidates(candidates, args)
    bucket_counts = Counter(candidate.market_bucket for candidate in selected)

    assert len(selected) == 4
    assert bucket_counts["H|OVER|0.5"] == 2
    assert bucket_counts["TB|UNDER|1.5"] == 2


def test_lookup_historical_bet_profile_prior_prefers_line_probability_bucket() -> None:
    priors = {
        "bet_profiles_target_probability": {
            "TB|UNDER|0.90-0.95": {"rows": 30, "win_rate": 0.82, "roi_per_bet": 0.18},
        },
        "bet_profiles_line_probability": {
            "TB|UNDER|1.5|0.90-0.95": {"rows": 14, "win_rate": 0.93, "roi_per_bet": 0.31},
        },
    }

    key, win_rate, support, source, roi = selector.lookup_historical_bet_profile_prior(
        priors,
        target="TB",
        direction="UNDER",
        market_line=1.5,
        graded_hit_rate=0.93,
        min_line_rows=12,
    )

    assert key == "TB|UNDER|1.5|0.90-0.95"
    assert win_rate == 0.93
    assert support == 14
    assert source == "line_probability"
    assert roi == 0.31


def test_lookup_historical_market_availability_prior_falls_back_to_target_direction() -> None:
    priors = {
        "availability_target_direction": {
            "TB|UNDER": {"rows": 40, "availability_rate": 0.58, "avg_books": 6.2},
        },
        "availability_line_buckets": {
            "TB|UNDER|3.5": {"rows": 6, "availability_rate": 0.33, "avg_books": 7.0},
        },
    }

    key, rate, support, source, avg_books = selector.lookup_historical_market_availability_prior(
        priors,
        target="TB",
        direction="UNDER",
        market_line=3.5,
        min_line_rows=12,
    )

    assert key == "TB|UNDER"
    assert rate == 0.58
    assert support == 40
    assert source == "target_direction"
    assert avg_books == 6.2
