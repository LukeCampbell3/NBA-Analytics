from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import select_high_precision_predictions as selector


def test_supported_count_targets_cover_all_playable_count_props() -> None:
    assert selector.SUPPORTED_COUNT_TARGETS == {"H", "TB", "R", "HR", "RBI", "K", "ER"}
    assert set(selector.HISTORICAL_TARGET_SPECS) == selector.SUPPORTED_COUNT_TARGETS
    assert set(selector.HISTORICAL_BET_TARGET_SPECS) == selector.SUPPORTED_COUNT_TARGETS


def test_standard_bettable_lines_reject_alternate_hitter_ladders() -> None:
    assert selector.is_standard_bettable_line("TB", 1.5)
    assert selector.is_standard_bettable_line("R", 0.5)
    assert selector.is_standard_bettable_line("K", 5.5)
    assert not selector.is_standard_bettable_line("TB", 5.5)
    assert not selector.is_standard_bettable_line("R", 1.5)
    assert not selector.is_standard_bettable_line("K", 5.0)


def test_report_path_is_repository_relative() -> None:
    assert selector.report_path(selector.REPO_ROOT / "sports" / "mlb") == "sports/mlb"


def test_american_price_helpers_reject_invalid_consensus_prices() -> None:
    assert selector.american_implied_probability(-1.67) is None
    assert selector.american_implied_probability(-168.686524) is None
    assert selector.american_profit_per_unit(-0.5) is None
    assert selector.american_implied_probability(-110.0) is not None


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
        "Market_Books": "5",
        "Market_Book_Keys": "caesars|draftkings|fanduel|fanatics|mgm",
        "Market_Common_Books": "5",
        "Market_Common_Book_Keys": "fanduel|draftkings|mgm|caesars|fanatics",
        "Market_Over_Book_Key": "draftkings",
        "Market_Over_Book": "DraftKings",
        "Market_Under_Book_Key": "fanduel",
        "Market_Under_Book": "FanDuel",
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


def test_historical_priors_exclude_rows_on_or_after_cutoff(tmp_path: Path) -> None:
    player_dir = tmp_path / "Example_Player"
    player_dir.mkdir()
    pd.DataFrame(
        [
            {
                "Date": "2026-04-01",
                "H": 2,
                "Market_H": 0.5,
                "H_market_gap": 0.5,
                "Market_Source_H": "real",
                "Market_H_books": 5,
                "Market_H_over_price": -110,
                "Market_H_under_price": -110,
            },
            {
                "Date": "2026-05-01",
                "H": 0,
                "Market_H": 0.5,
                "H_market_gap": 0.5,
                "Market_Source_H": "real",
                "Market_H_books": 5,
                "Market_H_over_price": -110,
                "Market_H_under_price": -110,
            },
        ]
    ).to_csv(player_dir / "2026_processed_processed.csv", index=False)

    cutoff = date(2026, 4, 15)
    calibration = selector.build_historical_bucket_priors(tmp_path, 2026, cutoff)
    bet_profiles = selector.build_historical_bet_profile_priors(tmp_path, 2026, cutoff)

    assert calibration["history_before_date"] == "2026-04-15"
    assert calibration["line_buckets"]["H|OVER|0.5"]["graded_rows"] == 1
    assert calibration["line_buckets"]["H|OVER|0.5"]["win_rate"] == 1.0
    profile = bet_profiles["bet_profiles_line_probability"]["H|OVER|0.5|0.60-0.65"]
    assert profile["rows"] == 1
    assert profile["win_rate"] == 1.0


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


def test_build_candidate_applies_live_confidence_to_final_probability() -> None:
    row = _row(
        player="Calibrated Over",
        team="AAA",
        game_id="game_calibrated",
        target="H",
        prediction=1.2,
        line=0.5,
        edge=0.7,
    )
    live_calibration = {
        "segments": {
            "H|OVER": {"active": True, "graded_rows": 8, "adjustment": -0.04},
        }
    }

    candidate = selector.build_candidate(
        row,
        calibration=None,
        live_confidence_calibration=live_calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert candidate is not None
    assert candidate.live_confidence_calibration_key == "H|OVER"
    assert candidate.live_confidence_calibration_support == 8
    assert candidate.live_confidence_calibration_adjustment == -0.04
    assert candidate.calibrated_graded_hit_rate == candidate.model_graded_hit_rate - 0.04
    assert candidate.calibrated_hit_probability == candidate.calibrated_graded_hit_rate


def test_build_candidate_recent_form_prior_can_raise_short_term_over_score() -> None:
    base_calibration = {
        "target_direction": {
            "H|OVER": {"graded_rows": 8000, "win_rate": 0.49},
        },
        "line_buckets": {
            "H|OVER|0.5": {"graded_rows": 5000, "win_rate": 0.48},
        },
    }
    recent_calibration = {
        **base_calibration,
        "recent_target_direction": {
            "H|OVER": {"graded_rows": 42, "win_rate": 0.71},
        },
        "recent_line_buckets": {
            "H|OVER|0.5": {"graded_rows": 18, "win_rate": 0.74},
        },
    }

    row = _row(
        player="Recent Form Over",
        team="AAA",
        game_id="recent_1",
        target="H",
        prediction=1.0,
        line=0.5,
        edge=0.5,
    )

    baseline = selector.build_candidate(
        row,
        calibration=base_calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    recent = selector.build_candidate(
        row,
        calibration=recent_calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert baseline is not None
    assert recent is not None
    assert recent.calibrated_hit_probability > baseline.calibrated_hit_probability
    assert recent.selection_score > baseline.selection_score


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
    assert candidate.calibrated_hit_probability <= selector.MAX_CALIBRATED_PROBABILITY
    assert candidate.calibrated_graded_hit_rate <= selector.MAX_CALIBRATED_PROBABILITY
    assert candidate.selected_sportsbook == "FanDuel"


def test_build_candidate_does_not_confirm_aggregate_price_without_named_book() -> None:
    row = _row(
        player="Untraceable Price",
        team="AAA",
        game_id="game_untraceable",
        target="TB",
        prediction=0.35,
        line=1.5,
        edge=-1.15,
    )
    row["Market_Under_Price"] = "-180"
    row["Market_Under_Book_Key"] = ""
    row["Market_Under_Book"] = ""

    candidate = selector.build_candidate(
        row,
        calibration=None,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert candidate is not None
    assert candidate.price_confirmed is False


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


def test_select_top_candidates_suppresses_duplicate_prop_across_game_ids() -> None:
    calibration = {
        "target_direction": {
            "TB|UNDER": {"graded_rows": 2900, "win_rate": 0.888},
        },
        "line_buckets": {
            "TB|UNDER|1.0": {"graded_rows": 1800, "win_rate": 0.86},
        },
    }
    rows = [
        _row(
            player="Duplicate Player",
            team="ATL",
            game_id="resumed_game",
            target="TB",
            prediction=0.25,
            line=1.0,
            edge=-0.75,
        ),
        _row(
            player="Duplicate Player",
            team="ATL",
            game_id="nightcap",
            target="TB",
            prediction=0.30,
            line=1.0,
            edge=-0.70,
        ),
    ]
    candidates = [
        selector.build_candidate(
            row,
            calibration=calibration,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        for row in rows
    ]
    args = selector.argparse.Namespace(
        top_n=2,
        max_per_player=2,
        max_per_game=2,
        max_per_team=3,
        max_per_market_bucket=4,
    )

    selected = selector.select_top_candidates(
        [candidate for candidate in candidates if candidate is not None],
        args,
    )

    assert len(selected) == 1
    assert selected[0].player == "Duplicate Player"


def test_select_top_candidates_soft_cap_allows_only_elite_expansion() -> None:
    candidates = []
    for idx, score in enumerate([0.95, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.79]):
        candidate = selector.build_candidate(
            _row(
                player=f"Volume Player {idx}",
                team=f"V{idx}",
                game_id=f"volume_{idx}",
                target="TB",
                prediction=0.20,
                line=1.5,
                edge=-1.30,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidate.selection_score = score
        candidates.append(candidate)

    args = selector.argparse.Namespace(
        top_n=10,
        daily_pick_soft_cap=6,
        post_cap_min_selection_score=0.80,
        min_over_picks=0,
        max_over_picks=0,
        max_per_player=1,
        max_per_game=2,
        max_per_team=3,
        max_per_market_bucket=10,
    )

    selected = selector.select_top_candidates(candidates, args)

    assert len(selected) == 7
    assert [candidate.selection_score for candidate in selected] == [
        0.95,
        0.92,
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
    ]


def test_survival_model_only_breaks_ties_inside_selection_score_band() -> None:
    candidates = []
    for idx, (score, survival_probability) in enumerate(
        ((0.704, 0.55), (0.701, 0.70), (0.714, 0.40))
    ):
        candidate = selector.build_candidate(
            _row(
                player=f"Tie Player {idx}",
                team=f"T{idx}",
                game_id=f"tie_{idx}",
                target="TB",
                prediction=1.9,
                line=1.5,
                edge=0.4,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidate.selection_score = score
        candidate.survival_probability = survival_probability
        candidate.survival_rank_active = True
        candidates.append(candidate)

    args = selector.argparse.Namespace(
        top_n=2,
        max_per_player=1,
        max_per_game=2,
        max_per_team=3,
        max_per_market_bucket=4,
    )

    selected = selector.select_top_candidates(candidates, args)

    assert [candidate.player for candidate in selected] == ["Tie Player 2", "Tie Player 1"]


def test_filter_candidates_uses_validated_over_profile_instead_of_general_probability_floor() -> None:
    row = _row(
        player="Moderate Over",
        team="ATL",
        game_id="over_profile",
        target="R",
        prediction=0.75,
        line=0.5,
        edge=0.25,
    )
    row.update(
        {
            "History_Rows": "60",
            "Market_Over_Price": "110",
            "Market_Under_Price": "-130",
        }
    )
    candidate = selector.build_candidate(
        row,
        calibration=None,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    assert candidate is not None
    assert 0.45 <= candidate.model_hit_probability <= 0.55
    assert candidate.expected_value_per_unit is not None
    assert candidate.expected_value_per_unit >= 0.10

    args = selector.argparse.Namespace(
        targets=["R"],
        optimized_over_targets=["R", "TB"],
        over_min_abs_edge=0.15,
        over_max_abs_edge=0.35,
        over_min_model_hit_probability=0.45,
        over_max_model_hit_probability=0.55,
        over_min_expected_value=0.10,
        over_max_american_price=125,
        over_min_history_rows=55,
        core_max_american_price=-200,
        allow_baseline=False,
        require_real_market_source=True,
        allow_synthetic_unders=False,
        min_abs_edge=0.35,
        min_history_rows=35,
        min_prediction=0.10,
        min_market_books=5,
        min_common_market_books=2,
        max_market_line_std=0.0,
        min_hit_probability=0.60,
        min_graded_hit_rate=0.75,
        min_expected_value=0.0,
        allow_unpriced_side=False,
        max_push_probability=0.10,
        max_days_since_history=4,
        min_historical_bet_profile_support=0,
        min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=0,
        min_historical_market_availability_rate=0.0,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == [candidate]
    assert rejected == Counter()
    assert selector.candidate_selection_profile(candidate, args) == selector.OPTIMIZED_OVER_SELECTION_PROFILE

    candidate.selected_side_price = 130.0
    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["optimized_over_price_too_long"] == 1

    candidate.selected_side_price = 110.0
    candidate.history_rows = 54
    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["optimized_over_history_too_short"] == 1


def test_filter_candidates_uses_probable_starter_pitcher_k_profile() -> None:
    row = _row(
        player="Workload Starter",
        team="SEA",
        game_id="pitcher_k_profile",
        target="K",
        prediction=5.1,
        line=4.5,
        edge=0.6,
    )
    row.update(
        {
            "Player_Type": "pitcher",
            "History_Rows": "20",
            "Starter_Confirmed": "1",
            "Starter_History_Rows": "18",
            "Projected_IP": "5.6",
            "Projected_Pitches": "88",
            "Last_History_Date": "2026-07-25",
            "Market_Over_Price": "110",
            "Market_Under_Price": "-130",
        }
    )
    candidate = selector.build_candidate(
        row,
        calibration=None,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    assert candidate is not None
    args = selector.argparse.Namespace(
        targets=["K"], optimized_over_targets=["R", "TB"], enable_pitcher_k_over_profile=True,
        pitcher_k_min_starter_history=15, pitcher_k_min_projected_ip=5.25,
        pitcher_k_min_projected_pitches=75.0, pitcher_k_max_days_since_history=14,
        pitcher_k_min_abs_edge=0.15,
        pitcher_k_max_abs_edge=1.0, pitcher_k_min_model_hit_probability=0.50,
        pitcher_k_max_model_hit_probability=0.65, pitcher_k_min_expected_value=0.0,
        pitcher_k_min_american_price=-130, pitcher_k_max_american_price=130,
        allow_baseline=False, require_real_market_source=True, allow_synthetic_unders=False,
        min_abs_edge=0.35, min_history_rows=35, min_prediction=0.10,
        min_market_books=5, min_common_market_books=2, max_market_line_std=0.0,
        min_hit_probability=0.60, min_graded_hit_rate=0.75, min_expected_value=0.0,
        allow_unpriced_side=False, max_push_probability=0.10, max_days_since_history=4,
        core_min_american_price=-250, core_max_american_price=-200,
        min_historical_bet_profile_support=0, min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=0, min_historical_market_availability_rate=0.0,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == [candidate]
    assert rejected == Counter()
    assert selector.candidate_selection_profile(candidate, args) == selector.PITCHER_K_OVER_SELECTION_PROFILE

    candidate.raw["Starter_Confirmed"] = "0"
    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["pitcher_starter_unconfirmed"] == 1

    candidate.raw["Starter_Confirmed"] = "1"
    candidate.days_since_history = 15
    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["history_too_stale"] == 1


def test_core_price_defense_rejects_expensive_non_optimized_pick() -> None:
    row = _row(
        player="Core Price Test",
        team="SEA",
        game_id="core_price",
        target="TB",
        prediction=0.2,
        line=1.5,
        edge=-1.3,
    )
    row.update({"History_Rows": "70", "Market_Under_Price": "-175", "Market_Over_Price": "145"})
    candidate = selector.build_candidate(
        row,
        calibration={"target_direction": {"TB|UNDER": {"graded_rows": 1000, "win_rate": 0.9}}},
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    assert candidate is not None
    args = selector.argparse.Namespace(
        targets=["TB"], optimized_over_targets=["R", "TB"], allow_baseline=False,
        require_real_market_source=True, allow_synthetic_unders=False, min_abs_edge=0.35,
        min_history_rows=35, min_prediction=0.0, min_market_books=2, min_common_market_books=1,
        max_market_line_std=0.0, min_hit_probability=0.0, min_graded_hit_rate=0.0,
        min_expected_value=-1.0, allow_unpriced_side=False, max_push_probability=1.0,
        max_days_since_history=4, core_max_american_price=-200,
        min_historical_bet_profile_support=0, min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=0, min_historical_market_availability_rate=0.0,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["core_price_too_long"] == 1


def test_core_price_defense_rejects_heavily_juiced_non_optimized_pick() -> None:
    row = _row(
        player="Core Juice Test",
        team="SEA",
        game_id="core_juice",
        target="TB",
        prediction=0.2,
        line=1.5,
        edge=-1.3,
    )
    row.update({"History_Rows": "70", "Market_Under_Price": "-300", "Market_Over_Price": "220"})
    candidate = selector.build_candidate(
        row,
        calibration={"target_direction": {"TB|UNDER": {"graded_rows": 1000, "win_rate": 0.9}}},
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    assert candidate is not None
    args = selector.argparse.Namespace(
        targets=["TB"], optimized_over_targets=["R", "TB"], allow_baseline=False,
        require_real_market_source=True, allow_synthetic_unders=False, min_abs_edge=0.35,
        min_history_rows=35, min_prediction=0.0, min_market_books=2, min_common_market_books=1,
        max_market_line_std=0.0, min_hit_probability=0.0, min_graded_hit_rate=0.0,
        min_expected_value=-1.0, allow_unpriced_side=False, max_push_probability=1.0,
        max_days_since_history=4, core_min_american_price=-250, core_max_american_price=-200,
        min_historical_bet_profile_support=0, min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=0, min_historical_market_availability_rate=0.0,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["core_price_too_heavily_juiced"] == 1


def test_select_top_candidates_reserves_and_caps_over_positions() -> None:
    candidates = []
    for idx in range(3):
        candidate = selector.build_candidate(
            _row(
                player=f"Reserved Over {idx}",
                team=f"O{idx}",
                game_id=f"over_{idx}",
                target="R",
                prediction=0.75,
                line=0.5,
                edge=0.25,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidates.append(candidate)
    for idx in range(4):
        candidate = selector.build_candidate(
            _row(
                player=f"Higher Ranked Under {idx}",
                team=f"U{idx}",
                game_id=f"under_{idx}",
                target="TB",
                prediction=0.20,
                line=1.5,
                edge=-1.30,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidates.append(candidate)

    selected = selector.select_top_candidates(
        candidates,
        selector.argparse.Namespace(
            top_n=4,
            min_over_picks=2,
            max_over_picks=2,
            optimized_over_targets=["R", "TB"],
            max_per_player=1,
            max_per_game=2,
            max_per_team=3,
            max_per_market_bucket=4,
        ),
    )

    assert len(selected) == 4
    assert Counter(candidate.direction for candidate in selected) == {"OVER": 2, "UNDER": 2}


def test_select_top_candidates_caps_under_fallback_positions() -> None:
    candidates = []
    for idx in range(3):
        candidate = selector.build_candidate(
            _row(
                player=f"Reserved Over {idx}",
                team=f"O{idx}",
                game_id=f"over_fallback_{idx}",
                target="R",
                prediction=0.75,
                line=0.5,
                edge=0.25,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidate.selection_score = 0.70 - (idx * 0.01)
        candidates.append(candidate)
    for idx in range(4):
        candidate = selector.build_candidate(
            _row(
                player=f"Higher Ranked Under {idx}",
                team=f"U{idx}",
                game_id=f"under_fallback_{idx}",
                target="TB",
                prediction=0.20,
                line=1.5,
                edge=-1.30,
            ),
            calibration=None,
            min_history_bucket_rows=50,
            max_history_prior_weight=0.35,
            history_prior_strength=400.0,
        )
        assert candidate is not None
        candidate.selection_score = 0.95 - (idx * 0.01)
        candidates.append(candidate)

    selected = selector.select_top_candidates(
        candidates,
        selector.argparse.Namespace(
            top_n=4,
            min_over_picks=3,
            max_over_picks=3,
            max_under_picks=1,
            optimized_over_targets=["R", "TB"],
            optimized_over_max_per_market_bucket=3,
            max_per_player=1,
            max_per_game=2,
            max_per_team=3,
            max_per_market_bucket=4,
        ),
    )

    assert len(selected) == 4
    assert Counter(candidate.direction for candidate in selected) == {"OVER": 3, "UNDER": 1}


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


def test_build_candidate_prefer_confident_side_keeps_model_side_when_flip_conflicts_with_model() -> None:
    calibration = {
        "target_direction": {
            "H|OVER": {"graded_rows": 800, "win_rate": 0.46},
            "H|UNDER": {"graded_rows": 800, "win_rate": 0.79},
        },
        "line_buckets": {
            "H|OVER|0.5": {"graded_rows": 500, "win_rate": 0.44},
            "H|UNDER|0.5": {"graded_rows": 500, "win_rate": 0.81},
        },
    }

    candidate = selector.build_candidate(
        _row(
            player="Strong Over",
            team="AAA",
            game_id="flip_1",
            target="H",
            prediction=1.7,
            line=0.5,
            edge=1.2,
        ),
        calibration=calibration,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
        prefer_confident_side=True,
    )

    assert candidate is not None
    assert candidate.direction == "OVER"
    assert candidate.original_direction == "OVER"
    assert candidate.direction_flip_applied is False


def test_filter_candidates_can_require_historical_market_availability() -> None:
    calibration = {
        "target_direction": {"TB|OVER": {"graded_rows": 1000, "win_rate": 0.63}},
        "line_buckets": {"TB|OVER|1.5": {"graded_rows": 700, "win_rate": 0.66}},
    }
    priors = {
        "availability_target_direction": {
            "TB|OVER": {"rows": 40, "availability_rate": 0.30, "avg_books": 2.8},
        },
        "availability_line_buckets": {},
        "bet_profiles_target_probability": {
            "TB|OVER|0.75-0.80": {"rows": 20, "win_rate": 0.61, "roi_per_bet": 0.08},
        },
        "bet_profiles_line_probability": {},
    }

    candidate = selector.build_candidate(
        _row(
            player="Availability Test",
            team="AAA",
            game_id="avail_1",
            target="TB",
                prediction=3.0,
                line=1.5,
            edge=1.5,
        ),
        calibration=calibration,
        bet_profile_priors=priors,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )

    assert candidate is not None
    args = selector.argparse.Namespace(
        targets=["TB"],
        allow_baseline=False,
        require_real_market_source=False,
        allow_synthetic_unders=False,
        min_abs_edge=0.45,
        min_history_rows=11,
        min_prediction=0.0,
        min_market_books=0,
        max_market_line_std=0.0,
        min_hit_probability=0.58,
        min_graded_hit_rate=0.68,
        min_expected_value=-1.0,
        allow_unpriced_side=True,
        max_push_probability=0.24,
        max_days_since_history=4,
        min_historical_bet_profile_support=0,
        min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=20,
        min_historical_market_availability_rate=0.45,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["historical_market_availability_rate_too_low"] == 1


def test_filter_candidates_rejects_zero_role_projection() -> None:
    row = _row(
        player="Role Risk",
        team="AAA",
        game_id="role_1",
        target="TB",
        prediction=0.0,
        line=1.5,
        edge=-1.5,
    )
    candidate = selector.build_candidate(
        row,
        calibration=None,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
    )
    assert candidate is not None
    args = selector.argparse.Namespace(
        targets=["TB"],
        allow_baseline=False,
        require_real_market_source=True,
        allow_synthetic_unders=False,
        min_abs_edge=0.45,
        min_history_rows=30,
        min_prediction=0.05,
        min_market_books=0,
        max_market_line_std=0.0,
        min_hit_probability=0.0,
        min_graded_hit_rate=0.0,
        min_expected_value=-1.0,
        allow_unpriced_side=True,
        max_push_probability=1.0,
        max_days_since_history=4,
        min_historical_bet_profile_support=0,
        min_historical_bet_profile_win_rate=0.0,
        min_historical_market_availability_support=0,
        min_historical_market_availability_rate=0.0,
    )

    kept, rejected = selector.filter_candidates([candidate], args)

    assert kept == []
    assert rejected["prediction_too_low"] == 1
