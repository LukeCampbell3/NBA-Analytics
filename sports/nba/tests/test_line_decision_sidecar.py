from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from decision_engine.line_decision import LineDecisionConfig, build_line_decision_lookup, estimate_line_decision
from post_process_market_plays import compute_final_board
from select_market_plays import _supply_dependent_context, build_history_lookup, build_play_rows
from validate_rebound_diagnostics import _compare_to_baseline


def _synthetic_history() -> pd.DataFrame:
    rows: list[dict] = []
    residual_pattern = [1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, -0.1, 0.4, 0.7]
    for idx in range(80):
        pred_pts = 21.0 + 0.15 * (idx % 4)
        market_pts = 19.5 + float(idx % 2)
        residual = residual_pattern[idx % len(residual_pattern)]
        actual_pts = pred_pts + residual
        rows.append(
            {
                "player": f"Hist Player {idx}",
                "market_date": f"2026-03-{(idx % 28) + 1:02d}",
                "pred_PTS": pred_pts,
                "market_PTS": market_pts,
                "actual_PTS": actual_pts,
                "pred_TRB": None,
                "market_TRB": None,
                "actual_TRB": None,
                "pred_AST": None,
                "market_AST": None,
                "actual_AST": None,
                "did_not_play": 0,
                "minutes": 28.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_slate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Strong Sidecar",
                "market_date": "2026-04-01",
                "market_player_raw": "Strong Sidecar",
                "market_event_id": "game_a",
                "market_commence_time_utc": "2026-04-01T23:00:00Z",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "history_rows": 120,
                "last_history_date": "2026-03-31",
                "csv": "strong.csv",
                "belief_uncertainty": 0.42,
                "feasibility": 0.93,
                "fallback_blend": 0.0,
                "pred_PTS": 22.6,
                "baseline_PTS": 21.0,
                "market_PTS": 19.5,
                "baseline_edge_PTS": 1.5,
                "PTS_uncertainty_sigma": 0.75,
                "PTS_spike_probability": 0.32,
                "market_books_PTS": 6,
                "pred_TRB": None,
                "baseline_TRB": None,
                "market_TRB": None,
                "baseline_edge_TRB": None,
                "TRB_uncertainty_sigma": None,
                "TRB_spike_probability": None,
                "market_books_TRB": None,
                "pred_AST": None,
                "baseline_AST": None,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
            },
            {
                "player": "Fragile Sidecar",
                "market_date": "2026-04-01",
                "market_player_raw": "Fragile Sidecar",
                "market_event_id": "game_b",
                "market_commence_time_utc": "2026-04-01T23:30:00Z",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "history_rows": 18,
                "last_history_date": "2026-03-31",
                "csv": "fragile.csv",
                "belief_uncertainty": 0.86,
                "feasibility": 0.63,
                "fallback_blend": 0.35,
                "pred_PTS": 19.9,
                "baseline_PTS": 19.7,
                "market_PTS": 19.5,
                "baseline_edge_PTS": 0.2,
                "PTS_uncertainty_sigma": 4.2,
                "PTS_spike_probability": 0.58,
                "market_books_PTS": 2,
                "pred_TRB": None,
                "baseline_TRB": None,
                "market_TRB": None,
                "baseline_edge_TRB": None,
                "TRB_uncertainty_sigma": None,
                "TRB_spike_probability": None,
                "market_books_TRB": None,
                "pred_AST": None,
                "baseline_AST": None,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
            },
        ]
    )


def _board_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Keep Me",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 24.0,
                "market_line": 21.5,
                "abs_edge": 2.5,
                "edge": 2.5,
                "expected_win_rate": 0.68,
                "expected_push_rate": 0.05,
                "posterior_variance": 0.03,
                "belief_confidence_factor": 0.88,
                "feasibility": 0.91,
                "recommendation": "strong",
                "history_rows": 140,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_keep",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "risk_penalty": 0.20,
                "market_books": 6,
                "gap_percentile": 0.95,
                "confidence_score": 0.20,
                "line_decision_trade_eligible": True,
                "line_decision_action": "OVER",
            },
            {
                "player": "Drop Me",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 20.0,
                "market_line": 19.5,
                "abs_edge": 0.5,
                "edge": 0.5,
                "expected_win_rate": 0.56,
                "expected_push_rate": 0.38,
                "posterior_variance": 0.05,
                "belief_confidence_factor": 0.62,
                "feasibility": 0.66,
                "recommendation": "pass",
                "history_rows": 18,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_drop",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "risk_penalty": 0.41,
                "market_books": 2,
                "gap_percentile": 0.58,
                "confidence_score": 0.04,
                "line_decision_trade_eligible": False,
                "line_decision_action": "NO_TRADE",
            },
        ]
    )


def _opposite_flip_history() -> pd.DataFrame:
    rows: list[dict] = []
    for idx in range(80):
        if idx < 56:
            pred_pts = 20.3 + 0.05 * (idx % 3)
            market_pts = 19.5
            actual_pts = 18.6 + 0.05 * (idx % 2)
        else:
            pred_pts = 18.8 - 0.05 * (idx % 3)
            market_pts = 19.5
            actual_pts = 18.7 + 0.05 * (idx % 2)
        rows.append(
            {
                "player": f"Flip Hist {idx}",
                "market_date": f"2026-03-{(idx % 28) + 1:02d}",
                "pred_PTS": pred_pts,
                "market_PTS": market_pts,
                "actual_PTS": actual_pts,
                "pred_TRB": None,
                "market_TRB": None,
                "actual_TRB": None,
                "pred_AST": None,
                "market_AST": None,
                "actual_AST": None,
                "did_not_play": 0,
                "minutes": 30.0,
            }
        )
    return pd.DataFrame(rows)


def _base_rebound_slate_row() -> dict:
    return {
        "player": "Rebound Candidate",
        "market_date": "2026-04-01",
        "market_player_raw": "Rebound Candidate",
        "market_event_id": "game_trb",
        "market_commence_time_utc": "2026-04-01T23:00:00Z",
        "market_home_team": "NYK",
        "market_away_team": "CLE",
        "history_rows": 88,
        "last_history_date": "2026-03-31",
        "csv": "rebound.csv",
        "belief_uncertainty": 0.58,
        "feasibility": 0.91,
        "fallback_blend": 0.0,
        "pred_PTS": None,
        "baseline_PTS": None,
        "market_PTS": None,
        "baseline_edge_PTS": None,
        "PTS_uncertainty_sigma": None,
        "PTS_spike_probability": None,
        "market_books_PTS": None,
        "pred_TRB": 13.2,
        "baseline_TRB": 10.9,
        "market_TRB": 11.5,
        "baseline_edge_TRB": -0.6,
        "TRB_uncertainty_sigma": 1.2,
        "TRB_spike_probability": 0.42,
        "market_books_TRB": 6,
        "market_over_price_TRB": -112.0,
        "market_under_price_TRB": -108.0,
        "pred_AST": 6.8,
        "baseline_AST": 4.6,
        "market_AST": None,
        "baseline_edge_AST": None,
        "AST_uncertainty_sigma": None,
        "AST_spike_probability": None,
        "market_books_AST": None,
        "rebound_supply_score": 0.18,
        "rebound_share_stability": 0.34,
        "rebound_share_stability_score": 0.34,
        "rebound_share_estimate": 0.23,
        "player_team_rebound_share_recent": 0.23,
        "player_rebound_share_std": 0.09,
        "team_shooting_efficiency_stress": 0.74,
        "opponent_shooting_efficiency_stress": 0.70,
        "wing_rebound_leakage_score": 0.72,
        "teammate_rebound_competition": 0.68,
        "teammate_rebound_competition_score": 0.68,
        "center_rebound_share_pressure": 0.62,
        "frontcourt_rebound_overlap_score": 0.66,
        "projected_team_missed_fga": 36.5,
        "projected_opponent_missed_fga": 37.2,
        "projected_team_missed_fta": 4.0,
        "projected_opponent_missed_fta": 4.3,
        "projected_missed_fga_total": 73.7,
        "projected_missed_fta_total": 8.3,
        "projected_available_rebound_events": 78.1,
        "expected_rebound_chances": 78.1,
        "team_rebound_pool_size": 52.4,
        "pace_rebound_environment": 0.28,
        "long_rebound_profile": 0.69,
        "free_throw_rebound_suppression": 0.11,
        "projected_team_fg_pct": 0.528,
        "projected_opponent_fg_pct": 0.522,
        "recent_games_count": 10,
        "trb_median_recent": 8.9,
        "trb_q75_recent": 10.7,
        "trb_q90_recent": 12.3,
        "minutes_floor_recent": 24.0,
        "minutes_p25_recent": 29.0,
        "minutes_median_recent": 33.0,
        "minutes_range_recent": 10.0,
        "expected_minutes_band_low": 29.0,
        "expected_minutes_band_high": 35.0,
        "expected_minutes_band_width": 6.0,
        "bench_role_flag": False,
        "starter_status_recent": 1.0,
        "starter_status_change_count": 0,
        "rotation_volatility_score": 0.28,
        "blowout_minutes_sensitivity": 0.22,
        "foul_rate_minutes_loss_risk": 0.18,
        "coach_trust_score": 0.74,
    }


def test_line_decision_prefers_trade_for_strong_edge() -> None:
    lookup = build_line_decision_lookup(_synthetic_history())
    decision = estimate_line_decision(
        lookup=lookup,
        target="PTS",
        prediction=22.6,
        market_line=19.5,
        direction="OVER",
        gap_percentile=0.96,
        uncertainty_sigma=0.75,
        belief_confidence_factor=0.90,
        feasibility=0.93,
        history_rows=120,
        market_books=6,
        fallback_blend=0.0,
        prior_direction_win_rate=0.66,
        prior_neutral_rate=0.03,
        config=LineDecisionConfig(),
    )
    assert decision["action"] == "OVER"
    assert decision["trade_eligible"] is True
    assert decision["chosen_direction_prob"] > decision["opposite_direction_prob"]
    assert decision["chosen_direction_conditional_prob"] >= 0.57
    assert decision["no_trade_prob"] < 0.36


def test_line_decision_gate_uses_conditional_trade_confidence() -> None:
    decision = estimate_line_decision(
        lookup={},
        target="PTS",
        prediction=20.2,
        market_line=19.5,
        direction="OVER",
        gap_percentile=0.80,
        uncertainty_sigma=0.0,
        belief_confidence_factor=1.0,
        feasibility=1.0,
        history_rows=100,
        market_books=5,
        fallback_blend=0.0,
        prior_direction_win_rate=0.55,
        prior_neutral_rate=0.20,
        config=LineDecisionConfig(no_trade_threshold=0.45, min_trade_prob=0.57, min_trade_prob_gap=0.06),
    )
    assert decision["chosen_direction_prob"] < 0.57
    assert decision["chosen_direction_conditional_prob"] > 0.57
    assert decision["trade_eligible"] is True


def test_line_decision_can_flip_to_opposite_side() -> None:
    lookup = build_line_decision_lookup(_opposite_flip_history())
    decision = estimate_line_decision(
        lookup=lookup,
        target="PTS",
        prediction=20.35,
        market_line=19.5,
        direction="OVER",
        gap_percentile=0.82,
        uncertainty_sigma=0.55,
        belief_confidence_factor=0.88,
        feasibility=0.92,
        history_rows=120,
        market_books=5,
        fallback_blend=0.0,
        prior_direction_win_rate=0.56,
        prior_neutral_rate=0.02,
        config=LineDecisionConfig(no_trade_threshold=0.45, min_trade_prob=0.57, min_trade_prob_gap=0.06),
    )

    assert decision["trade_eligible"] is True
    assert decision["action"] == "UNDER"
    assert decision["action_is_opposite"] is True
    assert decision["under_prob"] > decision["over_prob"]
    assert decision["preferred_direction_conditional_prob"] >= 0.57


def test_build_play_rows_marks_fragile_near_line_case_as_no_trade() -> None:
    history = _synthetic_history()
    slate = _synthetic_slate()
    plays = build_play_rows(
        slate,
        build_history_lookup(history),
        line_decision_lookup=build_line_decision_lookup(history),
        line_decision_enabled=True,
        line_decision_config=LineDecisionConfig(),
    )
    assert {"line_decision_action", "line_no_trade_prob", "line_decision_trade_eligible"}.issubset(set(plays.columns))
    by_player = plays.set_index("player")
    assert by_player.loc["Strong Sidecar", "line_decision_action"] == "OVER"
    assert bool(by_player.loc["Strong Sidecar", "line_decision_trade_eligible"]) is True
    assert by_player.loc["Strong Sidecar", "expected_push_rate"] == by_player.loc["Strong Sidecar", "historical_push_rate"]
    assert by_player.loc["Strong Sidecar", "recommendation"] in {"consider", "strong", "elite"}
    assert by_player.loc["Fragile Sidecar", "line_decision_action"] == "NO_TRADE"
    assert bool(by_player.loc["Fragile Sidecar", "line_decision_trade_eligible"]) is False
    assert by_player.loc["Fragile Sidecar", "recommendation"] == "pass"
    assert by_player.loc["Fragile Sidecar", "line_no_trade_prob"] > 0.36


def test_build_play_rows_downgrades_upper_band_rebound_over_when_supply_is_weak() -> None:
    slate = pd.DataFrame(
        [
            {
                "player": "Supply Fragile Rebound Over",
                "market_date": "2026-04-01",
                "market_player_raw": "Supply Fragile Rebound Over",
                "market_event_id": "game_trb",
                "market_commence_time_utc": "2026-04-01T23:00:00Z",
                "market_home_team": "NYK",
                "market_away_team": "CLE",
                "history_rows": 88,
                "last_history_date": "2026-03-31",
                "csv": "fragile_rebound.csv",
                "belief_uncertainty": 0.58,
                "feasibility": 0.91,
                "fallback_blend": 0.0,
                "pred_PTS": None,
                "baseline_PTS": None,
                "market_PTS": None,
                "baseline_edge_PTS": None,
                "PTS_uncertainty_sigma": None,
                "PTS_spike_probability": None,
                "market_books_PTS": None,
                "pred_TRB": 13.2,
                "baseline_TRB": 10.9,
                "market_TRB": 11.5,
                "baseline_edge_TRB": -0.6,
                "TRB_uncertainty_sigma": 1.2,
                "TRB_spike_probability": 0.42,
                "market_books_TRB": 6,
                "pred_AST": 6.8,
                "baseline_AST": 4.6,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
                "rebound_supply_score": 0.18,
                "rebound_share_stability": 0.34,
                "rebound_share_estimate": 0.23,
                "team_shooting_efficiency_stress": 0.74,
                "opponent_shooting_efficiency_stress": 0.70,
                "wing_rebound_leakage_score": 0.72,
                "teammate_rebound_competition": 0.68,
                    "projected_team_missed_fga": 36.5,
                    "projected_opponent_missed_fga": 37.2,
                    "projected_available_rebound_events": 78.1,
                    "recent_games_count": 9,
                    "trb_median_recent": 8.9,
                    "trb_q75_recent": 10.7,
                    "trb_q90_recent": 12.3,
                }
        ]
    )

    plays = build_play_rows(
        slate,
        build_history_lookup(_synthetic_history()),
        line_decision_lookup={},
        line_decision_enabled=False,
    )

    assert len(plays) == 1
    row = plays.iloc[0]
    assert row["target"] == "TRB"
    assert row["raw_recommendation"] == "strong"
    assert row["recommendation"] == "consider"
    assert bool(row["supply_dependency_active"]) is True
    assert float(row["supply_dependency_score"]) > 0.0
    assert float(row["upper_band_line_penalty"]) > 0.0
    assert float(row["total_rebound_penalty"]) > 0.0
    assert "TRB_OVER_UPPER_BAND" in str(row["trb_over_bucket"])
    assert float(row["expected_win_rate_pre_sidecar"]) < float(row["bayesian_expected_win_rate"])
    assert float(row["risk_penalty"]) > 0.25


def test_upper_band_diagnostics_trigger_only_above_recent_upper_band() -> None:
    base_row = _base_rebound_slate_row()
    near_band = _supply_dependent_context(pd.Series(base_row), "TRB", "OVER", 10.95)
    above_band = _supply_dependent_context(pd.Series(base_row), "TRB", "OVER", 11.6)

    assert bool(near_band["upper_band_line_flag"]) is False
    assert float(near_band["upper_band_line_penalty"]) == 0.0
    assert bool(above_band["upper_band_line_flag"]) is True
    assert float(above_band["upper_band_line_penalty"]) > 0.0
    assert "TRB_OVER_UPPER_BAND" in str(above_band["trb_over_bucket"])


def test_low_line_role_volatility_triggers_on_unstable_minutes_despite_low_line() -> None:
    unstable_row = _base_rebound_slate_row()
    unstable_row.update(
        {
            "market_TRB": 4.5,
            "pred_TRB": 5.4,
            "trb_median_recent": 5.0,
            "trb_q75_recent": 6.0,
            "trb_q90_recent": 7.0,
            "minutes_floor_recent": 12.0,
            "minutes_p25_recent": 15.0,
            "minutes_median_recent": 20.0,
            "minutes_range_recent": 18.0,
            "expected_minutes_band_low": 14.0,
            "expected_minutes_band_high": 26.0,
            "expected_minutes_band_width": 12.0,
            "bench_role_flag": True,
            "starter_status_recent": 0.2,
            "starter_status_change_count": 3,
            "rotation_volatility_score": 0.81,
            "coach_trust_score": 0.31,
        }
    )
    context = _supply_dependent_context(pd.Series(unstable_row), "TRB", "OVER", 4.5)

    assert bool(context["upper_band_line_flag"]) is False
    assert bool(context["low_line_role_volatility_flag"]) is True
    assert float(context["low_line_role_volatility_penalty"]) > 0.0
    assert "TRB_OVER_LOW_LINE_ROLE_VOLATILE" in str(context["trb_over_bucket"])


def test_rebound_supply_penalty_increases_when_projected_missed_shot_pool_is_low() -> None:
    weak_env = _base_rebound_slate_row()
    weak_env.update(
        {
            "projected_missed_fga_total": 70.0,
            "projected_team_fg_pct": 0.531,
            "projected_opponent_fg_pct": 0.526,
            "pace_rebound_environment": 0.22,
        }
    )
    supportive_env = _base_rebound_slate_row()
    supportive_env.update(
        {
            "projected_missed_fga_total": 90.0,
            "projected_team_fg_pct": 0.478,
            "projected_opponent_fg_pct": 0.482,
            "pace_rebound_environment": 0.71,
        }
    )
    weak_context = _supply_dependent_context(pd.Series(weak_env), "TRB", "OVER", 11.5)
    supportive_context = _supply_dependent_context(pd.Series(supportive_env), "TRB", "OVER", 11.5)

    assert float(weak_context["rebound_supply_penalty"]) > float(supportive_context["rebound_supply_penalty"])
    assert "TRB_OVER_SUPPLY_DEPENDENT" in str(weak_context["trb_over_bucket"])


def test_rebound_share_penalty_increases_when_teammate_competition_is_high() -> None:
    high_comp = _base_rebound_slate_row()
    high_comp.update(
        {
            "teammate_rebound_competition": 0.79,
            "teammate_rebound_competition_score": 0.79,
            "player_rebound_share_std": 0.11,
            "wing_rebound_leakage_score": 0.74,
            "frontcourt_rebound_overlap_score": 0.72,
        }
    )
    stable_share = _base_rebound_slate_row()
    stable_share.update(
        {
            "teammate_rebound_competition": 0.52,
            "teammate_rebound_competition_score": 0.52,
            "player_rebound_share_std": 0.03,
            "wing_rebound_leakage_score": 0.42,
            "frontcourt_rebound_overlap_score": 0.48,
        }
    )
    high_context = _supply_dependent_context(pd.Series(high_comp), "TRB", "OVER", 11.5)
    stable_context = _supply_dependent_context(pd.Series(stable_share), "TRB", "OVER", 11.5)

    assert float(high_context["rebound_share_competition_penalty"]) > float(stable_context["rebound_share_competition_penalty"])
    assert "TRB_OVER_SHARE_COMPETITION" in str(high_context["trb_over_bucket"])


def test_diagnostics_do_not_fire_on_stable_trb_over_cases() -> None:
    stable_row = _base_rebound_slate_row()
    stable_row.update(
        {
            "market_TRB": 9.5,
            "pred_TRB": 10.6,
            "trb_median_recent": 10.0,
            "trb_q75_recent": 11.0,
            "trb_q90_recent": 12.0,
            "rebound_supply_score": 0.72,
            "rebound_share_stability": 0.76,
            "rebound_share_stability_score": 0.76,
            "teammate_rebound_competition": 0.44,
            "teammate_rebound_competition_score": 0.44,
            "player_rebound_share_std": 0.03,
            "wing_rebound_leakage_score": 0.40,
            "center_rebound_share_pressure": 0.32,
            "frontcourt_rebound_overlap_score": 0.38,
            "projected_missed_fga_total": 86.0,
            "projected_team_fg_pct": 0.486,
            "projected_opponent_fg_pct": 0.491,
            "pace_rebound_environment": 0.68,
        }
    )
    context = _supply_dependent_context(pd.Series(stable_row), "TRB", "OVER", 9.5)

    assert str(context["trb_over_bucket"]) == "TRB_OVER_STABLE"
    assert float(context["total_rebound_penalty"]) == 0.0
    assert bool(context["upper_band_line_flag"]) is False
    assert bool(context["low_line_role_volatility_flag"]) is False


def test_opposite_side_trb_under_candidate_is_created_when_over_is_penalized() -> None:
    rebound_row = _base_rebound_slate_row()
    rebound_row["market_under_price_TRB"] = 250.0
    slate = pd.DataFrame([rebound_row])
    plays = build_play_rows(
        slate,
        build_history_lookup(_synthetic_history()),
        line_decision_lookup={},
        line_decision_enabled=False,
    )

    assert len(plays) == 2
    over_row = plays.loc[plays["direction"] == "OVER"].iloc[0]
    under_row = plays.loc[plays["direction"] == "UNDER"].iloc[0]
    assert bool(over_row["opposite_side_candidate_flag"]) is True
    assert over_row["opposite_side_decision"] == "promote_under_candidate"
    assert under_row["rebound_diagnostic_segment"] == "TRB_UNDER_FROM_OPPOSITE_SIDE_DISCOVERY"
    assert under_row["recommendation"] in {"consider", "strong"}


def test_full_penalty_downgrades_candidate_without_silently_deleting_it() -> None:
    penalized_row = _base_rebound_slate_row()
    penalized_row["market_under_price_TRB"] = np.nan
    slate = pd.DataFrame([penalized_row])
    plays = build_play_rows(
        slate,
        build_history_lookup(_synthetic_history()),
        line_decision_lookup={},
        line_decision_enabled=False,
    )

    assert len(plays) == 1
    row = plays.iloc[0]
    assert row["direction"] == "OVER"
    assert float(row["total_rebound_penalty"]) > 0.0
    assert row["recommendation"] in {"pass", "consider"}
    assert row["rebound_diagnostic_segment"] in {
        "TRB_OVER_UPPER_BAND",
        "TRB_OVER_SUPPLY_DEPENDENT",
        "TRB_OVER_SHARE_COMPETITION",
    }


def test_validation_reports_removed_wins_and_losses_separately() -> None:
    baseline = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "Loss Removed", "target": "TRB", "direction": "OVER", "market_line": 11.5, "market_id": "TRB_OVER", "result": "loss"},
            {"run_date": "2026-04-01", "player": "Win Kept", "target": "TRB", "direction": "OVER", "market_line": 10.5, "market_id": "TRB_OVER", "result": "win"},
            {"run_date": "2026-04-01", "player": "Non Rebound", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win"},
        ]
    )
    variant = pd.DataFrame(
        [
            {"run_date": "2026-04-01", "player": "Win Kept", "target": "TRB", "direction": "OVER", "market_line": 10.5, "market_id": "TRB_OVER", "result": "win"},
            {"run_date": "2026-04-01", "player": "Non Rebound", "target": "PTS", "direction": "OVER", "market_line": 19.5, "market_id": "PTS_OVER", "result": "win"},
        ]
    )

    metrics = _compare_to_baseline(variant, baseline)

    assert metrics["removed_trb_over_losses"] == 1
    assert metrics["removed_trb_over_wins"] == 0
    assert metrics["kept_trb_over_wins"] == 1
    assert metrics["kept_trb_over_losses"] == 0


def test_build_play_rows_flips_effective_direction_when_opposite_side_is_preferred() -> None:
    history = _opposite_flip_history()
    slate = pd.DataFrame(
        [
            {
                "player": "Flip Candidate",
                "market_date": "2026-04-01",
                "market_player_raw": "Flip Candidate",
                "market_event_id": "game_flip",
                "market_commence_time_utc": "2026-04-01T23:00:00Z",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "history_rows": 140,
                "last_history_date": "2026-03-31",
                "csv": "flip.csv",
                "belief_uncertainty": 0.50,
                "feasibility": 0.92,
                "fallback_blend": 0.0,
                "pred_PTS": 20.35,
                "baseline_PTS": 19.8,
                "market_PTS": 19.5,
                "baseline_edge_PTS": 0.30,
                "PTS_uncertainty_sigma": 0.55,
                "PTS_spike_probability": 0.22,
                "market_books_PTS": 5,
                "pred_TRB": None,
                "baseline_TRB": None,
                "market_TRB": None,
                "baseline_edge_TRB": None,
                "TRB_uncertainty_sigma": None,
                "TRB_spike_probability": None,
                "market_books_TRB": None,
                "pred_AST": None,
                "baseline_AST": None,
                "market_AST": None,
                "baseline_edge_AST": None,
                "AST_uncertainty_sigma": None,
                "AST_spike_probability": None,
                "market_books_AST": None,
            }
        ]
    )

    plays = build_play_rows(
        slate,
        build_history_lookup(history),
        line_decision_lookup=build_line_decision_lookup(history),
        line_decision_enabled=True,
        line_decision_config=LineDecisionConfig(no_trade_threshold=0.45, min_trade_prob=0.57, min_trade_prob_gap=0.06),
    )

    assert len(plays) == 1
    row = plays.iloc[0]
    assert row["model_direction"] == "OVER"
    assert row["direction"] == "UNDER"
    assert row["line_decision_action"] == "UNDER"
    assert bool(row["line_action_is_opposite"]) is True
    assert bool(row["line_decision_trade_eligible"]) is True
    assert float(row["expected_win_rate"]) > 0.50
    assert float(row["prediction"]) < float(row["market_line"])


def test_compute_final_board_keeps_no_trade_rows_when_line_gate_is_disabled() -> None:
    board = compute_final_board(
        _board_input(),
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        max_plays_per_player=5,
        max_plays_per_target=0,
        max_total_plays=5,
        max_target_plays={"PTS": 5, "TRB": 5, "AST": 5},
        max_plays_per_game=5,
        max_plays_per_script_cluster=5,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.49,
        medium_bet_win_rate=0.52,
        full_bet_win_rate=0.56,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )
    assert not board.empty
    assert set(board["player"].tolist()) == {"Keep Me", "Drop Me"}
    assert set(board["line_decision_gate_mode"].unique().tolist()) == {"disabled_annotation_only"}
