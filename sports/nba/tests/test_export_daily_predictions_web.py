from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from export_daily_predictions_web import (
    apply_adaptive_board_sizing,
    apply_variance_aware_reexpand,
    build_selector_pool_fallback,
    enrich_selector_pool_candidates,
)


def test_build_selector_pool_fallback_keeps_only_quality_rows() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Ayo Dosunmu",
                "game_key": "g1",
                "target": "TRB",
                "direction": "OVER",
                "expected_win_rate": 0.5143,
                "ev": 0.0184,
                "final_confidence": 0.0674,
                "abs_edge": 0.7018,
            },
            {
                "player": "Josh Hart",
                "game_key": "g2",
                "target": "TRB",
                "direction": "UNDER",
                "expected_win_rate": 0.5101,
                "ev": 0.0142,
                "final_confidence": 0.0682,
                "abs_edge": 0.5885,
            },
            {
                "player": "Nickeil Alexander-Walker",
                "game_key": "g2",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.5008,
                "ev": -0.0006,
                "final_confidence": 0.1616,
                "abs_edge": 1.2571,
            },
            {
                "player": "Jalen Duren",
                "game_key": "g3",
                "target": "PTS",
                "direction": "OVER",
                "expected_win_rate": 0.5006,
                "ev": -0.0017,
                "final_confidence": 0.1525,
                "abs_edge": 1.3027,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback["player"].tolist() == ["Ayo Dosunmu", "Josh Hart"]


def test_build_selector_pool_fallback_returns_empty_when_slate_is_too_weak() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Weak One",
                "game_key": "g1",
                "expected_win_rate": 0.503,
                "ev": -0.001,
                "final_confidence": 0.06,
                "abs_edge": 0.8,
            },
            {
                "player": "Weak Two",
                "game_key": "g2",
                "expected_win_rate": 0.509,
                "ev": 0.003,
                "final_confidence": 0.03,
                "abs_edge": 0.7,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback.empty


def test_build_selector_pool_fallback_applies_adaptive_sizing_then_reexpands_to_three() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "Core One",
                "game_key": "g1",
                "expected_win_rate": 0.5196,
                "ev": 0.0288,
                "final_confidence": 0.14,
                "abs_edge": 1.20,
                "uncertainty_sigma": 3.0,
                "history_rows": 70,
                "spike_probability": 0.20,
                "contradiction_score": 0.05,
                "recoverability_score": 0.80,
                "agreement_count": 3,
            },
            {
                "player": "Core Two",
                "game_key": "g2",
                "expected_win_rate": 0.5195,
                "ev": 0.0287,
                "final_confidence": 0.139,
                "abs_edge": 1.19,
                "uncertainty_sigma": 3.0,
                "history_rows": 69,
                "spike_probability": 0.21,
                "contradiction_score": 0.051,
                "recoverability_score": 0.79,
                "agreement_count": 3,
            },
            {
                "player": "Expand Three",
                "game_key": "g1",
                "expected_win_rate": 0.5194,
                "ev": 0.0286,
                "final_confidence": 0.138,
                "abs_edge": 1.18,
                "uncertainty_sigma": 3.0,
                "history_rows": 68,
                "spike_probability": 0.22,
                "contradiction_score": 0.052,
                "recoverability_score": 0.78,
                "agreement_count": 3,
            },
            {
                "player": "Expand Four",
                "game_key": "g2",
                "expected_win_rate": 0.5193,
                "ev": 0.0285,
                "final_confidence": 0.137,
                "abs_edge": 1.17,
                "uncertainty_sigma": 3.0,
                "history_rows": 67,
                "spike_probability": 0.23,
                "contradiction_score": 0.053,
                "recoverability_score": 0.77,
                "agreement_count": 3,
            },
        ]
    )

    fallback = build_selector_pool_fallback(plays)

    assert fallback["player"].tolist() == ["Core One", "Core Two", "Expand Three"]


def test_enrich_selector_pool_candidates_prefers_balanced_profile() -> None:
    plays = pd.DataFrame(
        [
            {
                "player": "High Prob Fragile",
                "game_key": "g1",
                "expected_win_rate": 0.515,
                "p_calibrated": 0.500,
                "ev": 0.001,
                "final_confidence": 0.05,
                "abs_edge": 0.50,
                "uncertainty_sigma": 9.0,
                "history_rows": 25,
                "spike_probability": 0.80,
                "contradiction_score": 0.30,
                "recoverability_score": 0.20,
                "agreement_count": 1,
            },
            {
                "player": "Balanced Strong",
                "game_key": "g2",
                "expected_win_rate": 0.511,
                "p_calibrated": 0.522,
                "ev": 0.010,
                "final_confidence": 0.15,
                "abs_edge": 1.20,
                "uncertainty_sigma": 3.0,
                "history_rows": 70,
                "spike_probability": 0.20,
                "contradiction_score": 0.05,
                "recoverability_score": 0.80,
                "agreement_count": 3,
            },
        ]
    )

    enriched = enrich_selector_pool_candidates(plays).sort_values("pool_selection_score", ascending=False).reset_index(drop=True)

    assert enriched.loc[0, "player"] == "Balanced Strong"


def test_apply_adaptive_board_sizing_trims_marginal_tail() -> None:
    plays = pd.DataFrame(
        [
            {"player": "Top One", "pool_selection_score": 2.60, "selection_confidence": 0.90},
            {"player": "Top Two", "pool_selection_score": 2.42, "selection_confidence": 0.86},
            {"player": "Marginal Three", "pool_selection_score": 2.05, "selection_confidence": 0.66},
            {"player": "Marginal Four", "pool_selection_score": 1.98, "selection_confidence": 0.60},
        ]
    )

    sized = apply_adaptive_board_sizing(plays)

    assert sized["player"].tolist() == ["Top One", "Top Two"]


def test_apply_variance_aware_reexpand_adds_third_play_on_low_conviction_two_leg_board() -> None:
    universe = pd.DataFrame(
        [
            {"player": "Top One", "selection_probability": 0.530, "selection_confidence": 0.11, "selection_ev": 0.02},
            {"player": "Top Two", "selection_probability": 0.519, "selection_confidence": 0.13, "selection_ev": 0.02},
            {"player": "Third Qualifier", "selection_probability": 0.513, "selection_confidence": 0.15, "selection_ev": 0.005},
        ]
    )
    trimmed = universe.iloc[:2].copy().reset_index(drop=True)

    expanded = apply_variance_aware_reexpand(
        trimmed,
        universe,
        probability_field="selection_probability",
        confidence_field="selection_confidence",
        ev_field="selection_ev",
        max_top2_avg_probability=0.526,
        min_third_probability=0.512,
        min_third_confidence=0.12,
        min_third_ev=0.0,
    )

    assert expanded["player"].tolist() == ["Top One", "Top Two", "Third Qualifier"]
