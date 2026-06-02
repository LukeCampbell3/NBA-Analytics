from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "scripts"))

from export_daily_predictions_web import PlayerIdentityLookup, normalize_play_rows
from post_process_market_plays import compute_final_board
from research.market_quality.parlay_price_defense import evaluate_parlay_price_defense
from research.market_quality.price_normalization import (
    american_odds_to_break_even,
    american_odds_to_decimal,
    select_side_specific_price,
)
from research.market_quality.priced_event_ledger import build_priced_event_ledger_frame


def _ledger_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": "candidate::ledger",
        "player": "Ledger Player",
        "market_player_raw": "Ledger Player",
        "player_name": "Ledger Player",
        "market_event_id": "game_ledger",
        "game_id": "game_ledger",
        "game_date": "2026-04-01",
        "market_date": "2026-04-01",
        "run_date": "2026-04-01",
        "market_commence_time_utc": "2026-04-02T00:00:00+00:00",
        "team": "AAA",
        "opponent": "BBB",
        "market_home_team": "AAA",
        "market_away_team": "BBB",
        "target": "PTS",
        "direction": "OVER",
        "side": "OVER",
        "market_id": "PTS_OVER",
        "market_type": "PTS_OVER",
        "market_line": 22.5,
        "line": 22.5,
        "prediction": 24.0,
        "market_side_price": -110.0,
        "over_price": -110.0,
        "under_price": -110.0,
        "price_source": "current_market_snapshot_pre_event",
        "price_source_hint": "",
        "odds_snapshot_time": "2026-04-01T16:00:00+00:00",
        "prediction_snapshot_time": "2026-04-01T16:10:00+00:00",
        "selector_run_time": "2026-04-01T16:12:00+00:00",
        "expected_win_rate": 0.56,
        "model_probability": 0.57,
        "stress_probability": 0.56,
        "lcb_probability": 0.55,
        "p_push": 0.0,
        "forecastability_score": 0.80,
        "scenario_agreement": 0.75,
        "chaos_score": 0.20,
        "belief_uncertainty": 0.20,
        "posterior_variance": 0.01,
        "feasibility": 0.90,
        "belief_confidence_factor": 0.85,
        "coach_trust_score": 0.80,
        "rotation_volatility_score": 0.10,
        "recommendation": "strong",
        "history_rows": 100,
        "selected_rank": 1,
        "selected_on_board": True,
    }
    row.update(overrides)
    return row


def _selector_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "Board One",
                "market_player_raw": "Board One",
                "target": "PTS",
                "direction": "OVER",
                "prediction": 25.2,
                "market_line": 22.5,
                "abs_edge": 2.7,
                "edge": 2.7,
                "expected_win_rate": 0.67,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.76,
                "feasibility": 0.92,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_board_one",
                "market_home_team": "AAA",
                "market_away_team": "BBB",
                "gap_percentile": 0.96,
            },
            {
                "player": "Board Two",
                "market_player_raw": "Board Two",
                "target": "AST",
                "direction": "UNDER",
                "prediction": 5.1,
                "market_line": 6.0,
                "abs_edge": 0.9,
                "edge": -0.9,
                "expected_win_rate": 0.63,
                "expected_push_rate": 0.03,
                "posterior_variance": 0.03,
                "belief_uncertainty": 0.75,
                "feasibility": 0.90,
                "recommendation": "strong",
                "history_rows": 120,
                "market_date": "2026-04-01",
                "last_history_date": "2026-03-31",
                "market_event_id": "game_board_two",
                "market_home_team": "CCC",
                "market_away_team": "DDD",
                "gap_percentile": 0.93,
            },
        ]
    )


def _run_board(plays: pd.DataFrame) -> pd.DataFrame:
    return compute_final_board(
        plays,
        selection_mode="ev_adjusted",
        ranking_mode="ev_adjusted",
        min_recommendation="consider",
        min_ev=-1.0,
        min_final_confidence=0.0,
        max_total_plays=10,
        max_plays_per_game=0,
        max_plays_per_script_cluster=0,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.40,
        medium_bet_win_rate=0.50,
        full_bet_win_rate=0.60,
    )


def test_side_specific_price_is_selected_correctly() -> None:
    assert select_side_specific_price("OVER", over_price=-115.0, under_price=105.0) == -115.0
    assert select_side_specific_price("UNDER", over_price=-115.0, under_price=105.0) == 105.0
    assert select_side_specific_price("OVER", explicit_price=-108.0, over_price=-115.0, under_price=105.0) == -108.0


def test_missing_price_creates_edge_untrusted_price() -> None:
    row = _ledger_row(
        market_side_price=np.nan,
        over_price=np.nan,
        under_price=np.nan,
        price_source="",
    )
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    assert ledger.iloc[0]["price_validity_status"] == "MISSING_PRICE"
    assert ledger.iloc[0]["edge_defendability_tier"] == "EDGE_UNTRUSTED_PRICE"


def test_close_only_price_creates_edge_diagnostic_only() -> None:
    row = _ledger_row(price_source_hint="close_only_snapshot")
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    assert ledger.iloc[0]["price_source_type"] == "CLOSE_ONLY_DIAGNOSTIC"
    assert ledger.iloc[0]["edge_defendability_tier"] == "EDGE_DIAGNOSTIC_ONLY"


def test_timestamp_unsafe_price_cannot_be_price_valid() -> None:
    row = _ledger_row(
        odds_snapshot_time="2026-04-02T00:05:00+00:00",
        prediction_snapshot_time="2026-04-01T16:10:00+00:00",
    )
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    assert bool(ledger.iloc[0]["timestamp_safe_flag"]) is False
    assert ledger.iloc[0]["price_validity_status"] != "PRICE_VALID"


def test_lcb_edge_calculation_is_correct() -> None:
    row = _ledger_row(market_side_price=-110.0, lcb_probability=0.56)
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    expected = 0.56 - american_odds_to_break_even(-110.0)
    assert float(ledger.iloc[0]["lcb_edge"]) == pytest.approx(expected, abs=1e-9)


def test_minimum_acceptable_odds_calculation_is_correct() -> None:
    row = _ledger_row(stress_probability=0.55, lcb_probability=0.53, p_push=0.0)
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    assert float(ledger.iloc[0]["minimum_acceptable_odds"]) == pytest.approx(-122.2222222222, abs=1e-6)


def test_edge_defendable_requires_price_stress_lcb_and_forecastability() -> None:
    defendable = build_priced_event_ledger_frame(pd.DataFrame([_ledger_row()]), record_scope="selected")
    fragile = build_priced_event_ledger_frame(
        pd.DataFrame([_ledger_row(forecastability_score=0.20)]),
        record_scope="selected",
    )
    assert defendable.iloc[0]["edge_defendability_tier"] == "EDGE_DEFENDABLE"
    assert fragile.iloc[0]["edge_defendability_tier"] == "EDGE_FAILS_PRICE"
    assert fragile.iloc[0]["edge_defendability_reason"] == "forecastability_below_threshold"


def test_edge_price_dependent_emits_required_odds() -> None:
    row = _ledger_row(
        market_side_price=-130.0,
        model_probability=0.58,
        stress_probability=0.56,
        lcb_probability=0.54,
    )
    ledger = build_priced_event_ledger_frame(pd.DataFrame([row]), record_scope="selected")
    assert ledger.iloc[0]["edge_defendability_tier"] == "EDGE_PRICE_DEPENDENT"
    assert ledger.iloc[0]["price_valid_decision"] == "PRICE_DEPENDENT"
    assert pd.notna(ledger.iloc[0]["minimum_acceptable_odds"])


def test_parlay_break_even_uses_book_quoted_odds_when_available() -> None:
    legs = [
        {
            "game_id": "g1",
            "team": "AAA",
            "player": "A",
            "market_side_decimal_odds": american_odds_to_decimal(-110.0),
            "model_probability": 0.60,
            "stress_probability": 0.58,
            "lcb_probability": 0.55,
        },
        {
            "game_id": "g2",
            "team": "BBB",
            "player": "B",
            "market_side_decimal_odds": american_odds_to_decimal(-105.0),
            "model_probability": 0.59,
            "stress_probability": 0.57,
            "lcb_probability": 0.54,
        },
    ]
    defense = evaluate_parlay_price_defense(legs, parlay_american_odds=260.0)
    assert defense["parlay_price_mode"] == "BOOK_QUOTED_PARLAY"
    assert defense["parlay_break_even"] == pytest.approx(1.0 / 3.6, abs=1e-9)


def test_synthetic_product_odds_are_not_validation_safe_same_game_sgp() -> None:
    legs = [
        {
            "game_id": "g_same",
            "team": "AAA",
            "player": "A",
            "market_side_decimal_odds": american_odds_to_decimal(-110.0),
            "model_probability": 0.60,
            "stress_probability": 0.58,
            "lcb_probability": 0.55,
        },
        {
            "game_id": "g_same",
            "team": "AAA",
            "player": "B",
            "market_side_decimal_odds": american_odds_to_decimal(-105.0),
            "model_probability": 0.59,
            "stress_probability": 0.57,
            "lcb_probability": 0.54,
        },
    ]
    defense = evaluate_parlay_price_defense(legs)
    assert defense["parlay_price_mode"] == "SYNTHETIC_DIAGNOSTIC"
    assert defense["parlay_price_validity_status"] == "DIAGNOSTIC_ONLY"


def test_audit_columns_do_not_change_production_board_results() -> None:
    baseline_rows = _selector_rows()
    enriched_rows = baseline_rows.copy()
    enriched_rows["market_side_price"] = [-110.0, np.nan]
    enriched_rows["market_side_break_even"] = [american_odds_to_break_even(-110.0), np.nan]
    enriched_rows["price_source"] = ["current_market_snapshot_pre_event", ""]
    enriched_rows["price_validity_status"] = ["PRICE_VALID", "MISSING_PRICE"]
    enriched_rows["odds_snapshot_time"] = ["2026-04-01T16:00:00+00:00", ""]
    enriched_rows["prediction_snapshot_time"] = ["2026-04-01T16:05:00+00:00", ""]

    baseline = _run_board(baseline_rows)
    enriched = _run_board(enriched_rows)

    assert baseline["player"].tolist() == enriched["player"].tolist()
    assert baseline["selected_rank"].tolist() == enriched["selected_rank"].tolist()
    assert baseline["recommendation"].tolist() == enriched["recommendation"].tolist()


def test_selected_board_exports_include_price_provenance_fields() -> None:
    ledger = build_priced_event_ledger_frame(pd.DataFrame([_ledger_row()]), record_scope="selected")
    exported = normalize_play_rows(ledger, PlayerIdentityLookup(name_to_id={}, abbr_to_id={}))
    row = exported[0]
    assert row["market_side_price"] == -110.0
    assert row["market_side_break_even"] == pytest.approx(american_odds_to_break_even(-110.0), abs=1e-9)
    assert row["edge_defendability_tier"] == "EDGE_DEFENDABLE"
    assert row["price_valid_decision"] == "KEEP"
    assert row["odds_snapshot_time"] == "2026-04-01 16:00:00+00:00"
