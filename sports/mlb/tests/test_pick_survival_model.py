from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


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


def _winner_signature_rows() -> pd.DataFrame:
    """v12 Phase 1 fixture: the same shape train_winner_signature_model()
    actually expects -- the v11-eligible, real-settled population build_
    v11_eligible_training_set.py produces -- which carries the richer
    enrichment columns pick_survival_model's own _training_rows() doesn't
    (historical bucket/bet-profile/market-availability, survival_probability)
    plus the raw inputs _with_disagreement_features() derives new columns
    from."""
    rows = _training_rows().copy()
    rows["abs_edge"] = rows["directional_edge"].abs()
    rows["market_common_books"] = 3.0
    rows["survival_probability"] = rows["model_hit_probability"]
    rows["historical_bucket_win_rate"] = 0.6
    rows["historical_bucket_support"] = 50.0
    rows["historical_bet_profile_win_rate"] = 0.55
    rows["historical_bet_profile_roi"] = 0.05
    rows["historical_bet_profile_support"] = 20.0
    rows["historical_market_availability_rate"] = 0.9
    rows["historical_market_availability_support"] = 10.0
    rows["live_confidence_calibration_adjustment"] = 0.0
    return rows


def test_winner_signature_training_uses_its_own_feature_set_and_identity() -> None:
    payload = survival.train_winner_signature_model(_winner_signature_rows(), top_k=3)

    assert payload["status"] == "shadow"
    assert payload["model_version"] == survival.WINNER_SIGNATURE_MODEL_VERSION
    assert set(payload["feature_contract"]["coefficients"]) == set(
        survival.WINNER_SIGNATURE_NUMERIC_FEATURES + survival.CATEGORICAL_FEATURES
    )
    # The real disagreement features _with_disagreement_features() derives
    # (not present at all in pick_survival_model's own NUMERIC_FEATURES)
    # must actually have been fit on, not just listed.
    assert "model_market_disagreement" in payload["feature_contract"]["means"]
    assert "model_survival_disagreement" in payload["feature_contract"]["means"]


def test_winner_signature_training_fails_closed_on_the_real_v11_eligible_row_count() -> None:
    """Real disclosed result: build_v11_eligible_training_set.py currently
    produces 130 real v11-eligible settled rows against MIN_TRAIN_ROWS=180
    -- this must report insufficient_support, never fabricate a fit on too
    little evidence."""
    rows = _winner_signature_rows().head(130)

    payload = survival.train_winner_signature_model(rows, top_k=3)

    assert payload["status"] == "insufficient_support"
    assert payload["shadow_only"] is True
    assert payload["model_version"] == survival.WINNER_SIGNATURE_MODEL_VERSION
    assert payload["training_rows"] == 130


def _safe_ev_candidate(**overrides) -> SimpleNamespace:
    base = dict(
        raw={},
        target="R",
        direction="OVER",
        run_date=date(2026, 5, 1),
        calibrated_hit_probability=0.80,
        selected_side_price=110.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fixed_probability_payload(probability: float, **overrides) -> dict:
    """A hand-built payload whose feature_contract has NO numeric or
    categorical features -- so apply_winner_signature_model()'s logit is
    exactly the intercept, giving full control over the resulting
    winner_signature_probability regardless of the candidate's own
    features. This isolates the min()-clamp/veto logic under test from
    the real model fit's randomness."""
    logit = math.log(probability / (1.0 - probability))
    payload = {
        "status": "shadow",
        "shadow_only": True,
        "training_end_date": "2026-01-01",
        "segment_support": {"R|OVER": 50},
        "minimum_segment_rows": 10,
        "feature_contract": {
            "numeric_features": [],
            "categorical_features": [],
            "means": {},
            "scales": {},
            "coefficients": {},
            "intercept": logit,
        },
    }
    payload.update(overrides)
    return payload


def test_apply_winner_signature_model_never_raises_probability_above_v11s_own() -> None:
    """The core negative-authority-only guarantee: a winner-signature
    output MORE confident than v11's own calibrated probability must NOT
    raise safe_probability -- it stays at v11's own (lower) bar."""
    candidate = _safe_ev_candidate(calibrated_hit_probability=0.80)
    payload = _fixed_probability_payload(0.95)  # winner-signature says 95%, v11 says 80%

    (
        winner_signature_probability,
        safe_probability,
        safe_expected_value,
        safe_probability_edge,
        safe_ev_veto,
        status,
        support,
    ) = survival.apply_winner_signature_model(candidate, payload)

    assert winner_signature_probability == pytest.approx(0.95)
    assert safe_probability == pytest.approx(0.80)  # clamped to v11's own, not the higher 0.95
    assert safe_ev_veto is False  # not materially below v11's own probability
    assert status == "shadow"
    assert support == 50
    decimal_price = 2.1  # +110 -> profit 1.1
    assert safe_expected_value == pytest.approx(0.80 * decimal_price - 1.0)
    assert safe_probability_edge == pytest.approx(0.80 - 1.0 / decimal_price)


def test_apply_winner_signature_model_pulls_probability_down_when_it_disagrees() -> None:
    """The other half of the same guarantee: when the winner-signature
    model is LESS confident than v11, safe_probability must actually drop
    to match it -- this is the real veto authority."""
    candidate = _safe_ev_candidate(calibrated_hit_probability=0.80)
    payload = _fixed_probability_payload(0.50)  # winner-signature says 50%, v11 says 80%

    winner_signature_probability, safe_probability, _, _, safe_ev_veto, _, _ = survival.apply_winner_signature_model(
        candidate, payload
    )

    assert winner_signature_probability == pytest.approx(0.50)
    assert safe_probability == pytest.approx(0.50)
    assert safe_ev_veto is True  # 0.50 is well below v11's 0.80 - 0.02 margin


def test_apply_winner_signature_model_veto_margin_is_not_tripped_exactly_at_the_boundary() -> None:
    candidate = _safe_ev_candidate(calibrated_hit_probability=0.80)
    payload = _fixed_probability_payload(0.78)  # exactly v11 - 0.02

    _, safe_probability, _, _, safe_ev_veto, _, _ = survival.apply_winner_signature_model(candidate, payload)

    assert safe_probability == pytest.approx(0.78)
    assert safe_ev_veto is False  # strictly-less-than, boundary itself does not veto


def test_apply_winner_signature_model_disabled_payload_passes_v11_probability_through_unchanged() -> None:
    candidate = _safe_ev_candidate(calibrated_hit_probability=0.80)

    result = survival.apply_winner_signature_model(candidate, None)

    assert result == (None, pytest.approx(0.80), None, None, False, "disabled", 0)


def test_apply_winner_signature_model_fails_closed_on_cutoff_and_low_segment_support() -> None:
    candidate = _safe_ev_candidate(calibrated_hit_probability=0.80, run_date=date(2026, 1, 1))
    payload = _fixed_probability_payload(0.95)  # training_end_date defaults to 2026-01-01

    result = survival.apply_winner_signature_model(candidate, payload)
    assert result[1] == pytest.approx(0.80)  # v11 probability still passed through on fail-closed
    assert result[5] == "cutoff_violation"

    candidate.run_date = date(2026, 5, 1)
    candidate.target = "K"  # not in the fixture's segment_support map
    result = survival.apply_winner_signature_model(candidate, payload)
    assert result[5] == "insufficient_segment_support"


def test_apply_winner_signature_model_safe_probability_is_none_without_a_v11_probability() -> None:
    candidate = _safe_ev_candidate(calibrated_hit_probability=None)
    payload = _fixed_probability_payload(0.95)

    winner_signature_probability, safe_probability, safe_expected_value, safe_probability_edge, safe_ev_veto, status, _ = (
        survival.apply_winner_signature_model(candidate, payload)
    )

    assert winner_signature_probability == pytest.approx(0.95)
    assert safe_probability is None
    assert safe_expected_value is None
    assert safe_probability_edge is None
    assert safe_ev_veto is False


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
