from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from sports.mlb.scripts.latent_parlay_model import (
    DEFAULT_ARTIFACT_PATH,
    NUMERIC_FEATURES,
    LatentParlayBundle,
    candidate_features,
)
from sports.mlb.scripts.select_daily_parlay import _latent_set_profile


def load_bundle() -> LatentParlayBundle:
    bundle = LatentParlayBundle.load(DEFAULT_ARTIFACT_PATH)
    assert bundle is not None
    return bundle


def supported_leg(bundle: LatentParlayBundle, offset: float) -> tuple[dict[str, float], dict[str, str]]:
    numeric = {
        name: float(bundle.mean[index] + offset * 0.05 * bundle.scale[index])
        for index, name in enumerate(NUMERIC_FEATURES)
    }
    categories = {
        "player": f"player-{offset}",
        "pitcher": f"pitcher-{offset}",
        "team": f"team-{offset}",
        "opponent": f"opponent-{offset}",
    }
    return numeric, categories


def test_ticket_prediction_is_permutation_invariant() -> None:
    bundle = load_bundle()
    legs = [supported_leg(bundle, value) for value in (1.0, 2.0, 3.0)]

    forward = bundle.predict_ticket(legs)
    reverse = bundle.predict_ticket(list(reversed(legs)))

    assert forward.probability == pytest.approx(reverse.probability, abs=1e-12)
    assert forward.raw_probability == pytest.approx(reverse.raw_probability, abs=1e-12)
    assert forward.in_support is True


def test_out_of_support_leg_is_flagged_without_widening_confidence() -> None:
    bundle = load_bundle()
    numeric, categories = supported_leg(bundle, 0.0)
    numeric = {name: value + 1_000_000.0 for name, value in numeric.items()}

    prediction = bundle.predict_leg(numeric, categories)

    assert prediction.in_support is False
    assert prediction.support_fraction < bundle.minimum_support_fraction


def test_candidate_features_do_not_use_projection_or_edge() -> None:
    raw = {
        "Baseline": 1.1,
        "Edge": 9.0,
        "Prediction_Run_Date": "2026-08-14",
        "Is_Home": 1,
        "Opponent": "NYY",
    }
    first = SimpleNamespace(
        raw=dict(raw),
        run_date=date(2026, 8, 14),
        prediction=1.2,
        history_rows=80,
        player_id="player-1",
        player="Player One",
        team="BOS",
    )
    second = SimpleNamespace(**{**first.__dict__, "prediction": 4.8, "raw": {**raw, "Edge": 3.7}})

    first_numeric, first_categories = candidate_features(first, last_hits=2.0, batting_order=3.0)
    second_numeric, second_categories = candidate_features(second, last_hits=2.0, batting_order=3.0)

    assert first_numeric == second_numeric
    assert first_categories == second_categories
    assert "prediction" not in first_numeric
    assert "edge" not in first_numeric


def test_selector_profile_exposes_shadow_score_without_replacing_active_score() -> None:
    bundle = load_bundle()
    model_inputs = [supported_leg(bundle, value) for value in (1.0, 2.0)]
    legs = [
        {
            "target": "H",
            "parlay_leg_probability": 0.66 + index * 0.01,
            "latent_leg_probability": 0.66 + index * 0.01,
            "market_implied_probability": 0.68 + index * 0.01,
            "latent_leg_ensemble_std": 0.02,
            "latent_probability_disagreement": 0.03,
            "latent_numeric_features": numeric,
            "latent_categorical_features": categories,
        }
        for index, (numeric, categories) in enumerate(model_inputs)
    ]

    profile = _latent_set_profile(legs, projected_probability=0.44, latent_bundle=bundle)

    assert profile["set_consistency_score"] == pytest.approx(0.44 * (1.0 - 0.0275))
    assert 0.0 < profile["shadow_joint_probability"] < 1.0
    assert profile["shadow_independent_leg_product"] == pytest.approx(0.66 * 0.67)
    assert profile["shadow_market_leg_product"] == pytest.approx(0.68 * 0.69)
    assert 0.0 < profile["shadow_hybrid_leg_product"] < 1.0
    assert profile["shadow_authorization"] == "diagnostic_only"
