from __future__ import annotations

from dataclasses import replace

from sports.mlb.advanced.game_conditioned_moe import (
    EXPERT_NAMES,
    MODEL_VERSION,
    build_expert_state,
    condition_probability,
)
from sports.mlb.advanced.schema import AdvancedCandidateContext, BatterProcessProfile, PitcherProcessProfile
from sports.mlb.advanced.sequential_pa_model import simulate_hitter_market


def batter(**overrides):
    base = BatterProcessProfile(
        player_id=11,
        player_name="Batter",
        as_of_date="2026-09-05",
        sample_pa=400,
        sample_bbe=260,
        k_rate=0.19,
        bb_rate=0.10,
        hbp_rate=0.01,
        hr_rate=0.045,
        contact_rate=0.81,
        whiff_rate=0.19,
        xwoba=0.365,
        xba=0.285,
        xslg=0.515,
        hard_hit_rate=0.45,
        barrel_rate=0.105,
        support=0.95,
    )
    return replace(base, **overrides)


def pitcher(**overrides):
    base = PitcherProcessProfile(
        player_id=22,
        player_name="Pitcher",
        as_of_date="2026-09-05",
        sample_pa=500,
        sample_bbe=310,
        k_rate=0.25,
        bb_rate=0.08,
        hbp_rate=0.01,
        hr_rate=0.03,
        k_minus_bb_rate=0.17,
        whiff_rate=0.255,
        xwoba_allowed=0.315,
        xba_allowed=0.245,
        xslg_allowed=0.405,
        hard_hit_rate_allowed=0.37,
        barrel_rate_allowed=0.07,
        gb_rate=0.45,
        xfip=3.60,
        siera=3.55,
        projected_ip=5.8,
        support=0.95,
    )
    return replace(base, **overrides)


def context(*, b=None, p=None, target_runs=4.8, order=2, defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY", defense_residual=0.0, temperature_f=72.0):
    return AdvancedCandidateContext(
        game_id="game",
        run_date="2026-09-05",
        batter=b or batter(),
        pitcher=p or pitcher(),
        direct_matchup=None,
        batting_order=order,
        is_home=False,
        team_expected_runs=target_runs,
        park_factor=1.0,
        defense_residual=defense_residual,
        defense_status=defense_status,
        data_freshness_status="FRESH",
        missing_components=(),
        temperature_f=temperature_f,
    )


def fitted_artifact(*, positive=False, evidence_class="ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION"):
    coefficients = {
        "strikeout_contact": 0.22,
        "contact_quality": 0.18,
        "power_tb": 0.16,
        "defense_conversion": 0.08,
        "pa_opportunity": 0.14,
        "bullpen_transition": 0.10,
    }
    return {
        "schema_version": "mlb_game_conditioned_hitter_moe_v2",
        "model_version": MODEL_VERSION,
        "training_status": "FITTED_TEST",
        "evidence_class": evidence_class,
        "max_abs_residual_logit": 0.35,
        "targets": {
            target: {
                "intercept": 0.0,
                "coefficients": coefficients,
                "feature_means": {name: 0.0 for name in EXPERT_NAMES},
                "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
                "positive_authority": positive,
                "validation": {"prior_brier": 0.24, "candidate_brier": 0.22},
            }
            for target in ("H", "TB")
        },
    }


def state(ctx, target="H"):
    seq = simulate_hitter_market(ctx, target=target, market_line=0.5 if target == "H" else 1.5, trials=4000)
    return build_expert_state(ctx, seq, target=target, pitch_compatibility_score=0.20), seq


def test_high_k_matchup_increases_strikeout_expert_activation():
    low_state, _ = state(context(p=pitcher(k_rate=0.17, k_minus_bb_rate=0.09, whiff_rate=0.18)))
    high_state, _ = state(context(p=pitcher(k_rate=0.36, k_minus_bb_rate=0.28, whiff_rate=0.35)))
    assert high_state.activations["strikeout_contact"] > low_state.activations["strikeout_contact"]
    assert high_state.signals["strikeout_contact"] < low_state.signals["strikeout_contact"]


def test_tb_market_weights_power_more_than_hits_market():
    ctx = context()
    h_state, _ = state(ctx, "H")
    tb_state, _ = state(ctx, "TB")
    assert tb_state.activations["power_tb"] > h_state.activations["power_tb"]


def test_specific_defense_activates_defense_expert_but_average_context_does_not_fake_it():
    average, _ = state(context(defense_status="AVERAGE_CONTEXT_RESIDUAL_ONLY", defense_residual=0.0))
    elite, _ = state(context(defense_status="SPECIFIC_DEFENSE_AVAILABLE", defense_residual=-0.03))
    assert average.signals["defense_conversion"] == 0.0
    assert elite.signals["defense_conversion"] < 0.0
    assert elite.activations["defense_conversion"] > average.activations["defense_conversion"]


def test_game_state_changes_effective_pa_opportunity_feature():
    weak, _ = state(context(target_runs=3.2, order=8))
    strong, _ = state(context(target_runs=6.2, order=1))
    assert strong.signals["pa_opportunity"] > weak.signals["pa_opportunity"]
    assert strong.effective_features["pa_opportunity"] > weak.effective_features["pa_opportunity"]


def test_rolling_form_changes_game_specific_contact_state_without_replacing_season_prior():
    hot_rolling = {
        "last_15": {"pa": 15, "k_rate": 0.10, "whiff_rate": 0.10, "xwoba_contact": 0.470, "xslg_contact": 0.700, "hard_hit_rate": 0.62, "barrel_rate": 0.18},
        "last_30": {"pa": 30, "k_rate": 0.13, "whiff_rate": 0.12, "xwoba_contact": 0.430, "xslg_contact": 0.640, "hard_hit_rate": 0.56, "barrel_rate": 0.15},
        "last_60": {"pa": 60, "k_rate": 0.16, "whiff_rate": 0.15, "xwoba_contact": 0.400, "xslg_contact": 0.590, "hard_hit_rate": 0.51, "barrel_rate": 0.13},
    }
    cold_rolling = {
        "last_15": {"pa": 15, "k_rate": 0.34, "whiff_rate": 0.33, "xwoba_contact": 0.240, "xslg_contact": 0.310, "hard_hit_rate": 0.25, "barrel_rate": 0.03},
        "last_30": {"pa": 30, "k_rate": 0.30, "whiff_rate": 0.29, "xwoba_contact": 0.270, "xslg_contact": 0.340, "hard_hit_rate": 0.29, "barrel_rate": 0.04},
        "last_60": {"pa": 60, "k_rate": 0.27, "whiff_rate": 0.26, "xwoba_contact": 0.300, "xslg_contact": 0.380, "hard_hit_rate": 0.33, "barrel_rate": 0.05},
    }
    hot, _ = state(context(b=batter(rolling=hot_rolling)))
    cold, _ = state(context(b=batter(rolling=cold_rolling)))
    assert hot.signals["strikeout_contact"] > cold.signals["strikeout_contact"]
    assert hot.signals["contact_quality"] > cold.signals["contact_quality"]
    assert hot.activations["contact_quality"] >= cold.activations["contact_quality"]


def test_warm_weather_changes_power_context_but_not_pa_opportunity():
    cold, _ = state(context(temperature_f=45.0), "TB")
    warm, _ = state(context(temperature_f=95.0), "TB")
    assert warm.signals["power_tb"] > cold.signals["power_tb"]
    assert warm.signals["pa_opportunity"] == cold.signals["pa_opportunity"]


def test_shadow_model_can_raise_candidate_but_not_production_probability():
    ctx = context(b=batter(k_rate=0.12, whiff_rate=0.12, xwoba=0.410, xba=0.325, xslg=0.620), target_runs=5.8, order=1)
    expert_state, seq = state(ctx, "H")
    result = condition_probability(0.62, target="H", state=expert_state, artifact=fitted_artifact(), sequential_uncertainty=seq.uncertainty)
    assert result.candidate_probability > 0.62
    assert result.production_probability <= 0.62
    assert result.positive_authority is False
    assert abs(sum(result.expert_weights.values()) - 1.0) < 1e-12


def test_exact_validated_promotion_can_apply_bidirectional_residual():
    ctx = context(b=batter(k_rate=0.12, whiff_rate=0.12, xwoba=0.410, xba=0.325, xslg=0.620), target_runs=5.8, order=1)
    expert_state, seq = state(ctx, "H")
    artifact = fitted_artifact(positive=True, evidence_class="EXACT_POINT_IN_TIME_LOCKED_VALIDATION")
    result = condition_probability(0.62, target="H", state=expert_state, artifact=artifact, sequential_uncertainty=seq.uncertainty)
    assert result.positive_authority is True
    assert result.production_probability > 0.62
    assert result.authority_status.startswith("PROMOTED")


def test_non_exact_evidence_cannot_unlock_positive_authority_even_if_flag_is_true():
    expert_state, seq = state(context(), "H")
    artifact = fitted_artifact(positive=True, evidence_class="ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION")
    result = condition_probability(0.62, target="H", state=expert_state, artifact=artifact, sequential_uncertainty=seq.uncertainty)
    assert result.positive_authority is False
    assert result.production_probability <= 0.62


def test_residual_logit_is_bounded():
    ctx = context(b=batter(k_rate=0.02, whiff_rate=0.02, xwoba=0.600, xba=0.500, xslg=1.100, barrel_rate=0.30), p=pitcher(k_rate=0.05, xwoba_allowed=0.450, xba_allowed=0.400, xslg_allowed=0.800))
    expert_state, seq = state(ctx, "TB")
    result = condition_probability(0.55, target="TB", state=expert_state, artifact=fitted_artifact(), sequential_uncertainty=seq.uncertainty)
    assert abs(result.residual_logit) <= 0.35 + 1e-12
