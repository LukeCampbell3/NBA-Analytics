from sports.mlb.advanced.game_conditioned_moe import (
    EXPERT_NAMES,
    ExpertState,
    condition_probability,
)


def _state() -> ExpertState:
    return ExpertState(
        signals={name: 0.0 for name in EXPERT_NAMES},
        activations={name: 1.0 for name in EXPERT_NAMES},
        effective_features={name: 0.0 for name in EXPERT_NAMES},
        evidence_strength=1.0,
        diagnostics={},
    )


def _artifact(*, parity: bool, intercept: float = -0.25, positive_authority: bool = False):
    validation = {
        "fit_rows": 600,
        "validation_rows": 240,
        "fold_count": 5,
        "fold_pass_rate": 0.80,
        "prior_brier": 0.220,
        "candidate_brier": 0.210,
        "prior_log_loss": 0.620,
        "candidate_log_loss": 0.605,
        # These claims alone must never grant runtime authority.
        "statistical_gate_passed": True,
        "negative_authority_allowed": True,
    }
    target = {
        "intercept": intercept,
        "coefficients": {name: 0.0 for name in EXPERT_NAMES},
        "feature_means": {name: 0.0 for name in EXPERT_NAMES},
        "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
        "positive_authority": positive_authority,
        "validation": validation,
    }
    return {
        "schema_version": "mlb_game_conditioned_hitter_moe_v2",
        "model_version": "game_conditioned_hitter_moe_v2",
        "training_status": "TEST_FITTED",
        "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        "validation_design": "expanding_window_strictly_prior_dates",
        "train_serve_feature_parity_proven": parity,
        "training_feature_contract": {
            "parity_proven": parity,
            "live_only_features": [] if parity else ["pitch_compatibility"],
        },
        "max_abs_residual_logit": 0.35,
        "targets": {market: dict(target) for market in ("H", "TB", "HR")},
    }


def test_runtime_ignores_forged_gate_when_train_serve_parity_is_missing():
    prior = 0.65
    result = condition_probability(
        prior,
        target="H",
        state=_state(),
        artifact=_artifact(parity=False),
        sequential_uncertainty=0.0,
    )

    assert result.candidate_probability < prior
    assert result.production_probability == prior
    assert result.authority_status == "SHADOW_ONLY_NO_PRODUCTION_AUTHORITY"
    audit = result.validation["independent_authority_audit"]
    assert audit["claimed_negative_authority"] is True
    assert audit["negative_authority_allowed"] is False
    assert "TRAIN_SERVE_FEATURE_PARITY_NOT_PROVEN" in audit["reasons"]


def test_runtime_allows_only_downward_residual_after_independent_negative_gate():
    prior = 0.65
    result = condition_probability(
        prior,
        target="TB",
        state=_state(),
        artifact=_artifact(parity=True, intercept=-0.25),
        sequential_uncertainty=0.0,
    )

    assert result.candidate_probability < prior
    assert result.production_probability < prior
    assert result.production_probability <= result.lower_bound_probability + 1e-12
    assert result.positive_authority is False
    assert result.authority_status == "INDEPENDENTLY_VALIDATED_NEGATIVE_AUTHORITY_ONLY"
    assert result.validation["independent_authority_audit"]["negative_authority_allowed"] is True


def test_runtime_blocks_upward_residual_without_exact_positive_certificate():
    prior = 0.55
    result = condition_probability(
        prior,
        target="HR",
        state=_state(),
        artifact=_artifact(parity=True, intercept=0.25, positive_authority=True),
        sequential_uncertainty=0.0,
    )

    assert result.candidate_probability > prior
    assert result.production_probability == prior
    assert result.positive_authority is False
    audit = result.validation["independent_authority_audit"]
    assert audit["negative_authority_allowed"] is True
    assert audit["positive_authority_allowed"] is False
    assert "EXACT_CERTIFICATION_EVIDENCE_REQUIRED" in audit["reasons"]
