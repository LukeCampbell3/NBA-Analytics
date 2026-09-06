import json
from pathlib import Path

from sports.mlb.advanced.game_conditioned_authority import (
    audit_authority_report,
    validate_target_authority,
)


ROOT = Path(__file__).resolve().parents[3]


def _valid_target(*, positive_authority=False):
    folds = [
        {
            "fold": index + 1,
            "both_improved": True,
            "prior_brier": 0.20,
            "candidate_brier": 0.19,
            "prior_log_loss": 0.50,
            "candidate_log_loss": 0.48,
        }
        for index in range(5)
    ]
    return {
        "positive_authority": positive_authority,
        "validation": {
            "fit_rows": 300,
            "validation_rows": 250,
            "fold_count": 5,
            "fold_pass_rate": 1.0,
            "folds": folds,
            "prior_brier": 0.20,
            "candidate_brier": 0.19,
            "prior_log_loss": 0.50,
            "candidate_log_loss": 0.48,
            "negative_authority_allowed": True,
            "statistical_gate_passed": True,
        },
    }


def _report(target_payload=None):
    target_payload = target_payload or _valid_target()
    return {
        "validation_design": "expanding_window_strictly_prior_dates",
        "evidence_class": "ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION",
        "train_serve_feature_parity_proven": True,
        "training_feature_contract": {
            "parity_proven": True,
            "status": "TEST_PARITY_PROVEN",
        },
        "targets": {
            "H": target_payload,
            "TB": _valid_target(),
            "HR": _valid_target(),
        },
    }


def test_negative_authority_is_recomputed_not_trusted_from_boolean():
    payload = _valid_target()
    payload["validation"]["candidate_brier"] = 0.23
    payload["validation"]["candidate_log_loss"] = 0.55

    decision = validate_target_authority(_report(payload), "H")

    assert decision.claimed_negative_authority is True
    assert decision.negative_authority_allowed is False
    assert "BRIER_GAIN_BELOW_GATE" in decision.reasons
    assert "LOGLOSS_GAIN_BELOW_GATE" in decision.reasons


def test_negative_authority_requires_repeatable_fold_performance():
    payload = _valid_target()
    payload["validation"]["fold_pass_rate"] = 0.40

    decision = validate_target_authority(_report(payload), "H")

    assert decision.negative_authority_allowed is False
    assert "FOLD_PASS_RATE_BELOW_GATE" in decision.reasons


def test_negative_authority_requires_strict_prior_date_validation_design():
    report = _report()
    report["validation_design"] = "random_split"

    decision = validate_target_authority(report, "TB")

    assert decision.negative_authority_allowed is False
    assert "STRICT_PRIOR_DATE_VALIDATION_NOT_PROVEN" in decision.reasons


def test_negative_authority_requires_train_serve_feature_parity():
    report = _report()
    report["train_serve_feature_parity_proven"] = False
    report["training_feature_contract"] = {
        "parity_proven": False,
        "live_only_features": ["pitch_compatibility"],
    }

    decision = validate_target_authority(report, "TB")

    assert decision.negative_authority_allowed is False
    assert "TRAIN_SERVE_FEATURE_PARITY_NOT_PROVEN" in decision.reasons
    assert decision.checks["train_serve_feature_parity_proven"] is False


def test_positive_authority_is_blocked_without_exact_pit_certificate():
    payload = _valid_target(positive_authority=True)
    report = _report(payload)

    decision = validate_target_authority(report, "H")

    assert decision.negative_authority_allowed is True
    assert decision.positive_authority_allowed is False
    assert "EXACT_CERTIFICATION_EVIDENCE_REQUIRED" in decision.reasons
    assert "POINT_IN_TIME_INTEGRITY_NOT_PROVEN" in decision.reasons


def test_positive_authority_accepts_locked_exact_pit_evidence():
    payload = _valid_target(positive_authority=True)
    payload["exact_certification"] = {
        "evidence_class": "EXACT_CERTIFICATION_ELIGIBLE",
        "exact_selection_count": 50,
        "independent_slates": 20,
        "point_in_time_integrity_passed": True,
        "locked_policy_hash": "frozen-model-policy-sha256",
    }
    report = _report(payload)
    report["evidence_class"] = "EXACT_CERTIFICATION_ELIGIBLE"

    decision = validate_target_authority(report, "H")

    assert decision.negative_authority_allowed is True
    assert decision.positive_authority_allowed is True
    assert decision.claimed_positive_authority is True


def test_audit_detects_forged_authority_claim():
    payload = _valid_target()
    payload["validation"]["candidate_brier"] = 0.22
    report = _report(payload)

    audit = audit_authority_report(report)

    assert audit["valid"] is False
    assert "H:INVALID_NEGATIVE_AUTHORITY_CLAIM" in audit["violations"]


def test_audit_detects_authority_claim_when_live_features_were_not_trained():
    report = _report()
    report["train_serve_feature_parity_proven"] = False
    report["training_feature_contract"] = {
        "parity_proven": False,
        "live_only_features": [
            "pitch_compatibility",
            "direct_matchup",
            "handedness_interaction",
        ],
    }

    audit = audit_authority_report(report)

    assert audit["valid"] is False
    assert set(audit["violations"]) == {
        "H:INVALID_NEGATIVE_AUTHORITY_CLAIM",
        "TB:INVALID_NEGATIVE_AUTHORITY_CLAIM",
        "HR:INVALID_NEGATIVE_AUTHORITY_CLAIM",
    }


def test_committed_validation_report_is_fail_closed_and_claim_consistent():
    path = ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    audit = audit_authority_report(report)

    # The broad committed fit currently has no production residual authority.
    # This test intentionally avoids pinning a transient target result; it checks
    # that any claim present in the artifact is independently supportable.
    assert audit["valid"] is True
    for target in ("H", "TB", "HR"):
        decision = audit["targets"][target]
        validation = report["targets"][target]["validation"]
        claimed = bool(
            validation.get("negative_authority_allowed", False)
            or validation.get("statistical_gate_passed", False)
        )
        assert decision["claimed_negative_authority"] is claimed
        assert decision["claimed_positive_authority"] is bool(
            report["targets"][target].get("positive_authority", False)
        )
        assert decision["positive_authority_allowed"] is False
