"""Independent promotion/authority checks for the game-conditioned hitter MoE.

The runtime scorer intentionally consumes a compact model artifact. This module
recomputes whether an artifact is *allowed* to move the baseline probability
rather than trusting booleans embedded in that artifact.

Negative authority means the residual adapter may lower the baseline when
rolling-origin validation demonstrates repeatable out-of-sample improvement
*and* the feature contract used in validation matches the live scoring path.

Positive authority is deliberately harder: it additionally requires exact,
point-in-time certification evidence. Diagnostic/reconstructed evidence can
never grant it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple


TARGETS: Tuple[str, ...] = ("H", "TB", "HR")
DEFAULT_MIN_FIT_ROWS = 300
DEFAULT_MIN_VALIDATION_ROWS = 50
DEFAULT_MIN_FOLDS = 3
DEFAULT_MIN_FOLD_PASS_RATE = 0.60
DEFAULT_MIN_BRIER_GAIN = 0.0025
DEFAULT_MIN_LOGLOSS_GAIN = 0.0025
DEFAULT_MIN_EXACT_SELECTIONS = 50
DEFAULT_MIN_INDEPENDENT_SLATES = 20


@dataclass(frozen=True)
class AuthorityDecision:
    target: str
    negative_authority_allowed: bool
    positive_authority_allowed: bool
    claimed_negative_authority: bool
    claimed_positive_authority: bool
    reasons: Tuple[str, ...]
    checks: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _bool(value: Any) -> bool:
    return value is True


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validation_payload(target_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = target_payload.get("validation")
    if isinstance(validation, Mapping):
        return validation
    gate = target_payload.get("statistical_gate")
    if isinstance(gate, Mapping):
        return gate
    return {}


def _threshold(report: Mapping[str, Any], name: str, default: float) -> float:
    controls = _mapping(report.get("controls"))
    value = _finite(controls.get(name))
    return default if value is None else value


def _exact_certificate(
    report: Mapping[str, Any], target_payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    for key in ("exact_certification", "certification", "promotion_evidence"):
        nested = target_payload.get(key)
        if isinstance(nested, Mapping):
            return nested
        nested = report.get(key)
        if isinstance(nested, Mapping):
            return nested
    return {}


def _is_exact_evidence(report: Mapping[str, Any], cert: Mapping[str, Any]) -> bool:
    evidence = str(
        cert.get("evidence_class")
        or report.get("evidence_class")
        or ""
    ).upper()
    return (
        "EXACT" in evidence
        and "DIAGNOSTIC" not in evidence
        and "NOT_CERTIFICATION" not in evidence
    )


def _train_serve_parity(report: Mapping[str, Any]) -> tuple[bool, Mapping[str, Any]]:
    """Return whether validation and live scoring use the same feature contract.

    This is intentionally fail-closed. A historical fit that omits live-only
    pitch compatibility, direct matchup, handedness interactions, chase/EV
    state, or arsenal-velocity state cannot authorize a live residual merely
    because aggregate validation metrics looked favorable.
    """

    contract = _mapping(report.get("training_feature_contract"))
    explicit = report.get("train_serve_feature_parity_proven")
    if explicit is None:
        explicit = contract.get("parity_proven")
    return _bool(explicit), contract


def validate_target_authority(
    report: Mapping[str, Any],
    target: str,
) -> AuthorityDecision:
    """Recompute authority for one market target from auditable metrics.

    This function is fail-closed. Missing validation metrics or feature-parity
    evidence do not count as evidence. It supports both the committed
    validation-report shape and the compact runtime model-artifact shape.
    """

    target = str(target).upper()
    target_payload = _mapping(_mapping(report.get("targets")).get(target))
    validation = _validation_payload(target_payload)
    cert = _exact_certificate(report, target_payload)
    parity_proven, feature_contract = _train_serve_parity(report)

    min_fit_rows = int(_threshold(report, "min_rows_to_fit", DEFAULT_MIN_FIT_ROWS))
    min_validation_rows = int(
        _threshold(report, "min_validation_rows", DEFAULT_MIN_VALIDATION_ROWS)
    )
    min_folds = int(_threshold(report, "min_folds", DEFAULT_MIN_FOLDS))
    min_fold_pass_rate = _threshold(
        report, "gate_min_fold_pass_rate", DEFAULT_MIN_FOLD_PASS_RATE
    )
    min_brier_gain = _threshold(
        report, "gate_min_brier_gain", DEFAULT_MIN_BRIER_GAIN
    )
    min_logloss_gain = _threshold(
        report, "gate_min_logloss_gain", DEFAULT_MIN_LOGLOSS_GAIN
    )

    prior_brier = _finite(validation.get("prior_brier"))
    candidate_brier = _finite(validation.get("candidate_brier"))
    prior_logloss = _finite(validation.get("prior_log_loss"))
    candidate_logloss = _finite(validation.get("candidate_log_loss"))
    fit_rows = int(_finite(validation.get("fit_rows")) or 0)
    validation_rows = int(_finite(validation.get("validation_rows")) or 0)
    fold_count = int(_finite(validation.get("fold_count")) or 0)
    fold_pass_rate = _finite(validation.get("fold_pass_rate"))

    brier_gain = (
        prior_brier - candidate_brier
        if prior_brier is not None and candidate_brier is not None
        else None
    )
    logloss_gain = (
        prior_logloss - candidate_logloss
        if prior_logloss is not None and candidate_logloss is not None
        else None
    )

    design = str(report.get("validation_design") or validation.get("validation_design") or "")
    rolling_origin = (
        "strictly_prior" in design.lower()
        or "rolling_origin" in design.lower()
        or "expanding_window" in design.lower()
    )

    checks: Dict[str, Any] = {
        "fit_rows": fit_rows,
        "validation_rows": validation_rows,
        "fold_count": fold_count,
        "fold_pass_rate": fold_pass_rate,
        "brier_gain": brier_gain,
        "logloss_gain": logloss_gain,
        "rolling_origin_design": rolling_origin,
        "train_serve_feature_parity_proven": parity_proven,
        "training_feature_contract": dict(feature_contract),
        "thresholds": {
            "min_fit_rows": min_fit_rows,
            "min_validation_rows": min_validation_rows,
            "min_folds": min_folds,
            "min_fold_pass_rate": min_fold_pass_rate,
            "min_brier_gain": min_brier_gain,
            "min_logloss_gain": min_logloss_gain,
        },
    }

    negative_failures = []
    if not target_payload:
        negative_failures.append("TARGET_PAYLOAD_MISSING")
    if fit_rows < min_fit_rows:
        negative_failures.append("INSUFFICIENT_FIT_ROWS")
    if validation_rows < min_validation_rows:
        negative_failures.append("INSUFFICIENT_VALIDATION_ROWS")
    if fold_count < min_folds:
        negative_failures.append("INSUFFICIENT_ROLLING_FOLDS")
    if fold_pass_rate is None or fold_pass_rate < min_fold_pass_rate:
        negative_failures.append("FOLD_PASS_RATE_BELOW_GATE")
    if brier_gain is None or brier_gain < min_brier_gain:
        negative_failures.append("BRIER_GAIN_BELOW_GATE")
    if logloss_gain is None or logloss_gain < min_logloss_gain:
        negative_failures.append("LOGLOSS_GAIN_BELOW_GATE")
    if not rolling_origin:
        negative_failures.append("STRICT_PRIOR_DATE_VALIDATION_NOT_PROVEN")
    if not parity_proven:
        negative_failures.append("TRAIN_SERVE_FEATURE_PARITY_NOT_PROVEN")

    negative_allowed = not negative_failures

    claimed_negative = _bool(validation.get("negative_authority_allowed")) or _bool(
        validation.get("statistical_gate_passed")
    )
    claimed_positive = _bool(target_payload.get("positive_authority"))

    exact_selections = int(
        _finite(cert.get("exact_selection_count") or cert.get("selection_count")) or 0
    )
    independent_slates = int(_finite(cert.get("independent_slates")) or 0)
    pit_integrity = _bool(
        cert.get("point_in_time_integrity_passed")
        if "point_in_time_integrity_passed" in cert
        else cert.get("pit_integrity_passed")
    )
    locked_policy_hash = str(
        cert.get("locked_policy_hash")
        or cert.get("policy_hash")
        or cert.get("model_hash")
        or ""
    ).strip()
    exact_evidence = _is_exact_evidence(report, cert)

    checks["exact_certification"] = {
        "exact_evidence": exact_evidence,
        "exact_selection_count": exact_selections,
        "independent_slates": independent_slates,
        "point_in_time_integrity_passed": pit_integrity,
        "locked_policy_hash_present": bool(locked_policy_hash),
        "thresholds": {
            "min_exact_selections": DEFAULT_MIN_EXACT_SELECTIONS,
            "min_independent_slates": DEFAULT_MIN_INDEPENDENT_SLATES,
        },
    }

    positive_failures = []
    if not negative_allowed:
        positive_failures.append("NEGATIVE_AUTHORITY_GATE_NOT_CLEARED")
    if not exact_evidence:
        positive_failures.append("EXACT_CERTIFICATION_EVIDENCE_REQUIRED")
    if exact_selections < DEFAULT_MIN_EXACT_SELECTIONS:
        positive_failures.append("INSUFFICIENT_EXACT_SELECTIONS")
    if independent_slates < DEFAULT_MIN_INDEPENDENT_SLATES:
        positive_failures.append("INSUFFICIENT_INDEPENDENT_SLATES")
    if not pit_integrity:
        positive_failures.append("POINT_IN_TIME_INTEGRITY_NOT_PROVEN")
    if not locked_policy_hash:
        positive_failures.append("LOCKED_POLICY_HASH_MISSING")

    positive_allowed = claimed_positive and not positive_failures

    reasons = tuple(negative_failures + positive_failures)
    if negative_allowed and not claimed_positive:
        reasons = reasons + ("POSITIVE_AUTHORITY_NOT_CLAIMED",)

    return AuthorityDecision(
        target=target,
        negative_authority_allowed=negative_allowed,
        positive_authority_allowed=positive_allowed,
        claimed_negative_authority=claimed_negative,
        claimed_positive_authority=claimed_positive,
        reasons=reasons,
        checks=checks,
    )


def audit_authority_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Audit all market targets and report contradictory authority claims."""

    decisions = {
        target: validate_target_authority(report, target).to_dict()
        for target in TARGETS
    }
    violations = []
    for target, decision in decisions.items():
        if decision["claimed_negative_authority"] and not decision[
            "negative_authority_allowed"
        ]:
            violations.append(f"{target}:INVALID_NEGATIVE_AUTHORITY_CLAIM")
        if decision["claimed_positive_authority"] and not decision[
            "positive_authority_allowed"
        ]:
            violations.append(f"{target}:INVALID_POSITIVE_AUTHORITY_CLAIM")

    return {
        "schema_version": "mlb_game_conditioned_authority_audit_v2",
        "valid": not violations,
        "violations": violations,
        "targets": decisions,
    }
