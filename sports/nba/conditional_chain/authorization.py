from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

import numpy as np
import pandas as pd

from .protocol import PARLAY_AUTHORIZATION_PROTOCOL, ParlayAuthorizationProtocol


ACTIVE_CERTIFICATE_STATUS = "ACTIVE"
SUPPORTED_PATH_STATUS = "PATH_INCREMENTAL_VALUE_SUPPORTED"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class QuoteEvidenceStatus(str, Enum):
    VERIFIED_EXECUTABLE_QUOTE = "VERIFIED_EXECUTABLE_QUOTE"
    INCOMPLETE_IDENTITY = "INCOMPLETE_IDENTITY"
    INVALID_LINE = "INVALID_LINE"
    INVALID_PRICE = "INVALID_PRICE"
    MISSING_BOOK = "MISSING_BOOK"
    MISSING_SOURCE_PROVENANCE = "MISSING_SOURCE_PROVENANCE"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    POST_START_QUOTE = "POST_START_QUOTE"
    FUTURE_QUOTE = "FUTURE_QUOTE"
    STALE_QUOTE = "STALE_QUOTE"


@dataclass(frozen=True)
class ParlayAuthorization:
    authorized: bool
    status: str
    reasons: tuple[str, ...]
    quote_status_counts: dict[str, int]
    policy_version: str
    certificate_id: str | None
    staking_enabled: bool


def _text_present(value: object) -> bool:
    return bool(pd.notna(value) and str(value).strip())


def _valid_sha256(value: object) -> bool:
    return bool(_text_present(value) and SHA256_PATTERN.fullmatch(str(value).lower()))


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scope_values(value: object) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value}


def assess_quote_evidence(
    candidates: pd.DataFrame,
    *,
    qualification_time: str | pd.Timestamp,
    protocol: ParlayAuthorizationProtocol = PARLAY_AUTHORIZATION_PROTOCOL,
) -> pd.DataFrame:
    """Validate that each candidate points to a fresh, auditable book quote."""

    required = {
        "event_id",
        "event_start_time_utc",
        "snapshot_time_utc",
        "player",
        "market",
        "side",
        "line",
        "book",
        "decimal_odds",
        "source",
        "raw_source_hash",
        "parser_version",
    }
    frame = candidates.copy()
    for column in required - set(frame.columns):
        frame[column] = pd.NA

    qualified_at = pd.Timestamp(qualification_time)
    qualified_at = (
        qualified_at.tz_localize("UTC")
        if qualified_at.tzinfo is None
        else qualified_at.tz_convert("UTC")
    )
    frame["_snapshot_time"] = pd.to_datetime(
        frame["snapshot_time_utc"], errors="coerce", utc=True
    )
    frame["_event_start"] = pd.to_datetime(
        frame["event_start_time_utc"], errors="coerce", utc=True
    )
    frame["quote_age_minutes"] = (
        qualified_at - frame["_snapshot_time"]
    ).dt.total_seconds() / 60.0

    statuses: list[str] = []
    for _, row in frame.iterrows():
        identity = all(
            _text_present(row[column])
            for column in ("event_id", "player", "market", "side")
        ) and str(row["side"]).upper() in {"OVER", "UNDER"}
        line = pd.to_numeric(pd.Series([row["line"]]), errors="coerce").iloc[0]
        price = pd.to_numeric(pd.Series([row["decimal_odds"]]), errors="coerce").iloc[0]
        if not identity:
            status = QuoteEvidenceStatus.INCOMPLETE_IDENTITY
        elif not np.isfinite(line) or float(line) <= 0.0:
            status = QuoteEvidenceStatus.INVALID_LINE
        elif (
            not np.isfinite(price)
            or float(price) < protocol.minimum_leg_decimal_odds
            or float(price) > protocol.maximum_leg_decimal_odds
        ):
            status = QuoteEvidenceStatus.INVALID_PRICE
        elif not _text_present(row["book"]):
            status = QuoteEvidenceStatus.MISSING_BOOK
        elif not (
            _text_present(row["source"])
            and _valid_sha256(row["raw_source_hash"])
            and _text_present(row["parser_version"])
        ):
            status = QuoteEvidenceStatus.MISSING_SOURCE_PROVENANCE
        elif pd.isna(row["_snapshot_time"]) or pd.isna(row["_event_start"]):
            status = QuoteEvidenceStatus.INVALID_TIMESTAMP
        elif row["_snapshot_time"] >= row["_event_start"]:
            status = QuoteEvidenceStatus.POST_START_QUOTE
        elif float(row["quote_age_minutes"]) < 0.0:
            status = QuoteEvidenceStatus.FUTURE_QUOTE
        elif float(row["quote_age_minutes"]) > protocol.maximum_quote_age_minutes:
            status = QuoteEvidenceStatus.STALE_QUOTE
        else:
            status = QuoteEvidenceStatus.VERIFIED_EXECUTABLE_QUOTE
        statuses.append(status.value)

    frame["quote_evidence_status"] = statuses
    frame["odds_validated_as_true"] = frame["quote_evidence_status"].eq(
        QuoteEvidenceStatus.VERIFIED_EXECUTABLE_QUOTE.value
    )
    return frame.drop(columns=["_snapshot_time", "_event_start"])


def authorize_parlay(
    candidates: pd.DataFrame,
    *,
    qualification_time: str | pd.Timestamp,
    active_policy_version: str,
    policy_certificate: dict[str, Any] | None,
    path_certificate: dict[str, Any] | None,
    protocol: ParlayAuthorizationProtocol = PARLAY_AUTHORIZATION_PROTOCOL,
) -> ParlayAuthorization:
    """Authorize a candidate set only through exact policy-level evidence."""

    reasons: list[str] = []
    certificate = policy_certificate or {}
    certificate_scope = certificate.get("scope", {})
    expected_leg_count = _safe_int(
        certificate_scope.get("leg_count"), protocol.parlay_legs
    )
    if expected_leg_count not in protocol.allowed_leg_counts:
        reasons.append("UNSUPPORTED_LEG_COUNT_SCOPE")
    if len(candidates) != expected_leg_count:
        reasons.append("WRONG_LEG_COUNT")
    if "player" not in candidates or candidates.get(
        "player", pd.Series(dtype=object)
    ).nunique() != len(candidates):
        reasons.append("DUPLICATE_PLAYER_EXPOSURE")

    policy_versions = (
        set(candidates["policy_version"].dropna().astype(str))
        if "policy_version" in candidates
        else set()
    )
    if policy_versions != {active_policy_version}:
        reasons.append("POLICY_VERSION_MISMATCH")

    if certificate.get("certificate_status") != ACTIVE_CERTIFICATE_STATUS:
        reasons.append("POLICY_CERTIFICATE_NOT_ACTIVE")
    if certificate.get("policy_version") != active_policy_version:
        reasons.append("CERTIFICATE_POLICY_VERSION_MISMATCH")
    if certificate_scope.get("league") != protocol.league:
        reasons.append("CERTIFICATE_LEAGUE_SCOPE_MISMATCH")
    evidence = certificate.get("evidence", {})
    if (
        _safe_int(evidence.get("resolved_action_slates"), 0)
        < protocol.minimum_resolved_action_slates
    ):
        reasons.append("INSUFFICIENT_RESOLVED_ACTION_SLATES")
    minimum_selections = (
        protocol.minimum_resolved_selections_per_leg * expected_leg_count
    )
    if _safe_int(evidence.get("resolved_selections"), 0) < minimum_selections:
        reasons.append("INSUFFICIENT_RESOLVED_SELECTIONS")
    if (
        _safe_float(evidence.get("slate_coverage"), 0.0)
        < protocol.minimum_slate_coverage
    ):
        reasons.append("INSUFFICIENT_SLATE_COVERAGE")
    if not bool(certificate.get("eligible_for_candidate_authorization", False)):
        reasons.append("CERTIFICATE_AUTHORIZATION_DISABLED")

    evaluation = certificate.get("evaluation", {})
    deployment_margin = _safe_float(
        evaluation.get("deployment_margin"), protocol.minimum_deployment_margin
    )
    return_lcb = _safe_float(evaluation.get("anytime_valid_return_lcb"), float("-inf"))
    if (
        deployment_margin < protocol.minimum_deployment_margin
        or return_lcb <= deployment_margin
    ):
        reasons.append("RETURN_LOWER_BOUND_BELOW_DEPLOYMENT_MARGIN")
    if certificate.get("support", {}).get("current_status") != "IN_SUPPORT":
        reasons.append("CANDIDATE_SUPPORT_NOT_CONFIRMED")
    if certificate.get("shift", {}).get("current_status") not in {
        "STABLE",
        "TOLERABLE",
    }:
        reasons.append("DISTRIBUTION_SHIFT_NOT_TOLERABLE")

    path = path_certificate or {}
    if protocol.require_path_certificate and not (
        path.get("status") == SUPPORTED_PATH_STATUS
        and bool(path.get("path_authorized", False))
    ):
        reasons.append("PATH_INCREMENTAL_VALUE_NOT_CERTIFIED")

    candidate_requirements = {
        "lineup_state": "CONFIRMED",
        "player_state": "ACTIVE",
        "identity_status": "MATCHED",
        "support_status": "IN_SUPPORT",
        "exposure_status": "PASS",
    }
    for column, expected in candidate_requirements.items():
        if (
            column not in candidates
            or not candidates[column].astype(str).eq(expected).all()
        ):
            reasons.append(f"CANDIDATE_{column.upper()}_GATE_FAILED")
    if (
        "eligible_by_input_rules" not in candidates
        or not candidates["eligible_by_input_rules"].fillna(False).astype(bool).all()
    ):
        reasons.append("CANDIDATE_INPUT_ELIGIBILITY_GATE_FAILED")

    expected_model_version = str(certificate_scope.get("model_version", ""))
    candidate_model_versions = (
        set(candidates["model_version"].dropna().astype(str))
        if "model_version" in candidates
        else set()
    )
    if not expected_model_version or candidate_model_versions != {
        expected_model_version
    }:
        reasons.append("MODEL_VERSION_OUTSIDE_CERTIFICATE_SCOPE")
    expected_representation = str(path.get("representation_version", ""))
    candidate_representations = (
        set(candidates["path_representation_version"].dropna().astype(str))
        if "path_representation_version" in candidates
        else set()
    )
    if not expected_representation or candidate_representations != {
        expected_representation
    }:
        reasons.append("PATH_REPRESENTATION_VERSION_MISMATCH")

    feature_cutoffs = pd.to_datetime(
        candidates.get("feature_cutoff_utc", pd.Series(pd.NaT, index=candidates.index)),
        errors="coerce",
        utc=True,
    )
    qualified_at = pd.Timestamp(qualification_time)
    qualified_at = (
        qualified_at.tz_localize("UTC")
        if qualified_at.tzinfo is None
        else qualified_at.tz_convert("UTC")
    )
    if bool(feature_cutoffs.isna().any() or feature_cutoffs.gt(qualified_at).any()):
        reasons.append("FEATURE_CUTOFF_GATE_FAILED")

    assessed = assess_quote_evidence(
        candidates,
        qualification_time=qualification_time,
        protocol=protocol,
    )
    quote_counts = {
        str(key): int(value)
        for key, value in assessed["quote_evidence_status"]
        .value_counts()
        .to_dict()
        .items()
    }
    if not bool(assessed["odds_validated_as_true"].all()):
        reasons.append("UNVERIFIED_OR_UNEXECUTABLE_QUOTE")
    allowed_markets = _scope_values(certificate_scope.get("markets"))
    if not allowed_markets or not set(assessed["market"].astype(str)).issubset(
        allowed_markets
    ):
        reasons.append("MARKET_OUTSIDE_CERTIFICATE_SCOPE")
    allowed_books = _scope_values(certificate_scope.get("books"))
    if not allowed_books or not set(assessed["book"].astype(str)).issubset(
        allowed_books
    ):
        reasons.append("BOOK_OUTSIDE_CERTIFICATE_SCOPE")
    scope_min_odds = _safe_float(
        certificate_scope.get("minimum_decimal_odds"), protocol.minimum_leg_decimal_odds
    )
    scope_max_odds = _safe_float(
        certificate_scope.get("maximum_decimal_odds"), protocol.maximum_leg_decimal_odds
    )
    prices = pd.to_numeric(assessed["decimal_odds"], errors="coerce")
    if bool((prices.lt(scope_min_odds) | prices.gt(scope_max_odds)).any()):
        reasons.append("PRICE_OUTSIDE_CERTIFICATE_SCOPE")

    unique_reasons = tuple(dict.fromkeys(reasons))
    authorized = not unique_reasons
    return ParlayAuthorization(
        authorized=authorized,
        status="AUTHORIZED" if authorized else "REJECTED",
        reasons=unique_reasons,
        quote_status_counts=quote_counts,
        policy_version=active_policy_version,
        certificate_id=(
            str(certificate["certificate_id"])
            if certificate.get("certificate_id") is not None
            else None
        ),
        staking_enabled=bool(authorized and protocol.staking_enabled),
    )
