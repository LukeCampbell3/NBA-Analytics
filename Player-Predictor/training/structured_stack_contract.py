#!/usr/bin/env python3
"""Shared artifact-contract helpers for structured stack training/inference."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


ARTIFACT_CONTRACT_VERSION = 2

SCHEMA_KEYS = (
    "feature_columns",
    "feature_spec",
    "baseline_features",
    "target_columns",
    "counts",
    "player_mapping",
    "team_mapping",
    "opponent_mapping",
    "seq_len",
    "n_features",
    "n_targets",
    "scaler_x_n_features",
    "scaler_y_n_features",
    "artifact_contract_version",
    "schema_signature",
)

FEATURE_SPEC_MIN_KEYS = (
    "player_idx",
    "team_idx",
    "opp_idx",
    "pts_idx",
    "trb_idx",
    "ast_idx",
    "pts_lag_idx",
    "trb_lag_idx",
    "ast_lag_idx",
    "pts_roll_idx",
    "trb_roll_idx",
    "ast_roll_idx",
)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def infer_catboost_feature_versions(model_bundle) -> list[str]:
    versions = []
    if isinstance(model_bundle, dict):
        for member in model_bundle.get("members", []):
            if not isinstance(member, dict):
                continue
            value = str(member.get("feature_version", "")).strip()
            if value:
                versions.append(value)
    return _dedupe(versions)


def normalize_catboost_model_info(catboost_model_info, target_columns, cb_models=None):
    target_columns = list(target_columns or [])
    cb_models = list(cb_models or [])
    source_rows = []
    if isinstance(catboost_model_info, list):
        source_rows = [row for row in catboost_model_info if isinstance(row, dict)]
    source_by_target = {
        str(row.get("target")): row
        for row in source_rows
        if str(row.get("target", "")).strip()
    }

    normalized = []
    for idx, target in enumerate(target_columns):
        row = dict(source_by_target.get(target, {}))
        if not row and idx < len(source_rows):
            row = dict(source_rows[idx])
        row["target"] = target

        versions = []
        raw_versions = row.get("feature_versions")
        if isinstance(raw_versions, (list, tuple)):
            versions.extend(str(v).strip() for v in raw_versions if str(v).strip())
        raw_version = str(row.get("feature_version", "")).strip()
        if raw_version:
            versions.append(raw_version)
        if idx < len(cb_models):
            versions.extend(infer_catboost_feature_versions(cb_models[idx]))

        versions = _dedupe(versions)
        if not versions:
            versions = ["v3"]
        row["feature_versions"] = versions
        row["feature_version"] = versions[0]
        normalized.append(row)
    return normalized


def build_schema_signature(
    feature_columns,
    target_columns,
    baseline_features,
    seq_len,
    n_features,
    n_targets,
    counts,
) -> str:
    payload = {
        "feature_columns": list(feature_columns or []),
        "target_columns": list(target_columns or []),
        "baseline_features": list(baseline_features or []),
        "seq_len": _as_int(seq_len, -1),
        "n_features": _as_int(n_features, -1),
        "n_targets": _as_int(n_targets, -1),
        "counts": dict(counts or {}),
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def build_schema_payload(metadata):
    payload = {}
    for key in SCHEMA_KEYS:
        if key in metadata:
            payload[key] = deepcopy(metadata[key])
    payload["artifact_contract_version"] = _as_int(
        payload.get("artifact_contract_version"),
        ARTIFACT_CONTRACT_VERSION,
    )
    if not payload.get("schema_signature"):
        payload["schema_signature"] = build_schema_signature(
            payload.get("feature_columns"),
            payload.get("target_columns"),
            payload.get("baseline_features"),
            payload.get("seq_len"),
            payload.get("n_features"),
            payload.get("n_targets"),
            payload.get("counts"),
        )
    return payload


def apply_schema_sidecar(metadata, schema_payload):
    repaired = dict(metadata or {})
    schema_payload = dict(schema_payload or {})
    for key in SCHEMA_KEYS:
        if key in schema_payload:
            repaired[key] = deepcopy(schema_payload[key])
    return repaired


def validate_metadata_contract(metadata, scaler_x=None, scaler_y=None, cb_models=None):
    errors = []
    if not isinstance(metadata, dict):
        return ["metadata payload is not a JSON object"]

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        errors.append("feature_columns must be a non-empty list")
        feature_columns = []
    baseline_features = metadata.get("baseline_features")
    if not isinstance(baseline_features, list) or not baseline_features:
        errors.append("baseline_features must be a non-empty list")
        baseline_features = []
    target_columns = metadata.get("target_columns")
    if not isinstance(target_columns, list) or not target_columns:
        errors.append("target_columns must be a non-empty list")
        target_columns = []

    n_features = _as_int(metadata.get("n_features"), -1)
    n_targets = _as_int(metadata.get("n_targets"), -1)
    if n_features <= 0:
        errors.append("n_features must be a positive integer")
    if n_targets <= 0:
        errors.append("n_targets must be a positive integer")
    if feature_columns and n_features > 0 and len(feature_columns) != n_features:
        errors.append(
            f"n_features mismatch: metadata={n_features}, feature_columns={len(feature_columns)}"
        )
    if target_columns and n_targets > 0 and len(target_columns) != n_targets:
        errors.append(
            f"n_targets mismatch: metadata={n_targets}, target_columns={len(target_columns)}"
        )
    if baseline_features and target_columns and len(baseline_features) != len(target_columns):
        errors.append(
            f"baseline/target mismatch: baseline_features={len(baseline_features)}, target_columns={len(target_columns)}"
        )

    if len(feature_columns) >= 3:
        expected_cats = ["Player_ID", "Team_ID", "Opponent_ID"]
        if feature_columns[:3] != expected_cats:
            errors.append(f"first 3 feature columns must be {expected_cats}")
    elif feature_columns:
        errors.append("feature_columns must include categorical slots at indices 0..2")

    spec = metadata.get("feature_spec")
    if not isinstance(spec, dict):
        errors.append("feature_spec must be a dict")
        spec = {}
    else:
        for key in FEATURE_SPEC_MIN_KEYS:
            if key not in spec:
                errors.append(f"feature_spec missing required key '{key}'")
        for key, value in spec.items():
            if not str(key).endswith("_idx"):
                continue
            idx = _as_int(value, None)
            if idx is None:
                errors.append(f"feature_spec[{key}] is not an int")
                continue
            if idx < -1:
                errors.append(f"feature_spec[{key}]={idx} must be >= -1")
            if feature_columns and idx >= len(feature_columns):
                errors.append(
                    f"feature_spec[{key}]={idx} out of range for feature_columns={len(feature_columns)}"
                )
        if spec:
            if _as_int(spec.get("player_idx"), -999) != 0:
                errors.append("feature_spec.player_idx must be 0")
            if _as_int(spec.get("team_idx"), -999) != 1:
                errors.append("feature_spec.team_idx must be 1")
            if _as_int(spec.get("opp_idx"), -999) != 2:
                errors.append("feature_spec.opp_idx must be 2")

    counts = metadata.get("counts", {})
    if not isinstance(counts, dict):
        errors.append("counts must be a dict")
        counts = {}
    mapping_specs = [
        ("player_mapping", "players"),
        ("team_mapping", "teams"),
        ("opponent_mapping", "opponents"),
    ]
    for mapping_key, count_key in mapping_specs:
        mapping = metadata.get(mapping_key)
        if not isinstance(mapping, dict) or not mapping:
            errors.append(f"{mapping_key} must be a non-empty dict")
            continue
        expected_count = _as_int(counts.get(count_key), -1)
        if expected_count > 0 and len(mapping) != expected_count:
            errors.append(
                f"{mapping_key} size mismatch: mapping={len(mapping)}, counts.{count_key}={expected_count}"
            )

    if scaler_x is not None:
        scaler_x_features = _as_int(getattr(scaler_x, "n_features_in_", -1), -1)
        if scaler_x_features > 0:
            if feature_columns and len(feature_columns) - 3 != scaler_x_features:
                errors.append(
                    f"scaler_x expects {scaler_x_features} numeric features but metadata implies {len(feature_columns) - 3}"
                )
            if n_features > 0 and n_features != scaler_x_features + 3:
                errors.append(
                    f"n_features mismatch with scaler_x: n_features={n_features}, scaler_x+cats={scaler_x_features + 3}"
                )

    if scaler_y is not None:
        scaler_y_features = _as_int(getattr(scaler_y, "n_features_in_", -1), -1)
        if scaler_y_features > 0:
            if target_columns and len(target_columns) != scaler_y_features:
                errors.append(
                    f"scaler_y expects {scaler_y_features} targets but metadata has {len(target_columns)}"
                )
            if baseline_features and len(baseline_features) != scaler_y_features:
                errors.append(
                    f"scaler_y expects {scaler_y_features} baselines but metadata has {len(baseline_features)}"
                )
            if n_targets > 0 and n_targets != scaler_y_features:
                errors.append(
                    f"n_targets mismatch with scaler_y: n_targets={n_targets}, scaler_y={scaler_y_features}"
                )

    cb_info = metadata.get("catboost_model_info")
    if not isinstance(cb_info, list) or not cb_info:
        errors.append("catboost_model_info must be a non-empty list")
    else:
        if target_columns and len(cb_info) != len(target_columns):
            errors.append(
                f"catboost_model_info length mismatch: info={len(cb_info)}, targets={len(target_columns)}"
            )
        for idx, row in enumerate(cb_info):
            if not isinstance(row, dict):
                errors.append(f"catboost_model_info[{idx}] must be an object")
                continue
            expected_target = target_columns[idx] if idx < len(target_columns) else None
            target = row.get("target")
            if expected_target and target != expected_target:
                errors.append(
                    f"catboost_model_info[{idx}].target mismatch: {target!r} != {expected_target!r}"
                )
            feature_versions = row.get("feature_versions")
            if isinstance(feature_versions, list):
                feature_versions = [str(v).strip() for v in feature_versions if str(v).strip()]
            else:
                feature_versions = []
            if not feature_versions:
                single = str(row.get("feature_version", "")).strip()
                if single:
                    feature_versions = [single]
            if not feature_versions:
                errors.append(f"catboost_model_info[{idx}] missing feature_version(s)")

    if cb_models is not None and target_columns:
        model_count = len(list(cb_models))
        if model_count != len(target_columns):
            errors.append(
                f"catboost model bundle count mismatch: models={model_count}, targets={len(target_columns)}"
            )

    signature = str(metadata.get("schema_signature", "")).strip()
    if signature:
        expected = build_schema_signature(
            feature_columns,
            target_columns,
            baseline_features,
            metadata.get("seq_len"),
            n_features,
            n_targets,
            counts,
        )
        if signature != expected:
            errors.append("schema_signature does not match metadata payload")

    return _dedupe(errors)
