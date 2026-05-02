from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalize_month_token(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    return ""


def _segment_key(target: str | None, direction: str | None) -> str:
    return f"{str(target or '').strip().upper()}|{str(direction or '').strip().upper()}"


def _safe_threshold(value: Any, default: float = float("-inf")) -> float:
    try:
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def resolve_month_payload(payload: dict[str, Any] | None, run_date_hint: str | None = None) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return "", {}
    months = payload.get("months", {})
    if not isinstance(months, dict) or not months:
        return "", {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, month_payload in months.items():
        key = _normalize_month_token(str(raw_key))
        if key and isinstance(month_payload, dict):
            normalized[key] = month_payload
    if not normalized:
        return "", {}

    requested = _normalize_month_token(run_date_hint)
    keys = sorted(normalized.keys())
    if requested:
        if requested in normalized:
            return requested, normalized[requested]
        prior = [key for key in keys if key <= requested]
        if prior:
            key = prior[-1]
            return key, normalized[key]
        future = [key for key in keys if key > requested]
        if future:
            key = future[0]
            return key, normalized[key]
        return "", {}

    key = keys[-1]
    return key, normalized[key]


def apply_learned_pool_gate(
    frame: pd.DataFrame,
    payload: dict[str, Any] | None,
    run_date_hint: str | None = None,
    prob_col: str = "expected_win_rate",
    target_col: str = "target",
    direction_col: str = "direction",
) -> tuple[pd.Series, pd.Series, pd.Series, str, dict[str, Any]]:
    index = frame.index
    probs = pd.to_numeric(frame.get(prob_col), errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    pass_all = pd.Series(True, index=index, dtype=bool)
    no_threshold = pd.Series(float("-inf"), index=index, dtype="float64")
    identity_source = pd.Series("identity", index=index, dtype="object")

    if not isinstance(payload, dict):
        return pass_all, no_threshold, identity_source, "", {"enabled": False, "reason": "missing_payload"}

    month_key, month_payload = resolve_month_payload(payload, run_date_hint=run_date_hint)
    if not month_payload:
        return pass_all, no_threshold, identity_source, month_key, {"enabled": False, "reason": "missing_month_payload", "month": month_key}

    global_payload = month_payload.get("global", {})
    if not isinstance(global_payload, dict):
        global_payload = {}
    global_threshold = _safe_threshold(global_payload.get("threshold"), default=float("-inf"))

    thresholds = pd.Series(global_threshold, index=index, dtype="float64")
    threshold_source = pd.Series("global", index=index, dtype="object")
    segments = month_payload.get("segments", {})
    if isinstance(segments, dict) and segments:
        targets = frame.get(target_col, pd.Series("", index=index)).astype(str).str.upper().str.strip()
        directions = frame.get(direction_col, pd.Series("", index=index)).astype(str).str.upper().str.strip()
        keys = targets + "|" + directions
        for key, segment_payload in segments.items():
            if not isinstance(segment_payload, dict):
                continue
            segment_threshold = _safe_threshold(segment_payload.get("threshold"), default=np.nan)
            if not np.isfinite(segment_threshold):
                continue
            norm_key = _segment_key(*str(key).split("|", 1)) if "|" in str(key) else str(key).strip().upper()
            mask = keys.eq(norm_key)
            if not mask.any():
                continue
            thresholds.loc[mask] = float(segment_threshold)
            threshold_source.loc[mask] = f"segment:{norm_key}"

    pass_mask = probs >= thresholds
    details = {
        "enabled": True,
        "month": month_key,
        "rows": int(len(frame)),
        "pass_rows": int(pass_mask.sum()),
        "pass_rate": float(pass_mask.mean()) if len(pass_mask) else np.nan,
        "global_threshold": float(global_threshold) if np.isfinite(global_threshold) else None,
    }
    return pass_mask.astype(bool), thresholds.astype("float64"), threshold_source, month_key, details

