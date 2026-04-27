from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .learned_pool_gate import apply_learned_pool_gate, resolve_month_payload


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAYLOAD_CANDIDATES = (
    PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate_pooltrained_tmp.json",
    PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
)


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def _clip01(values: pd.Series | np.ndarray | float) -> pd.Series:
    return pd.Series(values).clip(lower=0.0, upper=1.0)


def _normalize_probability(values: pd.Series) -> pd.Series:
    return ((pd.to_numeric(values, errors="coerce").fillna(0.5) - 0.50) / 0.30).clip(lower=0.0, upper=1.0)


def _segment_key(target: pd.Series, direction: pd.Series) -> pd.Series:
    return target.astype(str).str.upper().str.strip() + "|" + direction.astype(str).str.upper().str.strip()


def infer_run_date_hint(frame: pd.DataFrame) -> str:
    for column in ("run_date", "market_date", "target_date"):
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().any():
            return str(parsed.max().strftime("%Y-%m-%d"))
    return ""


@lru_cache(maxsize=8)
def _load_cached_payload(path_text: str) -> dict[str, Any] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_learned_gate_payload(config: Any) -> dict[str, Any] | None:
    inline_payload = getattr(config, "learned_gate_payload", None)
    if isinstance(inline_payload, dict):
        return inline_payload

    candidates: list[Path] = []
    raw_path = str(getattr(config, "learned_gate_payload_path", "") or "").strip()
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PLAYER_PREDICTOR_ROOT / candidate
        candidates.append(candidate)
    candidates.extend(DEFAULT_PAYLOAD_CANDIDATES)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = _load_cached_payload(resolved)
        if isinstance(payload, dict):
            return payload
    return None


def annotate_final_pool_quality(
    frame: pd.DataFrame,
    *,
    payload: dict[str, Any] | None,
    run_date_hint: str | None = None,
    probability_col: str = "expected_win_rate",
    target_col: str = "target",
    direction_col: str = "direction",
    belief_uncertainty_lower: float = 0.75,
    belief_uncertainty_upper: float = 1.15,
    near_miss_margin: float = 0.003,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    run_date_hint = str(run_date_hint or infer_run_date_hint(out))

    pass_mask, thresholds, sources, month_key, details = apply_learned_pool_gate(
        out,
        payload=payload,
        run_date_hint=run_date_hint,
        prob_col=probability_col,
        target_col=target_col,
        direction_col=direction_col,
    )
    out["learned_gate_enabled"] = bool(details.get("enabled", False))
    out["learned_gate_pass"] = pd.to_numeric(pass_mask, errors="coerce").fillna(0).astype(bool)
    out["learned_gate_threshold"] = pd.to_numeric(thresholds, errors="coerce").fillna(float("-inf"))
    out["learned_gate_source"] = sources.astype(str)
    out["learned_gate_month"] = str(month_key or "")

    probabilities = pd.to_numeric(out.get(probability_col), errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    finite_threshold = np.isfinite(pd.to_numeric(out["learned_gate_threshold"], errors="coerce"))
    raw_margin = probabilities - pd.to_numeric(out["learned_gate_threshold"], errors="coerce").where(finite_threshold, probabilities)
    out["learned_gate_margin"] = raw_margin.astype("float64")

    month_payload = resolve_month_payload(payload, run_date_hint=run_date_hint)[1] if isinstance(payload, dict) else {}
    global_payload = month_payload.get("global", {}) if isinstance(month_payload, dict) else {}
    try:
        global_hit_rate = float(global_payload.get("hit_rate"))
    except Exception:
        global_hit_rate = np.nan

    segment_hit_map: dict[str, float] = {}
    segment_lift_map: dict[str, float] = {}
    segments = month_payload.get("segments", {}) if isinstance(month_payload, dict) else {}
    if isinstance(segments, dict):
        for key, segment_payload in segments.items():
            if not isinstance(segment_payload, dict):
                continue
            norm_key = str(key).strip().upper()
            try:
                segment_hit_map[norm_key] = float(segment_payload.get("hit_rate"))
            except Exception:
                pass
            try:
                segment_lift_map[norm_key] = float(segment_payload.get("lift_pp"))
            except Exception:
                pass

    targets = out.get(target_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    directions = out.get(direction_col, pd.Series("", index=out.index)).fillna("").astype(str).str.upper().str.strip()
    segment_keys = _segment_key(targets, directions)
    out["final_pool_segment_key"] = segment_keys
    out["final_pool_segment_hit_rate"] = pd.to_numeric(segment_keys.map(segment_hit_map), errors="coerce")
    if np.isfinite(global_hit_rate):
        out["final_pool_segment_hit_rate"] = out["final_pool_segment_hit_rate"].fillna(float(global_hit_rate))
    out["final_pool_segment_lift_pp"] = pd.to_numeric(segment_keys.map(segment_lift_map), errors="coerce").fillna(0.0)
    out["final_pool_global_hit_rate"] = float(global_hit_rate) if np.isfinite(global_hit_rate) else np.nan

    margin_window = max(float(near_miss_margin), 1e-6)
    margin_component = pd.Series(
        np.where(
            finite_threshold.to_numpy(dtype=bool, copy=False),
            np.clip((raw_margin.to_numpy(dtype="float64", copy=False) + margin_window) / (2.0 * margin_window), 0.0, 1.0),
            _normalize_probability(probabilities).to_numpy(dtype="float64", copy=False),
        ),
        index=out.index,
        dtype="float64",
    )
    segment_component = _normalize_probability(out["final_pool_segment_hit_rate"].fillna(global_hit_rate if np.isfinite(global_hit_rate) else 0.60))
    confidence_component = _clip01(_numeric_series(out, "final_confidence", 0.0))
    gap_component = _clip01(_numeric_series(out, "gap_percentile", 0.0))
    recency_component = _clip01(_numeric_series(out, "recency_factor", 0.50))
    recoverability_component = _clip01(_numeric_series(out, "recoverability_score", 0.50))
    noise_penalty = _clip01(_numeric_series(out, "noise_score", 0.0))
    contradiction_penalty = _clip01(_numeric_series(out, "contradiction_score", 0.0))

    belief_uncertainty = _numeric_series(out, "belief_uncertainty", 1.0)
    belief_uncertainty_normalized = _numeric_series(out, "belief_uncertainty_normalized", np.nan)
    if belief_uncertainty_normalized.isna().any():
        span = max(float(belief_uncertainty_upper) - float(belief_uncertainty_lower), 1e-9)
        fallback_unc = ((belief_uncertainty - float(belief_uncertainty_lower)) / span).clip(lower=0.0, upper=1.0)
        belief_uncertainty_normalized = belief_uncertainty_normalized.fillna(fallback_unc)
    uncertainty_component = (1.0 - belief_uncertainty_normalized).clip(lower=0.0, upper=1.0)

    edge_scale = _numeric_series(out, "edge_scale", np.nan)
    if edge_scale.isna().any():
        abs_edge = _numeric_series(out, "abs_edge", 0.0).clip(lower=0.0)
        edge_baseline = abs_edge.groupby(targets).transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
        edge_scale = edge_scale.fillna((abs_edge / edge_baseline).clip(lower=0.50, upper=2.50))
    edge_component = ((edge_scale - 0.50) / 2.00).clip(lower=0.0, upper=1.0)

    quality_score = (
        0.24 * margin_component
        + 0.18 * segment_component
        + 0.16 * confidence_component
        + 0.12 * gap_component
        + 0.10 * edge_component
        + 0.08 * uncertainty_component
        + 0.06 * recency_component
        + 0.04 * recoverability_component
        - 0.06 * noise_penalty
        - 0.06 * contradiction_penalty
    ).clip(lower=0.0, upper=1.0)

    out["final_pool_quality_score"] = pd.to_numeric(quality_score, errors="coerce").fillna(0.0)
    out["parlay_leg_quality_score"] = out["final_pool_quality_score"]
    out["final_pool_quality_rank"] = out["final_pool_quality_score"].rank(method="first", ascending=False)
    return out
