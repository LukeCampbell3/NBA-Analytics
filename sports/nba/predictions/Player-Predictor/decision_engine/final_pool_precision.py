from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
VALIDATION_ROOT = WORKSPACE_ROOT / "validation"
TARGETS = {"PTS", "TRB", "AST"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _numeric_series(frame: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _string_series(frame: pd.DataFrame, columns: tuple[str, ...], default: str = "") -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype="object")


def _percentile_rank(series: pd.Series, default: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(default, index=series.index, dtype="float64")
    return numeric.rank(method="average", pct=True).fillna(default).astype("float64")


def _coerce_event_datetime(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    out = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ymd_mask = numeric.between(19000101, 21001231, inclusive="both")
    if bool(ymd_mask.any()):
        ymd_text = numeric.round().astype("Int64").astype(str)
        parsed_ymd = pd.to_datetime(ymd_text.where(ymd_mask), format="%Y%m%d", errors="coerce")
        out = out.where(~ymd_mask, parsed_ymd)
    return out


def _discover_history_csv() -> Path | None:
    candidates = sorted(
        VALIDATION_ROOT.glob("validation_recent_pool_selector_*_rows.csv"),
        key=lambda path: (path.stat().st_size, path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _discover_feedback_csv() -> Path | None:
    candidates = sorted(
        VALIDATION_ROOT.glob("precision_pool_*_rows.csv"),
        key=lambda path: (path.stat().st_size, path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _prepare_history(history: pd.DataFrame, *, cutoff_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    out = history.copy()
    if "market_date" not in out.columns:
        return pd.DataFrame()
    out["event_date"] = _coerce_event_datetime(out["market_date"]).dt.normalize()
    out = out.loc[out["event_date"].notna()].copy()
    if pd.notna(cutoff_date):
        out = out.loc[out["event_date"] < pd.Timestamp(cutoff_date).normalize()].copy()
    if out.empty:
        return out
    out["target"] = _string_series(out, ("target",), "").str.upper().str.strip()
    out["direction"] = _string_series(out, ("direction",), "").str.upper().str.strip()
    out = out.loc[out["target"].isin(TARGETS) & out["direction"].isin({"OVER", "UNDER"})].copy()
    result = _string_series(out, ("result", "outcome"), "").str.lower().str.strip()
    out["win"] = result.map({"win": 1.0, "loss": 0.0})
    out = out.loc[out["win"].notna()].copy()
    if out.empty:
        return out
    out["prob"] = _numeric_series(out, ("estimated_win_rate", "expected_win_rate", "p_calibrated"), 0.5).fillna(0.5).clip(0.01, 0.99)
    out["confidence"] = _numeric_series(out, ("selection_confidence", "final_confidence"), 0.0).fillna(0.0).clip(0.0, 1.0)
    out["abs_edge"] = _numeric_series(out, ("abs_edge",), 0.0).fillna(0.0)
    out["uncertainty_sigma"] = _numeric_series(out, ("uncertainty_sigma",), np.nan)
    out["spike_probability"] = _numeric_series(out, ("spike_probability",), 0.0).fillna(0.0).clip(0.0, 1.0)
    out["segment_key"] = out["target"] + "|" + out["direction"]
    out["prob_bucket"] = pd.cut(
        out["prob"],
        bins=[0.0, 0.52, 0.56, 0.60, 0.66, 1.0],
        labels=["p000_052", "p052_056", "p056_060", "p060_066", "p066_100"],
        include_lowest=True,
    ).astype(str)
    return out


def _prepare_feedback(feedback: pd.DataFrame, *, cutoff_date: pd.Timestamp | pd.NaT) -> pd.DataFrame:
    if feedback.empty:
        return pd.DataFrame()
    out = feedback.copy()
    date_col = "run_date" if "run_date" in out.columns else "market_date" if "market_date" in out.columns else ""
    if not date_col:
        return pd.DataFrame()
    out["event_date"] = _coerce_event_datetime(out[date_col]).dt.normalize()
    out = out.loc[out["event_date"].notna()].copy()
    if pd.notna(cutoff_date):
        out = out.loc[out["event_date"] < pd.Timestamp(cutoff_date).normalize()].copy()
    if out.empty:
        return out
    if "mode" in out.columns:
        out = out.loc[out["mode"].astype(str).str.strip().str.lower().eq("precision_pool")].copy()
    out["target"] = _string_series(out, ("target",), "").str.upper().str.strip()
    out["direction"] = _string_series(out, ("direction",), "").str.upper().str.strip()
    out = out.loc[out["target"].isin(TARGETS) & out["direction"].isin({"OVER", "UNDER"})].copy()
    result = _string_series(out, ("result", "outcome"), "").str.lower().str.strip()
    out["win"] = result.map({"win": 1.0, "loss": 0.0})
    out = out.loc[out["win"].notna()].copy()
    if out.empty:
        return out
    out["segment_key"] = out["target"] + "|" + out["direction"]
    return out


def _posterior_stats(wins: float, rows: int, *, prior_mean: float, prior_strength: float) -> dict[str, float]:
    rows = int(max(0, rows))
    alpha = float(prior_mean) * float(prior_strength) + float(wins)
    beta = (1.0 - float(prior_mean)) * float(prior_strength) + float(max(0, rows) - float(wins))
    total = max(alpha + beta, 1e-9)
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    lcb = float(np.clip(mean - 1.15 * math.sqrt(max(variance, 0.0)), 0.01, 0.99))
    return {"rows": float(rows), "wins": float(wins), "mean": float(mean), "lcb": lcb}


def _profile_map(history: pd.DataFrame, key_col: str, *, prior_mean: float, prior_strength: float) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    if history.empty or key_col not in history.columns:
        return profiles
    grouped = history.groupby(key_col, dropna=False)["win"].agg(["size", "sum"]).reset_index()
    for _, row in grouped.iterrows():
        key = str(row.get(key_col, "")).strip()
        if not key:
            continue
        profiles[key] = _posterior_stats(
            _safe_float(row.get("sum"), 0.0),
            int(row.get("size", 0) or 0),
            prior_mean=prior_mean,
            prior_strength=prior_strength,
        )
    return profiles


def _window_history(history: pd.DataFrame, days: int) -> pd.DataFrame:
    if history.empty or "event_date" not in history.columns:
        return pd.DataFrame()
    window_days = int(max(1, days))
    cutoff = history["event_date"].max() - pd.Timedelta(days=window_days - 1)
    return history.loc[history["event_date"] >= cutoff].copy()


def _history_frame(history_frame: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if history_frame is not None:
        return history_frame.copy(), {"source": "provided_history", "path": ""}
    path = _discover_history_csv()
    if path is None:
        return pd.DataFrame(), {"source": "missing_history", "path": ""}
    try:
        return pd.read_csv(path), {"source": "discovered_history_csv", "path": str(path)}
    except Exception as exc:
        return pd.DataFrame(), {"source": "history_read_error", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _feedback_frame(feedback_frame: pd.DataFrame | None, *, allow_discovery: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feedback_frame is not None:
        return feedback_frame.copy(), {"source": "provided_feedback", "path": ""}
    if not allow_discovery:
        return pd.DataFrame(), {"source": "feedback_discovery_disabled", "path": ""}
    path = _discover_feedback_csv()
    if path is None:
        return pd.DataFrame(), {"source": "missing_feedback", "path": ""}
    try:
        return pd.read_csv(path), {"source": "discovered_precision_pool_feedback_csv", "path": str(path)}
    except Exception as exc:
        return pd.DataFrame(), {"source": "feedback_read_error", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _run_cutoff(frame: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    for column in ("market_date", "run_date", "event_date"):
        if column in frame.columns:
            dates = _coerce_event_datetime(frame[column]).dropna()
            if not dates.empty:
                return pd.Timestamp(dates.min()).normalize()
    return pd.NaT


def annotate_precision_pool(
    candidates: pd.DataFrame,
    *,
    history_frame: pd.DataFrame | None = None,
    feedback_frame: pd.DataFrame | None = None,
    target_accuracy: float = 0.83,
    recent_days: int = 14,
    prior_strength: float = 18.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = candidates.copy()
    if out.empty:
        return out, {"enabled": False, "reason": "empty_candidates"}

    raw_history, source_meta = _history_frame(history_frame)
    cutoff = _run_cutoff(out)
    history = _prepare_history(raw_history, cutoff_date=cutoff)
    raw_feedback, feedback_source_meta = _feedback_frame(feedback_frame, allow_discovery=history_frame is None)
    feedback = _prepare_feedback(raw_feedback, cutoff_date=cutoff)
    if history.empty:
        out["precision_pool_enabled"] = False
        out["precision_pool_prob"] = pd.to_numeric(out.get("board_play_win_prob", out.get("expected_win_rate")), errors="coerce").fillna(0.5)
        out["precision_pool_lcb"] = out["precision_pool_prob"] * 0.90
        out["precision_pool_score"] = out["precision_pool_lcb"]
        return out, {"enabled": False, "reason": "empty_history", "history_source": source_meta}

    global_mean = float(history["win"].mean())
    short_days = int(max(3, round(float(recent_days) / 2.0)))
    medium_days = int(max(int(recent_days) * 2, 28))
    short = _window_history(history, short_days)
    recent = _window_history(history, int(recent_days))
    medium = _window_history(history, medium_days)
    if short.empty:
        short = recent.copy() if not recent.empty else history.copy()
    if recent.empty:
        recent = history.copy()
    if medium.empty:
        medium = history.copy()

    segment_long = _profile_map(history, "segment_key", prior_mean=global_mean, prior_strength=prior_strength)
    segment_medium = _profile_map(medium, "segment_key", prior_mean=global_mean, prior_strength=prior_strength * 0.85)
    segment_recent = _profile_map(recent, "segment_key", prior_mean=global_mean, prior_strength=prior_strength * 0.65)
    segment_short = _profile_map(short, "segment_key", prior_mean=global_mean, prior_strength=prior_strength * 0.50)
    target_long = _profile_map(history, "target", prior_mean=global_mean, prior_strength=prior_strength)
    direction_long = _profile_map(history, "direction", prior_mean=global_mean, prior_strength=prior_strength)
    bucket_long = _profile_map(history, "prob_bucket", prior_mean=global_mean, prior_strength=prior_strength)
    feedback_segment = _profile_map(feedback, "segment_key", prior_mean=global_mean, prior_strength=12.0) if not feedback.empty else {}
    feedback_target = _profile_map(feedback, "target", prior_mean=global_mean, prior_strength=14.0) if not feedback.empty else {}
    feedback_direction = _profile_map(feedback, "direction", prior_mean=global_mean, prior_strength=18.0) if not feedback.empty else {}

    target = _string_series(out, ("target",), "").str.upper().str.strip()
    direction = _string_series(out, ("direction",), "").str.upper().str.strip()
    segment_key = target + "|" + direction
    model_prob = _numeric_series(out, ("robust_reranker_prob", "selected_board_prob_raw", "board_play_win_prob", "expected_win_rate"), 0.5).fillna(0.5).clip(0.01, 0.99)
    confidence = _numeric_series(out, ("final_confidence", "selection_confidence"), 0.0).fillna(0.0).clip(0.0, 1.0)
    quality = _numeric_series(out, ("final_pool_quality_score", "parlay_leg_quality_score"), 0.5).fillna(0.5).clip(0.0, 1.0)
    ev_signal = _numeric_series(out, ("ev_adjusted", "ev"), 0.0).fillna(0.0)
    abs_edge_signal = _numeric_series(out, ("abs_edge",), 0.0).fillna(0.0)
    market_strength = (
        0.35 * _percentile_rank(ev_signal)
        + 0.30 * _percentile_rank(confidence)
        + 0.20 * _percentile_rank(abs_edge_signal)
        + 0.15 * _percentile_rank(model_prob)
    ).clip(0.0, 1.0)
    uncertainty = _numeric_series(out, ("belief_uncertainty_normalized", "uncertainty_sigma"), 0.0).fillna(0.0)
    spike = _numeric_series(out, ("spike_probability",), 0.0).fillna(0.0).clip(0.0, 1.0)
    prob_bucket = pd.cut(
        model_prob,
        bins=[0.0, 0.52, 0.56, 0.60, 0.66, 1.0],
        labels=["p000_052", "p052_056", "p056_060", "p060_066", "p066_100"],
        include_lowest=True,
    ).astype(str)

    probs: list[float] = []
    lcbs: list[float] = []
    rows: list[int] = []
    sources: list[str] = []
    short_probs: list[float] = []
    long_probs: list[float] = []
    consistency_scores: list[float] = []
    regime_deltas: list[float] = []
    regime_trusts: list[float] = []
    feedback_probs: list[float] = []
    feedback_lcbs: list[float] = []
    feedback_rows: list[int] = []
    feedback_adjustments: list[float] = []

    # --- Direction-balanced fallback: when a segment (e.g. PTS|OVER) has no
    # history, fall back to the *target-level* profile (PTS) and the
    # *opposite-direction* segment (PTS|UNDER) with a conservative haircut
    # instead of dropping the segment entirely.  This lets qualified OVERs
    # surface when the system has only published UNDERs historically.
    def _segment_fallback_profile(
        profile_map: dict,
        seg_key: str,
        tgt_key: str,
        direc_key: str,
    ) -> tuple[dict | None, str]:
        """Return (profile, source_label).  Tries the exact segment first,
        then the target-level profile, then the opposite-direction segment
        with a conservative haircut applied inline by the caller."""
        profile = profile_map.get(seg_key)
        if profile and float(profile.get("rows", 0)) > 0:
            return profile, "segment"
        # Try target-level (direction-agnostic)
        target_profile = profile_map.get(tgt_key)
        if target_profile and float(target_profile.get("rows", 0)) > 0:
            return target_profile, "target_fallback"
        # Try opposite direction segment
        opposite_dir = "UNDER" if direc_key == "OVER" else "OVER"
        opposite_seg = tgt_key + "|" + opposite_dir
        opp_profile = profile_map.get(opposite_seg)
        if opp_profile and float(opp_profile.get("rows", 0)) > 0:
            # Return with a flag; caller applies haircut
            return opp_profile, "opposite_fallback"
        return None, "missing"

    for idx in out.index:
        pieces: list[tuple[float, float, float, str, float]] = []
        seg = str(segment_key.loc[idx])
        tgt = str(target.loc[idx])
        direc = str(direction.loc[idx])
        bucket = str(prob_bucket.loc[idx])
        for profile_map, key, base_weight, label in (
            (segment_short, seg, 0.16, "segment_short"),
            (segment_recent, seg, 0.20, "segment_recent"),
            (segment_medium, seg, 0.18, "segment_medium"),
            (segment_long, seg, 0.22, "segment_long"),
            (bucket_long, bucket, 0.12, "prob_bucket"),
            (target_long, tgt, 0.08, "target"),
            (direction_long, direc, 0.06, "direction"),
        ):
            profile = profile_map.get(key)
            if profile:
                support = float(np.clip(profile["rows"] / 40.0, 0.15, 1.0))
                pieces.append((float(profile["mean"]), float(profile["lcb"]), base_weight * support, label, float(profile["rows"])))
                continue
            # Direction-balanced fallback for segment-level maps only
            if label.startswith("segment_"):
                fb_profile, fb_source = _segment_fallback_profile(profile_map, seg, tgt, direc)
                if fb_profile:
                    fb_support = float(np.clip(fb_profile["rows"] / 40.0, 0.15, 1.0))
                    fb_mean = float(fb_profile["mean"])
                    fb_lcb = float(fb_profile["lcb"])
                    # Apply conservative haircut for fallback sources
                    if fb_source == "opposite_fallback":
                        fb_mean = fb_mean * 0.85  # 15% haircut for opposite direction (OVERs win at ~50% vs UNDERs at ~67%)
                        fb_lcb = fb_lcb * 0.80    # 20% haircut on lower bound
                        fb_support *= 0.45         # significantly reduce weight
                    elif fb_source == "target_fallback":
                        fb_mean = fb_mean * 0.93  # 7% haircut for target-level
                        fb_lcb = fb_lcb * 0.90
                        fb_support *= 0.65
                    pieces.append((fb_mean, fb_lcb, base_weight * fb_support, f"{label}_{fb_source}", float(fb_profile["rows"])))
        pieces.append((float(model_prob.loc[idx]), float(model_prob.loc[idx]) - 0.04, 0.22 + 0.18 * float(confidence.loc[idx]), "model", 0.0))
        weight_sum = sum(piece[2] for piece in pieces) or 1.0
        empirical_prob = sum(piece[0] * piece[2] for piece in pieces) / weight_sum
        empirical_lcb = sum(piece[1] * piece[2] for piece in pieces) / weight_sum

        short_profile = segment_short.get(seg) or segment_recent.get(seg)
        recent_profile = segment_recent.get(seg) or short_profile
        medium_profile = segment_medium.get(seg) or segment_long.get(seg)
        long_profile = segment_long.get(seg)
        horizon_profiles = [profile for profile in (short_profile, recent_profile, medium_profile, long_profile) if profile]
        horizon_means = [float(profile["mean"]) for profile in horizon_profiles]
        horizon_rows = [float(profile["rows"]) for profile in horizon_profiles]
        short_signal = float(short_profile["mean"]) if short_profile else float(model_prob.loc[idx])
        long_signal = float(long_profile["mean"]) if long_profile else float(np.mean(horizon_means)) if horizon_means else float(model_prob.loc[idx])
        horizon_std = float(np.std(horizon_means)) if len(horizon_means) >= 2 else 0.0
        regime_delta = float(short_signal - long_signal)
        support_factor = float(np.clip(max(horizon_rows, default=0.0) / 60.0, 0.0, 1.0))
        agreement = float(np.clip(1.0 - (horizon_std / 0.16) - (abs(regime_delta) / 0.32), 0.0, 1.0))
        consistency_score = float(np.clip(0.35 + 0.65 * support_factor * agreement, 0.0, 1.0))
        recent_rows = float(recent_profile["rows"]) if recent_profile else 0.0
        regime_trust = float(np.clip((recent_rows / 35.0) * (0.45 + 0.55 * agreement), 0.0, 1.0))
        regime_boost = float(np.clip(regime_delta, -0.12, 0.12) * 0.20 * regime_trust)
        consistency_haircut = 0.040 * (1.0 - consistency_score)
        unsupported_hot_haircut = 0.030 * max(0.0, regime_delta) * (1.0 - regime_trust)

        feedback_pieces: list[tuple[float, float, float, float]] = []
        for profile_map, key, base_weight in (
            (feedback_segment, seg, 0.62),
            (feedback_target, tgt, 0.22),
            (feedback_direction, direc, 0.16),
        ):
            profile = profile_map.get(key)
            if not profile:
                continue
            support = float(np.clip(profile["rows"] / 16.0, 0.20, 1.0))
            feedback_pieces.append((float(profile["mean"]), float(profile["lcb"]), base_weight * support, float(profile["rows"])))
        if feedback_pieces:
            feedback_weight = sum(piece[2] for piece in feedback_pieces) or 1.0
            feedback_prob = sum(piece[0] * piece[2] for piece in feedback_pieces) / feedback_weight
            feedback_lcb = sum(piece[1] * piece[2] for piece in feedback_pieces) / feedback_weight
            feedback_support = max(piece[3] for piece in feedback_pieces)
            feedback_trust = float(np.clip(feedback_support / 12.0, 0.0, 1.0))
            feedback_adjustment = float(np.clip(feedback_prob - empirical_prob, -0.18, 0.12) * 0.55 * feedback_trust)
            feedback_lcb_haircut = float(max(0.0, empirical_lcb - feedback_lcb) * 0.45 * feedback_trust)
        else:
            feedback_prob = np.nan
            feedback_lcb = np.nan
            feedback_support = 0.0
            feedback_adjustment = 0.0
            feedback_lcb_haircut = 0.0

        risk_haircut = 0.035 * float(np.clip(uncertainty.loc[idx], 0.0, 1.0)) + 0.030 * float(spike.loc[idx])
        quality_boost = 0.035 * (float(quality.loc[idx]) - 0.50)
        final_prob = float(np.clip(empirical_prob + quality_boost + regime_boost + feedback_adjustment - risk_haircut - consistency_haircut - unsupported_hot_haircut, 0.01, 0.99))
        final_lcb = float(np.clip(empirical_lcb + 0.50 * quality_boost + 0.50 * regime_boost + 0.60 * feedback_adjustment - feedback_lcb_haircut - risk_haircut - consistency_haircut - unsupported_hot_haircut, 0.01, 0.99))
        probs.append(final_prob)
        lcbs.append(final_lcb)
        rows.append(int(max([piece[4] for piece in pieces], default=0.0)))
        sources.append("+".join(piece[3] for piece in pieces))
        short_probs.append(short_signal)
        long_probs.append(long_signal)
        consistency_scores.append(consistency_score)
        regime_deltas.append(regime_delta)
        regime_trusts.append(regime_trust)
        feedback_probs.append(float(feedback_prob) if feedback_prob == feedback_prob else np.nan)
        feedback_lcbs.append(float(feedback_lcb) if feedback_lcb == feedback_lcb else np.nan)
        feedback_rows.append(int(feedback_support))
        feedback_adjustments.append(feedback_adjustment)

    out["precision_pool_enabled"] = True
    out["precision_pool_target_accuracy"] = float(target_accuracy)
    out["precision_pool_prob"] = pd.Series(probs, index=out.index, dtype="float64")
    out["precision_pool_lcb"] = pd.Series(lcbs, index=out.index, dtype="float64")
    out["precision_pool_support_rows"] = pd.Series(rows, index=out.index, dtype="int64")
    out["precision_pool_source"] = pd.Series(sources, index=out.index, dtype="object")
    out["precision_pool_short_prob"] = pd.Series(short_probs, index=out.index, dtype="float64")
    out["precision_pool_long_prob"] = pd.Series(long_probs, index=out.index, dtype="float64")
    out["precision_pool_consistency_score"] = pd.Series(consistency_scores, index=out.index, dtype="float64")
    out["precision_pool_regime_delta"] = pd.Series(regime_deltas, index=out.index, dtype="float64")
    out["precision_pool_regime_trust"] = pd.Series(regime_trusts, index=out.index, dtype="float64")
    out["precision_pool_feedback_prob"] = pd.Series(feedback_probs, index=out.index, dtype="float64")
    out["precision_pool_feedback_lcb"] = pd.Series(feedback_lcbs, index=out.index, dtype="float64")
    out["precision_pool_feedback_rows"] = pd.Series(feedback_rows, index=out.index, dtype="int64")
    out["precision_pool_feedback_adjustment"] = pd.Series(feedback_adjustments, index=out.index, dtype="float64")
    out["precision_pool_market_strength"] = market_strength.astype("float64")
    out["precision_pool_score"] = (
        0.32 * out["precision_pool_lcb"]
        + 0.18 * out["precision_pool_prob"]
        + 0.18 * out["precision_pool_market_strength"]
        + 0.07 * out["precision_pool_consistency_score"]
        + 0.15 * out["precision_pool_feedback_lcb"].fillna(out["precision_pool_lcb"])
        + 0.07 * model_prob.astype("float64")
        + 0.03 * quality.astype("float64")
    ).clip(0.0, 1.0)
    existing_parlay_quality = _numeric_series(out, ("parlay_leg_quality_score",), 0.0).fillna(0.0)
    out["parlay_leg_quality_score"] = np.maximum(existing_parlay_quality, out["precision_pool_score"])
    return out, {
        "enabled": True,
        "history_source": source_meta,
        "feedback_source": feedback_source_meta,
        "history_rows": int(len(history)),
        "feedback_rows": int(len(feedback)),
        "short_rows": int(len(short)),
        "recent_rows": int(len(recent)),
        "medium_rows": int(len(medium)),
        "global_hit_rate": float(global_mean),
        "target_accuracy": float(target_accuracy),
        "short_days": int(short_days),
        "recent_days": int(max(1, recent_days)),
        "medium_days": int(medium_days),
    }


def choose_precision_board_size(
    ranked: pd.DataFrame,
    *,
    max_total_plays: int,
    min_board_plays: int = 0,
    target_accuracy: float = 0.83,
    miss_penalty: float = 2.4,
    volume_reward: float = 0.015,
) -> tuple[int, dict[str, Any]]:
    if ranked.empty:
        return 0, {"enabled": False, "reason": "empty_ranked"}
    max_k = int(max_total_plays) if int(max_total_plays) > 0 else int(len(ranked))
    max_k = max(1, min(max_k, int(len(ranked))))
    min_k = max(1, min(int(max(0, min_board_plays)) or 1, max_k))
    fallback_prob = _numeric_series(ranked, ("expected_win_rate",), 0.5).fillna(0.5)
    prob_series = _numeric_series(ranked, ("precision_pool_prob",), np.nan).fillna(fallback_prob).clip(0.01, 0.99)
    lcb_series = _numeric_series(ranked, ("precision_pool_lcb",), np.nan).fillna(prob_series).clip(0.01, 0.99)
    consistency_series = _numeric_series(ranked, ("precision_pool_consistency_score",), 1.0).fillna(1.0).clip(0.0, 1.0)
    strength_series = _numeric_series(ranked, ("precision_pool_market_strength",), 0.5).fillna(0.5).clip(0.0, 1.0)
    prob = prob_series.to_numpy(dtype="float64")
    lcb = lcb_series.to_numpy(dtype="float64")
    consistency = consistency_series.to_numpy(dtype="float64")
    strength = strength_series.to_numpy(dtype="float64")
    scores: list[dict[str, float]] = []
    best_k = min_k
    best_score = -np.inf
    target = float(np.clip(target_accuracy, 0.50, 0.95))
    for k in range(min_k, max_k + 1):
        avg_prob = float(np.mean(prob[:k]))
        avg_lcb = float(np.mean(lcb[:k]))
        shortfall = max(0.0, target - avg_lcb)
        avg_consistency = float(np.mean(consistency[:k]))
        avg_strength = float(np.mean(strength[:k]))
        variance = float(np.sqrt(np.sum(np.clip(prob[:k] * (1.0 - prob[:k]), 0.0, 1.0))) / max(1, k))
        objective = avg_prob + float(volume_reward) * math.log1p(k) - float(miss_penalty) * shortfall * shortfall - 0.18 * variance - 0.050 * (1.0 - avg_consistency)
        fail_closed_objective = avg_lcb + 0.08 * avg_strength + 0.04 * avg_consistency - 0.04 * math.log1p(k) - 0.12 * variance
        scores.append(
            {
                "k": float(k),
                "avg_prob": avg_prob,
                "avg_lcb": avg_lcb,
                "avg_consistency": avg_consistency,
                "avg_strength": avg_strength,
                "objective": objective,
                "fail_closed_objective": fail_closed_objective,
            }
        )
        if objective > best_score:
            best_score = objective
            best_k = k
    if not any(float(row["avg_lcb"]) >= target for row in scores):
        best_row = max(scores, key=lambda row: (float(row["fail_closed_objective"]), -float(row["k"])))
        best_k = int(best_row["k"])
    selected = next((row for row in scores if int(row["k"]) == int(best_k)), scores[0])
    return int(best_k), {
        "enabled": True,
        "target_accuracy": target,
        "selected_size": int(best_k),
        "selected_avg_prob": float(selected["avg_prob"]),
        "selected_avg_lcb": float(selected["avg_lcb"]),
        "selected_avg_consistency": float(selected.get("avg_consistency", 1.0)),
        "selected_avg_strength": float(selected.get("avg_strength", 0.5)),
        "selected_objective": float(selected["objective"]),
        "selected_fail_closed_objective": float(selected.get("fail_closed_objective", selected["objective"])),
        "target_attainable": bool(float(selected["avg_lcb"]) >= target),
        "evaluated_sizes": scores,
    }
