from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def _clip_prob(values: np.ndarray | pd.Series, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    return np.clip(np.asarray(values, dtype="float64"), float(low), float(high))


def _month_token(value: str | pd.Timestamp | datetime | None) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m")


def _log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = _clip_prob(p, 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = _clip_prob(p, 0.0, 1.0)
    y = np.asarray(y, dtype="float64")
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0
    n = max(1, len(p))
    for b in range(len(edges) - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (float(np.sum(mask)) / n) * abs(acc - conf)
    return float(ece)


def _monotonic_accumulate(rates: np.ndarray) -> np.ndarray:
    out = np.asarray(rates, dtype="float64").copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def fit_monotonic_bin_calibrator(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> dict[str, Any] | None:
    p = _clip_prob(probs, 0.01, 0.99)
    y = np.asarray(labels, dtype="float64")
    if len(p) == 0 or len(y) == 0 or len(p) != len(y):
        return None
    if len(p) < 20:
        return None

    try:
        # Quantile bins are robust for narrow-score ranges.
        q = pd.qcut(pd.Series(p), q=int(max(3, n_bins)), duplicates="drop")
    except Exception:
        return None

    if q is None:
        return None
    bin_codes = q.cat.codes.to_numpy(dtype="int64")
    valid_mask = bin_codes >= 0
    if not np.any(valid_mask):
        return None
    p = p[valid_mask]
    y = y[valid_mask]
    bin_codes = bin_codes[valid_mask]

    n_used = int(bin_codes.max()) + 1
    if n_used <= 0:
        return None
    centers = np.zeros(n_used, dtype="float64")
    rates = np.zeros(n_used, dtype="float64")
    counts = np.zeros(n_used, dtype="int64")

    for b in range(n_used):
        mask = bin_codes == b
        c = int(np.sum(mask))
        if c <= 0:
            continue
        pb = p[mask]
        yb = y[mask]
        wins = float(np.sum(yb))
        centers[b] = float(np.mean(pb))
        counts[b] = c
        rates[b] = float((wins + float(alpha)) / (c + float(alpha) + float(beta)))

    valid = counts > 0
    if not np.any(valid):
        return None
    centers = centers[valid]
    rates = rates[valid]
    counts = counts[valid]
    if len(centers) <= 1:
        return None

    order = np.argsort(centers)
    centers = centers[order]
    rates = rates[order]
    counts = counts[order]
    rates = _monotonic_accumulate(rates)
    rates = _clip_prob(rates, 0.01, 0.99)

    return {
        "kind": "monotonic_bin",
        "bin_centers": centers.tolist(),
        "bin_rates": rates.tolist(),
        "bin_counts": counts.tolist(),
        "rows": int(len(p)),
        "wins": int(np.sum(y)),
        "mean_raw_prob": float(np.mean(p)),
        "mean_label": float(np.mean(y)),
    }


def apply_monotonic_bin_calibrator(probs: np.ndarray, calibrator: dict[str, Any] | None) -> np.ndarray:
    p = _clip_prob(probs, 0.01, 0.99)
    if not calibrator:
        return p
    centers = np.asarray(calibrator.get("bin_centers", []), dtype="float64")
    rates = np.asarray(calibrator.get("bin_rates", []), dtype="float64")
    if len(centers) <= 1 or len(rates) <= 1 or len(centers) != len(rates):
        return p
    return _clip_prob(np.interp(p, centers, rates), 0.01, 0.99)


@dataclass
class CalibratorFitConfig:
    lookback_days: int = 120
    min_rows_global: int = 250
    min_rows_segment: int = 80
    n_bins: int = 10
    recent_window_days: int = 21
    recent_min_rows_global: int = 40
    recent_min_rows_segment: int = 18
    recent_strength: float = 20.0
    recent_max_adjustment: float = 0.08
    safety_min_rows: int = 60
    safety_bin_edges: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00)
    safety_min_bucket_rows: int = 12
    safety_high_prob_threshold: float = 0.65
    safety_gap_threshold: float = 0.04
    safety_gap_retention: float = 0.35
    safety_margin: float = 0.02


def _fit_global_shrink_factor(probs: np.ndarray, labels: np.ndarray) -> float:
    p = _clip_prob(probs, 0.01, 0.99)
    y = np.asarray(labels, dtype="float64")
    if len(p) < 20 or len(y) != len(p):
        return 1.0
    centered = p - float(np.mean(p))
    variance = float(np.mean(centered ** 2))
    if variance <= 1e-9:
        return 1.0
    covariance = float(np.mean(centered * (y - float(np.mean(y)))))
    slope = covariance / variance
    return float(np.clip(slope, 0.50, 1.00))


def fit_empirical_safety_profile(
    probs: np.ndarray,
    labels: np.ndarray,
    config: CalibratorFitConfig | None = None,
) -> dict[str, Any] | None:
    cfg = config or CalibratorFitConfig()
    p = _clip_prob(probs, 0.01, 0.99)
    y = np.asarray(labels, dtype="float64")
    if len(p) < int(max(20, cfg.safety_min_rows)) or len(y) != len(p):
        return None

    edges = np.asarray(cfg.safety_bin_edges, dtype="float64")
    if len(edges) < 2:
        return None
    edges = np.clip(edges, 0.0, 1.0)
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)
    edges = np.unique(edges)
    if len(edges) < 2:
        return None

    mean_prob = float(np.mean(p))
    mean_label = float(np.mean(y))
    shrink_factor = 1.0
    if (mean_prob - mean_label) >= 0.02:
        high_conf_mask = p >= 0.55
        shrink_probs = p[high_conf_mask] if int(np.sum(high_conf_mask)) >= 20 else p
        shrink_labels = y[high_conf_mask] if int(np.sum(high_conf_mask)) >= 20 else y
        shrink_factor = _fit_global_shrink_factor(shrink_probs, shrink_labels)
    high_prob_buckets: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        upper_bound = upper if upper < 1.0 else 1.000001
        mask = (p >= float(lower)) & (p < float(upper_bound))
        count = int(np.sum(mask))
        if count < int(max(1, cfg.safety_min_bucket_rows)):
            continue
        avg_prob = float(np.mean(p[mask]))
        if avg_prob < float(cfg.safety_high_prob_threshold):
            continue
        wins = float(np.sum(y[mask]))
        smoothed_hit_rate = float((wins + 2.0) / (count + 4.0))
        gap = float(avg_prob - smoothed_hit_rate)
        if gap < float(cfg.safety_gap_threshold):
            continue
        cap = smoothed_hit_rate + float(cfg.safety_margin) + float(cfg.safety_gap_retention) * gap
        high_prob_buckets.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "rows": count,
                "wins": int(wins),
                "avg_prob": avg_prob,
                "smoothed_hit_rate": smoothed_hit_rate,
                "gap": gap,
                "cap": float(np.clip(cap, 0.50, avg_prob)),
            }
        )

    if not high_prob_buckets and shrink_factor >= 0.995:
        return None
    return {
        "rows": int(len(p)),
        "mean_prob": mean_prob,
        "mean_label": mean_label,
        "global_shrink_factor": shrink_factor,
        "high_prob_buckets": high_prob_buckets,
    }


def apply_empirical_safety_profile(
    probs: np.ndarray,
    safety_profile: dict[str, Any] | None,
) -> tuple[np.ndarray, bool]:
    p = _clip_prob(probs, 0.01, 0.99)
    if not safety_profile:
        return p, False

    transformed = p.copy()
    try:
        shrink_factor = float(safety_profile.get("global_shrink_factor", 1.0))
    except Exception:
        shrink_factor = 1.0
    shrink_factor = float(np.clip(shrink_factor, 0.50, 1.00))
    transformed = 0.5 + shrink_factor * (transformed - 0.5)
    applied = shrink_factor < 0.995

    for bucket in safety_profile.get("high_prob_buckets", []) or []:
        try:
            lower = float(bucket.get("lower", 0.0))
            upper = float(bucket.get("upper", 1.0))
            cap = float(bucket.get("cap", 1.0))
        except Exception:
            continue
        upper_bound = upper if upper < 1.0 else 1.000001
        mask = (p >= lower) & (p < upper_bound)
        if not np.any(mask):
            continue
        transformed[mask] = np.minimum(transformed[mask], float(np.clip(cap, 0.50, 0.99)))
        applied = True

    return _clip_prob(transformed, 0.01, 0.99), applied


def _recent_profile_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "mean_label": float(np.clip(frame["_label"].mean(), 0.0, 1.0)) if not frame.empty else 0.5,
        "mean_raw_prob": float(np.clip(frame["_prob"].mean(), 0.0, 1.0)) if not frame.empty else 0.5,
    }


def _apply_recent_regime_adjustment(
    calibrated: np.ndarray,
    seg_keys: pd.Series,
    month_payload: dict[str, Any],
    config_payload: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    cfg = config_payload or {}
    recent_segments = month_payload.get("recent_segments", {}) if isinstance(month_payload, dict) else {}
    recent_global = month_payload.get("recent_global") if isinstance(month_payload, dict) else None
    long_segments = month_payload.get("segments", {}) if isinstance(month_payload, dict) else {}
    safety_profile = month_payload.get("safety_profile") if isinstance(month_payload, dict) else None

    if not isinstance(recent_segments, dict) and not isinstance(recent_global, dict):
        return calibrated, np.full(len(calibrated), "", dtype=object), False

    global_long_rate = 0.5
    if isinstance(safety_profile, dict):
        try:
            global_long_rate = float(np.clip(safety_profile.get("mean_label", global_long_rate), 0.0, 1.0))
        except Exception:
            global_long_rate = 0.5

    recent_strength = max(1e-6, float(cfg.get("recent_strength", 20.0) or 20.0))
    recent_max_adjustment = float(np.clip(cfg.get("recent_max_adjustment", 0.08) or 0.08, 0.0, 0.25))
    recent_min_rows_segment = int(max(1, cfg.get("recent_min_rows_segment", 18) or 18))
    recent_min_rows_global = int(max(1, cfg.get("recent_min_rows_global", 40) or 40))

    adjusted = np.asarray(calibrated, dtype="float64").copy()
    recent_sources = np.full(len(adjusted), "", dtype=object)
    keys = seg_keys.astype(str).str.upper().str.strip()
    applied = False

    for idx, key in enumerate(keys.tolist()):
        segment_long = long_segments.get(key, {}) if isinstance(long_segments, dict) else {}
        segment_recent = recent_segments.get(key, {}) if isinstance(recent_segments, dict) else {}

        use_segment = isinstance(segment_recent, dict) and int(segment_recent.get("rows", 0) or 0) >= recent_min_rows_segment
        use_global = isinstance(recent_global, dict) and int(recent_global.get("rows", 0) or 0) >= recent_min_rows_global
        if not use_segment and not use_global:
            continue

        if use_segment:
            recent_rate = float(np.clip(segment_recent.get("mean_label", 0.5), 0.0, 1.0))
            long_rate = float(np.clip(segment_long.get("mean_label", global_long_rate), 0.0, 1.0))
            rows = int(segment_recent.get("rows", 0) or 0)
            source = f"segment:{key}"
        else:
            recent_rate = float(np.clip(recent_global.get("mean_label", 0.5), 0.0, 1.0))
            long_rate = global_long_rate
            rows = int(recent_global.get("rows", 0) or 0)
            source = "global"

        support = float(rows) / (float(rows) + recent_strength)
        shift = float(np.clip(recent_rate - long_rate, -recent_max_adjustment, recent_max_adjustment)) * float(
            np.clip(support, 0.0, 1.0)
        )
        if abs(shift) < 1e-6:
            continue
        adjusted[idx] = float(np.clip(adjusted[idx] + shift, 0.01, 0.99))
        recent_sources[idx] = source
        applied = True

    return adjusted, recent_sources, applied


def fit_selected_board_calibrator_payload(
    rows_df: pd.DataFrame,
    run_date_col: str = "run_date",
    prob_col: str = "expected_win_rate",
    label_col: str = "is_win",
    target_col: str = "target",
    direction_col: str = "direction",
    config: CalibratorFitConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CalibratorFitConfig()
    if rows_df.empty:
        return {"version": 1, "config": cfg.__dict__.copy(), "months": {}, "segments": []}

    df = rows_df.copy()
    raw_dates = df[run_date_col]
    parsed_token = pd.to_datetime(raw_dates.astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    parsed_generic = pd.to_datetime(raw_dates, errors="coerce")
    df["_run_date"] = parsed_token.fillna(parsed_generic)
    df = df.loc[df["_run_date"].notna()].copy()
    if df.empty:
        return {"version": 1, "config": cfg.__dict__.copy(), "months": {}, "segments": []}

    df["_month"] = df["_run_date"].dt.strftime("%Y-%m")
    df["_prob"] = _clip_prob(pd.to_numeric(df[prob_col], errors="coerce").fillna(0.5).to_numpy(dtype="float64"), 0.01, 0.99)
    df["_label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    df["_target"] = df[target_col].astype(str).str.upper().str.strip()
    df["_direction"] = df[direction_col].astype(str).str.upper().str.strip()

    months = sorted(df["_month"].dropna().unique().tolist())
    months_to_fit = months[:]
    if months:
        last_month = pd.to_datetime(f"{months[-1]}-01", errors="coerce")
        if pd.notna(last_month):
            next_month = (last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
            if next_month not in months_to_fit:
                months_to_fit.append(next_month)
    payload: dict[str, Any] = {
        "version": 1,
        "config": cfg.__dict__.copy(),
        "segments": ["GLOBAL", "PTS_OVER", "PTS_UNDER", "TRB_OVER", "TRB_UNDER", "AST_OVER", "AST_UNDER"],
        "months": {},
    }

    for month in months_to_fit:
        month_start = pd.to_datetime(f"{month}-01", errors="coerce")
        if pd.isna(month_start):
            continue
        lookback_start = month_start - pd.Timedelta(days=int(max(1, cfg.lookback_days)))
        train = df.loc[(df["_run_date"] < month_start) & (df["_run_date"] >= lookback_start)].copy()
        if train.empty:
            continue

        global_cal = None
        if len(train) >= int(cfg.min_rows_global):
            global_cal = fit_monotonic_bin_calibrator(
                train["_prob"].to_numpy(dtype="float64"),
                train["_label"].to_numpy(dtype="float64"),
                n_bins=int(max(3, cfg.n_bins)),
            )

        segments: dict[str, Any] = {}
        for target in ("PTS", "TRB", "AST"):
            for direction in ("OVER", "UNDER"):
                key = f"{target}_{direction}"
                seg = train.loc[(train["_target"] == target) & (train["_direction"] == direction)].copy()
                if len(seg) < int(cfg.min_rows_segment):
                    continue
                cal = fit_monotonic_bin_calibrator(
                    seg["_prob"].to_numpy(dtype="float64"),
                    seg["_label"].to_numpy(dtype="float64"),
                    n_bins=int(max(3, cfg.n_bins)),
                )
                if cal:
                    segments[key] = cal

        recent_train = train.loc[train["_run_date"] >= (month_start - pd.Timedelta(days=int(max(1, cfg.recent_window_days))))].copy()
        recent_global = None
        if len(recent_train) >= int(cfg.recent_min_rows_global):
            recent_global = _recent_profile_summary(recent_train)
        recent_segments: dict[str, Any] = {}
        for target in ("PTS", "TRB", "AST"):
            for direction in ("OVER", "UNDER"):
                key = f"{target}_{direction}"
                recent_seg = recent_train.loc[
                    (recent_train["_target"] == target) & (recent_train["_direction"] == direction)
                ].copy()
                if len(recent_seg) < int(cfg.recent_min_rows_segment):
                    continue
                recent_segments[key] = _recent_profile_summary(recent_seg)

        payload["months"][month] = {
            "train_rows": int(len(train)),
            "train_start": lookback_start.strftime("%Y-%m-%d"),
            "train_end": (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "global": global_cal,
            "segments": segments,
            "recent_train_rows": int(len(recent_train)),
            "recent_train_start": (month_start - pd.Timedelta(days=int(max(1, cfg.recent_window_days)))).strftime("%Y-%m-%d"),
            "recent_global": recent_global,
            "recent_segments": recent_segments,
            "safety_profile": fit_empirical_safety_profile(
                train["_prob"].to_numpy(dtype="float64"),
                train["_label"].to_numpy(dtype="float64"),
                config=cfg,
            ),
        }

    return payload


def _resolve_month_payload(payload: dict[str, Any], month: str) -> tuple[str, dict[str, Any] | None]:
    months = payload.get("months", {}) if isinstance(payload, dict) else {}
    if not isinstance(months, dict) or not months:
        return "", None
    if month in months:
        return month, months[month]
    prior = sorted([m for m in months.keys() if str(m) <= str(month)])
    if prior:
        key = prior[-1]
        return key, months[key]
    future = sorted([m for m in months.keys() if str(m) > str(month)])
    if future:
        # Bootstrap fallback: if we only have a cold-start calibrator trained on
        # the immediately preceding resolved window, allow the earliest fitted
        # month to service the current live month instead of forcing identity.
        key = future[0]
        return key, months[key]
    return "", None


def apply_selected_board_calibration(
    frame: pd.DataFrame,
    payload: dict[str, Any] | None,
    run_date_hint: str | None = None,
    prob_col: str = "board_play_win_prob",
    target_col: str = "target",
    direction_col: str = "direction",
) -> tuple[pd.Series, pd.Series, str]:
    if frame.empty:
        return (
            pd.Series(dtype="float64", index=frame.index),
            pd.Series(dtype="object", index=frame.index),
            "",
        )
    probs = pd.to_numeric(frame.get(prob_col), errors="coerce").fillna(0.5).astype("float64")
    base = _clip_prob(probs.to_numpy(dtype="float64"), 0.01, 0.99)
    if not payload:
        return (
            pd.Series(base, index=frame.index, dtype="float64"),
            pd.Series("identity_no_payload", index=frame.index, dtype="object"),
            "",
        )

    month_hint = _month_token(run_date_hint)
    if not month_hint and "market_date" in frame.columns:
        month_hint = _month_token(pd.to_datetime(frame["market_date"], errors="coerce").max())
    if not month_hint:
        month_hint = datetime.utcnow().strftime("%Y-%m")

    resolved_month, month_payload = _resolve_month_payload(payload, month_hint)
    if not month_payload:
        return (
            pd.Series(base, index=frame.index, dtype="float64"),
            pd.Series("identity_no_month", index=frame.index, dtype="object"),
            resolved_month,
        )

    global_cal = month_payload.get("global")
    segment_calibrators = month_payload.get("segments", {}) if isinstance(month_payload.get("segments"), dict) else {}
    safety_profile = month_payload.get("safety_profile") if isinstance(month_payload, dict) else None
    targets = frame.get(target_col, pd.Series("", index=frame.index)).astype(str).str.upper().str.strip()
    directions = frame.get(direction_col, pd.Series("", index=frame.index)).astype(str).str.upper().str.strip()
    seg_keys = targets + "_" + directions

    calibrated = np.asarray(base, dtype="float64").copy()
    sources = np.full(len(frame), "identity", dtype=object)
    if global_cal:
        calibrated = apply_monotonic_bin_calibrator(calibrated, global_cal)
        sources[:] = "global"

    for key, cal in segment_calibrators.items():
        mask = (seg_keys == str(key)).to_numpy(dtype=bool)
        if not np.any(mask):
            continue
        calibrated[mask] = apply_monotonic_bin_calibrator(calibrated[mask], cal)
        sources[mask] = f"segment:{key}"

    calibrated, recent_sources, recent_applied = _apply_recent_regime_adjustment(
        calibrated,
        seg_keys,
        month_payload if isinstance(month_payload, dict) else {},
        payload.get("config") if isinstance(payload, dict) else None,
    )
    if recent_applied:
        sources = np.asarray(
            [
                (
                    f"{str(source)}+recent:{recent_source}"
                    if str(source) != "identity"
                    else f"recent:{recent_source}"
                )
                if recent_source
                else str(source)
                for source, recent_source in zip(sources, recent_sources, strict=False)
            ],
            dtype=object,
        )

    calibrated, safety_applied = apply_empirical_safety_profile(calibrated, safety_profile)
    if safety_applied:
        sources = np.asarray(
            [
                f"{str(source)}+safety" if str(source) != "identity" else "safety"
                for source in sources
            ],
            dtype=object,
        )

    return (
        pd.Series(_clip_prob(calibrated, 0.01, 0.99), index=frame.index, dtype="float64"),
        pd.Series(sources, index=frame.index, dtype="object"),
        resolved_month,
    )


def evaluate_calibration(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = _clip_prob(probs, 1e-6, 1.0 - 1e-6)
    y = np.asarray(labels, dtype="float64")
    return {
        "rows": float(len(p)),
        "mean_prob": float(np.mean(p)) if len(p) else np.nan,
        "mean_label": float(np.mean(y)) if len(y) else np.nan,
        "gap": float(np.mean(y) - np.mean(p)) if len(p) else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(p) else np.nan,
        "log_loss": _log_loss_binary(p, y) if len(p) else np.nan,
        "ece_10": _ece_binary(p, y, n_bins=10) if len(p) else np.nan,
    }
