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

        payload["months"][month] = {
            "train_rows": int(len(train)),
            "train_start": lookback_start.strftime("%Y-%m-%d"),
            "train_end": (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "global": global_cal,
            "segments": segments,
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
