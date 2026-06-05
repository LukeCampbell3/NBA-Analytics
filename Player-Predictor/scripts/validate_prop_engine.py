#!/usr/bin/env python3
"""
Paired shadow validation for the v9 NBA prop engine.

Compares:
  - current no-vig market baseline
  - optional closing no-vig baseline when closing fields are available
  - v9 raw probabilities
  - v9 calibrated probabilities
  - v9 calibrated + selection gate

The output is intentionally segment-heavy. Global Brier is useful, but the
promotion decision should be made on calibrated, gated, out-of-sample pockets.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TRAINING_DIR = ROOT / "training"
sys.path.insert(0, str(TRAINING_DIR))

from nba_v9_prop_engine import american_to_decimal, no_vig_probs
from nba_v10_probability_stack import (
    build_blender_frame,
    clip_prob,
    fit_predict_v10,
    fit_v10_probability_stack,
    logit,
    sigmoid,
    _categorical_columns,
    _numeric_columns,
)

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover
    IsotonicRegression = None


DEFAULT_PROMOTION_GATES = {
    "min_resolved": 1500,
    "max_ece": 0.045,
    "brier_must_beat_market": True,
    "min_low_uncertainty_roi": 0.03,
    "max_high_uncertainty_roi": 0.00,
    "min_clv_correlation": 0.10,
    "no_market_side_collapse": True,
    "no_single_player_dependency": True,
    "no_single_market_dependency": True,
}

FORBIDDEN_FEATURE_TOKENS = [
    "actual",
    "result",
    "hit",
    "win",
    "loss",
    "push",
    "residual",
    "abs_residual",
    "error",
    "settled",
    "outcome",
    "postgame",
    "final",
    "box_score_after",
    "stat_actual",
    "actual_stat",
    "minutes",
    "usage",
    "fga",
    "rebounds",
    "assists",
]


def _resolve_manifest_path(path_text: str, manifest_path: Path) -> Path:
    path_text = str(path_text)
    if path_text.startswith("/workspace/"):
        return REPO_ROOT / path_text.replace("/workspace/", "", 1)
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve() if not path.exists() else path.resolve()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(manifest: dict, manifest_path: Path) -> pd.DataFrame:
    if manifest.get("model_version") == "prop_engine_v10":
        source = manifest.get("source_v9_manifest")
        if not source:
            raise FileNotFoundError("v10 manifest is missing source_v9_manifest")
        source_path = Path(source)
        if str(source).startswith("/workspace/"):
            source_path = REPO_ROOT / str(source).replace("/workspace/", "", 1)
        elif not source_path.is_absolute():
            source_path = (manifest_path.parent / source_path).resolve() if not source_path.exists() else source_path.resolve()
        source_manifest = _load_manifest(source_path)
        return _load_rows(source_manifest, source_path)

    output_dir = _resolve_manifest_path(manifest.get("output", manifest_path.parent), manifest_path)
    if manifest.get("model_version") == "prop_engine_v9_1_honest_distribution_baseline":
        source = manifest.get("source_v9_manifest")
        if source:
            source_path = Path(source)
            if str(source).startswith("/workspace/"):
                source_path = REPO_ROOT / str(source).replace("/workspace/", "", 1)
            elif not source_path.is_absolute():
                source_path = (manifest_path.parent / source_path).resolve() if not source_path.exists() else source_path.resolve()
            source_manifest = _load_manifest(source_path)
            return _load_rows(source_manifest, source_path)
    candidates = [
        output_dir / "data" / "prop_training_rows.csv",
        manifest_path.parent / "data" / "prop_training_rows.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            rows = pd.read_csv(candidate)
            rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
            return rows.dropna(subset=["date"]).copy()
    raise FileNotFoundError("Could not find data/prop_training_rows.csv beside v9 artifacts")


def _load_calibrator(manifest: dict, manifest_path: Path, market: str):
    output_dir = _resolve_manifest_path(manifest.get("output", manifest_path.parent), manifest_path)
    cal_dir = output_dir / "calibration"
    market_path = cal_dir / f"{market}_adaptive_calibrator.pkl"
    global_path = cal_dir / "global_adaptive_calibrator.pkl"
    path = market_path if market_path.exists() else global_path
    if not path.exists():
        return None, "raw"
    data = joblib.load(str(path))
    return data.get("isotonic"), market if path == market_path else "GLOBAL"


def _fit_validation_calibrator(train_rows: pd.DataFrame, market: str, min_samples: int = 30):
    if IsotonicRegression is None or train_rows.empty:
        return None, "raw"
    market_rows = train_rows[train_rows["market"] == market]
    if len(market_rows) >= min_samples:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(_clip_prob(market_rows["p_over_raw"]), market_rows["result_over"].to_numpy())
        return iso, f"{market}_prefit"
    if len(train_rows) >= min_samples:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(_clip_prob(train_rows["p_over_raw"]), train_rows["result_over"].to_numpy())
        return iso, "GLOBAL_prefit"
    return None, "raw"


def _clip_prob(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).clip(0.001, 0.999)


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2)) if len(outcomes) else 0.0


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    if len(outcomes) == 0:
        return 0.0
    p = _clip_prob(probs)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    if len(outcomes) == 0:
        return 0.0
    p = _clip_prob(probs)
    y = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    ece = 0.0
    for idx in range(n_bins):
        mask = (p >= bins[idx]) & (p < bins[idx + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _profit_for_side(row: pd.Series, side_col: str = "selected_side") -> float:
    side = row.get(side_col, "OVER")
    if side == "OVER":
        won = row["result_over"] == 1
        odds = row.get("over_odds", -110)
    else:
        won = row["result_over"] == 0 and row.get("push", 0) == 0
        odds = row.get("under_odds", -110)
    if row.get("push", 0) == 1:
        return 0.0
    if pd.isna(odds):
        odds = -110
    return american_to_decimal(odds) - 1.0 if won else -1.0


def _closing_odds_columns(rows: pd.DataFrame) -> tuple[str | None, str | None]:
    over_col = "close_over_odds" if "close_over_odds" in rows.columns else "closing_over_odds" if "closing_over_odds" in rows.columns else None
    under_col = "close_under_odds" if "close_under_odds" in rows.columns else "closing_under_odds" if "closing_under_odds" in rows.columns else None
    return over_col, under_col


def _add_market_tracking_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    close_over_col, close_under_col = _closing_odds_columns(rows)
    if close_over_col and close_under_col and rows[close_over_col].notna().any() and rows[close_under_col].notna().any():
        valid_close = rows[close_over_col].notna() & rows[close_under_col].notna()
        rows["close_no_vig_over"] = np.nan
        rows["close_no_vig_under"] = np.nan
        pairs = rows.loc[valid_close].apply(lambda r: no_vig_probs(r[close_over_col], r[close_under_col]), axis=1)
        if len(pairs):
            over_probs, under_probs = zip(*pairs)
            rows.loc[valid_close, "close_no_vig_over"] = over_probs
            rows.loc[valid_close, "close_no_vig_under"] = under_probs
        rows["closing_no_vig_over"] = rows["close_no_vig_over"]
        rows["closing_no_vig_under"] = rows["close_no_vig_under"]
    if "current_line" not in rows.columns:
        rows["current_line"] = rows["line"]
    if "close_line" not in rows.columns and "closing_line" in rows.columns:
        rows["close_line"] = rows["closing_line"]
    if {"close_no_vig_over", "close_no_vig_under"}.issubset(rows.columns):
        rows["clv_no_vig_over"] = rows["close_no_vig_over"] - rows["market_no_vig_over"]
        rows["clv_no_vig_under"] = rows["close_no_vig_under"] - rows["market_no_vig_under"]
        rows["clv_no_vig_selected"] = np.where(rows["selected_side"] == "OVER", rows["clv_no_vig_over"], rows["clv_no_vig_under"])
    if "close_line" in rows.columns:
        rows["clv_line_selected"] = np.where(
            rows["selected_side"] == "OVER",
            rows["close_line"].astype(float) - rows["current_line"].astype(float),
            rows["current_line"].astype(float) - rows["close_line"].astype(float),
        )
    return rows


def _metrics(frame: pd.DataFrame, prob_col: str, outcome_col: str = "result_over", baseline_brier: float = 0.25) -> dict:
    if frame.empty or prob_col not in frame.columns or outcome_col not in frame.columns:
        return {
            "n": 0,
            "brier_skill_score": 0.0,
            "brier": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "roi_shadow": 0.0,
            "hit_rate": 0.0,
        }
    frame = frame.copy()
    frame[prob_col] = pd.to_numeric(frame[prob_col], errors="coerce")
    frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors="coerce")
    frame = frame.dropna(subset=[prob_col, outcome_col])
    if frame.empty:
        return {
            "n": 0,
            "brier": 0.0,
            "brier_skill_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "roi_shadow": 0.0,
            "hit_rate": 0.0,
        }
    probs = frame[prob_col].to_numpy()
    outcomes = frame[outcome_col].to_numpy()
    brier = _brier(probs, outcomes)
    return {
        "n": int(len(frame)),
        "brier": brier,
        "brier_skill_score": float(1.0 - brier / baseline_brier) if baseline_brier > 0 else 0.0,
        "log_loss": _log_loss(probs, outcomes),
        "ece": _ece(probs, outcomes),
        "roi_shadow": float(frame["profit"].mean()) if "profit" in frame.columns else 0.0,
        "hit_rate": float(outcomes.mean()),
    }


def _market_source_report(rows: pd.DataFrame) -> dict:
    required_current = {"over_odds", "under_odds", "market_no_vig_over", "market_no_vig_under"}
    close_over_col, close_under_col = _closing_odds_columns(rows)
    has_current = required_current.issubset(rows.columns)
    has_close = bool(close_over_col and close_under_col and rows[close_over_col].notna().any() and rows[close_under_col].notna().any())
    odds_pairs = rows[["over_odds", "under_odds"]].dropna().drop_duplicates() if {"over_odds", "under_odds"}.issubset(rows.columns) else pd.DataFrame()
    neutral_only = bool(len(odds_pairs) == 1 and float(odds_pairs.iloc[0]["over_odds"]) == -110 and float(odds_pairs.iloc[0]["under_odds"]) == -110) if len(odds_pairs) else True
    close_status_counts = rows["close_status"].dropna().value_counts().to_dict() if "close_status" in rows.columns else {}
    reliable_clv_rows = int(rows["close_status"].fillna("").eq("same_day_latest_snapshot_proxy_commence_missing").sum()) if "close_status" in rows.columns else 0
    return {
        "current_odds_available": bool(has_current),
        "closing_odds_available": has_close,
        "real_market_probability_available": bool(has_current and not neutral_only),
        "neutral_110_only": neutral_only,
        "books": sorted(str(v) for v in rows["book"].dropna().unique()) if "book" in rows.columns else [],
        "snapshot_count": int(rows["snapshot_time"].notna().sum()) if "snapshot_time" in rows.columns else 0,
        "close_over_odds_column": close_over_col,
        "close_under_odds_column": close_under_col,
        "close_status_counts": close_status_counts,
        "clv_reliable_rows": reliable_clv_rows,
        "clv_reliability": "limited_same_day_proxy" if reliable_clv_rows else "unavailable_or_archived_only",
    }


def _true_market_subset_report(rows: pd.DataFrame, gated: pd.DataFrame) -> dict:
    if not {"book", "market_no_vig_over", "market_no_vig_under"}.issubset(rows.columns):
        return {"available": False, "reason": "book/no-vig columns unavailable"}
    market_rows = rows[rows["book"].notna()].copy()
    if market_rows.empty:
        return {"available": False, "reason": "no rows matched true market snapshots"}
    market_rows["profit"] = market_rows.apply(_profit_for_side, axis=1)
    market_gated = gated[gated.index.isin(market_rows.index)].copy()
    if not market_gated.empty:
        market_gated["profit"] = market_gated.apply(_profit_for_side, axis=1)
    report = {
        "available": True,
        "rows": int(len(market_rows)),
        "gated_rows": int(len(market_gated)),
        "model": _metrics(market_rows.assign(profit=0.0), "p_over_calibrated"),
        "market_no_vig": _metrics(market_rows.assign(profit=0.0), "market_no_vig_over"),
        "side_prior": _metrics(market_rows.assign(profit=0.0), "side_prior_over") if "side_prior_over" in market_rows.columns else None,
        "gated_model": _metrics(market_gated, "p_selected", outcome_col="selected_outcome") if len(market_gated) else None,
        "books": sorted(str(v) for v in market_rows["book"].dropna().unique()),
    }
    report["model_bss_vs_market"] = (
        float(1.0 - report["model"]["brier"] / report["market_no_vig"]["brier"])
        if report["market_no_vig"]["brier"] > 0
        else None
    )
    return report


def _bucketize(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["abs_edge"] = rows["edge"].abs()
    rows["edge_bucket"] = pd.cut(
        rows["abs_edge"],
        bins=[-0.001, 0.02, 0.04, 0.06, 0.08, 1.0],
        labels=["0-2", "2-4", "4-6", "6-8", "8+"],
    ).astype(str)
    rows["uncertainty_bucket"] = pd.cut(
        rows["uncertainty"],
        bins=[-0.001, 0.35, 0.55, 0.70, 1.0],
        labels=["low", "medium", "high", "extreme"],
    ).astype(str)
    rows["line_bucket"] = pd.cut(
        rows["line"],
        bins=[-0.001, 3.5, 7.5, 12.5, 20.5, 99],
        labels=["tiny", "small", "medium", "large", "star"],
    ).astype(str)
    rows["odds_bucket"] = "standard"
    rows.loc[rows["over_odds"] <= -125, "odds_bucket"] = "over_taxed"
    rows.loc[rows["under_odds"] <= -125, "odds_bucket"] = "under_taxed"
    player_counts = rows.groupby("player")["player"].transform("size")
    rows["player_volume_bucket"] = pd.cut(
        player_counts,
        bins=[0, 20, 60, 120, 100000],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)
    if "minutes" in rows.columns:
        minutes_proxy = rows.groupby("player")["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
        rows["minutes_volatility_bucket"] = pd.cut(
            minutes_proxy.fillna(minutes_proxy.median()).fillna(0.0),
            bins=[-0.001, 3, 6, 10, 100],
            labels=["low", "medium", "high", "extreme"],
        ).astype(str)
    else:
        rows["minutes_volatility_bucket"] = "unknown"

    uncertainty_source = rows["belief_uncertainty"] if "belief_uncertainty" in rows.columns else rows["uncertainty"]
    rows["regime"] = "normal_market"
    rows.loc[uncertainty_source >= uncertainty_source.quantile(0.75), "regime"] = "uncertainty_risk"
    if "minutes" in rows.columns:
        rows.loc[rows["minutes"] <= rows["minutes"].quantile(0.20), "regime"] = "minutes_risk"
    rows.loc[(rows["abs_edge"] >= 0.08) & (rows["uncertainty"] <= 0.55), "regime"] = "line_lag"
    if "book" in rows.columns:
        book_counts = rows.groupby(["date", "player", "market"])["book"].transform("nunique")
        rows["book_disagreement_bucket"] = np.where(book_counts > 1, "multi_book", "single_book")
    else:
        rows["book_disagreement_bucket"] = "unavailable"
    return rows


def _prepare_rows(
    rows: pd.DataFrame,
    manifest: dict,
    manifest_path: Path,
    calibration_train_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows = rows.copy()
    for odds_col in ["over_odds", "under_odds"]:
        if odds_col not in rows.columns:
            rows[odds_col] = -110
    if "market_no_vig_over" not in rows.columns:
        pairs = rows.apply(lambda r: no_vig_probs(r["over_odds"], r["under_odds"]), axis=1)
        rows["market_no_vig_over"], rows["market_no_vig_under"] = zip(*pairs)
    if "market_no_vig_under" not in rows.columns:
        rows["market_no_vig_under"] = 1.0 - rows["market_no_vig_over"]

    rows["p_over_raw"] = _clip_prob(rows["p_over_raw"])
    rows["p_over_calibrated"] = rows["p_over_raw"]
    rows["calibration_source"] = "raw"
    source_counts: dict[str, int] = {}
    for market, idx in rows.groupby("market").groups.items():
        if calibration_train_rows is not None and not calibration_train_rows.empty:
            iso, source = _fit_validation_calibrator(calibration_train_rows, str(market))
        else:
            iso, source = _load_calibrator(manifest, manifest_path, str(market))
        source_counts[source] = source_counts.get(source, 0) + len(idx)
        if iso is not None:
            rows.loc[idx, "p_over_calibrated"] = iso.predict(rows.loc[idx, "p_over_raw"].to_numpy()).clip(0.01, 0.99)
        rows.loc[idx, "calibration_source"] = source

    rows["p_under_calibrated"] = 1.0 - rows["p_over_calibrated"]
    rows["edge_over"] = rows["p_over_calibrated"] - rows["market_no_vig_over"]
    rows["edge_under"] = rows["p_under_calibrated"] - rows["market_no_vig_under"]
    rows["selected_side"] = np.where(rows["edge_over"] >= rows["edge_under"], "OVER", "UNDER")
    rows["side"] = rows["selected_side"]
    rows["p_selected"] = np.where(rows["selected_side"] == "OVER", rows["p_over_calibrated"], rows["p_under_calibrated"])
    rows["market_selected"] = np.where(rows["selected_side"] == "OVER", rows["market_no_vig_over"], rows["market_no_vig_under"])
    rows["edge"] = rows["p_selected"] - rows["market_selected"]
    uncertainty_raw = rows.get("belief_uncertainty", pd.Series(0.5, index=rows.index)).astype(float)
    rows["uncertainty_raw"] = uncertainty_raw
    if uncertainty_raw.nunique(dropna=True) > 1:
        # Operational uncertainty is a rank-normalized risk score so the
        # production cap behaves consistently across model generations.
        rows["uncertainty"] = uncertainty_raw.rank(pct=True, method="average").astype(float)
    else:
        rows["uncertainty"] = 0.5
    rows["profit"] = rows.apply(_profit_for_side, axis=1)
    rows["selected_outcome"] = np.where(rows["selected_side"] == "OVER", rows["result_over"], 1.0 - rows["result_over"])
    rows["calibration_delta"] = rows["p_over_calibrated"] - rows["p_over_raw"]
    if "side_prior_over" not in rows.columns and calibration_train_rows is not None and not calibration_train_rows.empty:
        train = calibration_train_rows.copy()
        train["line_bucket"] = pd.cut(
            train["line"],
            bins=[-0.001, 3.5, 7.5, 12.5, 20.5, 99],
            labels=["tiny", "small", "medium", "large", "star"],
        ).astype(str)
        rows["line_bucket"] = pd.cut(
            rows["line"],
            bins=[-0.001, 3.5, 7.5, 12.5, 20.5, 99],
            labels=["tiny", "small", "medium", "large", "star"],
        ).astype(str)
        market_side = train.groupby("market").agg(n=("result_over", "size"), over_rate=("result_over", "mean"))
        market_line = train.groupby(["market", "line_bucket"]).agg(n=("result_over", "size"), over_rate=("result_over", "mean"))
        rows = rows.merge(
            market_side.rename(columns={"n": "side_prior_market_n", "over_rate": "side_prior_market_raw"}),
            left_on="market",
            right_index=True,
            how="left",
        )
        rows = rows.merge(
            market_line.rename(columns={"n": "side_prior_line_n", "over_rate": "side_prior_line_raw"}),
            left_on=["market", "line_bucket"],
            right_index=True,
            how="left",
        )
        shrink_k = 300.0
        market_n = rows["side_prior_market_n"].fillna(0.0)
        line_n = rows["side_prior_line_n"].fillna(0.0)
        market_prior = 0.5 + (rows["side_prior_market_raw"].fillna(0.5) - 0.5) * market_n / (market_n + shrink_k)
        line_prior = 0.5 + (rows["side_prior_line_raw"].fillna(0.5) - 0.5) * line_n / (line_n + shrink_k)
        rows["side_prior_over"] = (0.65 * market_prior + 0.35 * line_prior).clip(0.01, 0.99)
    rows = _add_market_tracking_columns(rows)
    rows = _bucketize(rows)
    return rows, source_counts


def _apply_calibration_blend(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Blend raw and calibrated probabilities, then recompute side/edge fields."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 0.999:
        return rows
    rows = rows.copy()
    rows["p_over_calibrated_prefit"] = rows["p_over_calibrated"]
    rows["p_over_calibrated"] = ((1.0 - alpha) * rows["p_over_raw"] + alpha * rows["p_over_calibrated_prefit"]).clip(0.01, 0.99)
    rows["p_under_calibrated"] = 1.0 - rows["p_over_calibrated"]
    rows["edge_over"] = rows["p_over_calibrated"] - rows["market_no_vig_over"]
    rows["edge_under"] = rows["p_under_calibrated"] - rows["market_no_vig_under"]
    rows["selected_side"] = np.where(rows["edge_over"] >= rows["edge_under"], "OVER", "UNDER")
    rows["side"] = rows["selected_side"]
    rows["p_selected"] = np.where(rows["selected_side"] == "OVER", rows["p_over_calibrated"], rows["p_under_calibrated"])
    rows["market_selected"] = np.where(rows["selected_side"] == "OVER", rows["market_no_vig_over"], rows["market_no_vig_under"])
    rows["edge"] = rows["p_selected"] - rows["market_selected"]
    rows["profit"] = rows.apply(_profit_for_side, axis=1)
    rows["selected_outcome"] = np.where(rows["selected_side"] == "OVER", rows["result_over"], 1.0 - rows["result_over"])
    rows["calibration_delta"] = rows["p_over_calibrated"] - rows["p_over_raw"]
    rows = _add_market_tracking_columns(rows)
    rows = _bucketize(rows)
    return rows


def _comparison_table(rows: pd.DataFrame, gated: pd.DataFrame, include_closing: bool) -> dict:
    market_frame = rows.copy()
    market_frame["profit"] = 0.0
    table = {
        "current_market_no_vig": _metrics(market_frame, "market_no_vig_over"),
        "side_prior": _metrics(rows.assign(profit=0.0), "side_prior_over") if "side_prior_over" in rows.columns else {"available": False},
        "v9_raw": _metrics(rows.assign(profit=0.0), "p_over_raw"),
        "v9_calibrated": _metrics(rows.assign(profit=0.0), "p_v9_calibrated")
        if "p_v9_calibrated" in rows.columns
        else _metrics(rows.assign(profit=0.0), "p_over_calibrated"),
        "v9_calibrated_gate": _metrics(gated, "p_v9_selected", outcome_col="v9_selected_outcome")
        if "p_v9_selected" in gated.columns
        else _metrics(gated, "p_selected", outcome_col="selected_outcome"),
    }
    if "p_v10_raw" in rows.columns:
        table["v10_raw"] = _metrics(rows.assign(profit=0.0), "p_v10_raw")
        if "p_v10_calibrated" in rows.columns:
            table["v10_calibrated"] = _metrics(rows.assign(profit=0.0), "p_v10_calibrated")
        table["v10_calibrated_gate"] = _metrics(gated, "p_selected", outcome_col="selected_outcome")
    if include_closing:
        close_over_col, close_under_col = _closing_odds_columns(rows)
        if close_over_col and close_under_col and rows[close_over_col].notna().any() and rows[close_under_col].notna().any():
            closing = rows.copy()
            valid_close = closing[close_over_col].notna() & closing[close_under_col].notna()
            closing["closing_no_vig_over"] = np.nan
            pairs = closing.loc[valid_close].apply(lambda r: no_vig_probs(r[close_over_col], r[close_under_col]), axis=1)
            if len(pairs):
                closing.loc[valid_close, "closing_no_vig_over"] = [pair[0] for pair in pairs]
            table["closing_market_no_vig"] = _metrics(closing, "closing_no_vig_over")
        else:
            table["closing_market_no_vig"] = {
                "available": False,
                "reason": "closing odds columns were absent or empty in validation rows",
            }
    return table


def _clv_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty or "clv_no_vig_selected" not in frame.columns:
        return {"available": False, "n": int(len(frame))}
    clv = pd.to_numeric(frame["clv_no_vig_selected"], errors="coerce").dropna()
    if clv.empty:
        return {"available": False, "n": int(len(frame))}
    edge = pd.to_numeric(frame.loc[clv.index, "edge"], errors="coerce")
    corr = float(edge.corr(clv)) if len(clv) >= 3 and edge.nunique(dropna=True) > 1 and clv.nunique(dropna=True) > 1 else None
    out = {
        "available": True,
        "n": int(len(clv)),
        "avg_clv_no_vig": float(clv.mean()),
        "median_clv_no_vig": float(clv.median()),
        "positive_clv_rate": float((clv > 0).mean()),
        "clv_edge_correlation": corr,
        "roi_actual_odds": float(frame.loc[clv.index, "profit"].mean()) if "profit" in frame.columns else None,
    }
    if "clv_line_selected" in frame.columns:
        line_clv = pd.to_numeric(frame.loc[clv.index, "clv_line_selected"], errors="coerce").dropna()
        if not line_clv.empty:
            out["avg_clv_line"] = float(line_clv.mean())
            out["positive_line_clv_rate"] = float((line_clv > 0).mean())
    return out


def _clv_segment_report(rows: pd.DataFrame, gated: pd.DataFrame, min_rows: int) -> dict:
    report = {
        "all": _clv_metrics(rows),
        "gated": _clv_metrics(gated),
    }
    for segment in ["side", "market", "edge_bucket"]:
        parts = []
        if segment in gated.columns and "clv_no_vig_selected" in gated.columns:
            for key, frame in gated.groupby(segment, dropna=False):
                if len(frame) < min_rows:
                    continue
                metrics = _clv_metrics(frame)
                metrics["segment_value"] = str(key)
                parts.append(metrics)
        report[f"gated_by_{segment}"] = sorted(parts, key=lambda item: item.get("n", 0), reverse=True)
    return report


def _segment_report(rows: pd.DataFrame, segment_cols: list[str], min_rows: int) -> dict:
    report = {}
    for segment in segment_cols:
        if segment not in rows.columns:
            report[segment] = {"available": False}
            continue
        parts = []
        for key, frame in rows.groupby(segment, dropna=False):
            if len(frame) < min_rows:
                continue
            metrics = _metrics(frame, "p_selected", outcome_col="selected_outcome")
            metrics["segment_value"] = str(key)
            metrics["avg_edge"] = float(frame["edge"].mean())
            metrics["avg_uncertainty"] = float(frame["uncertainty"].mean())
            parts.append(metrics)
        report[segment] = sorted(parts, key=lambda x: x["n"], reverse=True)

    combo_cols = [c for c in ["edge_bucket", "uncertainty_bucket", "market", "side"] if c in rows.columns]
    combo = []
    if len(combo_cols) == 4:
        for keys, frame in rows.groupby(combo_cols, dropna=False):
            if len(frame) < min_rows:
                continue
            metrics = _metrics(frame, "p_selected", outcome_col="selected_outcome")
            metrics.update({col: str(value) for col, value in zip(combo_cols, keys)})
            metrics["avg_edge"] = float(frame["edge"].mean())
            combo.append(metrics)
    report["edge_bucket_x_uncertainty_bucket_x_market_x_side"] = sorted(
        combo,
        key=lambda x: (x.get("roi_shadow", 0.0), x.get("n", 0)),
        reverse=True,
    )
    return report


def _calibration_diagnostics(rows: pd.DataFrame, source_counts: dict) -> dict:
    by_source = {}
    for source, frame in rows.groupby("calibration_source"):
        by_source[source] = {
            "n": int(len(frame)),
            "fallback_rate": float(len(frame) / len(rows)),
            "raw_mean": float(frame["p_over_raw"].mean()),
            "calibrated_mean": float(frame["p_over_calibrated"].mean()),
            "avg_delta": float(frame["calibration_delta"].mean()),
            "abs_delta_mean": float(frame["calibration_delta"].abs().mean()),
        }
    return {
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "raw_distribution": rows["p_over_raw"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "calibrated_distribution": rows["p_over_calibrated"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "calibration_delta_distribution": rows["calibration_delta"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "by_source": by_source,
    }


def _promotion_gates(rows: pd.DataFrame, gated: pd.DataFrame, comparison: dict) -> dict:
    gates = dict(DEFAULT_PROMOTION_GATES)
    low_uncertainty = gated[gated["uncertainty_bucket"] == "low"]
    high_uncertainty = gated[gated["uncertainty_bucket"].isin(["high", "extreme"])]
    side_share = gated["side"].value_counts(normalize=True).to_dict() if len(gated) else {}
    player_share = float(gated["player"].value_counts(normalize=True).iloc[0]) if len(gated) else 1.0
    market_share = float(gated["market"].value_counts(normalize=True).iloc[0]) if len(gated) else 1.0
    clv = _clv_metrics(gated)

    model_key = "v10_calibrated" if "v10_calibrated" in comparison else "v9_calibrated"
    gate_key = "v10_calibrated_gate" if "v10_calibrated_gate" in comparison else "v9_calibrated_gate"
    checks = {
        "min_resolved": len(rows) >= gates["min_resolved"],
        "max_ece": comparison[model_key]["ece"] <= gates["max_ece"],
        "brier_must_beat_market": comparison[model_key]["brier"] < comparison["current_market_no_vig"]["brier"],
        "brier_must_beat_side_prior": (
            comparison[model_key]["brier"] < comparison["side_prior"]["brier"]
            if isinstance(comparison.get("side_prior"), dict) and "brier" in comparison["side_prior"]
            else None
        ),
        "min_low_uncertainty_roi": (not low_uncertainty.empty) and float(low_uncertainty["profit"].mean()) >= gates["min_low_uncertainty_roi"],
        "max_high_uncertainty_roi": high_uncertainty.empty or float(high_uncertainty["profit"].mean()) <= gates["max_high_uncertainty_roi"],
        "min_clv_correlation": (
            clv["clv_edge_correlation"] >= gates["min_clv_correlation"]
            if clv.get("available") and clv.get("clv_edge_correlation") is not None
            else None
        ),
        "no_market_side_collapse": all(value <= 0.70 for value in side_share.values()) if side_share else False,
        "no_single_player_dependency": player_share <= 0.10,
        "no_single_market_dependency": market_share <= 0.60,
    }
    passed = all(value is True for value in checks.values() if value is not None)
    return {
        "status": "pass" if passed else "fail",
        "target_status": "shadow_candidate" if passed else "not_promoted",
        "gates": gates,
        "checks": checks,
        "diagnostics": {
            "gated_count": int(len(gated)),
            "low_uncertainty_roi": float(low_uncertainty["profit"].mean()) if not low_uncertainty.empty else None,
            "high_uncertainty_roi": float(high_uncertainty["profit"].mean()) if not high_uncertainty.empty else None,
            "side_share": {str(k): float(v) for k, v in side_share.items()},
            "largest_player_share": player_share,
            "largest_market_share": market_share,
            "clv_available": bool(clv.get("available")),
            "clv": clv,
            "model_key": model_key,
            "gate_key": gate_key,
        },
    }


def _fit_prefit_isotonic(train_rows: pd.DataFrame, raw_col: str):
    if IsotonicRegression is None or train_rows.empty or raw_col not in train_rows.columns:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(_clip_prob(train_rows[raw_col]), train_rows["result_over"].to_numpy())
    return iso


def _apply_v10_stack(
    rows: pd.DataFrame,
    calibration_train_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calibration_train_rows.empty:
        return rows, calibration_train_rows
    _, scored = fit_predict_v10(calibration_train_rows, rows)
    _, train_scored = fit_predict_v10(calibration_train_rows, calibration_train_rows)
    # The v10 stack is intentionally higher-resolution than v9. Use a
    # conservative temperature shrink for the shadow validator so resolution
    # gains do not become uncalibrated 0.01/0.99 confidence.
    shrink = 0.75
    scored["p_v10_calibrated"] = (0.5 + (scored["p_v10_raw"] - 0.5) * shrink).clip(0.01, 0.99)
    train_scored["p_v10_calibrated"] = (0.5 + (train_scored["p_v10_raw"] - 0.5) * shrink).clip(0.01, 0.99)
    scored["v10_calibration_method"] = f"temperature_shrink_{shrink:.2f}"
    train_scored["v10_calibration_method"] = f"temperature_shrink_{shrink:.2f}"
    return scored, train_scored


def _component_cutoff_report(
    manifest: dict,
    rows: pd.DataFrame,
    calibration_train_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    validation_start = str(pd.Timestamp(args.start).date()) if args.start else str(rows["date"].min().date())
    training_cutoff = None
    if not calibration_train_rows.empty:
        training_cutoff = str(calibration_train_rows["date"].max().date())
    cutoff_safe = bool(training_cutoff and pd.Timestamp(training_cutoff) < pd.Timestamp(validation_start))
    is_v10 = manifest.get("model_version") == "prop_engine_v10"
    is_v92 = manifest.get("model_version") == "prop_engine_v9_2_minutes_adjusted_distribution"
    is_v93 = manifest.get("model_version") == "prop_engine_v9_3_market_validated_distribution"
    is_v94 = manifest.get("model_version") in {
        "prop_engine_v9_4_lineup_delta_ready_distribution",
        "prop_engine_v9_4_oracle_lineup_adjusted_distribution",
    }
    is_v95 = manifest.get("model_version") == "prop_engine_v9_5_pregame_lineup_distribution"
    return {
        "training_cutoff": training_cutoff,
        "validation_start": validation_start,
        "model_components_policy": (
            "prefit_before_start_inside_validator"
            if is_v10
            else "walk_forward_projected_minutes_plus_prefit_calibrator"
            if is_v92 or is_v93 or is_v94 or is_v95
            else "v9_artifact_plus_prefit_calibrator"
        ),
        "model_components_cutoff_safe": cutoff_safe if (is_v10 or is_v92 or is_v93 or is_v94 or is_v95) else None,
        "calibrator_cutoff_safe": cutoff_safe,
        "side_prior_cutoff_safe": cutoff_safe if (is_v10 or is_v92 or is_v93 or is_v94 or is_v95) else None,
        "blender_cutoff_safe": cutoff_safe if is_v10 else None,
        "risk_model_cutoff_safe": cutoff_safe if is_v10 else None,
        "status": "cutoff_safe" if (cutoff_safe or not args.start) else "not_shadow_safe",
    }


def _audit_leakage_report(rows: pd.DataFrame, model_feature_columns: list[str]) -> dict:
    feature_hits = {}
    oracle_features = [col for col in model_feature_columns if "oracle" in col.lower()]
    for col in model_feature_columns:
        lower = col.lower()
        hits = [token for token in FORBIDDEN_FEATURE_TOKENS if token in lower]
        if lower.startswith("projected_minutes") or lower.startswith("pregame_minutes"):
            hits = [token for token in hits if token != "minutes"]
        if lower in {"minutes_roll5_mean_shifted", "minutes_roll10_mean_shifted"}:
            hits = [token for token in hits if token != "minutes"]
        if lower.startswith("v92_minutes"):
            hits = [token for token in hits if token != "minutes"]
        if lower == "p_over_raw_minutes_branch":
            hits = [token for token in hits if token != "minutes"]
        if lower.startswith("pregame_usage_"):
            hits = [token for token in hits if token != "usage"]
        if hits:
            feature_hits[col] = hits
    available_row_flags = {
        "has_actual_minutes_column": "minutes" in rows.columns,
        "has_actual_value_column": "actual_value" in rows.columns,
        "has_result_over_column": "result_over" in rows.columns,
        "has_push_column": "push" in rows.columns,
    }
    passed = len(feature_hits) == 0 and not oracle_features
    return {
        "status": "pass" if passed else "research_only_oracle_features" if oracle_features and not feature_hits else "fail",
        "model_feature_columns_checked": model_feature_columns,
        "forbidden_feature_hits": feature_hits,
        "oracle_feature_hits": oracle_features,
        "row_target_columns_present_but_not_model_features": available_row_flags,
        "note": (
            "Target/result columns may exist in validation rows; audit fails only if selected model features contain forbidden tokens. "
            "Oracle lineup features are retrospective research-only unless supplied by a pregame availability feed."
        ),
    }


def _ablation_report(
    train_rows: pd.DataFrame,
    rows: pd.DataFrame,
    scored_rows: pd.DataFrame,
    gated: pd.DataFrame,
) -> dict:
    if train_rows.empty or "p_v10_raw" not in scored_rows.columns:
        return {"available": False, "reason": "v10 scored rows and pre-start training rows are required"}

    # Branches are trained from pre-start rows by fit_predict_v10. Evaluate each
    # branch on the validation window and use the same edge/risk gate mechanics.
    variants = {
        "market_prior_only": "market_no_vig_over",
        "side_prior_only": "side_prior_over",
        "distribution_only": "p_over_raw",
        "direct_classifier_only": "p_direct",
        "market_residual_only": "p_market_residual",
        "full_v10_raw": "p_v10_raw",
        "full_v10_calibrated": "p_v10_calibrated",
    }

    report = {}
    for name, col in variants.items():
        if col not in scored_rows.columns:
            continue
        frame = scored_rows.copy()
        frame["variant_prob"] = clip_prob(frame[col])
        frame["variant_under"] = 1.0 - frame["variant_prob"]
        frame["variant_edge_over"] = frame["variant_prob"] - frame["market_no_vig_over"]
        frame["variant_edge_under"] = frame["variant_under"] - frame["market_no_vig_under"]
        frame["selected_side"] = np.where(frame["variant_edge_over"] >= frame["variant_edge_under"], "OVER", "UNDER")
        frame["p_selected"] = np.where(frame["selected_side"] == "OVER", frame["variant_prob"], frame["variant_under"])
        frame["selected_outcome"] = np.where(frame["selected_side"] == "OVER", frame["result_over"], 1.0 - frame["result_over"])
        frame["edge"] = np.maximum(frame["variant_edge_over"], frame["variant_edge_under"])
        frame["profit"] = frame.apply(_profit_for_side, axis=1)
        risk_col = "brier_risk" if "brier_risk" in frame.columns else "uncertainty"
        risk_cap = frame[risk_col].quantile(0.70) if risk_col == "brier_risk" else 0.70
        gated_variant = frame[(frame["edge"] >= 0.045) & (frame[risk_col] <= risk_cap)].copy()
        report[name] = {
            "all": _metrics(frame.assign(profit=0.0), "variant_prob"),
            "gated": _metrics(gated_variant, "p_selected", outcome_col="selected_outcome"),
            "gated_n": int(len(gated_variant)),
        }

    if {"p_distribution", "p_direct", "p_market_residual"}.issubset(scored_rows.columns):
        no_side = pd.DataFrame(
            {
                "logit_distribution": logit(scored_rows["p_distribution"]),
                "logit_direct": logit(scored_rows["p_direct"]),
                "logit_market_residual": logit(scored_rows["p_market_residual"]),
            }
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Simple no-side branch blend: average logits from the three non-prior branches.
        scored_rows = scored_rows.copy()
        scored_rows["p_full_no_side_prior"] = sigmoid(no_side.mean(axis=1))
        report["full_v10_no_side_prior"] = {
            "all": _metrics(scored_rows.assign(profit=0.0), "p_full_no_side_prior"),
        }
    return report


def _label_shuffle_report(train_rows: pd.DataFrame, rows: pd.DataFrame, random_state: int = 42) -> dict:
    if train_rows.empty:
        return {"available": False, "reason": "pre-start training rows are required"}
    shuffled = train_rows.copy()
    rng = np.random.default_rng(random_state)
    shuffled["result_over"] = (
        shuffled.groupby(["market", "date"])["result_over"]
        .transform(lambda s: rng.permutation(s.to_numpy()))
        .astype(float)
    )
    _, scored = fit_predict_v10(shuffled, rows)
    p = 0.5 + (scored["p_v10_raw"] - 0.5) * 0.75
    return {
        "status": "pass" if abs(_brier(p, rows["result_over"].to_numpy()) - 0.25) < 0.03 else "fail",
        "brier": _brier(p, rows["result_over"].to_numpy()),
        "ece": _ece(p, rows["result_over"].to_numpy()),
        "expected": "Brier should collapse toward 0.250 if labels are shuffled inside market/date buckets.",
    }


def _run_monthly_walk_forward(
    manifest: dict,
    manifest_path: Path,
    all_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    fold_starts = pd.date_range(start=start, end=end, freq="MS")
    folds = []
    for fold_start in fold_starts:
        fold_end = min(fold_start + pd.offsets.MonthEnd(0), end)
        train_rows = all_rows[all_rows["date"] < fold_start].copy()
        test_rows = all_rows[(all_rows["date"] >= fold_start) & (all_rows["date"] <= fold_end)].copy()
        if len(train_rows) < 500 or test_rows.empty:
            continue
        prepared, _ = _prepare_rows(test_rows, manifest, manifest_path, train_rows)
        prepared = _apply_calibration_blend(prepared, args.calibration_blend_alpha)
        if manifest.get("model_version") == "prop_engine_v10":
            prepared["p_v9_calibrated"] = prepared["p_over_calibrated"]
            prepared, _ = _apply_v10_stack(prepared, train_rows)
            prepared["p_over_calibrated"] = prepared["p_v10_calibrated"]
            prepared["p_under_calibrated"] = 1.0 - prepared["p_over_calibrated"]
            prepared["edge_over"] = prepared["p_over_calibrated"] - prepared["market_no_vig_over"]
            prepared["edge_under"] = prepared["p_under_calibrated"] - prepared["market_no_vig_under"]
            prepared["selected_side"] = np.where(prepared["edge_over"] >= prepared["edge_under"], "OVER", "UNDER")
            prepared["side"] = prepared["selected_side"]
            prepared["p_selected"] = np.where(prepared["selected_side"] == "OVER", prepared["p_over_calibrated"], prepared["p_under_calibrated"])
            prepared["market_selected"] = np.where(prepared["selected_side"] == "OVER", prepared["market_no_vig_over"], prepared["market_no_vig_under"])
            prepared["edge"] = prepared["p_selected"] - prepared["market_selected"]
            prepared["profit"] = prepared.apply(_profit_for_side, axis=1)
            prepared["selected_outcome"] = np.where(prepared["selected_side"] == "OVER", prepared["result_over"], 1.0 - prepared["result_over"])
            prepared = _add_market_tracking_columns(prepared)
        risk_col = "brier_risk" if "brier_risk" in prepared.columns else "uncertainty"
        risk_cap = prepared[risk_col].quantile(args.max_uncertainty) if risk_col == "brier_risk" else args.max_uncertainty
        edge_over = args.min_edge if args.min_edge_over is None else args.min_edge_over
        edge_under = args.min_edge if args.min_edge_under is None else args.min_edge_under
        gated = prepared[
            (
                ((prepared["side"] == "OVER") & (prepared["edge"] >= edge_over))
                | ((prepared["side"] == "UNDER") & (prepared["edge"] >= edge_under))
            )
            & (prepared[risk_col] <= risk_cap)
        ]
        folds.append(
            {
                "fold_start": str(fold_start.date()),
                "fold_end": str(fold_end.date()),
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(prepared)),
                "gated_rows": int(len(gated)),
                "market": _metrics(prepared.assign(profit=0.0), "market_no_vig_over"),
                "side_prior": _metrics(prepared.assign(profit=0.0), "side_prior_over") if "side_prior_over" in prepared.columns else {"available": False},
                "model": _metrics(prepared.assign(profit=0.0), "p_over_calibrated"),
                "gated": _metrics(gated, "p_selected", outcome_col="selected_outcome"),
            }
        )
    return {
        "mode": "walk_forward",
        "fold": args.fold,
        "folds": folds,
        "summary": {
            "n_folds": len(folds),
            "avg_model_brier": float(np.mean([f["model"]["brier"] for f in folds])) if folds else None,
            "avg_gated_brier": float(np.mean([f["gated"]["brier"] for f in folds if f["gated"]["n"] > 0])) if folds else None,
            "all_folds_beat_market": all(f["model"]["brier"] < f["market"]["brier"] for f in folds) if folds else False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v9 prop engine against market baselines")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--compare-market-baseline", action="store_true")
    parser.add_argument("--compare-closing-line", action="store_true")
    parser.add_argument("--compare-side-prior", action="store_true")
    parser.add_argument("--audit-leakage", action="store_true")
    parser.add_argument("--check-component-cutoffs", action="store_true")
    parser.add_argument("--ablate-branches", action="store_true")
    parser.add_argument("--label-shuffle-test", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--fold", type=str, default="monthly", choices=["monthly"])
    parser.add_argument("--segment", nargs="+", default=["market", "side", "regime", "uncertainty_bucket", "edge_bucket"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-edge", type=float, default=0.045)
    parser.add_argument("--min-edge-over", type=float, default=None)
    parser.add_argument("--min-edge-under", type=float, default=None)
    parser.add_argument("--min-ev", type=float, default=0.025)
    parser.add_argument("--max-uncertainty", type=float, default=0.70)
    parser.add_argument(
        "--calibration-blend-alpha",
        type=float,
        default=1.0,
        help="1.0 uses prefit calibration only; lower values blend raw and calibrated probabilities.",
    )
    parser.add_argument(
        "--uncertainty-scale",
        type=str,
        default="percentile",
        choices=["percentile"],
        help="Validation uses percentile-normalized uncertainty for cross-run comparability.",
    )
    parser.add_argument("--min-segment-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    all_rows = _load_rows(manifest, manifest_path)
    if args.walk_forward:
        if not args.start or not args.end:
            raise ValueError("--walk-forward requires --start and --end")
        report = _run_monthly_walk_forward(manifest, manifest_path, all_rows, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"walk_forward": report["summary"], "output": str(args.output)}, indent=2, default=str))
        return

    calibration_train_rows = pd.DataFrame()
    if args.start:
        calibration_train_rows = all_rows[all_rows["date"] < pd.Timestamp(args.start)].copy()

    rows = all_rows.copy()
    if args.start:
        rows = rows[rows["date"] >= pd.Timestamp(args.start)]
    if args.end:
        rows = rows[rows["date"] <= pd.Timestamp(args.end)]
    if rows.empty:
        raise ValueError("no validation rows remain after date filters")

    rows, source_counts = _prepare_rows(rows, manifest, manifest_path, calibration_train_rows)
    rows = _apply_calibration_blend(rows, args.calibration_blend_alpha)
    train_scored = calibration_train_rows
    if manifest.get("model_version") == "prop_engine_v10":
        rows["p_v9_calibrated"] = rows["p_over_calibrated"]
        rows["v9_edge_over"] = rows["edge_over"]
        rows["v9_edge_under"] = rows["edge_under"]
        rows["v9_selected_side"] = rows["selected_side"]
        rows["p_v9_selected"] = rows["p_selected"]
        rows["v9_selected_outcome"] = rows["selected_outcome"]
        rows, train_scored = _apply_v10_stack(rows, calibration_train_rows)
        rows["p_over_calibrated"] = rows["p_v10_calibrated"]
        rows["p_under_calibrated"] = 1.0 - rows["p_over_calibrated"]
        rows["edge_over"] = rows["p_over_calibrated"] - rows["market_no_vig_over"]
        rows["edge_under"] = rows["p_under_calibrated"] - rows["market_no_vig_under"]
        rows["selected_side"] = np.where(rows["edge_over"] >= rows["edge_under"], "OVER", "UNDER")
        rows["side"] = rows["selected_side"]
        rows["p_selected"] = np.where(rows["selected_side"] == "OVER", rows["p_over_calibrated"], rows["p_under_calibrated"])
        rows["market_selected"] = np.where(rows["selected_side"] == "OVER", rows["market_no_vig_over"], rows["market_no_vig_under"])
        rows["edge"] = rows["p_selected"] - rows["market_selected"]
        rows["profit"] = rows.apply(_profit_for_side, axis=1)
        rows["selected_outcome"] = np.where(rows["selected_side"] == "OVER", rows["result_over"], 1.0 - rows["result_over"])
        rows = _add_market_tracking_columns(rows)
        rows = _bucketize(rows)

    risk_col = "brier_risk" if "brier_risk" in rows.columns else "uncertainty"
    risk_cap = rows[risk_col].quantile(args.max_uncertainty) if risk_col == "brier_risk" else args.max_uncertainty
    edge_over = args.min_edge if args.min_edge_over is None else args.min_edge_over
    edge_under = args.min_edge if args.min_edge_under is None else args.min_edge_under
    gated = rows[
        (
            ((rows["side"] == "OVER") & (rows["edge"] >= edge_over))
            | ((rows["side"] == "UNDER") & (rows["edge"] >= edge_under))
        )
        & (rows[risk_col] <= risk_cap)
    ].copy()

    comparison = _comparison_table(rows, gated, include_closing=args.compare_closing_line)
    segments = _segment_report(rows, args.segment, args.min_segment_rows)
    calibration = _calibration_diagnostics(rows, source_counts)
    market_source = _market_source_report(rows)
    market_validation = {
        "market_source": market_source,
        "clv": _clv_segment_report(rows, gated, args.min_segment_rows),
        "brier_vs_true_no_vig_market": comparison.get("current_market_no_vig"),
        "true_market_subset": _true_market_subset_report(rows, gated),
        "roi_using_actual_available_odds": (
            comparison.get("v9_calibrated_gate", {}).get("roi_shadow")
            if market_source["real_market_probability_available"]
            else None
        ),
    }
    promotion = _promotion_gates(rows, gated, comparison)
    model_feature_columns = []
    if manifest.get("model_version") == "prop_engine_v10":
        model_feature_columns = _numeric_columns(rows) + _categorical_columns(rows)
    elif manifest.get("model_version") in {
        "prop_engine_v9_2_minutes_adjusted_distribution",
        "prop_engine_v9_3_market_validated_distribution",
        "prop_engine_v9_4_lineup_delta_ready_distribution",
        "prop_engine_v9_4_oracle_lineup_adjusted_distribution",
        "prop_engine_v9_5_pregame_lineup_distribution",
        "prop_engine_v9_5_market_clv_validated_distribution",
    }:
        model_feature_columns = [
            "projected_minutes_mean",
            "projected_minutes_sigma",
            "minutes_roll5_mean_shifted",
            "minutes_roll10_mean_shifted",
            "v92_minutes_ratio",
            "v92_minutes_adjustment",
            "v92_sigma",
            "p_over_raw_minutes_branch",
            "p_over_raw_v91",
        ]
        if manifest.get("model_version") == "prop_engine_v9_4_oracle_lineup_adjusted_distribution":
            model_feature_columns.extend([
                "lineup_oracle_teammates_out_count",
                "lineup_oracle_delta_weighted",
                "lineup_oracle_adjustment",
                "lineup_oracle_confidence_sum",
                "lineup_oracle_max_abs_delta",
                "v94_lineup_model_mean",
                "v94_lineup_sigma",
            ])
        if manifest.get("model_version") in {
            "prop_engine_v9_5_pregame_lineup_distribution",
            "prop_engine_v9_5_market_clv_validated_distribution",
        }:
            model_feature_columns.extend([
                "pregame_teammate_out_prob_sum",
                "pregame_teammate_out_expected_count",
                "pregame_lineup_delta_weighted",
                "pregame_lineup_adjustment",
                "pregame_availability_confidence",
                "pregame_usage_removed_expected",
                "pregame_ast_shift_expected",
                "pregame_reb_shift_expected",
                "v95_pregame_lineup_model_mean",
                "v95_pregame_lineup_sigma",
            ])
    audit = _audit_leakage_report(rows, model_feature_columns) if args.audit_leakage else None
    lineup_field_safety = manifest.get("pregame_lineup_application", {}).get("lineup_field_safety")
    cutoff_report = _component_cutoff_report(manifest, rows, calibration_train_rows, args) if args.check_component_cutoffs else None
    ablations = _ablation_report(calibration_train_rows, rows, rows, gated) if args.ablate_branches else None
    shuffle = _label_shuffle_report(calibration_train_rows, rows) if args.label_shuffle_test else None

    report = {
        "manifest": str(manifest_path),
        "date_range": f"{rows['date'].min().date()}_to_{rows['date'].max().date()}",
        "resolved": int(len(rows)),
        "gated_resolved": int(len(gated)),
        "calibration_training_rows": int(len(calibration_train_rows)),
        "filters": {
            "start": args.start,
            "end": args.end,
            "min_edge": args.min_edge,
            "min_edge_over": edge_over,
            "min_edge_under": edge_under,
            "min_ev": args.min_ev,
            "max_uncertainty": args.max_uncertainty,
            "uncertainty_scale": args.uncertainty_scale,
            "calibration_policy": "prefit_before_start" if len(calibration_train_rows) else "artifact_or_raw",
            "calibration_blend_alpha": args.calibration_blend_alpha,
            "risk_gate_column": risk_col,
            "risk_gate_value": float(risk_cap),
        },
        "comparison": comparison,
        "market_validation": market_validation,
        "segments": segments,
        "calibration": calibration,
        "promotion": promotion,
        "lineup_field_safety": lineup_field_safety,
        "cutoff_safety": cutoff_report,
        "leakage_audit": audit,
        "branch_ablation": ablations,
        "label_shuffle_test": shuffle,
        "limitations": [
            "Current no-vig baseline is neutral unless real over_odds/under_odds are attached." if not market_validation["market_source"]["real_market_probability_available"] else "Current no-vig baseline uses attached real odds snapshots.",
            "Closing no-vig comparison is only computed when close_over_odds/close_under_odds or closing aliases exist.",
            "Regime segmentation uses pregame proxy buckets unless row-level v9 regime labels are present.",
            "CLV promotion gate is unavailable until closing-line snapshots are attached." if not market_validation["market_source"]["closing_odds_available"] else "CLV is computed from attached closing-line snapshots.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "resolved": report["resolved"],
        "gated_resolved": report["gated_resolved"],
        "comparison": comparison,
        "promotion": promotion,
        "cutoff_safety": cutoff_report,
        "leakage_audit": audit,
        "label_shuffle_test": shuffle,
        "output": str(args.output),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
