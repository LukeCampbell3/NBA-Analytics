#!/usr/bin/env python3
"""
v9.8 Weak Line Detector

Sits after the full distribution model and before the final gate.
Scores each prop on three dimensions:

  1. Outcome edge:     p_model_side - p_market_side
  2. CLV edge:         probability that line/price moves toward model side
  3. Execution quality: how much better this book's price is vs consensus

Features:
  - Book-vs-consensus outlier score
  - Line-vs-projection z-score
  - Market movement velocity (side-aware)
  - Stale-line score
  - Availability/news sensitivity

Output per prop:
  weak_line_score, expected_clv, book_outlier_score, stale_line_score,
  model_market_disagreement, selected_side, reason_codes
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from market_odds_quality import is_valid_american_odds


def _ensure_snapshot_identity(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    out = snapshot_df.copy()
    if "date" not in out.columns and "snapshot_date" in out.columns:
        out["date"] = out["snapshot_date"]
    if "book" not in out.columns and "book_key" in out.columns:
        out["book"] = out["book_key"]
    if "line" in out.columns:
        out["line"] = pd.to_numeric(out["line"], errors="coerce")
    if "snapshot_time" in out.columns:
        out["_snapshot_ts"] = pd.to_datetime(out["snapshot_time"], errors="coerce", utc=True, format="mixed")
    else:
        out["_snapshot_ts"] = pd.NaT
        out["snapshot_time"] = ""
    return out


def _prop_keys(df: pd.DataFrame, *, include_snapshot: bool = False, include_book: bool = False) -> list[str]:
    keys = ["player"]
    if "game_id" in df.columns:
        keys.append("game_id")
    elif "market_event_id" in df.columns:
        keys.append("market_event_id")
    keys.extend(["market", "line", "date"])
    if include_snapshot and "snapshot_time" in df.columns:
        keys.append("snapshot_time")
    if include_book and "book" in df.columns:
        keys.append("book")
    return [key for key in keys if key in df.columns]


def _latest_per_prop(df: pd.DataFrame, prop_keys: list[str]) -> pd.DataFrame:
    if df.empty or "_snapshot_ts" not in df.columns:
        return df
    sortable = df.copy()
    sortable["_snapshot_ts_sort"] = sortable["_snapshot_ts"].fillna(pd.Timestamp.min.tz_localize("UTC"))
    return sortable.sort_values("_snapshot_ts_sort").groupby(prop_keys, dropna=False).tail(1).drop(columns=["_snapshot_ts_sort"])


def _american_to_implied(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _no_vig(over_odds: float, under_odds: float) -> tuple[float, float]:
    over = _american_to_implied(over_odds)
    under = _american_to_implied(under_odds)
    total = over + under
    if not np.isfinite(total) or total <= 0:
        return 0.5, 0.5
    return over / total, under / total


def _p_over_from_distribution(model_mean: float, line: float, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1.0
    z = (line - model_mean) / sigma
    return float(np.clip(0.5 * (1.0 - math.erf(z / math.sqrt(2.0))), 0.01, 0.99))


# ─── Feature Computation ──────────────────────────────────────────

def compute_consensus(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute consensus no-vig probability across all books for each prop.

    Returns one row per (player, market, line, date, snapshot_time) with
    consensus_no_vig_over/under = mean across books.
    """
    snapshot_df = _ensure_snapshot_identity(snapshot_df)
    valid = snapshot_df[snapshot_df["is_valid_american_odds"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    # Compute no-vig per row
    nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    valid["book_no_vig_over"], valid["book_no_vig_under"] = zip(*nv)

    prop_keys = _prop_keys(valid)
    timestamp_keys = _prop_keys(valid, include_snapshot=True)
    main_line_keys = [key for key in prop_keys if key != "line"]
    line_counts = valid.groupby(main_line_keys, dropna=False)["line"].transform("nunique") if main_line_keys else 1
    main_lines = valid.groupby(main_line_keys, dropna=False)["line"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").mode().iloc[0] if not pd.to_numeric(s, errors="coerce").mode().empty else np.nan
    ) if main_line_keys else valid["line"]
    valid["consensus_line_type"] = np.where(line_counts > 1, np.where(valid["line"] == main_lines, "main_line", "alt_line"), "main_line")

    consensus = valid.groupby(timestamp_keys, dropna=False).agg(
        consensus_no_vig_over=("book_no_vig_over", "mean"),
        consensus_no_vig_under=("book_no_vig_under", "mean"),
        n_books=("book", "nunique"),
        books_list=("book", lambda x: sorted(x.unique().tolist())),
        odds_std=("book_no_vig_over", "std"),
        consensus_line_type=("consensus_line_type", "first"),
        consensus_snapshot_time=("snapshot_time", "first"),
        _snapshot_ts=("_snapshot_ts", "first"),
    ).reset_index()

    return consensus


def compute_book_outlier_features(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """For each book's price, compute deviation from consensus.

    A book is an outlier if its no-vig probability deviates significantly
    from the consensus of all books for the same prop.
    """
    snapshot_df = _ensure_snapshot_identity(snapshot_df)
    valid = snapshot_df[snapshot_df["is_valid_american_odds"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    valid["book_no_vig_over"], valid["book_no_vig_under"] = zip(*nv)

    group_keys = _prop_keys(valid, include_snapshot=True)
    consensus = valid.groupby(group_keys)["book_no_vig_over"].transform("mean")
    valid["consensus_no_vig_over"] = consensus
    valid["book_deviation_over"] = valid["book_no_vig_over"] - valid["consensus_no_vig_over"]
    valid["book_deviation_under"] = -valid["book_deviation_over"]

    return valid


def compute_stale_line_score(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Detect books that haven't moved while others have.

    A stale line is one where the book's price hasn't changed between
    the first and last snapshot, but the consensus has moved.

    Improved: specifically detects when consensus moved TOWARD model side
    but a specific book stayed put.
    """
    snapshot_df = _ensure_snapshot_identity(snapshot_df)
    valid = snapshot_df[snapshot_df["is_valid_american_odds"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["snapshot_ts"] = valid["_snapshot_ts"]
    nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    valid["book_no_vig_over"], valid["book_no_vig_under"] = zip(*nv)

    group_keys = _prop_keys(valid, include_book=True)

    # Get first and last snapshot per book/prop
    sorted_df = valid.sort_values("snapshot_ts")
    first = sorted_df.groupby(group_keys).first().reset_index()
    last = sorted_df.groupby(group_keys).last().reset_index()

    merged = first[group_keys + ["book_no_vig_over"]].merge(
        last[group_keys + ["book_no_vig_over"]],
        on=group_keys,
        suffixes=("_first", "_last"),
    )
    merged["book_moved"] = (merged["book_no_vig_over_last"] - merged["book_no_vig_over_first"]).abs()

    # Consensus movement per prop
    prop_keys = _prop_keys(valid)
    consensus_movement = merged.groupby(prop_keys)["book_moved"].mean().reset_index()
    consensus_movement = consensus_movement.rename(columns={"book_moved": "consensus_avg_movement"})

    merged = merged.merge(consensus_movement, on=prop_keys, how="left")
    merged["stale_line_score"] = np.clip(
        merged["consensus_avg_movement"] - merged["book_moved"], 0, 1
    )

    return merged


def compute_market_velocity(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute side-aware market movement velocity between snapshots."""
    snapshot_df = _ensure_snapshot_identity(snapshot_df)
    valid = snapshot_df[snapshot_df["is_valid_american_odds"] == True].copy()
    if valid.empty:
        return pd.DataFrame()

    valid["snapshot_ts"] = valid["_snapshot_ts"]
    nv = valid.apply(lambda r: _no_vig(r["over_odds"], r["under_odds"]), axis=1)
    valid["book_no_vig_over"], valid["book_no_vig_under"] = zip(*nv)

    group_keys = _prop_keys(valid)

    # Consensus at first and last timestamp
    sorted_df = valid.sort_values("snapshot_ts")
    first_ts = sorted_df.groupby(group_keys)["snapshot_ts"].transform("min")
    last_ts = sorted_df.groupby(group_keys)["snapshot_ts"].transform("max")

    first_consensus = sorted_df[sorted_df["snapshot_ts"] == first_ts].groupby(group_keys)["book_no_vig_over"].mean()
    last_consensus = sorted_df[sorted_df["snapshot_ts"] == last_ts].groupby(group_keys)["book_no_vig_over"].mean()

    velocity = (last_consensus - first_consensus).reset_index()
    velocity.columns = group_keys + ["over_velocity"]
    velocity["under_velocity"] = -velocity["over_velocity"]

    return velocity


# ─── Weak Line Scoring ─────────────────────────────────────────────

def score_weak_lines(
    predictions: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Score each prediction for weak-line characteristics.

    Args:
        predictions: DataFrame with model predictions (player, market, date,
            model_mean, line, sigma, p_model_over, p_model_under)
        snapshot_df: Raw collected book snapshots with multi-book odds
        weights: Optional weight overrides for scoring components

    Returns:
        predictions with added weak_line_score and component features
    """
    if weights is None:
        weights = {
            "model_edge": 0.35,
            "book_outlier": 0.25,
            "stale_line": 0.10,
            "velocity_alignment": 0.15,
            "projection_z": 0.15,
        }

    from market_odds_quality import add_american_odds_quality
    snapshot_df = _ensure_snapshot_identity(add_american_odds_quality(snapshot_df))

    preds = predictions.copy()
    if "line" in preds.columns:
        preds["line"] = pd.to_numeric(preds["line"], errors="coerce")
    # Preserve line column through merges.
    preds["_line_orig"] = preds["line"].copy()
    exact_prop_keys = [key for key in _prop_keys(snapshot_df) if key in preds.columns]

    # 1. Model edge (already have p_model_over)
    # Need entry market no-vig to compute edge
    consensus = compute_consensus(snapshot_df)
    if not consensus.empty:
        # Avoid line collision — consensus has 'line' as a group key
        # Only take the consensus probability columns, aggregated across lines
        consensus_latest = _latest_per_prop(consensus, exact_prop_keys)
        consensus_cols = exact_prop_keys + [
            "consensus_no_vig_over",
            "consensus_no_vig_under",
            "n_books",
            "odds_std",
            "consensus_line_type",
            "consensus_snapshot_time",
        ]
        preds = preds.merge(consensus_latest[consensus_cols], on=exact_prop_keys, how="left")
    else:
        preds["consensus_no_vig_over"] = 0.5
        preds["consensus_no_vig_under"] = 0.5
        preds["n_books"] = 0
        preds["odds_std"] = 0.0
        preds["consensus_line_type"] = "unavailable"
        preds["consensus_snapshot_time"] = ""

    preds["consensus_no_vig_over"] = preds["consensus_no_vig_over"].fillna(0.5)
    preds["consensus_no_vig_under"] = preds["consensus_no_vig_under"].fillna(0.5)
    preds["n_books"] = pd.to_numeric(preds.get("n_books"), errors="coerce").fillna(0.0)
    preds["odds_std"] = pd.to_numeric(preds.get("odds_std"), errors="coerce").fillna(0.0)
    preds["consensus_line_type"] = preds.get("consensus_line_type", pd.Series("unavailable", index=preds.index)).fillna("unavailable")
    preds["line_mismatch_guard_pass"] = preds["n_books"] > 0
    preds["same_line_consensus_confirmed"] = preds["line_mismatch_guard_pass"] & (preds["n_books"] >= 2)
    preds["main_line_consensus"] = np.where(preds["consensus_line_type"] == "main_line", preds["n_books"], 0.0)
    preds["alt_line_consensus"] = np.where(preds["consensus_line_type"] == "alt_line", preds["n_books"], 0.0)

    # Model edge vs consensus
    preds["model_edge_over"] = preds["p_model_over"] - preds["consensus_no_vig_over"]
    preds["model_edge_under"] = preds["p_model_under"] - preds["consensus_no_vig_under"]
    preds["selected_side"] = np.where(
        preds["model_edge_over"] >= preds["model_edge_under"], "OVER", "UNDER"
    )
    preds["model_edge"] = np.where(
        preds["selected_side"] == "OVER",
        preds["model_edge_over"],
        preds["model_edge_under"],
    )
    preds["model_edge_normalized"] = np.clip(preds["model_edge"] / 0.15, 0, 1)

    # 2. Book outlier score — find the best book for the selected side
    book_features = compute_book_outlier_features(snapshot_df)
    if not book_features.empty:
        book_features = _latest_per_prop(book_features, _prop_keys(book_features, include_book=True))
        # For each prop (player/market/date), find the book with the best price for the model's side
        prop_keys = exact_prop_keys
        best_books = []
        for prop_values, pred_group in preds.groupby(prop_keys, dropna=False):
            if not isinstance(prop_values, tuple):
                prop_values = (prop_values,)
            if pred_group.empty:
                continue
            side = pred_group.iloc[0]["selected_side"]
            mask = pd.Series(True, index=book_features.index)
            for key, value in zip(prop_keys, prop_values):
                mask &= book_features[key].eq(value)
            book_group = book_features[mask]
            prop_payload = dict(zip(prop_keys, prop_values))
            if book_group.empty:
                best_books.append({
                    **prop_payload,
                    "best_book": None, "best_book_deviation": 0.0, "book_spread": 0.0,
                    "book_specific_line_edge": 0.0, "book_odds_timestamp": "",
                })
                continue
            if side == "OVER":
                best_idx = book_group["book_no_vig_over"].idxmin()
                best_dev = float(-book_group.loc[best_idx, "book_deviation_over"])
            else:
                best_idx = book_group["book_no_vig_under"].idxmin()
                best_dev = float(-book_group.loc[best_idx, "book_deviation_under"])
            best_books.append({
                **prop_payload,
                "best_book": book_group.loc[best_idx, "book"],
                "best_book_deviation": best_dev,
                "book_spread": float(book_group["book_no_vig_over"].max() - book_group["book_no_vig_over"].min()),
                "book_specific_line_edge": best_dev,
                "book_odds_timestamp": book_group.loc[best_idx, "snapshot_time"],
            })
        if best_books:
            best_df = pd.DataFrame(best_books)
            preds = preds.merge(best_df, on=prop_keys, how="left")
        else:
            preds["best_book"] = None
            preds["best_book_deviation"] = 0.0
            preds["book_spread"] = 0.0
            preds["book_specific_line_edge"] = 0.0
            preds["book_odds_timestamp"] = ""
    else:
        preds["best_book"] = None
        preds["best_book_deviation"] = 0.0
        preds["book_spread"] = 0.0
        preds["book_specific_line_edge"] = 0.0
        preds["book_odds_timestamp"] = ""

    preds["best_book_deviation"] = preds["best_book_deviation"].fillna(0.0)
    preds["book_spread"] = preds["book_spread"].fillna(0.0)
    preds["book_specific_line_edge"] = preds["book_specific_line_edge"].fillna(0.0)
    preds["book_outlier_score"] = np.clip(preds["best_book_deviation"] / 0.03, 0, 1)

    # 3. Stale line score
    stale = compute_stale_line_score(snapshot_df)
    if not stale.empty:
        # Average stale score per prop (across books)
        prop_stale = stale.groupby(exact_prop_keys, dropna=False)["stale_line_score"].max().reset_index()
        preds = preds.merge(prop_stale, on=exact_prop_keys, how="left")
    else:
        preds["stale_line_score"] = 0.0
    preds["stale_line_score"] = preds["stale_line_score"].fillna(0.0)

    # 4. Market velocity alignment
    velocity = compute_market_velocity(snapshot_df)
    if not velocity.empty:
        preds = preds.merge(velocity, on=exact_prop_keys, how="left")
    else:
        preds["over_velocity"] = 0.0
        preds["under_velocity"] = 0.0
    preds["over_velocity"] = preds["over_velocity"].fillna(0.0)
    preds["under_velocity"] = preds["under_velocity"].fillna(0.0)

    # Side-aware velocity: positive = market moving toward model side
    preds["side_velocity"] = np.where(
        preds["selected_side"] == "OVER",
        preds["over_velocity"],
        preds["under_velocity"],
    )
    preds["velocity_alignment_score"] = np.clip(preds["side_velocity"] / 0.02, -1, 1)
    preds["velocity_alignment_normalized"] = np.clip((preds["velocity_alignment_score"] + 1) / 2, 0, 1)

    # 5. Projection z-score
    # Restore line if lost during merges
    if "line" not in preds.columns and "_line_orig" in preds.columns:
        preds["line"] = preds["_line_orig"]
    preds["projection_z"] = (preds["line"] - preds["model_mean"]) / preds["sigma"].clip(lower=0.5)
    # For OVER: lower z = better (line is below model mean)
    # For UNDER: higher z = better (line is above model mean)
    preds["projection_z_score"] = np.where(
        preds["selected_side"] == "OVER",
        np.clip(-preds["projection_z"] / 2, 0, 1),
        np.clip(preds["projection_z"] / 2, 0, 1),
    )

    # ─── Composite Weak Line Score ────────────────────────────────
    preds["weak_line_score"] = (
        weights["model_edge"] * preds["model_edge_normalized"]
        + weights["book_outlier"] * preds["book_outlier_score"]
        + weights["stale_line"] * preds["stale_line_score"]
        + weights["velocity_alignment"] * preds["velocity_alignment_normalized"]
        + weights["projection_z"] * preds["projection_z_score"]
    ).clip(0, 1)

    # ─── Market Drift Regime Detection ────────────────────────────
    # Detect which direction the market is broadly drifting in this slate.
    # This is NOT hard-coded to OVER or UNDER — it adapts per collection window.
    over_velocity_all = preds["over_velocity"].mean()
    drift_rate = float(np.clip((over_velocity_all + 0.02) / 0.04, 0, 1))  # 0=strong under drift, 1=strong over drift
    if over_velocity_all > 0.003:
        market_drift_side = "OVER"
        market_drift_strength = float(np.clip(over_velocity_all / 0.01, 0, 1))
    elif over_velocity_all < -0.003:
        market_drift_side = "UNDER"
        market_drift_strength = float(np.clip(-over_velocity_all / 0.01, 0, 1))
    else:
        market_drift_side = "NEUTRAL"
        market_drift_strength = 0.0

    preds["market_drift_side"] = market_drift_side
    preds["market_drift_strength"] = market_drift_strength
    preds["fighting_drift"] = preds["selected_side"] != market_drift_side

    # Risk penalty
    uncertainty = preds.get("belief_uncertainty", pd.Series(0.5, index=preds.index))
    preds["risk_penalty"] = np.clip(uncertainty * 0.3, 0, 0.3)

    # Regime-adaptive side penalty: penalize the side fighting market drift.
    # The penalty scales with drift strength — stronger drift = more penalty for opposing side.
    # If drift is NEUTRAL, no side penalty.
    drift_penalty_amount = 0.08 * market_drift_strength  # max 0.08 penalty
    preds["side_drift_penalty"] = np.where(
        (market_drift_side != "NEUTRAL") & (preds["selected_side"] != market_drift_side),
        drift_penalty_amount,
        0.0,
    )

    preds["weak_line_score_adjusted"] = (
        preds["weak_line_score"] - preds["risk_penalty"] - preds["side_drift_penalty"]
    ).clip(0, 1)

    # ─── Reliability Tier Assignment (regime-adaptive) ────────────
    # Base thresholds
    shadow_threshold = 0.55
    monitor_threshold = 0.40

    # Side fighting drift needs higher threshold
    drift_side_boost = 0.05 * market_drift_strength  # up to +0.05 for strong drift
    preds["effective_shadow_threshold"] = np.where(
        preds["fighting_drift"] & (market_drift_side != "NEUTRAL"),
        shadow_threshold + drift_side_boost,
        shadow_threshold,
    )
    preds["effective_monitor_threshold"] = np.where(
        preds["fighting_drift"] & (market_drift_side != "NEUTRAL"),
        monitor_threshold + drift_side_boost,
        monitor_threshold,
    )

    preds["reliability_tier"] = "no_action"
    for idx in preds.index:
        if not bool(preds.loc[idx, "line_mismatch_guard_pass"]):
            preds.loc[idx, "reliability_tier"] = "no_action"
            continue
        wls = preds.loc[idx, "weak_line_score_adjusted"]
        model_edge_high = preds.loc[idx, "model_edge"] > 0.05
        market_weakness_high = preds.loc[idx, "book_outlier_score"] >= 0.33 or preds.loc[idx, "stale_line_score"] > 0.3
        if wls >= preds.loc[idx, "effective_shadow_threshold"] and model_edge_high and market_weakness_high:
            preds.loc[idx, "reliability_tier"] = "shadow_A"
        elif model_edge_high and not market_weakness_high and wls >= preds.loc[idx, "effective_monitor_threshold"]:
            preds.loc[idx, "reliability_tier"] = "model_edge_monitor"
        elif market_weakness_high and wls >= preds.loc[idx, "effective_monitor_threshold"]:
            preds.loc[idx, "reliability_tier"] = "execution_monitor"
        elif wls >= preds.loc[idx, "effective_monitor_threshold"]:
            preds.loc[idx, "reliability_tier"] = "monitor"

    # Reason codes
    def _reason_codes(row):
        codes = []
        if row["model_edge"] > 0.05:
            codes.append("model_edge_high")
        if row.get("best_book_deviation", 0) > 0.01:
            codes.append("book_price_below_consensus")
        if bool(row.get("same_line_consensus_confirmed", False)):
            codes.append("same_line_consensus_confirmed")
        if bool(row.get("line_mismatch_guard_pass", False)):
            codes.append("not_alt_line_mismatch")
        else:
            codes.append("alt_line_mismatch")
        if row.get("side_velocity", 0) > 0.005:
            codes.append("market_moving_toward_model_side")
        if row.get("side_velocity", 0) < -0.005:
            codes.append("market_moving_against_model_side")
        if row.get("stale_line_score", 0) > 0.3:
            codes.append("stale_line_detected")
        if row.get("projection_z_score", 0) > 0.6:
            codes.append("strong_distribution_edge")
        if uncertainty.loc[row.name] < 0.3:
            codes.append("low_uncertainty")
        if row.get("fighting_drift", False) and market_drift_side != "NEUTRAL":
            codes.append(f"against_{market_drift_side.lower()}_drift")
        if row.get("lineup_confidence", row.get("availability_confidence", 1.0)) >= 0.5:
            codes.append("lineup_confidence_ok")
        if str(row.get("close_status", "")).lower() != "stale_after_lock":
            codes.append("not_stale_after_lock")
        return codes

    preds["reason_codes"] = preds.apply(_reason_codes, axis=1)
    required_production_codes = {
        "model_edge_high",
        "book_price_below_consensus",
        "same_line_consensus_confirmed",
        "lineup_confidence_ok",
        "not_alt_line_mismatch",
        "not_stale_after_lock",
    }
    preds["production_reason_codes_valid"] = preds["reason_codes"].apply(
        lambda codes: required_production_codes.issubset(set(codes or []))
    )
    missing_production_codes = preds["reliability_tier"].eq("shadow_A") & ~preds["production_reason_codes_valid"]
    preds.loc[missing_production_codes, "reliability_tier"] = "execution_monitor"

    # Final reliability filter: demote picks fighting strong drift without overwhelming evidence
    strong_drift_fight = (
        preds["fighting_drift"]
        & (market_drift_strength > 0.3)
        & (preds["side_velocity"] < -0.003)
        & (preds["model_edge"] < 0.12)
    )
    preds.loc[strong_drift_fight, "reliability_tier"] = "no_action"
    preds.loc[strong_drift_fight, "weak_line_score_adjusted"] = (
        preds.loc[strong_drift_fight, "weak_line_score_adjusted"] * 0.5
    )

    return preds


# ─── Main ─────────────────────────────────────────────────────────

def validate_wls_buckets(scored: pd.DataFrame, attachable_path: Path) -> dict:
    """Validate weak-line score by buckets with full diagnostics.

    Tests:
    1. Monotonicity: higher WLS → higher positive CLV rate
    2. Bootstrap confidence intervals per bucket
    3. Book-level CLV (is one book carrying the signal?)
    4. Market × side health (does it work across markets and sides?)
    5. Promotion ladder readiness
    """
    from attach_model_predictions_to_clv import _no_vig as _nv

    att = pd.read_csv(attachable_path)
    from market_odds_quality import add_american_odds_quality
    att = add_american_odds_quality(att)
    true_clv = att[att.get("close_status", pd.Series()) == "true_sequence_close"].copy()

    # Normalize for join
    true_clv["player_norm"] = true_clv["player"].str.replace("_", " ").str.lower().str.strip()
    scored_norm = scored.copy()
    scored_norm["player_norm"] = scored_norm["player"].str.replace("_", " ").str.lower().str.strip()
    true_clv["date"] = pd.to_datetime(true_clv["date"], errors="coerce").dt.date.astype(str)
    scored_norm["date"] = pd.to_datetime(scored_norm["date"], errors="coerce").dt.date.astype(str)

    merge_cols = ["player_norm", "market", "date", "weak_line_score_adjusted",
                  "selected_side", "model_edge", "book_outlier_score",
                  "stale_line_score", "velocity_alignment_normalized"]
    if "best_book" in scored_norm.columns:
        merge_cols.append("best_book")

    merged = true_clv.merge(
        scored_norm[merge_cols],
        on=["player_norm", "market", "date"],
        how="inner",
    )

    if len(merged) == 0:
        return {"status": "no_matches", "buckets": []}

    # Compute CLV
    entry_valid = merged["over_odds"].apply(is_valid_american_odds) & merged["under_odds"].apply(is_valid_american_odds)
    close_valid = merged["close_over_odds"].apply(is_valid_american_odds) & merged["close_under_odds"].apply(is_valid_american_odds)
    both = entry_valid & close_valid
    valid = merged.loc[both].copy()

    if len(valid) == 0:
        return {"status": "no_valid_odds", "buckets": []}

    entry_nv = valid.apply(lambda r: _nv(r["over_odds"], r["under_odds"]), axis=1)
    close_nv = valid.apply(lambda r: _nv(r["close_over_odds"], r["close_under_odds"]), axis=1)
    valid["entry_nv_over"], valid["entry_nv_under"] = zip(*entry_nv)
    valid["close_nv_over"], valid["close_nv_under"] = zip(*close_nv)
    valid["clv_over"] = valid["close_nv_over"] - valid["entry_nv_over"]
    valid["clv_under"] = valid["close_nv_under"] - valid["entry_nv_under"]
    valid["model_side_clv"] = np.where(
        valid["selected_side"] == "OVER", valid["clv_over"], valid["clv_under"]
    )

    # Only rows with movement
    moved = valid[valid["clv_over"].abs() > 1e-6].copy()
    if len(moved) < 20:
        return {"status": "insufficient_moved_rows", "moved": int(len(moved)), "buckets": []}

    # ── 1. Bucket validation with bootstrap CI ──
    bins = [0.0, 0.25, 0.40, 0.55, 0.70, 1.01]
    labels = ["0.00-0.25", "0.25-0.40", "0.40-0.55", "0.55-0.70", "0.70+"]
    moved["wls_bucket"] = pd.cut(moved["weak_line_score_adjusted"], bins=bins, labels=labels).astype(str)

    buckets = []
    for bucket in labels:
        group = moved[moved["wls_bucket"] == bucket]
        if len(group) == 0:
            continue
        pos_rate = float((group["model_side_clv"] > 0).mean())
        mean_clv = float(group["model_side_clv"].mean())

        # Bootstrap 95% CI for positive CLV rate
        n_boot = 1000
        rng = np.random.default_rng(42)
        if len(group) >= 10:
            boot_rates = []
            for _ in range(n_boot):
                sample = rng.choice(group["model_side_clv"].values, size=len(group), replace=True)
                boot_rates.append(float((sample > 0).mean()))
            ci_lower = float(np.percentile(boot_rates, 2.5))
            ci_upper = float(np.percentile(boot_rates, 97.5))
        else:
            ci_lower = ci_upper = pos_rate

        side_counts = group["selected_side"].value_counts(normalize=True).to_dict()
        market_counts = group["market"].value_counts(normalize=True).to_dict()

        buckets.append({
            "bucket": bucket,
            "rows": int(len(group)),
            "unique_players": int(group["player_norm"].nunique()) if "player_norm" in group.columns else 0,
            "positive_clv_rate": pos_rate,
            "positive_clv_rate_ci_lower": ci_lower,
            "positive_clv_rate_ci_upper": ci_upper,
            "mean_side_clv": mean_clv,
            "mean_model_edge": float(group["model_edge"].mean()),
            "edge_clv_corr": float(group["model_edge"].corr(group["model_side_clv"])) if len(group) > 5 else None,
            "side_share": side_counts,
            "market_mix": market_counts,
        })

    # Monotonicity test
    rates = [b["positive_clv_rate"] for b in buckets if b["rows"] >= 10]
    monotonic_pairs = sum(1 for i in range(len(rates) - 1) if rates[i + 1] >= rates[i])
    total_pairs = max(len(rates) - 1, 1)
    monotonicity_score = monotonic_pairs / total_pairs if total_pairs > 0 else 0.0

    # ── 2. Book-level validation ──
    book_validation = {}
    if "book" in moved.columns:
        for book_name, book_group in moved.groupby("book"):
            if len(book_group) >= 10:
                book_validation[str(book_name)] = {
                    "rows": int(len(book_group)),
                    "positive_clv_rate": float((book_group["model_side_clv"] > 0).mean()),
                    "mean_clv": float(book_group["model_side_clv"].mean()),
                }
    # Check if one book dominates
    if book_validation:
        book_rows = {k: v["rows"] for k, v in book_validation.items()}
        total_book_rows = sum(book_rows.values())
        max_book_share = max(book_rows.values()) / total_book_rows if total_book_rows > 0 else 0
        book_concentration_ok = max_book_share < 0.50
    else:
        book_concentration_ok = True
        max_book_share = 0.0

    # ── 3. Market × side health ──
    side_market_health = {}
    for side in ["OVER", "UNDER"]:
        side_group = moved[moved["selected_side"] == side]
        if len(side_group) >= 10:
            side_market_health[side] = {
                "rows": int(len(side_group)),
                "positive_clv_rate": float((side_group["model_side_clv"] > 0).mean()),
                "mean_clv": float(side_group["model_side_clv"].mean()),
            }
    for market_name in ["PTS", "TRB", "AST"]:
        mkt_group = moved[moved["market"] == market_name]
        if len(mkt_group) >= 10:
            side_market_health[market_name] = {
                "rows": int(len(mkt_group)),
                "positive_clv_rate": float((mkt_group["model_side_clv"] > 0).mean()),
                "mean_clv": float(mkt_group["model_side_clv"].mean()),
            }

    # Count markets with positive CLV
    markets_with_positive_clv = sum(
        1 for k, v in side_market_health.items()
        if k in ["PTS", "TRB", "AST"] and v["positive_clv_rate"] > 0.50
    )
    sides_with_positive_clv = sum(
        1 for k, v in side_market_health.items()
        if k in ["OVER", "UNDER"] and v["positive_clv_rate"] > 0.50
    )

    # ── 4. Promotion ladder ──
    tier_a_rows = moved[moved["weak_line_score_adjusted"] >= 0.55]
    tier_a_pos_rate = float((tier_a_rows["model_side_clv"] > 0).mean()) if len(tier_a_rows) > 0 else 0.0
    tier_a_mean_clv = float(tier_a_rows["model_side_clv"].mean()) if len(tier_a_rows) > 0 else 0.0

    promotion_ladder = {
        "tier_a_shadow": {
            "min_wls": 0.55,
            "tier_a_rows": int(len(tier_a_rows)),
            "tier_a_positive_clv_rate": tier_a_pos_rate,
            "tier_a_mean_clv": tier_a_mean_clv,
            "status": "pass" if len(tier_a_rows) >= 10 and tier_a_pos_rate > 0.50 else "insufficient",
        },
        "tier_a_production_candidate": {
            "require_100_tier_a_rows": int(len(tier_a_rows)) >= 100,
            "require_55pct_positive_clv": tier_a_pos_rate >= 0.55,
            "require_positive_mean_clv": tier_a_mean_clv > 0,
            "require_both_sides_positive": sides_with_positive_clv >= 2,
            "require_2_markets_positive": markets_with_positive_clv >= 2,
            "require_settled_brier": False,  # Pending game outcomes
            "require_settled_roi": False,  # Pending game outcomes
            "status": "blocked_pending_outcomes_and_volume",
        },
        "live_production": {
            "require_500_clv_rows": int(len(moved)) >= 500,
            "require_150_tier_a_rows": int(len(tier_a_rows)) >= 150,
            "require_positive_clv_edge_corr": monotonicity_score >= 0.5,
            "require_settled_roi_positive": False,  # Pending
            "require_model_brier_beats_market": False,  # Pending
            "status": "blocked",
        },
    }

    return {
        "status": "computed",
        "total_moved_rows": int(len(moved)),
        "buckets": buckets,
        "monotonicity_score": monotonicity_score,
        "monotonicity_pass": monotonicity_score >= 0.5,
        "book_validation": book_validation,
        "book_concentration_ok": book_concentration_ok,
        "max_book_share": max_book_share,
        "side_market_health": side_market_health,
        "markets_with_positive_clv": markets_with_positive_clv,
        "sides_with_positive_clv": sides_with_positive_clv,
        "promotion_ladder": promotion_ladder,
        "note": "monotonicity_score = fraction of adjacent bucket pairs where higher WLS has higher positive CLV rate",
    }


def main():
    from attach_model_predictions_to_clv import attach_and_evaluate

    predictions_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "model_slate_for_clv.csv"
    snapshot_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "collected_book_snapshots.csv"
    attachable_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "market_snapshot_attachable.csv"

    if not predictions_path.exists():
        print("ERROR: Run build_clv_slate_from_live_odds.py first")
        return

    predictions = pd.read_csv(predictions_path)
    snapshot_df = pd.read_csv(snapshot_path)

    print("=" * 70)
    print("V9.8 WEAK LINE DETECTOR")
    print("=" * 70)
    print(f"\nInput: {len(predictions)} model predictions, {len(snapshot_df)} book snapshots")

    scored = score_weak_lines(predictions, snapshot_df)

    print(f"\nWeak Line Score Distribution:")
    print(f"  Mean:   {scored['weak_line_score'].mean():.3f}")
    print(f"  Median: {scored['weak_line_score'].median():.3f}")
    print(f"  Std:    {scored['weak_line_score'].std():.3f}")
    print(f"  Min:    {scored['weak_line_score'].min():.3f}")
    print(f"  Max:    {scored['weak_line_score'].max():.3f}")

    print(f"\nSide Distribution:")
    side_counts = scored["selected_side"].value_counts()
    for side, count in side_counts.items():
        print(f"  {side}: {count} ({count/len(scored):.1%})")

    print(f"\nComponent Scores (mean):")
    print(f"  Model edge (normalized):     {scored['model_edge_normalized'].mean():.3f}")
    print(f"  Book outlier score:          {scored['book_outlier_score'].mean():.3f}")
    print(f"  Stale line score:            {scored['stale_line_score'].mean():.3f}")
    print(f"  Velocity alignment:          {scored['velocity_alignment_normalized'].mean():.3f}")
    print(f"  Projection z-score:          {scored['projection_z_score'].mean():.3f}")
    print(f"  Risk penalty:                {scored['risk_penalty'].mean():.3f}")
    print(f"  Adjusted score:              {scored['weak_line_score_adjusted'].mean():.3f}")

    # Top weak lines
    top = scored.nlargest(10, "weak_line_score_adjusted")
    print(f"\nTop 10 Weak Lines:")
    print(f"  {'Player':<20s} {'Mkt':>4s} {'Side':>5s} {'Edge':>6s} {'WLS':>5s} {'Book':<12s} {'Reasons'}")
    for _, row in top.iterrows():
        reasons = ", ".join(row["reason_codes"][:2]) if row["reason_codes"] else ""
        book = str(row.get("best_book", ""))[:12] if pd.notna(row.get("best_book")) else ""
        print(f"  {row['player']:<20s} {row['market']:>4s} {row['selected_side']:>5s} {row['model_edge']:>+.3f} {row['weak_line_score_adjusted']:.3f} {book:<12s} {reasons}")

    # CLV evaluation
    print(f"\n{'=' * 70}")
    print("CLV EVALUATION")
    print("=" * 70)

    result = attach_and_evaluate(scored, attachable_path)
    if "metrics" in result:
        m = result["metrics"]
        print(f"\n  Overall model-side CLV:    {m['mean_model_side_clv']:+.6f}")
        print(f"  Positive CLV rate:         {m['positive_clv_rate']:.3f}")
        print(f"  Edge-CLV correlation:      {m['edge_clv_correlation']:+.4f}")

    # WLS bucket validation
    print(f"\n{'=' * 70}")
    print("WEAK LINE SCORE BUCKET VALIDATION")
    print("=" * 70)

    bucket_result = validate_wls_buckets(scored, attachable_path)
    if bucket_result["status"] == "computed":
        print(f"\n  {'Bucket':<12s} {'Rows':>5s} {'Pos CLV%':>9s} {'CI 95%':>14s} {'Mean CLV':>10s} {'Edge':>7s} {'Side'}")
        for b in bucket_result["buckets"]:
            side_str = "/".join(f"{k}:{v:.0%}" for k, v in b["side_share"].items())
            ci_str = f"[{b['positive_clv_rate_ci_lower']:.1%},{b['positive_clv_rate_ci_upper']:.1%}]"
            print(f"  {b['bucket']:<12s} {b['rows']:>5d} {b['positive_clv_rate']:>8.1%} {ci_str:>14s} {b['mean_side_clv']:>+10.6f} {b['mean_model_edge']:>+.4f} {side_str}")
        print(f"\n  Monotonicity score: {bucket_result['monotonicity_score']:.2f} ({'PASS' if bucket_result['monotonicity_pass'] else 'FAIL'})")

        # Book validation
        if bucket_result.get("book_validation"):
            print(f"\n  Book-Level CLV (concentration ok: {bucket_result['book_concentration_ok']}, max share: {bucket_result['max_book_share']:.1%}):")
            for book, bv in sorted(bucket_result["book_validation"].items(), key=lambda x: -x[1]["rows"]):
                print(f"    {book:<12s} n={bv['rows']:>4d}  pos_clv={bv['positive_clv_rate']:.1%}  mean_clv={bv['mean_clv']:+.6f}")

        # Side × market health
        if bucket_result.get("side_market_health"):
            print(f"\n  Side/Market Health (sides_positive: {bucket_result['sides_with_positive_clv']}, markets_positive: {bucket_result['markets_with_positive_clv']}):")
            for key, sv in bucket_result["side_market_health"].items():
                print(f"    {key:<6s} n={sv['rows']:>4d}  pos_clv={sv['positive_clv_rate']:.1%}  mean_clv={sv['mean_clv']:+.6f}")

        # Promotion ladder
        ladder = bucket_result.get("promotion_ladder", {})
        print(f"\n  Promotion Ladder:")
        for tier, info in ladder.items():
            print(f"    {tier}: {info['status']}")
            for k, v in info.items():
                if k != "status" and isinstance(v, bool):
                    marker = "✓" if v else "✗"
                    print(f"      {marker} {k}")
    else:
        print(f"  Status: {bucket_result['status']}")
        if "moved" in bucket_result:
            print(f"  Moved rows: {bucket_result['moved']}")

    # Gate summary
    print(f"\n{'=' * 70}")
    print("MARKET DRIFT REGIME")
    print("=" * 70)
    drift_side = scored["market_drift_side"].iloc[0] if len(scored) > 0 else "UNKNOWN"
    drift_strength = scored["market_drift_strength"].iloc[0] if len(scored) > 0 else 0.0
    fighting = scored[scored["fighting_drift"] == True]
    with_drift = scored[scored["fighting_drift"] == False]
    print(f"  Drift side:      {drift_side}")
    print(f"  Drift strength:  {drift_strength:.3f}")
    print(f"  Plays with drift:    {len(with_drift)} ({len(with_drift)/len(scored):.0%})")
    print(f"  Plays against drift: {len(fighting)} ({len(fighting)/len(scored):.0%})")

    print(f"\n{'=' * 70}")
    print("GATE SUMMARY")
    print("=" * 70)
    tier_a = scored[scored["weak_line_score_adjusted"] >= 0.55]
    tier_b = scored[(scored["weak_line_score_adjusted"] >= 0.40) & (scored["weak_line_score_adjusted"] < 0.55)]
    print(f"  Tier A (WLS >= 0.55): {len(tier_a)} plays")
    print(f"  Tier B (WLS 0.40-0.55): {len(tier_b)} plays (monitor only)")
    print(f"  Below threshold: {len(scored) - len(tier_a) - len(tier_b)} plays (no action)")
    if len(tier_a) > 0:
        a_side = tier_a["selected_side"].value_counts(normalize=True)
        print(f"  Tier A side balance: {a_side.to_dict()}")
        print(f"  Tier A mean edge: {tier_a['model_edge'].mean():+.3f}")

    # Save
    import json
    output_path = ROOT / "data" / "market_odds" / "nba" / "v9_6_sequence" / "weak_line_scored_predictions.csv"
    scored.to_csv(output_path, index=False)

    report = {
        "evaluated_at": str(pd.Timestamp.now(tz="UTC")),
        "predictions": int(len(scored)),
        "market_drift_regime": {
            "drift_side": drift_side,
            "drift_strength": drift_strength,
            "plays_with_drift": int(len(with_drift)),
            "plays_against_drift": int(len(fighting)),
        },
        "wls_distribution": {
            "mean": float(scored["weak_line_score_adjusted"].mean()),
            "median": float(scored["weak_line_score_adjusted"].median()),
            "std": float(scored["weak_line_score_adjusted"].std()),
        },
        "side_distribution": scored["selected_side"].value_counts(normalize=True).to_dict(),
        "clv_result": result.get("metrics", {}),
        "bucket_validation": bucket_result,
        "tier_a_count": int(len(tier_a)),
        "tier_b_count": int(len(tier_b)),
    }
    report_path = ROOT / "model" / "props" / "v9_6" / "validation" / "weak_line_detector_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Saved predictions: {output_path}")
    print(f"  Saved report: {report_path}")


if __name__ == "__main__":
    main()
