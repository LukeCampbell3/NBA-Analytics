from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .gating import StrategyConfig
from .sizing import american_profit_per_unit


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax().replace(0.0, np.nan)
    drawdowns = (equity_curve - running_max) / running_max
    return float(drawdowns.min()) if drawdowns.notna().any() else 0.0


def numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def summarize_slice(df: pd.DataFrame, payout_per_unit: float) -> dict[str, Any]:
    rows = int(len(df))
    wins = int((df["result"] == "win").sum()) if "result" in df.columns else 0
    pushes = int((df["result"] == "push").sum()) if "result" in df.columns else 0
    losses = int((df["result"] == "loss").sum()) if "result" in df.columns else 0
    stake = float(numeric_series(df, "stake", 0.0).sum())
    profit = float(numeric_series(df, "profit", 0.0).sum())
    expected_profit = float(numeric_series(df, "expected_profit", 0.0).sum())
    win_rate = wins / rows if rows else None
    push_rate = pushes / rows if rows else None
    roi = profit / stake if stake > 0.0 else None
    expected_roi = expected_profit / stake if stake > 0.0 else None
    avg_ev = safe_float(numeric_series(df, "ev", np.nan).mean())
    avg_gap_percentile = safe_float(numeric_series(df, "gap_percentile", np.nan).mean())
    avg_final_confidence = safe_float(numeric_series(df, "final_confidence", np.nan).mean())
    flat_unit_profit = wins * payout_per_unit - losses
    flat_unit_expected_profit = float(numeric_series(df, "ev", 0.0).sum())
    return {
        "rows": rows,
        "wins": wins,
        "pushes": pushes,
        "losses": losses,
        "win_rate": win_rate,
        "push_rate": push_rate,
        "profit": profit,
        "expected_profit": expected_profit,
        "roi": roi,
        "expected_roi": expected_roi,
        "avg_ev": avg_ev,
        "avg_gap_percentile": avg_gap_percentile,
        "avg_final_confidence": avg_final_confidence,
        "profit_per_opportunity": (profit / rows) if rows else None,
        "flat_unit_profit": flat_unit_profit,
        "flat_unit_expected_profit": flat_unit_expected_profit,
        "flat_unit_roi": (flat_unit_profit / rows) if rows else None,
    }


def build_bucket_curve(df: pd.DataFrame, column: str, label: str, payout_per_unit: float, buckets: int = 5) -> list[dict[str, Any]]:
    working = df.loc[df[column].notna()].copy()
    if working.empty:
        return []
    if working[column].nunique() <= 1:
        return [
            {
                "bucket": f"{label}_all",
                **summarize_slice(working, payout_per_unit),
                f"avg_{column}": safe_float(working[column].mean()),
            }
        ]

    ranks = working[column].rank(method="first", pct=True)
    labels = [f"{label}_Q{idx}" for idx in range(1, buckets + 1)]
    working["bucket"] = pd.cut(ranks, bins=np.linspace(0.0, 1.0, buckets + 1), labels=labels, include_lowest=True)
    rows: list[dict[str, Any]] = []
    for bucket in labels:
        subset = working.loc[working["bucket"] == bucket].copy()
        if subset.empty:
            continue
        payload = summarize_slice(subset, payout_per_unit)
        payload["bucket"] = bucket
        payload[f"avg_{column}"] = safe_float(subset[column].mean())
        rows.append(payload)
    return rows


def build_ev_calibration(df: pd.DataFrame, payout_per_unit: float, buckets: int = 5) -> list[dict[str, Any]]:
    working = df.loc[df["selected"]].copy()
    if working.empty:
        return []
    return build_bucket_curve(working, "ev", "ev", payout_per_unit=payout_per_unit, buckets=buckets)


def build_selection_effectiveness(df: pd.DataFrame, payout_per_unit: float) -> dict[str, Any]:
    eligible = df.loc[df["gating_passed"]].copy()
    if eligible.empty:
        return {"eligible_pool": summarize_slice(eligible, payout_per_unit), "top_half": None, "bottom_half": None}
    midpoint = max(1, int(np.ceil(len(eligible) * 0.5)))
    ordered = eligible.sort_values("selection_rank", ascending=True)
    return {
        "eligible_pool": summarize_slice(eligible, payout_per_unit),
        "top_half": summarize_slice(ordered.head(midpoint), payout_per_unit),
        "bottom_half": summarize_slice(ordered.iloc[midpoint:], payout_per_unit) if len(ordered) > midpoint else None,
    }


def build_target_breakdown(df: pd.DataFrame, payout_per_unit: float) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for target in sorted(df["target"].dropna().unique()):
        payload[str(target)] = summarize_slice(df.loc[df["target"] == target].copy(), payout_per_unit)
    return payload


def build_alerts(decisions_df: pd.DataFrame, selected_df: pd.DataFrame, payout_per_unit: float) -> list[str]:
    alerts: list[str] = []
    if selected_df.empty:
        alerts.append("No plays were selected; policy is too restrictive or calibration history is too thin.")
        return alerts

    expected_profit = float(selected_df["expected_profit"].sum())
    realized_profit = float(selected_df["profit"].sum())
    if expected_profit > 0.0 and realized_profit < expected_profit * 0.5:
        alerts.append("Expected EV materially exceeded realized profit; calibration may be optimistic.")

    percentile_curve = build_bucket_curve(selected_df, "gap_percentile", "percentile", payout_per_unit=payout_per_unit)
    if len(percentile_curve) >= 2:
        first_rate = percentile_curve[0]["win_rate"] or 0.0
        last_rate = percentile_curve[-1]["win_rate"] or 0.0
        if last_rate < first_rate:
            alerts.append("Higher disagreement percentiles are not outperforming lower-percentile plays.")

    confidence_curve = build_bucket_curve(selected_df, "final_confidence", "confidence", payout_per_unit=payout_per_unit)
    if len(confidence_curve) >= 2:
        first_rate = confidence_curve[0]["win_rate"] or 0.0
        last_rate = confidence_curve[-1]["win_rate"] or 0.0
        if last_rate < first_rate:
            alerts.append("Higher-confidence plays are not winning more often than lower-confidence plays.")

    under_share = float((selected_df["direction"] == "UNDER").mean())
    if under_share >= 0.75:
        alerts.append("Unders dominate the selected book; directional bias may be creeping in.")
    if under_share <= 0.25:
        alerts.append("Overs dominate the selected book; directional bias may be creeping in.")

    if decisions_df.loc[decisions_df["gating_passed"]].empty:
        alerts.append("No gated opportunities were produced after scoring.")

    return alerts


def summarize_simulation(decisions_df: pd.DataFrame, daily_df: pd.DataFrame, config: StrategyConfig) -> dict[str, Any]:
    payout_per_unit = american_profit_per_unit(config.american_odds)
    selected_df = decisions_df.loc[decisions_df["selected"]].copy()
    all_candidates = summarize_slice(decisions_df, payout_per_unit)
    gated_candidates = summarize_slice(decisions_df.loc[decisions_df["gating_passed"]].copy(), payout_per_unit)
    selected_summary = summarize_slice(selected_df, payout_per_unit)

    ending_bankroll = float(daily_df["bankroll_end"].iloc[-1]) if not daily_df.empty else float(config.starting_bankroll)
    total_profit = float(selected_df["profit"].sum()) if not selected_df.empty else 0.0
    stake = float(selected_df["stake"].sum()) if not selected_df.empty else 0.0
    log_growth = float(np.log(ending_bankroll / config.starting_bankroll)) if ending_bankroll > 0.0 and config.starting_bankroll > 0.0 else None
    max_drawdown = compute_max_drawdown(daily_df["bankroll_end"]) if not daily_df.empty else 0.0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": config.name,
        "config": config.to_dict(),
        "overall": {
            "opportunities": int(len(decisions_df)),
            "gated_opportunities": int(decisions_df["gating_passed"].sum()) if "gating_passed" in decisions_df.columns else 0,
            "selected_plays": int(selected_df["selected"].sum()) if not selected_df.empty else 0,
            "days_replayed": int(daily_df["target_date"].nunique()) if not daily_df.empty else 0,
            "total_profit": total_profit,
            "total_staked": stake,
            "roi": (total_profit / stake) if stake > 0.0 else None,
            "profit_per_opportunity": (total_profit / len(decisions_df)) if len(decisions_df) else None,
            "expected_profit": float(selected_df["expected_profit"].sum()) if not selected_df.empty else 0.0,
            "ev_realization_gap": (
                float(selected_df["profit"].sum()) - float(selected_df["expected_profit"].sum())
            ) if not selected_df.empty else 0.0,
            "starting_bankroll": float(config.starting_bankroll),
            "ending_bankroll": ending_bankroll,
            "log_bankroll_growth": log_growth,
            "max_drawdown": max_drawdown,
        },
        "gating_effectiveness": {
            "all_candidates": all_candidates,
            "gated_candidates": gated_candidates,
            "selected_candidates": selected_summary,
        },
        "selection_effectiveness": build_selection_effectiveness(decisions_df, payout_per_unit),
        "ev_calibration": build_ev_calibration(decisions_df, payout_per_unit),
        "percentile_curve": build_bucket_curve(selected_df, "gap_percentile", "percentile", payout_per_unit=payout_per_unit),
        "confidence_curve": build_bucket_curve(selected_df, "final_confidence", "confidence", payout_per_unit=payout_per_unit),
        "target_breakdown": build_target_breakdown(selected_df, payout_per_unit),
        "alerts": build_alerts(decisions_df, selected_df, payout_per_unit),
    }
