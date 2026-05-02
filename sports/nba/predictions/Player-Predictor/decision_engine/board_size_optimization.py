from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BREAKEVEN_RATE_NEG110 = 110.0 / 210.0
WILSON_Z_90 = 1.2815515655446004


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z_90) -> float:
    n = int(max(0, total))
    if n <= 0:
        return 0.0
    p = float(successes) / float(n)
    z_sq = float(z) ** 2
    denom = 1.0 + z_sq / float(n)
    center = p + z_sq / (2.0 * float(n))
    margin = float(z) * np.sqrt((p * (1.0 - p) + z_sq / (4.0 * float(n))) / float(n))
    return float(np.clip((center - margin) / denom, 0.0, 1.0))


def summarize_board_size_history(
    daily_df: pd.DataFrame,
    *,
    payout_per_win: float = (100.0 / 110.0),
    breakeven_rate: float = BREAKEVEN_RATE_NEG110,
    prior_strength: float = 24.0,
    min_resolved_for_full_weight: int = 40,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    required = {"board_size_requested", "board_size_realized", "resolved", "wins", "losses", "units"}
    missing = [column for column in required if column not in daily_df.columns]
    if missing:
        raise ValueError(f"Daily board history missing required columns: {missing}")

    working = daily_df.copy()
    working["board_size_requested"] = pd.to_numeric(working["board_size_requested"], errors="coerce").fillna(0).astype(int)
    working["board_size_realized"] = pd.to_numeric(working["board_size_realized"], errors="coerce").fillna(0).astype(int)
    working["resolved"] = pd.to_numeric(working["resolved"], errors="coerce").fillna(0).astype(int)
    working["wins"] = pd.to_numeric(working["wins"], errors="coerce").fillna(0).astype(int)
    working["losses"] = pd.to_numeric(working["losses"], errors="coerce").fillna(0).astype(int)
    working["units"] = pd.to_numeric(working["units"], errors="coerce").fillna(0.0).astype("float64")
    working["expected_win_rate_mean"] = pd.to_numeric(working.get("expected_win_rate_mean"), errors="coerce").fillna(np.nan)
    working["board_full"] = working["board_size_realized"] >= working["board_size_requested"]
    working["resolved_day"] = working["resolved"] > 0
    working["positive_units_day"] = working["units"] > 0.0

    rows: list[dict[str, Any]] = []
    for requested_size, part in working.groupby("board_size_requested", dropna=False):
        days = int(len(part))
        resolved = int(part["resolved"].sum())
        wins = int(part["wins"].sum())
        losses = int(part["losses"].sum())
        units = float(part["units"].sum())
        hit_rate = float(wins / resolved) if resolved > 0 else np.nan
        roi_per_play = float(units / resolved) if resolved > 0 else np.nan
        avg_realized_size = float(part["board_size_realized"].mean()) if len(part) else 0.0
        fulfillment_rate = float(part["board_full"].mean()) if len(part) else 0.0
        realized_share = float(avg_realized_size / max(int(requested_size), 1)) if int(requested_size) > 0 else 0.0
        resolved_days = int(part["resolved_day"].sum())
        positive_days = int(part["positive_units_day"].sum())
        expected_win_rate_mean = float(part["expected_win_rate_mean"].mean()) if part["expected_win_rate_mean"].notna().any() else np.nan

        hit_rate_lcb_90 = wilson_lower_bound(wins, resolved, z=WILSON_Z_90)
        prior_wins = float(prior_strength) * float(breakeven_rate)
        posterior_hit_rate = float((wins + prior_wins) / max(resolved + float(prior_strength), 1e-9))
        conservative_edge = float(hit_rate_lcb_90 - float(breakeven_rate))
        posterior_edge = float(posterior_hit_rate - float(breakeven_rate))
        sample_weight = float(np.clip(resolved / max(int(min_resolved_for_full_weight), 1), 0.0, 1.0))
        availability = float(np.clip(0.55 * fulfillment_rate + 0.45 * np.clip(realized_share, 0.0, 1.0), 0.0, 1.0))
        objective_score = float(
            availability
            * (
                0.65 * conservative_edge
                + 0.25 * posterior_edge
                + 0.10 * np.clip(roi_per_play if np.isfinite(roi_per_play) else 0.0, -0.25, 0.25)
            )
            * (0.35 + 0.65 * sample_weight)
        )

        rows.append(
            {
                "board_size_requested": int(requested_size),
                "days": days,
                "resolved_days": resolved_days,
                "positive_unit_days": positive_days,
                "resolved": resolved,
                "wins": wins,
                "losses": losses,
                "hit_rate": hit_rate,
                "hit_rate_lcb_90": hit_rate_lcb_90,
                "posterior_hit_rate": posterior_hit_rate,
                "units": units,
                "roi_per_play": roi_per_play,
                "avg_realized_size": avg_realized_size,
                "size_fulfillment_rate": fulfillment_rate,
                "size_realization_share": realized_share,
                "avg_expected_win_rate": expected_win_rate_mean,
                "breakeven_rate": float(breakeven_rate),
                "conservative_edge": conservative_edge,
                "posterior_edge": posterior_edge,
                "sample_weight": sample_weight,
                "availability_score": availability,
                "objective_score": objective_score,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["objective_score", "hit_rate_lcb_90", "roi_per_play", "board_size_requested"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def recommend_board_size(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {
            "recommended_board_size": 0,
            "reason": "no_summary_rows",
        }

    ranked = summary_df.sort_values(
        ["objective_score", "hit_rate_lcb_90", "roi_per_play", "board_size_requested"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    return {
        "recommended_board_size": int(best.get("board_size_requested", 0)),
        "objective_score": float(best.get("objective_score", 0.0)),
        "hit_rate": float(best.get("hit_rate", np.nan)),
        "hit_rate_lcb_90": float(best.get("hit_rate_lcb_90", np.nan)),
        "roi_per_play": float(best.get("roi_per_play", np.nan)),
        "avg_realized_size": float(best.get("avg_realized_size", np.nan)),
        "size_fulfillment_rate": float(best.get("size_fulfillment_rate", np.nan)),
        "resolved": int(best.get("resolved", 0)),
        "days": int(best.get("days", 0)),
        "reason": "max_objective_score",
    }
