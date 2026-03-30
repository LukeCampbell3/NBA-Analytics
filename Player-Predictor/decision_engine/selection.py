from __future__ import annotations

import numpy as np
import pandas as pd

from .gating import StrategyConfig
from .sizing import american_profit_per_unit
try:
    from .uncertainty import belief_confidence_factor
except Exception:  # pragma: no cover - fallback when uncertainty helper module is unavailable
    def belief_confidence_factor(value, default: float = 1.0, lower: float = 0.75, upper: float = 1.15):
        series = pd.to_numeric(value, errors="coerce") if isinstance(value, pd.Series) else value
        if isinstance(series, pd.Series):
            span = max(float(upper) - float(lower), 1e-9)
            normalized = ((series.fillna(float(default)) - float(lower)) / span).clip(lower=0.0, upper=1.0)
            return (1.0 - normalized).clip(lower=0.0, upper=1.0)
        try:
            numeric = float(series)
            if np.isnan(numeric):
                numeric = float(default)
        except Exception:
            numeric = float(default)
        span = max(float(upper) - float(lower), 1e-9)
        normalized = float(np.clip((numeric - float(lower)) / span, 0.0, 1.0))
        return float(np.clip(1.0 - normalized, 0.0, 1.0))


def recommendation_rank(label: str) -> int:
    order = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}
    return order.get(str(label), 3)


def minimum_recommendation_rank(label: str) -> int:
    return {"elite": 0, "strong": 1, "consider": 2}[label]


def resolve_target_caps(config: StrategyConfig) -> dict[str, int]:
    if config.max_plays_per_target > 0:
        cap = int(config.max_plays_per_target)
        return {"PTS": cap, "TRB": cap, "AST": cap}
    return {"PTS": int(config.max_pts_plays), "TRB": int(config.max_trb_plays), "AST": int(config.max_ast_plays)}


def numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def compute_ev_columns(
    df: pd.DataFrame,
    american_odds: int,
    edge_adjust_k: float,
    belief_uncertainty_lower: float,
    belief_uncertainty_upper: float,
) -> pd.DataFrame:
    out = df.copy()
    payout = american_profit_per_unit(american_odds)
    out["expected_win_rate"] = numeric_series(out, "expected_win_rate", 0.0)
    out["expected_push_rate"] = numeric_series(out, "expected_push_rate", 0.0).clip(lower=0.0, upper=1.0)
    out["expected_loss_rate"] = (1.0 - out["expected_win_rate"] - out["expected_push_rate"]).clip(lower=0.0, upper=1.0)
    out["gap_percentile"] = numeric_series(out, "gap_percentile", 0.0)
    out["belief_uncertainty"] = numeric_series(out, "belief_uncertainty", 1.0)
    out["belief_confidence_factor"] = numeric_series(out, "belief_confidence_factor", np.nan).fillna(
        belief_confidence_factor(
            out["belief_uncertainty"],
            default=1.0,
            lower=float(belief_uncertainty_lower),
            upper=float(belief_uncertainty_upper),
        )
    )
    out["feasibility"] = numeric_series(out, "feasibility", 0.0)
    out["abs_edge"] = numeric_series(out, "abs_edge", 0.0)
    out["final_confidence"] = numeric_series(out, "final_confidence", np.nan).fillna(
        out["gap_percentile"] * out["belief_confidence_factor"] * np.clip(out["feasibility"], 0.0, None)
    )
    out["ev"] = out["expected_win_rate"] * payout - out["expected_loss_rate"]
    out["recommendation_rank"] = out["recommendation"].map(recommendation_rank)
    edge_baseline = out.groupby("target")["abs_edge"].transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
    out["edge_scale"] = (out["abs_edge"] / edge_baseline).clip(lower=0.50, upper=2.50)
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))
    return out


def apply_policy(scored: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    out = compute_ev_columns(
        scored,
        config.american_odds,
        config.edge_adjust_k,
        config.belief_uncertainty_lower,
        config.belief_uncertainty_upper,
    )
    out["passes_recommendation"] = out["recommendation_rank"] <= minimum_recommendation_rank(config.min_recommendation)
    out["passes_ev"] = out["ev"] >= float(config.min_ev)
    out["passes_final_confidence"] = out["final_confidence"] >= float(config.min_final_confidence)
    out["passes_non_pts_threshold"] = (out["target"] == "PTS") | (out["gap_percentile"] >= float(config.non_pts_min_gap_percentile))
    out["gating_passed"] = (
        out["passes_recommendation"]
        & out["passes_ev"]
        & out["passes_final_confidence"]
        & out["passes_non_pts_threshold"]
    )
    out["selected"] = False
    out["decision_stage"] = np.where(out["gating_passed"], "eligible", "rejected_gate")
    out["selection_rank"] = np.nan
    out["selected_order"] = np.nan

    eligible = out.loc[out["gating_passed"]].copy()
    if eligible.empty:
        return out.drop(columns=["recommendation_rank"])

    eligible = eligible.sort_values(
        ["recommendation_rank", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
        ascending=[True, False, False, False, False],
    ).reset_index()
    eligible["selection_rank"] = np.arange(1, len(eligible) + 1, dtype=float)

    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    total_selected = 0
    caps = resolve_target_caps(config)

    for _, row in eligible.iterrows():
        original_index = int(row["index"])
        player = str(row["player"])
        target = str(row["target"])

        if config.max_total_plays > 0 and total_selected >= int(config.max_total_plays):
            out.at[original_index, "decision_stage"] = "capped_total"
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            continue
        if config.max_plays_per_player > 0 and player_counts.get(player, 0) >= int(config.max_plays_per_player):
            out.at[original_index, "decision_stage"] = "capped_player"
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            continue
        if caps.get(target, 0) > 0 and target_counts.get(target, 0) >= int(caps[target]):
            out.at[original_index, "decision_stage"] = "capped_target"
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            continue

        total_selected += 1
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        out.at[original_index, "selected"] = True
        out.at[original_index, "decision_stage"] = "selected"
        out.at[original_index, "selection_rank"] = float(row["selection_rank"])
        out.at[original_index, "selected_order"] = float(total_selected)

    return out.drop(columns=["recommendation_rank"])
