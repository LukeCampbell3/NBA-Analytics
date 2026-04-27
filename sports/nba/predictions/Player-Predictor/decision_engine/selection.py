from __future__ import annotations

import numpy as np
import pandas as pd

from .board_optimizer import optimize_board, prepare_board_candidates
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
    return order.get(str(label).strip().lower(), 3)


def minimum_recommendation_rank(label: str) -> int:
    return {"elite": 0, "strong": 1, "consider": 2, "pass": 3}[str(label).strip().lower()]


def resolve_target_caps(config: StrategyConfig) -> dict[str, int]:
    if config.max_plays_per_target > 0:
        cap = int(config.max_plays_per_target)
        return {"PTS": cap, "TRB": cap, "AST": cap}
    return {"PTS": int(config.max_pts_plays), "TRB": int(config.max_trb_plays), "AST": int(config.max_ast_plays)}


def numeric_series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def string_series(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def resolve_selection_mode(config: StrategyConfig) -> str:
    return str(config.selection_mode or config.ranking_mode or "ev_adjusted").strip().lower()


def normalize_script_cluster(value: object) -> str:
    token = str(value or "").strip()
    if not token or token.lower() == "script=unknown":
        return ""
    return token


def build_game_key(df: pd.DataFrame) -> pd.Series:
    existing = string_series(df, "game_key").str.strip()
    event = string_series(df, "market_event_id").str.strip()
    home = string_series(df, "market_home_team").str.strip()
    away = string_series(df, "market_away_team").str.strip()
    teams = (home + "::" + away).str.strip(":")
    fallback = event.where(event != "", teams)
    return existing.where(existing != "", fallback)


def ranking_columns_for_mode(df: pd.DataFrame, mode: str) -> list[str]:
    columns = {
        "edge": ["recommendation_rank", "edge", "abs_edge", "expected_win_rate", "final_confidence", "ev_adjusted"],
        "abs_edge": ["recommendation_rank", "abs_edge", "ev_adjusted", "expected_win_rate", "final_confidence"],
        "thompson_ev": ["recommendation_rank", "thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
        "board_objective": ["board_objective_base_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
    }.get(mode, ["recommendation_rank", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"])
    return [column for column in columns if column in df.columns]


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
    out["edge"] = numeric_series(out, "edge", 0.0)
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
    out["recommendation_rank"] = out["recommendation"].map(recommendation_rank).fillna(3)
    edge_baseline = out.groupby("target")["abs_edge"].transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
    out["edge_scale"] = (out["abs_edge"] / edge_baseline).clip(lower=0.50, upper=2.50)
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))
    out["thompson_ev"] = numeric_series(out, "thompson_ev", np.nan).fillna(out["ev_adjusted"])
    out["direction"] = string_series(out, "direction").str.upper().str.strip()
    missing_direction = out["direction"] == ""
    if missing_direction.any():
        out.loc[missing_direction, "direction"] = np.where(
            out.loc[missing_direction, "edge"] > 0.0,
            "OVER",
            np.where(out.loc[missing_direction, "edge"] < 0.0, "UNDER", "PUSH"),
        )
    out["game_key"] = build_game_key(out).fillna("").astype(str).str.strip()
    out["script_cluster_id"] = string_series(out, "script_cluster_id").map(normalize_script_cluster)
    return out


def apply_portfolio_caps(eligible: pd.DataFrame, config: StrategyConfig, mode: str) -> pd.DataFrame:
    ranked = eligible.copy()
    sort_columns = ranking_columns_for_mode(ranked, mode)
    if not sort_columns:
        sort_columns = ["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"]
    ascending = [column == "recommendation_rank" for column in sort_columns]
    ranked = ranked.sort_values(sort_columns, ascending=ascending).reset_index()
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1, dtype=float)
    return ranked


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

    mode = resolve_selection_mode(config)
    if mode == "board_objective":
        prepared = prepare_board_candidates(eligible, config)
        ranked = apply_portfolio_caps(prepared, config, mode)
        candidate_result = optimize_board(prepared, config)
        selected_board = candidate_result.selected_board.copy()
        candidate_pool = candidate_result.candidate_pool.copy()

        candidate_indices = set(candidate_pool.index.tolist())
        selected_indices = set(selected_board.index.tolist())
        for _, row in ranked.iterrows():
            original_index = int(row["index"])
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            if original_index in selected_indices:
                continue
            out.at[original_index, "decision_stage"] = "portfolio_excluded" if original_index in candidate_indices else "portfolio_candidate_cut"

        if not selected_board.empty:
            selected_ranked = selected_board.sort_values(
                ["board_objective_base_score", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
                ascending=[False, False, False, False, False],
            )
            for order, original_index in enumerate(selected_ranked.index.tolist(), start=1):
                out.at[int(original_index), "selected"] = True
                out.at[int(original_index), "decision_stage"] = "selected"
                out.at[int(original_index), "selected_order"] = float(order)
                out.at[int(original_index), "board_objective_score"] = float(candidate_result.board_score)
                out.at[int(original_index), "board_objective_beam_width"] = float(candidate_result.beam_width)
                out.at[int(original_index), "board_objective_candidate_count"] = float(len(candidate_pool))
        return out.drop(columns=["recommendation_rank"])

    eligible_ranked = apply_portfolio_caps(eligible, config, mode)

    player_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    script_cluster_counts: dict[str, int] = {}
    total_selected = 0
    caps = resolve_target_caps(config)

    for _, row in eligible_ranked.iterrows():
        original_index = int(row["index"])
        player = str(row["player"])
        target = str(row["target"])
        game_key = str(row.get("game_key", "")).strip()
        script_cluster = normalize_script_cluster(row.get("script_cluster_id", ""))

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
        if config.max_plays_per_game > 0 and game_key and game_counts.get(game_key, 0) >= int(config.max_plays_per_game):
            out.at[original_index, "decision_stage"] = "capped_game"
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            continue
        if (
            config.max_plays_per_script_cluster > 0
            and script_cluster
            and script_cluster_counts.get(script_cluster, 0) >= int(config.max_plays_per_script_cluster)
        ):
            out.at[original_index, "decision_stage"] = "capped_script_cluster"
            out.at[original_index, "selection_rank"] = float(row["selection_rank"])
            continue

        total_selected += 1
        player_counts[player] = player_counts.get(player, 0) + 1
        target_counts[target] = target_counts.get(target, 0) + 1
        if game_key:
            game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if script_cluster:
            script_cluster_counts[script_cluster] = script_cluster_counts.get(script_cluster, 0) + 1
        out.at[original_index, "selected"] = True
        out.at[original_index, "decision_stage"] = "selected"
        out.at[original_index, "selection_rank"] = float(row["selection_rank"])
        out.at[original_index, "selected_order"] = float(total_selected)

    return out.drop(columns=["recommendation_rank"])
