from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .uncertainty import BELIEF_UNCERTAINTY_LOWER, BELIEF_UNCERTAINTY_UPPER, belief_confidence_factor
except Exception:  # pragma: no cover - fallback when uncertainty helper module is unavailable
    BELIEF_UNCERTAINTY_LOWER = 0.75
    BELIEF_UNCERTAINTY_UPPER = 1.15

    def belief_confidence_factor(value, default: float = 1.0, lower: float = BELIEF_UNCERTAINTY_LOWER, upper: float = BELIEF_UNCERTAINTY_UPPER):
        try:
            numeric = float(value)
            if np.isnan(numeric):
                numeric = float(default)
        except Exception:
            numeric = float(default)
        span = max(float(upper) - float(lower), 1e-9)
        normalized = (numeric - float(lower)) / span
        return float(np.clip(1.0 - normalized, 0.0, 1.0))


TARGETS = ("PTS", "TRB", "AST")
DEFAULT_TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.75, "strong_pct": 0.90},
    "TRB": {"consider_pct": 0.85, "strong_pct": 0.95},
    "AST": {"consider_pct": 0.85, "strong_pct": 0.95},
}


@dataclass
class StrategyConfig:
    name: str
    american_odds: int = -110
    elite_pct: float = 0.95
    probability_shrink_factor: float = 0.75
    ranking_mode: str = "ev_adjusted"
    xgb_ltr_min_train_rows: int = 4000
    xgb_ltr_num_pair_per_sample: int = 12
    robust_reranker_min_train_rows: int = 4000
    robust_reranker_holdout_days: int = 45
    robust_reranker_min_holdout_rows: int = 250
    robust_reranker_num_pair_per_sample: int = 12
    robust_reranker_min_candidate_win_rate: float = 0.55
    robust_reranker_min_candidate_final_confidence: float = 0.03
    robust_reranker_min_candidate_recommendation: str = "consider"
    accept_reject_enabled: bool = False
    accept_reject_min_train_rows: int = 3000
    accept_reject_holdout_days: int = 45
    accept_reject_min_holdout_rows: int = 250
    accept_reject_min_accept_rate: float = 0.05
    accept_reject_threshold_floor: float = 0.0
    min_ev: float = 0.0
    min_final_confidence: float = 0.03
    max_history_staleness_days: int = 0
    min_recency_factor: float = 0.0
    min_recommendation: str = "consider"
    max_plays_per_player: int = 1
    max_plays_per_target: int = 0
    max_pts_plays: int = 6
    max_trb_plays: int = 4
    max_ast_plays: int = 2
    max_total_plays: int = 12
    min_board_plays: int = 0
    non_pts_min_gap_percentile: float = 0.90
    edge_adjust_k: float = 0.30
    selection_mode: str = "ev_adjusted"
    append_agreement_min: int = 3
    append_edge_percentile_min: float = 0.90
    append_max_extra_plays: int = 3
    board_objective_overfetch: float = 4.0
    board_objective_candidate_limit: int = 36
    board_objective_max_search_nodes: int = 750000
    board_objective_lambda_corr: float = 0.12
    board_objective_lambda_conc: float = 0.07
    board_objective_lambda_unc: float = 0.06
    board_objective_corr_same_game: float = 0.65
    board_objective_corr_same_player: float = 1.0
    board_objective_corr_same_target: float = 0.15
    board_objective_corr_same_direction: float = 0.05
    board_objective_corr_same_script_cluster: float = 0.30
    board_objective_swap_candidates: int = 18
    board_objective_swap_rounds: int = 2
    board_objective_instability_enabled: bool = False
    board_objective_lambda_shadow_disagreement: float = 0.0
    board_objective_lambda_segment_weakness: float = 0.0
    board_objective_instability_near_cutoff_window: int = 3
    board_objective_instability_top_protected: int = 3
    board_objective_instability_veto_enabled: bool = False
    board_objective_instability_veto_quantile: float = 0.85
    board_objective_dynamic_size_enabled: bool = False
    board_objective_dynamic_size_max_shrink: int = 0
    board_objective_dynamic_size_trigger: float = 0.62
    board_objective_fp_veto_enabled: bool = False
    board_objective_fp_veto_live: bool = False
    board_objective_fp_veto_tail_slots: int = 2
    board_objective_fp_veto_top_protected: int = 6
    board_objective_fp_veto_threshold: float = 0.80
    board_objective_fp_veto_max_drops: int = 1
    learned_gate_enabled: bool = False
    learned_gate_min_rows: int = 0
    max_plays_per_game: int = 2
    max_plays_per_script_cluster: int = 2
    thompson_temperature: float = 1.0
    thompson_seed: int = 17
    market_regression_floor: float = 0.25
    market_regression_ceiling: float = 0.95
    min_bet_win_rate: float = 0.57
    medium_bet_win_rate: float = 0.60
    full_bet_win_rate: float = 0.65
    medium_tier_percentile: float = 0.80
    strong_tier_percentile: float = 0.90
    elite_tier_percentile: float = 0.95
    belief_uncertainty_lower: float = BELIEF_UNCERTAINTY_LOWER
    belief_uncertainty_upper: float = BELIEF_UNCERTAINTY_UPPER
    min_history_rows: int = 90
    min_history_rows_per_target: int = 25
    starting_bankroll: float = 1000.0
    sizing_method: str = "kelly"
    flat_stake: float = 10.0
    base_bet_fraction: float = 0.015
    kelly_fraction: float = 0.25
    small_bet_fraction: float = 0.005
    medium_bet_fraction: float = 0.010
    full_bet_fraction: float = 0.015
    coarse_low_bet_fraction: float = 0.003
    coarse_mid_bet_fraction: float = 0.005
    coarse_high_bet_fraction: float = 0.007
    coarse_high_max_share: float = 0.30
    coarse_mid_max_share: float = 0.50
    coarse_high_max_plays: int = 0
    coarse_mid_max_plays: int = 0
    coarse_score_alpha_uncertainty: float = 0.18
    coarse_score_beta_dependency: float = 0.12
    coarse_score_gamma_support: float = 0.08
    coarse_score_model: str = "legacy"
    coarse_score_delta_prob_weight: float = 0.0
    coarse_score_ev_weight: float = 0.0
    coarse_score_risk_weight: float = 0.0
    coarse_score_recency_weight: float = 0.0
    staking_bucket_model_enabled: bool = False
    staking_bucket_model_min_rows: int = 0
    edge_scale_start_percentile: float = 0.75
    edge_scale_span: float = 0.25
    edge_scale_lift: float = 0.15
    edge_scale_max_multiplier: float = 1.25
    max_bet_fraction: float = 0.05
    max_total_bet_fraction: float = 0.05
    min_bet_fraction: float = 0.0
    conditional_framework_enabled: bool = True
    conditional_framework_mode: str = "auto"
    conditional_anchor_min_probability: float = 0.57
    conditional_anchor_min_confidence: float = 0.05
    conditional_anchor_max_risk_penalty: float = 0.62
    conditional_min_anchor_count: int = 2
    conditional_recoverability_threshold: float = 0.52
    conditional_contradiction_threshold: float = 0.62
    conditional_noise_threshold: float = 0.68
    conditional_lambda: float = 0.45
    conditional_max_score: float = 0.35
    conditional_lift_shrinkage_k: float = 40.0
    conditional_min_pair_count: float = 25.0
    conditional_min_recent_pair_count: float = 10.0
    conditional_min_regime_pair_count: float = 8.0
    conditional_min_support: float = 0.10
    conditional_promotion_min_probability: float = 0.56
    conditional_promotion_min_ev: float = 0.02
    conditional_max_promotions_per_slate: int = 3
    conditional_max_promotions_per_game: int = 1
    conditional_max_promotions_per_player: int = 1
    conditional_max_promotions_per_script_cluster: int = 2
    conditional_max_promoted_share_of_recoverable: float = 0.35
    conditional_contrastive_weight: float = 0.20
    conditional_contrastive_clip: float = 0.08
    conditional_market_modifier_strength: float = 0.04
    conditional_market_modifier_clip: float = 0.03
    conditional_max_failure_memory_penalty: float = 0.06
    conditional_recency_half_life_days: float = 35.0
    conditional_stale_history_days: int = 21
    conditional_min_script_anchors_per_game: int = 1
    conditional_kill_switch_failure_rate: float = 0.60
    conditional_kill_switch_min_failures: int = 25
    conditional_promoted_min_recommendation: str = "consider"
    conditional_baseline_min_recommendation: str = "consider"
    conditional_failure_memory_path: str = "model/analysis/conditional_failure_memory.json"
    target_thresholds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {target: values.copy() for target, values in DEFAULT_TARGET_THRESHOLDS.items()}
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_thresholds"] = {target: values.copy() for target, values in self.target_thresholds.items()}
        return payload


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def prepare_historical_decisions(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Historical decisions CSV not found: {path}")
        df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError("Historical decisions input is empty.")

    required = {"player", "target", "prediction", "market_line", "actual", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical decisions input is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["target"] = out["target"].astype(str).str.upper()
    out = out.loc[out["target"].isin(TARGETS)].copy()
    out["target_date"] = pd.to_datetime(out.get("target_date"), errors="coerce")
    out = out.loc[out["target_date"].notna()].copy()

    bool_cols = ["active_only_row", "model_beats_market_error", "schema_repaired", "used_default_ids", "nan_feature_repaired"]
    for column in bool_cols:
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool)

    numeric_cols = [
        "prediction",
        "market_line",
        "actual",
        "baseline",
        "edge",
        "abs_edge",
        "actual_minus_market",
        "belief_uncertainty",
        "feasibility",
        "uncertainty_sigma",
        "spike_probability",
        "fallback_blend",
        "target_index",
    ]
    for column in numeric_cols:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "edge" not in out.columns:
        out["edge"] = out["prediction"] - out["market_line"]
    else:
        out["edge"] = out["edge"].fillna(out["prediction"] - out["market_line"])
    if "abs_edge" not in out.columns:
        out["abs_edge"] = out["edge"].abs()
    else:
        out["abs_edge"] = out["abs_edge"].fillna(out["edge"].abs())
    if "actual_minus_market" not in out.columns:
        out["actual_minus_market"] = out["actual"] - out["market_line"]
    else:
        out["actual_minus_market"] = out["actual_minus_market"].fillna(out["actual"] - out["market_line"])

    if "direction" not in out.columns:
        out["direction"] = np.where(out["edge"] > 0.0, "OVER", np.where(out["edge"] < 0.0, "UNDER", "PUSH"))
    else:
        out["direction"] = out["direction"].fillna("PUSH").astype(str).str.upper()

    if "active_only_row" in out.columns:
        out = out.loc[out["active_only_row"]].copy()

    out = out.sort_values(["target_date", "player", "target", "target_index"], na_position="last").reset_index(drop=True)
    target_index = out["target_index"] if "target_index" in out.columns else pd.Series(np.arange(len(out)), index=out.index, dtype="int64")
    out["opportunity_id"] = (
        out["target_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["player"].astype(str)
        + "|"
        + out["target"].astype(str)
        + "|"
        + target_index.fillna(-1).astype(int).astype(str)
    )
    return out


def percentile_of_gap(gaps_sorted: np.ndarray, gap: float) -> float:
    if gaps_sorted.size == 0:
        return 0.0
    rank = np.searchsorted(gaps_sorted, gap, side="right")
    return float(rank / gaps_sorted.size)


def classify_play(target: str, percentile: float, thresholds: dict[str, dict[str, float]]) -> str:
    target_thresholds = thresholds[target]
    elite_pct = float(target_thresholds.get("elite_pct", 0.95))
    if percentile >= elite_pct:
        return "elite"
    if percentile >= target_thresholds["strong_pct"]:
        return "strong"
    if percentile >= target_thresholds["consider_pct"]:
        return "consider"
    return "pass"


def build_history_lookup(history_df: pd.DataFrame, config: StrategyConfig) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if history_df.empty:
        return lookup

    working = history_df.copy()
    working = working.loc[working["edge"].notna() & working["abs_edge"].notna() & working["result"].notna()].copy()
    working = working.loc[working["direction"] != "PUSH"].copy()
    if working.empty:
        return lookup

    for target in TARGETS:
        target_df = working.loc[working["target"] == target].copy()
        if len(target_df) < int(config.min_history_rows_per_target):
            continue

        quartile_cut = float(target_df["abs_edge"].quantile(0.75))
        decile_cut = float(target_df["abs_edge"].quantile(0.90))

        def summarize(mask: pd.Series) -> dict[str, Any] | None:
            subset = target_df.loc[mask].copy()
            if subset.empty:
                return None
            wins = float((subset["result"] == "win").mean())
            pushes = float((subset["result"] == "push").mean())
            losses = 1.0 - wins - pushes
            return {
                "rows": int(len(subset)),
                "win_rate": wins,
                "push_rate": max(0.0, pushes),
                "loss_rate": max(0.0, losses),
            }

        lookup[target] = {
            "rows": int(len(target_df)),
            "quartile_cut": quartile_cut,
            "decile_cut": decile_cut,
            "all": summarize(pd.Series(True, index=target_df.index)),
            "top_quartile": summarize(target_df["abs_edge"] >= quartile_cut),
            "top_decile": summarize(target_df["abs_edge"] >= decile_cut),
            "gaps_sorted": np.sort(target_df["abs_edge"].to_numpy(dtype=float)),
            "last_history_date": target_df["target_date"].max(),
        }
    return lookup


def expected_rates_for(target: str, percentile: float, history_info: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> dict[str, Any]:
    target_thresholds = thresholds[target]
    bucket = history_info["all"]
    label = "all"
    elite_pct = float(target_thresholds.get("elite_pct", 0.95))
    if percentile >= elite_pct and history_info.get("top_decile"):
        bucket = history_info["top_decile"]
        label = "elite_top_decile"
    elif percentile >= target_thresholds["strong_pct"] and history_info.get("top_decile"):
        bucket = history_info["top_decile"]
        label = "top_decile"
    elif percentile >= target_thresholds["consider_pct"] and history_info.get("top_quartile"):
        bucket = history_info["top_quartile"]
        label = "top_quartile"
    return {
        "bucket": label,
        "raw_expected_win_rate": float(bucket["win_rate"]),
        "raw_expected_push_rate": float(bucket["push_rate"]),
        "raw_expected_loss_rate": float(bucket["loss_rate"]),
        "history_rows": int(bucket["rows"]),
    }


def shrink_probability_triplet(
    win_rate: float,
    push_rate: float,
    loss_rate: float,
    shrink_factor: float,
) -> tuple[float, float, float]:
    shrink = float(np.clip(shrink_factor, 0.0, 1.0))
    push = float(np.clip(push_rate, 0.0, 1.0))
    non_push = max(0.0, 1.0 - push)
    if non_push <= 0.0:
        return 0.0, push, 0.0

    win_conditional = float(np.clip(win_rate / non_push, 0.0, 1.0))
    win_conditional_shrunk = 0.5 + shrink * (win_conditional - 0.5)
    win_conditional_shrunk = float(np.clip(win_conditional_shrunk, 0.0, 1.0))
    win_adj = win_conditional_shrunk * non_push
    loss_adj = non_push - win_adj
    return win_adj, push, loss_adj


def score_candidates(current_df: pd.DataFrame, history_lookup: dict[str, dict[str, Any]], config: StrategyConfig) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()

    rows: list[dict[str, Any]] = []
    thresholds = {
        target: {
            **values,
            "elite_pct": float(values.get("elite_pct", config.elite_pct)),
        }
        for target, values in config.target_thresholds.items()
    }
    for _, row in current_df.iterrows():
        target = str(row["target"]).upper()
        history_info = history_lookup.get(target)
        if history_info is None:
            continue

        edge = safe_float(row.get("edge"))
        abs_edge = abs(edge)
        gap_percentile = percentile_of_gap(history_info["gaps_sorted"], abs_edge)
        expected_rates = expected_rates_for(target, gap_percentile, history_info, thresholds)

        belief = safe_float(row.get("belief_uncertainty"), default=1.0)
        belief_conf = belief_confidence_factor(
            belief,
            default=1.0,
            lower=float(config.belief_uncertainty_lower),
            upper=float(config.belief_uncertainty_upper),
        )
        feasibility = max(0.0, safe_float(row.get("feasibility"), default=0.0))
        confidence_score = abs_edge * belief_conf * feasibility
        final_confidence = gap_percentile * belief_conf * feasibility

        rows.append(
            {
                **row.to_dict(),
                "strategy_name": config.name,
                "gap_percentile": gap_percentile,
                "recommendation": classify_play(target, gap_percentile, thresholds),
                "raw_expected_win_rate": expected_rates["raw_expected_win_rate"],
                "raw_expected_push_rate": expected_rates["raw_expected_push_rate"],
                "raw_expected_loss_rate": expected_rates["raw_expected_loss_rate"],
                "confidence_score": confidence_score,
                "final_confidence": final_confidence,
                "history_rows": int(history_info["rows"]),
                "bucket_history_rows": int(expected_rates["history_rows"]),
                "calibration_bucket": expected_rates["bucket"],
                "last_history_date": history_info["last_history_date"],
            }
        )

    scored = pd.DataFrame.from_records(rows)
    if scored.empty:
        return scored

    shrunk = scored.apply(
        lambda row: shrink_probability_triplet(
            win_rate=float(row["raw_expected_win_rate"]),
            push_rate=float(row["raw_expected_push_rate"]),
            loss_rate=float(row["raw_expected_loss_rate"]),
            shrink_factor=config.probability_shrink_factor,
        ),
        axis=1,
    )
    shrunk_df = pd.DataFrame(shrunk.tolist(), columns=["expected_win_rate", "expected_push_rate", "expected_loss_rate"], index=scored.index)
    scored[["expected_win_rate", "expected_push_rate", "expected_loss_rate"]] = shrunk_df

    scored = scored.sort_values(
        ["target_date", "target", "gap_percentile", "abs_edge"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    return scored
