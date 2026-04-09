#!/usr/bin/env python3
"""
Accepted-pick gate research orchestration (shadow-only by design).

This script keeps live behavior frozen while continuously learning a robust
keep/drop gate on accepted picks:
1) snapshot: freeze daily production artifacts and build repeat-run stability
2) append-settled: append settled outcomes/attribution to accepted-pick history
3) train-shadow: walk-forward train/eval candidate gate and emit promotion report
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.accepted_pick_gate import (
    CatBoostGateConfig,
    CatBoostKeepDropGate,
    GatePolicyConfig,
    LogisticGateConfig,
    PromotionCriteria,
    RegularizedLogisticGate,
    UpliftCatBoostGate,
    UpliftCatBoostGateConfig,
    WalkForwardConfig,
    apply_accepted_pick_gate,
    apply_shadow_gate_policy,
    build_meta_cohort_columns,
    build_pick_key,
    build_promotion_recommendation,
    concentration_stats,
    expected_utility_from_result,
    iter_walk_forward_windows,
    normalize_player_name,
    predict_keep_harm_scores,
    result_to_keep_label,
    rolling_window_paired_deltas,
    summarize_paired_outcomes,
)
from post_process_market_plays import compute_final_board


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "model" / "analysis" / "accepted_pick_gate"
DEFAULT_HISTORY_CSV = DEFAULT_OUTPUT_ROOT / "history" / "accepted_pick_history.csv"
DEFAULT_REPLAY_HISTORY_CSV = DEFAULT_OUTPUT_ROOT / "history" / "accepted_pick_history_replay.csv"
DEFAULT_REPLAY_HISTORY_REPORT_JSON = DEFAULT_OUTPUT_ROOT / "history" / "accepted_pick_history_replay_build_report.json"
DEFAULT_CANDIDATE_JSON = DEFAULT_OUTPUT_ROOT / "candidates" / "accepted_pick_gate_candidate.json"
DEFAULT_REPORT_JSON = DEFAULT_OUTPUT_ROOT / "reports" / "accepted_pick_gate_report.json"
DEFAULT_SCORED_ROWS_CSV = DEFAULT_OUTPUT_ROOT / "reports" / "accepted_pick_gate_scored_rows.csv"
DEFAULT_PAIRED_EVAL_JSON = DEFAULT_OUTPUT_ROOT / "eval" / "paired_eval_summary.json"
DEFAULT_PAIRED_EVAL_ROWS_CSV = DEFAULT_OUTPUT_ROOT / "eval" / "paired_eval_scored_rows.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _player_concentration_features(concentration: dict[str, Any] | None) -> dict[str, float]:
    payload = concentration if isinstance(concentration, dict) else {}
    top_player = _safe_float(payload.get("top_player_share"), 0.0)
    player_hhi = _safe_float(payload.get("player_hhi"), 0.0)
    player_diversity = _safe_float(payload.get("player_diversity"), 0.0)
    unique_players = float(max(0, _safe_int(payload.get("unique_players"), 0)))
    veto_rows = float(max(0, _safe_int(payload.get("veto_rows"), 0)))
    if player_diversity <= 0.0 and veto_rows > 0.0:
        player_diversity = float(min(1.0, unique_players / veto_rows))
    return {
        "top_player_share": float(max(0.0, top_player)),
        "player_hhi": float(max(0.0, player_hhi)),
        "player_diversity": float(np.clip(player_diversity, 0.0, 1.0)),
    }


def _player_concentration_objective_adjustment(
    concentration: dict[str, Any] | None,
    *,
    top_share_penalty: float,
    hhi_penalty: float,
    diversity_bonus: float,
) -> float:
    features = _player_concentration_features(concentration)
    return (
        -float(max(0.0, top_share_penalty)) * float(features["top_player_share"])
        - float(max(0.0, hhi_penalty)) * float(features["player_hhi"])
        + float(max(0.0, diversity_bonus)) * float(features["player_diversity"])
    )


def _effective_target_cap(
    *,
    base_cap: float,
    top_player_share: float,
    recent_profit: float,
    recent_hit_pp: float,
    conditional_enabled: bool,
    conditional_relaxed_cap: float,
    conditional_player_share_max: float,
    conditional_require_recent_non_negative: bool,
    conditional_recent_profit_floor: float,
    conditional_recent_hit_floor_pp: float,
) -> tuple[float, bool]:
    cap = float(base_cap)
    active = False
    if not bool(conditional_enabled):
        return cap, active
    player_ok = float(top_player_share) <= float(conditional_player_share_max)
    recent_ok = True
    if bool(conditional_require_recent_non_negative):
        recent_ok = bool(
            float(recent_profit) >= float(conditional_recent_profit_floor)
            and (
                float(recent_hit_pp) != float(recent_hit_pp)
                or float(recent_hit_pp) >= float(conditional_recent_hit_floor_pp)
            )
        )
    if player_ok and recent_ok:
        active = True
        cap = float(max(float(base_cap), float(conditional_relaxed_cap)))
    return float(cap), bool(active)


def _bootstrap_paired_deltas(
    frame: pd.DataFrame,
    *,
    veto_col: str,
    result_col: str,
    iterations: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    out = {
        "enabled": False,
        "iterations": int(max(0, iterations)),
        "alpha": float(alpha),
        "sample_rows": int(len(frame)),
        "profit_units_ci": [float("nan"), float("nan")],
        "hit_rate_pp_ci": [float("nan"), float("nan")],
        "profit_units_mean": float("nan"),
        "hit_rate_pp_mean": float("nan"),
    }
    if frame.empty or int(iterations) <= 0:
        return out
    resolved = frame.loc[frame[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
    if resolved.empty:
        return out
    rng = np.random.default_rng(int(seed))
    n = int(len(resolved))
    if n <= 0:
        return out
    profit_vals: list[float] = []
    hit_vals: list[float] = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, n, n)
        sample = resolved.iloc[idx].copy()
        summary = summarize_paired_outcomes(sample, veto_col=veto_col, result_col=result_col)
        delta = summary.get("delta", {})
        profit_vals.append(_safe_float(delta.get("profit_units"), float("nan")))
        hit_vals.append(_safe_float(delta.get("hit_rate_pp"), float("nan")))
    profit_arr = np.asarray(profit_vals, dtype="float64")
    hit_arr = np.asarray(hit_vals, dtype="float64")
    lo_q = float(np.clip(float(alpha) / 2.0, 0.0, 0.5))
    hi_q = float(1.0 - lo_q)
    out["enabled"] = True
    out["profit_units_ci"] = [float(np.nanquantile(profit_arr, lo_q)), float(np.nanquantile(profit_arr, hi_q))]
    out["hit_rate_pp_ci"] = [float(np.nanquantile(hit_arr, lo_q)), float(np.nanquantile(hit_arr, hi_q))]
    out["profit_units_mean"] = float(np.nanmean(profit_arr))
    out["hit_rate_pp_mean"] = float(np.nanmean(hit_arr))
    return out


def _leave_one_player_out_validation(
    frame: pd.DataFrame,
    *,
    player_col: str,
    veto_col: str,
    result_col: str,
    min_player_rows: int,
    profit_floor: float,
    hit_floor_pp: float,
) -> dict[str, Any]:
    out = {
        "enabled": False,
        "players_evaluated": 0,
        "profit_pass_rate": float("nan"),
        "hit_pass_rate": float("nan"),
        "profit_floor": float(profit_floor),
        "hit_floor_pp": float(hit_floor_pp),
        "min_player_rows": int(max(1, min_player_rows)),
        "worst_profit_units": float("nan"),
        "worst_hit_rate_pp": float("nan"),
    }
    if frame.empty:
        return out
    resolved = frame.loc[frame[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
    if resolved.empty:
        return out
    players = resolved.get(player_col, pd.Series("", index=resolved.index)).map(normalize_player_name)
    counts = players.value_counts()
    keep_players = [str(name) for name, cnt in counts.items() if int(cnt) >= int(max(1, min_player_rows)) and str(name).strip()]
    if not keep_players:
        return out
    profits: list[float] = []
    hits: list[float] = []
    for player in keep_players:
        mask = players.ne(player)
        part = resolved.loc[mask].copy()
        if part.empty:
            continue
        summary = summarize_paired_outcomes(part, veto_col=veto_col, result_col=result_col)
        delta = summary.get("delta", {})
        profits.append(_safe_float(delta.get("profit_units"), float("nan")))
        hits.append(_safe_float(delta.get("hit_rate_pp"), float("nan")))
    if not profits:
        return out
    profit_arr = np.asarray(profits, dtype="float64")
    hit_arr = np.asarray(hits, dtype="float64")
    out["enabled"] = True
    out["players_evaluated"] = int(len(profit_arr))
    out["profit_pass_rate"] = float(np.nanmean(profit_arr >= float(profit_floor)))
    out["hit_pass_rate"] = float(np.nanmean(hit_arr >= float(hit_floor_pp)))
    out["worst_profit_units"] = float(np.nanmin(profit_arr))
    out["worst_hit_rate_pp"] = float(np.nanmin(hit_arr))
    return out


def _apply_small_lift_validation_to_recommendation(
    recommendation: dict[str, Any],
    *,
    lopo: dict[str, Any],
    bootstrap: dict[str, Any],
    enforce: bool,
    min_lopo_profit_pass_rate: float,
    min_lopo_hit_pass_rate: float,
    min_bootstrap_profit_ci_lower: float,
    min_bootstrap_hit_ci_lower_pp: float,
) -> dict[str, Any]:
    rec = dict(recommendation or {})
    failures = [str(x) for x in rec.get("failures", [])]
    if bool(enforce):
        if bool(lopo.get("enabled", False)):
            if _safe_float(lopo.get("profit_pass_rate"), float("-inf")) < float(min_lopo_profit_pass_rate):
                failures.append("lopo_profit_pass_rate_below_floor")
            if _safe_float(lopo.get("hit_pass_rate"), float("-inf")) < float(min_lopo_hit_pass_rate):
                failures.append("lopo_hit_pass_rate_below_floor")
        profit_ci = lopo_ci_default = [float("nan"), float("nan")]
        hit_ci = lopo_ci_default
        if isinstance(bootstrap.get("profit_units_ci"), list) and len(bootstrap.get("profit_units_ci")) >= 2:
            profit_ci = bootstrap.get("profit_units_ci")
        if isinstance(bootstrap.get("hit_rate_pp_ci"), list) and len(bootstrap.get("hit_rate_pp_ci")) >= 2:
            hit_ci = bootstrap.get("hit_rate_pp_ci")
        if bool(bootstrap.get("enabled", False)):
            if _safe_float(profit_ci[0], float("-inf")) < float(min_bootstrap_profit_ci_lower):
                failures.append("bootstrap_profit_ci_lower_below_floor")
            if _safe_float(hit_ci[0], float("-inf")) < float(min_bootstrap_hit_ci_lower_pp):
                failures.append("bootstrap_hit_ci_lower_below_floor")
    rec["small_lift_validation"] = {
        "enabled": bool(enforce),
        "leave_one_player_out": lopo,
        "bootstrap": bootstrap,
        "floors": {
            "min_lopo_profit_pass_rate": float(min_lopo_profit_pass_rate),
            "min_lopo_hit_pass_rate": float(min_lopo_hit_pass_rate),
            "min_bootstrap_profit_ci_lower": float(min_bootstrap_profit_ci_lower),
            "min_bootstrap_hit_ci_lower_pp": float(min_bootstrap_hit_ci_lower_pp),
        },
    }
    rec["failures"] = sorted(set(failures))
    rec["pass"] = bool(len(rec["failures"]) == 0)
    return rec


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_run_stamp(run_date: str) -> str:
    ts = pd.to_datetime(run_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid run date: {run_date}")
    return ts.strftime("%Y%m%d")


def _to_run_date_iso(run_date: str) -> str:
    ts = pd.to_datetime(run_date, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid run date: {run_date}")
    return ts.strftime("%Y-%m-%d")


def _resolve_manifest(
    *,
    run_date: str | None,
    run_dir: Path | None,
    manifest: Path | None,
    daily_runs_dir: Path,
) -> tuple[Path, Path, str]:
    if manifest is not None:
        manifest_path = manifest.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if run_dir is None:
            run_dir = manifest_path.parent
        run_stamp = run_dir.name if run_dir is not None else ""
        if not (run_stamp.isdigit() and len(run_stamp) == 8):
            run_stamp = _to_run_stamp(run_date or "")
        return manifest_path, run_dir.resolve(), run_stamp

    if run_dir is not None:
        run_path = run_dir.resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        run_stamp = run_path.name
        if not (run_stamp.isdigit() and len(run_stamp) == 8):
            run_stamp = _to_run_stamp(run_date or "")
        manifest_path = run_path / f"daily_market_pipeline_manifest_{run_stamp}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return manifest_path.resolve(), run_path, run_stamp

    if not run_date:
        raise ValueError("Provide --run-date, --run-dir, or --manifest.")
    run_stamp = _to_run_stamp(run_date)
    run_path = (daily_runs_dir / run_stamp).resolve()
    manifest_path = run_path / f"daily_market_pipeline_manifest_{run_stamp}.json"
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return manifest_path.resolve(), run_path, run_stamp


def _load_optional_payload(path_like: Any) -> dict[str, Any] | None:
    try:
        if path_like is None:
            return None
        path = Path(str(path_like))
        if not path.exists():
            return None
        payload = _read_json(path)
        return payload if payload else None
    except Exception:
        return None


def _resolve_shadow_run(
    manifest_payload: dict[str, Any],
    preferred_profile: str | None = None,
) -> dict[str, Any] | None:
    runs = manifest_payload.get("shadow_runs")
    if not isinstance(runs, list) or not runs:
        return None
    if preferred_profile:
        for item in runs:
            if str(item.get("policy_profile", "")) == str(preferred_profile):
                return item if isinstance(item, dict) else None
    first = runs[0]
    return first if isinstance(first, dict) else None


def _build_board_kwargs_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("Missing policy payload in final JSON.")
    known_keys = {
        "american_odds",
        "min_ev",
        "min_final_confidence",
        "min_recommendation",
        "selection_mode",
        "ranking_mode",
        "max_plays_per_player",
        "max_plays_per_target",
        "max_total_plays",
        "min_board_plays",
        "max_plays_per_game",
        "max_plays_per_script_cluster",
        "non_pts_min_gap_percentile",
        "edge_adjust_k",
        "thompson_temperature",
        "thompson_seed",
        "min_bet_win_rate",
        "medium_bet_win_rate",
        "full_bet_win_rate",
        "medium_tier_percentile",
        "strong_tier_percentile",
        "elite_tier_percentile",
        "small_bet_fraction",
        "medium_bet_fraction",
        "full_bet_fraction",
        "max_bet_fraction",
        "max_total_bet_fraction",
        "sizing_method",
        "flat_bet_fraction",
        "coarse_low_bet_fraction",
        "coarse_mid_bet_fraction",
        "coarse_high_bet_fraction",
        "coarse_high_max_share",
        "coarse_mid_max_share",
        "coarse_high_max_plays",
        "coarse_mid_max_plays",
        "coarse_score_alpha_uncertainty",
        "coarse_score_beta_dependency",
        "coarse_score_gamma_support",
        "coarse_score_model",
        "coarse_score_delta_prob_weight",
        "coarse_score_ev_weight",
        "coarse_score_risk_weight",
        "coarse_score_recency_weight",
        "belief_uncertainty_lower",
        "belief_uncertainty_upper",
        "append_agreement_min",
        "append_edge_percentile_min",
        "append_max_extra_plays",
        "board_objective_overfetch",
        "board_objective_candidate_limit",
        "board_objective_max_search_nodes",
        "board_objective_lambda_corr",
        "board_objective_lambda_conc",
        "board_objective_lambda_unc",
        "board_objective_corr_same_game",
        "board_objective_corr_same_player",
        "board_objective_corr_same_target",
        "board_objective_corr_same_direction",
        "board_objective_corr_same_script_cluster",
        "board_objective_swap_candidates",
        "board_objective_swap_rounds",
        "board_objective_instability_enabled",
        "board_objective_lambda_shadow_disagreement",
        "board_objective_lambda_segment_weakness",
        "board_objective_instability_near_cutoff_window",
        "board_objective_instability_top_protected",
        "board_objective_instability_veto_enabled",
        "board_objective_instability_veto_quantile",
        "board_objective_dynamic_size_enabled",
        "board_objective_dynamic_size_max_shrink",
        "board_objective_dynamic_size_trigger",
        "board_objective_fp_veto_enabled",
        "board_objective_fp_veto_live",
        "board_objective_fp_veto_tail_slots",
        "board_objective_fp_veto_top_protected",
        "board_objective_fp_veto_threshold",
        "board_objective_fp_veto_max_drops",
        "board_objective_fp_veto_quantile",
        "board_objective_fp_veto_max_swaps",
        "board_objective_fp_veto_swap_candidates",
        "board_objective_fp_veto_min_swap_gain",
        "board_objective_fp_veto_risk_lambda",
        "board_objective_fp_veto_ml_weight",
        "max_history_staleness_days",
        "min_recency_factor",
        "learned_gate_min_rows",
    }
    out = {key: policy[key] for key in known_keys if key in policy}
    out["max_target_plays"] = {
        "PTS": int(max(0, _safe_int(policy.get("max_pts_plays"), 0))),
        "TRB": int(max(0, _safe_int(policy.get("max_trb_plays"), 0))),
        "AST": int(max(0, _safe_int(policy.get("max_ast_plays"), 0))),
    }
    out["learned_gate_rescue_enabled"] = True
    return out


def _parse_seed_list(raw: str | None, repeat_runs: int, start: int, step: int) -> list[int]:
    if raw and str(raw).strip():
        out: list[int] = []
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            out.append(int(token))
        if not out:
            raise ValueError("seed-list was provided but no valid integer seed was parsed.")
        return out
    count = int(max(0, repeat_runs))
    return [int(start + idx * step) for idx in range(count)]


def _cooccurrence_instability(
    baseline_keys: list[str],
    repeats_long: pd.DataFrame,
    run_col: str,
    key_col: str,
) -> dict[str, float]:
    if repeats_long.empty:
        return {key: float("nan") for key in baseline_keys}
    run_sets: dict[Any, set[str]] = {}
    for run_id, part in repeats_long.groupby(run_col, dropna=False):
        run_sets[run_id] = set(str(v) for v in part[key_col].dropna().astype(str).tolist())
    baseline_set = set(str(v) for v in baseline_keys)
    out: dict[str, float] = {}
    for key in baseline_keys:
        sims: list[float] = []
        target_other = baseline_set - {str(key)}
        for run_set in run_sets.values():
            if str(key) not in run_set:
                continue
            run_other = run_set - {str(key)}
            union = run_other | target_other
            if not union:
                sims.append(1.0)
                continue
            sims.append(float(len(run_other & target_other)) / float(len(union)))
        if not sims:
            out[str(key)] = float("nan")
        else:
            out[str(key)] = float(1.0 - np.mean(sims))
    return out


def _build_repeat_stability_features(
    baseline: pd.DataFrame,
    repeats_long: pd.DataFrame,
    *,
    total_repeat_runs: int,
    board_size: int,
) -> pd.DataFrame:
    if baseline.empty:
        return pd.DataFrame(columns=["pick_key"])
    keys = baseline["pick_key"].astype(str).tolist()
    if repeats_long.empty or total_repeat_runs <= 0:
        out = baseline[["pick_key"]].copy()
        out["stability_accept_count"] = 0.0
        out["stability_accept_rate"] = 0.0
        out["stability_prob_mean"] = np.nan
        out["stability_prob_std"] = np.nan
        out["stability_edge_mean"] = np.nan
        out["stability_edge_std"] = np.nan
        out["stability_rank_mean"] = np.nan
        out["stability_rank_std"] = np.nan
        out["stability_topk_frequency"] = 0.0
        out["stability_cutoff_gap_mean"] = np.nan
        out["stability_cutoff_gap_min"] = np.nan
        out["stability_cooccurrence_instability"] = np.nan
        out["stability_direction_consistency"] = np.nan
        return out

    rep = repeats_long.copy()
    rep["pick_key"] = rep["pick_key"].astype(str)
    rep["board_prob"] = pd.to_numeric(
        rep.get("board_play_win_prob", rep.get("p_calibrated", rep.get("expected_win_rate"))),
        errors="coerce",
    )
    rep["edge_num"] = pd.to_numeric(rep.get("edge"), errors="coerce")
    rep["rank_num"] = pd.to_numeric(rep.get("selected_rank"), errors="coerce")
    rep["topk"] = rep["rank_num"].le(float(max(1, board_size)))
    rep["cutoff_gap"] = (rep["rank_num"] - float(max(1, board_size))).abs()

    agg = (
        rep.groupby("pick_key", dropna=False)
        .agg(
            stability_accept_count=("repeat_index", "nunique"),
            stability_prob_mean=("board_prob", "mean"),
            stability_prob_std=("board_prob", "std"),
            stability_edge_mean=("edge_num", "mean"),
            stability_edge_std=("edge_num", "std"),
            stability_rank_mean=("rank_num", "mean"),
            stability_rank_std=("rank_num", "std"),
            stability_topk_frequency=("topk", "mean"),
            stability_cutoff_gap_mean=("cutoff_gap", "mean"),
            stability_cutoff_gap_min=("cutoff_gap", "min"),
        )
        .reset_index()
    )
    agg["stability_accept_rate"] = agg["stability_accept_count"] / float(max(1, total_repeat_runs))

    cooc = _cooccurrence_instability(
        baseline_keys=keys,
        repeats_long=rep,
        run_col="repeat_index",
        key_col="pick_key",
    )
    agg["stability_cooccurrence_instability"] = agg["pick_key"].map(lambda k: cooc.get(str(k), float("nan")))

    rep_player = rep.get("market_player_raw", rep.get("player", pd.Series("", index=rep.index)))
    rep["player_norm"] = rep_player.map(normalize_player_name)
    rep["target_norm"] = rep.get("target", pd.Series("", index=rep.index)).fillna("").astype(str).str.upper().str.strip()
    rep["direction_norm"] = rep.get("direction", pd.Series("", index=rep.index)).fillna("").astype(str).str.upper().str.strip()
    rep["player_target_key"] = (rep["player_norm"] + "|" + rep["target_norm"]).astype(str)

    dir_share = (
        rep.groupby(["player_target_key", "direction_norm"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    total_share = dir_share.groupby("player_target_key", dropna=False)["rows"].transform("sum")
    dir_share["direction_share"] = dir_share["rows"] / total_share.replace(0, np.nan)
    dir_lookup = {
        (str(r.player_target_key), str(r.direction_norm)): _safe_float(r.direction_share, np.nan)
        for r in dir_share.itertuples(index=False)
    }

    base = baseline[["pick_key"]].copy()
    base_player = baseline.get("market_player_raw", baseline.get("player", pd.Series("", index=baseline.index)))
    base["player_norm"] = base_player.map(normalize_player_name)
    base["target_norm"] = baseline.get("target", pd.Series("", index=baseline.index)).fillna("").astype(str).str.upper().str.strip()
    base["direction_norm"] = baseline.get("direction", pd.Series("", index=baseline.index)).fillna("").astype(str).str.upper().str.strip()
    base["player_target_key"] = (base["player_norm"] + "|" + base["target_norm"]).astype(str)
    base["stability_direction_consistency"] = [
        _safe_float(dir_lookup.get((str(pt), str(dn)), np.nan), np.nan)
        for pt, dn in zip(base["player_target_key"], base["direction_norm"])
    ]

    out = base[["pick_key", "stability_direction_consistency"]].merge(agg, on="pick_key", how="left")
    out["stability_accept_count"] = pd.to_numeric(out["stability_accept_count"], errors="coerce").fillna(0.0)
    out["stability_accept_rate"] = pd.to_numeric(out["stability_accept_rate"], errors="coerce").fillna(0.0)
    out["stability_topk_frequency"] = pd.to_numeric(out["stability_topk_frequency"], errors="coerce").fillna(0.0)
    return out


def cmd_snapshot(args: argparse.Namespace) -> None:
    manifest_path, run_dir, run_stamp = _resolve_manifest(
        run_date=args.run_date,
        run_dir=args.run_dir,
        manifest=args.manifest,
        daily_runs_dir=args.daily_runs_dir,
    )
    manifest_payload = _read_json(manifest_path)
    run_date = str(manifest_payload.get("run_date") or _to_run_date_iso(args.run_date or run_stamp))
    out_root = args.output_root.resolve()
    snap_dir = out_root / "snapshots" / run_stamp
    repeat_dir = snap_dir / "repeats"
    if snap_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Snapshot directory already exists: {snap_dir} (use --overwrite).")
    snap_dir.mkdir(parents=True, exist_ok=True)
    repeat_dir.mkdir(parents=True, exist_ok=True)

    final_csv = Path(str(manifest_payload.get("final_csv", "")))
    selector_csv = Path(str(manifest_payload.get("selector_csv", "")))
    final_json = Path(str(manifest_payload.get("final_json", "")))
    if not final_csv.exists():
        raise FileNotFoundError(f"Baseline final CSV not found: {final_csv}")
    if not selector_csv.exists():
        raise FileNotFoundError(f"Selector CSV not found: {selector_csv}")
    if not final_json.exists():
        raise FileNotFoundError(f"Final JSON not found: {final_json}")

    baseline = pd.read_csv(final_csv).copy()
    baseline["snapshot_run_date"] = run_date
    baseline["snapshot_run_stamp"] = run_stamp
    baseline["pick_key"] = build_pick_key(baseline)
    baseline["accepted_pick_id"] = run_stamp + "|" + baseline["pick_key"].astype(str)
    baseline_out = snap_dir / "accepted_pool_baseline.csv"
    baseline.to_csv(baseline_out, index=False)

    shadow_profile = ""
    shadow_out = snap_dir / "accepted_pool_shadow_frozen.csv"
    shadow_rows = 0
    if not bool(args.skip_shadow):
        shadow = _resolve_shadow_run(manifest_payload, preferred_profile=args.shadow_policy_profile)
        if shadow:
            shadow_profile = str(shadow.get("policy_profile", ""))
            shadow_csv = Path(str(shadow.get("final_csv", "")))
            if shadow_csv.exists():
                shadow_df = pd.read_csv(shadow_csv).copy()
                shadow_df = shadow_df.assign(
                    snapshot_run_date=run_date,
                    snapshot_run_stamp=run_stamp,
                )
                shadow_df["pick_key"] = build_pick_key(shadow_df)
                shadow_df.to_csv(shadow_out, index=False)
                shadow_rows = int(len(shadow_df))

    final_payload = _read_json(final_json)
    policy_payload = final_payload.get("policy", {})
    board_kwargs = _build_board_kwargs_from_policy(policy_payload if isinstance(policy_payload, dict) else {})

    learned_payload = None
    learned_month = ""
    learned_meta = final_payload.get("learned_pool_gate", {})
    if isinstance(learned_meta, dict) and bool(learned_meta.get("enabled", False)):
        learned_payload = _load_optional_payload(learned_meta.get("path"))
        learned_month = str(learned_meta.get("month") or "")

    calibrator_payload = None
    calibrator_month = ""
    calibrator_meta = final_payload.get("selected_board_calibrator", {})
    if isinstance(calibrator_meta, dict) and bool(calibrator_meta.get("enabled", False)):
        calibrator_payload = _load_optional_payload(calibrator_meta.get("path"))
        calibrator_month = str(calibrator_meta.get("calibration_month") or "")

    selector = pd.read_csv(selector_csv)
    seeds = _parse_seed_list(
        raw=args.seed_list,
        repeat_runs=int(args.repeat_runs),
        start=int(args.seed_start),
        step=int(args.seed_step),
    )
    repeat_frames: list[pd.DataFrame] = []
    if not bool(args.skip_repeats):
        for idx, seed in enumerate(seeds):
            repeat_kwargs = dict(board_kwargs)
            repeat_kwargs["thompson_seed"] = int(seed)
            repeat_kwargs["selected_board_calibrator"] = calibrator_payload
            repeat_kwargs["selected_board_calibration_month"] = calibrator_month
            repeat_kwargs["learned_gate_payload"] = learned_payload
            repeat_kwargs["learned_gate_month"] = learned_month
            board = compute_final_board(
                selector,
                **repeat_kwargs,
            )
            board = board.copy()
            board["pick_key"] = build_pick_key(board)
            board["repeat_index"] = int(idx)
            board["repeat_seed"] = int(seed)
            board["snapshot_run_date"] = run_date
            board["snapshot_run_stamp"] = run_stamp
            out_path = repeat_dir / f"final_market_plays_repeat_{idx:03d}_seed_{int(seed)}.csv"
            board.to_csv(out_path, index=False)
            repeat_frames.append(board)

    repeats_long = pd.concat(repeat_frames, ignore_index=True) if repeat_frames else pd.DataFrame()
    repeats_long_out = snap_dir / "repeated_boards_long.csv"
    if not repeats_long.empty:
        repeats_long.to_csv(repeats_long_out, index=False)

    board_size = int(max(1, _safe_int(policy_payload.get("max_total_plays"), len(baseline))))
    stability = _build_repeat_stability_features(
        baseline=baseline,
        repeats_long=repeats_long,
        total_repeat_runs=len(seeds) if not bool(args.skip_repeats) else 0,
        board_size=board_size,
    )
    baseline_stability = baseline.merge(stability, on="pick_key", how="left")
    baseline_stability["snapshot_repeat_runs"] = int(0 if bool(args.skip_repeats) else len(seeds))
    baseline_stability["snapshot_board_size_requested"] = board_size
    baseline_stability["snapshot_selector_rows"] = int(len(selector))
    baseline_stability_out = snap_dir / "accepted_pool_with_stability.csv"
    baseline_stability.to_csv(baseline_stability_out, index=False)

    snapshot_manifest = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "run_date": run_date,
        "run_stamp": run_stamp,
        "source_run_dir": str(run_dir),
        "source_manifest": str(manifest_path),
        "source_final_csv": str(final_csv),
        "source_selector_csv": str(selector_csv),
        "source_final_json": str(final_json),
        "policy_profile": str(manifest_payload.get("policy_profile", "")),
        "baseline_rows": int(len(baseline)),
        "shadow_rows": int(shadow_rows),
        "shadow_policy_profile": shadow_profile,
        "selector_rows": int(len(selector)),
        "repeat_runs_requested": int(len(seeds)),
        "repeat_runs_executed": int(0 if bool(args.skip_repeats) else len(seeds)),
        "repeat_seeds": [int(s) for s in seeds],
        "snapshot_dir": str(snap_dir),
        "accepted_pool_baseline_csv": str(baseline_out),
        "accepted_pool_with_stability_csv": str(baseline_stability_out),
        "accepted_pool_shadow_csv": str(shadow_out) if shadow_rows > 0 else "",
        "repeated_boards_long_csv": str(repeats_long_out) if not repeats_long.empty else "",
        "notes": "Frozen production artifacts only. No automatic live-policy mutation.",
    }
    snapshot_manifest_path = snap_dir / "snapshot_manifest.json"
    _write_json(snapshot_manifest_path, snapshot_manifest)

    print("Accepted-pick gate snapshot complete.")
    print(f"Run date:                {run_date}")
    print(f"Run directory:           {run_dir}")
    print(f"Baseline accepted rows:  {len(baseline)}")
    print(f"Shadow accepted rows:    {shadow_rows}")
    print(f"Repeat runs executed:    {0 if bool(args.skip_repeats) else len(seeds)}")
    print(f"Snapshot manifest:       {snapshot_manifest_path}")


def _resolve_snapshot_manifest_path(
    *,
    snapshot_manifest: Path | None,
    snapshot_dir: Path | None,
    output_root: Path,
    run_date: str | None,
) -> Path:
    if snapshot_manifest is not None:
        path = snapshot_manifest.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found: {path}")
        return path
    if snapshot_dir is not None:
        path = snapshot_dir.resolve() / "snapshot_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found: {path}")
        return path
    if run_date:
        run_stamp = _to_run_stamp(run_date)
        path = output_root.resolve() / "snapshots" / run_stamp / "snapshot_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot manifest not found: {path}")
        return path
    raise ValueError("Provide --snapshot-manifest, --snapshot-dir, or --run-date.")


def _load_actual_lookup():
    cache: dict[str, pd.DataFrame] = {}

    def resolve(csv_path: str, market_date: str, target: str) -> float | None:
        if not isinstance(csv_path, str) or not csv_path:
            return None
        if csv_path not in cache:
            path = Path(csv_path)
            if not path.exists():
                cache[csv_path] = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
            else:
                try:
                    frame = pd.read_csv(path, usecols=["Date", "PTS", "TRB", "AST"])
                except Exception:
                    frame = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
                frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                cache[csv_path] = frame
        frame = cache[csv_path]
        if frame.empty:
            return None
        key = str(target).upper().strip()
        if key not in {"PTS", "TRB", "AST"}:
            return None
        rows = frame.loc[frame["Date"] == str(market_date)]
        if rows.empty:
            return None
        value = pd.to_numeric(rows.iloc[-1][key], errors="coerce")
        if pd.isna(value):
            return None
        return float(value)

    return resolve


def _classify_result(direction: str, line: float | None, actual: float | None, tol: float = 1e-9) -> str:
    if actual is None or line is None:
        return "missing"
    if abs(float(actual) - float(line)) <= tol:
        return "push"
    d = str(direction).upper().strip()
    if d == "OVER":
        return "win" if float(actual) > float(line) else "loss"
    if d == "UNDER":
        return "win" if float(actual) < float(line) else "loss"
    return "missing"


def _dedupe_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    out = history.copy()
    out["pick_key"] = out.get("pick_key", pd.Series("", index=out.index)).astype(str)
    out["accepted_pick_id"] = out.get("accepted_pick_id", pd.Series("", index=out.index)).astype(str)
    out["result_rank"] = out.get("result", pd.Series("", index=out.index)).astype(str).str.lower().map(
        {"win": 4, "loss": 3, "push": 2, "missing": 1}
    ).fillna(0)
    out["settled_at_utc"] = pd.to_datetime(out.get("settled_at_utc"), errors="coerce")
    out["event_date"] = pd.to_datetime(out.get("market_date"), errors="coerce")
    out = out.sort_values(["event_date", "result_rank", "settled_at_utc"], ascending=[True, True, True])
    key_mask = out["accepted_pick_id"].str.len() > 0
    with_id = out.loc[key_mask].drop_duplicates(subset=["accepted_pick_id"], keep="last")
    no_id = out.loc[~key_mask].drop_duplicates(subset=["pick_key"], keep="last")
    merged = pd.concat([with_id, no_id], ignore_index=True)
    merged = merged.drop(columns=["result_rank", "event_date"], errors="ignore")
    merged = merged.sort_values("settled_at_utc", na_position="first").reset_index(drop=True)
    return merged


def cmd_append_settled(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    snapshot_manifest_path = _resolve_snapshot_manifest_path(
        snapshot_manifest=args.snapshot_manifest,
        snapshot_dir=args.snapshot_dir,
        output_root=output_root,
        run_date=args.run_date,
    )
    snapshot_manifest = _read_json(snapshot_manifest_path)
    snapshot_dir = snapshot_manifest_path.parent
    accepted_path = snapshot_manifest.get("accepted_pool_with_stability_csv") or snapshot_manifest.get("accepted_pool_baseline_csv")
    accepted_csv = Path(str(accepted_path))
    if not accepted_csv.exists():
        raise FileNotFoundError(f"Accepted-pool CSV not found: {accepted_csv}")
    accepted = pd.read_csv(accepted_csv).copy()
    if accepted.empty:
        raise RuntimeError(f"Accepted-pool CSV has no rows: {accepted_csv}")

    if "pick_key" not in accepted.columns:
        accepted["pick_key"] = build_pick_key(accepted)
    if "accepted_pick_id" not in accepted.columns:
        run_stamp = str(snapshot_manifest.get("run_stamp", ""))
        accepted["accepted_pick_id"] = run_stamp + "|" + accepted["pick_key"].astype(str)

    lookup_actual = _load_actual_lookup()
    line_vals = pd.to_numeric(accepted.get("market_line"), errors="coerce")
    dates = pd.to_datetime(accepted.get("market_date"), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    targets = accepted.get("target", pd.Series("", index=accepted.index)).fillna("").astype(str).str.upper().str.strip()
    directions = accepted.get("direction", pd.Series("", index=accepted.index)).fillna("").astype(str).str.upper().str.strip()
    csv_paths = accepted.get("csv", pd.Series("", index=accepted.index)).fillna("").astype(str)

    actual_values: list[float] = []
    results: list[str] = []
    utility_units: list[float] = []
    keep_labels: list[float] = []
    for csv_path, market_date, target, direction, line in zip(csv_paths, dates, targets, directions, line_vals):
        line_value = None if pd.isna(line) else float(line)
        actual = lookup_actual(str(csv_path), str(market_date), str(target))
        outcome = _classify_result(str(direction), line_value, actual)
        utility = expected_utility_from_result(outcome)
        keep_label = result_to_keep_label(outcome)
        actual_values.append(float(actual) if actual is not None else float("nan"))
        results.append(str(outcome))
        utility_units.append(float(utility))
        keep_labels.append(float(keep_label))

    accepted["actual_value"] = pd.Series(actual_values, index=accepted.index, dtype="float64")
    accepted["result"] = pd.Series(results, index=accepted.index, dtype="object")
    accepted["utility_units"] = pd.Series(utility_units, index=accepted.index, dtype="float64")
    accepted["keep_label"] = pd.Series(keep_labels, index=accepted.index, dtype="float64")
    accepted["settled_at_utc"] = _utc_now_iso()
    accepted["resolved"] = accepted["result"].isin(["win", "loss"]).astype(bool)

    settled_csv = snapshot_dir / "accepted_pool_settled.csv"
    accepted.to_csv(settled_csv, index=False)

    history_csv = args.history_csv.resolve()
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    if history_csv.exists():
        existing = pd.read_csv(history_csv)
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, accepted], ignore_index=True, sort=False)
    combined = _dedupe_history(combined)
    combined.to_csv(history_csv, index=False)

    resolved_rows = int(accepted["resolved"].sum())
    append_report = {
        "version": 1,
        "updated_at_utc": _utc_now_iso(),
        "snapshot_manifest": str(snapshot_manifest_path),
        "snapshot_settled_csv": str(settled_csv),
        "history_csv": str(history_csv),
        "snapshot_rows": int(len(accepted)),
        "snapshot_resolved_rows": resolved_rows,
        "snapshot_win_rows": int((accepted["result"] == "win").sum()),
        "snapshot_loss_rows": int((accepted["result"] == "loss").sum()),
        "snapshot_push_rows": int((accepted["result"] == "push").sum()),
        "history_total_rows": int(len(combined)),
        "notes": "Outcomes appended for accepted picks only; no live policy mutation.",
    }
    append_report_path = snapshot_dir / "append_settled_report.json"
    _write_json(append_report_path, append_report)

    print("Accepted-pick settlement append complete.")
    print(f"Snapshot manifest:       {snapshot_manifest_path}")
    print(f"Settled snapshot rows:   {len(accepted)}")
    print(f"Resolved rows:           {resolved_rows}")
    print(f"History CSV:             {history_csv}")
    print(f"Append report:           {append_report_path}")


def _resolve_rows_date_col(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    for candidate in ["market_date", "run_date", "event_date"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not resolve date column in rows frame. "
        f"Tried preferred={preferred} and fallback columns."
    )


def _resolve_rows_player_col(frame: pd.DataFrame) -> str:
    if "market_player_raw" in frame.columns:
        return "market_player_raw"
    if "player" in frame.columns:
        return "player"
    frame["player"] = ""
    return "player"


def cmd_prepare_replay_history(args: argparse.Namespace) -> None:
    rows_csv = args.rows_csv.resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")
    rows = pd.read_csv(rows_csv)
    if rows.empty:
        raise RuntimeError(f"Rows CSV has no rows: {rows_csv}")

    if args.mode and "mode" in rows.columns:
        rows = rows.loc[rows["mode"].astype(str).str.lower().str.strip() == str(args.mode).lower().strip()].copy()
    if rows.empty:
        raise RuntimeError("No rows remain after mode/date filters.")

    date_col = _resolve_rows_date_col(rows, preferred=args.date_col)
    result_col = str(args.result_col).strip() or "result"
    if result_col not in rows.columns:
        raise ValueError(f"Result column not found in rows CSV: {result_col}")

    out = rows.copy()
    out["_event_date"] = _coerce_event_datetime(out.get(date_col))
    out = out.loc[out["_event_date"].notna()].copy()
    if out.empty:
        raise RuntimeError("No valid dated rows found after parsing replay rows.")

    if args.start_date:
        out = out.loc[out["_event_date"] >= pd.to_datetime(args.start_date, errors="coerce")].copy()
    if args.end_date:
        out = out.loc[out["_event_date"] <= pd.to_datetime(args.end_date, errors="coerce")].copy()
    if out.empty:
        raise RuntimeError("No rows remain after start/end date filters.")

    out[result_col] = out.get(result_col, pd.Series("missing", index=out.index)).astype(str).str.lower().str.strip()
    out["market_date"] = out["_event_date"].dt.strftime("%Y-%m-%d")
    player_col = _resolve_rows_player_col(out)
    if "market_player_raw" not in out.columns:
        out["market_player_raw"] = out.get(player_col, pd.Series("", index=out.index)).fillna("").astype(str)
    if "target" not in out.columns:
        out["target"] = ""
    if "direction" not in out.columns:
        out["direction"] = ""
    if "market_line" not in out.columns:
        out["market_line"] = np.nan
    out["target"] = out["target"].fillna("").astype(str).str.upper().str.strip()
    out["direction"] = out["direction"].fillna("").astype(str).str.upper().str.strip()
    out["market_line"] = pd.to_numeric(out["market_line"], errors="coerce")

    out["keep_label"] = out[result_col].map({"win": 1.0, "loss": 0.0})
    out["utility_units"] = out[result_col].map(expected_utility_from_result)
    out["resolved"] = out[result_col].isin(["win", "loss"]).astype(bool)
    if "pick_key" not in out.columns:
        out["pick_key"] = build_pick_key(out)

    run_token = out.get("run_date", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    run_token = run_token.where(run_token.str.len() > 0, out["market_date"].astype(str))
    out["accepted_pick_id"] = run_token + "|" + out["pick_key"].astype(str)
    out["history_source"] = "board_objective_replay_rows"
    out["settled_at_utc"] = _utc_now_iso()

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if bool(args.append) and output_csv.exists():
        existing = pd.read_csv(output_csv)
        combined = pd.concat([existing, out], ignore_index=True, sort=False)
    else:
        combined = out.copy()
    combined = _dedupe_history(combined)
    combined.to_csv(output_csv, index=False)

    report = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "rows_csv": str(rows_csv),
        "output_csv": str(output_csv),
        "mode_filter": str(args.mode),
        "date_col": date_col,
        "result_col": result_col,
        "start_date": str(args.start_date or ""),
        "end_date": str(args.end_date or ""),
        "rows_written": int(len(combined)),
        "rows_resolved": int(pd.to_numeric(combined.get("resolved"), errors="coerce").fillna(0).astype(bool).sum()),
        "rows_win": int((combined.get(result_col, pd.Series("", index=combined.index)).astype(str).str.lower().str.strip() == "win").sum()),
        "rows_loss": int((combined.get(result_col, pd.Series("", index=combined.index)).astype(str).str.lower().str.strip() == "loss").sum()),
        "notes": "Replay rows converted to accepted-pick history format for policy-aligned shadow training.",
    }
    report_out = args.report_out.resolve() if args.report_out else output_csv.with_name(output_csv.stem + "_build_report.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_out, report)

    print("Replay-history build complete.")
    print(f"Rows CSV:                {rows_csv}")
    print(f"Output history CSV:      {output_csv}")
    print(f"Rows written:            {len(combined)}")
    print(f"Resolved rows:           {int(pd.to_numeric(combined.get('resolved'), errors='coerce').fillna(0).astype(bool).sum())}")
    print(f"Build report:            {report_out}")


def cmd_paired_eval(args: argparse.Namespace) -> None:
    rows_csv = args.rows_csv.resolve()
    if not rows_csv.exists():
        raise FileNotFoundError(f"Rows CSV not found: {rows_csv}")
    rows = pd.read_csv(rows_csv)
    if rows.empty:
        raise RuntimeError(f"Rows CSV has no rows: {rows_csv}")

    if args.mode and "mode" in rows.columns:
        rows = rows.loc[rows["mode"].astype(str).str.lower().str.strip() == str(args.mode).lower().strip()].copy()
    if rows.empty:
        raise RuntimeError("No rows remain after mode filter.")

    date_col = _resolve_rows_date_col(rows, preferred=args.date_col)
    result_col = str(args.result_col).strip() or "result"
    if result_col not in rows.columns:
        raise ValueError(f"Result column not found in rows CSV: {result_col}")
    player_col = _resolve_rows_player_col(rows)
    target_col = _resolve_target_col(rows)
    direction_col = _resolve_direction_col(rows)

    rows["_event_date"] = _coerce_event_datetime(rows.get(date_col))
    rows = rows.loc[rows["_event_date"].notna()].copy()
    if args.start_date:
        rows = rows.loc[rows["_event_date"] >= pd.to_datetime(args.start_date, errors="coerce")].copy()
    if args.end_date:
        rows = rows.loc[rows["_event_date"] <= pd.to_datetime(args.end_date, errors="coerce")].copy()
    if rows.empty:
        raise RuntimeError("No rows remain after date filters.")

    gate_payload = None
    if args.gate_json:
        gate_path = args.gate_json.resolve()
        if not gate_path.exists():
            raise FileNotFoundError(f"Gate JSON not found: {gate_path}")
        gate_payload = _read_json(gate_path)

    scored, gate_details = apply_accepted_pick_gate(
        rows,
        gate_payload,
        run_date_hint=args.run_date_hint,
        date_col=date_col,
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
        live=False,
        min_rows=int(max(0, args.min_rows)),
    )

    broad_summary = summarize_paired_outcomes(scored, veto_col="accepted_pick_gate_veto", result_col=result_col)
    max_date = _coerce_event_datetime(scored.get("_event_date")).max()
    short_days = int(max(1, args.short_days))
    recent_days = int(max(1, args.recent_days))
    short_start = max_date - pd.Timedelta(days=short_days - 1) if pd.notna(max_date) else pd.NaT
    recent_start = max_date - pd.Timedelta(days=recent_days - 1) if pd.notna(max_date) else pd.NaT
    if pd.notna(short_start):
        short_frame = scored.loc[_coerce_event_datetime(scored.get("_event_date")) >= short_start].copy()
    else:
        short_frame = scored.iloc[0:0].copy()
    if pd.notna(recent_start):
        recent_frame = scored.loc[_coerce_event_datetime(scored.get("_event_date")) >= recent_start].copy()
    else:
        recent_frame = scored.iloc[0:0].copy()

    short_summary = summarize_paired_outcomes(short_frame, veto_col="accepted_pick_gate_veto", result_col=result_col)
    recent_summary = summarize_paired_outcomes(recent_frame, veto_col="accepted_pick_gate_veto", result_col=result_col)

    resolved = scored.loc[scored[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
    affected_share = _safe_float(pd.to_numeric(resolved.get("accepted_pick_gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
    concentration = concentration_stats(
        scored,
        veto_col="accepted_pick_gate_veto",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
    )
    rolling = rolling_window_paired_deltas(
        scored,
        date_col="_event_date",
        veto_col="accepted_pick_gate_veto",
        result_col=result_col,
        window_days=int(args.rolling_window_days),
        step_days=int(args.rolling_step_days),
    )
    criteria = PromotionCriteria(
        min_broad_profit_delta_units=float(args.min_broad_profit_delta_units),
        min_recent_profit_delta_units=float(args.min_recent_profit_delta_units),
        min_recent_hit_rate_delta_pp=float(args.min_recent_hit_rate_delta_pp),
        min_broad_hit_rate_delta_pp=float(args.min_broad_hit_rate_delta_pp),
        min_coverage_retention=float(args.min_coverage_retention),
        min_affected_share=float(args.min_affected_share),
        max_top_removed_player_share=float(args.max_top_removed_player_share),
        max_top_removed_segment_share=float(args.max_top_removed_segment_share),
        max_top_removed_target_share=float(args.max_top_removed_target_share),
        concentration_min_veto_rows=int(args.concentration_min_veto_rows),
        conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
        conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
        conditional_target_player_share_max=float(args.conditional_target_player_share_max),
        conditional_target_require_recent_non_negative=bool(
            not args.disable_conditional_target_require_recent_non_negative
        ),
        conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
        conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
        min_rolling_pass_rate=float(args.min_rolling_pass_rate),
        max_observed_fire_rate=float(args.max_observed_fire_rate),
        rolling_profit_delta_floor=float(args.rolling_profit_delta_floor),
        rolling_hit_rate_delta_floor_pp=float(args.rolling_hit_rate_delta_floor_pp),
    )
    recommendation = build_promotion_recommendation(
        broad_delta=broad_summary.get("delta", {}),
        recent_delta=recent_summary.get("delta", {}),
        affected_share=float(affected_share),
        concentration=concentration,
        rolling=rolling,
        observed_fire_rate=float(affected_share),
        criteria=criteria,
    )
    lopo = _leave_one_player_out_validation(
        scored,
        player_col=player_col,
        veto_col="accepted_pick_gate_veto",
        result_col=result_col,
        min_player_rows=int(args.lopo_min_player_rows),
        profit_floor=float(args.lopo_profit_floor),
        hit_floor_pp=float(args.lopo_hit_floor_pp),
    )
    bootstrap = _bootstrap_paired_deltas(
        scored,
        veto_col="accepted_pick_gate_veto",
        result_col=result_col,
        iterations=int(args.bootstrap_iterations),
        alpha=float(args.bootstrap_alpha),
        seed=int(args.bootstrap_seed),
    )
    recommendation = _apply_small_lift_validation_to_recommendation(
        recommendation,
        lopo=lopo,
        bootstrap=bootstrap,
        enforce=bool(not args.disable_small_lift_validation),
        min_lopo_profit_pass_rate=float(args.min_lopo_profit_pass_rate),
        min_lopo_hit_pass_rate=float(args.min_lopo_hit_pass_rate),
        min_bootstrap_profit_ci_lower=float(args.min_bootstrap_profit_ci_lower),
        min_bootstrap_hit_ci_lower_pp=float(args.min_bootstrap_hit_ci_lower_pp),
    )

    out_payload = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "rows_csv": str(rows_csv),
        "gate_json": str(args.gate_json.resolve()) if args.gate_json else "",
        "mode_filter": str(args.mode),
        "date_col": date_col,
        "result_col": result_col,
        "window": {
            "start_date": str(args.start_date or ""),
            "end_date": str(args.end_date or ""),
            "max_date": str(max_date.date()) if pd.notna(max_date) else "",
            "short_days": int(short_days),
            "recent_days": int(recent_days),
            "short_start": str(short_start.date()) if pd.notna(short_start) else "",
            "recent_start": str(recent_start.date()) if pd.notna(recent_start) else "",
        },
        "rows_total": int(len(scored)),
        "rows_resolved": int(len(resolved)),
        "gate_details": gate_details,
        "broad_summary": broad_summary,
        "short_summary": short_summary,
        "recent_summary": recent_summary,
        "affected_share": float(affected_share),
        "concentration": concentration,
        "rolling_window": {
            "days": int(args.rolling_window_days),
            "step_days": int(args.rolling_step_days),
            "rows": int(len(rolling)),
        },
        "promotion_criteria": asdict(criteria),
        "promotion_recommendation": recommendation,
    }

    summary_out = args.summary_out.resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary_out, out_payload)

    if args.scored_rows_out:
        scored_out = args.scored_rows_out.resolve()
        scored_out.parent.mkdir(parents=True, exist_ok=True)
        export = scored.copy()
        export["event_date"] = pd.to_datetime(export["_event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        export.to_csv(scored_out, index=False)

    print("Deterministic paired evaluation complete.")
    print(f"Rows CSV:                {rows_csv}")
    print(f"Summary JSON:            {summary_out}")
    print(f"Broad delta hit-rate:    {_safe_float(broad_summary.get('delta', {}).get('hit_rate_pp'), np.nan):.4f} pp")
    print(f"Broad delta profit:      {_safe_float(broad_summary.get('delta', {}).get('profit_units'), np.nan):.4f} units")
    print(f"Recent delta hit-rate:   {_safe_float(recent_summary.get('delta', {}).get('hit_rate_pp'), np.nan):.4f} pp")
    print(f"Recent delta profit:     {_safe_float(recent_summary.get('delta', {}).get('profit_units'), np.nan):.4f} units")
    print(f"Promotion pass:          {bool(recommendation.get('pass', False))}")


def _resolve_history_date_col(frame: pd.DataFrame, preferred: str) -> str:
    if preferred in frame.columns:
        return preferred
    for candidate in ["market_date", "snapshot_run_date", "run_date", "event_date"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not resolve date column in history. "
        f"Tried preferred={preferred} and fallback columns."
    )


def _resolve_player_col(frame: pd.DataFrame) -> str:
    if "market_player_raw" in frame.columns:
        return "market_player_raw"
    if "player" in frame.columns:
        return "player"
    frame["player"] = ""
    return "player"


def _resolve_target_col(frame: pd.DataFrame) -> str:
    if "target" in frame.columns:
        return "target"
    frame["target"] = ""
    return "target"


def _resolve_direction_col(frame: pd.DataFrame) -> str:
    if "direction" in frame.columns:
        return "direction"
    frame["direction"] = ""
    return "direction"


def _resolve_result_col(frame: pd.DataFrame) -> str:
    if "result" in frame.columns:
        return "result"
    frame["result"] = "missing"
    return "result"


def _prepare_history_frame(
    history: pd.DataFrame,
    *,
    date_col: str,
    label_col: str,
    result_col: str,
) -> pd.DataFrame:
    out = history.copy()
    out["_event_date"] = _coerce_event_datetime(out.get(date_col))
    out = out.loc[out["_event_date"].notna()].copy()
    out[result_col] = out.get(result_col, pd.Series("missing", index=out.index)).astype(str).str.lower().str.strip()
    if label_col not in out.columns:
        out[label_col] = out[result_col].map({"win": 1.0, "loss": 0.0})
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    if "utility_units" not in out.columns:
        out["utility_units"] = out[result_col].map(expected_utility_from_result)
    out["utility_units"] = pd.to_numeric(out["utility_units"], errors="coerce")
    if "selected_rank" not in out.columns:
        out["selected_rank"] = np.nan
    out["selected_rank"] = pd.to_numeric(out["selected_rank"], errors="coerce").fillna(999.0)
    if "pick_key" not in out.columns:
        out["pick_key"] = build_pick_key(out)
    return out


def _feature_column_candidates(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_seed = [
        "selected_rank",
        "market_line",
        "edge",
        "abs_edge",
        "gap_percentile",
        "raw_gap_percentile",
        "expected_win_rate",
        "selector_expected_win_rate",
        "p_selector",
        "p_calibrated",
        "board_play_win_prob",
        "selected_board_prob_raw",
        "ev",
        "ev_adjusted",
        "final_confidence",
        "confidence_score",
        "market_books",
        "market_books_norm",
        "history_rows",
        "recency_factor",
        "belief_uncertainty",
        "belief_uncertainty_normalized",
        "belief_confidence_factor",
        "posterior_alpha",
        "posterior_beta",
        "posterior_variance",
        "calibration_blend_weight",
        "board_play_strength",
        "board_uncertainty_penalty",
        "board_prob_dispersion",
        "board_dependency_burden",
        "board_shadow_disagreement",
        "board_segment_recent_weakness",
        "board_instability_score",
        "board_instability_penalty",
        "snapshot_repeat_runs",
        "snapshot_board_size_requested",
        "snapshot_selector_rows",
        "stability_accept_count",
        "stability_accept_rate",
        "stability_prob_mean",
        "stability_prob_std",
        "stability_edge_mean",
        "stability_edge_std",
        "stability_rank_mean",
        "stability_rank_std",
        "stability_topk_frequency",
        "stability_cutoff_gap_mean",
        "stability_cutoff_gap_min",
        "stability_cooccurrence_instability",
        "stability_direction_consistency",
    ]
    for column in frame.columns:
        if str(column).startswith("stability_") and str(column) not in numeric_seed:
            numeric_seed.append(str(column))

    categorical_seed = [
        "target",
        "direction",
        "recommendation",
        "decision_tier",
        "calibration_source",
        "selected_board_calibration_source",
        "learned_gate_source",
        "learned_gate_month",
        "weak_bucket",
        "board_segment_recent_weakness_source",
    ]
    numeric_features = [col for col in numeric_seed if col in frame.columns]
    categorical_features = [col for col in categorical_seed if col in frame.columns]
    return numeric_features, categorical_features


def _build_gate_model_configs(
    args: argparse.Namespace,
) -> tuple[str, LogisticGateConfig | None, CatBoostGateConfig | None, UpliftCatBoostGateConfig | None]:
    model_family = str(getattr(args, "model_family", "logistic")).strip().lower()
    if model_family not in {"logistic", "catboost", "uplift_catboost"}:
        raise ValueError(f"Unsupported --model-family: {model_family}")
    logistic_cfg: LogisticGateConfig | None = None
    catboost_cfg: CatBoostGateConfig | None = None
    uplift_cfg: UpliftCatBoostGateConfig | None = None
    if model_family == "catboost":
        catboost_cfg = CatBoostGateConfig(
            iterations=int(getattr(args, "catboost_iterations", 320)),
            depth=int(getattr(args, "catboost_depth", 5)),
            learning_rate=float(getattr(args, "catboost_learning_rate", 0.05)),
            l2_leaf_reg=float(getattr(args, "catboost_l2_leaf_reg", 8.0)),
            random_strength=float(getattr(args, "catboost_random_strength", 1.0)),
            bagging_temperature=float(getattr(args, "catboost_bagging_temperature", 0.0)),
            min_data_in_leaf=int(getattr(args, "catboost_min_data_in_leaf", 20)),
            random_seed=int(getattr(args, "catboost_seed", 42)),
            class_weight_positive=float(getattr(args, "class_weight_positive", 1.0)),
            class_weight_negative=float(getattr(args, "class_weight_negative", 1.0)),
            segment_weight_power=float(getattr(args, "catboost_segment_weight_power", 0.0)),
            segment_weight_cap=float(getattr(args, "catboost_segment_weight_cap", 3.0)),
        )
    elif model_family == "uplift_catboost":
        uplift_cfg = UpliftCatBoostGateConfig(
            iterations=int(getattr(args, "catboost_iterations", 320)),
            depth=int(getattr(args, "catboost_depth", 5)),
            learning_rate=float(getattr(args, "catboost_learning_rate", 0.05)),
            l2_leaf_reg=float(getattr(args, "catboost_l2_leaf_reg", 8.0)),
            random_strength=float(getattr(args, "catboost_random_strength", 1.0)),
            bagging_temperature=float(getattr(args, "catboost_bagging_temperature", 0.0)),
            min_data_in_leaf=int(getattr(args, "catboost_min_data_in_leaf", 20)),
            random_seed=int(getattr(args, "catboost_seed", 42)),
            class_weight_positive=float(getattr(args, "class_weight_positive", 1.0)),
            class_weight_negative=float(getattr(args, "class_weight_negative", 1.0)),
            segment_weight_power=float(getattr(args, "catboost_segment_weight_power", 0.0)),
            segment_weight_cap=float(getattr(args, "catboost_segment_weight_cap", 3.0)),
            keep_prob_temperature=float(getattr(args, "uplift_keep_prob_temperature", 0.25)),
            harm_head_enabled=bool(not getattr(args, "disable_uplift_harm_head", False)),
            harm_positive_weight=float(getattr(args, "uplift_harm_positive_weight", 1.5)),
            harm_temperature=float(getattr(args, "uplift_harm_temperature", 1.0)),
        )
    else:
        logistic_cfg = LogisticGateConfig(
            learning_rate=float(getattr(args, "learning_rate", 0.05)),
            l2_strength=float(getattr(args, "l2_strength", 2.5)),
            max_iter=int(getattr(args, "max_iter", 3500)),
            tolerance=float(getattr(args, "tolerance", 1e-7)),
            class_weight_positive=float(getattr(args, "class_weight_positive", 1.0)),
            class_weight_negative=float(getattr(args, "class_weight_negative", 1.0)),
        )
    return model_family, logistic_cfg, catboost_cfg, uplift_cfg


def _new_gate_model(
    model_family: str,
    *,
    logistic_cfg: LogisticGateConfig | None,
    catboost_cfg: CatBoostGateConfig | None,
    uplift_cfg: UpliftCatBoostGateConfig | None,
) -> Any:
    family = str(model_family).strip().lower()
    if family == "catboost":
        if catboost_cfg is None:
            raise RuntimeError("catboost model family selected but no CatBoost config was provided.")
        return CatBoostKeepDropGate(config=catboost_cfg)
    if family == "uplift_catboost":
        if uplift_cfg is None:
            raise RuntimeError("uplift_catboost model family selected but no uplift config was provided.")
        return UpliftCatBoostGate(config=uplift_cfg)
    if logistic_cfg is None:
        raise RuntimeError("logistic model family selected but no Logistic config was provided.")
    return RegularizedLogisticGate(config=logistic_cfg)


def _feature_signature_hash(
    *,
    model_family: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> str:
    payload = {
        "model_family": str(model_family).strip().lower(),
        "numeric_features": sorted(str(v) for v in numeric_features),
        "categorical_features": sorted(str(v) for v in categorical_features),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _segment_paired_deltas(
    frame: pd.DataFrame,
    *,
    target_col: str,
    direction_col: str,
    result_col: str,
    veto_col: str = "gate_veto",
    min_resolved_rows: int = 10,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    target = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    direction = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    segment = (target + "|" + direction).where(target.str.len().gt(0) & direction.str.len().gt(0), "")
    for seg_name, idx in segment.groupby(segment).groups.items():
        seg = str(seg_name).strip().upper()
        if not seg:
            continue
        part = frame.loc[idx].copy()
        summary = summarize_paired_outcomes(part, veto_col=veto_col, result_col=result_col)
        baseline = summary.get("baseline", {})
        delta = summary.get("delta", {})
        resolved_rows = int(_safe_int(baseline.get("resolved"), 0))
        if resolved_rows < int(max(1, min_resolved_rows)):
            continue
        out.append(
            {
                "segment": seg,
                "resolved_rows": resolved_rows,
                "rows": int(_safe_int(summary.get("rows"), 0)),
                "baseline_hit_rate": _safe_float(baseline.get("hit_rate"), float("nan")),
                "gated_hit_rate": _safe_float(summary.get("gated", {}).get("hit_rate"), float("nan")),
                "delta_hit_rate_pp": _safe_float(delta.get("hit_rate_pp"), float("nan")),
                "delta_profit_units": _safe_float(delta.get("profit_units"), float("nan")),
                "delta_ev_per_resolved": _safe_float(delta.get("ev_per_resolved"), float("nan")),
                "delta_resolved": int(_safe_int(delta.get("resolved"), 0)),
            }
        )
    out.sort(
        key=lambda row: (
            _safe_float(row.get("delta_profit_units"), float("-inf")),
            _safe_float(row.get("delta_hit_rate_pp"), float("-inf")),
            int(row.get("resolved_rows", 0)),
        ),
        reverse=True,
    )
    return out


def _parse_quantiles(raw: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(np.clip(float(token), 0.0, 1.0)))
    if not values:
        return [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    return sorted(set(values))


def _parse_float_grid(raw: str, *, default: list[float], clip_min: float | None = None, clip_max: float | None = None) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except Exception:
            continue
        if clip_min is not None:
            value = max(float(clip_min), value)
        if clip_max is not None:
            value = min(float(clip_max), value)
        values.append(float(value))
    if not values:
        values = [float(v) for v in default]
    return sorted(set(float(v) for v in values))


def _parse_int_grid(raw: str, *, default: list[int], min_value: int = 0) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        values.append(int(max(int(min_value), value)))
    if not values:
        values = [int(max(int(min_value), v)) for v in default]
    return sorted(set(int(v) for v in values))


def _build_threshold_grid(
    frame: pd.DataFrame,
    keep_prob_col: str,
    quantiles: list[float],
    *,
    max_candidates: int = 48,
) -> list[float]:
    probs = pd.to_numeric(frame.get(keep_prob_col), errors="coerce").dropna().clip(lower=0.0, upper=1.0)
    if probs.empty:
        return [-1.0, 0.50]
    base_quantiles = sorted(set(float(np.clip(q, 0.0, 1.0)) for q in quantiles))
    default_ladder = [0.50, 0.60, 0.70, 0.80, 0.90]
    ladder = sorted(set(base_quantiles + default_ladder))
    out = sorted(set(float(probs.quantile(float(q))) for q in ladder))
    # Add micro-threshold candidates so we can evaluate tiny (1-3 row) veto behavior.
    # Quantile-only grids can skip these and force over-/under-pruning.
    unique = sorted(set(float(v) for v in probs.tolist()))
    if unique:
        slices: list[list[float]] = []
        slices.append(unique[: min(len(unique), 12)])
        slices.append(unique[max(0, len(unique) - 12) :])
        for edge in slices:
            if not edge:
                continue
            if len(edge) == 1:
                out.append(float(np.clip(edge[0] + 1e-9, 0.0, 1.0)))
            else:
                for i in range(len(edge) - 1):
                    lo = float(edge[i])
                    hi = float(edge[i + 1])
                    mid = float((lo + hi) / 2.0)
                    if mid > lo:
                        out.append(float(np.clip(mid, 0.0, 1.0)))
                out.append(float(np.clip(edge[0] + 1e-9, 0.0, 1.0)))
                out.append(float(np.clip(edge[-1] - 1e-9, 0.0, 1.0)))
    out = sorted(set(out + [-1.0]))
    max_candidates = int(max(8, max_candidates))
    if len(out) > max_candidates:
        core = sorted(v for v in out if float(v) > -0.999999)
        if core:
            q = np.linspace(0.0, 1.0, num=max(2, max_candidates - 1))
            sampled = sorted(set(float(np.quantile(core, float(qq))) for qq in q))
            out = sorted(set([-1.0] + sampled))
        else:
            out = [-1.0]
    if not out:
        return [-1.0, 0.50]
    return out


def _segment_subsets_for_search(
    frame: pd.DataFrame,
    *,
    target_col: str,
    direction_col: str,
    result_col: str,
    max_size: int,
    max_subsets: int,
    min_segment_rows: int,
    segment_seed_limit: int,
) -> list[tuple[str, ...]]:
    target = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    direction = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    seg = (target + "|" + direction).astype("object")
    result = frame.get(result_col, pd.Series("", index=frame.index)).astype(str).str.lower().str.strip()
    win = result.eq("win").astype(float)
    loss = result.eq("loss").astype(float)
    resolved_mask = (win + loss) > 0
    baseline_hit = float(win.loc[resolved_mask].mean()) if bool(resolved_mask.any()) else float("nan")

    per_segment_rows: list[dict[str, Any]] = []
    if bool(resolved_mask.any()):
        tmp = pd.DataFrame(
            {
                "_seg": seg,
                "_win": win,
                "_resolved": resolved_mask.astype(int),
            }
        )
        grouped = (
            tmp.loc[tmp["_resolved"] > 0]
            .groupby("_seg", dropna=False)
            .agg(rows=("_win", "size"), hit_rate=("_win", "mean"))
            .reset_index()
        )
        min_rows = int(max(1, min_segment_rows))
        grouped = grouped.loc[grouped["rows"] >= min_rows].copy()
        for _, row in grouped.iterrows():
            segment = str(row.get("_seg", "")).strip().upper()
            if "|" not in segment:
                continue
            rows = int(row.get("rows", 0))
            hit = _safe_float(row.get("hit_rate"), float("nan"))
            delta_pp = float((hit - baseline_hit) * 100.0) if (hit == hit and baseline_hit == baseline_hit) else 0.0
            weakness = max(0.0, -delta_pp)
            score = float(weakness * np.log1p(max(1, rows)))
            per_segment_rows.append(
                {
                    "segment": segment,
                    "rows": rows,
                    "hit_rate": hit,
                    "delta_pp": delta_pp,
                    "score": score,
                }
            )

    if not per_segment_rows:
        segments = sorted(seg.loc[seg.str.len() > 1].unique().tolist())
        per_segment_rows = [{"segment": str(s), "rows": 0, "hit_rate": float("nan"), "delta_pp": 0.0, "score": 0.0} for s in segments]

    per_segment_rows = sorted(
        per_segment_rows,
        key=lambda row: (
            float(row.get("score", 0.0)),
            -float(row.get("delta_pp", 0.0)),
            int(row.get("rows", 0)),
        ),
        reverse=True,
    )
    segment_seed_limit = int(max(1, segment_seed_limit))
    ranked_segments = [str(row.get("segment", "")) for row in per_segment_rows if str(row.get("segment", ""))]
    segments = ranked_segments[:segment_seed_limit]
    if not segments:
        segments = ranked_segments
    if not segments:
        return [tuple()]
    max_size = int(max(0, max_size))
    max_subsets = int(max(1, max_subsets))
    subsets: list[tuple[str, ...]] = [tuple()]  # empty tuple means no segment filter
    if max_size <= 0:
        return subsets
    segment_score = {str(row.get("segment", "")): float(row.get("score", 0.0)) for row in per_segment_rows}
    ranked_combos: list[tuple[float, tuple[str, ...]]] = []
    for size in range(1, min(len(segments), max_size) + 1):
        for combo in itertools.combinations(segments, size):
            combo_tuple = tuple(sorted(str(v) for v in combo))
            combo_score = float(sum(float(segment_score.get(seg_name, 0.0)) for seg_name in combo_tuple))
            # Favor broader packs when raw weakness score is similar.
            combo_score += 0.10 * float(size - 1)
            ranked_combos.append((combo_score, combo_tuple))
    ranked_combos.sort(key=lambda item: (float(item[0]), len(item[1])), reverse=True)
    for _, combo in ranked_combos:
        subsets.append(combo)
        if len(subsets) >= max_subsets:
            break
    return subsets


def _meta_cohort_packs_for_search(
    frame: pd.DataFrame,
    *,
    target_col: str,
    direction_col: str,
    result_col: str,
    max_size: int,
    max_subsets: int,
    min_cohort_rows: int,
    seed_limit: int,
    min_negative_delta_pp: float,
    enabled: bool,
) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = [tuple()]
    if not bool(enabled):
        return subsets
    if frame.empty:
        return subsets

    result = frame.get(result_col, pd.Series("", index=frame.index)).astype(str).str.lower().str.strip()
    win = result.eq("win").astype(float)
    loss = result.eq("loss").astype(float)
    resolved_mask = (win + loss) > 0
    if not bool(resolved_mask.any()):
        return subsets
    baseline_hit = _safe_float(win.loc[resolved_mask].mean(), float("nan"))
    if baseline_hit != baseline_hit:
        return subsets

    cohort_cols = build_meta_cohort_columns(frame, target_col=target_col, direction_col=direction_col)
    if cohort_cols.empty:
        return subsets

    min_rows = int(max(1, min_cohort_rows))
    min_neg_delta = float(min_negative_delta_pp)
    rows: list[dict[str, Any]] = []
    for cohort_col in cohort_cols.columns:
        labels = cohort_cols.get(cohort_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
        tmp = pd.DataFrame(
            {
                "_label": labels,
                "_win": win,
                "_resolved": resolved_mask.astype(int),
            }
        )
        grouped = (
            tmp.loc[(tmp["_resolved"] > 0) & tmp["_label"].str.len().gt(0)]
            .groupby("_label", dropna=False)
            .agg(rows=("_win", "size"), hit_rate=("_win", "mean"))
            .reset_index()
        )
        if grouped.empty:
            continue
        for _, row in grouped.iterrows():
            label = str(row.get("_label", "")).strip().upper()
            if not label:
                continue
            cohort_rows = int(row.get("rows", 0))
            if cohort_rows < min_rows:
                continue
            hit = _safe_float(row.get("hit_rate"), float("nan"))
            if hit != hit:
                continue
            delta_pp = float((hit - baseline_hit) * 100.0)
            if delta_pp > min_neg_delta:
                continue
            score = float(max(0.0, -delta_pp) * np.log1p(max(1, cohort_rows)))
            rows.append(
                {
                    "label": label,
                    "rows": cohort_rows,
                    "hit_rate": hit,
                    "delta_pp": delta_pp,
                    "score": score,
                }
            )

    if not rows:
        return subsets

    rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("score", 0.0)),
            -float(row.get("delta_pp", 0.0)),
            int(row.get("rows", 0)),
        ),
        reverse=True,
    )
    seed_limit = int(max(1, seed_limit))
    seeds = [str(row.get("label", "")) for row in rows[:seed_limit] if str(row.get("label", ""))]
    if not seeds:
        return subsets

    max_size = int(max(0, max_size))
    max_subsets = int(max(1, max_subsets))
    if max_size <= 0:
        return subsets

    label_score = {str(row.get("label", "")): float(row.get("score", 0.0)) for row in rows}
    ranked_combos: list[tuple[float, tuple[str, ...]]] = []
    for size in range(1, min(len(seeds), max_size) + 1):
        for combo in itertools.combinations(seeds, size):
            combo_tuple = tuple(sorted(str(v) for v in combo))
            combo_score = float(sum(float(label_score.get(item, 0.0)) for item in combo_tuple))
            combo_score += 0.10 * float(size - 1)
            ranked_combos.append((combo_score, combo_tuple))
    ranked_combos.sort(key=lambda item: (float(item[0]), len(item[1])), reverse=True)
    for _, combo in ranked_combos:
        subsets.append(combo)
        if len(subsets) >= max_subsets:
            break
    return subsets


def _policy_variants_for_search(policy: GatePolicyConfig) -> list[GatePolicyConfig]:
    variants: list[GatePolicyConfig] = []

    def _emit(gap: float, tail_slots: int, require_keep_prob_filter: bool) -> None:
        variants.append(
            GatePolicyConfig(
                max_fire_rate=float(policy.max_fire_rate),
                min_coverage_rate=float(policy.min_coverage_rate),
                max_removed_per_day=int(policy.max_removed_per_day),
                max_removed_per_player_per_day=int(policy.max_removed_per_player_per_day),
                max_removed_per_segment_per_day=int(policy.max_removed_per_segment_per_day),
                max_removed_per_target_per_day=int(policy.max_removed_per_target_per_day),
                max_global_removed_per_player=int(policy.max_global_removed_per_player),
                max_global_removed_share_per_player=float(policy.max_global_removed_share_per_player),
                global_player_cap_min_veto_rows=int(policy.global_player_cap_min_veto_rows),
                relax_player_global_cap_for_budget=bool(policy.relax_player_global_cap_for_budget),
                player_penalty_alpha=float(policy.player_penalty_alpha),
                player_penalty_power=float(policy.player_penalty_power),
                optimizer_enabled=bool(policy.optimizer_enabled),
                optimizer_beam_width=int(policy.optimizer_beam_width),
                optimizer_max_candidates_per_day=int(policy.optimizer_max_candidates_per_day),
                optimizer_keep_prob_weight=float(policy.optimizer_keep_prob_weight),
                optimizer_gap_weight=float(policy.optimizer_gap_weight),
                optimizer_harm_weight=float(policy.optimizer_harm_weight),
                optimizer_rank_weight=float(policy.optimizer_rank_weight),
                segment_threshold_overrides=dict(policy.segment_threshold_overrides or {}),
                segment_budget_overrides=dict(policy.segment_budget_overrides or {}),
                tail_slots_only=int(max(0, tail_slots)),
                min_veto_gap=float(max(0.0, gap)),
                require_keep_prob_filter=bool(require_keep_prob_filter),
                allowed_segments=tuple(policy.allowed_segments or ()),
                allowed_meta_cohorts=tuple(policy.allowed_meta_cohorts or ()),
            )
        )

    _emit(float(policy.min_veto_gap), int(policy.tail_slots_only), True)
    if float(policy.min_veto_gap) > 1e-9:
        _emit(0.0, int(policy.tail_slots_only), True)
    if int(policy.tail_slots_only) > 0:
        _emit(float(policy.min_veto_gap), 0, True)
    if float(policy.min_veto_gap) > 1e-9 and int(policy.tail_slots_only) > 0:
        _emit(0.0, 0, True)
    # Cohort-only gating variant: allow strong cohort rules even when keep_prob is not below threshold.
    _emit(0.0, 0, False)

    dedup: list[GatePolicyConfig] = []
    seen: set[tuple[float, int, int]] = set()
    for item in variants:
        key = (
            round(float(item.min_veto_gap), 6),
            int(item.tail_slots_only),
            1 if bool(item.require_keep_prob_filter) else 0,
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def _search_train_threshold(
    train_frame: pd.DataFrame,
    *,
    keep_prob_col: str,
    date_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
    result_col: str,
    policy: GatePolicyConfig,
    quantiles: list[float],
    threshold_max_candidates: int,
    segment_search_max_size: int,
    segment_search_max_subsets: int,
    segment_search_min_rows: int,
    segment_search_seed_limit: int,
    cohort_search_enabled: bool,
    cohort_search_min_delta_pp: float,
    min_affected_share: float,
    max_top_removed_player_share: float,
    max_top_removed_segment_share: float,
    max_top_removed_target_share: float,
    concentration_min_veto_rows: int,
    min_coverage_retention: float,
    prefer_non_noop_when_close: bool,
    non_noop_objective_margin: float,
    objective_player_top_share_penalty: float,
    objective_player_hhi_penalty: float,
    objective_player_diversity_bonus: float,
    conditional_target_cap_enabled: bool,
    conditional_target_relaxed_cap: float,
    conditional_target_player_share_max: float,
    conditional_target_require_recent_non_negative: bool,
    conditional_target_recent_hit_floor_pp: float,
    conditional_target_recent_profit_floor: float,
) -> tuple[float, dict[str, Any]]:
    if train_frame.empty:
        return -1.0, {"threshold": -1.0, "reason": "empty_train"}
    candidates = _build_threshold_grid(
        train_frame,
        keep_prob_col=keep_prob_col,
        quantiles=quantiles,
        max_candidates=int(threshold_max_candidates),
    )
    segment_subsets = _segment_subsets_for_search(
        train_frame,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        max_size=int(segment_search_max_size),
        max_subsets=int(segment_search_max_subsets),
        min_segment_rows=int(segment_search_min_rows),
        segment_seed_limit=int(segment_search_seed_limit),
    )
    cohort_packs = _meta_cohort_packs_for_search(
        train_frame,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        max_size=int(segment_search_max_size),
        max_subsets=int(segment_search_max_subsets),
        min_cohort_rows=int(segment_search_min_rows),
        seed_limit=int(segment_search_seed_limit),
        min_negative_delta_pp=float(cohort_search_min_delta_pp),
        enabled=bool(cohort_search_enabled),
    )
    if len(cohort_packs) > 1:
        # Cohort packs already encode segment context; keep segment search neutral
        # to avoid a combinatorial explosion and over-constraining.
        segment_subsets = [tuple()]
    best_threshold = -1.0
    best_payload: dict[str, Any] | None = None
    best_relaxed: dict[str, Any] | None = None
    best_non_noop: dict[str, Any] | None = None
    best_relaxed_structural: dict[str, Any] | None = None
    best_non_noop_structural: dict[str, Any] | None = None
    policy_variants = _policy_variants_for_search(policy)
    for threshold in candidates:
        for seg_subset in segment_subsets:
            for cohort_pack in cohort_packs:
                for variant in policy_variants:
                    if not bool(variant.require_keep_prob_filter) and not cohort_pack:
                        continue
                    cand_policy = GatePolicyConfig(
                        max_fire_rate=float(variant.max_fire_rate),
                        min_coverage_rate=float(variant.min_coverage_rate),
                        max_removed_per_day=int(variant.max_removed_per_day),
                        max_removed_per_player_per_day=int(variant.max_removed_per_player_per_day),
                        max_removed_per_segment_per_day=int(variant.max_removed_per_segment_per_day),
                        max_removed_per_target_per_day=int(variant.max_removed_per_target_per_day),
                        max_global_removed_per_player=int(variant.max_global_removed_per_player),
                        max_global_removed_share_per_player=float(variant.max_global_removed_share_per_player),
                        global_player_cap_min_veto_rows=int(variant.global_player_cap_min_veto_rows),
                        relax_player_global_cap_for_budget=bool(variant.relax_player_global_cap_for_budget),
                        player_penalty_alpha=float(variant.player_penalty_alpha),
                        player_penalty_power=float(variant.player_penalty_power),
                        optimizer_enabled=bool(variant.optimizer_enabled),
                        optimizer_beam_width=int(variant.optimizer_beam_width),
                        optimizer_max_candidates_per_day=int(variant.optimizer_max_candidates_per_day),
                        optimizer_keep_prob_weight=float(variant.optimizer_keep_prob_weight),
                        optimizer_gap_weight=float(variant.optimizer_gap_weight),
                        optimizer_harm_weight=float(variant.optimizer_harm_weight),
                        optimizer_rank_weight=float(variant.optimizer_rank_weight),
                        segment_threshold_overrides=dict(variant.segment_threshold_overrides or {}),
                        segment_budget_overrides=dict(variant.segment_budget_overrides or {}),
                        tail_slots_only=int(variant.tail_slots_only),
                        min_veto_gap=float(variant.min_veto_gap),
                        require_keep_prob_filter=bool(variant.require_keep_prob_filter),
                        allowed_segments=tuple(seg_subset),
                        allowed_meta_cohorts=tuple(cohort_pack),
                    )
                    scored = apply_shadow_gate_policy(
                        train_frame,
                        keep_prob_col=keep_prob_col,
                        date_col=date_col,
                        player_col=player_col,
                        target_col=target_col,
                        direction_col=direction_col,
                        threshold=float(threshold),
                        policy=cand_policy,
                    )
                    summary = summarize_paired_outcomes(scored, veto_col="gate_veto", result_col=result_col)
                    delta = summary.get("delta", {})
                    broad_profit = _safe_float(delta.get("profit_units"), float("-inf"))
                    broad_hit = _safe_float(delta.get("hit_rate_pp"), float("-inf"))
                    fire_rate = _safe_float(scored.get("gate_veto", pd.Series(dtype="float64")).mean(), 0.0)
                    resolved = scored.loc[scored[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
                    affected_share = _safe_float(pd.to_numeric(resolved.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
                    base_resolved = _safe_float(summary.get("baseline", {}).get("resolved"), 0.0)
                    gated_resolved = _safe_float(summary.get("gated", {}).get("resolved"), 0.0)
                    coverage_retention = float(gated_resolved / base_resolved) if base_resolved > 0.0 else 1.0
                    concentration = concentration_stats(
                        scored,
                        veto_col="gate_veto",
                        player_col=player_col,
                        target_col=target_col,
                        direction_col=direction_col,
                    )
                    top_player_share = _safe_float(concentration.get("top_player_share"), 0.0)
                    top_segment_share = _safe_float(concentration.get("top_segment_share"), 0.0)
                    top_target_share = _safe_float(concentration.get("top_target_share"), 0.0)
                    veto_rows = int(max(0, _safe_int(concentration.get("veto_rows"), 0)))
                    concentration_enforced = bool(veto_rows >= int(max(1, concentration_min_veto_rows)))
                    effective_target_cap, conditional_target_cap_active = _effective_target_cap(
                        base_cap=float(max_top_removed_target_share),
                        top_player_share=float(top_player_share),
                        recent_profit=float(broad_profit),
                        recent_hit_pp=float(broad_hit),
                        conditional_enabled=bool(conditional_target_cap_enabled),
                        conditional_relaxed_cap=float(conditional_target_relaxed_cap),
                        conditional_player_share_max=float(conditional_target_player_share_max),
                        conditional_require_recent_non_negative=bool(conditional_target_require_recent_non_negative),
                        conditional_recent_profit_floor=float(conditional_target_recent_profit_floor),
                        conditional_recent_hit_floor_pp=float(conditional_target_recent_hit_floor_pp),
                    )
                    effective_player_cap = float(max_top_removed_player_share if concentration_enforced else 1.0)
                    effective_segment_cap = float(max_top_removed_segment_share if concentration_enforced else 1.0)
                    effective_target_cap = float(effective_target_cap if concentration_enforced else 1.0)
                    concentration_features = _player_concentration_features(concentration)
                    player_hhi = float(concentration_features.get("player_hhi", 0.0))
                    player_diversity = float(concentration_features.get("player_diversity", 0.0))
                    constraints_passed = bool(
                        affected_share >= float(min_affected_share)
                        and coverage_retention >= float(min_coverage_retention)
                        and top_player_share <= float(effective_player_cap)
                        and top_segment_share <= float(effective_segment_cap)
                        and top_target_share <= float(effective_target_cap)
                    )
                    structural_pass = bool(
                        coverage_retention >= float(min_coverage_retention)
                        and affected_share >= float(min_affected_share)
                        and top_player_share <= float(effective_player_cap)
                        and top_segment_share <= float(effective_segment_cap)
                        and top_target_share <= float(effective_target_cap)
                        and fire_rate <= float(variant.max_fire_rate)
                    )
                    objective_score = (
                        float(broad_profit)
                        + 0.05 * float(broad_hit)
                        + 0.20 * float(affected_share)
                        + _player_concentration_objective_adjustment(
                            concentration,
                            top_share_penalty=float(objective_player_top_share_penalty),
                            hhi_penalty=float(objective_player_hhi_penalty),
                            diversity_bonus=float(objective_player_diversity_bonus),
                        )
                        - 2.0 * max(0.0, float(min_affected_share) - float(affected_share))
                        - 3.0 * max(0.0, float(min_coverage_retention) - float(coverage_retention))
                        - 2.5 * max(0.0, float(top_player_share) - float(max_top_removed_player_share))
                        - 2.5 * max(0.0, float(top_segment_share) - float(max_top_removed_segment_share))
                        - 2.0 * max(0.0, float(top_target_share) - float(effective_target_cap))
                        - 1.0 * max(0.0, float(fire_rate) - float(variant.max_fire_rate))
                    )
                    payload = {
                        "threshold": float(threshold),
                        "allowed_segments": [str(v) for v in seg_subset],
                        "segment_count": int(len(seg_subset)),
                        "allowed_meta_cohorts": [str(v) for v in cohort_pack],
                        "meta_cohort_count": int(len(cohort_pack)),
                        "tail_slots_only": int(cand_policy.tail_slots_only),
                        "min_veto_gap": float(cand_policy.min_veto_gap),
                        "require_keep_prob_filter": bool(cand_policy.require_keep_prob_filter),
                        "max_global_removed_per_player": int(cand_policy.max_global_removed_per_player),
                        "max_global_removed_share_per_player": float(cand_policy.max_global_removed_share_per_player),
                        "global_player_cap_min_veto_rows": int(cand_policy.global_player_cap_min_veto_rows),
                        "relax_player_global_cap_for_budget": bool(cand_policy.relax_player_global_cap_for_budget),
                        "player_penalty_alpha": float(cand_policy.player_penalty_alpha),
                        "player_penalty_power": float(cand_policy.player_penalty_power),
                        "optimizer_enabled": bool(cand_policy.optimizer_enabled),
                        "optimizer_beam_width": int(cand_policy.optimizer_beam_width),
                        "optimizer_max_candidates_per_day": int(cand_policy.optimizer_max_candidates_per_day),
                        "optimizer_keep_prob_weight": float(cand_policy.optimizer_keep_prob_weight),
                        "optimizer_gap_weight": float(cand_policy.optimizer_gap_weight),
                        "optimizer_harm_weight": float(cand_policy.optimizer_harm_weight),
                        "optimizer_rank_weight": float(cand_policy.optimizer_rank_weight),
                        "segment_threshold_overrides": dict(cand_policy.segment_threshold_overrides or {}),
                        "segment_budget_overrides": dict(cand_policy.segment_budget_overrides or {}),
                        "delta_profit_units": broad_profit,
                        "delta_hit_rate_pp": broad_hit,
                        "fire_rate": fire_rate,
                        "affected_share": float(affected_share),
                        "coverage_retention": float(coverage_retention),
                        "top_player_share": float(top_player_share),
                        "top_segment_share": float(top_segment_share),
                        "top_target_share": float(top_target_share),
                        "veto_rows": int(veto_rows),
                        "concentration_enforced": bool(concentration_enforced),
                        "effective_player_cap": float(effective_player_cap),
                        "effective_segment_cap": float(effective_segment_cap),
                        "effective_target_cap": float(effective_target_cap),
                        "conditional_target_cap_active": bool(conditional_target_cap_active),
                        "player_hhi": float(player_hhi),
                        "player_diversity": float(player_diversity),
                        "constraints_passed": bool(constraints_passed),
                        "structural_pass": bool(structural_pass),
                        "objective_score": float(objective_score),
                        "summary": summary,
                    }
                    if float(threshold) > -0.999999:
                        if best_non_noop is None or payload["objective_score"] > best_non_noop.get("objective_score", float("-inf")) + 1e-12:
                            best_non_noop = payload
                        if bool(payload.get("structural_pass", False)):
                            if (
                                best_non_noop_structural is None
                                or payload["objective_score"] > best_non_noop_structural.get("objective_score", float("-inf")) + 1e-12
                            ):
                                best_non_noop_structural = payload
                    if best_relaxed is None or payload["objective_score"] > best_relaxed.get("objective_score", float("-inf")) + 1e-12:
                        best_relaxed = payload
                    if bool(payload.get("structural_pass", False)):
                        if (
                            best_relaxed_structural is None
                            or payload["objective_score"] > best_relaxed_structural.get("objective_score", float("-inf")) + 1e-12
                        ):
                            best_relaxed_structural = payload
                    if not bool(payload["constraints_passed"]):
                        continue
                    if best_payload is None:
                        best_payload = payload
                        best_threshold = float(threshold)
                        continue
                    if payload["objective_score"] > best_payload.get("objective_score", float("-inf")) + 1e-12:
                        best_payload = payload
                        best_threshold = float(threshold)
                        continue
                    if (
                        abs(payload["objective_score"] - best_payload.get("objective_score", float("-inf"))) <= 1e-12
                        and payload["delta_profit_units"] > best_payload["delta_profit_units"] + 1e-12
                    ):
                        best_payload = payload
                        best_threshold = float(threshold)
                        continue
                    if (
                        abs(payload["objective_score"] - best_payload.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(payload["delta_profit_units"] - best_payload["delta_profit_units"]) <= 1e-12
                        and payload["delta_hit_rate_pp"] > best_payload["delta_hit_rate_pp"] + 1e-12
                    ):
                        best_payload = payload
                        best_threshold = float(threshold)
                        continue
                    if (
                        abs(payload["objective_score"] - best_payload.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(payload["delta_profit_units"] - best_payload["delta_profit_units"]) <= 1e-12
                        and abs(payload["delta_hit_rate_pp"] - best_payload["delta_hit_rate_pp"]) <= 1e-12
                        and payload["affected_share"] > best_payload.get("affected_share", 0.0) + 1e-12
                    ):
                        best_payload = payload
                        best_threshold = float(threshold)
                        continue
                    if (
                        abs(payload["objective_score"] - best_payload.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(payload["delta_profit_units"] - best_payload["delta_profit_units"]) <= 1e-12
                        and abs(payload["delta_hit_rate_pp"] - best_payload["delta_hit_rate_pp"]) <= 1e-12
                        and abs(payload["affected_share"] - best_payload.get("affected_share", 0.0)) <= 1e-12
                        and payload["fire_rate"] < best_payload["fire_rate"] - 1e-12
                    ):
                        best_payload = payload
                        best_threshold = float(threshold)
    if best_payload is not None:
        if (
            bool(prefer_non_noop_when_close)
            and float(best_payload.get("threshold", -1.0)) <= -0.999999
            and best_non_noop_structural is not None
            and float(best_non_noop_structural.get("objective_score", float("-inf")))
            >= float(best_payload.get("objective_score", float("-inf"))) - float(max(0.0, non_noop_objective_margin))
        ):
            out = dict(best_non_noop_structural)
            out["selection_mode"] = "exploratory_non_noop"
            out["exploratory_margin"] = float(max(0.0, non_noop_objective_margin))
            return float(out.get("threshold", -1.0)), out
        best_payload["selection_mode"] = "constraint_aware"
        return float(best_threshold), best_payload
    if best_relaxed_structural is not None:
        best_relaxed_structural["selection_mode"] = "relaxed_objective_structural"
        return float(best_relaxed_structural["threshold"]), best_relaxed_structural
    if best_relaxed is not None:
        return -1.0, {
            "threshold": -1.0,
            "selection_mode": "no_op_structural_fallback",
            "reason": "no_relaxed_candidate_met_structural_constraints",
            "rejected_relaxed_candidate": best_relaxed,
        }
    return float(best_threshold), {"threshold": float(best_threshold), "reason": "fallback"}


def _select_final_threshold_from_oof(
    oof: pd.DataFrame,
    *,
    keep_prob_col: str,
    date_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
    result_col: str,
    policy: GatePolicyConfig,
    quantiles: list[float],
    threshold_max_candidates: int,
    recent_days: int,
    recent_hit_floor_pp: float,
    recent_profit_floor: float,
    broad_profit_floor: float,
    broad_hit_floor_pp: float,
    min_coverage_retention: float,
    max_fire_rate: float,
    segment_search_max_size: int,
    segment_search_max_subsets: int,
    segment_search_min_rows: int,
    segment_search_seed_limit: int,
    cohort_search_enabled: bool,
    cohort_search_min_delta_pp: float,
    min_affected_share: float,
    max_top_removed_player_share: float,
    max_top_removed_segment_share: float,
    max_top_removed_target_share: float,
    concentration_min_veto_rows: int,
    prefer_relaxed_non_noop_when_close: bool,
    relaxed_non_noop_objective_margin: float,
    objective_player_top_share_penalty: float,
    objective_player_hhi_penalty: float,
    objective_player_diversity_bonus: float,
    conditional_target_cap_enabled: bool,
    conditional_target_relaxed_cap: float,
    conditional_target_player_share_max: float,
    conditional_target_require_recent_non_negative: bool,
    conditional_target_recent_hit_floor_pp: float,
    conditional_target_recent_profit_floor: float,
) -> tuple[float, dict[str, Any]]:
    if oof.empty:
        return -1.0, {"reason": "empty_oof", "threshold": -1.0}
    candidates = _build_threshold_grid(
        oof,
        keep_prob_col=keep_prob_col,
        quantiles=quantiles,
        max_candidates=int(threshold_max_candidates),
    )
    dates = _coerce_event_datetime(oof.get(date_col))
    max_date = dates.max()
    recent_start = max_date - pd.Timedelta(days=int(max(1, recent_days)) - 1) if pd.notna(max_date) else pd.NaT
    segment_subsets = _segment_subsets_for_search(
        oof,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        max_size=int(segment_search_max_size),
        max_subsets=int(segment_search_max_subsets),
        min_segment_rows=int(segment_search_min_rows),
        segment_seed_limit=int(segment_search_seed_limit),
    )
    cohort_packs = _meta_cohort_packs_for_search(
        oof,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        max_size=int(segment_search_max_size),
        max_subsets=int(segment_search_max_subsets),
        min_cohort_rows=int(segment_search_min_rows),
        seed_limit=int(segment_search_seed_limit),
        min_negative_delta_pp=float(cohort_search_min_delta_pp),
        enabled=bool(cohort_search_enabled),
    )
    if len(cohort_packs) > 1:
        segment_subsets = [tuple()]
    best: dict[str, Any] | None = None
    best_relaxed: dict[str, Any] | None = None
    best_relaxed_structural: dict[str, Any] | None = None
    all_candidates: list[dict[str, Any]] = []
    policy_variants = _policy_variants_for_search(policy)
    for threshold in candidates:
        for seg_subset in segment_subsets:
            for cohort_pack in cohort_packs:
                for variant in policy_variants:
                    if not bool(variant.require_keep_prob_filter) and not cohort_pack:
                        continue
                    cand_policy = GatePolicyConfig(
                        max_fire_rate=float(variant.max_fire_rate),
                        min_coverage_rate=float(variant.min_coverage_rate),
                        max_removed_per_day=int(variant.max_removed_per_day),
                        max_removed_per_player_per_day=int(variant.max_removed_per_player_per_day),
                        max_removed_per_segment_per_day=int(variant.max_removed_per_segment_per_day),
                        max_removed_per_target_per_day=int(variant.max_removed_per_target_per_day),
                        max_global_removed_per_player=int(variant.max_global_removed_per_player),
                        max_global_removed_share_per_player=float(variant.max_global_removed_share_per_player),
                        global_player_cap_min_veto_rows=int(variant.global_player_cap_min_veto_rows),
                        relax_player_global_cap_for_budget=bool(variant.relax_player_global_cap_for_budget),
                        player_penalty_alpha=float(variant.player_penalty_alpha),
                        player_penalty_power=float(variant.player_penalty_power),
                        optimizer_enabled=bool(variant.optimizer_enabled),
                        optimizer_beam_width=int(variant.optimizer_beam_width),
                        optimizer_max_candidates_per_day=int(variant.optimizer_max_candidates_per_day),
                        optimizer_keep_prob_weight=float(variant.optimizer_keep_prob_weight),
                        optimizer_gap_weight=float(variant.optimizer_gap_weight),
                        optimizer_harm_weight=float(variant.optimizer_harm_weight),
                        optimizer_rank_weight=float(variant.optimizer_rank_weight),
                        segment_threshold_overrides=dict(variant.segment_threshold_overrides or {}),
                        segment_budget_overrides=dict(variant.segment_budget_overrides or {}),
                        tail_slots_only=int(variant.tail_slots_only),
                        min_veto_gap=float(variant.min_veto_gap),
                        require_keep_prob_filter=bool(variant.require_keep_prob_filter),
                        allowed_segments=tuple(seg_subset),
                        allowed_meta_cohorts=tuple(cohort_pack),
                    )
                    scored = apply_shadow_gate_policy(
                        oof,
                        keep_prob_col=keep_prob_col,
                        date_col=date_col,
                        player_col=player_col,
                        target_col=target_col,
                        direction_col=direction_col,
                        threshold=float(threshold),
                        policy=cand_policy,
                    )
                    broad_summary = summarize_paired_outcomes(scored, veto_col="gate_veto", result_col=result_col)
                    if pd.notna(recent_start):
                        recent_part = scored.loc[dates >= recent_start].copy()
                    else:
                        recent_part = scored.iloc[0:0].copy()
                    recent_summary = summarize_paired_outcomes(recent_part, veto_col="gate_veto", result_col=result_col)
                    broad_delta = broad_summary.get("delta", {})
                    recent_delta = recent_summary.get("delta", {})
                    broad_profit = _safe_float(broad_delta.get("profit_units"), float("-inf"))
                    broad_hit = _safe_float(broad_delta.get("hit_rate_pp"), float("-inf"))
                    recent_profit = _safe_float(recent_delta.get("profit_units"), float("-inf"))
                    recent_hit = _safe_float(recent_delta.get("hit_rate_pp"), float("-inf"))
                    fire_rate = _safe_float(pd.to_numeric(scored.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
                    resolved = scored.loc[scored[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
                    affected_share = _safe_float(pd.to_numeric(resolved.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
                    base_resolved = _safe_float(broad_summary.get("baseline", {}).get("resolved"), 0.0)
                    gated_resolved = _safe_float(broad_summary.get("gated", {}).get("resolved"), 0.0)
                    coverage_retention = float(gated_resolved / base_resolved) if base_resolved > 0.0 else 1.0
                    concentration = concentration_stats(
                        scored,
                        veto_col="gate_veto",
                        player_col=player_col,
                        target_col=target_col,
                        direction_col=direction_col,
                    )
                    top_player_share = _safe_float(concentration.get("top_player_share"), 0.0)
                    top_segment_share = _safe_float(concentration.get("top_segment_share"), 0.0)
                    top_target_share = _safe_float(concentration.get("top_target_share"), 0.0)
                    veto_rows = int(max(0, _safe_int(concentration.get("veto_rows"), 0)))
                    concentration_enforced = bool(veto_rows >= int(max(1, concentration_min_veto_rows)))
                    effective_target_cap, conditional_target_cap_active = _effective_target_cap(
                        base_cap=float(max_top_removed_target_share),
                        top_player_share=float(top_player_share),
                        recent_profit=float(recent_profit),
                        recent_hit_pp=float(recent_hit),
                        conditional_enabled=bool(conditional_target_cap_enabled),
                        conditional_relaxed_cap=float(conditional_target_relaxed_cap),
                        conditional_player_share_max=float(conditional_target_player_share_max),
                        conditional_require_recent_non_negative=bool(conditional_target_require_recent_non_negative),
                        conditional_recent_profit_floor=float(conditional_target_recent_profit_floor),
                        conditional_recent_hit_floor_pp=float(conditional_target_recent_hit_floor_pp),
                    )
                    effective_player_cap = float(max_top_removed_player_share if concentration_enforced else 1.0)
                    effective_segment_cap = float(max_top_removed_segment_share if concentration_enforced else 1.0)
                    effective_target_cap = float(effective_target_cap if concentration_enforced else 1.0)
                    concentration_features = _player_concentration_features(concentration)
                    player_hhi = float(concentration_features.get("player_hhi", 0.0))
                    player_diversity = float(concentration_features.get("player_diversity", 0.0))
                    constraints_passed = bool(
                        broad_profit >= float(broad_profit_floor)
                        and broad_hit >= float(broad_hit_floor_pp)
                        and recent_hit >= float(recent_hit_floor_pp)
                        and recent_profit >= float(recent_profit_floor)
                        and coverage_retention >= float(min_coverage_retention)
                        and fire_rate <= float(max_fire_rate)
                        and affected_share >= float(min_affected_share)
                        and top_player_share <= float(effective_player_cap)
                        and top_segment_share <= float(effective_segment_cap)
                        and top_target_share <= float(effective_target_cap)
                    )
                    structural_pass = bool(
                        coverage_retention >= float(min_coverage_retention)
                        and fire_rate <= float(max_fire_rate)
                        and affected_share >= float(min_affected_share)
                        and top_player_share <= float(effective_player_cap)
                        and top_segment_share <= float(effective_segment_cap)
                        and top_target_share <= float(effective_target_cap)
                    )
                    objective_score = (
                        float(broad_profit)
                        + 0.80 * float(recent_profit)
                        + 0.02 * float(broad_hit)
                        + 0.03 * float(recent_hit)
                        + 0.40 * float(affected_share)
                        + _player_concentration_objective_adjustment(
                            concentration,
                            top_share_penalty=float(objective_player_top_share_penalty),
                            hhi_penalty=float(objective_player_hhi_penalty),
                            diversity_bonus=float(objective_player_diversity_bonus),
                        )
                        - 3.0 * max(0.0, float(min_coverage_retention) - float(coverage_retention))
                        - 3.0 * max(0.0, float(min_affected_share) - float(affected_share))
                        - 4.0 * max(0.0, float(top_player_share) - float(max_top_removed_player_share))
                        - 4.0 * max(0.0, float(top_segment_share) - float(max_top_removed_segment_share))
                        - 3.0 * max(0.0, float(top_target_share) - float(effective_target_cap))
                        - 2.0 * max(0.0, float(fire_rate) - float(max_fire_rate))
                    )
                    candidate = {
                        "threshold": float(threshold),
                        "allowed_segments": list(seg_subset),
                        "segment_count": int(len(seg_subset)),
                        "allowed_meta_cohorts": [str(v) for v in cohort_pack],
                        "meta_cohort_count": int(len(cohort_pack)),
                        "tail_slots_only": int(cand_policy.tail_slots_only),
                        "min_veto_gap": float(cand_policy.min_veto_gap),
                        "require_keep_prob_filter": bool(cand_policy.require_keep_prob_filter),
                        "max_global_removed_per_player": int(cand_policy.max_global_removed_per_player),
                        "max_global_removed_share_per_player": float(cand_policy.max_global_removed_share_per_player),
                        "global_player_cap_min_veto_rows": int(cand_policy.global_player_cap_min_veto_rows),
                        "relax_player_global_cap_for_budget": bool(cand_policy.relax_player_global_cap_for_budget),
                        "player_penalty_alpha": float(cand_policy.player_penalty_alpha),
                        "player_penalty_power": float(cand_policy.player_penalty_power),
                        "optimizer_enabled": bool(cand_policy.optimizer_enabled),
                        "optimizer_beam_width": int(cand_policy.optimizer_beam_width),
                        "optimizer_max_candidates_per_day": int(cand_policy.optimizer_max_candidates_per_day),
                        "optimizer_keep_prob_weight": float(cand_policy.optimizer_keep_prob_weight),
                        "optimizer_gap_weight": float(cand_policy.optimizer_gap_weight),
                        "optimizer_harm_weight": float(cand_policy.optimizer_harm_weight),
                        "optimizer_rank_weight": float(cand_policy.optimizer_rank_weight),
                        "segment_threshold_overrides": dict(cand_policy.segment_threshold_overrides or {}),
                        "segment_budget_overrides": dict(cand_policy.segment_budget_overrides or {}),
                        "broad_profit_units": float(broad_profit),
                        "broad_hit_rate_pp": float(broad_hit),
                        "recent_profit_units": float(recent_profit),
                        "recent_hit_rate_pp": float(recent_hit),
                        "coverage_retention": float(coverage_retention),
                        "fire_rate": float(fire_rate),
                        "affected_share": float(affected_share),
                        "top_player_share": float(top_player_share),
                        "top_segment_share": float(top_segment_share),
                        "top_target_share": float(top_target_share),
                        "veto_rows": int(veto_rows),
                        "concentration_enforced": bool(concentration_enforced),
                        "effective_player_cap": float(effective_player_cap),
                        "effective_segment_cap": float(effective_segment_cap),
                        "effective_target_cap": float(effective_target_cap),
                        "conditional_target_cap_active": bool(conditional_target_cap_active),
                        "player_hhi": float(player_hhi),
                        "player_diversity": float(player_diversity),
                        "objective_score": float(objective_score),
                        "broad_summary": broad_summary,
                        "recent_summary": recent_summary,
                        "constraints_passed": constraints_passed,
                        "structural_pass": structural_pass,
                        "recent_days": int(max(1, recent_days)),
                        "constraints": {
                            "broad_profit_floor": float(broad_profit_floor),
                            "broad_hit_floor_pp": float(broad_hit_floor_pp),
                            "recent_profit_floor": float(recent_profit_floor),
                            "recent_hit_floor_pp": float(recent_hit_floor_pp),
                            "min_coverage_retention": float(min_coverage_retention),
                            "max_fire_rate": float(max_fire_rate),
                            "segment_search_max_size": int(segment_search_max_size),
                            "segment_search_max_subsets": int(segment_search_max_subsets),
                            "segment_search_min_rows": int(segment_search_min_rows),
                            "segment_search_seed_limit": int(segment_search_seed_limit),
                            "cohort_search_enabled": bool(cohort_search_enabled),
                            "cohort_search_min_delta_pp": float(cohort_search_min_delta_pp),
                            "min_affected_share": float(min_affected_share),
                            "max_top_removed_player_share": float(max_top_removed_player_share),
                            "max_top_removed_segment_share": float(max_top_removed_segment_share),
                            "max_top_removed_target_share": float(max_top_removed_target_share),
                            "concentration_min_veto_rows": int(max(1, concentration_min_veto_rows)),
                            "effective_target_cap": float(effective_target_cap),
                            "max_global_removed_per_player": int(cand_policy.max_global_removed_per_player),
                            "max_global_removed_share_per_player": float(cand_policy.max_global_removed_share_per_player),
                            "global_player_cap_min_veto_rows": int(cand_policy.global_player_cap_min_veto_rows),
                            "relax_player_global_cap_for_budget": bool(cand_policy.relax_player_global_cap_for_budget),
                            "player_penalty_alpha": float(cand_policy.player_penalty_alpha),
                            "player_penalty_power": float(cand_policy.player_penalty_power),
                            "objective_player_top_share_penalty": float(objective_player_top_share_penalty),
                            "objective_player_hhi_penalty": float(objective_player_hhi_penalty),
                            "objective_player_diversity_bonus": float(objective_player_diversity_bonus),
                            "conditional_target_cap_enabled": bool(conditional_target_cap_enabled),
                            "conditional_target_relaxed_cap": float(conditional_target_relaxed_cap),
                            "conditional_target_player_share_max": float(conditional_target_player_share_max),
                            "conditional_target_require_recent_non_negative": bool(conditional_target_require_recent_non_negative),
                            "conditional_target_recent_hit_floor_pp": float(conditional_target_recent_hit_floor_pp),
                            "conditional_target_recent_profit_floor": float(conditional_target_recent_profit_floor),
                        },
                    }
                    all_candidates.append(candidate)
                    if candidate["constraints_passed"]:
                        if best is None:
                            best = candidate
                        elif candidate["objective_score"] > best.get("objective_score", float("-inf")) + 1e-12:
                            best = candidate
                        elif candidate["broad_profit_units"] > best["broad_profit_units"] + 1e-12:
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and candidate["recent_profit_units"] > best["recent_profit_units"] + 1e-12
                        ):
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
                            and candidate["broad_hit_rate_pp"] > best["broad_hit_rate_pp"] + 1e-12
                        ):
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
                            and abs(candidate["broad_hit_rate_pp"] - best["broad_hit_rate_pp"]) <= 1e-12
                            and candidate["recent_hit_rate_pp"] > best["recent_hit_rate_pp"] + 1e-12
                        ):
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
                            and abs(candidate["broad_hit_rate_pp"] - best["broad_hit_rate_pp"]) <= 1e-12
                            and abs(candidate["recent_hit_rate_pp"] - best["recent_hit_rate_pp"]) <= 1e-12
                            and candidate["affected_share"] > best.get("affected_share", 0.0) + 1e-12
                        ):
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
                            and abs(candidate["broad_hit_rate_pp"] - best["broad_hit_rate_pp"]) <= 1e-12
                            and abs(candidate["recent_hit_rate_pp"] - best["recent_hit_rate_pp"]) <= 1e-12
                            and abs(candidate["affected_share"] - best.get("affected_share", 0.0)) <= 1e-12
                            and candidate["fire_rate"] < best["fire_rate"] - 1e-12
                        ):
                            best = candidate
                        elif (
                            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
                            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
                            and abs(candidate["broad_hit_rate_pp"] - best["broad_hit_rate_pp"]) <= 1e-12
                            and abs(candidate["recent_hit_rate_pp"] - best["recent_hit_rate_pp"]) <= 1e-12
                            and abs(candidate["affected_share"] - best.get("affected_share", 0.0)) <= 1e-12
                            and abs(candidate["fire_rate"] - best["fire_rate"]) <= 1e-12
                            and candidate["segment_count"] < best.get("segment_count", 0)
                        ):
                            best = candidate
                    if best_relaxed is None:
                        best_relaxed = candidate
                    elif candidate["objective_score"] > best_relaxed.get("objective_score", float("-inf")) + 1e-12:
                        best_relaxed = candidate
                    elif (
                        abs(candidate["objective_score"] - best_relaxed.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(candidate["broad_profit_units"] - best_relaxed["broad_profit_units"]) <= 1e-12
                        and candidate["broad_hit_rate_pp"] > best_relaxed["broad_hit_rate_pp"] + 1e-12
                    ):
                        best_relaxed = candidate
                    elif (
                        abs(candidate["objective_score"] - best_relaxed.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(candidate["broad_profit_units"] - best_relaxed["broad_profit_units"]) <= 1e-12
                        and abs(candidate["broad_hit_rate_pp"] - best_relaxed["broad_hit_rate_pp"]) <= 1e-12
                        and candidate.get("affected_share", 0.0) > best_relaxed.get("affected_share", 0.0) + 1e-12
                    ):
                        best_relaxed = candidate
                    elif (
                        abs(candidate["objective_score"] - best_relaxed.get("objective_score", float("-inf"))) <= 1e-12
                        and abs(candidate["broad_profit_units"] - best_relaxed["broad_profit_units"]) <= 1e-12
                        and abs(candidate["broad_hit_rate_pp"] - best_relaxed["broad_hit_rate_pp"]) <= 1e-12
                        and abs(candidate.get("affected_share", 0.0) - best_relaxed.get("affected_share", 0.0)) <= 1e-12
                        and candidate["fire_rate"] < best_relaxed["fire_rate"] - 1e-12
                    ):
                        best_relaxed = candidate
                    if bool(candidate.get("structural_pass", False)):
                        if best_relaxed_structural is None:
                            best_relaxed_structural = candidate
                        elif candidate["objective_score"] > best_relaxed_structural.get("objective_score", float("-inf")) + 1e-12:
                            best_relaxed_structural = candidate
                        elif (
                            abs(candidate["objective_score"] - best_relaxed_structural.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best_relaxed_structural["broad_profit_units"]) <= 1e-12
                            and candidate["broad_hit_rate_pp"] > best_relaxed_structural["broad_hit_rate_pp"] + 1e-12
                        ):
                            best_relaxed_structural = candidate
                        elif (
                            abs(candidate["objective_score"] - best_relaxed_structural.get("objective_score", float("-inf"))) <= 1e-12
                            and abs(candidate["broad_profit_units"] - best_relaxed_structural["broad_profit_units"]) <= 1e-12
                            and abs(candidate["broad_hit_rate_pp"] - best_relaxed_structural["broad_hit_rate_pp"]) <= 1e-12
                            and candidate.get("affected_share", 0.0) > best_relaxed_structural.get("affected_share", 0.0) + 1e-12
                        ):
                            best_relaxed_structural = candidate

    if best is not None:
        best["selection_mode"] = "constrained"
        return float(best["threshold"]), best
    no_op_candidates = [
        cand
        for cand in all_candidates
        if float(cand.get("threshold", 0.0)) <= -0.999999
        and abs(_safe_float(cand.get("affected_share"), 0.0)) <= 1e-12
    ]
    if not no_op_candidates:
        no_op_candidates = [
            cand
            for cand in all_candidates
            if float(cand.get("threshold", 0.0)) <= -0.999999
            and bool(cand.get("require_keep_prob_filter", True))
        ]
    if not no_op_candidates:
        no_op_candidates = [cand for cand in all_candidates if float(cand.get("threshold", 0.0)) <= -0.999999]
    no_op = (
        max(no_op_candidates, key=lambda cand: float(cand.get("objective_score", float("-inf"))))
        if no_op_candidates
        else None
    )
    if (
        bool(prefer_relaxed_non_noop_when_close)
        and best_relaxed_structural is not None
        and float(best_relaxed_structural.get("threshold", -1.0)) > -0.999999
        and no_op is not None
        and float(best_relaxed_structural.get("objective_score", float("-inf")))
        >= float(no_op.get("objective_score", float("-inf"))) - float(max(0.0, relaxed_non_noop_objective_margin))
    ):
        best_relaxed_structural["selection_mode"] = "relaxed_exploratory"
        best_relaxed_structural["reason"] = "no_constrained_candidate_selected_relaxed_non_noop"
        best_relaxed_structural["relaxed_objective_margin"] = float(max(0.0, relaxed_non_noop_objective_margin))
        return float(best_relaxed_structural["threshold"]), best_relaxed_structural
    if no_op is not None:
        no_op["selection_mode"] = "no_op_fallback"
        no_op["reason"] = "no_threshold_passed_constraints"
        no_op["failed_candidate_count"] = int(sum(1 for cand in all_candidates if not bool(cand.get("constraints_passed", False))))
        no_op["failed_structural_count"] = int(sum(1 for cand in all_candidates if not bool(cand.get("structural_pass", False))))
        if best_relaxed is not None and not bool(best_relaxed.get("structural_pass", False)):
            no_op["relaxed_rejected_reason"] = "best_relaxed_failed_structural_constraints"
            no_op["rejected_relaxed_candidate"] = best_relaxed
        return float(no_op["threshold"]), no_op
    if best_relaxed_structural is not None:
        best_relaxed_structural["selection_mode"] = "relaxed_no_noop_structural"
        return float(best_relaxed_structural["threshold"]), best_relaxed_structural
    if best_relaxed is not None:
        return -1.0, {
            "reason": "no_relaxed_candidate_met_structural_constraints",
            "threshold": -1.0,
            "selection_mode": "fallback_structural_noop",
            "rejected_relaxed_candidate": best_relaxed,
        }
    return -1.0, {"reason": "no_candidates", "threshold": -1.0, "selection_mode": "fallback"}


def _select_fullframe_cohort_fallback(
    frame: pd.DataFrame,
    *,
    keep_prob_col: str,
    date_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
    result_col: str,
    base_policy: GatePolicyConfig,
    recent_days: int,
    recent_hit_floor_pp: float,
    recent_profit_floor: float,
    broad_profit_floor: float,
    broad_hit_floor_pp: float,
    min_coverage_retention: float,
    max_fire_rate: float,
    segment_search_max_size: int,
    segment_search_max_subsets: int,
    segment_search_min_rows: int,
    segment_search_seed_limit: int,
    cohort_search_min_delta_pp: float,
    min_affected_share: float,
    max_top_removed_player_share: float,
    max_top_removed_segment_share: float,
    max_top_removed_target_share: float,
    concentration_min_veto_rows: int,
    conditional_target_cap_enabled: bool,
    conditional_target_relaxed_cap: float,
    conditional_target_player_share_max: float,
    conditional_target_require_recent_non_negative: bool,
    conditional_target_recent_hit_floor_pp: float,
    conditional_target_recent_profit_floor: float,
    objective_player_top_share_penalty: float,
    objective_player_hhi_penalty: float,
    objective_player_diversity_bonus: float,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    cohort_packs = _meta_cohort_packs_for_search(
        frame,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        max_size=int(segment_search_max_size),
        max_subsets=int(segment_search_max_subsets),
        min_cohort_rows=int(segment_search_min_rows),
        seed_limit=int(segment_search_seed_limit),
        min_negative_delta_pp=float(cohort_search_min_delta_pp),
        enabled=True,
    )
    cohort_packs = [pack for pack in cohort_packs if pack]
    if not cohort_packs:
        return None

    dates = _coerce_event_datetime(frame.get(date_col))
    max_date = dates.max()
    recent_start = max_date - pd.Timedelta(days=int(max(1, recent_days)) - 1) if pd.notna(max_date) else pd.NaT
    effective_fire_rate = float(min(1.0, max(float(base_policy.max_fire_rate), float(max_fire_rate))))
    effective_min_coverage = float(max(0.0, min(float(base_policy.min_coverage_rate), float(min_coverage_retention))))
    best: dict[str, Any] | None = None
    for cohort_pack in cohort_packs:
        policy = GatePolicyConfig(
            max_fire_rate=float(effective_fire_rate),
            min_coverage_rate=float(effective_min_coverage),
            max_removed_per_day=int(base_policy.max_removed_per_day),
            max_removed_per_player_per_day=int(base_policy.max_removed_per_player_per_day),
            max_removed_per_segment_per_day=int(base_policy.max_removed_per_segment_per_day),
            max_removed_per_target_per_day=int(base_policy.max_removed_per_target_per_day),
            max_global_removed_per_player=int(base_policy.max_global_removed_per_player),
            max_global_removed_share_per_player=float(base_policy.max_global_removed_share_per_player),
            global_player_cap_min_veto_rows=int(base_policy.global_player_cap_min_veto_rows),
            relax_player_global_cap_for_budget=bool(base_policy.relax_player_global_cap_for_budget),
            player_penalty_alpha=float(base_policy.player_penalty_alpha),
            player_penalty_power=float(base_policy.player_penalty_power),
            tail_slots_only=0,
            min_veto_gap=0.0,
            require_keep_prob_filter=False,
            allowed_segments=tuple(),
            allowed_meta_cohorts=tuple(cohort_pack),
        )
        scored = apply_shadow_gate_policy(
            frame,
            keep_prob_col=keep_prob_col,
            date_col=date_col,
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            threshold=-1.0,
            policy=policy,
        )
        broad_summary = summarize_paired_outcomes(scored, veto_col="gate_veto", result_col=result_col)
        if pd.notna(recent_start):
            recent_part = scored.loc[dates >= recent_start].copy()
        else:
            recent_part = scored.iloc[0:0].copy()
        recent_summary = summarize_paired_outcomes(recent_part, veto_col="gate_veto", result_col=result_col)
        broad_delta = broad_summary.get("delta", {})
        recent_delta = recent_summary.get("delta", {})
        broad_profit = _safe_float(broad_delta.get("profit_units"), float("-inf"))
        broad_hit = _safe_float(broad_delta.get("hit_rate_pp"), float("-inf"))
        recent_profit = _safe_float(recent_delta.get("profit_units"), float("-inf"))
        recent_hit = _safe_float(recent_delta.get("hit_rate_pp"), float("-inf"))
        fire_rate = _safe_float(pd.to_numeric(scored.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
        resolved = scored.loc[scored[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
        affected_share = _safe_float(pd.to_numeric(resolved.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
        base_resolved = _safe_float(broad_summary.get("baseline", {}).get("resolved"), 0.0)
        gated_resolved = _safe_float(broad_summary.get("gated", {}).get("resolved"), 0.0)
        coverage_retention = float(gated_resolved / base_resolved) if base_resolved > 0.0 else 1.0
        concentration = concentration_stats(
            scored,
            veto_col="gate_veto",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
        )
        top_player_share = _safe_float(concentration.get("top_player_share"), 0.0)
        top_segment_share = _safe_float(concentration.get("top_segment_share"), 0.0)
        top_target_share = _safe_float(concentration.get("top_target_share"), 0.0)
        veto_rows = int(max(0, _safe_int(concentration.get("veto_rows"), 0)))
        concentration_enforced = bool(veto_rows >= int(max(1, concentration_min_veto_rows)))
        effective_target_cap, conditional_target_cap_active = _effective_target_cap(
            base_cap=float(max_top_removed_target_share),
            top_player_share=float(top_player_share),
            recent_profit=float(recent_profit),
            recent_hit_pp=float(recent_hit),
            conditional_enabled=bool(conditional_target_cap_enabled),
            conditional_relaxed_cap=float(conditional_target_relaxed_cap),
            conditional_player_share_max=float(conditional_target_player_share_max),
            conditional_require_recent_non_negative=bool(conditional_target_require_recent_non_negative),
            conditional_recent_profit_floor=float(conditional_target_recent_profit_floor),
            conditional_recent_hit_floor_pp=float(conditional_target_recent_hit_floor_pp),
        )
        effective_player_cap = float(max_top_removed_player_share if concentration_enforced else 1.0)
        effective_segment_cap = float(max_top_removed_segment_share if concentration_enforced else 1.0)
        effective_target_cap = float(effective_target_cap if concentration_enforced else 1.0)
        concentration_features = _player_concentration_features(concentration)
        player_hhi = float(concentration_features.get("player_hhi", 0.0))
        player_diversity = float(concentration_features.get("player_diversity", 0.0))
        constraints_passed = bool(
            broad_profit >= float(broad_profit_floor)
            and broad_hit >= float(broad_hit_floor_pp)
            and recent_hit >= float(recent_hit_floor_pp)
            and recent_profit >= float(recent_profit_floor)
            and coverage_retention >= float(min_coverage_retention)
            and fire_rate <= float(max_fire_rate)
            and affected_share >= float(min_affected_share)
            and top_player_share <= float(effective_player_cap)
            and top_segment_share <= float(effective_segment_cap)
            and top_target_share <= float(effective_target_cap)
        )
        if not constraints_passed:
            continue
        objective_score = (
            float(broad_profit)
            + 0.80 * float(recent_profit)
            + 0.03 * float(broad_hit)
            + 0.04 * float(recent_hit)
            + 0.30 * float(affected_share)
            + _player_concentration_objective_adjustment(
                concentration,
                top_share_penalty=float(objective_player_top_share_penalty),
                hhi_penalty=float(objective_player_hhi_penalty),
                diversity_bonus=float(objective_player_diversity_bonus),
            )
        )
        candidate = {
            "threshold": -1.0,
            "allowed_segments": [],
            "segment_count": 0,
            "allowed_meta_cohorts": [str(v) for v in cohort_pack],
            "meta_cohort_count": int(len(cohort_pack)),
            "tail_slots_only": 0,
            "min_veto_gap": 0.0,
            "require_keep_prob_filter": False,
            "max_global_removed_per_player": int(policy.max_global_removed_per_player),
            "max_global_removed_share_per_player": float(policy.max_global_removed_share_per_player),
            "global_player_cap_min_veto_rows": int(policy.global_player_cap_min_veto_rows),
            "relax_player_global_cap_for_budget": bool(policy.relax_player_global_cap_for_budget),
            "player_penalty_alpha": float(policy.player_penalty_alpha),
            "player_penalty_power": float(policy.player_penalty_power),
            "broad_profit_units": float(broad_profit),
            "broad_hit_rate_pp": float(broad_hit),
            "recent_profit_units": float(recent_profit),
            "recent_hit_rate_pp": float(recent_hit),
            "coverage_retention": float(coverage_retention),
            "fire_rate": float(fire_rate),
            "affected_share": float(affected_share),
            "top_player_share": float(top_player_share),
            "top_segment_share": float(top_segment_share),
            "top_target_share": float(top_target_share),
            "veto_rows": int(veto_rows),
            "concentration_enforced": bool(concentration_enforced),
            "effective_player_cap": float(effective_player_cap),
            "effective_segment_cap": float(effective_segment_cap),
            "effective_target_cap": float(effective_target_cap),
            "conditional_target_cap_active": bool(conditional_target_cap_active),
            "player_hhi": float(player_hhi),
            "player_diversity": float(player_diversity),
            "objective_score": float(objective_score),
            "broad_summary": broad_summary,
            "recent_summary": recent_summary,
            "constraints_passed": True,
            "recent_days": int(max(1, recent_days)),
            "selection_mode": "fullframe_cohort_fallback",
            "reason": "oof_noop_fullframe_cohort_fallback_selected",
        }
        if best is None:
            best = candidate
            continue
        if candidate["objective_score"] > best.get("objective_score", float("-inf")) + 1e-12:
            best = candidate
            continue
        if (
            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
            and candidate["broad_profit_units"] > best["broad_profit_units"] + 1e-12
        ):
            best = candidate
            continue
        if (
            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
            and candidate["recent_profit_units"] > best["recent_profit_units"] + 1e-12
        ):
            best = candidate
            continue
        if (
            abs(candidate["objective_score"] - best.get("objective_score", float("-inf"))) <= 1e-12
            and abs(candidate["broad_profit_units"] - best["broad_profit_units"]) <= 1e-12
            and abs(candidate["recent_profit_units"] - best["recent_profit_units"]) <= 1e-12
            and candidate["recent_hit_rate_pp"] > best["recent_hit_rate_pp"] + 1e-12
        ):
            best = candidate
    return best


def _search_segment_override_on_oof(
    frame: pd.DataFrame,
    *,
    keep_prob_col: str,
    date_col: str,
    player_col: str,
    target_col: str,
    direction_col: str,
    result_col: str,
    base_threshold: float,
    base_policy: GatePolicyConfig,
    segment_key: str,
    threshold_candidates: list[float],
    budget_candidates: list[int],
    recent_days: int,
    recent_hit_floor_pp: float,
    recent_profit_floor: float,
    broad_profit_floor: float,
    broad_hit_floor_pp: float,
    min_coverage_retention: float,
    max_fire_rate: float,
    min_affected_share: float,
    max_top_removed_player_share: float,
    max_top_removed_segment_share: float,
    max_top_removed_target_share: float,
    concentration_min_veto_rows: int,
    conditional_target_cap_enabled: bool,
    conditional_target_relaxed_cap: float,
    conditional_target_player_share_max: float,
    conditional_target_require_recent_non_negative: bool,
    conditional_target_recent_hit_floor_pp: float,
    conditional_target_recent_profit_floor: float,
    objective_player_top_share_penalty: float,
    objective_player_hhi_penalty: float,
    objective_player_diversity_bonus: float,
    min_objective_gain: float,
    min_broad_profit_gain: float,
    min_recent_profit_gain: float,
    min_affected_share_gain: float,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    segment = str(segment_key).upper().strip()
    if not segment:
        return None
    targets = frame.get(target_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    directions = frame.get(direction_col, pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    segments = (targets + "|" + directions).astype(str)
    if not bool(segments.eq(segment).any()):
        return None

    dates = _coerce_event_datetime(frame.get(date_col))
    max_date = dates.max()
    recent_start = max_date - pd.Timedelta(days=int(max(1, recent_days)) - 1) if pd.notna(max_date) else pd.NaT

    def _evaluate(scored: pd.DataFrame) -> dict[str, Any]:
        broad_summary = summarize_paired_outcomes(scored, veto_col="gate_veto", result_col=result_col)
        if pd.notna(recent_start):
            recent_part = scored.loc[dates >= recent_start].copy()
        else:
            recent_part = scored.iloc[0:0].copy()
        recent_summary = summarize_paired_outcomes(recent_part, veto_col="gate_veto", result_col=result_col)
        broad_delta = broad_summary.get("delta", {})
        recent_delta = recent_summary.get("delta", {})
        broad_profit = _safe_float(broad_delta.get("profit_units"), float("-inf"))
        broad_hit = _safe_float(broad_delta.get("hit_rate_pp"), float("-inf"))
        recent_profit = _safe_float(recent_delta.get("profit_units"), float("-inf"))
        recent_hit = _safe_float(recent_delta.get("hit_rate_pp"), float("-inf"))
        fire_rate = _safe_float(pd.to_numeric(scored.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
        resolved = scored.loc[scored[result_col].astype(str).str.lower().isin(["win", "loss"])].copy()
        affected_share = _safe_float(pd.to_numeric(resolved.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
        base_resolved = _safe_float(broad_summary.get("baseline", {}).get("resolved"), 0.0)
        gated_resolved = _safe_float(broad_summary.get("gated", {}).get("resolved"), 0.0)
        coverage_retention = float(gated_resolved / base_resolved) if base_resolved > 0.0 else 1.0
        concentration = concentration_stats(
            scored,
            veto_col="gate_veto",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
        )
        top_player_share = _safe_float(concentration.get("top_player_share"), 0.0)
        top_segment_share = _safe_float(concentration.get("top_segment_share"), 0.0)
        top_target_share = _safe_float(concentration.get("top_target_share"), 0.0)
        veto_rows = int(max(0, _safe_int(concentration.get("veto_rows"), 0)))
        concentration_enforced = bool(veto_rows >= int(max(1, concentration_min_veto_rows)))
        effective_target_cap, conditional_target_cap_active = _effective_target_cap(
            base_cap=float(max_top_removed_target_share),
            top_player_share=float(top_player_share),
            recent_profit=float(recent_profit),
            recent_hit_pp=float(recent_hit),
            conditional_enabled=bool(conditional_target_cap_enabled),
            conditional_relaxed_cap=float(conditional_target_relaxed_cap),
            conditional_player_share_max=float(conditional_target_player_share_max),
            conditional_require_recent_non_negative=bool(conditional_target_require_recent_non_negative),
            conditional_recent_profit_floor=float(conditional_target_recent_profit_floor),
            conditional_recent_hit_floor_pp=float(conditional_target_recent_hit_floor_pp),
        )
        effective_player_cap = float(max_top_removed_player_share if concentration_enforced else 1.0)
        effective_segment_cap = float(max_top_removed_segment_share if concentration_enforced else 1.0)
        effective_target_cap = float(effective_target_cap if concentration_enforced else 1.0)
        constraints_passed = bool(
            broad_profit >= float(broad_profit_floor)
            and broad_hit >= float(broad_hit_floor_pp)
            and recent_hit >= float(recent_hit_floor_pp)
            and recent_profit >= float(recent_profit_floor)
            and coverage_retention >= float(min_coverage_retention)
            and fire_rate <= float(max_fire_rate)
            and affected_share >= float(min_affected_share)
            and top_player_share <= float(effective_player_cap)
            and top_segment_share <= float(effective_segment_cap)
            and top_target_share <= float(effective_target_cap)
        )
        objective_score = (
            float(broad_profit)
            + 0.80 * float(recent_profit)
            + 0.03 * float(broad_hit)
            + 0.04 * float(recent_hit)
            + 0.30 * float(affected_share)
            + _player_concentration_objective_adjustment(
                concentration,
                top_share_penalty=float(objective_player_top_share_penalty),
                hhi_penalty=float(objective_player_hhi_penalty),
                diversity_bonus=float(objective_player_diversity_bonus),
            )
        )
        return {
            "broad_summary": broad_summary,
            "recent_summary": recent_summary,
            "broad_profit_units": float(broad_profit),
            "broad_hit_rate_pp": float(broad_hit),
            "recent_profit_units": float(recent_profit),
            "recent_hit_rate_pp": float(recent_hit),
            "coverage_retention": float(coverage_retention),
            "fire_rate": float(fire_rate),
            "affected_share": float(affected_share),
            "top_player_share": float(top_player_share),
            "top_segment_share": float(top_segment_share),
            "top_target_share": float(top_target_share),
            "veto_rows": int(veto_rows),
            "concentration_enforced": bool(concentration_enforced),
            "effective_player_cap": float(effective_player_cap),
            "effective_segment_cap": float(effective_segment_cap),
            "effective_target_cap": float(effective_target_cap),
            "conditional_target_cap_active": bool(conditional_target_cap_active),
            "objective_score": float(objective_score),
            "constraints_passed": bool(constraints_passed),
        }

    baseline_scored = apply_shadow_gate_policy(
        frame,
        keep_prob_col=keep_prob_col,
        date_col=date_col,
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
        threshold=float(base_threshold),
        policy=base_policy,
    )
    baseline_eval = _evaluate(baseline_scored)

    best: dict[str, Any] | None = None
    threshold_values = sorted(set(float(np.clip(v, 0.0, 1.0)) for v in threshold_candidates))
    budget_values = sorted(set(int(max(1, v)) for v in budget_candidates))
    for seg_threshold in threshold_values:
        for seg_budget in budget_values:
            if seg_threshold <= -0.999999:
                continue
            merged_segment_thresholds = dict(base_policy.segment_threshold_overrides or {})
            merged_segment_thresholds[segment] = float(max(seg_threshold, base_threshold))
            merged_segment_budgets = dict(base_policy.segment_budget_overrides or {})
            merged_segment_budgets[segment] = int(seg_budget)
            policy_payload = asdict(base_policy)
            policy_payload["segment_threshold_overrides"] = merged_segment_thresholds
            policy_payload["segment_budget_overrides"] = merged_segment_budgets
            cand_policy = GatePolicyConfig(**policy_payload)
            scored = apply_shadow_gate_policy(
                frame,
                keep_prob_col=keep_prob_col,
                date_col=date_col,
                player_col=player_col,
                target_col=target_col,
                direction_col=direction_col,
                threshold=float(base_threshold),
                policy=cand_policy,
            )
            veto_mask = pd.to_numeric(scored.get("gate_veto"), errors="coerce").fillna(0).astype(bool)
            veto_in_segment = int(veto_mask.loc[segments.eq(segment)].sum())
            if veto_in_segment <= 0:
                continue
            eval_payload = _evaluate(scored)
            if not bool(eval_payload.get("constraints_passed", False)):
                continue
            if float(eval_payload.get("objective_score", float("-inf"))) < float(
                baseline_eval.get("objective_score", float("-inf"))
            ) + float(min_objective_gain):
                continue
            if float(eval_payload.get("broad_profit_units", float("-inf"))) < float(
                baseline_eval.get("broad_profit_units", float("-inf"))
            ) + float(min_broad_profit_gain):
                continue
            if float(eval_payload.get("recent_profit_units", float("-inf"))) < float(
                baseline_eval.get("recent_profit_units", float("-inf"))
            ) + float(min_recent_profit_gain):
                continue
            if float(eval_payload.get("affected_share", 0.0)) < float(
                baseline_eval.get("affected_share", 0.0)
            ) + float(min_affected_share_gain):
                continue
            candidate = {
                **eval_payload,
                "segment_key": segment,
                "segment_threshold": float(seg_threshold),
                "segment_budget": int(seg_budget),
                "segment_veto_rows": int(veto_in_segment),
                "segment_threshold_overrides": merged_segment_thresholds,
                "segment_budget_overrides": merged_segment_budgets,
                "selection_mode": "segment_override",
            }
            if best is None:
                best = candidate
                continue
            if float(candidate["objective_score"]) > float(best.get("objective_score", float("-inf"))) + 1e-12:
                best = candidate
                continue
            if float(candidate["broad_profit_units"]) > float(best.get("broad_profit_units", float("-inf"))) + 1e-12:
                best = candidate
                continue
            if float(candidate["recent_profit_units"]) > float(best.get("recent_profit_units", float("-inf"))) + 1e-12:
                best = candidate
                continue
            if int(candidate["segment_veto_rows"]) > int(best.get("segment_veto_rows", 0)):
                best = candidate
    if best is None:
        return None
    return {
        "baseline": baseline_eval,
        "selected": best,
        "constraints": {
            "segment_key": str(segment),
            "threshold_candidates": [float(v) for v in threshold_values],
            "budget_candidates": [int(v) for v in budget_values],
            "min_objective_gain": float(min_objective_gain),
            "min_broad_profit_gain": float(min_broad_profit_gain),
            "min_recent_profit_gain": float(min_recent_profit_gain),
            "min_affected_share_gain": float(min_affected_share_gain),
        },
    }


def cmd_train_shadow(args: argparse.Namespace) -> None:
    history_csv = args.history_csv.resolve()
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")
    history = pd.read_csv(history_csv)
    if history.empty:
        raise RuntimeError(f"History CSV has no rows: {history_csv}")

    date_col = _resolve_history_date_col(history, preferred=args.run_date_col)
    result_col = _resolve_result_col(history)
    player_col = _resolve_player_col(history)
    target_col = _resolve_target_col(history)
    direction_col = _resolve_direction_col(history)
    label_col = str(args.label_col).strip() or "keep_label"

    working = _prepare_history_frame(
        history,
        date_col=date_col,
        label_col=label_col,
        result_col=result_col,
    )
    working = working.loc[working[label_col].notna() & working[result_col].isin(["win", "loss"])].copy()
    if working.empty:
        raise RuntimeError("No resolved win/loss accepted-pick rows available after preprocessing.")
    working = working.sort_values("_event_date").reset_index(drop=True)

    numeric_features, categorical_features = _feature_column_candidates(working)
    if not numeric_features and not categorical_features:
        raise RuntimeError("No usable features found for gate training.")

    model_family, logistic_cfg, catboost_cfg, uplift_cfg = _build_gate_model_configs(args)
    feature_hash = _feature_signature_hash(
        model_family=model_family,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    walk_cfg = WalkForwardConfig(
        train_window_days=int(args.train_window_days),
        test_window_days=int(args.test_window_days),
        step_days=int(args.step_days),
        min_train_rows=int(args.min_train_rows),
        min_test_rows=int(args.min_test_rows),
    )
    gate_policy = GatePolicyConfig(
        max_fire_rate=float(args.max_fire_rate),
        min_coverage_rate=float(args.min_coverage_rate),
        max_removed_per_day=int(args.max_removed_per_day),
        max_removed_per_player_per_day=int(args.max_removed_per_player_per_day),
        max_removed_per_segment_per_day=int(args.max_removed_per_segment_per_day),
        max_removed_per_target_per_day=int(args.max_removed_per_target_per_day),
        max_global_removed_per_player=int(args.max_global_removed_per_player),
        max_global_removed_share_per_player=float(args.max_global_removed_share_per_player),
        global_player_cap_min_veto_rows=int(args.global_player_cap_min_veto_rows),
        relax_player_global_cap_for_budget=bool(not args.disable_player_cap_relaxation),
        player_penalty_alpha=float(args.player_penalty_alpha),
        player_penalty_power=float(args.player_penalty_power),
        optimizer_enabled=bool(not args.disable_optimizer),
        optimizer_beam_width=int(args.optimizer_beam_width),
        optimizer_max_candidates_per_day=int(args.optimizer_max_candidates_per_day),
        optimizer_keep_prob_weight=float(args.optimizer_keep_prob_weight),
        optimizer_gap_weight=float(args.optimizer_gap_weight),
        optimizer_harm_weight=float(args.optimizer_harm_weight),
        optimizer_rank_weight=float(args.optimizer_rank_weight),
        tail_slots_only=int(args.tail_slots_only),
        min_veto_gap=float(args.min_veto_gap),
        require_keep_prob_filter=True,
    )
    criteria = PromotionCriteria(
        min_broad_profit_delta_units=float(args.min_broad_profit_delta_units),
        min_recent_profit_delta_units=float(args.min_recent_profit_delta_units),
        min_recent_hit_rate_delta_pp=float(args.min_recent_hit_rate_delta_pp),
        min_broad_hit_rate_delta_pp=float(args.min_broad_hit_rate_delta_pp),
        min_coverage_retention=float(args.min_coverage_retention),
        min_affected_share=float(args.min_affected_share),
        max_top_removed_player_share=float(args.max_top_removed_player_share),
        max_top_removed_segment_share=float(args.max_top_removed_segment_share),
        max_top_removed_target_share=float(args.max_top_removed_target_share),
        concentration_min_veto_rows=int(args.concentration_min_veto_rows),
        conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
        conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
        conditional_target_player_share_max=float(args.conditional_target_player_share_max),
        conditional_target_require_recent_non_negative=bool(
            not args.disable_conditional_target_require_recent_non_negative
        ),
        conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
        conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
        min_rolling_pass_rate=float(args.min_rolling_pass_rate),
        max_observed_fire_rate=float(args.max_observed_fire_rate),
        rolling_profit_delta_floor=float(args.rolling_profit_delta_floor),
        rolling_hit_rate_delta_floor_pp=float(args.rolling_hit_rate_delta_floor_pp),
    )

    quantiles = _parse_quantiles(args.threshold_quantiles)
    folds = iter_walk_forward_windows(working, date_col="_event_date", config=walk_cfg)
    fallback_split_used = False
    if not folds:
        split_idx = int(math.floor(0.70 * len(working)))
        min_train_fallback = 40
        min_test_fallback = 12
        if split_idx >= min_train_fallback and (len(working) - split_idx) >= min_test_fallback:
            train_mask = pd.Series(False, index=working.index, dtype=bool)
            test_mask = pd.Series(False, index=working.index, dtype=bool)
            train_mask.iloc[:split_idx] = True
            test_mask.iloc[split_idx:] = True
            train_start = str(working.loc[train_mask, "_event_date"].min().date())
            train_end = str(working.loc[train_mask, "_event_date"].max().date())
            test_start = str(working.loc[test_mask, "_event_date"].min().date())
            test_end = str(working.loc[test_mask, "_event_date"].max().date())
            folds = [
                {
                    "train_mask": train_mask,
                    "test_mask": test_mask,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_rows": int(train_mask.sum()),
                    "test_rows": int(test_mask.sum()),
                    "fallback_split": True,
                }
            ]
            fallback_split_used = True
        else:
            raise RuntimeError("No valid walk-forward folds were produced. Loosen window/min-row settings.")

    fold_rows: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    fold_thresholds: list[float] = []
    for fold_idx, fold in enumerate(folds):
        train = working.loc[fold["train_mask"]].copy()
        test = working.loc[fold["test_mask"]].copy()
        model = _new_gate_model(
            model_family,
            logistic_cfg=logistic_cfg,
            catboost_cfg=catboost_cfg,
            uplift_cfg=uplift_cfg,
        )
        model.fit_dataframe(
            train,
            label_col=label_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        train_keep_prob, train_harm = predict_keep_harm_scores(model, train)
        train["gate_keep_prob"] = pd.Series(train_keep_prob, index=train.index, dtype="float64")
        train["gate_harm_score"] = pd.Series(train_harm, index=train.index, dtype="float64")
        threshold, threshold_diag = _search_train_threshold(
            train,
            keep_prob_col="gate_keep_prob",
            date_col="_event_date",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            result_col=result_col,
            policy=gate_policy,
            quantiles=quantiles,
            threshold_max_candidates=int(args.threshold_max_candidates),
            segment_search_max_size=int(args.segment_search_max_size),
            segment_search_max_subsets=int(args.segment_search_max_subsets),
            segment_search_min_rows=int(args.segment_search_min_rows),
            segment_search_seed_limit=int(args.segment_search_seed_limit),
            cohort_search_enabled=bool(not args.disable_cohort_search),
            cohort_search_min_delta_pp=float(args.cohort_search_min_delta_pp),
            min_affected_share=float(args.train_threshold_min_affected_share),
            max_top_removed_player_share=float(args.train_threshold_max_top_removed_player_share),
            max_top_removed_segment_share=float(args.train_threshold_max_top_removed_segment_share),
            max_top_removed_target_share=float(args.train_threshold_max_top_removed_target_share),
            concentration_min_veto_rows=int(args.concentration_min_veto_rows),
            min_coverage_retention=float(args.train_threshold_min_coverage_retention),
            prefer_non_noop_when_close=bool(not args.disable_train_non_noop_exploration),
            non_noop_objective_margin=float(args.train_non_noop_objective_margin),
            objective_player_top_share_penalty=float(args.objective_player_top_share_penalty),
            objective_player_hhi_penalty=float(args.objective_player_hhi_penalty),
            objective_player_diversity_bonus=float(args.objective_player_diversity_bonus),
            conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
            conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
            conditional_target_player_share_max=float(args.conditional_target_player_share_max),
            conditional_target_require_recent_non_negative=bool(
                not args.disable_conditional_target_require_recent_non_negative
            ),
            conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
            conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
        )
        fold_thresholds.append(float(threshold))
        fold_allowed_segments = tuple(
            str(seg).upper().strip()
            for seg in threshold_diag.get("allowed_segments", [])
            if str(seg).strip()
        )
        fold_allowed_meta_cohorts = tuple(
            str(seg).upper().strip()
            for seg in threshold_diag.get("allowed_meta_cohorts", [])
            if str(seg).strip()
        )
        fold_policy = GatePolicyConfig(
            max_fire_rate=float(gate_policy.max_fire_rate),
            min_coverage_rate=float(gate_policy.min_coverage_rate),
            max_removed_per_day=int(gate_policy.max_removed_per_day),
            max_removed_per_player_per_day=int(gate_policy.max_removed_per_player_per_day),
            max_removed_per_segment_per_day=int(gate_policy.max_removed_per_segment_per_day),
            max_removed_per_target_per_day=int(gate_policy.max_removed_per_target_per_day),
            max_global_removed_per_player=int(
                _safe_int(threshold_diag.get("max_global_removed_per_player"), gate_policy.max_global_removed_per_player)
            ),
            max_global_removed_share_per_player=float(
                _safe_float(
                    threshold_diag.get("max_global_removed_share_per_player"),
                    gate_policy.max_global_removed_share_per_player,
                )
            ),
            global_player_cap_min_veto_rows=int(
                _safe_int(
                    threshold_diag.get("global_player_cap_min_veto_rows"),
                    gate_policy.global_player_cap_min_veto_rows,
                )
            ),
            relax_player_global_cap_for_budget=bool(
                threshold_diag.get(
                    "relax_player_global_cap_for_budget",
                    gate_policy.relax_player_global_cap_for_budget,
                )
            ),
            player_penalty_alpha=float(
                _safe_float(threshold_diag.get("player_penalty_alpha"), gate_policy.player_penalty_alpha)
            ),
            player_penalty_power=float(
                _safe_float(threshold_diag.get("player_penalty_power"), gate_policy.player_penalty_power)
            ),
            optimizer_enabled=bool(threshold_diag.get("optimizer_enabled", gate_policy.optimizer_enabled)),
            optimizer_beam_width=int(
                _safe_int(threshold_diag.get("optimizer_beam_width"), gate_policy.optimizer_beam_width)
            ),
            optimizer_max_candidates_per_day=int(
                _safe_int(
                    threshold_diag.get("optimizer_max_candidates_per_day"),
                    gate_policy.optimizer_max_candidates_per_day,
                )
            ),
            optimizer_keep_prob_weight=float(
                _safe_float(threshold_diag.get("optimizer_keep_prob_weight"), gate_policy.optimizer_keep_prob_weight)
            ),
            optimizer_gap_weight=float(
                _safe_float(threshold_diag.get("optimizer_gap_weight"), gate_policy.optimizer_gap_weight)
            ),
            optimizer_harm_weight=float(
                _safe_float(threshold_diag.get("optimizer_harm_weight"), gate_policy.optimizer_harm_weight)
            ),
            optimizer_rank_weight=float(
                _safe_float(threshold_diag.get("optimizer_rank_weight"), gate_policy.optimizer_rank_weight)
            ),
            segment_threshold_overrides={
                str(seg).upper().strip(): float(val)
                for seg, val in dict(threshold_diag.get("segment_threshold_overrides", gate_policy.segment_threshold_overrides) or {}).items()
                if str(seg).strip()
            },
            segment_budget_overrides={
                str(seg).upper().strip(): int(max(0, int(val)))
                for seg, val in dict(threshold_diag.get("segment_budget_overrides", gate_policy.segment_budget_overrides) or {}).items()
                if str(seg).strip()
            },
            tail_slots_only=int(_safe_int(threshold_diag.get("tail_slots_only"), gate_policy.tail_slots_only)),
            min_veto_gap=float(_safe_float(threshold_diag.get("min_veto_gap"), gate_policy.min_veto_gap)),
            require_keep_prob_filter=bool(threshold_diag.get("require_keep_prob_filter", gate_policy.require_keep_prob_filter)),
            allowed_segments=fold_allowed_segments,
            allowed_meta_cohorts=fold_allowed_meta_cohorts,
        )

        test_keep_prob, test_harm = predict_keep_harm_scores(model, test)
        test["gate_keep_prob"] = pd.Series(test_keep_prob, index=test.index, dtype="float64")
        test["gate_harm_score"] = pd.Series(test_harm, index=test.index, dtype="float64")
        scored = apply_shadow_gate_policy(
            test,
            keep_prob_col="gate_keep_prob",
            date_col="_event_date",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            threshold=float(threshold),
            policy=fold_policy,
        )
        scored["fold_index"] = int(fold_idx)
        scored["fold_threshold"] = float(threshold)
        scored["fold_allowed_segments"] = "||".join(fold_allowed_segments)
        scored["fold_allowed_meta_cohorts"] = "||".join(fold_allowed_meta_cohorts)
        fold_rows.append(scored)

        fold_summary = summarize_paired_outcomes(scored, veto_col="gate_veto", result_col=result_col)
        fold_reports.append(
            {
                "fold_index": int(fold_idx),
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "train_rows": int(fold.get("train_rows", len(train))),
                "test_rows": int(fold.get("test_rows", len(test))),
                "threshold": float(threshold),
                "allowed_segments": [str(v) for v in fold_allowed_segments],
                "allowed_meta_cohorts": [str(v) for v in fold_allowed_meta_cohorts],
                "threshold_diagnostics": threshold_diag,
                "summary": fold_summary,
            }
        )

    oof = pd.concat(fold_rows, ignore_index=True).sort_values("_event_date").reset_index(drop=True)
    broad_summary = summarize_paired_outcomes(oof, veto_col="gate_veto", result_col=result_col)
    max_date = oof["_event_date"].max()
    recent_start = max_date - pd.Timedelta(days=int(max(1, args.recent_days)) - 1)
    recent = oof.loc[oof["_event_date"] >= recent_start].copy()
    recent_summary = summarize_paired_outcomes(recent, veto_col="gate_veto", result_col=result_col)

    resolved = oof.loc[oof[result_col].isin(["win", "loss"])].copy()
    affected_share = _safe_float(resolved.get("gate_veto", pd.Series(dtype="float64")).mean(), 0.0)
    observed_fire_rate = affected_share
    concentration = concentration_stats(
        oof,
        veto_col="gate_veto",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
    )
    rolling = rolling_window_paired_deltas(
        oof,
        date_col="_event_date",
        veto_col="gate_veto",
        result_col=result_col,
        window_days=int(args.rolling_window_days),
        step_days=int(args.rolling_step_days),
    )
    recommendation = build_promotion_recommendation(
        broad_delta=broad_summary.get("delta", {}),
        recent_delta=recent_summary.get("delta", {}),
        affected_share=float(affected_share),
        concentration=concentration,
        rolling=rolling,
        observed_fire_rate=float(observed_fire_rate),
        criteria=criteria,
    )
    oof_lopo = _leave_one_player_out_validation(
        oof,
        player_col=player_col,
        veto_col="gate_veto",
        result_col=result_col,
        min_player_rows=int(args.lopo_min_player_rows),
        profit_floor=float(args.lopo_profit_floor),
        hit_floor_pp=float(args.lopo_hit_floor_pp),
    )
    oof_bootstrap = _bootstrap_paired_deltas(
        oof,
        veto_col="gate_veto",
        result_col=result_col,
        iterations=int(args.bootstrap_iterations),
        alpha=float(args.bootstrap_alpha),
        seed=int(args.bootstrap_seed),
    )
    recommendation = _apply_small_lift_validation_to_recommendation(
        recommendation,
        lopo=oof_lopo,
        bootstrap=oof_bootstrap,
        enforce=bool(not args.disable_small_lift_validation),
        min_lopo_profit_pass_rate=float(args.min_lopo_profit_pass_rate),
        min_lopo_hit_pass_rate=float(args.min_lopo_hit_pass_rate),
        min_bootstrap_profit_ci_lower=float(args.min_bootstrap_profit_ci_lower),
        min_bootstrap_hit_ci_lower_pp=float(args.min_bootstrap_hit_ci_lower_pp),
    )

    final_threshold, threshold_selection = _select_final_threshold_from_oof(
        oof=oof,
        keep_prob_col="gate_keep_prob",
        date_col="_event_date",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        policy=gate_policy,
        quantiles=quantiles,
        threshold_max_candidates=int(args.threshold_max_candidates),
        recent_days=int(args.threshold_recent_days),
        recent_hit_floor_pp=float(args.threshold_recent_hit_floor_pp),
        recent_profit_floor=float(args.threshold_recent_profit_floor),
        broad_profit_floor=float(args.threshold_broad_profit_floor),
        broad_hit_floor_pp=float(args.threshold_broad_hit_floor_pp),
        min_coverage_retention=float(args.threshold_min_coverage_retention),
        max_fire_rate=float(args.threshold_max_fire_rate),
        segment_search_max_size=int(args.segment_search_max_size),
        segment_search_max_subsets=int(args.segment_search_max_subsets),
        segment_search_min_rows=int(args.segment_search_min_rows),
        segment_search_seed_limit=int(args.segment_search_seed_limit),
        cohort_search_enabled=bool(not args.disable_cohort_search),
        cohort_search_min_delta_pp=float(args.cohort_search_min_delta_pp),
        min_affected_share=float(args.threshold_min_affected_share),
        max_top_removed_player_share=float(args.threshold_max_top_removed_player_share),
        max_top_removed_segment_share=float(args.threshold_max_top_removed_segment_share),
        max_top_removed_target_share=float(args.threshold_max_top_removed_target_share),
        concentration_min_veto_rows=int(args.concentration_min_veto_rows),
        prefer_relaxed_non_noop_when_close=bool(not args.disable_final_non_noop_exploration),
        relaxed_non_noop_objective_margin=float(args.final_non_noop_objective_margin),
        objective_player_top_share_penalty=float(args.objective_player_top_share_penalty),
        objective_player_hhi_penalty=float(args.objective_player_hhi_penalty),
        objective_player_diversity_bonus=float(args.objective_player_diversity_bonus),
        conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
        conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
        conditional_target_player_share_max=float(args.conditional_target_player_share_max),
        conditional_target_require_recent_non_negative=bool(
            not args.disable_conditional_target_require_recent_non_negative
        ),
        conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
        conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
    )
    selected_allowed_segments = tuple(
        str(seg).upper().strip()
        for seg in threshold_selection.get("allowed_segments", [])
        if str(seg).strip()
    )
    selected_allowed_meta_cohorts = tuple(
        str(seg).upper().strip()
        for seg in threshold_selection.get("allowed_meta_cohorts", [])
        if str(seg).strip()
    )
    final_gate_policy = GatePolicyConfig(
        max_fire_rate=float(gate_policy.max_fire_rate),
        min_coverage_rate=float(gate_policy.min_coverage_rate),
        max_removed_per_day=int(gate_policy.max_removed_per_day),
        max_removed_per_player_per_day=int(gate_policy.max_removed_per_player_per_day),
        max_removed_per_segment_per_day=int(gate_policy.max_removed_per_segment_per_day),
        max_removed_per_target_per_day=int(gate_policy.max_removed_per_target_per_day),
        max_global_removed_per_player=int(
            _safe_int(threshold_selection.get("max_global_removed_per_player"), gate_policy.max_global_removed_per_player)
        ),
        max_global_removed_share_per_player=float(
            _safe_float(
                threshold_selection.get("max_global_removed_share_per_player"),
                gate_policy.max_global_removed_share_per_player,
            )
        ),
        global_player_cap_min_veto_rows=int(
            _safe_int(
                threshold_selection.get("global_player_cap_min_veto_rows"),
                gate_policy.global_player_cap_min_veto_rows,
            )
        ),
        relax_player_global_cap_for_budget=bool(
            threshold_selection.get(
                "relax_player_global_cap_for_budget",
                gate_policy.relax_player_global_cap_for_budget,
            )
        ),
        player_penalty_alpha=float(
            _safe_float(threshold_selection.get("player_penalty_alpha"), gate_policy.player_penalty_alpha)
        ),
        player_penalty_power=float(
            _safe_float(threshold_selection.get("player_penalty_power"), gate_policy.player_penalty_power)
        ),
        optimizer_enabled=bool(threshold_selection.get("optimizer_enabled", gate_policy.optimizer_enabled)),
        optimizer_beam_width=int(
            _safe_int(threshold_selection.get("optimizer_beam_width"), gate_policy.optimizer_beam_width)
        ),
        optimizer_max_candidates_per_day=int(
            _safe_int(
                threshold_selection.get("optimizer_max_candidates_per_day"),
                gate_policy.optimizer_max_candidates_per_day,
            )
        ),
        optimizer_keep_prob_weight=float(
            _safe_float(
                threshold_selection.get("optimizer_keep_prob_weight"),
                gate_policy.optimizer_keep_prob_weight,
            )
        ),
        optimizer_gap_weight=float(
            _safe_float(threshold_selection.get("optimizer_gap_weight"), gate_policy.optimizer_gap_weight)
        ),
        optimizer_harm_weight=float(
            _safe_float(threshold_selection.get("optimizer_harm_weight"), gate_policy.optimizer_harm_weight)
        ),
        optimizer_rank_weight=float(
            _safe_float(threshold_selection.get("optimizer_rank_weight"), gate_policy.optimizer_rank_weight)
        ),
        segment_threshold_overrides={
            str(seg).upper().strip(): float(val)
            for seg, val in dict(
                threshold_selection.get("segment_threshold_overrides", gate_policy.segment_threshold_overrides) or {}
            ).items()
            if str(seg).strip()
        },
        segment_budget_overrides={
            str(seg).upper().strip(): int(max(0, int(val)))
            for seg, val in dict(
                threshold_selection.get("segment_budget_overrides", gate_policy.segment_budget_overrides) or {}
            ).items()
            if str(seg).strip()
        },
        tail_slots_only=int(_safe_int(threshold_selection.get("tail_slots_only"), gate_policy.tail_slots_only)),
        min_veto_gap=float(_safe_float(threshold_selection.get("min_veto_gap"), gate_policy.min_veto_gap)),
        require_keep_prob_filter=bool(threshold_selection.get("require_keep_prob_filter", gate_policy.require_keep_prob_filter)),
        allowed_segments=selected_allowed_segments,
        allowed_meta_cohorts=selected_allowed_meta_cohorts,
    )
    segment_override_result: dict[str, Any] | None = None
    if bool(not args.disable_segment_override_search):
        segment_override_result = _search_segment_override_on_oof(
            oof,
            keep_prob_col="gate_keep_prob",
            date_col="_event_date",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            result_col=result_col,
            base_threshold=float(final_threshold),
            base_policy=final_gate_policy,
            segment_key=str(args.segment_override_key),
            threshold_candidates=_parse_float_grid(
                str(args.segment_override_thresholds),
                default=[0.58, 0.60, 0.62, 0.64, 0.66],
                clip_min=0.0,
                clip_max=1.0,
            ),
            budget_candidates=_parse_int_grid(
                str(args.segment_override_budgets),
                default=[1, 2],
                min_value=1,
            ),
            recent_days=int(args.threshold_recent_days),
            recent_hit_floor_pp=float(args.threshold_recent_hit_floor_pp),
            recent_profit_floor=float(args.threshold_recent_profit_floor),
            broad_profit_floor=float(args.threshold_broad_profit_floor),
            broad_hit_floor_pp=float(args.threshold_broad_hit_floor_pp),
            min_coverage_retention=float(args.threshold_min_coverage_retention),
            max_fire_rate=float(args.threshold_max_fire_rate),
            min_affected_share=float(args.threshold_min_affected_share),
            max_top_removed_player_share=float(args.threshold_max_top_removed_player_share),
            max_top_removed_segment_share=float(args.threshold_max_top_removed_segment_share),
            max_top_removed_target_share=float(args.threshold_max_top_removed_target_share),
            concentration_min_veto_rows=int(args.concentration_min_veto_rows),
            conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
            conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
            conditional_target_player_share_max=float(args.conditional_target_player_share_max),
            conditional_target_require_recent_non_negative=bool(
                not args.disable_conditional_target_require_recent_non_negative
            ),
            conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
            conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
            objective_player_top_share_penalty=float(args.objective_player_top_share_penalty),
            objective_player_hhi_penalty=float(args.objective_player_hhi_penalty),
            objective_player_diversity_bonus=float(args.objective_player_diversity_bonus),
            min_objective_gain=float(args.segment_override_min_objective_gain),
            min_broad_profit_gain=float(args.segment_override_min_broad_profit_gain),
            min_recent_profit_gain=float(args.segment_override_min_recent_profit_gain),
            min_affected_share_gain=float(args.segment_override_min_affected_share_gain),
        )
    threshold_selection["segment_override_search"] = segment_override_result or {
        "selection_mode": "segment_override",
        "applied": False,
        "reason": "no_candidate_passed_or_disabled",
    }
    if segment_override_result and isinstance(segment_override_result.get("selected"), dict):
        selected_override = segment_override_result.get("selected", {})
        threshold_selection["segment_override_search"]["applied"] = True
        threshold_selection["segment_override_search"]["reason"] = "candidate_selected"
        threshold_selection["segment_override_search"]["selected"] = selected_override
        final_gate_policy = GatePolicyConfig(
            **{
                **asdict(final_gate_policy),
                "segment_threshold_overrides": {
                    str(seg).upper().strip(): float(val)
                    for seg, val in dict(selected_override.get("segment_threshold_overrides", {})).items()
                    if str(seg).strip()
                },
                "segment_budget_overrides": {
                    str(seg).upper().strip(): int(max(0, int(val)))
                    for seg, val in dict(selected_override.get("segment_budget_overrides", {})).items()
                    if str(seg).strip()
                },
            }
        )

    final_model = _new_gate_model(
        model_family,
        logistic_cfg=logistic_cfg,
        catboost_cfg=catboost_cfg,
        uplift_cfg=uplift_cfg,
    )
    final_model.fit_dataframe(
        working,
        label_col=label_col,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    full_scored = working.copy()
    full_keep_prob, full_harm = predict_keep_harm_scores(final_model, full_scored)
    full_scored["gate_keep_prob"] = pd.Series(full_keep_prob, index=full_scored.index, dtype="float64")
    full_scored["gate_harm_score"] = pd.Series(full_harm, index=full_scored.index, dtype="float64")
    fallback_base_policy = GatePolicyConfig(
        max_fire_rate=float(max(float(gate_policy.max_fire_rate), float(args.threshold_max_fire_rate))),
        min_coverage_rate=float(min(float(gate_policy.min_coverage_rate), float(args.threshold_min_coverage_retention))),
        max_removed_per_day=int(gate_policy.max_removed_per_day),
        max_removed_per_player_per_day=int(gate_policy.max_removed_per_player_per_day),
        max_removed_per_segment_per_day=int(gate_policy.max_removed_per_segment_per_day),
        max_removed_per_target_per_day=int(gate_policy.max_removed_per_target_per_day),
        max_global_removed_per_player=int(gate_policy.max_global_removed_per_player),
        max_global_removed_share_per_player=float(gate_policy.max_global_removed_share_per_player),
        global_player_cap_min_veto_rows=int(gate_policy.global_player_cap_min_veto_rows),
        relax_player_global_cap_for_budget=bool(gate_policy.relax_player_global_cap_for_budget),
        player_penalty_alpha=float(gate_policy.player_penalty_alpha),
        player_penalty_power=float(gate_policy.player_penalty_power),
        optimizer_enabled=bool(gate_policy.optimizer_enabled),
        optimizer_beam_width=int(gate_policy.optimizer_beam_width),
        optimizer_max_candidates_per_day=int(gate_policy.optimizer_max_candidates_per_day),
        optimizer_keep_prob_weight=float(gate_policy.optimizer_keep_prob_weight),
        optimizer_gap_weight=float(gate_policy.optimizer_gap_weight),
        optimizer_harm_weight=float(gate_policy.optimizer_harm_weight),
        optimizer_rank_weight=float(gate_policy.optimizer_rank_weight),
        segment_threshold_overrides=dict(gate_policy.segment_threshold_overrides or {}),
        segment_budget_overrides=dict(gate_policy.segment_budget_overrides or {}),
        tail_slots_only=0,
        min_veto_gap=0.0,
        require_keep_prob_filter=False,
    )
    segment_override_applied = bool(
        isinstance(threshold_selection.get("segment_override_search"), dict)
        and threshold_selection.get("segment_override_search", {}).get("applied", False)
    )
    fullframe_fallback_triggered = bool(
        (
            float(final_threshold) <= -0.999999
            or not bool(threshold_selection.get("constraints_passed", False))
        )
        and not bool(segment_override_applied)
    )
    if bool(not args.disable_fullframe_cohort_fallback) and fullframe_fallback_triggered:
        fallback_selection = _select_fullframe_cohort_fallback(
            full_scored,
            keep_prob_col="gate_keep_prob",
            date_col="_event_date",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            result_col=result_col,
            base_policy=fallback_base_policy,
            recent_days=int(args.threshold_recent_days),
            recent_hit_floor_pp=float(args.threshold_recent_hit_floor_pp),
            recent_profit_floor=float(args.threshold_recent_profit_floor),
            broad_profit_floor=float(args.threshold_broad_profit_floor),
            broad_hit_floor_pp=float(args.threshold_broad_hit_floor_pp),
            min_coverage_retention=float(args.threshold_min_coverage_retention),
            max_fire_rate=float(args.threshold_max_fire_rate),
            segment_search_max_size=int(args.segment_search_max_size),
            segment_search_max_subsets=int(args.segment_search_max_subsets),
            segment_search_min_rows=int(args.segment_search_min_rows),
            segment_search_seed_limit=int(args.segment_search_seed_limit),
            cohort_search_min_delta_pp=float(args.cohort_search_min_delta_pp),
            min_affected_share=float(args.threshold_min_affected_share),
            max_top_removed_player_share=float(args.threshold_max_top_removed_player_share),
            max_top_removed_segment_share=float(args.threshold_max_top_removed_segment_share),
            max_top_removed_target_share=float(args.threshold_max_top_removed_target_share),
            concentration_min_veto_rows=int(args.concentration_min_veto_rows),
            conditional_target_cap_enabled=bool(not args.disable_conditional_target_cap),
            conditional_target_relaxed_cap=float(args.conditional_target_relaxed_cap),
            conditional_target_player_share_max=float(args.conditional_target_player_share_max),
            conditional_target_require_recent_non_negative=bool(
                not args.disable_conditional_target_require_recent_non_negative
            ),
            conditional_target_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
            conditional_target_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
            objective_player_top_share_penalty=float(args.objective_player_top_share_penalty),
            objective_player_hhi_penalty=float(args.objective_player_hhi_penalty),
            objective_player_diversity_bonus=float(args.objective_player_diversity_bonus),
        )
        if fallback_selection is not None:
            threshold_selection["oof_selection"] = dict(threshold_selection)
            threshold_selection = fallback_selection
            final_threshold = float(threshold_selection.get("threshold", -1.0))
            selected_allowed_segments = tuple()
            selected_allowed_meta_cohorts = tuple(
                str(seg).upper().strip()
                for seg in threshold_selection.get("allowed_meta_cohorts", [])
                if str(seg).strip()
            )
            final_gate_policy = GatePolicyConfig(
                max_fire_rate=float(gate_policy.max_fire_rate),
                min_coverage_rate=float(gate_policy.min_coverage_rate),
                max_removed_per_day=int(gate_policy.max_removed_per_day),
                max_removed_per_player_per_day=int(gate_policy.max_removed_per_player_per_day),
                max_removed_per_segment_per_day=int(gate_policy.max_removed_per_segment_per_day),
                max_removed_per_target_per_day=int(gate_policy.max_removed_per_target_per_day),
                max_global_removed_per_player=int(
                    _safe_int(threshold_selection.get("max_global_removed_per_player"), gate_policy.max_global_removed_per_player)
                ),
                max_global_removed_share_per_player=float(
                    _safe_float(
                        threshold_selection.get("max_global_removed_share_per_player"),
                        gate_policy.max_global_removed_share_per_player,
                    )
                ),
                global_player_cap_min_veto_rows=int(
                    _safe_int(
                        threshold_selection.get("global_player_cap_min_veto_rows"),
                        gate_policy.global_player_cap_min_veto_rows,
                    )
                ),
                relax_player_global_cap_for_budget=bool(
                    threshold_selection.get(
                        "relax_player_global_cap_for_budget",
                        gate_policy.relax_player_global_cap_for_budget,
                    )
                ),
                player_penalty_alpha=float(
                    _safe_float(threshold_selection.get("player_penalty_alpha"), gate_policy.player_penalty_alpha)
                ),
                player_penalty_power=float(
                    _safe_float(threshold_selection.get("player_penalty_power"), gate_policy.player_penalty_power)
                ),
                optimizer_enabled=bool(threshold_selection.get("optimizer_enabled", gate_policy.optimizer_enabled)),
                optimizer_beam_width=int(
                    _safe_int(threshold_selection.get("optimizer_beam_width"), gate_policy.optimizer_beam_width)
                ),
                optimizer_max_candidates_per_day=int(
                    _safe_int(
                        threshold_selection.get("optimizer_max_candidates_per_day"),
                        gate_policy.optimizer_max_candidates_per_day,
                    )
                ),
                optimizer_keep_prob_weight=float(
                    _safe_float(
                        threshold_selection.get("optimizer_keep_prob_weight"),
                        gate_policy.optimizer_keep_prob_weight,
                    )
                ),
                optimizer_gap_weight=float(
                    _safe_float(threshold_selection.get("optimizer_gap_weight"), gate_policy.optimizer_gap_weight)
                ),
                optimizer_harm_weight=float(
                    _safe_float(threshold_selection.get("optimizer_harm_weight"), gate_policy.optimizer_harm_weight)
                ),
                optimizer_rank_weight=float(
                    _safe_float(threshold_selection.get("optimizer_rank_weight"), gate_policy.optimizer_rank_weight)
                ),
                segment_threshold_overrides={
                    str(seg).upper().strip(): float(val)
                    for seg, val in dict(
                        threshold_selection.get("segment_threshold_overrides", gate_policy.segment_threshold_overrides) or {}
                    ).items()
                    if str(seg).strip()
                },
                segment_budget_overrides={
                    str(seg).upper().strip(): int(max(0, int(val)))
                    for seg, val in dict(
                        threshold_selection.get("segment_budget_overrides", gate_policy.segment_budget_overrides) or {}
                    ).items()
                    if str(seg).strip()
                },
                tail_slots_only=0,
                min_veto_gap=0.0,
                require_keep_prob_filter=False,
                allowed_segments=selected_allowed_segments,
                allowed_meta_cohorts=selected_allowed_meta_cohorts,
            )
    full_scored_candidate = apply_shadow_gate_policy(
        full_scored,
        keep_prob_col="gate_keep_prob",
        date_col="_event_date",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
        threshold=float(final_threshold),
        policy=final_gate_policy,
    )
    guard_broad_summary = summarize_paired_outcomes(full_scored_candidate, veto_col="gate_veto", result_col=result_col)
    guard_max_date = _coerce_event_datetime(full_scored_candidate.get("_event_date")).max()
    guard_recent_start = (
        guard_max_date - pd.Timedelta(days=int(max(1, args.threshold_recent_days)) - 1)
        if pd.notna(guard_max_date)
        else pd.NaT
    )
    if pd.notna(guard_recent_start):
        guard_recent_frame = full_scored_candidate.loc[_coerce_event_datetime(full_scored_candidate.get("_event_date")) >= guard_recent_start].copy()
    else:
        guard_recent_frame = full_scored_candidate.iloc[0:0].copy()
    guard_recent_summary = summarize_paired_outcomes(guard_recent_frame, veto_col="gate_veto", result_col=result_col)
    guard_broad_delta = guard_broad_summary.get("delta", {})
    guard_recent_delta = guard_recent_summary.get("delta", {})
    guard_broad_profit = _safe_float(guard_broad_delta.get("profit_units"), float("-inf"))
    guard_broad_hit = _safe_float(guard_broad_delta.get("hit_rate_pp"), float("-inf"))
    guard_recent_profit = _safe_float(guard_recent_delta.get("profit_units"), float("-inf"))
    guard_recent_hit = _safe_float(guard_recent_delta.get("hit_rate_pp"), float("-inf"))
    guard_resolved = full_scored_candidate.loc[full_scored_candidate[result_col].isin(["win", "loss"])].copy()
    guard_affected_share = _safe_float(pd.to_numeric(guard_resolved.get("gate_veto"), errors="coerce").fillna(0).mean(), 0.0)
    guard_observed_fire_rate = float(guard_affected_share)
    guard_base_resolved = _safe_float(guard_broad_summary.get("baseline", {}).get("resolved"), 0.0)
    guard_gated_resolved = _safe_float(guard_broad_summary.get("gated", {}).get("resolved"), 0.0)
    guard_coverage_retention = float(guard_gated_resolved / guard_base_resolved) if guard_base_resolved > 0.0 else 1.0
    guard_concentration = concentration_stats(
        full_scored_candidate,
        veto_col="gate_veto",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
    )
    guard_top_player_share = _safe_float(guard_concentration.get("top_player_share"), 0.0)
    guard_top_segment_share = _safe_float(guard_concentration.get("top_segment_share"), 0.0)
    guard_top_target_share = _safe_float(guard_concentration.get("top_target_share"), 0.0)
    guard_veto_rows = int(max(0, _safe_int(guard_concentration.get("veto_rows"), 0)))
    guard_concentration_enforced = bool(guard_veto_rows >= int(max(1, args.concentration_min_veto_rows)))
    guard_effective_target_cap, guard_conditional_target_cap_active = _effective_target_cap(
        base_cap=float(args.threshold_max_top_removed_target_share),
        top_player_share=float(guard_top_player_share),
        recent_profit=float(guard_recent_profit),
        recent_hit_pp=float(guard_recent_hit),
        conditional_enabled=bool(not args.disable_conditional_target_cap),
        conditional_relaxed_cap=float(args.conditional_target_relaxed_cap),
        conditional_player_share_max=float(args.conditional_target_player_share_max),
        conditional_require_recent_non_negative=bool(
            not args.disable_conditional_target_require_recent_non_negative
        ),
        conditional_recent_profit_floor=float(args.conditional_target_recent_profit_floor),
        conditional_recent_hit_floor_pp=float(args.conditional_target_recent_hit_floor_pp),
    )
    guard_effective_player_cap = float(args.threshold_max_top_removed_player_share if guard_concentration_enforced else 1.0)
    guard_effective_segment_cap = float(args.threshold_max_top_removed_segment_share if guard_concentration_enforced else 1.0)
    guard_effective_target_cap = float(guard_effective_target_cap if guard_concentration_enforced else 1.0)
    full_fit_guard_pass = bool(
        guard_broad_profit >= float(args.threshold_broad_profit_floor)
        and guard_broad_hit >= float(args.threshold_broad_hit_floor_pp)
        and guard_recent_profit >= float(args.threshold_recent_profit_floor)
        and guard_recent_hit >= float(args.threshold_recent_hit_floor_pp)
        and guard_coverage_retention >= float(args.threshold_min_coverage_retention)
        and guard_observed_fire_rate <= float(args.threshold_max_fire_rate)
        and guard_affected_share >= float(args.threshold_min_affected_share)
        and guard_top_player_share <= float(guard_effective_player_cap)
        and guard_top_segment_share <= float(guard_effective_segment_cap)
        and guard_top_target_share <= float(guard_effective_target_cap)
    )
    threshold_selection["full_fit_guard"] = {
        "pass": bool(full_fit_guard_pass),
        "broad_profit_units": float(guard_broad_profit),
        "broad_hit_rate_pp": float(guard_broad_hit),
        "recent_profit_units": float(guard_recent_profit),
        "recent_hit_rate_pp": float(guard_recent_hit),
        "coverage_retention": float(guard_coverage_retention),
        "observed_fire_rate": float(guard_observed_fire_rate),
        "affected_share": float(guard_affected_share),
        "top_player_share": float(guard_top_player_share),
        "top_segment_share": float(guard_top_segment_share),
        "top_target_share": float(guard_top_target_share),
        "veto_rows": int(guard_veto_rows),
        "concentration_enforced": bool(guard_concentration_enforced),
        "effective_player_cap": float(guard_effective_player_cap),
        "effective_segment_cap": float(guard_effective_segment_cap),
        "effective_target_cap": float(guard_effective_target_cap),
        "conditional_target_cap_active": bool(guard_conditional_target_cap_active),
        "recent_days": int(max(1, args.threshold_recent_days)),
    }
    if not bool(full_fit_guard_pass) and float(final_threshold) > -0.999999:
        final_threshold = -1.0
        selected_allowed_segments = tuple()
        selected_allowed_meta_cohorts = tuple()
        final_gate_policy = GatePolicyConfig(
            max_fire_rate=float(gate_policy.max_fire_rate),
            min_coverage_rate=float(gate_policy.min_coverage_rate),
            max_removed_per_day=int(gate_policy.max_removed_per_day),
            max_removed_per_player_per_day=int(gate_policy.max_removed_per_player_per_day),
            max_removed_per_segment_per_day=int(gate_policy.max_removed_per_segment_per_day),
            max_removed_per_target_per_day=int(gate_policy.max_removed_per_target_per_day),
            max_global_removed_per_player=int(gate_policy.max_global_removed_per_player),
            max_global_removed_share_per_player=float(gate_policy.max_global_removed_share_per_player),
            global_player_cap_min_veto_rows=int(gate_policy.global_player_cap_min_veto_rows),
            relax_player_global_cap_for_budget=bool(gate_policy.relax_player_global_cap_for_budget),
            player_penalty_alpha=float(gate_policy.player_penalty_alpha),
            player_penalty_power=float(gate_policy.player_penalty_power),
            optimizer_enabled=bool(gate_policy.optimizer_enabled),
            optimizer_beam_width=int(gate_policy.optimizer_beam_width),
            optimizer_max_candidates_per_day=int(gate_policy.optimizer_max_candidates_per_day),
            optimizer_keep_prob_weight=float(gate_policy.optimizer_keep_prob_weight),
            optimizer_gap_weight=float(gate_policy.optimizer_gap_weight),
            optimizer_harm_weight=float(gate_policy.optimizer_harm_weight),
            optimizer_rank_weight=float(gate_policy.optimizer_rank_weight),
            segment_threshold_overrides={},
            segment_budget_overrides={},
            tail_slots_only=int(gate_policy.tail_slots_only),
            min_veto_gap=float(gate_policy.min_veto_gap),
            require_keep_prob_filter=True,
            allowed_segments=selected_allowed_segments,
            allowed_meta_cohorts=selected_allowed_meta_cohorts,
        )
        threshold_selection["selection_mode"] = "full_fit_guard_no_op_fallback"
        threshold_selection["reason"] = "full_fit_guard_failed_no_harm_fallback"
        threshold_selection["threshold"] = float(final_threshold)
        threshold_selection["allowed_segments"] = []
        threshold_selection["allowed_meta_cohorts"] = []
        threshold_selection["segment_count"] = 0
        threshold_selection["meta_cohort_count"] = 0
        threshold_selection["full_fit_guard"]["fallback_applied"] = True
        full_scored = apply_shadow_gate_policy(
            full_scored,
            keep_prob_col="gate_keep_prob",
            date_col="_event_date",
            player_col=player_col,
            target_col=target_col,
            direction_col=direction_col,
            threshold=float(final_threshold),
            policy=final_gate_policy,
        )
    else:
        threshold_selection["full_fit_guard"]["fallback_applied"] = False
        full_scored = full_scored_candidate
    full_scored["fold_index"] = -1
    full_scored["fold_threshold"] = float(final_threshold)

    final_broad_summary = summarize_paired_outcomes(full_scored, veto_col="gate_veto", result_col=result_col)
    final_max_date = _coerce_event_datetime(full_scored.get("_event_date")).max()
    final_recent_start = final_max_date - pd.Timedelta(days=int(max(1, args.recent_days)) - 1) if pd.notna(final_max_date) else pd.NaT
    if pd.notna(final_recent_start):
        final_recent = full_scored.loc[_coerce_event_datetime(full_scored.get("_event_date")) >= final_recent_start].copy()
    else:
        final_recent = full_scored.iloc[0:0].copy()
    final_recent_summary = summarize_paired_outcomes(final_recent, veto_col="gate_veto", result_col=result_col)
    final_resolved = full_scored.loc[full_scored[result_col].isin(["win", "loss"])].copy()
    final_affected_share = _safe_float(final_resolved.get("gate_veto", pd.Series(dtype="float64")).mean(), 0.0)
    final_observed_fire_rate = float(final_affected_share)
    final_concentration = concentration_stats(
        full_scored,
        veto_col="gate_veto",
        player_col=player_col,
        target_col=target_col,
        direction_col=direction_col,
    )
    final_rolling = rolling_window_paired_deltas(
        full_scored,
        date_col="_event_date",
        veto_col="gate_veto",
        result_col=result_col,
        window_days=int(args.rolling_window_days),
        step_days=int(args.rolling_step_days),
    )
    final_recommendation = build_promotion_recommendation(
        broad_delta=final_broad_summary.get("delta", {}),
        recent_delta=final_recent_summary.get("delta", {}),
        affected_share=float(final_affected_share),
        concentration=final_concentration,
        rolling=final_rolling,
        observed_fire_rate=float(final_observed_fire_rate),
        criteria=criteria,
    )
    final_lopo = _leave_one_player_out_validation(
        full_scored,
        player_col=player_col,
        veto_col="gate_veto",
        result_col=result_col,
        min_player_rows=int(args.lopo_min_player_rows),
        profit_floor=float(args.lopo_profit_floor),
        hit_floor_pp=float(args.lopo_hit_floor_pp),
    )
    final_bootstrap = _bootstrap_paired_deltas(
        full_scored,
        veto_col="gate_veto",
        result_col=result_col,
        iterations=int(args.bootstrap_iterations),
        alpha=float(args.bootstrap_alpha),
        seed=int(args.bootstrap_seed) + 1,
    )
    final_recommendation = _apply_small_lift_validation_to_recommendation(
        final_recommendation,
        lopo=final_lopo,
        bootstrap=final_bootstrap,
        enforce=bool(not args.disable_small_lift_validation),
        min_lopo_profit_pass_rate=float(args.min_lopo_profit_pass_rate),
        min_lopo_hit_pass_rate=float(args.min_lopo_hit_pass_rate),
        min_bootstrap_profit_ci_lower=float(args.min_bootstrap_profit_ci_lower),
        min_bootstrap_hit_ci_lower_pp=float(args.min_bootstrap_hit_ci_lower_pp),
    )
    train_start_ts = _coerce_event_datetime(working.get("_event_date")).min()
    train_end_ts = _coerce_event_datetime(working.get("_event_date")).max()
    segment_deltas_broad = _segment_paired_deltas(
        full_scored,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        veto_col="gate_veto",
        min_resolved_rows=10,
    )
    segment_deltas_recent = _segment_paired_deltas(
        final_recent,
        target_col=target_col,
        direction_col=direction_col,
        result_col=result_col,
        veto_col="gate_veto",
        min_resolved_rows=5,
    )

    model_payload = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "mode": "shadow_candidate",
        "model_family": str(model_family),
        "feature_hash": str(feature_hash),
        "label_column": label_col,
        "result_column": result_col,
        "date_column": date_col,
        "player_column": player_col,
        "target_column": target_col,
        "direction_column": direction_col,
        "threshold": float(final_threshold),
        "threshold_source": "oof_constrained_selection_or_noop",
        "gate_policy": asdict(final_gate_policy),
        "logistic_config": asdict(logistic_cfg) if logistic_cfg is not None else None,
        "catboost_config": asdict(catboost_cfg) if catboost_cfg is not None else None,
        "uplift_config": asdict(uplift_cfg) if uplift_cfg is not None else None,
        "walk_forward_config": asdict(walk_cfg),
        "promotion_criteria": asdict(criteria),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model": final_model.to_dict(),
        "shadow_only": True,
        "live_ready": bool(final_recommendation.get("pass", False)),
        "notes": "Candidate gate artifact; does not mutate production policy automatically.",
    }

    report_payload = {
        "version": 1,
        "created_at_utc": _utc_now_iso(),
        "model_family": str(model_family),
        "feature_hash": str(feature_hash),
        "logistic_config": asdict(logistic_cfg) if logistic_cfg is not None else None,
        "catboost_config": asdict(catboost_cfg) if catboost_cfg is not None else None,
        "uplift_config": asdict(uplift_cfg) if uplift_cfg is not None else None,
        "history_csv": str(history_csv),
        "rows_total": int(len(history)),
        "rows_resolved_trainable": int(len(working)),
        "date_col_resolved": date_col,
        "label_col": label_col,
        "result_col": result_col,
        "feature_counts": {
            "numeric": int(len(numeric_features)),
            "categorical": int(len(categorical_features)),
        },
        "walk_forward_folds": int(len(fold_reports)),
        "walk_forward_fallback_split_used": bool(fallback_split_used),
        "fold_thresholds": [float(v) for v in fold_thresholds],
        "final_threshold_selection": threshold_selection,
        "threshold_constraints": {
            "broad_profit_floor": float(args.threshold_broad_profit_floor),
            "broad_hit_floor_pp": float(args.threshold_broad_hit_floor_pp),
            "recent_days": int(args.threshold_recent_days),
            "threshold_max_candidates": int(args.threshold_max_candidates),
            "recent_profit_floor": float(args.threshold_recent_profit_floor),
            "recent_hit_floor_pp": float(args.threshold_recent_hit_floor_pp),
            "min_coverage_retention": float(args.threshold_min_coverage_retention),
            "max_fire_rate": float(args.threshold_max_fire_rate),
            "segment_search_max_size": int(args.segment_search_max_size),
            "segment_search_max_subsets": int(args.segment_search_max_subsets),
            "segment_search_min_rows": int(args.segment_search_min_rows),
            "segment_search_seed_limit": int(args.segment_search_seed_limit),
            "cohort_search_enabled": bool(not args.disable_cohort_search),
            "cohort_search_min_delta_pp": float(args.cohort_search_min_delta_pp),
            "fullframe_cohort_fallback_enabled": bool(not args.disable_fullframe_cohort_fallback),
            "min_affected_share": float(args.threshold_min_affected_share),
            "max_top_removed_player_share": float(args.threshold_max_top_removed_player_share),
            "max_top_removed_segment_share": float(args.threshold_max_top_removed_segment_share),
            "max_top_removed_target_share": float(args.threshold_max_top_removed_target_share),
            "concentration_min_veto_rows": int(args.concentration_min_veto_rows),
            "max_global_removed_per_player": int(args.max_global_removed_per_player),
            "max_global_removed_share_per_player": float(args.max_global_removed_share_per_player),
            "global_player_cap_min_veto_rows": int(args.global_player_cap_min_veto_rows),
            "player_cap_relaxation_enabled": bool(not args.disable_player_cap_relaxation),
            "player_penalty_alpha": float(args.player_penalty_alpha),
            "player_penalty_power": float(args.player_penalty_power),
            "optimizer_enabled": bool(not args.disable_optimizer),
            "optimizer_beam_width": int(args.optimizer_beam_width),
            "optimizer_max_candidates_per_day": int(args.optimizer_max_candidates_per_day),
            "optimizer_keep_prob_weight": float(args.optimizer_keep_prob_weight),
            "optimizer_gap_weight": float(args.optimizer_gap_weight),
            "optimizer_harm_weight": float(args.optimizer_harm_weight),
            "optimizer_rank_weight": float(args.optimizer_rank_weight),
            "objective_player_top_share_penalty": float(args.objective_player_top_share_penalty),
            "objective_player_hhi_penalty": float(args.objective_player_hhi_penalty),
            "objective_player_diversity_bonus": float(args.objective_player_diversity_bonus),
            "conditional_target_cap_enabled": bool(not args.disable_conditional_target_cap),
            "conditional_target_relaxed_cap": float(args.conditional_target_relaxed_cap),
            "conditional_target_player_share_max": float(args.conditional_target_player_share_max),
            "conditional_target_require_recent_non_negative": bool(
                not args.disable_conditional_target_require_recent_non_negative
            ),
            "conditional_target_recent_hit_floor_pp": float(args.conditional_target_recent_hit_floor_pp),
            "conditional_target_recent_profit_floor": float(args.conditional_target_recent_profit_floor),
            "final_non_noop_exploration_enabled": bool(not args.disable_final_non_noop_exploration),
            "final_non_noop_objective_margin": float(args.final_non_noop_objective_margin),
            "train_non_noop_exploration_enabled": bool(not args.disable_train_non_noop_exploration),
            "train_non_noop_objective_margin": float(args.train_non_noop_objective_margin),
            "segment_override_search_enabled": bool(not args.disable_segment_override_search),
            "segment_override_key": str(args.segment_override_key),
            "segment_override_thresholds": _parse_float_grid(
                str(args.segment_override_thresholds),
                default=[0.58, 0.60, 0.62, 0.64, 0.66],
                clip_min=0.0,
                clip_max=1.0,
            ),
            "segment_override_budgets": _parse_int_grid(
                str(args.segment_override_budgets),
                default=[1, 2],
                min_value=1,
            ),
            "segment_override_min_objective_gain": float(args.segment_override_min_objective_gain),
            "segment_override_min_broad_profit_gain": float(args.segment_override_min_broad_profit_gain),
            "segment_override_min_recent_profit_gain": float(args.segment_override_min_recent_profit_gain),
            "segment_override_min_affected_share_gain": float(args.segment_override_min_affected_share_gain),
            "small_lift_validation_enabled": bool(not args.disable_small_lift_validation),
            "lopo_min_player_rows": int(args.lopo_min_player_rows),
            "lopo_profit_floor": float(args.lopo_profit_floor),
            "lopo_hit_floor_pp": float(args.lopo_hit_floor_pp),
            "min_lopo_profit_pass_rate": float(args.min_lopo_profit_pass_rate),
            "min_lopo_hit_pass_rate": float(args.min_lopo_hit_pass_rate),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_alpha": float(args.bootstrap_alpha),
            "bootstrap_seed": int(args.bootstrap_seed),
            "min_bootstrap_profit_ci_lower": float(args.min_bootstrap_profit_ci_lower),
            "min_bootstrap_hit_ci_lower_pp": float(args.min_bootstrap_hit_ci_lower_pp),
        },
        "broad_summary": final_broad_summary,
        "recent_summary": final_recent_summary,
        "oof_broad_summary": broad_summary,
        "oof_recent_summary": recent_summary,
        "recent_window": {
            "days": int(args.recent_days),
            "start": str(final_recent_start.date()) if pd.notna(final_recent_start) else "",
            "end": str(final_max_date.date()) if pd.notna(final_max_date) else "",
            "rows": int(len(final_recent)),
        },
        "affected_share": float(final_affected_share),
        "observed_fire_rate": float(final_observed_fire_rate),
        "concentration": final_concentration,
        "rolling_window": {
            "days": int(args.rolling_window_days),
            "step_days": int(args.rolling_step_days),
            "rows": int(len(final_rolling)),
        },
        "promotion_recommendation": final_recommendation,
        "oof_promotion_recommendation": recommendation,
        "fold_reports": fold_reports,
        "model_threshold": float(final_threshold),
        "training_window": {
            "start": str(train_start_ts.date()) if pd.notna(train_start_ts) else "",
            "end": str(train_end_ts.date()) if pd.notna(train_end_ts) else "",
            "resolved_rows": int(len(working)),
        },
        "segment_level_deltas": {
            "broad": segment_deltas_broad,
            "recent": segment_deltas_recent,
        },
    }

    model_out = args.model_out.resolve()
    report_out = args.report_out.resolve()
    scored_rows_out = args.scored_rows_out.resolve()
    model_id = f"{str(model_family)}_{str(feature_hash)}_{model_out.stem}"
    model_payload["model_id"] = str(model_id)
    report_payload["model_id"] = str(model_id)
    report_payload["artifacts"] = {
        "model_json": str(model_out),
        "report_json": str(report_out),
        "scored_rows_csv": str(scored_rows_out),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    scored_rows_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(model_out, model_payload)
    _write_json(report_out, report_payload)

    export_rows = full_scored.copy()
    export_rows["event_date"] = pd.to_datetime(export_rows["_event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export_rows.to_csv(scored_rows_out, index=False)
    if not rolling.empty:
        rolling_out = scored_rows_out.with_name(scored_rows_out.stem + "_rolling.csv")
        rolling.to_csv(rolling_out, index=False)

    print("Accepted-pick gate shadow training complete.")
    print(f"Model family:           {model_family}")
    print(f"History rows (resolved): {len(working)}")
    print(f"Walk-forward folds:      {len(fold_reports)}")
    print(f"Broad delta hit-rate:    {_safe_float(final_broad_summary.get('delta', {}).get('hit_rate_pp'), np.nan):.4f} pp")
    print(f"Broad delta EV/resolved: {_safe_float(final_broad_summary.get('delta', {}).get('ev_per_resolved'), np.nan):.6f}")
    print(f"Promotion pass:          {bool(final_recommendation.get('pass', False))}")
    print(f"Candidate model JSON:    {model_out}")
    print(f"Training report JSON:    {report_out}")
    print(f"Scored rows CSV:         {scored_rows_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accepted-pick gate research orchestration (shadow-only).")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="Freeze daily artifacts and build repeat-run stability features.")
    snap.add_argument("--run-date", type=str, default=None, help="Run date token (YYYY-MM-DD or parseable).")
    snap.add_argument("--run-dir", type=Path, default=None, help="Explicit daily run directory (YYYYMMDD folder).")
    snap.add_argument("--manifest", type=Path, default=None, help="Explicit daily pipeline manifest path.")
    snap.add_argument("--daily-runs-dir", type=Path, default=REPO_ROOT / "model" / "analysis" / "daily_runs")
    snap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    snap.add_argument("--repeat-runs", type=int, default=25, help="Repeat-board count when --seed-list is omitted.")
    snap.add_argument("--seed-start", type=int, default=17, help="Start seed for repeat runs.")
    snap.add_argument("--seed-step", type=int, default=37, help="Seed step for repeat runs.")
    snap.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="Comma-separated explicit seed list (overrides --repeat-runs/--seed-start/--seed-step).",
    )
    snap.add_argument("--skip-repeats", action="store_true", help="Skip repeat-board generation.")
    snap.add_argument("--skip-shadow", action="store_true", help="Skip frozen-shadow artifact capture.")
    snap.add_argument(
        "--shadow-policy-profile",
        type=str,
        default=None,
        help="Optional preferred shadow policy profile (falls back to first shadow run).",
    )
    snap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing snapshot directory.")
    snap.set_defaults(func=cmd_snapshot)

    app = sub.add_parser("append-settled", help="Append settled outcomes from a snapshot into accepted-pick history.")
    app.add_argument("--snapshot-manifest", type=Path, default=None, help="Path to snapshot_manifest.json")
    app.add_argument("--snapshot-dir", type=Path, default=None, help="Path to snapshot directory (contains snapshot_manifest.json).")
    app.add_argument("--run-date", type=str, default=None, help="Resolve snapshot by run date via output-root/snapshots/YYYYMMDD.")
    app.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    app.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    app.set_defaults(func=cmd_append_settled)

    prep = sub.add_parser(
        "prepare-replay-history",
        help="Convert deterministic replay rows into accepted-pick history format for policy-aligned training.",
    )
    prep.add_argument("--rows-csv", type=Path, required=True, help="Replay rows CSV (for example validate_board_objective_mode rows output).")
    prep.add_argument("--output-csv", type=Path, default=DEFAULT_REPLAY_HISTORY_CSV)
    prep.add_argument("--report-out", type=Path, default=DEFAULT_REPLAY_HISTORY_REPORT_JSON)
    prep.add_argument("--mode", type=str, default="board_objective")
    prep.add_argument("--date-col", type=str, default="market_date")
    prep.add_argument("--result-col", type=str, default="result")
    prep.add_argument("--start-date", type=str, default=None)
    prep.add_argument("--end-date", type=str, default=None)
    prep.add_argument("--append", action="store_true", help="Append and de-duplicate against existing output history CSV.")
    prep.set_defaults(func=cmd_prepare_replay_history)

    pe = sub.add_parser(
        "paired-eval",
        help="Run deterministic paired evaluation on a fixed replay frame using accepted-pick veto tags.",
    )
    pe.add_argument("--rows-csv", type=Path, required=True)
    pe.add_argument("--gate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    pe.add_argument("--summary-out", type=Path, default=DEFAULT_PAIRED_EVAL_JSON)
    pe.add_argument("--scored-rows-out", type=Path, default=DEFAULT_PAIRED_EVAL_ROWS_CSV)
    pe.add_argument("--mode", type=str, default="board_objective")
    pe.add_argument("--date-col", type=str, default="market_date")
    pe.add_argument("--result-col", type=str, default="result")
    pe.add_argument("--start-date", type=str, default=None)
    pe.add_argument("--end-date", type=str, default=None)
    pe.add_argument("--run-date-hint", type=str, default=None)
    pe.add_argument("--min-rows", type=int, default=0)
    pe.add_argument("--short-days", type=int, default=35)
    pe.add_argument("--recent-days", type=int, default=14)
    pe.add_argument("--rolling-window-days", type=int, default=21)
    pe.add_argument("--rolling-step-days", type=int, default=7)
    pe.add_argument("--min-broad-profit-delta-units", type=float, default=0.0)
    pe.add_argument("--min-recent-profit-delta-units", type=float, default=0.0)
    pe.add_argument("--min-recent-hit-rate-delta-pp", type=float, default=0.0)
    pe.add_argument("--min-broad-hit-rate-delta-pp", type=float, default=-0.05)
    pe.add_argument("--min-coverage-retention", type=float, default=0.96)
    pe.add_argument("--min-affected-share", type=float, default=0.01)
    pe.add_argument("--max-top-removed-player-share", type=float, default=0.30)
    pe.add_argument("--max-top-removed-segment-share", type=float, default=0.35)
    pe.add_argument("--max-top-removed-target-share", type=float, default=0.55)
    pe.add_argument("--concentration-min-veto-rows", type=int, default=25)
    pe.add_argument("--disable-conditional-target-cap", action="store_true")
    pe.add_argument("--conditional-target-relaxed-cap", type=float, default=0.90)
    pe.add_argument("--conditional-target-player-share-max", type=float, default=0.15)
    pe.add_argument("--disable-conditional-target-require-recent-non-negative", action="store_true")
    pe.add_argument("--conditional-target-recent-hit-floor-pp", type=float, default=0.0)
    pe.add_argument("--conditional-target-recent-profit-floor", type=float, default=0.0)
    pe.add_argument("--min-rolling-pass-rate", type=float, default=0.55)
    pe.add_argument("--max-observed-fire-rate", type=float, default=0.20)
    pe.add_argument("--rolling-profit-delta-floor", type=float, default=0.0)
    pe.add_argument("--rolling-hit-rate-delta-floor-pp", type=float, default=-0.10)
    pe.add_argument("--disable-small-lift-validation", action="store_true")
    pe.add_argument("--lopo-min-player-rows", type=int, default=12)
    pe.add_argument("--lopo-profit-floor", type=float, default=0.0)
    pe.add_argument("--lopo-hit-floor-pp", type=float, default=0.0)
    pe.add_argument("--min-lopo-profit-pass-rate", type=float, default=0.60)
    pe.add_argument("--min-lopo-hit-pass-rate", type=float, default=0.60)
    pe.add_argument("--bootstrap-iterations", type=int, default=600)
    pe.add_argument("--bootstrap-alpha", type=float, default=0.10)
    pe.add_argument("--bootstrap-seed", type=int, default=29)
    pe.add_argument("--min-bootstrap-profit-ci-lower", type=float, default=0.0)
    pe.add_argument("--min-bootstrap-hit-ci-lower-pp", type=float, default=0.0)
    pe.set_defaults(func=cmd_paired_eval)

    tr = sub.add_parser("train-shadow", help="Train and evaluate candidate accepted-pick gate in shadow-only mode.")
    tr.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    tr.add_argument("--run-date-col", type=str, default="market_date")
    tr.add_argument("--label-col", type=str, default="keep_label")
    tr.add_argument("--model-family", choices=["logistic", "catboost", "uplift_catboost"], default="logistic")
    tr.add_argument("--learning-rate", type=float, default=0.05)
    tr.add_argument("--l2-strength", type=float, default=2.5)
    tr.add_argument("--max-iter", type=int, default=3500)
    tr.add_argument("--tolerance", type=float, default=1e-7)
    tr.add_argument("--class-weight-positive", type=float, default=1.0)
    tr.add_argument("--class-weight-negative", type=float, default=1.0)
    tr.add_argument("--catboost-iterations", type=int, default=320)
    tr.add_argument("--catboost-depth", type=int, default=5)
    tr.add_argument("--catboost-learning-rate", type=float, default=0.05)
    tr.add_argument("--catboost-l2-leaf-reg", type=float, default=8.0)
    tr.add_argument("--catboost-random-strength", type=float, default=1.0)
    tr.add_argument("--catboost-bagging-temperature", type=float, default=0.0)
    tr.add_argument("--catboost-min-data-in-leaf", type=int, default=20)
    tr.add_argument("--catboost-seed", type=int, default=42)
    tr.add_argument("--catboost-segment-weight-power", type=float, default=0.0)
    tr.add_argument("--catboost-segment-weight-cap", type=float, default=3.0)
    tr.add_argument("--uplift-keep-prob-temperature", type=float, default=0.25)
    tr.add_argument("--disable-uplift-harm-head", action="store_true")
    tr.add_argument("--uplift-harm-positive-weight", type=float, default=1.5)
    tr.add_argument("--uplift-harm-temperature", type=float, default=1.0)
    tr.add_argument("--train-window-days", type=int, default=120)
    tr.add_argument("--test-window-days", type=int, default=14)
    tr.add_argument("--step-days", type=int, default=7)
    tr.add_argument("--min-train-rows", type=int, default=250)
    tr.add_argument("--min-test-rows", type=int, default=20)
    tr.add_argument("--threshold-quantiles", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40")
    tr.add_argument("--threshold-max-candidates", type=int, default=48)
    tr.add_argument("--max-fire-rate", type=float, default=0.20)
    tr.add_argument("--min-coverage-rate", type=float, default=0.90)
    tr.add_argument("--max-removed-per-day", type=int, default=1)
    tr.add_argument("--max-removed-per-player-per-day", type=int, default=1)
    tr.add_argument("--max-removed-per-segment-per-day", type=int, default=2)
    tr.add_argument("--max-removed-per-target-per-day", type=int, default=2)
    tr.add_argument("--max-global-removed-per-player", type=int, default=0)
    tr.add_argument("--max-global-removed-share-per-player", type=float, default=1.0)
    tr.add_argument("--global-player-cap-min-veto-rows", type=int, default=8)
    tr.add_argument("--disable-player-cap-relaxation", action="store_true")
    tr.add_argument("--player-penalty-alpha", type=float, default=0.0)
    tr.add_argument("--player-penalty-power", type=float, default=1.0)
    tr.add_argument("--disable-optimizer", action="store_true")
    tr.add_argument("--optimizer-beam-width", type=int, default=96)
    tr.add_argument("--optimizer-max-candidates-per-day", type=int, default=64)
    tr.add_argument("--optimizer-keep-prob-weight", type=float, default=1.0)
    tr.add_argument("--optimizer-gap-weight", type=float, default=0.60)
    tr.add_argument("--optimizer-harm-weight", type=float, default=0.80)
    tr.add_argument("--optimizer-rank-weight", type=float, default=0.0)
    tr.add_argument("--tail-slots-only", type=int, default=2)
    tr.add_argument("--min-veto-gap", type=float, default=0.02)
    tr.add_argument("--objective-player-top-share-penalty", type=float, default=0.0)
    tr.add_argument("--objective-player-hhi-penalty", type=float, default=0.0)
    tr.add_argument("--objective-player-diversity-bonus", type=float, default=0.0)
    tr.add_argument("--recent-days", type=int, default=28)
    tr.add_argument("--rolling-window-days", type=int, default=21)
    tr.add_argument("--rolling-step-days", type=int, default=7)
    tr.add_argument("--threshold-recent-days", type=int, default=14)
    tr.add_argument("--threshold-recent-hit-floor-pp", type=float, default=0.0)
    tr.add_argument("--threshold-recent-profit-floor", type=float, default=0.0)
    tr.add_argument("--threshold-broad-profit-floor", type=float, default=0.0)
    tr.add_argument("--threshold-broad-hit-floor-pp", type=float, default=0.0)
    tr.add_argument("--threshold-min-coverage-retention", type=float, default=0.96)
    tr.add_argument("--threshold-max-fire-rate", type=float, default=0.20)
    tr.add_argument("--threshold-min-affected-share", type=float, default=0.01)
    tr.add_argument("--threshold-max-top-removed-player-share", type=float, default=0.30)
    tr.add_argument("--threshold-max-top-removed-segment-share", type=float, default=0.35)
    tr.add_argument("--threshold-max-top-removed-target-share", type=float, default=0.55)
    tr.add_argument("--disable-conditional-target-cap", action="store_true")
    tr.add_argument("--conditional-target-relaxed-cap", type=float, default=0.90)
    tr.add_argument("--conditional-target-player-share-max", type=float, default=0.15)
    tr.add_argument("--disable-conditional-target-require-recent-non-negative", action="store_true")
    tr.add_argument("--conditional-target-recent-hit-floor-pp", type=float, default=0.0)
    tr.add_argument("--conditional-target-recent-profit-floor", type=float, default=0.0)
    tr.add_argument("--disable-final-non-noop-exploration", action="store_true")
    tr.add_argument("--final-non-noop-objective-margin", type=float, default=0.20)
    tr.add_argument("--segment-search-min-rows", type=int, default=24)
    tr.add_argument("--segment-search-seed-limit", type=int, default=6)
    tr.add_argument("--segment-search-max-size", type=int, default=3)
    tr.add_argument("--segment-search-max-subsets", type=int, default=128)
    tr.add_argument("--disable-cohort-search", action="store_true")
    tr.add_argument("--cohort-search-min-delta-pp", type=float, default=-4.0)
    tr.add_argument("--disable-fullframe-cohort-fallback", action="store_true")
    tr.add_argument("--disable-segment-override-search", action="store_true")
    tr.add_argument("--segment-override-key", type=str, default="PTS|OVER")
    tr.add_argument("--segment-override-thresholds", type=str, default="0.58,0.60,0.62,0.64,0.66")
    tr.add_argument("--segment-override-budgets", type=str, default="1,2")
    tr.add_argument("--segment-override-min-objective-gain", type=float, default=0.0)
    tr.add_argument("--segment-override-min-broad-profit-gain", type=float, default=0.0)
    tr.add_argument("--segment-override-min-recent-profit-gain", type=float, default=0.0)
    tr.add_argument("--segment-override-min-affected-share-gain", type=float, default=0.0)
    tr.add_argument("--train-threshold-min-affected-share", type=float, default=0.01)
    tr.add_argument("--train-threshold-max-top-removed-player-share", type=float, default=0.40)
    tr.add_argument("--train-threshold-max-top-removed-segment-share", type=float, default=0.65)
    tr.add_argument("--train-threshold-max-top-removed-target-share", type=float, default=0.80)
    tr.add_argument("--train-threshold-min-coverage-retention", type=float, default=0.96)
    tr.add_argument("--disable-train-non-noop-exploration", action="store_true")
    tr.add_argument("--train-non-noop-objective-margin", type=float, default=0.30)
    tr.add_argument("--min-broad-profit-delta-units", type=float, default=0.0)
    tr.add_argument("--min-recent-profit-delta-units", type=float, default=0.0)
    tr.add_argument("--min-recent-hit-rate-delta-pp", type=float, default=0.0)
    tr.add_argument("--min-broad-hit-rate-delta-pp", type=float, default=-0.05)
    tr.add_argument("--min-coverage-retention", type=float, default=0.96)
    tr.add_argument("--min-affected-share", type=float, default=0.01)
    tr.add_argument("--max-top-removed-player-share", type=float, default=0.30)
    tr.add_argument("--max-top-removed-segment-share", type=float, default=0.35)
    tr.add_argument("--max-top-removed-target-share", type=float, default=0.55)
    tr.add_argument("--concentration-min-veto-rows", type=int, default=25)
    tr.add_argument("--min-rolling-pass-rate", type=float, default=0.55)
    tr.add_argument("--max-observed-fire-rate", type=float, default=0.20)
    tr.add_argument("--rolling-profit-delta-floor", type=float, default=0.0)
    tr.add_argument("--rolling-hit-rate-delta-floor-pp", type=float, default=-0.10)
    tr.add_argument("--disable-small-lift-validation", action="store_true")
    tr.add_argument("--lopo-min-player-rows", type=int, default=12)
    tr.add_argument("--lopo-profit-floor", type=float, default=0.0)
    tr.add_argument("--lopo-hit-floor-pp", type=float, default=0.0)
    tr.add_argument("--min-lopo-profit-pass-rate", type=float, default=0.60)
    tr.add_argument("--min-lopo-hit-pass-rate", type=float, default=0.60)
    tr.add_argument("--bootstrap-iterations", type=int, default=600)
    tr.add_argument("--bootstrap-alpha", type=float, default=0.10)
    tr.add_argument("--bootstrap-seed", type=int, default=29)
    tr.add_argument("--min-bootstrap-profit-ci-lower", type=float, default=0.0)
    tr.add_argument("--min-bootstrap-hit-ci-lower-pp", type=float, default=0.0)
    tr.add_argument("--model-out", type=Path, default=DEFAULT_CANDIDATE_JSON)
    tr.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_JSON)
    tr.add_argument("--scored-rows-out", type=Path, default=DEFAULT_SCORED_ROWS_CSV)
    tr.set_defaults(func=cmd_train_shadow)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
