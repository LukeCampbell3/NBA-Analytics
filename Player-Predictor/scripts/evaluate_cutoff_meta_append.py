#!/usr/bin/env python3
"""
Cutoff-band append-only evaluator.

Builds a near-cutoff historical dataset (default ranks 8-18 for B=12), trains a
walk-forward residual append-value meta-model, and compares:
1) edge baseline board
2) existing append shadow (agree=1, edge_pct>=0.90, max_extra=1)
3) append-only cutoff meta-model gate (max_extra=1, no replacements)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from post_process_market_plays import compute_final_board


SHADOW_APPEND_AGREEMENT_MIN = 1.0
SHADOW_APPEND_EDGE_PERCENTILE_MIN = 0.90
SHADOW_APPEND_MARKET_BOOKS_MIN = 4.0
SHADOW_APPEND_HISTORY_ROWS_MIN = 35.0
SHADOW_APPEND_CONFIDENCE_MIN = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cutoff-band append-only meta-model.")
    parser.add_argument(
        "--daily-runs-dir",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs",
        help="Path to daily run folders (YYYYMMDD).",
    )
    parser.add_argument("--start-date", type=str, default="2026-03-14", help="Inclusive start run/market date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2026-03-30", help="Inclusive end run/market date (YYYY-MM-DD).")
    parser.add_argument("--board-size", type=int, default=12, help="Baseline edge board size.")
    parser.add_argument("--cutoff-rank-low", type=int, default=8, help="Inclusive lower rank for cutoff-band training rows.")
    parser.add_argument("--cutoff-rank-high", type=int, default=18, help="Inclusive upper rank for cutoff-band training rows.")
    parser.add_argument("--append-window", type=int, default=6, help="Append candidate window from B+1..B+window.")
    parser.add_argument("--min-train-resolved", type=int, default=20, help="Minimum resolved append-pool rows before training.")
    parser.add_argument(
        "--model-uplift-threshold",
        type=float,
        default=0.03,
        help="Minimum predicted append value uplift (vs pool baseline) to append.",
    )
    parser.add_argument(
        "--model-uplift-margin",
        type=float,
        default=0.015,
        help="Minimum uplift gap between top-1 and top-2 near-miss candidates; otherwise abstain.",
    )
    parser.add_argument("--gap-quantile-threshold", type=float, default=0.60, help="Require day cutoff-gap <= this train quantile.")
    parser.add_argument("--max-corr-score", type=float, default=1.25, help="Maximum correlation score allowed for appended row.")
    parser.add_argument(
        "--research-pool-min-agreement",
        type=float,
        default=1.0,
        help="Minimum agreement_count for research append-pool opportunity diagnostics.",
    )
    parser.add_argument(
        "--research-corr-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "absolute", "percentile", "zscore"],
        help="Research corr feasibility mode for append-pool diagnostics.",
    )
    parser.add_argument(
        "--research-pool-max-corr-score",
        type=float,
        default=None,
        help="Optional corr_score ceiling for research append-pool diagnostics. Defaults to no ceiling.",
    )
    parser.add_argument(
        "--research-corr-percentile-max",
        type=float,
        default=None,
        help="Optional within-day corr percentile cap [0,1] for research diagnostics.",
    )
    parser.add_argument(
        "--research-corr-zscore-max",
        type=float,
        default=None,
        help="Optional within-day corr z-score cap for research diagnostics.",
    )
    parser.add_argument(
        "--unified-veto-corr-score",
        type=float,
        default=1.25,
        help="Unified veto ceiling for shadow candidate correlation score; set high to disable correlation veto.",
    )
    parser.add_argument(
        "--unified-shadow-uplift-floor",
        type=float,
        default=-0.03,
        help="Unified veto floor: veto shadow append only when predicted uplift is materially below this level.",
    )
    parser.add_argument(
        "--unified-require-shallow-day",
        action="store_true",
        help="If set, unified mode only allows append on shallow-cutoff days.",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_band_dataset_20260314_20260330_b12.csv",
        help="CSV output for cutoff-band and append-pool feature rows.",
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_meta_append_rows_20260314_20260330_b12.csv",
        help="CSV output for variant-level evaluated rows.",
    )
    parser.add_argument(
        "--daily-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_meta_append_daily_20260314_20260330_b12.csv",
        help="CSV output for per-day summary.",
    )
    parser.add_argument(
        "--daily-context-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_meta_append_daily_context_20260314_20260330_b12.csv",
        help="CSV output for per-day slate-context validation dataset.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_meta_append_summary_20260314_20260330_b12.json",
        help="JSON output for aggregate summary.",
    )
    parser.add_argument(
        "--abstain-out",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "daily_runs" / "cutoff_meta_append_abstain_20260314_20260330_b12.csv",
        help="CSV output for per-day abstention diagnostics.",
    )
    parser.add_argument(
        "--shadow-top1-miss-out",
        type=Path,
        default=None,
        help="Optional CSV output for positive top1 days missed by shadow append, with miss reasons.",
    )
    parser.add_argument(
        "--top1-shadow-miss-summary-out",
        type=Path,
        default=None,
        help="Optional JSON summary artifact for top1 shadow miss reasons.",
    )
    parser.add_argument(
        "--stage2-proposal-table-out",
        type=Path,
        default=None,
        help="Optional CSV output for Stage-2 proposal-ranker training rows (candidate-level).",
    )
    parser.add_argument(
        "--stage2-proposal-pairs-out",
        type=Path,
        default=None,
        help="Optional CSV output for Stage-2 pairwise ranking rows.",
    )
    parser.add_argument(
        "--stage2-proposal-summary-out",
        type=Path,
        default=None,
        help="Optional JSON output with Stage-2 proposal table diagnostics.",
    )
    parser.add_argument(
        "--exclude-snapshot-modes",
        nargs="*",
        default=[],
        help="Optional current_market_snapshot_meta.mode values to exclude (for example: stale_fallback).",
    )
    return parser.parse_args()


def _read_run_dirs(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child.name.isdigit() and len(child.name) == 8):
            continue
        run_dt = pd.to_datetime(child.name, format="%Y%m%d", errors="coerce")
        if pd.isna(run_dt):
            continue
        if start <= run_dt <= end:
            out.append(child)
    return out


def _load_snapshot_meta(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / f"daily_market_pipeline_manifest_{run_dir.name}.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    meta = payload.get("current_market_snapshot_meta")
    return meta if isinstance(meta, dict) else {}


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
        target_key = str(target).upper()
        if target_key not in {"PTS", "TRB", "AST"}:
            return None
        rows = frame.loc[frame["Date"] == str(market_date)]
        if rows.empty:
            return None
        value = pd.to_numeric(rows.iloc[-1][target_key], errors="coerce")
        if pd.isna(value):
            return None
        return float(value)

    return resolve


def _load_recent_volatility_lookup(lookback_games: int = 5):
    cache: dict[str, pd.DataFrame] = {}

    def resolve(csv_path: str, market_date: str, target: str) -> float:
        if not isinstance(csv_path, str) or not csv_path:
            return float("nan")
        target_key = str(target).upper()
        if target_key not in {"PTS", "TRB", "AST"}:
            return float("nan")
        if csv_path not in cache:
            path = Path(csv_path)
            if not path.exists():
                cache[csv_path] = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
            else:
                try:
                    frame = pd.read_csv(path, usecols=["Date", "PTS", "TRB", "AST"])
                except Exception:
                    frame = pd.DataFrame(columns=["Date", "PTS", "TRB", "AST"])
                frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
                cache[csv_path] = frame
        frame = cache[csv_path]
        if frame.empty:
            return float("nan")
        mdate = pd.to_datetime(market_date, errors="coerce")
        if pd.isna(mdate):
            return float("nan")
        prior = frame.loc[frame["Date"] < mdate, ["Date", target_key]].copy()
        if prior.empty:
            return float("nan")
        prior[target_key] = pd.to_numeric(prior[target_key], errors="coerce")
        prior = prior.loc[prior[target_key].notna()].sort_values("Date")
        if prior.empty:
            return float("nan")
        series = prior[target_key].tail(int(max(2, lookback_games)))
        if len(series) < 2:
            return float(0.0)
        return float(series.std(ddof=0))

    return resolve


def _classify_result(direction: str, line: float | None, actual: float | None) -> str:
    if actual is None or line is None:
        return "missing"
    d = str(direction).upper()
    if d == "OVER":
        if actual > line:
            return "win"
        if actual < line:
            return "loss"
        return "push"
    if d == "UNDER":
        if actual < line:
            return "win"
        if actual > line:
            return "loss"
        return "push"
    return "missing"


def _board_params(board_size: int) -> dict[str, Any]:
    return {
        "american_odds": -110,
        "min_ev": -1.0,
        "min_final_confidence": 0.0,
        "min_recommendation": "pass",
        "max_plays_per_player": 1,
        "max_plays_per_target": 0,
        "max_total_plays": int(board_size),
        "max_target_plays": {"PTS": 10, "TRB": 4, "AST": 4},
        "max_plays_per_game": 0,
        "max_plays_per_script_cluster": 3,
        "non_pts_min_gap_percentile": 0.0,
        "min_bet_win_rate": 0.49,
        "medium_bet_win_rate": 0.52,
        "full_bet_win_rate": 0.56,
        "medium_tier_percentile": 0.0,
        "strong_tier_percentile": 0.0,
        "elite_tier_percentile": 0.0,
    }


def _build_edge_board(selector_df: pd.DataFrame, board_size: int) -> pd.DataFrame:
    return compute_final_board(
        selector_df,
        selection_mode="edge",
        ranking_mode="edge",
        **_board_params(board_size),
    )


def _build_shadow_board(
    selector_df: pd.DataFrame,
    board_size: int,
    append_agreement_min: int = 1,
    append_edge_percentile_min: float = 0.90,
    append_max_extra_plays: int = 1,
) -> pd.DataFrame:
    return compute_final_board(
        selector_df,
        selection_mode="edge_append_shadow",
        ranking_mode="edge_append_shadow",
        append_agreement_min=int(append_agreement_min),
        append_edge_percentile_min=float(append_edge_percentile_min),
        append_max_extra_plays=int(append_max_extra_plays),
        **_board_params(board_size),
    )


def _build_extended_edge_universe(selector_df: pd.DataFrame, board_size: int, cutoff_rank_high: int, append_window: int) -> pd.DataFrame:
    # Build a ranked universe beyond B so we can inspect the near-cutoff zone.
    total_cap = max(int(board_size + append_window + 8), int(cutoff_rank_high + 2), 300)
    return compute_final_board(
        selector_df,
        selection_mode="edge",
        ranking_mode="edge",
        american_odds=-110,
        min_ev=-1.0,
        min_final_confidence=0.0,
        min_recommendation="pass",
        max_plays_per_player=1,
        max_plays_per_target=0,
        max_total_plays=total_cap,
        max_target_plays={"PTS": 20, "TRB": 10, "AST": 10},
        max_plays_per_game=0,
        max_plays_per_script_cluster=3,
        non_pts_min_gap_percentile=0.0,
        min_bet_win_rate=0.0,
        medium_bet_win_rate=1.0,
        full_bet_win_rate=1.0,
        medium_tier_percentile=0.0,
        strong_tier_percentile=0.0,
        elite_tier_percentile=0.0,
    )


def _add_agreement_features(df: pd.DataFrame, board_size: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    overfetch = int(np.clip(3 * int(board_size), 1, len(out)))
    idx_e = set(out.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).head(overfetch).index.tolist())
    idx_t = set(
        out.sort_values(["thompson_ev", "ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False, False])
        .head(overfetch)
        .index.tolist()
    )
    idx_v = set(out.sort_values(["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"], ascending=[False, False, False, False]).head(overfetch).index.tolist())
    out["agreement_count"] = out.index.to_series().map(lambda idx: int(idx in idx_e) + int(idx in idx_t) + int(idx in idx_v)).astype(int)
    out["edge_percentile"] = pd.to_numeric(out["abs_edge"], errors="coerce").fillna(0.0).rank(method="average", pct=True)
    out["set_sources"] = out.apply(
        lambda row: ",".join(
            part
            for part, enabled in (
                ("E", bool(row.name in idx_e)),
                ("T", bool(row.name in idx_t)),
                ("V", bool(row.name in idx_v)),
            )
            if enabled
        ),
        axis=1,
    )
    return out


def _safe_num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _metrics(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "rows": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "missing": 0,
            "resolved_hit_rate": np.nan,
            "ev_per_resolved": np.nan,
        }
    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    pushes = int((df["result"] == "push").sum())
    missing = int((df["result"] == "missing").sum())
    resolved = wins + losses
    hit_rate = (wins / resolved) if resolved else np.nan
    ev_per_resolved = ((wins * (100.0 / 110.0) - losses) / resolved) if resolved else np.nan
    return {
        "rows": int(len(df)),
        "resolved": int(resolved),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "missing": int(missing),
        "resolved_hit_rate": float(hit_rate) if not pd.isna(hit_rate) else np.nan,
        "ev_per_resolved": float(ev_per_resolved) if not pd.isna(ev_per_resolved) else np.nan,
    }


def _feature_spec() -> tuple[list[str], list[str]]:
    numeric = [
        "edge",
        "abs_edge",
        "edge_percentile",
        "win_rate_minus_breakeven",
        "raw_win_rate_minus_breakeven",
        "final_confidence",
        "market_books",
        "history_rows",
        "agreement_count",
        "risk_penalty",
        "uncertainty_sigma",
        "volatility_score",
        "player_recent_volatility",
        "tail_imbalance",
        "corr_score",
        "corr_same_game_count",
        "corr_same_target_dir_count",
        "corr_same_script_cluster_count",
        "board_cutoff_gap",
        "board_top_to_cutoff_gap",
        "board_edge_mean",
        "board_edge_std",
        "board_target_count_same",
        "pool_rank_edge",
        "pool_edge_z",
        "pool_conf_z",
        "pool_corr_z",
        "pool_agreement_z",
        "pool_edge_gap_to_top",
        "pool_edge_gap_to_median",
        "pool_conf_gap_to_median",
        "pool_size",
        "near_miss_local_slope",
        "board_concentration",
    ]
    categorical = ["target", "direction", "set_sources"]
    return numeric, categorical


def _build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn fallback
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", onehot),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_depth=6,
        random_state=17,
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def _play_key_from_values(player: Any, target: Any, direction: Any, market_line: Any) -> tuple[str, str, str, float | None]:
    line = pd.to_numeric(pd.Series([market_line]), errors="coerce").iloc[0]
    line_key = None if pd.isna(line) else round(float(line), 4)
    return (str(player), str(target), str(direction), line_key)


def _zscore_within_group(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean()) if numeric.notna().any() else 0.0
    std = float(numeric.std(ddof=0)) if numeric.notna().any() else 0.0
    if std <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return ((numeric - mean) / std).fillna(0.0)


def _add_slate_relative_features(feature_df: pd.DataFrame, board_size: int) -> pd.DataFrame:
    if feature_df.empty:
        return feature_df.copy()
    out = feature_df.copy()

    out["pool_rank_edge"] = np.nan
    out["pool_edge_z"] = np.nan
    out["pool_conf_z"] = np.nan
    out["pool_corr_z"] = np.nan
    out["pool_agreement_z"] = np.nan
    out["pool_edge_gap_to_top"] = np.nan
    out["pool_edge_gap_to_median"] = np.nan
    out["pool_conf_gap_to_median"] = np.nan
    out["pool_size"] = 0
    out["near_miss_local_slope"] = np.nan
    out["board_concentration"] = np.nan

    for run_date, group_idx in out.groupby("run_date").groups.items():
        group = out.loc[group_idx].copy()
        pool = group.loc[group["is_append_pool"] == True].copy()
        board = group.loc[group["is_base_board_member"] == True].copy()

        # Board concentration: higher means concentrated in fewer players.
        if not board.empty:
            player_counts = board["player"].astype(str).value_counts(normalize=True)
            concentration = float((player_counts**2).sum())
        else:
            concentration = np.nan

        # Local slope around cutoff ranks 10-16 captures board steepness.
        local = group.loc[group["edge_rank"].between(max(1, int(board_size) - 2), int(board_size) + 4)].sort_values("edge_rank")
        if len(local) >= 2:
            edge_vals = pd.to_numeric(local["edge"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            slope = float(edge_vals[0] - edge_vals[-1]) / max(1, len(edge_vals) - 1)
        else:
            slope = np.nan

        out.loc[group_idx, "board_concentration"] = concentration
        out.loc[group_idx, "near_miss_local_slope"] = slope

        if pool.empty:
            continue

        edge = pd.to_numeric(pool["edge"], errors="coerce").fillna(0.0)
        conf = pd.to_numeric(pool["final_confidence"], errors="coerce").fillna(0.0)
        corr = pd.to_numeric(pool["corr_score"], errors="coerce").fillna(0.0)
        agr = pd.to_numeric(pool["agreement_count"], errors="coerce").fillna(0.0)

        pool_features = pd.DataFrame(
            {
                "pool_rank_edge": edge.rank(method="dense", ascending=False),
                "pool_edge_z": _zscore_within_group(edge),
                "pool_conf_z": _zscore_within_group(conf),
                "pool_corr_z": _zscore_within_group(corr),
                "pool_agreement_z": _zscore_within_group(agr),
                "pool_edge_gap_to_top": edge - float(edge.max()),
                "pool_edge_gap_to_median": edge - float(edge.median()),
                "pool_conf_gap_to_median": conf - float(conf.median()),
                "pool_size": int(len(pool)),
            },
            index=pool.index,
        )
        for col in pool_features.columns:
            out.loc[pool_features.index, col] = pool_features[col]
    return out


def _decision_time_append_score(pool_df: pd.DataFrame) -> pd.Series:
    """Decision-time-only append score used for top-1 feasible diagnostics."""
    edge_z = pd.to_numeric(pool_df.get("pool_edge_z"), errors="coerce").fillna(0.0)
    conf_z = pd.to_numeric(pool_df.get("pool_conf_z"), errors="coerce").fillna(0.0)
    agr_z = pd.to_numeric(pool_df.get("pool_agreement_z"), errors="coerce").fillna(0.0)
    corr_z = pd.to_numeric(pool_df.get("pool_corr_z"), errors="coerce").fillna(0.0)
    edge_pct = pd.to_numeric(pool_df.get("edge_percentile"), errors="coerce").fillna(0.0)
    conf = pd.to_numeric(pool_df.get("final_confidence"), errors="coerce").fillna(0.0)

    # Weighted blend emphasizing edge/confidence/consensus with correlation penalty.
    return (
        0.55 * edge_z
        + 0.25 * conf_z
        + 0.20 * agr_z
        - 0.20 * corr_z
        + 0.10 * edge_pct
        + 0.05 * conf
    )


def _shadow_append_eligibility_mask(df: pd.DataFrame) -> pd.Series:
    agreement = pd.to_numeric(df.get("agreement_count"), errors="coerce").fillna(0.0)
    edge_pct = pd.to_numeric(df.get("edge_percentile"), errors="coerce").fillna(0.0)
    books = pd.to_numeric(df.get("market_books"), errors="coerce").fillna(0.0)
    history = pd.to_numeric(df.get("history_rows"), errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(df.get("final_confidence"), errors="coerce").fillna(0.0)
    return (
        (agreement >= float(SHADOW_APPEND_AGREEMENT_MIN))
        & (edge_pct >= float(SHADOW_APPEND_EDGE_PERCENTILE_MIN))
        & (books >= float(SHADOW_APPEND_MARKET_BOOKS_MIN))
        & (history >= float(SHADOW_APPEND_HISTORY_ROWS_MIN))
        & (confidence >= float(SHADOW_APPEND_CONFIDENCE_MIN))
    )


def _compose_shadow_gate_fail_reasons(
    *,
    pass_agreement: bool,
    pass_edge_pct: bool,
    pass_books: bool,
    pass_history: bool,
    pass_confidence: bool,
) -> str:
    reasons: list[str] = []
    if not pass_agreement:
        reasons.append("agreement_below_min")
    if not pass_edge_pct:
        reasons.append("edge_percentile_below_min")
    if not pass_books:
        reasons.append("market_books_below_min")
    if not pass_history:
        reasons.append("history_rows_below_min")
    if not pass_confidence:
        reasons.append("confidence_below_min")
    return "|".join(reasons)


def _build_shadow_top1_miss_report(daily_context_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_date",
        "snapshot_mode",
        "pool_size",
        "pool_feasible_count",
        "pool_feasible_resolved_count",
        "top1_feasible_player",
        "top1_feasible_target",
        "top1_feasible_direction",
        "top1_feasible_result",
        "top1_feasible_ev_label",
        "top1_feasible_value_delta",
        "top1_feasible_decision_score",
        "top1_feasible_market_line",
        "top1_feasible_edge_rank",
        "top1_feasible_edge",
        "top1_feasible_confidence",
        "top1_feasible_agreement",
        "top1_feasible_corr_score",
        "top1_feasible_corr_percentile_in_pool",
        "top1_feasible_market_books",
        "top1_feasible_history_rows",
        "top1_feasible_edge_percentile",
        "top1_shadow_rule_eligible",
        "top1_shadow_rule_rank_all_candidates",
        "top1_shadow_rule_rank_pool_candidates",
        "top1_shadow_rule_fail_reasons",
        "shadow_candidate_exists",
        "shadow_candidate_player",
        "shadow_candidate_target",
        "shadow_candidate_direction",
        "shadow_candidate_result",
        "shadow_candidate_edge_rank",
        "shadow_candidate_edge",
        "shadow_candidate_confidence",
        "shadow_candidate_agreement",
        "shadow_candidate_corr_score",
        "shadow_candidate_in_research_pool",
        "shadow_top1_miss_stage",
        "shadow_top1_miss_reason",
        "shadow_top1_generation_blocked",
        "shadow_top1_safety_blocked",
    ]
    if daily_context_df.empty:
        return pd.DataFrame(columns=cols)
    top1_positive = pd.to_numeric(daily_context_df.get("label_top1_feasible_positive"), errors="coerce")
    shadow_missed = pd.to_numeric(daily_context_df.get("label_shadow_missed_positive_top1"), errors="coerce")
    mask = (top1_positive == 1) & (shadow_missed == 1)
    out = daily_context_df.loc[mask].copy()
    if out.empty:
        return pd.DataFrame(columns=cols)
    keep_cols = [c for c in cols if c in out.columns]
    return out.loc[:, keep_cols].sort_values("run_date").reset_index(drop=True)


def _normalize_research_envelope_label(research_corr_mode: str, research_corr_threshold: float | None) -> str:
    mode = str(research_corr_mode or "none").strip().lower()
    if mode == "percentile" and research_corr_threshold is not None and np.isfinite(float(research_corr_threshold)):
        return f"pct_{float(research_corr_threshold):.2f}"
    if mode == "absolute" and research_corr_threshold is not None and np.isfinite(float(research_corr_threshold)):
        return f"abs_{float(research_corr_threshold):g}"
    if mode == "zscore" and research_corr_threshold is not None and np.isfinite(float(research_corr_threshold)):
        return f"z_{float(research_corr_threshold):g}"
    return mode


def _research_feasible_mask(
    df: pd.DataFrame,
    *,
    research_pool_min_agreement: float,
    research_corr_mode: str,
    research_corr_threshold: float | None,
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="bool")
    agreement = pd.to_numeric(df.get("agreement_count"), errors="coerce").fillna(0.0)
    corr = pd.to_numeric(df.get("corr_score"), errors="coerce")
    feasible = agreement >= float(research_pool_min_agreement)
    mode = str(research_corr_mode or "none").strip().lower()
    threshold = None if research_corr_threshold is None else float(research_corr_threshold)
    if mode == "absolute" and threshold is not None and not np.isinf(threshold):
        feasible = feasible & (corr.isna() | (corr <= float(threshold)))
    elif mode == "percentile" and threshold is not None:
        corr_pct = pd.to_numeric(corr.rank(method="average", pct=True), errors="coerce")
        feasible = feasible & (corr_pct <= float(threshold))
    elif mode == "zscore" and threshold is not None:
        corr_z = _zscore_within_group(corr.fillna(0.0))
        feasible = feasible & (pd.to_numeric(corr_z, errors="coerce") <= float(threshold))
    return feasible.astype(bool)


def _gate_ablation_recovery_counts(miss_df: pd.DataFrame) -> dict[str, int]:
    if miss_df.empty:
        return {
            "recover_if_relax_edge_percentile_only": 0,
            "recover_if_relax_market_books_only": 0,
            "recover_if_relax_history_rows_only": 0,
            "recover_if_relax_confidence_only": 0,
            "recover_if_relax_agreement_only": 0,
        }
    passes_edge = pd.to_numeric(miss_df.get("top1_shadow_rule_pass_edge_percentile"), errors="coerce").fillna(0).astype(int)
    passes_books = pd.to_numeric(miss_df.get("top1_shadow_rule_pass_market_books"), errors="coerce").fillna(0).astype(int)
    passes_hist = pd.to_numeric(miss_df.get("top1_shadow_rule_pass_history_rows"), errors="coerce").fillna(0).astype(int)
    passes_conf = pd.to_numeric(miss_df.get("top1_shadow_rule_pass_confidence"), errors="coerce").fillna(0).astype(int)
    passes_agr = pd.to_numeric(miss_df.get("top1_shadow_rule_pass_agreement"), errors="coerce").fillna(0).astype(int)
    return {
        "recover_if_relax_edge_percentile_only": int(((passes_edge == 0) & (passes_books == 1) & (passes_hist == 1) & (passes_conf == 1) & (passes_agr == 1)).sum()),
        "recover_if_relax_market_books_only": int(((passes_edge == 1) & (passes_books == 0) & (passes_hist == 1) & (passes_conf == 1) & (passes_agr == 1)).sum()),
        "recover_if_relax_history_rows_only": int(((passes_edge == 1) & (passes_books == 1) & (passes_hist == 0) & (passes_conf == 1) & (passes_agr == 1)).sum()),
        "recover_if_relax_confidence_only": int(((passes_edge == 1) & (passes_books == 1) & (passes_hist == 1) & (passes_conf == 0) & (passes_agr == 1)).sum()),
        "recover_if_relax_agreement_only": int(((passes_edge == 1) & (passes_books == 1) & (passes_hist == 1) & (passes_conf == 1) & (passes_agr == 0)).sum()),
    }


def _build_top1_shadow_miss_summary(
    daily_context_df: pd.DataFrame,
    miss_df: pd.DataFrame,
    *,
    research_corr_mode: str,
    research_corr_threshold: float | None,
) -> dict[str, Any]:
    envelope_label = _normalize_research_envelope_label(research_corr_mode, research_corr_threshold)
    top1_positive = pd.to_numeric(daily_context_df.get("label_top1_feasible_positive"), errors="coerce") if not daily_context_df.empty else pd.Series(dtype="float64")
    positive_top1_days = int((top1_positive == 1).sum()) if not top1_positive.empty else 0
    missed_positive_days = int(len(miss_df))
    recall = float((positive_top1_days - missed_positive_days) / positive_top1_days) if positive_top1_days > 0 else np.nan

    miss_stage_counts = (
        miss_df.get("shadow_top1_miss_stage", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
        if not miss_df.empty
        else {}
    )
    miss_reason_counts = (
        miss_df.get("shadow_top1_miss_reason", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
        if not miss_df.empty
        else {}
    )

    gate_key_map = {
        "edge_percentile_below_min": "edge_percentile",
        "market_books_below_min": "market_books",
        "history_rows_below_min": "history_rows",
        "confidence_below_min": "confidence",
        "agreement_below_min": "agreement",
    }
    gate_fail_counts: dict[str, int] = {v: 0 for v in gate_key_map.values()}
    gate_combo_counts: dict[str, int] = {}
    if not miss_df.empty:
        fail_series = miss_df.get("top1_shadow_rule_fail_reasons", pd.Series("", index=miss_df.index)).fillna("").astype(str)
        for raw in fail_series:
            parts = [p for p in raw.split("|") if p]
            mapped = [gate_key_map.get(p, p) for p in parts]
            for gate in mapped:
                gate_fail_counts[gate] = int(gate_fail_counts.get(gate, 0) + 1)
            combo_key = "|".join(mapped)
            gate_combo_counts[combo_key] = int(gate_combo_counts.get(combo_key, 0) + 1)

    threshold_map = {
        "edge_percentile": float(SHADOW_APPEND_EDGE_PERCENTILE_MIN),
        "market_books": float(SHADOW_APPEND_MARKET_BOOKS_MIN),
        "history_rows": float(SHADOW_APPEND_HISTORY_ROWS_MIN),
        "confidence": float(SHADOW_APPEND_CONFIDENCE_MIN),
        "agreement": float(SHADOW_APPEND_AGREEMENT_MIN),
    }
    value_cols = {
        "edge_percentile": "top1_feasible_edge_percentile",
        "market_books": "top1_feasible_market_books",
        "history_rows": "top1_feasible_history_rows",
        "confidence": "top1_feasible_confidence",
        "agreement": "top1_feasible_agreement",
    }
    pass_cols = {
        "edge_percentile": "top1_shadow_rule_pass_edge_percentile",
        "market_books": "top1_shadow_rule_pass_market_books",
        "history_rows": "top1_shadow_rule_pass_history_rows",
        "confidence": "top1_shadow_rule_pass_confidence",
        "agreement": "top1_shadow_rule_pass_agreement",
    }
    gate_gap_summary: dict[str, dict[str, float | int]] = {}
    for gate, threshold in threshold_map.items():
        if miss_df.empty:
            gate_gap_summary[gate] = {"failed_count": 0, "mean_gap": np.nan, "median_gap": np.nan}
            continue
        pass_series = pd.to_numeric(miss_df.get(pass_cols[gate]), errors="coerce")
        value_series = pd.to_numeric(miss_df.get(value_cols[gate]), errors="coerce")
        gap_series = value_series - float(threshold)
        fail_mask = pass_series == 0
        failed_gaps = gap_series.loc[fail_mask].dropna()
        gate_gap_summary[gate] = {
            "failed_count": int(len(failed_gaps)),
            "mean_gap": float(failed_gaps.mean()) if not failed_gaps.empty else np.nan,
            "median_gap": float(failed_gaps.median()) if not failed_gaps.empty else np.nan,
        }

    envelope_payload = {
        "research_envelope": envelope_label,
        "positive_top1_days": positive_top1_days,
        "shadow_recall_on_positive_top1_days": recall,
        "miss_stage_breakdown": miss_stage_counts,
        "gate_fail_counts": gate_fail_counts,
        "gate_fail_combinations": gate_combo_counts,
        "gate_gap_to_threshold": gate_gap_summary,
        "gate_ablation_counterfactual": _gate_ablation_recovery_counts(miss_df),
    }
    by_env = {
        "pct_0.25": None,
        "pct_0.50": None,
    }
    if envelope_label in by_env:
        by_env[envelope_label] = envelope_payload

    return {
        "research_envelope": envelope_label,
        "rows_missed_positive_top1": missed_positive_days,
        "positive_top1_days": positive_top1_days,
        "shadow_recall_on_positive_top1_days": recall,
        "miss_stage_breakdown": miss_stage_counts,
        "miss_reason_counts": miss_reason_counts,
        "gate_fail_counts": gate_fail_counts,
        "gate_fail_combinations": gate_combo_counts,
        "gate_gap_to_threshold": gate_gap_summary,
        "gate_ablation_counterfactual": _gate_ablation_recovery_counts(miss_df),
        "by_research_envelope": by_env,
    }


def _build_stage2_proposal_tables(
    feature_df: pd.DataFrame,
    daily_context_df: pd.DataFrame,
    *,
    research_pool_min_agreement: float,
    research_corr_mode: str,
    research_corr_threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if feature_df.empty or daily_context_df.empty:
        empty = pd.DataFrame()
        return empty, empty, {"candidate_rows": 0, "pair_rows": 0}

    pool = feature_df.loc[feature_df["is_append_pool"] == True].copy()
    if pool.empty:
        empty = pd.DataFrame()
        return empty, empty, {"candidate_rows": 0, "pair_rows": 0}

    pool["decision_time_score"] = _decision_time_append_score(pool)
    pool["corr_percentile_in_pool"] = (
        pool.groupby("run_date")["corr_score"].rank(method="average", pct=True)
    )
    pool["corr_z_in_pool"] = (
        pool.groupby("run_date")["corr_score"].transform(lambda s: _zscore_within_group(pd.to_numeric(s, errors="coerce").fillna(0.0)))
    )
    pool["is_research_feasible"] = (
        pool.groupby("run_date", group_keys=False)
        .apply(
            lambda g: _research_feasible_mask(
                g,
                research_pool_min_agreement=float(research_pool_min_agreement),
                research_corr_mode=research_corr_mode,
                research_corr_threshold=research_corr_threshold,
            )
        )
        .astype(bool)
    )

    daily_context = daily_context_df.copy()
    daily_context["run_date"] = daily_context["run_date"].astype(str)
    resolved_days = set(
        daily_context.loc[
            pd.to_numeric(daily_context.get("pool_feasible_resolved_count"), errors="coerce").fillna(0).gt(0),
            "run_date",
        ].astype(str)
    )

    candidates = pool.loc[
        pool["is_research_feasible"].astype(bool) & pool["run_date"].astype(str).isin(resolved_days)
    ].copy()
    if candidates.empty:
        empty = pd.DataFrame()
        return empty, empty, {"candidate_rows": 0, "pair_rows": 0}

    day_cols = [
        "run_date",
        "snapshot_mode",
        "pool_size",
        "pool_feasible_count",
        "pool_feasible_resolved_count",
        "board_cutoff_gap",
        "board_top_to_cutoff_gap",
        "near_miss_local_slope",
        "board_concentration",
        "pool_edge_gap_1_2",
        "pool_conf_gap_1_2",
        "pool_edge_mean",
        "pool_edge_std",
        "pool_conf_mean",
        "pool_conf_std",
        "pool_corr_mean",
        "pool_corr_std",
        "label_top1_feasible_positive",
        "label_top1_feasible_improves_edge",
        "top1_feasible_value_delta",
        "label_shadow_missed_positive_top1",
        "top1_feasible_player",
        "top1_feasible_target",
        "top1_feasible_direction",
        "top1_feasible_market_line",
        "top1_feasible_decision_score",
    ]
    day_cols = [c for c in day_cols if c in daily_context.columns]
    day_frame = daily_context.loc[:, day_cols].copy()
    candidates["run_date"] = candidates["run_date"].astype(str)
    candidates = candidates.merge(day_frame, on="run_date", how="left", suffixes=("", "_day"))

    candidates["candidate_key"] = candidates.apply(
        lambda r: _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line")),
        axis=1,
    )
    top1_market_line = pd.to_numeric(
        candidates.get("top1_feasible_market_line", pd.Series(np.nan, index=candidates.index)),
        errors="coerce",
    )
    candidates["top1_key"] = candidates.apply(
        lambda r: _play_key_from_values(
            r.get("top1_feasible_player"),
            r.get("top1_feasible_target"),
            r.get("top1_feasible_direction"),
            top1_market_line.loc[r.name] if r.name in top1_market_line.index else np.nan,
        ),
        axis=1,
    )
    fallback_top1_match = (
        (candidates["player"].astype(str) == candidates.get("top1_feasible_player", pd.Series("", index=candidates.index)).astype(str))
        & (candidates["target"].astype(str) == candidates.get("top1_feasible_target", pd.Series("", index=candidates.index)).astype(str))
        & (candidates["direction"].astype(str) == candidates.get("top1_feasible_direction", pd.Series("", index=candidates.index)).astype(str))
    )
    candidates["candidate_is_day_top1_feasible"] = (
        (candidates["candidate_key"].astype(str) == candidates["top1_key"].astype(str)) | fallback_top1_match
    ).astype(int)

    candidates["candidate_id"] = candidates["run_date"].astype(str) + "::" + candidates["candidate_key"].astype(str)
    candidates["label_day_top1_feasible_positive"] = pd.to_numeric(candidates.get("label_top1_feasible_positive"), errors="coerce")
    candidates["label_day_top1_feasible_improves_edge"] = pd.to_numeric(candidates.get("label_top1_feasible_improves_edge"), errors="coerce")
    candidates["label_day_top1_value_delta_vs_edge"] = pd.to_numeric(candidates.get("top1_feasible_value_delta"), errors="coerce")
    candidates["label_day_is_missed_positive_top1"] = (
        (
            (pd.to_numeric(candidates.get("label_top1_feasible_positive"), errors="coerce") == 1)
            & (pd.to_numeric(candidates.get("label_shadow_missed_positive_top1"), errors="coerce") == 1)
        )
        .astype(int)
    )
    candidates["label_candidate_is_matched_non_top1_from_missed_positive_day"] = (
        (candidates["label_day_is_missed_positive_top1"] == 1)
        & (candidates["candidate_is_day_top1_feasible"] == 0)
    ).astype(int)

    candidates["edge_percentile_minus_shadow_threshold"] = pd.to_numeric(candidates.get("edge_percentile"), errors="coerce") - float(SHADOW_APPEND_EDGE_PERCENTILE_MIN)
    candidates["market_books_minus_shadow_threshold"] = pd.to_numeric(candidates.get("market_books"), errors="coerce") - float(SHADOW_APPEND_MARKET_BOOKS_MIN)
    candidates["history_rows_minus_shadow_threshold"] = pd.to_numeric(candidates.get("history_rows"), errors="coerce") - float(SHADOW_APPEND_HISTORY_ROWS_MIN)
    candidates["confidence_minus_shadow_threshold"] = pd.to_numeric(candidates.get("final_confidence"), errors="coerce") - float(SHADOW_APPEND_CONFIDENCE_MIN)
    candidates["agreement_minus_shadow_threshold"] = pd.to_numeric(candidates.get("agreement_count"), errors="coerce") - float(SHADOW_APPEND_AGREEMENT_MIN)
    candidates["passes_shadow_gate_edge_percentile"] = (candidates["edge_percentile_minus_shadow_threshold"] >= 0.0).astype(int)
    candidates["passes_shadow_gate_market_books"] = (candidates["market_books_minus_shadow_threshold"] >= 0.0).astype(int)
    candidates["passes_shadow_gate_history_rows"] = (candidates["history_rows_minus_shadow_threshold"] >= 0.0).astype(int)
    candidates["passes_shadow_gate_confidence"] = (candidates["confidence_minus_shadow_threshold"] >= 0.0).astype(int)
    candidates["passes_shadow_gate_agreement"] = (candidates["agreement_minus_shadow_threshold"] >= 0.0).astype(int)
    candidates["research_envelope"] = _normalize_research_envelope_label(research_corr_mode, research_corr_threshold)

    # Pairwise rows: top1 candidate should outrank all other feasible candidates on the same day.
    pair_rows: list[dict[str, Any]] = []
    for run_date, day_df in candidates.groupby("run_date", sort=True):
        top_rows = day_df.loc[day_df["candidate_is_day_top1_feasible"] == 1].copy()
        if top_rows.empty:
            continue
        top_row = top_rows.sort_values(["decision_time_score", "edge"], ascending=[False, False]).iloc[0]
        others = day_df.loc[day_df["candidate_id"].astype(str) != str(top_row.get("candidate_id", ""))].copy()
        for _, other in others.iterrows():
            other_ev = pd.to_numeric(pd.Series([other.get("ev_label")]), errors="coerce").iloc[0]
            top1_positive = pd.to_numeric(pd.Series([top_row.get("label_day_top1_feasible_positive")]), errors="coerce").iloc[0]
            label_positive_vs_nonpositive = (
                int((top1_positive == 1) and (not pd.isna(other_ev)) and (float(other_ev) <= 0.0))
                if not pd.isna(top1_positive)
                else np.nan
            )
            pair_rows.append(
                {
                    "run_date": str(run_date),
                    "research_envelope": _normalize_research_envelope_label(research_corr_mode, research_corr_threshold),
                    "top_candidate_id": str(top_row.get("candidate_id", "")),
                    "other_candidate_id": str(other.get("candidate_id", "")),
                    "label_top1_outranks_other": 1,
                    "label_top1_positive_outranks_nonpositive_other": label_positive_vs_nonpositive,
                    "label_day_top1_feasible_positive": top1_positive,
                    "label_day_top1_feasible_improves_edge": pd.to_numeric(pd.Series([top_row.get("label_day_top1_feasible_improves_edge")]), errors="coerce").iloc[0],
                    "label_day_top1_value_delta_vs_edge": pd.to_numeric(pd.Series([top_row.get("label_day_top1_value_delta_vs_edge")]), errors="coerce").iloc[0],
                    "label_day_is_missed_positive_top1": int(top_row.get("label_day_is_missed_positive_top1", 0)),
                    "top_decision_time_score": pd.to_numeric(pd.Series([top_row.get("decision_time_score")]), errors="coerce").iloc[0],
                    "other_decision_time_score": pd.to_numeric(pd.Series([other.get("decision_time_score")]), errors="coerce").iloc[0],
                    "delta_decision_time_score": (
                        pd.to_numeric(pd.Series([top_row.get("decision_time_score")]), errors="coerce").iloc[0]
                        - pd.to_numeric(pd.Series([other.get("decision_time_score")]), errors="coerce").iloc[0]
                    ),
                    "top_edge": pd.to_numeric(pd.Series([top_row.get("edge")]), errors="coerce").iloc[0],
                    "other_edge": pd.to_numeric(pd.Series([other.get("edge")]), errors="coerce").iloc[0],
                    "delta_edge": (
                        pd.to_numeric(pd.Series([top_row.get("edge")]), errors="coerce").iloc[0]
                        - pd.to_numeric(pd.Series([other.get("edge")]), errors="coerce").iloc[0]
                    ),
                    "top_confidence": pd.to_numeric(pd.Series([top_row.get("final_confidence")]), errors="coerce").iloc[0],
                    "other_confidence": pd.to_numeric(pd.Series([other.get("final_confidence")]), errors="coerce").iloc[0],
                    "delta_confidence": (
                        pd.to_numeric(pd.Series([top_row.get("final_confidence")]), errors="coerce").iloc[0]
                        - pd.to_numeric(pd.Series([other.get("final_confidence")]), errors="coerce").iloc[0]
                    ),
                }
            )
    pair_df = pd.DataFrame.from_records(pair_rows)

    summary = {
        "research_envelope": _normalize_research_envelope_label(research_corr_mode, research_corr_threshold),
        "candidate_rows": int(len(candidates)),
        "pair_rows": int(len(pair_df)),
        "candidate_days": int(candidates["run_date"].astype(str).nunique()),
        "candidate_top1_rows": int((candidates["candidate_is_day_top1_feasible"] == 1).sum()),
        "candidate_rows_missed_positive_top1_days": int((candidates["label_day_is_missed_positive_top1"] == 1).sum()),
        "candidate_rows_matched_non_top1_from_missed_positive_days": int((candidates["label_candidate_is_matched_non_top1_from_missed_positive_day"] == 1).sum()),
        "pair_rows_missed_positive_top1_days": int((pd.to_numeric(pair_df.get("label_day_is_missed_positive_top1"), errors="coerce") == 1).sum()) if not pair_df.empty else 0,
    }
    return candidates, pair_df, summary


def _build_daily_context_dataset(
    feature_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    abstain_df: pd.DataFrame,
    *,
    research_pool_min_agreement: float = 1.0,
    research_corr_mode: str = "none",
    research_corr_threshold: float | None = None,
) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    def _nunique_nonempty(df: pd.DataFrame, col: str) -> int:
        if col not in df.columns or df.empty:
            return 0
        series = df[col].astype(str).str.strip()
        series = series.loc[series != ""]
        return int(series.nunique())

    def _max_count_nonempty(df: pd.DataFrame, col: str) -> int:
        if col not in df.columns or df.empty:
            return 0
        series = df[col].astype(str).str.strip()
        series = series.loc[series != ""]
        if series.empty:
            return 0
        return int(series.value_counts(dropna=False).max())

    abstain_by_date: dict[str, dict[str, Any]] = {}
    if not abstain_df.empty and "run_date" in abstain_df.columns:
        tmp = abstain_df.copy()
        tmp["run_date"] = tmp["run_date"].astype(str)
        for _, row in tmp.drop_duplicates(subset=["run_date"], keep="last").iterrows():
            abstain_by_date[str(row.get("run_date", ""))] = row.to_dict()

    run_dates = sorted(feature_df["run_date"].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []

    for run_date in run_dates:
        day_features = feature_df.loc[feature_df["run_date"].astype(str) == str(run_date)].copy()
        if day_features.empty:
            continue
        day_head = day_features.iloc[0]
        board = day_features.loc[day_features["is_base_board_member"] == True].copy()
        pool = day_features.loc[day_features["is_append_pool"] == True].copy()
        if not pool.empty:
            pool["decision_time_score"] = _decision_time_append_score(pool)
        else:
            pool["decision_time_score"] = np.nan
        pool_resolved = pool.loc[pd.to_numeric(pool.get("ev_label"), errors="coerce").notna()].copy()
        pool_agreement = pd.to_numeric(pool.get("agreement_count"), errors="coerce").fillna(0.0)
        pool_corr = pd.to_numeric(pool.get("corr_score"), errors="coerce")
        feasible_mask = pool_agreement >= float(research_pool_min_agreement)
        mode = str(research_corr_mode or "none").lower()
        threshold = None if research_corr_threshold is None else float(research_corr_threshold)
        if mode == "absolute" and threshold is not None and not np.isinf(threshold):
            feasible_mask = feasible_mask & (pool_corr.isna() | (pool_corr <= float(threshold)))
        elif mode == "percentile" and threshold is not None:
            corr_pct = pd.to_numeric(pool_corr.rank(method="average", pct=True), errors="coerce")
            feasible_mask = feasible_mask & (corr_pct <= float(threshold))
        elif mode == "zscore" and threshold is not None:
            corr_z = _zscore_within_group(pool_corr.fillna(0.0))
            feasible_mask = feasible_mask & (pd.to_numeric(corr_z, errors="coerce") <= float(threshold))
        pool_feasible = pool.loc[feasible_mask].copy()
        pool_feasible_resolved = pool_feasible.loc[pd.to_numeric(pool_feasible.get("ev_label"), errors="coerce").notna()].copy()

        edge_values = pd.to_numeric(pool.get("edge"), errors="coerce").dropna().sort_values(ascending=False).tolist()
        conf_values = pd.to_numeric(pool.get("final_confidence"), errors="coerce").dropna().sort_values(ascending=False).tolist()

        pool_edge_top1 = float(edge_values[0]) if len(edge_values) >= 1 else np.nan
        pool_edge_top2 = float(edge_values[1]) if len(edge_values) >= 2 else np.nan
        pool_edge_gap_1_2 = float(pool_edge_top1 - pool_edge_top2) if len(edge_values) >= 2 else np.nan
        pool_conf_top1 = float(conf_values[0]) if len(conf_values) >= 1 else np.nan
        pool_conf_top2 = float(conf_values[1]) if len(conf_values) >= 2 else np.nan
        pool_conf_gap_1_2 = float(pool_conf_top1 - pool_conf_top2) if len(conf_values) >= 2 else np.nan

        if not pool_resolved.empty:
            ranked_pool = pool_resolved.sort_values(
                ["ev_label", "edge", "final_confidence"],
                ascending=[False, False, False],
            ).copy()
            best_pool = ranked_pool.iloc[0]
            best_pool_player = str(best_pool.get("player", ""))
            best_pool_target = str(best_pool.get("target", ""))
            best_pool_direction = str(best_pool.get("direction", ""))
            best_pool_result = str(best_pool.get("result", ""))
            best_pool_ev_label = _safe_num(best_pool.get("ev_label"), default=np.nan)
            best_pool_edge_rank = int(_safe_num(best_pool.get("edge_rank"), default=np.nan)) if not pd.isna(_safe_num(best_pool.get("edge_rank"), default=np.nan)) else 0
        else:
            best_pool_player = ""
            best_pool_target = ""
            best_pool_direction = ""
            best_pool_result = ""
            best_pool_ev_label = np.nan
            best_pool_edge_rank = 0

        pool_resolved_mean_ev = float(pd.to_numeric(pool_resolved.get("ev_label"), errors="coerce").mean()) if not pool_resolved.empty else np.nan
        if not pool_feasible_resolved.empty:
            ranked_feasible = pool_feasible_resolved.sort_values(
                ["ev_label", "edge", "final_confidence"],
                ascending=[False, False, False],
            ).copy()
            best_feasible = ranked_feasible.iloc[0]
            best_feasible_player = str(best_feasible.get("player", ""))
            best_feasible_target = str(best_feasible.get("target", ""))
            best_feasible_direction = str(best_feasible.get("direction", ""))
            best_feasible_result = str(best_feasible.get("result", ""))
            best_feasible_ev_label = _safe_num(best_feasible.get("ev_label"), default=np.nan)
            best_feasible_edge_rank = int(_safe_num(best_feasible.get("edge_rank"), default=np.nan)) if not pd.isna(_safe_num(best_feasible.get("edge_rank"), default=np.nan)) else 0
            feasible_positive_count = int((pd.to_numeric(pool_feasible_resolved.get("ev_label"), errors="coerce") > 0.0).sum())
        else:
            best_feasible_player = ""
            best_feasible_target = ""
            best_feasible_direction = ""
            best_feasible_result = ""
            best_feasible_ev_label = np.nan
            best_feasible_edge_rank = 0
            feasible_positive_count = 0

        top1_feasible_exists = bool(not pool_feasible.empty)
        top1_feasible_player = ""
        top1_feasible_target = ""
        top1_feasible_direction = ""
        top1_feasible_result = ""
        top1_feasible_ev_label = np.nan
        top1_feasible_edge_rank = 0
        top1_feasible_score = np.nan
        top1_feasible_resolved = 0
        top1_feasible_edge = np.nan
        top1_feasible_confidence = np.nan
        top1_feasible_agreement = np.nan
        top1_feasible_corr_score = np.nan
        top1_feasible_corr_percentile_in_pool = np.nan
        top1_feasible_market_books = np.nan
        top1_feasible_history_rows = np.nan
        top1_feasible_edge_percentile = np.nan
        top1_shadow_rule_pass_agreement = np.nan
        top1_shadow_rule_pass_edge_percentile = np.nan
        top1_shadow_rule_pass_market_books = np.nan
        top1_shadow_rule_pass_history_rows = np.nan
        top1_shadow_rule_pass_confidence = np.nan
        top1_shadow_rule_eligible = np.nan
        top1_shadow_rule_fail_reasons = ""
        top1_shadow_rule_rank_all_candidates = np.nan
        top1_shadow_rule_rank_pool_candidates = np.nan
        top1_market_line = np.nan
        if top1_feasible_exists:
            ranked_top1 = pool_feasible.sort_values(
                ["decision_time_score", "edge", "final_confidence", "agreement_count"],
                ascending=[False, False, False, False],
            ).copy()
            top1 = ranked_top1.iloc[0]
            top1_feasible_player = str(top1.get("player", ""))
            top1_feasible_target = str(top1.get("target", ""))
            top1_feasible_direction = str(top1.get("direction", ""))
            top1_feasible_result = str(top1.get("result", ""))
            top1_feasible_score = _safe_num(top1.get("decision_time_score"), default=np.nan)
            top1_feasible_ev_label = _safe_num(top1.get("ev_label"), default=np.nan)
            top1_feasible_edge_rank = int(_safe_num(top1.get("edge_rank"), default=np.nan)) if not pd.isna(_safe_num(top1.get("edge_rank"), default=np.nan)) else 0
            top1_feasible_resolved = int(top1_feasible_result in {"win", "loss"})
            top1_feasible_edge = _safe_num(top1.get("edge"), default=np.nan)
            top1_feasible_confidence = _safe_num(top1.get("final_confidence"), default=np.nan)
            top1_feasible_agreement = _safe_num(top1.get("agreement_count"), default=np.nan)
            top1_feasible_corr_score = _safe_num(top1.get("corr_score"), default=np.nan)
            top1_feasible_market_books = _safe_num(top1.get("market_books"), default=np.nan)
            top1_feasible_history_rows = _safe_num(top1.get("history_rows"), default=np.nan)
            top1_feasible_edge_percentile = _safe_num(top1.get("edge_percentile"), default=np.nan)
            top1_market_line = _safe_num(top1.get("market_line"), default=np.nan)

            if not pool.empty:
                pool_corr_pct = pd.to_numeric(pool_corr.rank(method="average", pct=True), errors="coerce")
                top1_feasible_corr_percentile_in_pool = _safe_num(pool_corr_pct.loc[top1.name], default=np.nan) if top1.name in pool_corr_pct.index else np.nan

            pass_agreement = bool(top1_feasible_agreement >= float(SHADOW_APPEND_AGREEMENT_MIN))
            pass_edge_pct = bool(top1_feasible_edge_percentile >= float(SHADOW_APPEND_EDGE_PERCENTILE_MIN))
            pass_books = bool(top1_feasible_market_books >= float(SHADOW_APPEND_MARKET_BOOKS_MIN))
            pass_history = bool(top1_feasible_history_rows >= float(SHADOW_APPEND_HISTORY_ROWS_MIN))
            pass_confidence = bool(top1_feasible_confidence >= float(SHADOW_APPEND_CONFIDENCE_MIN))
            top1_shadow_rule_pass_agreement = int(pass_agreement)
            top1_shadow_rule_pass_edge_percentile = int(pass_edge_pct)
            top1_shadow_rule_pass_market_books = int(pass_books)
            top1_shadow_rule_pass_history_rows = int(pass_history)
            top1_shadow_rule_pass_confidence = int(pass_confidence)
            top1_shadow_rule_eligible = int(pass_agreement and pass_edge_pct and pass_books and pass_history and pass_confidence)
            top1_shadow_rule_fail_reasons = _compose_shadow_gate_fail_reasons(
                pass_agreement=pass_agreement,
                pass_edge_pct=pass_edge_pct,
                pass_books=pass_books,
                pass_history=pass_history,
                pass_confidence=pass_confidence,
            )

            # Rank position of top1 within shadow-rule eligible universes.
            shadow_universe = day_features.loc[~day_features["is_base_board_member"].astype(bool)].copy()
            if not shadow_universe.empty:
                shadow_eligible_all = shadow_universe.loc[_shadow_append_eligibility_mask(shadow_universe)].copy()
                if not shadow_eligible_all.empty:
                    shadow_eligible_all = shadow_eligible_all.sort_values(
                        ["agreement_count", "edge_percentile", "edge", "expected_win_rate", "final_confidence"],
                        ascending=[False, False, False, False, False],
                    ).copy()
                    top1_key_all = _play_key_from_values(
                        top1_feasible_player,
                        top1_feasible_target,
                        top1_feasible_direction,
                        top1_market_line,
                    )
                    shadow_eligible_all["shadow_key"] = shadow_eligible_all.apply(
                        lambda r: _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line")),
                        axis=1,
                    )
                    ranked_keys_all = shadow_eligible_all["shadow_key"].astype(str).tolist()
                    if top1_key_all in ranked_keys_all:
                        top1_shadow_rule_rank_all_candidates = int(ranked_keys_all.index(top1_key_all) + 1)

            if not pool.empty:
                shadow_eligible_pool = pool.loc[_shadow_append_eligibility_mask(pool)].copy()
                if not shadow_eligible_pool.empty:
                    shadow_eligible_pool = shadow_eligible_pool.sort_values(
                        ["agreement_count", "edge_percentile", "edge", "expected_win_rate", "final_confidence"],
                        ascending=[False, False, False, False, False],
                    ).copy()
                    top1_key_pool = _play_key_from_values(
                        top1_feasible_player,
                        top1_feasible_target,
                        top1_feasible_direction,
                        top1_market_line,
                    )
                    shadow_eligible_pool["shadow_key"] = shadow_eligible_pool.apply(
                        lambda r: _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line")),
                        axis=1,
                    )
                    ranked_keys_pool = shadow_eligible_pool["shadow_key"].astype(str).tolist()
                    if top1_key_pool in ranked_keys_pool:
                        top1_shadow_rule_rank_pool_candidates = int(ranked_keys_pool.index(top1_key_pool) + 1)

        day_eval = eval_df.loc[eval_df["run_date"].astype(str) == str(run_date)].copy() if not eval_df.empty else pd.DataFrame()
        edge_day = day_eval.loc[day_eval["variant"] == "edge_baseline"].copy()
        shadow_day = day_eval.loc[day_eval["variant"] == "shadow_append_a1_p90_x1"].copy()
        unified_day = day_eval.loc[day_eval["variant"] == "unified_shadow_meta_x1"].copy()

        edge_metrics = _metrics(edge_day)
        shadow_metrics = _metrics(shadow_day)
        unified_metrics = _metrics(unified_day)

        edge_keys = {
            _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line"))
            for _, r in edge_day.iterrows()
        }
        shadow_only = shadow_day.loc[
            ~shadow_day.apply(
                lambda r: _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line")) in edge_keys,
                axis=1,
            )
        ].copy() if not shadow_day.empty else pd.DataFrame()
        shadow_candidate_exists = bool(not shadow_only.empty)
        shadow_candidate_player = ""
        shadow_candidate_target = ""
        shadow_candidate_direction = ""
        shadow_candidate_result = ""
        shadow_candidate_edge = np.nan
        shadow_candidate_conf = np.nan
        shadow_candidate_agreement = np.nan
        shadow_candidate_corr = np.nan
        shadow_candidate_ev_label = np.nan
        shadow_candidate_edge_rank = 0
        shadow_candidate_score = np.nan
        shadow_candidate_resolved = 0
        shadow_candidate_in_research_pool = 0
        if shadow_candidate_exists:
            cand = shadow_only.iloc[0]
            shadow_candidate_player = str(cand.get("player", ""))
            shadow_candidate_target = str(cand.get("target", ""))
            shadow_candidate_direction = str(cand.get("direction", ""))
            shadow_candidate_result = str(cand.get("result", ""))
            shadow_candidate_ev_label = (100.0 / 110.0) if shadow_candidate_result == "win" else (-1.0 if shadow_candidate_result == "loss" else np.nan)
            shadow_candidate_resolved = int(shadow_candidate_result in {"win", "loss"})

            candidate_feature_row = day_features.loc[
                (day_features["player"].astype(str) == shadow_candidate_player)
                & (day_features["target"].astype(str) == shadow_candidate_target)
                & (day_features["direction"].astype(str) == shadow_candidate_direction)
            ].copy()
            if not candidate_feature_row.empty:
                candidate_feature_row = candidate_feature_row.sort_values(["edge_rank", "edge"], ascending=[True, False]).head(1)
                candidate_row = candidate_feature_row.iloc[0]
                shadow_candidate_edge = _safe_num(candidate_row.get("edge"), default=np.nan)
                shadow_candidate_conf = _safe_num(candidate_row.get("final_confidence"), default=np.nan)
                shadow_candidate_agreement = _safe_num(candidate_row.get("agreement_count"), default=np.nan)
                shadow_candidate_corr = _safe_num(candidate_row.get("corr_score"), default=np.nan)
                rank_value = _safe_num(candidate_row.get("edge_rank"), default=np.nan)
                shadow_candidate_edge_rank = int(rank_value) if not pd.isna(rank_value) else 0
                shadow_candidate_score = _safe_num(
                    _decision_time_append_score(candidate_feature_row).iloc[0],
                    default=np.nan,
                )

            shadow_pool_row = pool.loc[
                (pool["player"].astype(str) == shadow_candidate_player)
                & (pool["target"].astype(str) == shadow_candidate_target)
                & (pool["direction"].astype(str) == shadow_candidate_direction)
            ].copy()
            if not shadow_pool_row.empty:
                shadow_candidate_in_research_pool = 1
                shadow_pool_row = shadow_pool_row.sort_values(["edge_rank", "edge"], ascending=[True, False]).head(1)
                shadow_candidate_score = _safe_num(shadow_pool_row.iloc[0].get("decision_time_score"), default=np.nan)

        abstain_row = abstain_by_date.get(str(run_date), {})
        model_ready = bool(abstain_row.get("model_ready", False))
        day_shallow_enough = bool(abstain_row.get("day_shallow_enough", False))
        meta_reason = str(abstain_row.get("meta_reason", ""))
        unified_reason = str(abstain_row.get("unified_reason", ""))

        delta_shadow_edge_ev = (
            float(shadow_metrics["ev_per_resolved"]) - float(edge_metrics["ev_per_resolved"])
            if int(shadow_metrics.get("resolved", 0)) > 0 and int(edge_metrics.get("resolved", 0)) > 0
            else np.nan
        )
        delta_unified_edge_ev = (
            float(unified_metrics["ev_per_resolved"]) - float(edge_metrics["ev_per_resolved"])
            if int(unified_metrics.get("resolved", 0)) > 0 and int(edge_metrics.get("resolved", 0)) > 0
            else np.nan
        )

        label_best_append_positive = int(best_pool_ev_label > 0.0) if not pd.isna(best_pool_ev_label) else np.nan
        label_shadow_candidate_positive = (
            int(shadow_candidate_ev_label > 0.0) if not pd.isna(shadow_candidate_ev_label) else np.nan
        )
        if shadow_candidate_exists and shadow_candidate_result in {"win", "loss"}:
            label_abstain_correct = int(shadow_candidate_result == "loss")
        elif (not shadow_candidate_exists) and not pd.isna(best_pool_ev_label):
            label_abstain_correct = int(best_pool_ev_label <= 0.0)
        else:
            label_abstain_correct = np.nan

        label_shadow_day_improves_edge = int(delta_shadow_edge_ev > 0.0) if not pd.isna(delta_shadow_edge_ev) else np.nan
        label_unified_day_improves_edge = int(delta_unified_edge_ev > 0.0) if not pd.isna(delta_unified_edge_ev) else np.nan
        label_unified_veto_correct = (
            int(shadow_candidate_result == "loss")
            if ("veto" in unified_reason.lower() and shadow_candidate_result in {"win", "loss"})
            else np.nan
        )
        best_feasible_append_uplift_vs_pool_mean = (
            float(best_feasible_ev_label - pool_resolved_mean_ev)
            if (not pd.isna(best_feasible_ev_label) and not pd.isna(pool_resolved_mean_ev))
            else np.nan
        )
        best_feasible_append_delta_vs_edge_mean = (
            float(best_feasible_ev_label - float(edge_metrics.get("ev_per_resolved", np.nan)))
            if (not pd.isna(best_feasible_ev_label) and int(edge_metrics.get("resolved", 0)) > 0)
            else np.nan
        )
        top1_feasible_value_delta = (
            float(top1_feasible_ev_label - float(edge_metrics.get("ev_per_resolved", np.nan)))
            if (not pd.isna(top1_feasible_ev_label) and int(edge_metrics.get("resolved", 0)) > 0)
            else np.nan
        )
        top1_feasible_uplift_vs_pool_mean = (
            float(top1_feasible_ev_label - pool_resolved_mean_ev)
            if (not pd.isna(top1_feasible_ev_label) and not pd.isna(pool_resolved_mean_ev))
            else np.nan
        )
        label_top1_feasible_positive = (
            int(top1_feasible_ev_label > 0.0)
            if not pd.isna(top1_feasible_ev_label)
            else np.nan
        )
        label_top1_feasible_improves_edge = (
            int(top1_feasible_value_delta > 0.0)
            if not pd.isna(top1_feasible_value_delta)
            else np.nan
        )
        shadow_matches_top1_feasible = (
            int(
                shadow_candidate_exists
                and top1_feasible_exists
                and shadow_candidate_player == top1_feasible_player
                and shadow_candidate_target == top1_feasible_target
                and shadow_candidate_direction == top1_feasible_direction
            )
            if top1_feasible_exists
            else np.nan
        )
        label_shadow_missed_positive_top1 = (
            int((label_top1_feasible_positive == 1) and (shadow_matches_top1_feasible != 1))
            if not pd.isna(label_top1_feasible_positive)
            else np.nan
        )
        top1_shadow_score_gap = (
            float(top1_feasible_score - shadow_candidate_score)
            if (not pd.isna(top1_feasible_score) and not pd.isna(shadow_candidate_score))
            else np.nan
        )
        shadow_top1_miss_stage = ""
        shadow_top1_miss_reason = ""
        shadow_top1_generation_blocked = np.nan
        shadow_top1_safety_blocked = np.nan
        if (not pd.isna(label_shadow_missed_positive_top1)) and int(label_shadow_missed_positive_top1) == 1:
            if int(_safe_num(top1_shadow_rule_eligible, default=0.0)) != 1:
                shadow_top1_miss_stage = "candidate_generation"
                shadow_top1_miss_reason = (
                    f"shadow_gate_blocked:{top1_shadow_rule_fail_reasons}"
                    if top1_shadow_rule_fail_reasons
                    else "shadow_gate_blocked"
                )
                shadow_top1_generation_blocked = 1
                shadow_top1_safety_blocked = 0
            elif shadow_candidate_exists and int(shadow_candidate_in_research_pool) == 0:
                shadow_top1_miss_stage = "candidate_generation"
                shadow_top1_miss_reason = "shadow_selected_outside_research_pool"
                shadow_top1_generation_blocked = 1
                shadow_top1_safety_blocked = 0
            elif not pd.isna(top1_shadow_rule_rank_all_candidates) and float(top1_shadow_rule_rank_all_candidates) > 1.0:
                shadow_top1_miss_stage = "candidate_generation"
                shadow_top1_miss_reason = "shadow_ranked_other_candidate_higher"
                shadow_top1_generation_blocked = 1
                shadow_top1_safety_blocked = 0
            elif not shadow_candidate_exists:
                shadow_top1_miss_stage = "post_generation_safety"
                shadow_top1_miss_reason = "shadow_no_append_after_caps"
                shadow_top1_generation_blocked = 0
                shadow_top1_safety_blocked = 1
            else:
                shadow_top1_miss_stage = "post_generation_safety"
                shadow_top1_miss_reason = "shadow_selected_different_candidate_after_caps"
                shadow_top1_generation_blocked = 0
                shadow_top1_safety_blocked = 1
        label_any_feasible_positive = (
            int(feasible_positive_count > 0)
            if not pool_feasible_resolved.empty
            else np.nan
        )
        label_any_feasible_improves_edge = (
            int(best_feasible_append_delta_vs_edge_mean > 0.0)
            if not pd.isna(best_feasible_append_delta_vs_edge_mean)
            else np.nan
        )
        if not pd.isna(label_any_feasible_positive) and int(label_any_feasible_positive) == 1:
            label_shadow_missed_feasible_positive = int(shadow_candidate_result != "win")
        elif not pd.isna(label_any_feasible_positive):
            label_shadow_missed_feasible_positive = 0
        else:
            label_shadow_missed_feasible_positive = np.nan

        board_target_entropy = np.nan
        if "target" in board.columns and not board.empty:
            probs = board["target"].astype(str).value_counts(normalize=True)
            board_target_entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())

        pool_target_dir_series = (
            pool.get("target", pd.Series("", index=pool.index)).astype(str)
            + "|"
            + pool.get("direction", pd.Series("", index=pool.index)).astype(str)
        ) if not pool.empty else pd.Series(dtype="object")
        pool_target_dir_max_count = int(pool_target_dir_series.value_counts().max()) if not pool_target_dir_series.empty else 0

        rows.append(
            {
                "run_date": str(run_date),
                "snapshot_mode": str(day_head.get("snapshot_mode", "")),
                "snapshot_selected_market_date": str(day_head.get("snapshot_selected_market_date", "")),
                "snapshot_selected_row_count": _safe_num(day_head.get("snapshot_selected_row_count"), default=np.nan),
                "model_ready": model_ready,
                "day_shallow_enough": day_shallow_enough,
                "meta_reason": meta_reason,
                "unified_reason": unified_reason,
                "board_size": int(len(board)),
                "board_player_unique_count": _nunique_nonempty(board, "player"),
                "board_target_unique_count": _nunique_nonempty(board, "target"),
                "board_target_entropy": board_target_entropy,
                "board_edge_mean": _safe_num(day_features.get("board_edge_mean", pd.Series([np.nan])).iloc[0], default=np.nan),
                "board_edge_std": _safe_num(day_features.get("board_edge_std", pd.Series([np.nan])).iloc[0], default=np.nan),
                "board_cutoff_gap": _safe_num(day_features.get("board_cutoff_gap", pd.Series([np.nan])).iloc[0], default=np.nan),
                "board_top_to_cutoff_gap": _safe_num(day_features.get("board_top_to_cutoff_gap", pd.Series([np.nan])).iloc[0], default=np.nan),
                "near_miss_local_slope": _safe_num(day_features.get("near_miss_local_slope", pd.Series([np.nan])).iloc[0], default=np.nan),
                "board_concentration": _safe_num(day_features.get("board_concentration", pd.Series([np.nan])).iloc[0], default=np.nan),
                "pool_size": int(len(pool)),
                "pool_resolved_count": int(len(pool_resolved)),
                "pool_resolved_share": float(len(pool_resolved) / len(pool)) if len(pool) > 0 else np.nan,
                "pool_feasible_count": int(len(pool_feasible)),
                "pool_feasible_resolved_count": int(len(pool_feasible_resolved)),
                "pool_feasible_positive_count": int(feasible_positive_count),
                "pool_player_unique_count": _nunique_nonempty(pool, "player"),
                "pool_target_unique_count": _nunique_nonempty(pool, "target"),
                "pool_game_unique_count": _nunique_nonempty(pool, "game_key"),
                "pool_script_cluster_unique_count": _nunique_nonempty(pool, "script_cluster_id"),
                "pool_game_max_count": _max_count_nonempty(pool, "game_key"),
                "pool_target_dir_max_count": pool_target_dir_max_count,
                "pool_edge_mean": float(pd.to_numeric(pool.get("edge"), errors="coerce").mean()) if not pool.empty else np.nan,
                "pool_edge_std": float(pd.to_numeric(pool.get("edge"), errors="coerce").std(ddof=0)) if not pool.empty else np.nan,
                "pool_conf_mean": float(pd.to_numeric(pool.get("final_confidence"), errors="coerce").mean()) if not pool.empty else np.nan,
                "pool_conf_std": float(pd.to_numeric(pool.get("final_confidence"), errors="coerce").std(ddof=0)) if not pool.empty else np.nan,
                "pool_corr_mean": float(pd.to_numeric(pool.get("corr_score"), errors="coerce").mean()) if not pool.empty else np.nan,
                "pool_corr_std": float(pd.to_numeric(pool.get("corr_score"), errors="coerce").std(ddof=0)) if not pool.empty else np.nan,
                "pool_edge_top1": pool_edge_top1,
                "pool_edge_top2": pool_edge_top2,
                "pool_edge_gap_1_2": pool_edge_gap_1_2,
                "pool_conf_top1": pool_conf_top1,
                "pool_conf_top2": pool_conf_top2,
                "pool_conf_gap_1_2": pool_conf_gap_1_2,
                "pool_agreement_ge1_count": int((pd.to_numeric(pool.get("agreement_count"), errors="coerce").fillna(0.0) >= 1.0).sum()) if not pool.empty else 0,
                "pool_agreement_ge2_count": int((pd.to_numeric(pool.get("agreement_count"), errors="coerce").fillna(0.0) >= 2.0).sum()) if not pool.empty else 0,
                "pool_agreement_ge3_count": int((pd.to_numeric(pool.get("agreement_count"), errors="coerce").fillna(0.0) >= 3.0).sum()) if not pool.empty else 0,
                "best_pool_player": best_pool_player,
                "best_pool_target": best_pool_target,
                "best_pool_direction": best_pool_direction,
                "best_pool_result": best_pool_result,
                "best_pool_ev_label": best_pool_ev_label,
                "best_pool_edge_rank": best_pool_edge_rank,
                "best_feasible_player": best_feasible_player,
                "best_feasible_target": best_feasible_target,
                "best_feasible_direction": best_feasible_direction,
                "best_feasible_result": best_feasible_result,
                "best_feasible_ev_label": best_feasible_ev_label,
                "best_feasible_edge_rank": best_feasible_edge_rank,
                "best_feasible_append_uplift_vs_pool_mean": best_feasible_append_uplift_vs_pool_mean,
                "best_feasible_append_delta_vs_edge_mean": best_feasible_append_delta_vs_edge_mean,
                "top1_feasible_exists": int(top1_feasible_exists),
                "top1_feasible_player": top1_feasible_player,
                "top1_feasible_target": top1_feasible_target,
                "top1_feasible_direction": top1_feasible_direction,
                "top1_feasible_result": top1_feasible_result,
                "top1_feasible_resolved": int(top1_feasible_resolved),
                "top1_feasible_ev_label": top1_feasible_ev_label,
                "top1_feasible_edge_rank": top1_feasible_edge_rank,
                "top1_feasible_decision_score": top1_feasible_score,
                "top1_feasible_market_line": top1_market_line,
                "top1_feasible_uplift_vs_pool_mean": top1_feasible_uplift_vs_pool_mean,
                "top1_feasible_value_delta": top1_feasible_value_delta,
                "top1_feasible_edge": top1_feasible_edge,
                "top1_feasible_confidence": top1_feasible_confidence,
                "top1_feasible_agreement": top1_feasible_agreement,
                "top1_feasible_corr_score": top1_feasible_corr_score,
                "top1_feasible_corr_percentile_in_pool": top1_feasible_corr_percentile_in_pool,
                "top1_feasible_market_books": top1_feasible_market_books,
                "top1_feasible_history_rows": top1_feasible_history_rows,
                "top1_feasible_edge_percentile": top1_feasible_edge_percentile,
                "top1_shadow_rule_pass_agreement": top1_shadow_rule_pass_agreement,
                "top1_shadow_rule_pass_edge_percentile": top1_shadow_rule_pass_edge_percentile,
                "top1_shadow_rule_pass_market_books": top1_shadow_rule_pass_market_books,
                "top1_shadow_rule_pass_history_rows": top1_shadow_rule_pass_history_rows,
                "top1_shadow_rule_pass_confidence": top1_shadow_rule_pass_confidence,
                "top1_shadow_rule_eligible": top1_shadow_rule_eligible,
                "top1_shadow_rule_fail_reasons": top1_shadow_rule_fail_reasons,
                "top1_shadow_rule_rank_all_candidates": top1_shadow_rule_rank_all_candidates,
                "top1_shadow_rule_rank_pool_candidates": top1_shadow_rule_rank_pool_candidates,
                "shadow_candidate_exists": shadow_candidate_exists,
                "shadow_candidate_player": shadow_candidate_player,
                "shadow_candidate_target": shadow_candidate_target,
                "shadow_candidate_direction": shadow_candidate_direction,
                "shadow_candidate_result": shadow_candidate_result,
                "shadow_candidate_resolved": int(shadow_candidate_resolved),
                "shadow_candidate_in_research_pool": int(shadow_candidate_in_research_pool),
                "shadow_candidate_ev_label": shadow_candidate_ev_label,
                "shadow_candidate_edge": shadow_candidate_edge,
                "shadow_candidate_confidence": shadow_candidate_conf,
                "shadow_candidate_agreement": shadow_candidate_agreement,
                "shadow_candidate_corr_score": shadow_candidate_corr,
                "shadow_candidate_decision_score": shadow_candidate_score,
                "shadow_candidate_edge_rank": shadow_candidate_edge_rank,
                "shadow_candidate_uplift_vs_pool_mean": (
                    float(shadow_candidate_ev_label - float(pd.to_numeric(pool_resolved.get("ev_label"), errors="coerce").mean()))
                    if shadow_candidate_resolved and not pool_resolved.empty
                    else np.nan
                ),
                "edge_day_resolved": int(edge_metrics.get("resolved", 0)),
                "edge_day_hit_rate": float(edge_metrics.get("resolved_hit_rate", np.nan)),
                "edge_day_ev_per_resolved": float(edge_metrics.get("ev_per_resolved", np.nan)),
                "shadow_day_resolved": int(shadow_metrics.get("resolved", 0)),
                "shadow_day_hit_rate": float(shadow_metrics.get("resolved_hit_rate", np.nan)),
                "shadow_day_ev_per_resolved": float(shadow_metrics.get("ev_per_resolved", np.nan)),
                "unified_day_resolved": int(unified_metrics.get("resolved", 0)),
                "unified_day_hit_rate": float(unified_metrics.get("resolved_hit_rate", np.nan)),
                "unified_day_ev_per_resolved": float(unified_metrics.get("ev_per_resolved", np.nan)),
                "shadow_minus_edge_ev_per_resolved": delta_shadow_edge_ev,
                "unified_minus_edge_ev_per_resolved": delta_unified_edge_ev,
                "label_best_append_positive": label_best_append_positive,
                "label_shadow_candidate_positive": label_shadow_candidate_positive,
                "label_abstain_correct": label_abstain_correct,
                "label_shadow_day_improves_edge": label_shadow_day_improves_edge,
                "label_unified_day_improves_edge": label_unified_day_improves_edge,
                "label_unified_veto_correct": label_unified_veto_correct,
                "label_any_feasible_positive": label_any_feasible_positive,
                "label_any_feasible_improves_edge": label_any_feasible_improves_edge,
                "label_shadow_missed_feasible_positive": label_shadow_missed_feasible_positive,
                "label_top1_feasible_positive": label_top1_feasible_positive,
                "label_top1_feasible_improves_edge": label_top1_feasible_improves_edge,
                "shadow_matches_top1_feasible": shadow_matches_top1_feasible,
                "label_shadow_missed_positive_top1": label_shadow_missed_positive_top1,
                "top1_shadow_score_gap": top1_shadow_score_gap,
                "shadow_top1_miss_stage": shadow_top1_miss_stage,
                "shadow_top1_miss_reason": shadow_top1_miss_reason,
                "shadow_top1_generation_blocked": shadow_top1_generation_blocked,
                "shadow_top1_safety_blocked": shadow_top1_safety_blocked,
            }
        )

    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = parse_args()
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid date range.")
    if end < start:
        raise ValueError("End date is earlier than start date.")

    run_dirs = _read_run_dirs(args.daily_runs_dir.resolve(), start, end)
    if not run_dirs:
        raise RuntimeError("No daily run folders found in requested range.")

    resolve_actual = _load_actual_lookup()
    resolve_recent_vol = _load_recent_volatility_lookup(lookback_games=5)

    feature_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    day_candidates: dict[str, pd.DataFrame] = {}
    day_base_board: dict[str, pd.DataFrame] = {}
    day_shadow_board: dict[str, pd.DataFrame] = {}
    day_shapes: dict[str, dict[str, float]] = {}

    board_size = int(args.board_size)
    cutoff_low = int(args.cutoff_rank_low)
    cutoff_high = int(args.cutoff_rank_high)
    append_window = int(args.append_window)
    append_rank_low = board_size + 1
    append_rank_high = board_size + append_window
    exclude_snapshot_modes = {str(x).strip().lower() for x in (args.exclude_snapshot_modes or []) if str(x).strip()}

    for run_dir in run_dirs:
        run_date = pd.to_datetime(run_dir.name, format="%Y%m%d", errors="coerce")
        if pd.isna(run_date):
            continue
        run_date_str = run_date.strftime("%Y-%m-%d")
        snapshot_meta = _load_snapshot_meta(run_dir)
        snapshot_mode = str(snapshot_meta.get("mode", "")).strip()
        snapshot_mode_key = snapshot_mode.lower()
        if snapshot_mode_key in exclude_snapshot_modes:
            continue
        snapshot_selected_market_date = snapshot_meta.get("selected_market_date")
        snapshot_selected_row_count = snapshot_meta.get("selected_row_count")
        selector_path = run_dir / f"upcoming_market_play_selector_{run_dir.name}.csv"
        if not selector_path.exists():
            continue
        selector_df = pd.read_csv(selector_path)
        if selector_df.empty:
            continue

        base_board = _build_edge_board(selector_df, board_size)
        shadow_board = _build_shadow_board(selector_df, board_size)
        universe = _build_extended_edge_universe(selector_df, board_size, cutoff_rank_high=cutoff_high, append_window=append_window)
        if universe.empty or base_board.empty:
            continue

        universe = universe.sort_values(["edge", "abs_edge", "expected_win_rate", "final_confidence"], ascending=[False, False, False, False]).copy()
        universe["edge_rank"] = np.arange(1, len(universe) + 1)
        universe = _add_agreement_features(universe, board_size=board_size)

        edge_at_b = _safe_num(universe.loc[universe["edge_rank"] == board_size, "edge"].iloc[0], default=np.nan) if (universe["edge_rank"] == board_size).any() else np.nan
        edge_at_b1 = _safe_num(universe.loc[universe["edge_rank"] == board_size + 1, "edge"].iloc[0], default=np.nan) if (universe["edge_rank"] == board_size + 1).any() else np.nan
        edge_at_top = _safe_num(universe.loc[universe["edge_rank"] == 1, "edge"].iloc[0], default=np.nan) if (universe["edge_rank"] == 1).any() else np.nan
        board_top = universe.loc[universe["edge_rank"] <= board_size].copy()
        board_cutoff_gap = edge_at_b - edge_at_b1 if (not pd.isna(edge_at_b) and not pd.isna(edge_at_b1)) else np.nan
        board_top_to_cutoff = edge_at_top - edge_at_b if (not pd.isna(edge_at_top) and not pd.isna(edge_at_b)) else np.nan
        shape = {
            "board_cutoff_gap": float(board_cutoff_gap) if not pd.isna(board_cutoff_gap) else np.nan,
            "board_top_to_cutoff_gap": float(board_top_to_cutoff) if not pd.isna(board_top_to_cutoff) else np.nan,
            "board_edge_mean": float(pd.to_numeric(board_top["edge"], errors="coerce").mean()) if not board_top.empty else np.nan,
            "board_edge_std": float(pd.to_numeric(board_top["edge"], errors="coerce").std(ddof=0)) if not board_top.empty else np.nan,
        }
        day_shapes[run_date_str] = shape
        day_candidates[run_date_str] = universe.copy()
        day_base_board[run_date_str] = base_board.copy()
        day_shadow_board[run_date_str] = shadow_board.copy()

        base_by_game = base_board.groupby(base_board.get("game_key", pd.Series("", index=base_board.index)).astype(str)).size().to_dict()
        base_by_script = base_board.groupby(base_board.get("script_cluster_id", pd.Series("", index=base_board.index)).astype(str)).size().to_dict()
        base_by_target_dir = (
            base_board.assign(
                _td=base_board.get("target", pd.Series("", index=base_board.index)).astype(str)
                + "|"
                + base_board.get("direction", pd.Series("", index=base_board.index)).astype(str)
            )
            .groupby("_td")
            .size()
            .to_dict()
        )
        base_by_target = base_board.groupby(base_board.get("target", pd.Series("", index=base_board.index)).astype(str)).size().to_dict()

        for _, row in universe.iterrows():
            mdate = pd.to_datetime(row.get("market_date"), errors="coerce")
            if pd.isna(mdate):
                continue
            market_date_str = mdate.strftime("%Y-%m-%d")
            line = pd.to_numeric(row.get("market_line"), errors="coerce")
            line_value = None if pd.isna(line) else float(line)
            actual = resolve_actual(str(row.get("csv", "")), market_date_str, str(row.get("target", "")))
            result = _classify_result(str(row.get("direction", "")), line_value, actual)
            resolved_win_label = 1 if result == "win" else (0 if result == "loss" else np.nan)
            game_key = str(row.get("game_key", ""))
            target = str(row.get("target", ""))
            direction = str(row.get("direction", ""))
            td_key = f"{target}|{direction}"
            script_cluster = str(row.get("script_cluster_id", ""))
            corr_same_game = int(base_by_game.get(game_key, 0))
            corr_same_script = int(base_by_script.get(script_cluster, 0))
            corr_same_td = int(base_by_target_dir.get(td_key, 0))
            corr_score = float(corr_same_game + 0.75 * corr_same_td + 0.50 * corr_same_script)
            expected_wr = _safe_num(row.get("expected_win_rate"))
            raw_expected_wr = _safe_num(row.get("raw_expected_win_rate"))
            breakeven = 110.0 / 210.0

            feature_rows.append(
                {
                    "run_date": run_date_str,
                    "snapshot_mode": snapshot_mode,
                    "snapshot_selected_market_date": snapshot_selected_market_date,
                    "snapshot_selected_row_count": snapshot_selected_row_count,
                    "market_date": market_date_str,
                    "player": row.get("player"),
                    "target": target,
                    "direction": direction,
                    "edge_rank": int(row.get("edge_rank", 0)),
                    "is_cutoff_band": bool(cutoff_low <= int(row.get("edge_rank", 0)) <= cutoff_high),
                    "is_append_pool": bool(append_rank_low <= int(row.get("edge_rank", 0)) <= append_rank_high),
                    "is_base_board_member": bool(int(row.get("edge_rank", 0)) <= board_size),
                    "edge": _safe_num(row.get("edge")),
                    "abs_edge": _safe_num(row.get("abs_edge")),
                    "edge_percentile": _safe_num(row.get("edge_percentile")),
                    "expected_win_rate": expected_wr,
                    "raw_expected_win_rate": raw_expected_wr,
                    "win_rate_minus_breakeven": (expected_wr - breakeven) if not pd.isna(expected_wr) else np.nan,
                    "raw_win_rate_minus_breakeven": (raw_expected_wr - breakeven) if not pd.isna(raw_expected_wr) else np.nan,
                    "final_confidence": _safe_num(row.get("final_confidence")),
                    "market_books": _safe_num(row.get("market_books"), default=0.0),
                    "history_rows": _safe_num(row.get("history_rows")),
                    "agreement_count": _safe_num(row.get("agreement_count")),
                    "set_sources": str(row.get("set_sources", "")),
                    "game_key": str(row.get("game_key", "")),
                    "script_cluster_id": str(row.get("script_cluster_id", "")),
                    "risk_penalty": _safe_num(row.get("risk_penalty")),
                    "uncertainty_sigma": _safe_num(row.get("uncertainty_sigma")),
                    "volatility_score": _safe_num(row.get("volatility_score")),
                    "player_recent_volatility": resolve_recent_vol(str(row.get("csv", "")), market_date_str, target),
                    "tail_imbalance": _safe_num(row.get("tail_imbalance")),
                    "board_cutoff_gap": shape["board_cutoff_gap"],
                    "board_top_to_cutoff_gap": shape["board_top_to_cutoff_gap"],
                    "board_edge_mean": shape["board_edge_mean"],
                    "board_edge_std": shape["board_edge_std"],
                    "corr_same_game_count": corr_same_game,
                    "corr_same_target_dir_count": corr_same_td,
                    "corr_same_script_cluster_count": corr_same_script,
                    "board_target_count_same": int(base_by_target.get(target, 0)),
                    "corr_score": corr_score,
                    "market_line": line_value,
                    "actual": actual,
                    "result": result,
                    "resolved_win_label": resolved_win_label,
                    "win_residual": (resolved_win_label - _safe_num(row.get("expected_win_rate"))) if not pd.isna(resolved_win_label) else np.nan,
                    "ev_label": (100.0 / 110.0) if result == "win" else (-1.0 if result == "loss" else np.nan),
                }
            )

    feature_df = pd.DataFrame.from_records(feature_rows)
    if feature_df.empty:
        raise RuntimeError("No cutoff-band feature rows were produced.")
    feature_df = _add_slate_relative_features(feature_df, board_size=board_size)
    resolved_pool = feature_df.loc[
        (feature_df["is_append_pool"] == True)
        & (feature_df["ev_label"].notna())
    ].copy()
    pool_mean_map = resolved_pool.groupby("run_date")["ev_label"].mean().to_dict()
    pool_count_map = resolved_pool.groupby("run_date")["ev_label"].size().to_dict()
    feature_df["append_pool_resolved_mean_ev"] = feature_df["run_date"].map(pool_mean_map)
    feature_df["append_pool_resolved_count"] = feature_df["run_date"].map(pool_count_map).fillna(0).astype(int)
    feature_df["append_pool_baseline_ev"] = pd.to_numeric(feature_df["append_pool_resolved_mean_ev"], errors="coerce")
    feature_df.loc[
        feature_df["append_pool_baseline_ev"].isna() | (feature_df["append_pool_resolved_count"] < 2),
        "append_pool_baseline_ev",
    ] = 0.0
    feature_df["append_value_target"] = pd.to_numeric(feature_df["ev_label"], errors="coerce") - pd.to_numeric(
        feature_df["append_pool_baseline_ev"], errors="coerce"
    )

    numeric_features, categorical_features = _feature_spec()
    run_dates = sorted(feature_df["run_date"].dropna().astype(str).unique().tolist())
    all_variants = ["edge_baseline", "shadow_append_a1_p90_x1", "cutoff_meta_append_x1", "unified_shadow_meta_x1"]
    daily_rows: list[dict[str, Any]] = []
    abstain_rows: list[dict[str, Any]] = []

    for run_date in run_dates:
        base_board = day_base_board.get(run_date)
        shadow_board = day_shadow_board.get(run_date)
        universe = day_candidates.get(run_date)
        if base_board is None or shadow_board is None or universe is None:
            continue
        if base_board.empty or universe.empty:
            continue

        # Walk-forward training rows (strictly prior run dates, resolved only).
        train = feature_df.loc[
            (feature_df["run_date"] < run_date)
            & (feature_df["is_append_pool"] == True)
            & (feature_df["append_value_target"].notna())
        ].copy()
        model_ready = False
        model = None
        if len(train) >= int(args.min_train_resolved):
            y = pd.to_numeric(train["append_value_target"], errors="coerce").fillna(0.0)
            if float(y.std(ddof=0)) > 1e-9:
                model = _build_model(numeric_features, categorical_features)
                model.fit(train[numeric_features + categorical_features], y)
                model_ready = True

        train_day_shapes = feature_df.loc[
            (feature_df["run_date"] < run_date) & (feature_df["is_base_board_member"] == True),
            ["run_date", "board_cutoff_gap"],
        ].drop_duplicates(subset=["run_date"])
        if not train_day_shapes.empty:
            gap_threshold = float(pd.to_numeric(train_day_shapes["board_cutoff_gap"], errors="coerce").dropna().quantile(float(args.gap_quantile_threshold)))
        else:
            gap_threshold = float("inf")
        day_shape = day_shapes.get(run_date, {})
        day_gap = _safe_num(day_shape.get("board_cutoff_gap"), default=np.nan)
        day_shallow_enough = (not pd.isna(day_gap)) and (day_gap <= gap_threshold)

        # Append pool: B+1..B+append_window.
        day_features = feature_df.loc[feature_df["run_date"] == run_date].copy()
        pool = feature_df.loc[(feature_df["run_date"] == run_date) & (feature_df["is_append_pool"] == True)].copy()
        pool = pool.sort_values(["edge_rank", "edge"], ascending=[True, False])
        appended_player = None
        appended_market_line = None
        appended_target = None
        appended_direction = None
        meta_reason = "abstain_no_model"
        meta_top_value = np.nan
        meta_second_value = np.nan
        meta_value_margin = np.nan
        meta_top_corr = np.nan
        meta_top_agreement = np.nan
        meta_top_player = ""
        if pool.empty:
            meta_reason = "abstain_no_pool"
        elif not model_ready:
            meta_reason = "abstain_no_model"
        else:
            pool["meta_uplift"] = model.predict(pool[numeric_features + categorical_features])
            pool["meta_uplift"] = pd.to_numeric(pool["meta_uplift"], errors="coerce").fillna(0.0)
            pool = pool.sort_values(["meta_uplift", "edge", "expected_win_rate"], ascending=[False, False, False]).copy()
            top = pool.iloc[0]
            meta_top_value = _safe_num(top.get("meta_uplift"), default=-np.inf)
            meta_top_player = str(top.get("player", ""))
            meta_top_corr = _safe_num(top.get("corr_score"), default=np.inf)
            meta_top_agreement = _safe_num(top.get("agreement_count"), default=0.0)
            if len(pool) > 1:
                meta_second_value = _safe_num(pool.iloc[1].get("meta_uplift"), default=-np.inf)
            else:
                meta_second_value = -np.inf
            meta_value_margin = meta_top_value - meta_second_value if np.isfinite(meta_second_value) else np.inf

            passes = (
                bool(day_shallow_enough)
                and (meta_top_value >= float(args.model_uplift_threshold))
                and (meta_value_margin >= float(args.model_uplift_margin))
                and (meta_top_corr <= float(args.max_corr_score))
                and (meta_top_agreement >= 1.0)
            )
            if passes:
                appended_player = str(top.get("player", ""))
                appended_market_line = _safe_num(top.get("market_line"), default=np.nan)
                appended_target = str(top.get("target", ""))
                appended_direction = str(top.get("direction", ""))
                meta_reason = "append_selected"
            else:
                if not bool(day_shallow_enough):
                    meta_reason = "abstain_day_not_shallow"
                elif meta_top_value < float(args.model_uplift_threshold):
                    meta_reason = "abstain_uplift_below_floor"
                elif meta_value_margin < float(args.model_uplift_margin):
                    meta_reason = "abstain_uplift_margin_small"
                elif meta_top_corr > float(args.max_corr_score):
                    meta_reason = "abstain_corr_too_high"
                elif meta_top_agreement < 1.0:
                    meta_reason = "abstain_agreement_too_low"
                else:
                    meta_reason = "abstain_other"

        # Unified candidate: keep shadow append only when the cutoff meta-model confirms it.
        base_board_keys = {
            _play_key_from_values(row.get("player"), row.get("target"), row.get("direction"), row.get("market_line"))
            for _, row in base_board.iterrows()
        }
        shadow_only_rows = shadow_board.loc[
            ~shadow_board.apply(
                lambda r: _play_key_from_values(r.get("player"), r.get("target"), r.get("direction"), r.get("market_line")) in base_board_keys,
                axis=1,
            )
        ].copy()
        unified_appended_row: pd.DataFrame | None = None
        unified_reason = "abstain_no_shadow_candidate"
        unified_shadow_value = np.nan
        unified_shadow_corr = np.nan
        unified_shadow_player = ""
        if not shadow_only_rows.empty:
            shadow_row = shadow_only_rows.iloc[0]
            unified_shadow_player = str(shadow_row.get("player", ""))
            mask = (
                (day_features["player"].astype(str) == str(shadow_row.get("player", "")))
                & (day_features["target"].astype(str) == str(shadow_row.get("target", "")))
                & (day_features["direction"].astype(str) == str(shadow_row.get("direction", "")))
                & (pd.to_numeric(day_features["market_line"], errors="coerce").round(4) == round(_safe_num(shadow_row.get("market_line"), default=np.nan), 4))
            )
            shadow_pool_row = day_features.loc[mask].head(1).copy()
            # Veto-only mode: keep the shadow append by default, and only veto
            # when the learned residual score is materially negative.
            keep_shadow = True
            unified_reason = "append_shadow_default"
            if model_ready and not shadow_pool_row.empty:
                shadow_pool_row["meta_uplift"] = model.predict(shadow_pool_row[numeric_features + categorical_features])
                meta_uplift = _safe_num(shadow_pool_row.iloc[0].get("meta_uplift"), default=0.0)
                unified_shadow_value = meta_uplift
                unified_shadow_corr = _safe_num(shadow_pool_row.iloc[0].get("corr_score"), default=np.inf)
                if unified_shadow_corr > float(args.unified_veto_corr_score):
                    keep_shadow = False
                    unified_reason = "abstain_shadow_corr_veto"
                elif meta_uplift < float(args.unified_shadow_uplift_floor):
                    keep_shadow = False
                    unified_reason = "abstain_shadow_uplift_veto"
                else:
                    unified_reason = "append_shadow_confirmed"
            elif model_ready and shadow_pool_row.empty:
                unified_reason = "append_shadow_no_feature_row"
            if keep_shadow:
                unified_appended_row = shadow_only_rows.head(1).copy()
                unified_appended_row["meta_appended"] = True
                unified_appended_row["unified_source"] = "shadow_confirmed"

        variants_payload: dict[str, pd.DataFrame] = {
            "edge_baseline": base_board.copy(),
            "shadow_append_a1_p90_x1": shadow_board.copy(),
            "cutoff_meta_append_x1": base_board.copy(),
            "unified_shadow_meta_x1": base_board.copy(),
        }
        if appended_player is not None and appended_target is not None and appended_direction is not None and not pd.isna(appended_market_line):
            candidate_row = universe.loc[
                (universe["player"].astype(str) == appended_player)
                & (universe["target"].astype(str) == appended_target)
                & (universe["direction"].astype(str) == appended_direction)
                & (pd.to_numeric(universe["market_line"], errors="coerce").round(4) == round(float(appended_market_line), 4))
            ].head(1)
            if not candidate_row.empty:
                with_flag = candidate_row.copy()
                with_flag["meta_appended"] = True
                board = variants_payload["cutoff_meta_append_x1"].copy()
                board["meta_appended"] = False
                variants_payload["cutoff_meta_append_x1"] = pd.concat([board, with_flag], ignore_index=True)
            else:
                variants_payload["cutoff_meta_append_x1"]["meta_appended"] = False
        else:
            variants_payload["cutoff_meta_append_x1"]["meta_appended"] = False

        if unified_appended_row is not None and not unified_appended_row.empty:
            unified_board = variants_payload["unified_shadow_meta_x1"].copy()
            unified_board["meta_appended"] = False
            unified_board["unified_source"] = ""
            variants_payload["unified_shadow_meta_x1"] = pd.concat([unified_board, unified_appended_row], ignore_index=True)
        else:
            variants_payload["unified_shadow_meta_x1"]["meta_appended"] = False
            variants_payload["unified_shadow_meta_x1"]["unified_source"] = ""

        for variant in all_variants:
            board = variants_payload[variant].copy()
            if board.empty:
                continue
            if "meta_appended" in board.columns:
                board["meta_appended"] = pd.to_numeric(board["meta_appended"], errors="coerce").fillna(0).astype(bool)
            else:
                board["meta_appended"] = False
            for _, row in board.iterrows():
                mdate = pd.to_datetime(row.get("market_date"), errors="coerce")
                if pd.isna(mdate):
                    continue
                if mdate < start or mdate > end:
                    continue
                market_date_str = mdate.strftime("%Y-%m-%d")
                line = pd.to_numeric(row.get("market_line"), errors="coerce")
                line_value = None if pd.isna(line) else float(line)
                actual = resolve_actual(str(row.get("csv", "")), market_date_str, str(row.get("target", "")))
                result = _classify_result(str(row.get("direction", "")), line_value, actual)
                eval_rows.append(
                    {
                        "variant": variant,
                        "run_date": run_date,
                        "market_date": market_date_str,
                        "player": row.get("player"),
                        "target": row.get("target"),
                        "direction": row.get("direction"),
                        "market_line": line_value,
                        "expected_win_rate": _safe_num(row.get("expected_win_rate")),
                        "ev": _safe_num(row.get("ev")),
                        "edge": _safe_num(row.get("edge")),
                        "result": result,
                        "meta_appended": bool(row.get("meta_appended", False)),
                        "unified_source": str(row.get("unified_source", "")),
                        "model_ready": bool(model_ready),
                        "day_shallow_enough": bool(day_shallow_enough),
                        "day_cutoff_gap": day_gap,
                        "gap_threshold": gap_threshold,
                    }
                )

        abstain_rows.append(
            {
                "run_date": run_date,
                "model_ready": bool(model_ready),
                "day_shallow_enough": bool(day_shallow_enough),
                "day_cutoff_gap": day_gap,
                "gap_threshold": gap_threshold,
                "pool_size": int(len(pool)),
                "meta_top_player": meta_top_player,
                "meta_top_uplift": meta_top_value,
                "meta_second_uplift": meta_second_value,
                "meta_uplift_margin": meta_value_margin,
                "meta_top_corr": meta_top_corr,
                "meta_top_agreement": meta_top_agreement,
                "meta_reason": meta_reason,
                "shadow_candidate_exists": bool(not shadow_only_rows.empty),
                "unified_shadow_player": unified_shadow_player,
                "unified_shadow_uplift": unified_shadow_value,
                "unified_shadow_corr": unified_shadow_corr,
                "unified_reason": unified_reason,
            }
        )

        for variant in all_variants:
            subset = pd.DataFrame([r for r in eval_rows if r["run_date"] == run_date and r["variant"] == variant])
            metrics = _metrics(subset)
            daily_rows.append({"run_date": run_date, "variant": variant, **metrics})

    eval_df = pd.DataFrame.from_records(eval_rows)
    if eval_df.empty:
        raise RuntimeError("No evaluation rows produced.")
    daily_df = pd.DataFrame.from_records(daily_rows).drop_duplicates(subset=["run_date", "variant"], keep="last")
    abstain_df = pd.DataFrame.from_records(abstain_rows)
    research_corr_mode = str(args.research_corr_mode or "auto").lower()
    if research_corr_mode == "auto":
        if args.research_corr_percentile_max is not None:
            research_corr_mode = "percentile"
        elif args.research_corr_zscore_max is not None:
            research_corr_mode = "zscore"
        elif args.research_pool_max_corr_score is not None:
            research_corr_mode = "absolute"
        else:
            research_corr_mode = "none"

    if research_corr_mode == "none":
        research_corr_threshold = None
    elif research_corr_mode == "absolute":
        research_corr_threshold = (
            float(args.research_pool_max_corr_score)
            if args.research_pool_max_corr_score is not None
            else float("inf")
        )
    elif research_corr_mode == "percentile":
        research_corr_threshold = (
            float(args.research_corr_percentile_max)
            if args.research_corr_percentile_max is not None
            else 1.0
        )
        research_corr_threshold = min(max(research_corr_threshold, 0.0), 1.0)
    elif research_corr_mode == "zscore":
        research_corr_threshold = (
            float(args.research_corr_zscore_max)
            if args.research_corr_zscore_max is not None
            else 0.0
        )
    else:
        raise ValueError(f"Unsupported research corr mode: {research_corr_mode}")

    daily_context_df = _build_daily_context_dataset(
        feature_df,
        eval_df,
        abstain_df,
        research_pool_min_agreement=float(args.research_pool_min_agreement),
        research_corr_mode=research_corr_mode,
        research_corr_threshold=research_corr_threshold,
    )

    summary: dict[str, Any] = {}
    for variant in all_variants:
        subset = eval_df.loc[eval_df["variant"] == variant].copy()
        summary[variant] = _metrics(subset)

    base_daily = daily_df.loc[daily_df["variant"] == "edge_baseline"].copy()
    resolved_only_days = sorted(base_daily.loc[pd.to_numeric(base_daily["missing"], errors="coerce").fillna(0).eq(0), "run_date"].astype(str).tolist())
    summary["resolved_only_window"] = {
        "run_dates": resolved_only_days,
        "count": int(len(resolved_only_days)),
    }
    summary["resolved_only_metrics"] = {}
    if resolved_only_days:
        for variant in all_variants:
            subset = eval_df.loc[(eval_df["variant"] == variant) & (eval_df["run_date"].astype(str).isin(resolved_only_days))].copy()
            summary["resolved_only_metrics"][variant] = _metrics(subset)

    meta_append_only = eval_df.loc[(eval_df["variant"] == "cutoff_meta_append_x1") & (eval_df["meta_appended"] == True)].copy()
    summary["cutoff_meta_appended_only"] = _metrics(meta_append_only)
    unified_append_only = eval_df.loc[(eval_df["variant"] == "unified_shadow_meta_x1") & (eval_df["meta_appended"] == True)].copy()
    summary["unified_appended_only"] = _metrics(unified_append_only)

    base = summary.get("edge_baseline", {})
    meta = summary.get("cutoff_meta_append_x1", {})
    if base and meta and int(base.get("resolved", 0)) > 0 and int(meta.get("resolved", 0)) > 0:
        summary["delta_meta_minus_edge"] = {
            "delta_resolved": int(meta["resolved"]) - int(base["resolved"]),
            "delta_wins": int(meta["wins"]) - int(base["wins"]),
            "delta_losses": int(meta["losses"]) - int(base["losses"]),
            "delta_hit_rate_pp": 100.0 * (float(meta["resolved_hit_rate"]) - float(base["resolved_hit_rate"])),
            "delta_ev_per_resolved": float(meta["ev_per_resolved"]) - float(base["ev_per_resolved"]),
        }
    else:
        summary["delta_meta_minus_edge"] = {}

    unified = summary.get("unified_shadow_meta_x1", {})
    if base and unified and int(base.get("resolved", 0)) > 0 and int(unified.get("resolved", 0)) > 0:
        summary["delta_unified_minus_edge"] = {
            "delta_resolved": int(unified["resolved"]) - int(base["resolved"]),
            "delta_wins": int(unified["wins"]) - int(base["wins"]),
            "delta_losses": int(unified["losses"]) - int(base["losses"]),
            "delta_hit_rate_pp": 100.0 * (float(unified["resolved_hit_rate"]) - float(base["resolved_hit_rate"])),
            "delta_ev_per_resolved": float(unified["ev_per_resolved"]) - float(base["ev_per_resolved"]),
        }
    else:
        summary["delta_unified_minus_edge"] = {}

    model_day_stats = eval_df.loc[eval_df["variant"] == "cutoff_meta_append_x1", ["run_date", "model_ready", "day_shallow_enough"]].drop_duplicates()
    summary["meta_gate_diagnostics"] = {
        "run_days": int(len(model_day_stats)),
        "model_ready_days": int(model_day_stats["model_ready"].sum()) if not model_day_stats.empty else 0,
        "shallow_days": int(model_day_stats["day_shallow_enough"].sum()) if not model_day_stats.empty else 0,
        "appended_rows_total": int(meta_append_only.shape[0]),
        "appended_resolved_total": int((meta_append_only["result"].isin(["win", "loss"])).sum()) if not meta_append_only.empty else 0,
        "abstain_reasons": (
            abstain_df["meta_reason"].value_counts(dropna=False).to_dict()
            if not abstain_df.empty and "meta_reason" in abstain_df.columns
            else {}
        ),
    }
    summary["unified_gate_diagnostics"] = {
        "appended_rows_total": int(unified_append_only.shape[0]),
        "appended_resolved_total": int((unified_append_only["result"].isin(["win", "loss"])).sum()) if not unified_append_only.empty else 0,
        "appended_sources": (
            unified_append_only["unified_source"].value_counts(dropna=False).to_dict()
            if not unified_append_only.empty and "unified_source" in unified_append_only.columns
            else {}
        ),
        "abstain_reasons": (
            abstain_df["unified_reason"].value_counts(dropna=False).to_dict()
            if not abstain_df.empty and "unified_reason" in abstain_df.columns
            else {}
        ),
    }
    summary["unified_policy"] = {
        "shadow_pass_through_default": True,
        "meta_fallback_enabled": False,
        "veto_uplift_floor": float(args.unified_shadow_uplift_floor),
        "veto_corr_ceiling": float(args.unified_veto_corr_score),
    }
    summary["shadow_append_rule"] = {
        "agreement_min": float(SHADOW_APPEND_AGREEMENT_MIN),
        "edge_percentile_min": float(SHADOW_APPEND_EDGE_PERCENTILE_MIN),
        "market_books_min": float(SHADOW_APPEND_MARKET_BOOKS_MIN),
        "history_rows_min": float(SHADOW_APPEND_HISTORY_ROWS_MIN),
        "confidence_min": float(SHADOW_APPEND_CONFIDENCE_MIN),
    }
    summary["feature_spec"] = {
        "near_cutoff_rank_range": [cutoff_low, cutoff_high],
        "append_pool_rank_range": [append_rank_low, append_rank_high],
        "training_target": "append_value_target",
        "secondary_targets": ["win_residual", "resolved_win_label", "ev_label"],
        "meta_model": "RandomForestRegressor",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    summary["research_pool_config"] = {
        "min_agreement": float(args.research_pool_min_agreement),
        "corr_mode": str(research_corr_mode),
        "corr_threshold": (
            None
            if (research_corr_threshold is None or (isinstance(research_corr_threshold, float) and np.isinf(research_corr_threshold)))
            else float(research_corr_threshold)
        ),
        "max_corr_score_absolute": (
            None
            if args.research_pool_max_corr_score is None
            else float(args.research_pool_max_corr_score)
        ),
        "corr_percentile_max": (
            None
            if args.research_corr_percentile_max is None
            else float(args.research_corr_percentile_max)
        ),
        "corr_zscore_max": (
            None
            if args.research_corr_zscore_max is None
            else float(args.research_corr_zscore_max)
        ),
        "top1_decision_score": {
            "edge_z": 0.55,
            "confidence_z": 0.25,
            "agreement_z": 0.20,
            "corr_z_penalty": -0.20,
            "edge_percentile": 0.10,
            "confidence_level": 0.05,
        },
    }
    summary["daily_context_dataset"] = {
        "rows": int(len(daily_context_df)),
        "model_ready_days": int(pd.to_numeric(daily_context_df.get("model_ready"), errors="coerce").fillna(0).astype(bool).sum()) if not daily_context_df.empty else 0,
        "shallow_days": int(pd.to_numeric(daily_context_df.get("day_shallow_enough"), errors="coerce").fillna(0).astype(bool).sum()) if not daily_context_df.empty else 0,
        "shadow_candidate_days": int(pd.to_numeric(daily_context_df.get("shadow_candidate_exists"), errors="coerce").fillna(0).astype(bool).sum()) if not daily_context_df.empty else 0,
        "pool_resolved_days": int(pd.to_numeric(daily_context_df.get("pool_resolved_count"), errors="coerce").fillna(0).gt(0).sum()) if not daily_context_df.empty else 0,
        "pool_feasible_days": int(pd.to_numeric(daily_context_df.get("pool_feasible_count"), errors="coerce").fillna(0).gt(0).sum()) if not daily_context_df.empty else 0,
        "pool_feasible_resolved_days": int(pd.to_numeric(daily_context_df.get("pool_feasible_resolved_count"), errors="coerce").fillna(0).gt(0).sum()) if not daily_context_df.empty else 0,
        "snapshot_mode_counts": (
            daily_context_df.get("snapshot_mode", pd.Series(dtype="object")).fillna("").astype(str).value_counts(dropna=False).to_dict()
            if not daily_context_df.empty
            else {}
        ),
        "label_coverage": {
            "label_best_append_positive": int(pd.to_numeric(daily_context_df.get("label_best_append_positive"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_shadow_candidate_positive": int(pd.to_numeric(daily_context_df.get("label_shadow_candidate_positive"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_abstain_correct": int(pd.to_numeric(daily_context_df.get("label_abstain_correct"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_shadow_day_improves_edge": int(pd.to_numeric(daily_context_df.get("label_shadow_day_improves_edge"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_unified_veto_correct": int(pd.to_numeric(daily_context_df.get("label_unified_veto_correct"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_any_feasible_positive": int(pd.to_numeric(daily_context_df.get("label_any_feasible_positive"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_any_feasible_improves_edge": int(pd.to_numeric(daily_context_df.get("label_any_feasible_improves_edge"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_shadow_missed_feasible_positive": int(pd.to_numeric(daily_context_df.get("label_shadow_missed_feasible_positive"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_top1_feasible_positive": int(pd.to_numeric(daily_context_df.get("label_top1_feasible_positive"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_top1_feasible_improves_edge": int(pd.to_numeric(daily_context_df.get("label_top1_feasible_improves_edge"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "shadow_matches_top1_feasible": int(pd.to_numeric(daily_context_df.get("shadow_matches_top1_feasible"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "label_shadow_missed_positive_top1": int(pd.to_numeric(daily_context_df.get("label_shadow_missed_positive_top1"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "top1_shadow_rule_eligible": int(pd.to_numeric(daily_context_df.get("top1_shadow_rule_eligible"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "shadow_top1_generation_blocked": int(pd.to_numeric(daily_context_df.get("shadow_top1_generation_blocked"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
            "shadow_top1_safety_blocked": int(pd.to_numeric(daily_context_df.get("shadow_top1_safety_blocked"), errors="coerce").notna().sum()) if not daily_context_df.empty else 0,
        },
    }
    if not daily_context_df.empty:
        resolved_mask = pd.to_numeric(daily_context_df.get("pool_feasible_resolved_count"), errors="coerce").fillna(0).gt(0)
        resolved_df = daily_context_df.loc[resolved_mask].copy()
        any_positive = pd.to_numeric(resolved_df.get("label_any_feasible_positive"), errors="coerce")
        any_improves_edge = pd.to_numeric(resolved_df.get("label_any_feasible_improves_edge"), errors="coerce")
        shadow_missed = pd.to_numeric(resolved_df.get("label_shadow_missed_feasible_positive"), errors="coerce")
        best_delta = pd.to_numeric(resolved_df.get("best_feasible_append_delta_vs_edge_mean"), errors="coerce")
        best_uplift = pd.to_numeric(resolved_df.get("best_feasible_append_uplift_vs_pool_mean"), errors="coerce")
        positive_days = int((any_positive == 1).sum())
        shadow_hits = int(((any_positive == 1) & (shadow_missed == 0)).sum())
        top1_resolved_df = resolved_df.loc[pd.to_numeric(resolved_df.get("top1_feasible_resolved"), errors="coerce").fillna(0).astype(bool)].copy()
        top1_positive = pd.to_numeric(top1_resolved_df.get("label_top1_feasible_positive"), errors="coerce")
        top1_improves_edge = pd.to_numeric(top1_resolved_df.get("label_top1_feasible_improves_edge"), errors="coerce")
        top1_shadow_match = pd.to_numeric(top1_resolved_df.get("shadow_matches_top1_feasible"), errors="coerce")
        top1_shadow_missed = pd.to_numeric(top1_resolved_df.get("label_shadow_missed_positive_top1"), errors="coerce")
        top1_delta = pd.to_numeric(top1_resolved_df.get("top1_feasible_value_delta"), errors="coerce")
        top1_uplift = pd.to_numeric(top1_resolved_df.get("top1_feasible_uplift_vs_pool_mean"), errors="coerce")
        top1_score_gap = pd.to_numeric(top1_resolved_df.get("top1_shadow_score_gap"), errors="coerce")
        top1_positive_days = int((top1_positive == 1).sum())
        top1_shadow_hits = int(((top1_positive == 1) & (top1_shadow_match == 1)).sum())
        summary["research_pool_opportunity_diagnostics"] = {
            "diagnostic_target_primary": "top1_feasible_by_decision_time",
            "resolved_feasible_days": int(len(resolved_df)),
            "days_any_feasible_positive": positive_days,
            "days_any_feasible_improves_edge": int((any_improves_edge == 1).sum()),
            "days_shadow_missed_feasible_positive": int((shadow_missed == 1).sum()),
            "shadow_recall_on_feasible_positive_days": (float(shadow_hits / positive_days) if positive_days > 0 else np.nan),
            "best_feasible_delta_vs_edge_mean": {
                "mean": float(best_delta.mean()) if best_delta.notna().any() else np.nan,
                "median": float(best_delta.median()) if best_delta.notna().any() else np.nan,
                "p10": float(best_delta.quantile(0.10)) if best_delta.notna().any() else np.nan,
                "p90": float(best_delta.quantile(0.90)) if best_delta.notna().any() else np.nan,
            },
            "best_feasible_uplift_vs_pool_mean": {
                "mean": float(best_uplift.mean()) if best_uplift.notna().any() else np.nan,
                "median": float(best_uplift.median()) if best_uplift.notna().any() else np.nan,
                "p10": float(best_uplift.quantile(0.10)) if best_uplift.notna().any() else np.nan,
                "p90": float(best_uplift.quantile(0.90)) if best_uplift.notna().any() else np.nan,
            },
            "top1_feasible_by_decision_time": {
                "resolved_top1_days": int(len(top1_resolved_df)),
                "days_top1_positive": top1_positive_days,
                "days_top1_improves_edge": int((top1_improves_edge == 1).sum()),
                "days_shadow_matches_top1": int((top1_shadow_match == 1).sum()),
                "days_shadow_missed_positive_top1": int((top1_shadow_missed == 1).sum()),
                "shadow_recall_on_positive_top1_days": (float(top1_shadow_hits / top1_positive_days) if top1_positive_days > 0 else np.nan),
                "top1_value_delta_vs_edge": {
                    "mean": float(top1_delta.mean()) if top1_delta.notna().any() else np.nan,
                    "median": float(top1_delta.median()) if top1_delta.notna().any() else np.nan,
                    "p10": float(top1_delta.quantile(0.10)) if top1_delta.notna().any() else np.nan,
                    "p90": float(top1_delta.quantile(0.90)) if top1_delta.notna().any() else np.nan,
                },
                "top1_uplift_vs_pool_mean": {
                    "mean": float(top1_uplift.mean()) if top1_uplift.notna().any() else np.nan,
                    "median": float(top1_uplift.median()) if top1_uplift.notna().any() else np.nan,
                    "p10": float(top1_uplift.quantile(0.10)) if top1_uplift.notna().any() else np.nan,
                    "p90": float(top1_uplift.quantile(0.90)) if top1_uplift.notna().any() else np.nan,
                },
                "top1_shadow_score_gap": {
                    "mean": float(top1_score_gap.mean()) if top1_score_gap.notna().any() else np.nan,
                    "median": float(top1_score_gap.median()) if top1_score_gap.notna().any() else np.nan,
                    "p10": float(top1_score_gap.quantile(0.10)) if top1_score_gap.notna().any() else np.nan,
                    "p90": float(top1_score_gap.quantile(0.90)) if top1_score_gap.notna().any() else np.nan,
                },
            },
        }
    else:
        summary["research_pool_opportunity_diagnostics"] = {}

    shadow_top1_miss_df = _build_shadow_top1_miss_report(daily_context_df)
    shadow_top1_miss_summary = _build_top1_shadow_miss_summary(
        daily_context_df,
        shadow_top1_miss_df,
        research_corr_mode=research_corr_mode,
        research_corr_threshold=research_corr_threshold,
    )
    summary["shadow_top1_miss_diagnostics"] = {
        "positive_top1_days": int(shadow_top1_miss_summary.get("positive_top1_days", 0)),
        "missed_positive_top1_days": int(shadow_top1_miss_summary.get("rows_missed_positive_top1", 0)),
        "shadow_recall_on_positive_top1_days": shadow_top1_miss_summary.get("shadow_recall_on_positive_top1_days", np.nan),
        "miss_stage_counts": shadow_top1_miss_summary.get("miss_stage_breakdown", {}),
        "miss_reason_counts": shadow_top1_miss_summary.get("miss_reason_counts", {}),
        "generation_blocked_days": int(pd.to_numeric(shadow_top1_miss_df.get("shadow_top1_generation_blocked"), errors="coerce").fillna(0).sum()) if not shadow_top1_miss_df.empty else 0,
        "safety_blocked_days": int(pd.to_numeric(shadow_top1_miss_df.get("shadow_top1_safety_blocked"), errors="coerce").fillna(0).sum()) if not shadow_top1_miss_df.empty else 0,
    }
    summary["top1_shadow_miss_reasons_summary"] = shadow_top1_miss_summary

    stage2_candidate_df, stage2_pair_df, stage2_summary = _build_stage2_proposal_tables(
        feature_df,
        daily_context_df,
        research_pool_min_agreement=float(args.research_pool_min_agreement),
        research_corr_mode=research_corr_mode,
        research_corr_threshold=research_corr_threshold,
    )
    summary["stage2_proposal_ranker"] = stage2_summary

    append_pool_all = feature_df.loc[feature_df["is_append_pool"] == True].copy()
    if not append_pool_all.empty:
        corr_all = pd.to_numeric(append_pool_all.get("corr_score"), errors="coerce")
        agreement_all = pd.to_numeric(append_pool_all.get("agreement_count"), errors="coerce").fillna(0.0)
        corr_quantiles = {
            "min": float(corr_all.min()) if corr_all.notna().any() else np.nan,
            "p10": float(corr_all.quantile(0.10)) if corr_all.notna().any() else np.nan,
            "p25": float(corr_all.quantile(0.25)) if corr_all.notna().any() else np.nan,
            "median": float(corr_all.quantile(0.50)) if corr_all.notna().any() else np.nan,
            "p75": float(corr_all.quantile(0.75)) if corr_all.notna().any() else np.nan,
            "p90": float(corr_all.quantile(0.90)) if corr_all.notna().any() else np.nan,
            "max": float(corr_all.max()) if corr_all.notna().any() else np.nan,
        }
        canonical_caps = [1.25, 2.0, 3.0, 6.0, 8.0, 10.0]
        feasible_days_by_cap: dict[str, int] = {}
        for cap in canonical_caps:
            mask = (agreement_all >= float(args.research_pool_min_agreement)) & corr_all.notna() & (corr_all <= float(cap))
            day_count = int(append_pool_all.loc[mask, "run_date"].astype(str).nunique())
            feasible_days_by_cap[f"corr_le_{cap:g}"] = day_count
        summary["research_pool_corr_diagnostics"] = {
            "append_pool_rows": int(len(append_pool_all)),
            "run_days_with_append_pool_rows": int(append_pool_all["run_date"].astype(str).nunique()),
            "corr_quantiles": corr_quantiles,
            "feasible_run_days_by_corr_cap": feasible_days_by_cap,
        }

        # Corr-score calibration table on research pool rows.
        calib_rows = append_pool_all.loc[agreement_all >= float(args.research_pool_min_agreement)].copy()
        corr_bins = [-np.inf, 6.0, 8.0, 10.0, 12.0, 14.0, np.inf]
        corr_labels = ["<=6", "(6,8]", "(8,10]", "(10,12]", "(12,14]", ">14"]
        calib_rows["corr_bin"] = pd.cut(
            pd.to_numeric(calib_rows.get("corr_score"), errors="coerce"),
            bins=corr_bins,
            labels=corr_labels,
            right=True,
            include_lowest=True,
        )
        calibration_table: list[dict[str, Any]] = []
        for label in corr_labels:
            grp = calib_rows.loc[calib_rows["corr_bin"] == label].copy()
            resolved = grp.loc[pd.to_numeric(grp.get("ev_label"), errors="coerce").notna()].copy()
            wins = int((resolved.get("result", pd.Series("", index=resolved.index)).astype(str) == "win").sum())
            resolved_count = int(len(resolved))
            win_rate = float(wins / resolved_count) if resolved_count > 0 else np.nan
            ev_per_resolved = float(pd.to_numeric(resolved.get("ev_label"), errors="coerce").mean()) if resolved_count > 0 else np.nan
            same_game_dup_rate = float((pd.to_numeric(grp.get("corr_same_game_count"), errors="coerce").fillna(0.0) > 0.0).mean()) if not grp.empty else np.nan
            same_target_dup_rate = float((pd.to_numeric(grp.get("corr_same_target_dir_count"), errors="coerce").fillna(0.0) > 0.0).mean()) if not grp.empty else np.nan
            same_script_dup_rate = float((pd.to_numeric(grp.get("corr_same_script_cluster_count"), errors="coerce").fillna(0.0) > 0.0).mean()) if not grp.empty else np.nan
            calibration_table.append(
                {
                    "corr_bin": str(label),
                    "candidate_count": int(len(grp)),
                    "resolved_count": resolved_count,
                    "win_rate": win_rate,
                    "ev_per_resolved": ev_per_resolved,
                    "same_game_dup_rate": same_game_dup_rate,
                    "same_target_dup_rate": same_target_dup_rate,
                    "same_script_dup_rate": same_script_dup_rate,
                }
            )
        summary["research_pool_corr_calibration"] = calibration_table
    else:
        summary["research_pool_corr_diagnostics"] = {}
        summary["research_pool_corr_calibration"] = []
    summary["window"] = {
        "requested_start": args.start_date,
        "requested_end": args.end_date,
        "exclude_snapshot_modes": [str(x) for x in (args.exclude_snapshot_modes or [])],
        "covered_run_dates": run_dates,
        "covered_run_date_count": int(len(run_dates)),
    }
    shadow_top1_miss_out = args.shadow_top1_miss_out
    if shadow_top1_miss_out is None:
        default_stem = args.daily_context_out.stem.replace("daily_context", "shadow_top1_miss_reasons")
        if default_stem == args.daily_context_out.stem:
            default_stem = f"{args.daily_context_out.stem}_shadow_top1_miss_reasons"
        shadow_top1_miss_out = args.daily_context_out.with_name(f"{default_stem}{args.daily_context_out.suffix}")
    top1_shadow_miss_summary_out = args.top1_shadow_miss_summary_out
    if top1_shadow_miss_summary_out is None:
        top1_shadow_miss_summary_out = shadow_top1_miss_out.with_name(
            shadow_top1_miss_out.stem.replace("reasons", "reasons_summary") + ".json"
        )
    stage2_proposal_table_out = args.stage2_proposal_table_out
    if stage2_proposal_table_out is None:
        default_stem = args.daily_context_out.stem.replace("daily_context", "stage2_proposal_table")
        if default_stem == args.daily_context_out.stem:
            default_stem = f"{args.daily_context_out.stem}_stage2_proposal_table"
        stage2_proposal_table_out = args.daily_context_out.with_name(f"{default_stem}{args.daily_context_out.suffix}")
    stage2_proposal_pairs_out = args.stage2_proposal_pairs_out
    if stage2_proposal_pairs_out is None:
        stage2_proposal_pairs_out = stage2_proposal_table_out.with_name(
            stage2_proposal_table_out.stem.replace("table", "pairs") + stage2_proposal_table_out.suffix
        )
    stage2_proposal_summary_out = args.stage2_proposal_summary_out
    if stage2_proposal_summary_out is None:
        stage2_proposal_summary_out = stage2_proposal_table_out.with_name(
            stage2_proposal_table_out.stem.replace("table", "summary") + ".json"
        )
    summary["artifacts"] = {
        "shadow_top1_miss_out": str(shadow_top1_miss_out),
        "top1_shadow_miss_summary_out": str(top1_shadow_miss_summary_out),
        "stage2_proposal_table_out": str(stage2_proposal_table_out),
        "stage2_proposal_pairs_out": str(stage2_proposal_pairs_out),
        "stage2_proposal_summary_out": str(stage2_proposal_summary_out),
    }

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    args.daily_out.parent.mkdir(parents=True, exist_ok=True)
    args.daily_context_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.abstain_out.parent.mkdir(parents=True, exist_ok=True)
    shadow_top1_miss_out.parent.mkdir(parents=True, exist_ok=True)
    top1_shadow_miss_summary_out.parent.mkdir(parents=True, exist_ok=True)
    stage2_proposal_table_out.parent.mkdir(parents=True, exist_ok=True)
    stage2_proposal_pairs_out.parent.mkdir(parents=True, exist_ok=True)
    stage2_proposal_summary_out.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(args.dataset_out, index=False)
    eval_df.to_csv(args.rows_out, index=False)
    daily_df.to_csv(args.daily_out, index=False)
    daily_context_df.to_csv(args.daily_context_out, index=False)
    abstain_df.to_csv(args.abstain_out, index=False)
    shadow_top1_miss_df.to_csv(shadow_top1_miss_out, index=False)
    stage2_candidate_df.to_csv(stage2_proposal_table_out, index=False)
    stage2_pair_df.to_csv(stage2_proposal_pairs_out, index=False)
    with top1_shadow_miss_summary_out.open("w", encoding="utf-8") as f:
        json.dump(shadow_top1_miss_summary, f, indent=2)
    with stage2_proposal_summary_out.open("w", encoding="utf-8") as f:
        json.dump(stage2_summary, f, indent=2)
    with args.summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 96)
    print("CUTOFF META APPEND SUMMARY")
    print("=" * 96)
    for variant in [
        "edge_baseline",
        "shadow_append_a1_p90_x1",
        "cutoff_meta_append_x1",
        "unified_shadow_meta_x1",
        "cutoff_meta_appended_only",
        "unified_appended_only",
    ]:
        metrics = summary.get(variant, {})
        print(f"{variant}: {metrics}")
    print(f"delta_meta_minus_edge: {summary.get('delta_meta_minus_edge', {})}")
    print(f"delta_unified_minus_edge: {summary.get('delta_unified_minus_edge', {})}")
    print("\nSaved:")
    print(f"  Dataset: {args.dataset_out}")
    print(f"  Rows:    {args.rows_out}")
    print(f"  Daily:   {args.daily_out}")
    print(f"  Daily Context: {args.daily_context_out}")
    print(f"  Abstain: {args.abstain_out}")
    print(f"  Shadow Miss: {shadow_top1_miss_out}")
    print(f"  Shadow Miss Summary: {top1_shadow_miss_summary_out}")
    print(f"  Stage2 Table: {stage2_proposal_table_out}")
    print(f"  Stage2 Pairs: {stage2_proposal_pairs_out}")
    print(f"  Stage2 Summary: {stage2_proposal_summary_out}")
    print(f"  Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
