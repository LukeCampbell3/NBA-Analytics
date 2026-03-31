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

    for run_dir in run_dirs:
        run_date = pd.to_datetime(run_dir.name, format="%Y%m%d", errors="coerce")
        if pd.isna(run_date):
            continue
        run_date_str = run_date.strftime("%Y-%m-%d")
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
    summary["feature_spec"] = {
        "near_cutoff_rank_range": [cutoff_low, cutoff_high],
        "append_pool_rank_range": [append_rank_low, append_rank_high],
        "training_target": "append_value_target",
        "secondary_targets": ["win_residual", "resolved_win_label", "ev_label"],
        "meta_model": "RandomForestRegressor",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    summary["window"] = {
        "requested_start": args.start_date,
        "requested_end": args.end_date,
        "covered_run_dates": run_dates,
        "covered_run_date_count": int(len(run_dates)),
    }

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    args.daily_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.abstain_out.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(args.dataset_out, index=False)
    eval_df.to_csv(args.rows_out, index=False)
    daily_df.to_csv(args.daily_out, index=False)
    abstain_df.to_csv(args.abstain_out, index=False)
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
    print(f"  Abstain: {args.abstain_out}")
    print(f"  Summary: {args.summary_out}")


if __name__ == "__main__":
    main()
