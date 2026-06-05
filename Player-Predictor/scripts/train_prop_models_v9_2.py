#!/usr/bin/env python3
"""
Train v9.2 honest distribution upgrade.

v9.2 keeps the v9.1 distribution-led architecture, but replaces the missing
actual-minutes signal with leakage-safe pregame minutes projections generated
from shifted player history. It then adjusts the stat distribution only when
the pregame minutes projection disagrees with the player's pregame minutes
baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MINUTE_FEATURES = [
    "games_prior",
    "days_since_last_game",
    "minutes_lag1",
    "minutes_roll3_mean",
    "minutes_roll5_mean",
    "minutes_roll10_mean",
    "minutes_roll5_std",
    "minutes_roll10_std",
    "minutes_roll5_min",
    "minutes_roll10_min",
    "minutes_roll5_max",
    "minutes_roll10_max",
    "played_lag1",
    "dnp_rate_roll10",
]


def _resolve_manifest_path(path_text: str) -> Path:
    if str(path_text).startswith("/workspace/"):
        return ROOT.parent / str(path_text).replace("/workspace/", "", 1)
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT.parent / path).resolve()


def _load_v9_rows(v9_manifest: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(v9_manifest.read_text(encoding="utf-8"))
    output = _resolve_manifest_path(manifest["output"])
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date", "player"]).copy()
    return manifest, rows.sort_values(["date", "player", "market"]).reset_index(drop=True)


def _build_game_minutes_frame(rows: pd.DataFrame) -> pd.DataFrame:
    games = rows[["player", "date", "minutes"]].drop_duplicates(["player", "date"]).copy()
    games = games.sort_values(["player", "date"]).reset_index(drop=True)
    grouped = games.groupby("player", group_keys=False)
    games["games_prior"] = grouped.cumcount()
    games["days_since_last_game"] = grouped["date"].diff().dt.days.fillna(7).clip(0, 30)
    shifted = grouped["minutes"].shift(1)
    games["minutes_lag1"] = shifted
    games["played_lag1"] = (shifted.fillna(0) > 0).astype(float)
    for window in [3, 5, 10]:
        roll = shifted.groupby(games["player"]).rolling(window, min_periods=1)
        games[f"minutes_roll{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
        games[f"minutes_roll{window}_std"] = roll.std().reset_index(level=0, drop=True)
        games[f"minutes_roll{window}_min"] = roll.min().reset_index(level=0, drop=True)
        games[f"minutes_roll{window}_max"] = roll.max().reset_index(level=0, drop=True)
    dnp = (shifted.fillna(0) <= 0).astype(float)
    games["dnp_rate_roll10"] = dnp.groupby(games["player"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    games[MINUTE_FEATURES] = games[MINUTE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return games


def _minute_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.035,
                    max_leaf_nodes=18,
                    l2_regularization=0.08,
                    random_state=42,
                ),
            ),
        ]
    )


def add_walk_forward_minutes(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    games = _build_game_minutes_frame(rows)
    games["projected_minutes_mean"] = games["minutes_roll5_mean"].fillna(games["minutes_lag1"]).fillna(0.0)
    games["projected_minutes_sigma"] = games["minutes_roll10_std"].fillna(games["minutes_roll5_std"]).fillna(5.0).clip(1.0, 18.0)

    fold_reports = []
    month_starts = pd.date_range(games["date"].min().replace(day=1), games["date"].max(), freq="MS")
    for month_start in month_starts:
        train = games[(games["date"] < month_start) & (games["games_prior"] >= 3)].copy()
        score_idx = games[(games["date"] >= month_start) & (games["date"] < month_start + pd.offsets.MonthBegin(1))].index
        if len(train) < 500 or len(score_idx) == 0:
            continue
        model = _minute_model()
        model.fit(train[MINUTE_FEATURES], train["minutes"])
        preds = model.predict(games.loc[score_idx, MINUTE_FEATURES]).clip(0, 48)
        games.loc[score_idx, "projected_minutes_mean"] = preds
        if len(score_idx):
            fold_reports.append(
                {
                    "month": str(month_start.date()),
                    "train_rows": int(len(train)),
                    "score_rows": int(len(score_idx)),
                    "minutes_mae": float(np.mean(np.abs(games.loc[score_idx, "minutes"].to_numpy() - preds))),
                }
            )

    keep = ["player", "date", "projected_minutes_mean", "projected_minutes_sigma", "minutes_roll5_mean", "minutes_roll10_mean"]
    out = rows.merge(games[keep], on=["player", "date"], how="left")
    report = {
        "folds": fold_reports,
        "avg_fold_minutes_mae": float(np.mean([f["minutes_mae"] for f in fold_reports])) if fold_reports else None,
    }
    return out, report


def _normal_p_over(line: pd.Series, mean: pd.Series, sigma: pd.Series) -> np.ndarray:
    z = (line.to_numpy(dtype=float) - mean.to_numpy(dtype=float)) / sigma.to_numpy(dtype=float).clip(0.75, 30)
    return 0.5 * (1.0 - np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _choose_minutes_blend(rows: pd.DataFrame, p_minutes: np.ndarray, blend_train_end: str | None) -> tuple[float, list[dict]]:
    if not blend_train_end:
        return 0.22, []
    mask = rows["date"] <= pd.Timestamp(blend_train_end)
    if mask.sum() < 500:
        return 0.22, []
    y = rows.loc[mask, "result_over"].to_numpy(dtype=float)
    base = rows.loc[mask, "p_over_raw"].to_numpy(dtype=float)
    pmin = p_minutes[mask.to_numpy()]
    grid = np.arange(0.0, 0.51, 0.05)
    results = []
    for alpha in grid:
        pred = ((1.0 - alpha) * base + alpha * pmin).clip(0.01, 0.99)
        brier = float(np.mean((pred - y) ** 2))
        results.append({"alpha": float(alpha), "train_brier": brier})
    best = min(results, key=lambda item: item["train_brier"])
    return float(best["alpha"]), results


def add_v92_distribution_adjustment(rows: pd.DataFrame, blend_train_end: str | None = "2025-12-31") -> tuple[pd.DataFrame, dict]:
    rows = rows.copy()
    baseline_minutes = rows["minutes_roll5_mean"].replace(0, np.nan).fillna(rows["minutes_roll10_mean"]).fillna(rows["projected_minutes_mean"])
    ratio = (rows["projected_minutes_mean"] / baseline_minutes.clip(lower=8.0)).clip(0.75, 1.25)
    market_weights = rows["market"].map({"PTS": 0.72, "TRB": 0.64, "AST": 0.68}).fillna(0.65)
    adjustment = 1.0 + (ratio - 1.0) * market_weights
    rows["v92_minutes_ratio"] = ratio
    rows["v92_minutes_adjustment"] = adjustment
    rows["v92_model_mean"] = (rows["model_mean"] * adjustment).clip(lower=0.0)
    rows["v92_sigma"] = np.sqrt(rows["sigma"].clip(lower=0.75) ** 2 + (rows["projected_minutes_sigma"] * 0.08).clip(0.0, 3.0) ** 2)
    p_minutes = _normal_p_over(rows["line"], rows["v92_model_mean"], rows["v92_sigma"]).clip(0.01, 0.99)
    alpha, alpha_grid = _choose_minutes_blend(rows, p_minutes, blend_train_end)
    # Conservative blend learned before the validation window.
    rows["p_over_raw_v91"] = rows["p_over_raw"]
    rows["p_over_raw_minutes_branch"] = p_minutes
    rows["p_over_raw"] = ((1.0 - alpha) * rows["p_over_raw"] + alpha * p_minutes).clip(0.01, 0.99)
    rows["edge_over_raw"] = rows["p_over_raw"] - rows["market_no_vig_over"]
    rows["edge_under_raw"] = (1.0 - rows["p_over_raw"]) - rows["market_no_vig_under"]
    report = {
        "avg_abs_probability_delta": float((rows["p_over_raw"] - rows["p_over_raw_v91"]).abs().mean()),
        "p95_abs_probability_delta": float((rows["p_over_raw"] - rows["p_over_raw_v91"]).abs().quantile(0.95)),
        "avg_minutes_ratio": float(rows["v92_minutes_ratio"].mean()),
        "minutes_blend_alpha": alpha,
        "blend_train_end": blend_train_end,
        "alpha_grid": alpha_grid,
    }
    return rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v9.2 honest minutes-adjusted distribution artifacts")
    parser.add_argument("--v9-manifest", type=Path, default=ROOT / "model" / "props" / "v9" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_2")
    parser.add_argument("--blend-train-end", type=str, default="2025-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v9_manifest, rows = _load_v9_rows(args.v9_manifest)
    rows, minutes_report = add_walk_forward_minutes(rows)
    rows, adjustment_report = add_v92_distribution_adjustment(rows, args.blend_train_end)

    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(data_dir / "prop_training_rows.csv", index=False)

    manifest = {
        "model_version": "prop_engine_v9_2_minutes_adjusted_distribution",
        "status": "shadow_candidate_pending_validation",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_manifest": str(args.v9_manifest),
        "output": str(args.output),
        "rows": int(len(rows)),
        "players": int(rows["player"].nunique()),
        "date_min": str(rows["date"].min().date()),
        "date_max": str(rows["date"].max().date()),
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "minutes_projection": "embedded_walk_forward_shifted_minutes_features",
        },
        "summaries": {
            "minutes": minutes_report,
            "distribution_adjustment": adjustment_report,
            "forbidden_features": ["actual_minutes_current_game", "residual", "abs_residual", "postgame_box_score"],
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest["summaries"], indent=2, default=str))
    print(f"\nWrote v9.2 artifacts to {args.output}")


if __name__ == "__main__":
    main()
