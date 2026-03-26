#!/usr/bin/env python3
"""
Train a synthetic market-anchor model from processed rows that contain market lines.

This learns bookmaker-like center lines from our existing structured features so we
can backfill historical market anchors when real props are unavailable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data-Proc"
OUT_DIR = REPO_ROOT / "model" / "market_anchor"
TARGETS = ["PTS", "TRB", "AST"]

NUMERIC_FEATURES = [
    "PTS_rolling_avg",
    "TRB_rolling_avg",
    "AST_rolling_avg",
    "PTS_lag1",
    "TRB_lag1",
    "AST_lag1",
    "USG%",
    "USG%_rolling_avg",
    "MP",
    "Rest_Days",
    "Did_Not_Play",
    "oppDfRtg_3",
    "Month_sin",
    "Month_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "GmSc_rolling_avg",
    "ORTG_rolling_avg",
    "DRTG_rolling_avg",
]

CATEGORICAL_FEATURES = [
    "Player",
    "Opponent",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a synthetic market anchor model.")
    parser.add_argument("--limit-players", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum rows with real market targets required per target.")
    parser.add_argument("--outdir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--random-seed", type=int, default=42, help="CatBoost random seed.")
    return parser.parse_args()


def gather_csvs(limit_players: int | None) -> list[Path]:
    player_dirs = sorted([path for path in DATA_DIR.iterdir() if path.is_dir()])
    if limit_players is not None:
        player_dirs = player_dirs[:limit_players]
    csvs: list[Path] = []
    for player_dir in player_dirs:
        csvs.extend(sorted(player_dir.glob("*_processed_processed.csv")))
    return csvs


def load_training_frame(limit_players: int | None) -> pd.DataFrame:
    frames = []
    for path in gather_csvs(limit_players):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required_any = [f"Market_{target}" for target in TARGETS]
        if not any(col in df.columns and df[col].notna().any() for col in required_any):
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def restrict_to_real_market_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    source_col = f"Market_Source_{target}"
    if source_col not in df.columns:
        return df.loc[df[f"Market_{target}"].notna()].copy()
    mask = df[source_col].fillna("").astype(str).eq("real") & df[f"Market_{target}"].notna()
    return df.loc[mask].copy()


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL_FEATURES:
        if col not in work.columns:
            work[col] = "UNK"
        work[col] = work[col].fillna("UNK").astype(str)
    for col in NUMERIC_FEATURES:
        work[col] = work[col].ffill().bfill().fillna(0.0)
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    return work[feature_cols].copy(), feature_cols, CATEGORICAL_FEATURES.copy()


def train_target_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    random_seed: int,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        depth=6,
        learning_rate=0.05,
        iterations=600,
        l2_leaf_reg=6.0,
        random_seed=random_seed,
        verbose=False,
    )
    model.fit(X, y, cat_features=categorical_features)
    return model


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    train_df = load_training_frame(args.limit_players)
    if train_df.empty:
        raise RuntimeError("No processed rows with market columns were found. Merge market props first.")

    X_full, feature_cols, categorical_features = build_feature_frame(train_df)
    bundle = {
        "created_at_utc": utc_now_iso(),
        "feature_columns": feature_cols,
        "categorical_features": categorical_features,
        "targets": {},
    }

    for target in TARGETS:
        target_df = restrict_to_real_market_rows(train_df, target)
        if len(target_df) < args.min_rows:
            continue
        target_col = f"Market_{target}"
        X_target, _, _ = build_feature_frame(target_df)
        y_target = pd.to_numeric(target_df[target_col], errors="coerce")
        model = train_target_model(X_target, y_target, categorical_features, args.random_seed)
        pred = model.predict(X_target)
        mae = float(np.mean(np.abs(pred - y_target.to_numpy(dtype=float))))
        bundle["targets"][target] = {
            "model": model,
            "rows": int(len(target_df)),
            "train_mae": mae,
        }

    if not bundle["targets"]:
        raise RuntimeError(f"No targets met the minimum row count of {args.min_rows}.")

    joblib.dump(bundle, args.outdir / "latest_market_anchor.pkl")
    metadata = {
        "created_at_utc": bundle["created_at_utc"],
        "feature_columns": bundle["feature_columns"],
        "categorical_features": bundle["categorical_features"],
        "targets": {
            target: {
                "rows": payload["rows"],
                "train_mae": payload["train_mae"],
            }
            for target, payload in bundle["targets"].items()
        },
    }
    (args.outdir / "latest_market_anchor.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("MARKET ANCHOR TRAINING COMPLETE")
    print("=" * 80)
    for target, payload in metadata["targets"].items():
        print(f"{target}: rows={payload['rows']} train_mae={payload['train_mae']:.4f}")
    print(f"Artifacts: {args.outdir}")


if __name__ == "__main__":
    main()
