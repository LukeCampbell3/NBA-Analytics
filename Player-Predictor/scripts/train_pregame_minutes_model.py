#!/usr/bin/env python3
"""
Train a leakage-safe pregame minutes projection model.

Target uses actual historical minutes, but features are shifted rolling values
available before the game being predicted. Current-game actual minutes are
explicitly forbidden as model input.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [
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


def _load_rows(v9_manifest: Path) -> pd.DataFrame:
    manifest = json.loads(v9_manifest.read_text(encoding="utf-8"))
    output = Path(manifest["output"])
    if str(output).startswith("/workspace/"):
        output = ROOT.parent / str(output).replace("/workspace/", "", 1)
    if not output.is_absolute():
        output = (ROOT.parent / output).resolve()
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date", "player", "minutes"]).copy()
    games = rows[["player", "date", "minutes"]].drop_duplicates(["player", "date"])
    return games.sort_values(["player", "date"]).reset_index(drop=True)


def build_minutes_features(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.copy()
    grouped = frame.groupby("player", group_keys=False)
    frame["games_prior"] = grouped.cumcount()
    frame["days_since_last_game"] = grouped["date"].diff().dt.days.fillna(7).clip(0, 30)
    shifted = grouped["minutes"].shift(1)
    frame["minutes_lag1"] = shifted
    frame["played_lag1"] = (shifted.fillna(0) > 0).astype(float)

    for window in [3, 5, 10]:
        roll = grouped["minutes"].shift(1).groupby(frame["player"]).rolling(window, min_periods=1)
        frame[f"minutes_roll{window}_mean"] = roll.mean().reset_index(level=0, drop=True)
        frame[f"minutes_roll{window}_std"] = roll.std().reset_index(level=0, drop=True)
        frame[f"minutes_roll{window}_min"] = roll.min().reset_index(level=0, drop=True)
        frame[f"minutes_roll{window}_max"] = roll.max().reset_index(level=0, drop=True)

    dnp = (grouped["minutes"].shift(1).fillna(0) <= 0).astype(float)
    frame["dnp_rate_roll10"] = (
        dnp.groupby(frame["player"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    frame = frame[frame["games_prior"] >= 3].copy()
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def _fit_quantile_model(x: pd.DataFrame, y: pd.Series, alpha: float):
    model = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    n_estimators=180,
                    learning_rate=0.035,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def train_minutes_model(features: pd.DataFrame, train_end: str | None) -> tuple[dict, pd.DataFrame]:
    if train_end:
        train = features[features["date"] <= pd.Timestamp(train_end)].copy()
        holdout = features[features["date"] > pd.Timestamp(train_end)].copy()
    else:
        split_date = features["date"].quantile(0.80)
        train = features[features["date"] <= split_date].copy()
        holdout = features[features["date"] > split_date].copy()

    x_train = train[FEATURE_COLUMNS]
    y_train = train["minutes"]
    mean_model = Pipeline(
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
    mean_model.fit(x_train, y_train)
    quantiles = {
        "p10": _fit_quantile_model(x_train, y_train, 0.10),
        "p25": _fit_quantile_model(x_train, y_train, 0.25),
        "p50": _fit_quantile_model(x_train, y_train, 0.50),
        "p75": _fit_quantile_model(x_train, y_train, 0.75),
        "p90": _fit_quantile_model(x_train, y_train, 0.90),
    }
    bundle = {
        "model_version": "pregame_minutes_v1",
        "feature_columns": FEATURE_COLUMNS,
        "mean_model": mean_model,
        "quantile_models": quantiles,
        "forbidden_features": ["minutes_current_game", "actual_minutes", "postgame_box_score"],
    }

    eval_frame = holdout if not holdout.empty else train
    scored = eval_frame[["player", "date", "minutes"]].copy()
    x_eval = eval_frame[FEATURE_COLUMNS]
    scored["minutes_mean"] = mean_model.predict(x_eval).clip(0, 48)
    for name, model in quantiles.items():
        scored[f"minutes_{name}"] = model.predict(x_eval).clip(0, 48)
    scored["minutes_sigma"] = ((scored["minutes_p90"] - scored["minutes_p10"]) / 2.56).clip(0.5, 20)
    scored["abs_error"] = (scored["minutes"] - scored["minutes_mean"]).abs()
    return bundle, scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-safe pregame minutes model")
    parser.add_argument("--v9-manifest", type=Path, default=ROOT / "model" / "props" / "v9" / "manifest.json")
    parser.add_argument("--train-end", type=str, default="2025-12-31")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_1" / "minutes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = _load_rows(args.v9_manifest)
    features = build_minutes_features(games)
    bundle, scored = train_minutes_model(features, args.train_end)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output / "pregame_minutes_model.pkl")
    scored.to_csv(args.output / "minutes_holdout_predictions.csv", index=False)

    p10_cover = float((scored["minutes"] >= scored["minutes_p10"]).mean())
    p90_cover = float((scored["minutes"] <= scored["minutes_p90"]).mean())
    report = {
        "model_version": "pregame_minutes_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_end": args.train_end,
        "rows": int(len(features)),
        "holdout_rows": int(len(scored)),
        "minutes_mae": float(scored["abs_error"].mean()),
        "p10_coverage": p10_cover,
        "p90_coverage": p90_cover,
        "feature_columns": FEATURE_COLUMNS,
        "forbidden_features": bundle["forbidden_features"],
        "status": "shadow_feature_candidate",
    }
    (args.output / "minutes_model_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
