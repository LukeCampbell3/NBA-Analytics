from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .io_utils import ensure_dir, write_json


ROLE_TARGETS = {
    "hitter": ["H", "HR", "RBI"],
    "pitcher": ["K", "ER", "ERA"],
}


ROLE_LEAKY_COLUMNS = {
    "hitter": {
        "H",
        "HR",
        "RBI",
        "TB",
        "R",
        "PA",
        "AB",
        "BB",
        "SO",
        "SB",
        "2B",
        "3B",
        "HBP",
        "IBB",
        "SF",
        "SH",
        "Team_PA_share",
        "wOBA",
        "ISO",
    },
    "pitcher": {
        "K",
        "ER",
        "ERA",
        "IP",
        "BF",
        "Pitches",
        "BB_allowed",
        "H_allowed",
        "HR_allowed",
        "HBP_allowed",
    },
}


class RollingBaselineRegressor:
    def __init__(self, feature_col: str | None, fallback_value: float) -> None:
        self.feature_col = feature_col
        self.fallback_value = float(fallback_value)

    def fit(self, X: Any, y: Any) -> "RollingBaselineRegressor":
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.feature_col and isinstance(X, pd.DataFrame) and self.feature_col in X.columns:
            series = pd.to_numeric(X[self.feature_col], errors="coerce").fillna(self.fallback_value)
            return series.to_numpy(dtype=np.float64)
        try:
            n = len(X)
        except Exception:
            n = 1
        return np.full(shape=(int(n),), fill_value=self.fallback_value, dtype=np.float64)


def _load_processed_frame(processed_dir: Path, season: int, role: str) -> pd.DataFrame:
    aggregate_path = processed_dir / f"{int(season)}_{role}s_processed.csv"
    if aggregate_path.exists():
        df = pd.read_csv(aggregate_path)
    else:
        frames = []
        for csv_path in sorted(processed_dir.glob("*/*_processed_processed.csv")):
            frames.append(pd.read_csv(csv_path))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df["Player_Type"] = df.get("Player_Type", "").astype(str).str.lower()
    df = df.loc[df["Player_Type"] == role].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[df["Date"].notna()].sort_values(["Date", "Player", "Game_ID"]).reset_index(drop=True)
    return df


def _feature_columns(df: pd.DataFrame, targets: list[str], role: str) -> list[str]:
    exclude = {
        "Date",
        "Player",
        "Player_Type",
        "Team",
        "Opponent",
        "Game_ID",
        "Market_Fetched_At_UTC",
        "Position",
    }
    exclude.update(targets)
    exclude.update({f"Market_Source_{t}" for t in targets})
    exclude.update(ROLE_LEAKY_COLUMNS.get(role, set()))
    cols = []
    for col in df.columns:
        if col in exclude or col.startswith("__") or col.endswith("_game"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _time_split(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["Date"].dropna().dt.normalize().unique().tolist())
    if len(dates) < 2:
        split = int(len(df) * (1.0 - val_fraction))
        split = max(1, min(split, len(df) - 1))
        return df.iloc[:split].copy(), df.iloc[split:].copy()
    cutoff_idx = max(1, int(len(dates) * (1.0 - val_fraction)))
    cutoff_idx = min(cutoff_idx, len(dates) - 1)
    cutoff_date = dates[cutoff_idx]
    train = df.loc[df["Date"] < cutoff_date].copy()
    val = df.loc[df["Date"] >= cutoff_date].copy()
    if train.empty or val.empty:
        split = int(len(df) * (1.0 - val_fraction))
        split = max(1, min(split, len(df) - 1))
        train = df.iloc[:split].copy()
        val = df.iloc[split:].copy()
    return train, val


def _candidate_models() -> dict[str, Any]:
    return {
        "hgb": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_depth=6,
            l2_regularization=0.01,
            random_state=42,
        ),
        "rf": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }


def _fit_target_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target: str,
) -> dict:
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    y_val = pd.to_numeric(val_df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    baseline_col = f"{target}_rolling_avg"
    baseline_value = float(np.mean(y_train)) if len(y_train) else 0.0
    if baseline_col in train_df.columns:
        train_baseline_mean = pd.to_numeric(train_df[baseline_col], errors="coerce").dropna()
        if not train_baseline_mean.empty:
            baseline_value = float(train_baseline_mean.mean())
    baseline_model = RollingBaselineRegressor(
        feature_col=baseline_col if baseline_col in feature_cols else None,
        fallback_value=baseline_value,
    )
    baseline_pred = baseline_model.predict(val_df)
    baseline_mae = float(mean_absolute_error(y_val, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))

    candidates = _candidate_models()
    results = [
        {
            "name": "baseline",
            "model": baseline_model,
            "mae": baseline_mae,
            "rmse": baseline_rmse,
            "is_learned": False,
        }
    ]
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        results.append(
            {
                "name": name,
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "is_learned": True,
            }
        )
    results = sorted(results, key=lambda x: x["mae"])
    best = results[0]
    learned = [r for r in results if r.get("is_learned")]

    # Blend top-2 learned models by inverse MAE; accept only if validation improves.
    blended = None
    if len(learned) >= 2:
        a, b = learned[0], learned[1]
        w_a = 1.0 / max(a["mae"], 1e-8)
        w_b = 1.0 / max(b["mae"], 1e-8)
        w_sum = w_a + w_b
        weights = [float(w_a / w_sum), float(w_b / w_sum)]
        blend_pred = weights[0] * a["model"].predict(X_val) + weights[1] * b["model"].predict(X_val)
        blend_mae = float(mean_absolute_error(y_val, blend_pred))
        blend_rmse = float(np.sqrt(mean_squared_error(y_val, blend_pred)))
        if blend_mae < best["mae"]:
            blended = {
                "name": "blend",
                "models": [a["model"], b["model"]],
                "members": [a["name"], b["name"]],
                "weights": weights,
                "mae": blend_mae,
                "rmse": blend_rmse,
            }

    best_learned = blended if blended is not None else (
        {
            "name": learned[0]["name"],
            "models": [learned[0]["model"]],
            "members": [learned[0]["name"]],
            "weights": [1.0],
            "mae": learned[0]["mae"],
            "rmse": learned[0]["rmse"],
        }
        if learned
        else {
            "name": "baseline",
            "models": [baseline_model],
            "members": ["baseline"],
            "weights": [1.0],
            "mae": baseline_mae,
            "rmse": baseline_rmse,
        }
    )

    selected = (
        {
            "name": "baseline",
            "models": [baseline_model],
            "members": ["baseline"],
            "weights": [1.0],
            "mae": baseline_mae,
            "rmse": baseline_rmse,
        }
        if baseline_mae <= best_learned["mae"]
        else best_learned
    )
    return {
        "target": target,
        "feature_cols": feature_cols,
        "baseline_mae": baseline_mae,
        "candidate_scores": [
            {"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]}
            for r in results
        ],
        "selected": {
            "name": selected["name"],
            "members": selected["members"],
            "weights": selected["weights"],
            "mae": float(selected["mae"]),
            "rmse": float(selected["rmse"]),
        },
        "model_bundle": selected,
    }


@dataclass
class TrainConfig:
    processed_dir: Path
    model_dir: Path
    season: int = 2026
    min_rows: int = 200


def train_role_models(config: TrainConfig, role: str) -> dict:
    processed_dir = config.processed_dir.resolve()
    model_dir = ensure_dir(config.model_dir.resolve())
    role_dir = ensure_dir(model_dir / role)

    df = _load_processed_frame(processed_dir, season=config.season, role=role)
    if df.empty:
        raise RuntimeError(f"No {role} rows found in {processed_dir}")
    if len(df) < int(config.min_rows):
        raise RuntimeError(
            f"Not enough {role} rows for stable training: {len(df)} < min_rows={int(config.min_rows)}"
        )

    targets = ROLE_TARGETS[role]
    feature_cols = _feature_columns(df, targets=targets, role=role)
    if not feature_cols:
        raise RuntimeError(f"No numeric feature columns available for {role} training")

    train_df, val_df = _time_split(df, val_fraction=0.2)
    if train_df.empty or val_df.empty:
        raise RuntimeError(f"Could not create train/validation split for {role}")

    target_summaries = []
    for target in targets:
        payload = _fit_target_models(train_df, val_df, feature_cols=feature_cols, target=target)
        bundle = payload.pop("model_bundle")
        artifact = {
            "role": role,
            "target": target,
            "season": int(config.season),
            "feature_cols": payload["feature_cols"],
            "selected": payload["selected"],
            "models": bundle["models"],
            "weights": bundle["weights"],
            "members": bundle["members"],
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        model_path = role_dir / f"{target.lower()}_model.joblib"
        joblib.dump(artifact, model_path)
        payload["artifact_path"] = str(model_path)
        target_summaries.append(payload)

    summary = {
        "role": role,
        "season": int(config.season),
        "row_count": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "targets": target_summaries,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(role_dir / "training_summary.json", summary)
    return summary


def train_all_models(config: TrainConfig) -> dict:
    summaries = {}
    failures = {}
    for role in ["hitter", "pitcher"]:
        try:
            summaries[role] = train_role_models(config, role)
        except Exception as exc:
            failures[role] = str(exc)

    payload = {
        "season": int(config.season),
        "model_dir": str(config.model_dir.resolve()),
        "summaries": summaries,
        "failures": failures,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(config.model_dir.resolve() / "training_manifest.json", payload)
    if not summaries:
        raise RuntimeError(f"No role models were trained. Failures: {failures}")
    return payload
