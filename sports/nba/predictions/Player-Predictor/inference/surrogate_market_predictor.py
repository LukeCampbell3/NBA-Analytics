from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


TARGETS = ["PTS", "TRB", "AST"]
NUMERIC_STAT_COLUMNS = ["PTS", "TRB", "AST", "MP", "FGA", "FTA", "USG%", "TS%", "GmSc", "TOV", "PLUS_MINUS"]
WINDOWS_MEAN = [3, 5, 10]
WINDOWS_STD = [5, 10]
TEAM_ID_BY_ABBREV = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}
DEFAULT_BUNDLE_PATH = Path("surrogate_market_predictor.joblib")
DEFAULT_SUMMARY_PATH = Path("surrogate_market_predictor_summary.json")
MODEL_TAG = "surrogate_tabular_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(value: str | None) -> str:
    out = str(value or "")
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return "_".join(part for part in out.split("_") if part)


def team_abbr_from_matchup(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().split(" ")
    if not token:
        return None
    head = token[0].strip().upper()
    return head if 2 <= len(head) <= 4 else None


def is_home_from_matchup(value: str | None) -> float:
    return float("vs." in str(value or ""))


def _base_numeric_columns() -> list[str]:
    cols: list[str] = []
    for col in NUMERIC_STAT_COLUMNS:
        cols.append(f"{col}_lag1")
        for window in WINDOWS_MEAN:
            cols.append(f"{col}_mean_{window}")
        for window in WINDOWS_STD:
            cols.append(f"{col}_std_{window}")
    cols.extend(["rest_days", "dnp_rate_10", "games_played_before", "dow", "month", "is_home"])
    return cols


FEATURE_NUMERIC_COLUMNS = _base_numeric_columns()
FEATURE_CATEGORICAL_COLUMNS = ["player_key", "Team_ID", "Opponent_ID"]
FEATURE_COLUMNS = FEATURE_NUMERIC_COLUMNS + FEATURE_CATEGORICAL_COLUMNS


def _prepare_history_frame(history_df: pd.DataFrame) -> pd.DataFrame:
    working = history_df.copy()
    if "Date" not in working.columns:
        raise ValueError("history_df must include Date")
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working.loc[working["Date"].notna()].sort_values("Date").reset_index(drop=True)
    if working.empty:
        raise ValueError("history_df has no valid dated rows")
    for column in NUMERIC_STAT_COLUMNS + ["Rest_Days", "Did_Not_Play", "Team_ID", "Opponent_ID"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        else:
            working[column] = np.nan
    working["is_home"] = working.get("MATCHUP", pd.Series("", index=working.index)).map(is_home_from_matchup).astype("float64")
    player_series = working.get("Player", pd.Series("", index=working.index)).astype(str).map(normalize_name)
    working["player_key"] = player_series.replace("", np.nan).fillna(normalize_name(str(working.iloc[-1].get("Player", ""))))
    return working


def build_training_frame(data_proc_root: Path, min_history_games: int = 5) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    use_cols = {
        "Date",
        "Player",
        "PTS",
        "TRB",
        "AST",
        "MP",
        "FGA",
        "FTA",
        "USG%",
        "TS%",
        "GmSc",
        "TOV",
        "PLUS_MINUS",
        "Rest_Days",
        "Did_Not_Play",
        "MATCHUP",
        "Team_ID",
        "Opponent_ID",
    }
    for csv_path in sorted(data_proc_root.glob("*/*_processed_processed.csv")):
        try:
            raw = pd.read_csv(csv_path, usecols=lambda c: c in use_cols)
        except Exception:
            continue
        if raw.empty or "Date" not in raw.columns:
            continue
        try:
            frame = _prepare_history_frame(raw)
        except Exception:
            continue
        frame["player_key"] = normalize_name(csv_path.parent.name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["Date"] + FEATURE_COLUMNS + TARGETS)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["player_key", "Date"]).reset_index(drop=True)
    grouped = df.groupby("player_key", sort=False)

    for column in NUMERIC_STAT_COLUMNS:
        shifted = grouped[column].shift(1)
        df[f"{column}_lag1"] = shifted
        rolling_group = shifted.groupby(df["player_key"])
        for window in WINDOWS_MEAN:
            df[f"{column}_mean_{window}"] = rolling_group.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        for window in WINDOWS_STD:
            df[f"{column}_std_{window}"] = rolling_group.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)

    df["rest_days"] = pd.to_numeric(df.get("Rest_Days"), errors="coerce")
    dnp_shifted = grouped["Did_Not_Play"].shift(1)
    df["dnp_rate_10"] = dnp_shifted.groupby(df["player_key"]).rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    df["games_played_before"] = grouped.cumcount().astype("float64")
    df["dow"] = df["Date"].dt.dayofweek.astype("float64")
    df["month"] = df["Date"].dt.month.astype("float64")

    out = df.loc[df["games_played_before"] >= float(min_history_games)].copy()
    out = out.loc[out[TARGETS].notna().all(axis=1)].copy()
    out["Team_ID"] = pd.to_numeric(out["Team_ID"], errors="coerce").fillna(-1).astype("int64").astype(str)
    out["Opponent_ID"] = pd.to_numeric(out["Opponent_ID"], errors="coerce").fillna(-1).astype("int64").astype(str)
    out["player_key"] = out["player_key"].astype(str)
    return out.reset_index(drop=True)


def _history_sigma(values: pd.Series, fallback: float) -> float:
    recent = pd.to_numeric(values, errors="coerce").dropna().tail(10)
    if len(recent) >= 2:
        return float(np.std(recent.to_numpy(dtype=float), ddof=0))
    return float(max(0.0, fallback))


def _spike_probability(values: pd.Series) -> float:
    recent = pd.to_numeric(values, errors="coerce").dropna().tail(10)
    if len(recent) < 2:
        return 0.10
    sigma = float(np.std(recent.to_numpy(dtype=float), ddof=0))
    if sigma <= 1e-9:
        return 0.10
    z_score = float((recent.iloc[-1] - recent.mean()) / sigma)
    return float(np.clip(0.50 + 0.18 * z_score, 0.05, 0.95))


def build_feature_row(
    history_df: pd.DataFrame,
    *,
    market_context: dict[str, Any] | None = None,
    min_history_games: int = 5,
) -> dict[str, Any]:
    history = _prepare_history_frame(history_df)
    if len(history) < int(min_history_games):
        raise ValueError(f"at least {min_history_games} history rows are required")

    latest = history.iloc[-1]
    market_context = dict(market_context or {})
    market_date = pd.to_datetime(market_context.get("market_date"), errors="coerce")
    if pd.isna(market_date):
        market_date = pd.to_datetime(latest["Date"], errors="coerce")
    player_key = normalize_name(str(market_context.get("player") or latest.get("player_key") or latest.get("Player") or ""))

    player_team_abbr = team_abbr_from_matchup(latest.get("MATCHUP"))
    home_team = str(market_context.get("market_home_team") or "").strip().upper()
    away_team = str(market_context.get("market_away_team") or "").strip().upper()
    if player_team_abbr and home_team and away_team:
        is_home = 1.0 if player_team_abbr == home_team else 0.0 if player_team_abbr == away_team else float(latest.get("is_home", 0.0))
        opponent_abbr = away_team if player_team_abbr == home_team else home_team if player_team_abbr == away_team else None
    else:
        is_home = float(latest.get("is_home", 0.0))
        opponent_abbr = None

    player_team_id = int(pd.to_numeric(pd.Series([latest.get("Team_ID")]), errors="coerce").fillna(-1).iloc[0])
    opponent_id_default = int(pd.to_numeric(pd.Series([latest.get("Opponent_ID")]), errors="coerce").fillna(-1).iloc[0])
    opponent_id = TEAM_ID_BY_ABBREV.get(str(opponent_abbr), opponent_id_default)

    latest_history_date = pd.to_datetime(latest["Date"], errors="coerce")
    if pd.notna(market_date) and pd.notna(latest_history_date):
        rest_days = float(max(0.0, (market_date.normalize() - latest_history_date.normalize()).days))
    else:
        rest_days = float(pd.to_numeric(pd.Series([latest.get("Rest_Days")]), errors="coerce").fillna(2.0).iloc[0])

    row: dict[str, Any] = {}
    for column in NUMERIC_STAT_COLUMNS:
        values = pd.to_numeric(history[column], errors="coerce").dropna()
        row[f"{column}_lag1"] = float(values.iloc[-1]) if not values.empty else np.nan
        for window in WINDOWS_MEAN:
            recent = values.tail(window)
            row[f"{column}_mean_{window}"] = float(recent.mean()) if not recent.empty else np.nan
        for window in WINDOWS_STD:
            recent = values.tail(window)
            row[f"{column}_std_{window}"] = float(recent.std(ddof=1)) if len(recent) >= 2 else np.nan

    dnp_values = pd.to_numeric(history["Did_Not_Play"], errors="coerce").dropna().tail(10)
    row["rest_days"] = rest_days
    row["dnp_rate_10"] = float(dnp_values.mean()) if not dnp_values.empty else 0.0
    row["games_played_before"] = float(len(history))
    row["dow"] = float(market_date.dayofweek) if pd.notna(market_date) else float(latest_history_date.dayofweek)
    row["month"] = float(market_date.month) if pd.notna(market_date) else float(latest_history_date.month)
    row["is_home"] = float(is_home)
    row["player_key"] = player_key
    row["Team_ID"] = str(player_team_id if player_team_id > 0 else -1)
    row["Opponent_ID"] = str(opponent_id if int(opponent_id) > 0 else -1)
    return row


def train_surrogate_models(
    training_df: pd.DataFrame,
    *,
    holdout_days: int = 14,
    iterations: int = 400,
    depth: int = 6,
    learning_rate: float = 0.05,
    random_seed: int = 42,
) -> tuple[dict[str, CatBoostRegressor], dict[str, float], dict[str, Any]]:
    if training_df.empty:
        raise RuntimeError("Training frame is empty.")

    max_date = pd.to_datetime(training_df["Date"], errors="coerce").max()
    valid_start = max_date - pd.Timedelta(days=int(max(1, holdout_days)))
    train_df = training_df.loc[pd.to_datetime(training_df["Date"], errors="coerce") < valid_start].copy()
    valid_df = training_df.loc[pd.to_datetime(training_df["Date"], errors="coerce") >= valid_start].copy()
    if train_df.empty or valid_df.empty:
        split_index = max(1, int(len(training_df) * 0.85))
        train_df = training_df.iloc[:split_index].copy()
        valid_df = training_df.iloc[split_index:].copy()

    medians = {
        column: float(pd.to_numeric(train_df[column], errors="coerce").median())
        for column in FEATURE_NUMERIC_COLUMNS
    }

    train_x = train_df[FEATURE_COLUMNS].copy()
    valid_x = valid_df[FEATURE_COLUMNS].copy()
    for column in FEATURE_NUMERIC_COLUMNS:
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce").fillna(medians[column])
        valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce").fillna(medians[column])
    for column in FEATURE_CATEGORICAL_COLUMNS:
        train_x[column] = train_x[column].astype(str)
        valid_x[column] = valid_x[column].astype(str)

    models: dict[str, CatBoostRegressor] = {}
    target_metrics: dict[str, Any] = {}
    for target in TARGETS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=int(depth),
            learning_rate=float(learning_rate),
            iterations=int(iterations),
            l2_leaf_reg=8.0,
            random_seed=int(random_seed),
            verbose=False,
        )
        model.fit(train_x, train_df[target], cat_features=FEATURE_CATEGORICAL_COLUMNS)
        pred = np.asarray(model.predict(valid_x), dtype="float64")
        actual = pd.to_numeric(valid_df[target], errors="coerce").to_numpy(dtype="float64")
        residual = pred - actual
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        models[target] = model
        target_metrics[target] = {
            "mae": mae,
            "rmse": rmse,
            "residual_std": float(np.std(residual, ddof=0)),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
        }

    summary = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "train_date_min": str(pd.to_datetime(train_df["Date"], errors="coerce").min().date()) if not train_df.empty else None,
        "train_date_max": str(pd.to_datetime(train_df["Date"], errors="coerce").max().date()) if not train_df.empty else None,
        "valid_date_min": str(pd.to_datetime(valid_df["Date"], errors="coerce").min().date()) if not valid_df.empty else None,
        "valid_date_max": str(pd.to_datetime(valid_df["Date"], errors="coerce").max().date()) if not valid_df.empty else None,
    }
    return models, medians, {"targets": target_metrics, "split": summary}


class SurrogateMarketPredictor:
    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        bundle = joblib.load(self.bundle_path)
        self.bundle = bundle
        self.models: dict[str, CatBoostRegressor] = bundle["models"]
        self.numeric_columns: list[str] = list(bundle["numeric_columns"])
        self.categorical_columns: list[str] = list(bundle["categorical_columns"])
        self.feature_columns: list[str] = list(bundle["feature_columns"])
        self.numeric_medians: dict[str, float] = {str(k): float(v) for k, v in bundle["numeric_medians"].items()}
        self.target_metrics: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in bundle["target_metrics"].items()}
        self.run_id = str(bundle.get("run_id", MODEL_TAG))
        self.min_history_games = int(bundle.get("min_history_games", 5))

    def _prepare_model_input(self, history_df: pd.DataFrame, market_context: dict[str, Any] | None) -> pd.DataFrame:
        row = build_feature_row(history_df, market_context=market_context, min_history_games=self.min_history_games)
        frame = pd.DataFrame([row], columns=self.feature_columns)
        for column in self.numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(self.numeric_medians.get(column, 0.0)))
        for column in self.categorical_columns:
            frame[column] = frame[column].astype(str)
        return frame

    def predict(
        self,
        history_df: pd.DataFrame,
        *,
        market_context: dict[str, Any] | None = None,
        return_debug: bool = False,
    ) -> dict[str, Any]:
        history = _prepare_history_frame(history_df)
        model_input = self._prepare_model_input(history, market_context=market_context)
        baseline: dict[str, float] = {}
        predicted: dict[str, float] = {}
        target_factors: dict[str, dict[str, Any]] = {}
        sigma_values: list[float] = []

        for target in TARGETS:
            history_values = pd.to_numeric(history[target], errors="coerce").dropna()
            baseline_value = float(history_values.tail(5).mean()) if not history_values.empty else 0.0
            model = self.models[target]
            pred_value = float(np.clip(float(model.predict(model_input)[0]), 0.0, np.inf))
            residual_std = float(self.target_metrics.get(target, {}).get("residual_std", 0.0))
            sigma = _history_sigma(history_values, residual_std)
            spike_prob = _spike_probability(history_values)
            baseline[target] = baseline_value
            predicted[target] = pred_value
            sigma_values.append(sigma)
            target_factors[target] = {
                "baseline_anchor": baseline_value,
                "normal_adjustment": float(pred_value - baseline_value),
                "tail_adjustment": 0.0,
                "spike_probability": spike_prob,
                "uncertainty_sigma": sigma,
                "projected_mean_from_latent": pred_value,
                "split_model_prediction": pred_value,
                "production_prediction": pred_value,
            }

        avg_prediction = float(np.mean(list(predicted.values()))) if predicted else 0.0
        avg_sigma = float(np.mean(sigma_values)) if sigma_values else 0.0
        sigma_ratio = avg_sigma / max(1.0, avg_prediction)
        belief_uncertainty = float(np.clip(0.16 + 0.72 * sigma_ratio, 0.05, 0.95))
        mp_history = pd.to_numeric(history["MP"], errors="coerce").dropna()
        feasibility = float(np.clip(mp_history.tail(10).mean() / 34.0, 0.20, 0.99)) if not mp_history.empty else 0.68
        dnp_rate = float(pd.to_numeric(history["Did_Not_Play"], errors="coerce").dropna().tail(10).mean()) if "Did_Not_Play" in history.columns else 0.0
        role_shift_risk = float(np.clip(0.18 + 0.55 * dnp_rate + 0.22 * max(0.0, sigma_ratio - 0.30), 0.05, 0.95))
        volatility_risk = float(np.clip(0.12 + 0.85 * sigma_ratio, 0.05, 0.95))
        context_pressure_risk = float(np.clip(0.12 + 0.40 * max(0.0, 1.0 - feasibility), 0.05, 0.90))

        explanation = {
            "baseline": baseline,
            "predicted": predicted,
            "predicted_raw_model": dict(predicted),
            "predicted_split_model": dict(predicted),
            "catboost_feature_versions": {target: [MODEL_TAG] for target in TARGETS},
            "data_quality": {
                "schema_repaired": False,
                "used_default_ids": False,
                "repaired_columns": [],
                "nan_feature_repaired": False,
                "nan_feature_count": 0,
                "nan_feature_columns": [],
                "fallback_blend": 0.0,
                "fallback_reasons": [MODEL_TAG],
                "active_like": True,
                "floor_guard_applied": False,
                "pts_residual_split_applied": False,
                "pts_spike_gate": 0.0,
                "pts_spike_delta": 0.0,
                "pts_split_activation": 0.0,
                "model_mode": "surrogate_market_predictor",
            },
            "latent_environment": {
                "slow_state_strength": 0.0,
                "environment_strength": 0.0,
                "belief_uncertainty": belief_uncertainty,
                "feasibility": feasibility,
                "role_shift_risk": role_shift_risk,
                "volatility_regime_risk": volatility_risk,
                "context_pressure_risk": context_pressure_risk,
            },
            "target_factors": target_factors,
        }
        if return_debug:
            explanation["debug"] = {
                "feature_row": sanitize_for_json(model_input.iloc[0].to_dict()),
                "bundle_path": str(self.bundle_path),
                "target_metrics": sanitize_for_json(self.target_metrics),
            }
        return explanation


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def write_bundle(
    *,
    bundle_path: Path,
    summary_path: Path | None,
    models: dict[str, CatBoostRegressor],
    numeric_medians: dict[str, float],
    target_metrics: dict[str, dict[str, Any]],
    split_summary: dict[str, Any],
    min_history_games: int,
) -> None:
    bundle = {
        "version": 1,
        "created_at_utc": utc_now_iso(),
        "run_id": MODEL_TAG,
        "model_type": "surrogate_market_predictor",
        "targets": TARGETS,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": FEATURE_NUMERIC_COLUMNS,
        "categorical_columns": FEATURE_CATEGORICAL_COLUMNS,
        "numeric_medians": numeric_medians,
        "target_metrics": target_metrics,
        "split_summary": split_summary,
        "min_history_games": int(min_history_games),
        "models": models,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    if summary_path is not None:
        summary_payload = dict(bundle)
        summary_payload.pop("models", None)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(sanitize_for_json(summary_payload), indent=2), encoding="utf-8")
