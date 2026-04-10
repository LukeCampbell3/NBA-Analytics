from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from .io_utils import ensure_dir, write_csv, write_json

# Import training module so joblib can resolve custom estimator classes
# (for example baseline fallback models).
from . import training as _training  # noqa: F401


MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
ROLE_TARGETS = {
    "hitter": ["H", "HR", "RBI"],
    "pitcher": ["K", "ER", "ERA"],
}


@dataclass
class DailyInferenceConfig:
    run_date: str
    season: int
    processed_dir: Path
    model_dir: Path
    out_dir: Path
    game_type: str = "R"
    min_history_rows: int = 5


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
    max_retries: int = 4,
) -> dict:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(2.0 * attempt, 8.0))


def _normalize_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float(default))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if np.isnan(out):
            return None
        return out
    except Exception:
        return None


def _fetch_daily_slate(run_date: pd.Timestamp, game_type: str = "R") -> pd.DataFrame:
    session = requests.Session()
    payload = _request_json(
        session,
        f"{MLB_STATS_API_BASE}/schedule",
        params={
            "sportId": 1,
            "date": run_date.strftime("%Y-%m-%d"),
            "gameType": game_type,
        },
    )

    rows: list[dict] = []
    for day in payload.get("dates", []) or []:
        official_date = str(day.get("date", "")).strip()
        for game in day.get("games", []) or []:
            status = game.get("status", {}) or {}
            coded = str(status.get("codedGameState", "")).strip().upper()
            detailed = str(status.get("detailedState", "")).strip()
            if detailed.lower() in {"postponed", "cancelled", "suspended"}:
                continue

            game_id = str(game.get("gamePk", "")).strip()
            game_time_utc = str(game.get("gameDate", "")).strip()
            teams = game.get("teams", {}) or {}
            home = teams.get("home", {}).get("team", {}) or {}
            away = teams.get("away", {}).get("team", {}) or {}

            home_id = int(home.get("id") or 0)
            away_id = int(away.get("id") or 0)
            if home_id <= 0 or away_id <= 0 or not game_id:
                continue

            home_name = str(home.get("abbreviation") or home.get("name") or "")
            away_name = str(away.get("abbreviation") or away.get("name") or "")

            rows.append(
                {
                    "Game_Date": official_date,
                    "Slate_Game_ID": game_id,
                    "Commence_Time_UTC": game_time_utc,
                    "Game_Status_Code": coded,
                    "Game_Status_Detail": detailed,
                    "Team_ID": home_id,
                    "Team": home_name,
                    "Opponent_ID": away_id,
                    "Opponent": away_name,
                    "Is_Home": 1,
                }
            )
            rows.append(
                {
                    "Game_Date": official_date,
                    "Slate_Game_ID": game_id,
                    "Commence_Time_UTC": game_time_utc,
                    "Game_Status_Code": coded,
                    "Game_Status_Detail": detailed,
                    "Team_ID": away_id,
                    "Team": away_name,
                    "Opponent_ID": home_id,
                    "Opponent": home_name,
                    "Is_Home": 0,
                }
            )

    if not rows:
        return pd.DataFrame()
    slate = pd.DataFrame.from_records(rows)
    slate = slate.sort_values(["Game_Date", "Commence_Time_UTC", "Slate_Game_ID", "Is_Home"], ascending=[True, True, True, False]).reset_index(drop=True)
    return slate


def _load_role_history(processed_dir: Path, season: int, role: str) -> pd.DataFrame:
    aggregate_path = processed_dir / f"{int(season)}_{role}s_processed.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing processed aggregate for role={role}: {aggregate_path}")
    df = pd.read_csv(aggregate_path)
    if df.empty:
        return df
    if "Player_Type" in df.columns:
        df["Player_Type"] = df["Player_Type"].astype(str).str.lower()
        df = df.loc[df["Player_Type"] == role].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.loc[df["Date"].notna()].copy()
    if df.empty:
        return df
    df["Player_ID"] = df.get("Player_ID", "").map(_normalize_id)
    df = df.sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)
    return df


def _apply_pregame_defaults(frame: pd.DataFrame, run_date: pd.Timestamp, role: str) -> pd.DataFrame:
    out = frame.copy()
    out["Date"] = run_date.normalize()
    out["Did_Not_Play"] = 0
    out["Month_sin"] = float(np.sin(2 * np.pi * (run_date.month / 12.0)))
    out["Month_cos"] = float(np.cos(2 * np.pi * (run_date.month / 12.0)))
    out["DayOfWeek_sin"] = float(np.sin(2 * np.pi * (run_date.dayofweek / 7.0)))
    out["DayOfWeek_cos"] = float(np.cos(2 * np.pi * (run_date.dayofweek / 7.0)))
    out["Market_Fetched_At_UTC"] = run_date.strftime("%Y-%m-%dT16:00:00Z")

    if role == "hitter":
        targets = ["H", "HR", "RBI"]
    else:
        targets = ["K", "ER", "ERA"]

    for target in targets:
        roll_col = f"{target}_rolling_avg"
        lag_col = f"{target}_lag1"
        market_col = f"Market_{target}"
        synthetic_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        books_col = f"Market_{target}_books"
        over_col = f"Market_{target}_over_price"
        under_col = f"Market_{target}_under_price"
        std_col = f"Market_{target}_line_std"
        gap_col = f"{target}_market_gap"

        roll = _safe_num(out.get(roll_col, pd.Series(np.nan, index=out.index)), 0.0)
        lag = _safe_num(out.get(lag_col, pd.Series(np.nan, index=out.index)), 0.0)
        fallback = roll.fillna(lag).fillna(0.0)

        out[market_col] = _safe_num(out.get(market_col, pd.Series(np.nan, index=out.index)), np.nan).fillna(fallback)
        out[synthetic_col] = _safe_num(out.get(synthetic_col, pd.Series(np.nan, index=out.index)), np.nan).fillna(fallback)
        source = out.get(source_col, pd.Series("", index=out.index)).astype(str).str.lower()
        out[source_col] = np.where(source.isin(["real", "synthetic", "baseline_fallback"]), source, "synthetic")
        out[books_col] = _safe_num(out.get(books_col, pd.Series(np.nan, index=out.index)), 0.0)
        out[over_col] = _safe_num(out.get(over_col, pd.Series(np.nan, index=out.index)), -110.0)
        out[under_col] = _safe_num(out.get(under_col, pd.Series(np.nan, index=out.index)), -110.0)
        out[std_col] = _safe_num(out.get(std_col, pd.Series(np.nan, index=out.index)), 0.0).clip(lower=0.0)
        out[gap_col] = _safe_num(out[market_col], 0.0) - _safe_num(out.get(roll_col, pd.Series(np.nan, index=out.index)), 0.0)

    return out


def _build_role_candidates(
    history_df: pd.DataFrame,
    slate_df: pd.DataFrame,
    *,
    run_date: pd.Timestamp,
    season: int,
    role: str,
    min_history_rows: int,
) -> pd.DataFrame:
    if history_df.empty or slate_df.empty:
        return pd.DataFrame()

    hist = history_df.loc[history_df["Date"] < run_date.normalize()].copy()
    if hist.empty:
        return pd.DataFrame()
    hist = hist.sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)
    history_rows = hist.groupby("Player", sort=False).size()
    latest = hist.groupby("Player", sort=False).tail(1).copy()
    latest["History_Rows"] = latest["Player"].map(history_rows).fillna(0).astype(int)
    latest["Last_History_Date"] = pd.to_datetime(latest["Date"], errors="coerce")
    latest = latest.loc[latest["History_Rows"] >= int(min_history_rows)].copy()
    if latest.empty:
        return pd.DataFrame()

    latest["Team_ID"] = pd.to_numeric(latest["Team_ID"], errors="coerce")
    slate = slate_df.copy()
    slate["Team_ID"] = pd.to_numeric(slate["Team_ID"], errors="coerce")
    merged = latest.merge(
        slate,
        on="Team_ID",
        how="inner",
        suffixes=("", "_slate"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["Player", "Commence_Time_UTC", "Slate_Game_ID"]).reset_index(drop=True)
    merged["__seq"] = merged.groupby("Player", sort=False).cumcount() + 1
    merged["Game_Index"] = _safe_num(merged.get("Game_Index", pd.Series(0.0, index=merged.index)), 0.0).astype(int) + merged["__seq"].astype(int)
    rest_days = (run_date.normalize() - pd.to_datetime(merged["Last_History_Date"], errors="coerce")).dt.days - 1
    merged["Rest_Days"] = pd.to_numeric(rest_days, errors="coerce").fillna(0.0).clip(lower=0.0)

    merged["Season"] = int(season)
    merged["Game_ID"] = merged["Slate_Game_ID"].astype(str)
    merged["Team"] = merged["Team_slate"]
    merged["Opponent"] = merged["Opponent_slate"]
    merged["Opponent_ID"] = pd.to_numeric(merged["Opponent_ID_slate"], errors="coerce").fillna(0).astype(int)
    merged["Is_Home"] = pd.to_numeric(merged["Is_Home_slate"], errors="coerce").fillna(0).astype(int)
    merged["Player_Type"] = role
    merged["Prediction_Run_Date"] = run_date.strftime("%Y-%m-%d")
    merged["Game_Date"] = run_date.strftime("%Y-%m-%d")

    merged = _apply_pregame_defaults(merged, run_date=run_date, role=role)
    return merged


def _load_target_artifact(model_dir: Path, role: str, target: str) -> dict:
    model_path = model_dir / role / f"{target.lower()}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact for role={role} target={target}: {model_path}")
    return joblib.load(model_path)


def _predict_with_artifact(artifact: dict, rows: pd.DataFrame) -> np.ndarray:
    feature_cols = list(artifact.get("feature_cols", []))
    if not feature_cols:
        raise RuntimeError("Model artifact has no feature columns")
    X = rows.reindex(columns=feature_cols).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)

    models = artifact.get("models", [])
    weights = artifact.get("weights", [])
    if not models:
        raise RuntimeError("Model artifact has no model objects")
    if not weights or len(weights) != len(models):
        weights = [1.0 / float(len(models))] * len(models)
    weights = np.array(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        weights = np.full(shape=(len(models),), fill_value=(1.0 / float(len(models))), dtype=np.float64)
    else:
        weights = weights / weight_sum

    preds = np.zeros(shape=(len(X),), dtype=np.float64)
    for idx, model in enumerate(models):
        model_pred = np.asarray(model.predict(X), dtype=np.float64)
        preds = preds + (weights[idx] * model_pred)
    return preds


def _build_long_rows(role_rows: pd.DataFrame, role: str, artifacts: dict[str, dict]) -> pd.DataFrame:
    targets = ROLE_TARGETS[role]
    working = role_rows.copy()
    for target in targets:
        artifact = artifacts[target]
        pred = _predict_with_artifact(artifact, working)
        working[f"Prediction_{target}"] = np.clip(pred, a_min=0.0, a_max=None)
        working[f"Baseline_{target}"] = _safe_num(working.get(f"{target}_rolling_avg", pd.Series(np.nan, index=working.index)), 0.0)
        working[f"Market_{target}"] = _safe_num(working.get(f"Market_{target}", pd.Series(np.nan, index=working.index)), np.nan)
        working[f"Edge_{target}"] = working[f"Prediction_{target}"] - working[f"Market_{target}"]

    records: list[dict] = []
    for _, row in working.iterrows():
        for target in targets:
            artifact = artifacts[target]
            selected = artifact.get("selected", {}) or {}
            records.append(
                {
                    "Prediction_Run_Date": str(row.get("Prediction_Run_Date", "")),
                    "Game_Date": str(row.get("Game_Date", "")),
                    "Commence_Time_UTC": str(row.get("Commence_Time_UTC", "")),
                    "Game_ID": str(row.get("Game_ID", "")),
                    "Game_Status_Code": str(row.get("Game_Status_Code", "")),
                    "Game_Status_Detail": str(row.get("Game_Status_Detail", "")),
                    "Player": str(row.get("Player", "")),
                    "Player_ID": _normalize_id(row.get("Player_ID")),
                    "Player_Type": role,
                    "Team": str(row.get("Team", "")),
                    "Team_ID": int(pd.to_numeric(row.get("Team_ID"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Team_ID"), errors="coerce")) else 0,
                    "Opponent": str(row.get("Opponent", "")),
                    "Opponent_ID": int(pd.to_numeric(row.get("Opponent_ID"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Opponent_ID"), errors="coerce")) else 0,
                    "Is_Home": int(pd.to_numeric(row.get("Is_Home"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("Is_Home"), errors="coerce")) else 0,
                    "Target": target,
                    "Prediction": _safe_float(row.get(f"Prediction_{target}")),
                    "Baseline": _safe_float(row.get(f"Baseline_{target}")),
                    "Market_Line": _safe_float(row.get(f"Market_{target}")),
                    "Edge": _safe_float(row.get(f"Edge_{target}")),
                    "History_Rows": int(pd.to_numeric(row.get("History_Rows"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("History_Rows"), errors="coerce")) else 0,
                    "Last_History_Date": pd.to_datetime(row.get("Last_History_Date"), errors="coerce").strftime("%Y-%m-%d")
                    if pd.notna(pd.to_datetime(row.get("Last_History_Date"), errors="coerce"))
                    else None,
                    "Model_Selected": str(selected.get("name", "")),
                    "Model_Members": ",".join(str(x) for x in selected.get("members", [])),
                    "Model_Weights": ",".join(str(x) for x in selected.get("weights", [])),
                    "Model_Val_MAE": _safe_float((selected or {}).get("mae")),
                    "Model_Val_RMSE": _safe_float((selected or {}).get("rmse")),
                }
            )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["Game_Date", "Commence_Time_UTC", "Game_ID", "Player_Type", "Player", "Target"]).reset_index(drop=True)
    return out


def generate_daily_prediction_pool(config: DailyInferenceConfig) -> dict:
    run_date = pd.to_datetime(config.run_date, errors="coerce")
    if pd.isna(run_date):
        raise ValueError(f"Invalid run_date: {config.run_date}")
    run_date = run_date.normalize()

    processed_dir = config.processed_dir.resolve()
    model_dir = config.model_dir.resolve()
    out_dir = ensure_dir(config.out_dir.resolve())
    run_stamp = run_date.strftime("%Y%m%d")

    slate_df = _fetch_daily_slate(run_date=run_date, game_type=str(config.game_type))
    if slate_df.empty:
        raise RuntimeError(
            f"No MLB slate rows found for {run_date.strftime('%Y-%m-%d')} (game_type={config.game_type})."
        )

    role_payloads: dict[str, dict] = {}
    role_long_frames: list[pd.DataFrame] = []
    for role in ["hitter", "pitcher"]:
        history_df = _load_role_history(processed_dir=processed_dir, season=int(config.season), role=role)
        candidates = _build_role_candidates(
            history_df=history_df,
            slate_df=slate_df,
            run_date=run_date,
            season=int(config.season),
            role=role,
            min_history_rows=int(config.min_history_rows),
        )
        if candidates.empty:
            role_payloads[role] = {
                "history_rows": int(len(history_df)),
                "candidate_rows": 0,
                "prediction_rows": 0,
                "targets": ROLE_TARGETS[role],
                "status": "no_candidates",
            }
            continue

        artifacts = {target: _load_target_artifact(model_dir=model_dir, role=role, target=target) for target in ROLE_TARGETS[role]}
        long_rows = _build_long_rows(candidates, role=role, artifacts=artifacts)
        role_long_frames.append(long_rows)

        role_payloads[role] = {
            "history_rows": int(len(history_df)),
            "candidate_rows": int(len(candidates)),
            "prediction_rows": int(len(long_rows)),
            "targets": ROLE_TARGETS[role],
            "status": "ok",
        }

    if not role_long_frames:
        raise RuntimeError(
            "Daily inference produced no predictions. "
            f"Role status: {role_payloads}"
        )

    pool_df = pd.concat(role_long_frames, ignore_index=True)
    pool_df = pool_df.sort_values(["Game_Date", "Commence_Time_UTC", "Game_ID", "Player_Type", "Player", "Target"]).reset_index(drop=True)

    pool_csv = out_dir / f"daily_prediction_pool_{run_stamp}.csv"
    pool_json = out_dir / f"daily_prediction_pool_{run_stamp}.json"
    write_csv(pool_df, pool_csv)

    summary = {
        "run_date": run_date.strftime("%Y-%m-%d"),
        "season": int(config.season),
        "game_type": str(config.game_type),
        "processed_dir": str(processed_dir),
        "model_dir": str(model_dir),
        "pool_csv": str(pool_csv),
        "rows": int(len(pool_df)),
        "games": int(pool_df["Game_ID"].nunique()),
        "players": int(pool_df["Player"].nunique()),
        "rows_by_role": {
            role: int(len(pool_df.loc[pool_df["Player_Type"] == role]))
            for role in ["hitter", "pitcher"]
        },
        "rows_by_target": {
            str(target): int(count)
            for target, count in pool_df["Target"].value_counts().sort_index().to_dict().items()
        },
        "role_status": role_payloads,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(pool_json, summary)
    return summary
