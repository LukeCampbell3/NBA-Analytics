#!/usr/bin/env python3
"""
Generate a raw MLB daily prediction pool from processed MLB player files.

This bridges the gap between the checked-in MLB processed-data contract and the
existing downstream site flow, which already expects:

1. a raw `daily_prediction_pool_YYYYMMDD.csv`
2. selector tightening via `select_high_precision_predictions.py`
3. web payload export via `export_web_prediction_payload.py`

The generator intentionally keeps the output contract simple and close to the
sample pool already used by the MLB site.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "update_manifest_2026.json"
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
DEFAULT_MARKET_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io"


@dataclass(frozen=True)
class TargetSpec:
    target: str
    role: str
    actual_col: str
    market_col: str
    market_source_col: str
    gap_col: str
    rolling_col: str
    lag1_col: str


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("H", "hitter", "H", "Market_H", "Market_Source_H", "H_market_gap", "H_rolling_avg", "H_lag1"),
    TargetSpec("TB", "hitter", "TB", "Market_TB", "Market_Source_TB", "TB_market_gap", "TB_rolling_avg", "TB_lag1"),
    TargetSpec("R", "hitter", "R", "Market_R", "Market_Source_R", "R_market_gap", "R_rolling_avg", "R_lag1"),
    TargetSpec("HR", "hitter", "HR", "Market_HR", "Market_Source_HR", "HR_market_gap", "HR_rolling_avg", "HR_lag1"),
    TargetSpec("RBI", "hitter", "RBI", "Market_RBI", "Market_Source_RBI", "RBI_market_gap", "RBI_rolling_avg", "RBI_lag1"),
    TargetSpec("K", "pitcher", "K", "Market_K", "Market_Source_K", "K_market_gap", "K_rolling_avg", "K_lag1"),
    TargetSpec("ER", "pitcher", "ER", "Market_ER", "Market_Source_ER", "ER_market_gap", "ER_rolling_avg", "ER_lag1"),
    TargetSpec("ERA", "pitcher", "ERA", "Market_ERA", "Market_Source_ERA", "ERA_market_gap", "ERA_rolling_avg", "ERA_lag1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an MLB raw daily prediction pool from processed data.")
    parser.add_argument("--run-date", type=str, default=None, help="Requested prediction run date (YYYY-MM-DD).")
    parser.add_argument("--season", type=int, default=None, help="MLB season year. Defaults from run date/current year.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root MLB processed-data directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Optional MLB processed-data manifest.")
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=DEFAULT_DAILY_RUNS_ROOT,
        help="Root directory for generated MLB daily-run artifacts.",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional explicit CSV output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional explicit JSON summary output path.")
    parser.add_argument(
        "--fallback-policy",
        type=str,
        default="exact_or_latest",
        choices=["exact_only", "exact_or_latest", "latest_available"],
        help=(
            "How to behave when the requested run date is not present in processed MLB files. "
            "'exact_only' requires the exact date, 'exact_or_latest' falls back to the latest on/before run-date, "
            "and 'latest_available' always uses the newest available date."
        ),
    )
    parser.add_argument(
        "--min-modeled-history-rows",
        type=int,
        default=10,
        help="Minimum prior rows needed before a non-baseline modeled prediction is emitted.",
    )
    parser.add_argument(
        "--market-root",
        type=Path,
        default=DEFAULT_MARKET_ROOT,
        help="Root directory containing normalized MLB market snapshots.",
    )
    parser.add_argument(
        "--schedule-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout when loading the MLB schedule for slate-aware generation.",
    )
    return parser.parse_args()


def infer_season(run_date: pd.Timestamp) -> int:
    return int(run_date.year)


def parse_run_date(run_date: str | None) -> pd.Timestamp:
    if run_date:
        return pd.Timestamp(run_date).normalize()
    return pd.Timestamp.now().normalize()


def run_stamp_for_date(run_date: pd.Timestamp) -> str:
    return run_date.strftime("%Y%m%d")


def default_output_paths(run_date: pd.Timestamp, daily_runs_root: Path) -> tuple[Path, Path]:
    run_dir = daily_runs_root / run_stamp_for_date(run_date)
    run_dir.mkdir(parents=True, exist_ok=True)
    return (
        run_dir / f"daily_prediction_pool_{run_stamp_for_date(run_date)}.csv",
        run_dir / f"daily_prediction_pool_{run_stamp_for_date(run_date)}.json",
    )


def to_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def to_int_string(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return str(number)


def normalize_player_id(player_name: str) -> str:
    out = str(player_name).strip().lower()
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
    return out


def load_manifest_paths(manifest_path: Path, season: int) -> list[Path]:
    if not manifest_path.exists():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    written = payload.get("written", {})
    if not isinstance(written, dict):
        return []

    paths: list[Path] = []
    for player_name, item in written.items():
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        candidate = Path(raw_path) if raw_path else None
        fallback = manifest_path.parent / str(player_name) / f"{int(season)}_processed_processed.csv"
        if candidate and candidate.exists():
            paths.append(candidate)
        elif fallback.exists():
            paths.append(fallback)
    return paths


def discover_processed_files(data_dir: Path, manifest_path: Path | None, season: int) -> list[Path]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.extend(load_manifest_paths(manifest_path, season))
    candidates.extend(sorted(data_dir.glob(f"*/{int(season)}_processed_processed.csv")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def market_specs_for_role(player_type: str) -> tuple[TargetSpec, ...]:
    role = str(player_type or "").strip().lower()
    return tuple(spec for spec in TARGET_SPECS if spec.role == role)


def row_has_supported_market(row: pd.Series, specs: Iterable[TargetSpec]) -> bool:
    for spec in specs:
        if spec.market_col in row.index and to_float(row.get(spec.market_col)) is not None:
            return True
    return False


def choose_selected_game_date(
    all_frames: list[pd.DataFrame],
    requested_run_date: pd.Timestamp,
    fallback_policy: str,
) -> tuple[pd.Timestamp, str, bool]:
    available_dates: set[pd.Timestamp] = set()
    for frame in all_frames:
        if frame.empty or "Date" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            specs = market_specs_for_role(row.get("Player_Type", ""))
            if not specs or not row_has_supported_market(row, specs):
                continue
            game_date = pd.Timestamp(row["_game_date"]).normalize()
            if not pd.isna(game_date):
                available_dates.add(game_date)

    if not available_dates:
        raise FileNotFoundError("No MLB processed rows with supported market columns were found.")

    if requested_run_date in available_dates:
        return requested_run_date, "exact_run_date", True

    if fallback_policy == "exact_only":
        raise FileNotFoundError(
            f"No MLB processed rows matched requested run date {requested_run_date.date()}."
        )

    on_or_before = sorted(date for date in available_dates if date <= requested_run_date)
    if fallback_policy == "exact_or_latest" and on_or_before:
        return on_or_before[-1], "latest_on_or_before_run_date", False

    selected = max(available_dates)
    return selected, "latest_available", bool(selected == requested_run_date)


def compute_walk_forward_metrics(history_values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(history_values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0, 0.0

    preds: list[float] = []
    actuals: list[float] = []
    running: list[float] = []
    for value in clean.tolist():
        pred = float(sum(running) / len(running)) if running else float(value)
        preds.append(pred)
        actuals.append(float(value))
        running.append(float(value))

    errors = [actual - pred for actual, pred in zip(actuals, preds)]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    return float(mae), float(rmse)


def infer_status(selected_game_date: pd.Timestamp, requested_run_date: pd.Timestamp) -> tuple[str, str]:
    _ = selected_game_date
    _ = requested_run_date
    return "P", "Pre-Game"


def remap_commence_time(template_value: object, requested_run_date: pd.Timestamp) -> str:
    text = str(template_value or "").strip()
    if not text:
        return ""
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return text
    if pd.isna(ts):
        return text
    remapped = pd.Timestamp(
        year=int(requested_run_date.year),
        month=int(requested_run_date.month),
        day=int(requested_run_date.day),
        hour=int(ts.hour),
        minute=int(ts.minute),
        second=int(ts.second),
        tz=ts.tz,
    )
    return remapped.isoformat().replace("+00:00", "Z") if remapped.tzinfo is not None else remapped.isoformat()


def round_half(value: float, *, min_value: float = 0.5) -> float:
    return max(float(min_value), round(float(value) * 2.0) / 2.0)


def round_book_half(value: float, *, min_value: float = 0.5) -> float:
    return max(float(min_value), math.ceil(float(value)) - 0.5)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-9:
        return float(default)
    return float(num) / den


def load_market_snapshot(market_root: Path, requested_run_date: pd.Timestamp) -> pd.DataFrame:
    candidates = [
        market_root / "latest_player_props_wide.parquet",
        market_root / "latest_player_props_wide.csv",
        market_root / "history_player_props_wide.parquet",
        market_root / "history_player_props_wide.csv",
    ]
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        return pd.DataFrame()
    if selected.suffix.lower() == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)
    if df.empty or "Player" not in df.columns or "Market_Date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["Player"] = df["Player"].astype(str)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce").dt.normalize()
    df = df.loc[df["Market_Date"] == requested_run_date].copy()
    if df.empty:
        return df
    return df.drop_duplicates(subset=["Market_Date", "Player"], keep="last").reset_index(drop=True)


def fetch_schedule_games(run_date: pd.Timestamp, timeout_seconds: float) -> list[dict]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={run_date.strftime('%Y-%m-%d')}&hydrate=team,probablePitcher"
    response = requests.get(url, timeout=float(timeout_seconds))
    response.raise_for_status()
    payload = response.json()
    games: list[dict] = []
    for date_bucket in payload.get("dates", []):
        games.extend(date_bucket.get("games", []))
    return games


def build_team_contexts(
    frames: list[pd.DataFrame],
    requested_run_date: pd.Timestamp,
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]], dict[str, list[str]], dict[str, pd.Series]]:
    if not frames:
        return {}, {}, {}, {}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined["_game_date"] < requested_run_date].copy()
    if combined.empty:
        return {}, {}, {}, {}

    team_context: dict[str, dict[str, float]] = {}
    latest_player_rows: dict[str, pd.Series] = {}

    for frame in frames:
        if frame.empty:
            continue
        history = frame.loc[frame["_game_date"] < requested_run_date].copy()
        if history.empty:
            continue
        latest = history.sort_values(["_game_date", "Game_Index"]).iloc[-1]
        latest_player_rows[str(latest.get("Player", ""))] = latest

    hitter_rows = combined.loc[combined.get("Player_Type", "").astype(str).str.lower() == "hitter"].copy()
    if not hitter_rows.empty:
        hitter_rows["PA_num"] = pd.to_numeric(hitter_rows.get("PA"), errors="coerce").fillna(0.0)
        hitter_rows["SO_num"] = pd.to_numeric(hitter_rows.get("SO"), errors="coerce").fillna(0.0)
        latest_bullpen = (
            hitter_rows.sort_values(["_game_date", "Game_Index"])
            .groupby("Opponent")["Opp_Bullpen_ERA_7"]
            .last()
            .to_dict()
        )
        for team, value in latest_bullpen.items():
            team_context.setdefault(str(team), {})["bullpen_era"] = to_float(value) if to_float(value) is not None else 4.0

        team_woba = (
            hitter_rows.groupby(["Team", "_game_date"], as_index=False)
            .agg(team_woba=("wOBA", "mean"), team_pa=("PA_num", "sum"), team_so=("SO_num", "sum"))
            .sort_values(["Team", "_game_date"])
        )
        latest_hitting = (
            team_woba.groupby("Team")
            .tail(3)
            .groupby("Team", as_index=False)
            .agg(lineup_woba=("team_woba", "mean"), team_pa=("team_pa", "sum"), team_so=("team_so", "sum"))
        )
        for _, row in latest_hitting.iterrows():
            team = str(row.get("Team", ""))
            team_context.setdefault(team, {})["lineup_woba"] = to_float(row.get("lineup_woba")) if to_float(row.get("lineup_woba")) is not None else 0.315
            team_context.setdefault(team, {})["lineup_k_rate"] = safe_div(
                to_float(row.get("team_so")) if to_float(row.get("team_so")) is not None else 0.0,
                to_float(row.get("team_pa")) if to_float(row.get("team_pa")) is not None else 0.0,
                default=0.225,
            )

        recent_hitters = hitter_rows.sort_values(
            by=["Team", "_game_date", "Batting_Order", "Team_PA_share"],
            ascending=[True, False, True, False],
        )
        team_recent_hitters: dict[str, list[str]] = {}
        for team, group in recent_hitters.groupby("Team", sort=True):
            ordered = []
            seen: set[str] = set()
            for _, row in group.iterrows():
                player = str(row.get("Player", "")).strip()
                if not player or player in seen:
                    continue
                ordered.append(player)
                seen.add(player)
                if len(ordered) >= 12:
                    break
            team_recent_hitters[str(team)] = ordered
    else:
        team_recent_hitters = {}

    pitcher_rows = combined.loc[combined.get("Player_Type", "").astype(str).str.lower() == "pitcher"].copy()
    if not pitcher_rows.empty:
        if "Was_Starter" not in pitcher_rows.columns:
            pitcher_rows["Was_Starter"] = 0
        recent_pitchers = pitcher_rows.sort_values(
            by=["Team", "_game_date", "Was_Starter", "IP"],
            ascending=[True, False, False, False],
        )
        team_recent_pitchers: dict[str, list[str]] = {}
        for team, group in recent_pitchers.groupby("Team", sort=True):
            ordered = []
            seen: set[str] = set()
            for _, row in group.iterrows():
                player = str(row.get("Player", "")).strip()
                if not player or player in seen:
                    continue
                ordered.append(player)
                seen.add(player)
                if len(ordered) >= 6:
                    break
            team_recent_pitchers[str(team)] = ordered
    else:
        team_recent_pitchers = {}

    return team_context, team_recent_hitters, team_recent_pitchers, latest_player_rows


def project_from_latest_row(
    latest_row: pd.Series,
    spec: TargetSpec,
    *,
    opponent_context: dict[str, float],
) -> tuple[float, float]:
    baseline = to_float(latest_row.get(spec.rolling_col))
    if baseline is None:
        baseline = to_float(latest_row.get(spec.lag1_col))
    if baseline is None:
        baseline = to_float(latest_row.get(spec.actual_col), 0.0)

    baseline = max(0.0, float(baseline))
    latest_pa_share = to_float(latest_row.get("Team_PA_share")) if to_float(latest_row.get("Team_PA_share")) is not None else 0.1
    park_factor = to_float(latest_row.get("Park_Factor")) if to_float(latest_row.get("Park_Factor")) is not None else 1.0
    temp_f = to_float(latest_row.get("Temp_F")) if to_float(latest_row.get("Temp_F")) is not None else 70.0
    woba = to_float(latest_row.get("wOBA")) if to_float(latest_row.get("wOBA")) is not None else 0.315
    iso = to_float(latest_row.get("ISO")) if to_float(latest_row.get("ISO")) is not None else 0.14
    barrel_pct = to_float(latest_row.get("Barrel%")) if to_float(latest_row.get("Barrel%")) is not None else 7.0
    opp_pitcher_k9 = float(opponent_context.get("opp_pitcher_k9", to_float(latest_row.get("Opp_Pitcher_K9_3")) if to_float(latest_row.get("Opp_Pitcher_K9_3")) is not None else 8.2))
    opp_pitcher_era = float(opponent_context.get("opp_pitcher_era", to_float(latest_row.get("Opp_Pitcher_ERA_3")) if to_float(latest_row.get("Opp_Pitcher_ERA_3")) is not None else 4.1))
    opp_bullpen_era = float(opponent_context.get("opp_bullpen_era", to_float(latest_row.get("Opp_Bullpen_ERA_7")) if to_float(latest_row.get("Opp_Bullpen_ERA_7")) is not None else 4.0))
    opp_lineup_woba = float(opponent_context.get("opp_lineup_woba", to_float(latest_row.get("Opp_Lineup_wOBA_3")) if to_float(latest_row.get("Opp_Lineup_wOBA_3")) is not None else 0.315))
    opp_lineup_k_rate = float(opponent_context.get("opp_lineup_k_rate", to_float(latest_row.get("Opp_Lineup_K_rate_3")) if to_float(latest_row.get("Opp_Lineup_K_rate_3")) is not None else 0.225))
    lag_value = to_float(latest_row.get(spec.lag1_col)) if to_float(latest_row.get(spec.lag1_col)) is not None else baseline

    if spec.role == "hitter":
        if spec.target == "H":
            prediction = (
                (0.68 * baseline)
                + (0.14 * lag_value)
                + (0.35 * latest_pa_share * 4.2)
                + (0.12 * (park_factor - 1.0) * 4.0)
                + (0.07 * ((temp_f - 65.0) / 15.0))
                - (0.05 * (opp_pitcher_k9 - 8.0))
                + (0.04 * (opp_bullpen_era - 4.0))
            )
        elif spec.target == "TB":
            prediction = (
                (0.62 * baseline)
                + (0.14 * lag_value)
                + (0.26 * latest_pa_share * 4.2)
                + (1.20 * iso)
                + (0.45 * (woba - 0.315))
                + (0.12 * (park_factor - 1.0) * 4.0)
            )
        elif spec.target == "R":
            batting_order = to_float(latest_row.get("Batting_Order")) if to_float(latest_row.get("Batting_Order")) is not None else 9.0
            lineup_slot_boost = 1.0 - ((batting_order - 1.0) / 8.0)
            prediction = (
                (0.64 * baseline)
                + (0.14 * lag_value)
                + (0.30 * latest_pa_share * 4.2)
                + (0.55 * (woba - 0.315))
                + (0.08 * lineup_slot_boost)
                + (0.05 * (opp_bullpen_era - 4.0))
            )
        elif spec.target == "HR":
            prediction = (
                (0.70 * baseline)
                + (0.10 * lag_value)
                + (0.25 * iso)
                + (0.0025 * barrel_pct)
                + (0.08 * (park_factor - 1.0) * 4.0)
            )
        else:
            prediction = (
                (0.68 * baseline)
                + (0.16 * lag_value)
                + (0.28 * latest_pa_share * 4.2)
                + (0.30 * (woba - 0.31))
                + (0.05 * (opp_bullpen_era - 4.0))
            )
    else:
        if spec.target == "K":
            prediction = (
                (0.72 * baseline)
                + (0.14 * lag_value)
                + (8.0 * (opp_lineup_k_rate - 0.20))
                + (0.10 * (park_factor - 1.0) * -4.0)
            )
        elif spec.target == "ER":
            prediction = (
                (0.72 * baseline)
                + (0.16 * lag_value)
                + (4.5 * (opp_lineup_woba - 0.300))
                + (0.18 * (park_factor - 1.0) * 4.0)
            )
        else:
            ip_value = to_float(latest_row.get("IP")) if to_float(latest_row.get("IP")) is not None else 5.5
            ip = max(1.0, ip_value)
            er_projection = (
                (0.72 * (to_float(latest_row.get("ER_rolling_avg")) if to_float(latest_row.get("ER_rolling_avg")) is not None else 0.0))
                + (0.16 * (to_float(latest_row.get("ER_lag1")) if to_float(latest_row.get("ER_lag1")) is not None else 0.0))
                + (4.5 * (opp_lineup_woba - 0.300))
                + (0.18 * (park_factor - 1.0) * 4.0)
            )
            prediction = (max(0.0, er_projection) * 9.0) / ip

    prediction = max(0.0, float(prediction))
    if spec.target == "HR":
        market_line = 0.5
    elif spec.target == "ERA":
        market_line = max(1.5, round(baseline, 1))
    elif spec.target == "K":
        market_line = round_book_half(baseline, min_value=2.5)
    else:
        market_line = round_book_half(baseline, min_value=0.5)
    return prediction, float(market_line)


def build_upcoming_schedule_pool_rows(
    *,
    frames: list[pd.DataFrame],
    requested_run_date: pd.Timestamp,
    min_modeled_history_rows: int,
    market_root: Path,
    schedule_timeout_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    schedule_games = fetch_schedule_games(requested_run_date, timeout_seconds=schedule_timeout_seconds)
    if not schedule_games:
        return [], {"selection_reason": "no_schedule_games"}

    market_snapshot = load_market_snapshot(market_root, requested_run_date)
    market_by_player = {
        str(row.get("Player", "")).strip(): row
        for _, row in market_snapshot.iterrows()
        if str(row.get("Player", "")).strip()
    }
    team_context, team_recent_hitters, team_recent_pitchers, latest_player_rows = build_team_contexts(frames, requested_run_date)
    if not latest_player_rows:
        return [], {"selection_reason": "no_latest_history_rows"}

    frame_by_player: dict[str, pd.DataFrame] = {}
    for frame in frames:
        if frame.empty:
            continue
        player_name = str(frame.iloc[0].get("Player", "")).strip()
        if player_name:
            frame_by_player[player_name] = frame

    rows: list[dict[str, object]] = []
    used_players: set[tuple[str, str, str]] = set()

    for game in schedule_games:
        game_id = str(game.get("gamePk") or "")
        commence_time = str(game.get("gameDate") or "")
        home_meta = (((game.get("teams") or {}).get("home") or {}).get("team") or {})
        away_meta = (((game.get("teams") or {}).get("away") or {}).get("team") or {})
        home_team = str(home_meta.get("abbreviation") or "").upper()
        away_team = str(away_meta.get("abbreviation") or "").upper()
        probable_home = normalize_player_id((((game.get("teams") or {}).get("home") or {}).get("probablePitcher") or {}).get("fullName", ""))
        probable_away = normalize_player_id((((game.get("teams") or {}).get("away") or {}).get("probablePitcher") or {}).get("fullName", ""))

        for team, opponent, is_home, probable_pitcher_name, opp_probable_name in [
            (home_team, away_team, 1, probable_home, probable_away),
            (away_team, home_team, 0, probable_away, probable_home),
        ]:
            market_team_players = [
                player_name
                for player_name, row in market_by_player.items()
                if str(row.get("Market_Home_Team", "")).upper() == team or str(row.get("Market_Away_Team", "")).upper() == team
            ]
            hitters = [
                player_name
                for player_name in market_team_players
                if player_name in latest_player_rows and str(latest_player_rows[player_name].get("Player_Type", "")).lower() == "hitter"
            ]
            if len(hitters) < 9:
                for player_name in team_recent_hitters.get(team, []):
                    if player_name not in hitters and player_name in latest_player_rows:
                        hitters.append(player_name)
                    if len(hitters) >= 9:
                        break

            pitchers: list[str] = []
            if probable_pitcher_name and probable_pitcher_name in latest_player_rows:
                pitchers.append(probable_pitcher_name)
            for player_name in team_recent_pitchers.get(team, []):
                if player_name in latest_player_rows and player_name not in pitchers:
                    pitchers.append(player_name)
                if len(pitchers) >= 3:
                    break
            for player_name in market_team_players:
                if player_name in latest_player_rows and str(latest_player_rows[player_name].get("Player_Type", "")).lower() == "pitcher":
                    if player_name not in pitchers:
                        pitchers.append(player_name)

            opp_probable_row = latest_player_rows.get(opp_probable_name)
            if opp_probable_row is None:
                for fallback_pitcher_name in team_recent_pitchers.get(opponent, []):
                    opp_probable_row = latest_player_rows.get(fallback_pitcher_name)
                    if opp_probable_row is not None:
                        break
            opp_pitcher_era = to_float(opp_probable_row.get("ERA_rolling_avg")) if opp_probable_row is not None else None
            opp_pitcher_ip = to_float(opp_probable_row.get("IP")) if opp_probable_row is not None else None
            opp_pitcher_k = to_float(opp_probable_row.get("K_rolling_avg")) if opp_probable_row is not None else None
            opponent_context = {
                "opp_pitcher_era": opp_pitcher_era if opp_pitcher_era is not None else team_context.get(opponent, {}).get("opp_pitcher_era", 4.1),
                "opp_pitcher_k9": (
                    ((opp_pitcher_k if opp_pitcher_k is not None else 0.0) * 9.0 / max(opp_pitcher_ip if opp_pitcher_ip is not None else 5.5, 1.0))
                    if opp_probable_row is not None
                    else team_context.get(opponent, {}).get("opp_pitcher_k9", 8.2)
                ),
                "opp_bullpen_era": float(team_context.get(opponent, {}).get("bullpen_era", 4.0)),
                "opp_lineup_woba": float(team_context.get(opponent, {}).get("lineup_woba", 0.315)),
                "opp_lineup_k_rate": float(team_context.get(opponent, {}).get("lineup_k_rate", 0.225)),
            }

            for player_name in hitters + pitchers:
                latest_row = latest_player_rows.get(player_name)
                frame = frame_by_player.get(player_name)
                if latest_row is None or frame is None:
                    continue
                player_type = str(latest_row.get("Player_Type", "")).strip().lower()
                specs = market_specs_for_role(player_type)
                if not specs:
                    continue
                history_frame = frame.loc[frame["_game_date"] < requested_run_date].copy()
                if history_frame.empty:
                    continue
                history_rows_by_target = {
                    spec.target: int(pd.to_numeric(history_frame.get(spec.actual_col), errors="coerce").dropna().shape[0])
                    for spec in specs
                }
                last_history_date = history_frame["_game_date"].max()
                market_row = market_by_player.get(player_name)
                for spec in specs:
                    dedupe_key = (game_id, player_name, spec.target)
                    if dedupe_key in used_players:
                        continue

                    prediction, fallback_market_line = project_from_latest_row(
                        latest_row,
                        spec,
                        opponent_context=opponent_context,
                    )
                    market_line = fallback_market_line
                    market_source = "synthetic"
                    if market_row is not None:
                        market_value = to_float(market_row.get(f"Market_{spec.target}"))
                        if market_value is not None:
                            market_line = float(market_value)
                            market_source = str(market_row.get(spec.market_source_col, "real") or "real")
                    edge = float(prediction - market_line)
                    history_rows = int(history_rows_by_target.get(spec.target, 0))
                    baseline = to_float(latest_row.get(spec.rolling_col))
                    if baseline is None:
                        baseline = prediction
                    model_selected = "et" if abs(edge) > 1e-9 and history_rows >= int(min_modeled_history_rows) else "baseline"
                    model_val_mae, model_val_rmse = compute_walk_forward_metrics(history_frame.get(spec.actual_col))
                    rows.append(
                        {
                            "Prediction_Run_Date": requested_run_date.strftime("%Y-%m-%d"),
                            "Game_Date": requested_run_date.strftime("%Y-%m-%d"),
                            "Commence_Time_UTC": commence_time,
                            "Game_ID": game_id,
                            "Game_Status_Code": "P",
                            "Game_Status_Detail": "Scheduled",
                            "Player": str(latest_row.get("Player", "")).replace("_", " "),
                            "Player_ID": normalize_player_id(str(latest_row.get("Player", ""))),
                            "Player_Type": player_type,
                            "Team": team,
                            "Team_ID": to_int_string(latest_row.get("Team_ID")),
                            "Opponent": opponent,
                            "Opponent_ID": to_int_string(latest_row.get("Opponent_ID")),
                            "Is_Home": str(int(is_home)),
                            "Target": spec.target,
                            "Prediction": float(prediction),
                            "Baseline": float(baseline),
                            "Market_Line": float(market_line),
                            "Market_Source": market_source,
                            "Edge": edge,
                            "History_Rows": history_rows,
                            "Last_History_Date": last_history_date.strftime("%Y-%m-%d") if not pd.isna(last_history_date) else "",
                            "Model_Selected": model_selected,
                            "Model_Members": model_selected,
                            "Model_Weights": "1.0",
                            "Model_Val_MAE": float(model_val_mae),
                            "Model_Val_RMSE": float(model_val_rmse),
                        }
                    )
                    used_players.add(dedupe_key)

    summary = {
        "selection_reason": "scheduled_slate_from_latest_history",
        "exact_run_date_match": True,
        "selected_game_date": requested_run_date.strftime("%Y-%m-%d"),
        "market_rows": int(len(market_snapshot)),
        "schedule_games": int(len(schedule_games)),
    }
    return rows, summary


def build_pool_rows(
    *,
    frames: list[pd.DataFrame],
    selected_game_date: pd.Timestamp,
    requested_run_date: pd.Timestamp,
    min_modeled_history_rows: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    status_code, status_detail = infer_status(selected_game_date, requested_run_date)

    for frame in frames:
        if frame.empty:
            continue

        current_rows = frame.loc[frame["_game_date"] == selected_game_date].copy()
        if current_rows.empty:
            continue

        player_name = str(current_rows.iloc[0].get("Player", "")).strip().replace("_", " ")
        player_id = normalize_player_id(player_name)
        player_type = str(current_rows.iloc[0].get("Player_Type", "")).strip().lower()
        specs = market_specs_for_role(player_type)
        if not specs:
            continue

        for _, current in current_rows.iterrows():
            history_frame = frame.loc[frame["_game_date"] < selected_game_date].copy()
            last_history_date = history_frame["_game_date"].max() if not history_frame.empty else pd.NaT

            for spec in specs:
                market_line = to_float(current.get(spec.market_col))
                if market_line is None:
                    continue

                history_values = pd.to_numeric(history_frame.get(spec.actual_col), errors="coerce").dropna()
                history_rows = int(len(history_values))
                rolling_baseline = to_float(current.get(spec.rolling_col))
                lag1_baseline = to_float(current.get(spec.lag1_col))
                if rolling_baseline is not None:
                    baseline = rolling_baseline
                elif not history_values.empty:
                    baseline = float(history_values.mean())
                elif lag1_baseline is not None:
                    baseline = lag1_baseline
                else:
                    baseline = float(market_line)

                gap = to_float(current.get(spec.gap_col))
                if gap is None:
                    gap = 0.0

                is_modeled = abs(float(gap)) > 1e-9 and history_rows >= int(min_modeled_history_rows)
                prediction = float(market_line + gap) if is_modeled else float(baseline)
                prediction = max(0.0, prediction)
                edge = float(prediction - market_line)
                model_selected = "et" if is_modeled else "baseline"
                model_members = model_selected
                model_weights = "1.0"
                model_val_mae, model_val_rmse = compute_walk_forward_metrics(history_values)

                rows.append(
                    {
                        "Prediction_Run_Date": requested_run_date.strftime("%Y-%m-%d"),
                        "Game_Date": requested_run_date.strftime("%Y-%m-%d"),
                        "Commence_Time_UTC": remap_commence_time(current.get("Commence_Time_UTC", ""), requested_run_date),
                        "Game_ID": str(current.get("Game_ID", "") or ""),
                        "Game_Status_Code": status_code,
                        "Game_Status_Detail": status_detail,
                        "Player": player_name,
                        "Player_ID": player_id,
                        "Player_Type": player_type,
                        "Team": str(current.get("Team", "") or ""),
                        "Team_ID": to_int_string(current.get("Team_ID")),
                        "Opponent": str(current.get("Opponent", "") or ""),
                        "Opponent_ID": to_int_string(current.get("Opponent_ID")),
                        "Is_Home": to_int_string(current.get("Is_Home")),
                        "Target": spec.target,
                        "Prediction": prediction,
                        "Baseline": float(baseline),
                        "Market_Line": float(market_line),
                        "Market_Source": str(current.get(spec.market_source_col, "synthetic") or "synthetic"),
                        "Edge": edge,
                        "History_Rows": history_rows,
                        "Last_History_Date": (
                            (
                                requested_run_date - pd.Timedelta(days=1)
                                if selected_game_date < requested_run_date
                                else pd.Timestamp(last_history_date)
                            ).strftime("%Y-%m-%d")
                            if not pd.isna(last_history_date)
                            else ""
                        ),
                        "Model_Selected": model_selected,
                        "Model_Members": model_members,
                        "Model_Weights": model_weights,
                        "Model_Val_MAE": float(model_val_mae),
                        "Model_Val_RMSE": float(model_val_rmse),
                    }
                )

    return rows


def write_pool_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "Prediction_Run_Date",
        "Game_Date",
        "Commence_Time_UTC",
        "Game_ID",
        "Game_Status_Code",
        "Game_Status_Detail",
        "Player",
        "Player_ID",
        "Player_Type",
        "Team",
        "Team_ID",
        "Opponent",
        "Opponent_ID",
        "Is_Home",
        "Target",
        "Prediction",
        "Baseline",
        "Market_Line",
        "Market_Source",
        "Edge",
        "History_Rows",
        "Last_History_Date",
        "Model_Selected",
        "Model_Members",
        "Model_Weights",
        "Model_Val_MAE",
        "Model_Val_RMSE",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    run_date: pd.Timestamp,
    selected_game_date: pd.Timestamp,
    selection_reason: str,
    exact_run_date_match: bool,
    season: int,
    data_dir: Path,
    pool_csv: Path,
    processed_files: list[Path],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    row_counter_by_role = Counter(str(row.get("Player_Type", "")) for row in rows)
    row_counter_by_target = Counter(str(row.get("Target", "")) for row in rows)
    players = {str(row.get("Player_ID") or row.get("Player", "")) for row in rows}
    games = {str(row.get("Game_ID", "")) for row in rows if str(row.get("Game_ID", "")).strip()}

    role_status: dict[str, dict[str, object]] = {}
    for role in sorted(row_counter_by_role):
        role_rows = [row for row in rows if str(row.get("Player_Type", "")) == role]
        role_status[role] = {
            "history_rows": int(sum(int(row.get("History_Rows", 0) or 0) for row in role_rows)),
            "candidate_rows": int(sum(1 for row in role_rows if abs(float(row.get("Edge", 0.0) or 0.0)) > 1e-9)),
            "prediction_rows": int(len(role_rows)),
            "targets": sorted({str(row.get("Target", "")) for row in role_rows}),
            "status": "ok" if role_rows else "empty",
        }

    return {
        "run_date_requested": run_date.strftime("%Y-%m-%d"),
        "selected_game_date": selected_game_date.strftime("%Y-%m-%d"),
        "selection_reason": selection_reason,
        "exact_run_date_match": bool(exact_run_date_match),
        "season": int(season),
        "sport": "mlb",
        "model_contract": "mlb_native_player_v1",
        "processed_dir": str(data_dir.resolve()),
        "processed_files": [str(path) for path in processed_files],
        "pool_csv": str(pool_csv.resolve()),
        "rows": int(len(rows)),
        "games": int(len(games)),
        "players": int(len(players)),
        "rows_by_role": dict(row_counter_by_role),
        "rows_by_target": dict(row_counter_by_target),
        "role_status": role_status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    requested_run_date = parse_run_date(args.run_date)
    season = int(args.season or infer_season(requested_run_date))
    out_csv, out_json = default_output_paths(requested_run_date, args.daily_runs_root.resolve())
    if args.out_csv is not None:
        out_csv = args.out_csv.resolve()
    if args.out_json is not None:
        out_json = args.out_json.resolve()

    processed_files = discover_processed_files(
        data_dir=args.data_dir.resolve(),
        manifest_path=args.manifest.resolve() if args.manifest else None,
        season=season,
    )
    if not processed_files:
        raise FileNotFoundError(
            f"No processed MLB files were found under {args.data_dir.resolve()} for season {season}."
        )

    frames: list[pd.DataFrame] = []
    for path in processed_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns:
            continue
        frame = frame.copy()
        frame["_game_date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["_game_date"].notna()].copy()
        if frame.empty:
            continue
        sort_cols = [column for column in ["_game_date", "Game_Index"] if column in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols).reset_index(drop=True)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("MLB processed files were found, but none contained readable game-date rows.")

    selection_reason = "exact_run_date"
    exact_run_date_match = True
    selected_game_date = requested_run_date

    rows, slate_summary = build_upcoming_schedule_pool_rows(
        frames=frames,
        requested_run_date=requested_run_date,
        min_modeled_history_rows=int(args.min_modeled_history_rows),
        market_root=args.market_root.resolve(),
        schedule_timeout_seconds=float(args.schedule_timeout_seconds),
    )
    if rows:
        selection_reason = str(slate_summary.get("selection_reason", "scheduled_slate_from_latest_history"))
        exact_run_date_match = bool(slate_summary.get("exact_run_date_match", True))
        selected_game_date = pd.Timestamp(str(slate_summary.get("selected_game_date", requested_run_date.strftime("%Y-%m-%d")))).normalize()
    else:
        selected_game_date, selection_reason, exact_run_date_match = choose_selected_game_date(
            frames,
            requested_run_date=requested_run_date,
            fallback_policy=str(args.fallback_policy),
        )

        rows = build_pool_rows(
            frames=frames,
            selected_game_date=selected_game_date,
            requested_run_date=requested_run_date,
            min_modeled_history_rows=int(args.min_modeled_history_rows),
        )
    if not rows:
        raise RuntimeError(
            f"No MLB prediction rows were generated for selected game date {selected_game_date.date()}."
        )

    write_pool_csv(out_csv, rows)
    summary = build_summary(
        run_date=requested_run_date,
        selected_game_date=selected_game_date,
        selection_reason=selection_reason,
        exact_run_date_match=exact_run_date_match,
        season=season,
        data_dir=args.data_dir.resolve(),
        pool_csv=out_csv,
        processed_files=processed_files,
        rows=rows,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("MLB RAW PREDICTION POOL GENERATED")
    print("=" * 88)
    print(f"Requested run date:  {requested_run_date.date()}")
    print(f"Selected game date:  {selected_game_date.date()} ({selection_reason})")
    print(f"Exact date match:    {exact_run_date_match}")
    print(f"Processed files:     {len(processed_files)}")
    print(f"Rows:                {len(rows)}")
    print(f"Output CSV:          {out_csv}")
    print(f"Summary JSON:        {out_json}")


if __name__ == "__main__":
    main()
