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
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.decision_engine.matchup_network import (  # noqa: E402
    MatchupNetworkSignal,
    build_matchup_network_signal,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "update_manifest_2026.json"
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
DEFAULT_MARKET_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io"

PREFERRED_BOOKMAKER_KEYS: tuple[str, ...] = (
    "fanduel",
    "draftkings",
    "bet365",
    "mgm",
    "caesars",
    "fanatics",
)
BOOKMAKER_TITLES = {
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "bet365": "bet365",
    "mgm": "BetMGM",
    "caesars": "Caesars",
    "fanatics": "Fanatics",
}
STANDARD_MARKET_LINES = {
    "H": 0.5,
    "TB": 1.5,
    "R": 0.5,
    "HR": 0.5,
    "RBI": 0.5,
}
STARTER_LIKE_MIN_IP = 3.0
STARTER_LIKE_MIN_PITCHES = 45.0
PITCHER_WORKLOAD_RECENT_STARTS = 5


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

TARGET_MARKET_KEYS: dict[str, str] = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "HR": "batter_home_runs",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
    "ER": "pitcher_earned_runs",
}


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


def valid_american_price(value: object) -> float | None:
    price = to_float(value)
    if price is None or (-100.0 < price < 100.0) or abs(price - round(price)) > 1e-6:
        return None
    return price


def _best_book_price(rows: pd.DataFrame, price_column: str) -> tuple[float | None, str, str]:
    offers: list[tuple[float, int, str]] = []
    preference = {key: index for index, key in enumerate(PREFERRED_BOOKMAKER_KEYS)}
    for _, offer in rows.iterrows():
        book_key = str(offer.get("bookmaker_key", "")).strip().lower()
        price = valid_american_price(offer.get(price_column))
        if book_key not in preference or price is None:
            continue
        offers.append((price, -preference[book_key], book_key))
    if not offers:
        return None, "", ""
    price, _, book_key = max(offers)
    return price, book_key, BOOKMAKER_TITLES.get(book_key, book_key)


def select_bettable_market_line(market_part: pd.DataFrame, target: str) -> tuple[float, pd.DataFrame]:
    priced = market_part.loc[
        market_part["line_num"].notna()
        & market_part[["over_price_num", "under_price_num"]].notna().any(axis=1)
    ].copy()
    if priced.empty:
        raise ValueError("market has no priced lines")

    priced["bookmaker_key"] = priced.get("bookmaker_key", "").astype(str).str.strip().str.lower()
    preferred = priced.loc[priced["bookmaker_key"].isin(PREFERRED_BOOKMAKER_KEYS)].copy()
    line_candidates = preferred if not preferred.empty else priced
    standard_line = STANDARD_MARKET_LINES.get(target)
    if standard_line is not None:
        line_candidates = preferred.loc[(preferred["line_num"] - standard_line).abs() < 1e-9].copy()
        if line_candidates.empty:
            raise ValueError(f"no major sportsbook offers the standard {target} line")

    ranked: list[tuple[tuple[int, int, int, int, float], float]] = []
    for line, line_rows in line_candidates.groupby("line_num"):
        line_value = float(line)
        unique_books = int(line_rows["bookmaker_key"].nunique())
        side_offers = int(line_rows[["over_price_num", "under_price_num"]].notna().sum().sum())
        all_line_rows = priced.loc[priced["line_num"] == line_value]
        standard_match = int(standard_line is not None and abs(line_value - standard_line) < 1e-9)
        standard_distance = abs(line_value - standard_line) if standard_line is not None else 0.0
        ranked.append(((unique_books, standard_match, side_offers, len(all_line_rows), -standard_distance), line_value))

    _, selected_line = max(ranked)
    return selected_line, priced.loc[priced["line_num"] == selected_line].copy()


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


def market_player_key(player_name: object) -> str:
    return str(player_name or "").strip().replace(" ", "_")


def resolve_scheduled_player_team(
    market_row: pd.Series,
    latest_row: pd.Series | None,
    *,
    home_team: str,
    home_team_id: str,
    away_team: str,
    away_team_id: str,
) -> str:
    market_team = str(market_row.get("Market_Player_Team", "") or "").strip().upper()
    if market_team in {home_team, away_team}:
        return market_team
    if latest_row is None:
        return ""
    latest_team_id = to_int_string(latest_row.get("Team_ID"))
    if latest_team_id == home_team_id:
        return home_team
    if latest_team_id == away_team_id:
        return away_team
    latest_team = str(latest_row.get("Team", "") or "").strip().upper()
    return latest_team if latest_team in {home_team, away_team} else ""


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
    wide_candidates = [
        market_root / "latest_player_props_wide.parquet",
        market_root / "latest_player_props_wide.csv",
        market_root / "history_player_props_wide.parquet",
        market_root / "history_player_props_wide.csv",
    ]
    wide_df = pd.DataFrame()
    for wide_selected in wide_candidates:
        if not wide_selected.exists():
            continue
        candidate = (
            pd.read_parquet(wide_selected)
            if wide_selected.suffix.lower() == ".parquet"
            else pd.read_csv(wide_selected)
        )
        if candidate.empty or "Market_Date" not in candidate.columns:
            continue
        candidate = candidate.copy()
        candidate["Player"] = candidate["Player"].astype(str)
        candidate["Market_Date"] = pd.to_datetime(candidate["Market_Date"], errors="coerce").dt.normalize()
        candidate = candidate.loc[candidate["Market_Date"] == requested_run_date].copy()
        if not candidate.empty:
            wide_df = candidate.drop_duplicates(subset=["Market_Date", "Player"], keep="last").reset_index(drop=True)
            break

    long_candidates = [
        market_root / "latest_player_props_long.parquet",
        market_root / "latest_player_props_long.csv",
        market_root / "history_player_props_long.parquet",
        market_root / "history_player_props_long.csv",
    ]
    long_df = pd.DataFrame()
    for long_selected in long_candidates:
        if not long_selected.exists():
            continue
        candidate = (
            pd.read_parquet(long_selected)
            if long_selected.suffix.lower() == ".parquet"
            else pd.read_csv(long_selected)
        )
        if candidate.empty or "player_name_norm" not in candidate.columns or "event_date_et" not in candidate.columns:
            continue
        candidate = candidate.copy()
        candidate["Market_Date"] = pd.to_datetime(candidate["event_date_et"], errors="coerce").dt.normalize()
        candidate = candidate.loc[candidate["Market_Date"] == requested_run_date].copy()
        if not candidate.empty:
            long_df = candidate
            break

    if long_df.empty:
        return wide_df
    per_offer_closing = (
        "snapshot_mode" in long_df.columns
        and long_df["snapshot_mode"].astype(str).eq("per_offer_closing").all()
    )
    if "fetched_at_utc" in long_df.columns and not per_offer_closing:
        fetched = pd.to_datetime(long_df["fetched_at_utc"], errors="coerce", utc=True)
        if fetched.notna().any():
            long_df = long_df.loc[fetched == fetched.max()].copy()

    long_df["Player"] = long_df["player_name_norm"].astype(str)
    long_df["line_num"] = pd.to_numeric(long_df.get("line"), errors="coerce")
    long_df["over_price_num"] = pd.to_numeric(long_df.get("over_price"), errors="coerce")
    long_df["under_price_num"] = pd.to_numeric(long_df.get("under_price"), errors="coerce")

    supplement_rows: list[dict[str, object]] = []
    for (market_date, player), part in long_df.groupby(["Market_Date", "Player"], dropna=False):
        row: dict[str, object] = {"Market_Date": market_date, "Player": player}
        for target, market_key in TARGET_MARKET_KEYS.items():
            market_part = part.loc[part["market_key"].astype(str) == market_key].copy()
            if market_part.empty:
                continue
            try:
                consensus_line, consensus_rows = select_bettable_market_line(market_part, target)
            except ValueError:
                continue
            common_rows = consensus_rows.loc[
                consensus_rows["bookmaker_key"].astype(str).str.lower().isin(PREFERRED_BOOKMAKER_KEYS)
            ].copy()
            over_price, over_book_key, over_book = _best_book_price(common_rows, "over_price_num")
            under_price, under_book_key, under_book = _best_book_price(common_rows, "under_price_num")
            exact_book_keys = sorted(
                {
                    str(value).strip().lower()
                    for value in consensus_rows["bookmaker_key"].dropna()
                    if str(value).strip()
                }
            )
            common_book_keys = [key for key in PREFERRED_BOOKMAKER_KEYS if key in set(exact_book_keys)]
            row[f"Market_{target}"] = consensus_line
            row[f"Market_{target}_books"] = len(exact_book_keys)
            row[f"Market_{target}_book_keys"] = "|".join(exact_book_keys)
            row[f"Market_{target}_common_books"] = len(common_book_keys)
            row[f"Market_{target}_common_book_keys"] = "|".join(common_book_keys)
            row[f"Market_{target}_over_price"] = over_price if over_price is not None else float("nan")
            row[f"Market_{target}_under_price"] = under_price if under_price is not None else float("nan")
            row[f"Market_{target}_over_book_key"] = over_book_key
            row[f"Market_{target}_over_book"] = over_book
            row[f"Market_{target}_under_book_key"] = under_book_key
            row[f"Market_{target}_under_book"] = under_book
            line_std = pd.to_numeric(market_part["line_num"], errors="coerce").std(ddof=0)
            row[f"Market_{target}_line_std"] = float(line_std) if pd.notna(line_std) else 0.0
            row[f"Market_Source_{target}"] = "real"
        supplement_rows.append(row)

    supplement_df = pd.DataFrame(supplement_rows)
    if supplement_df.empty:
        return wide_df

    if wide_df.empty:
        return supplement_df

    merged = wide_df.merge(supplement_df, on=["Market_Date", "Player"], how="outer", suffixes=("", "__supp"))
    for target in TARGET_MARKET_KEYS:
        for column in [
            f"Market_{target}",
            f"Market_{target}_books",
            f"Market_{target}_book_keys",
            f"Market_{target}_common_books",
            f"Market_{target}_common_book_keys",
            f"Market_{target}_over_price",
            f"Market_{target}_under_price",
            f"Market_{target}_over_book_key",
            f"Market_{target}_over_book",
            f"Market_{target}_under_book_key",
            f"Market_{target}_under_book",
            f"Market_{target}_line_std",
            f"Market_Source_{target}",
        ]:
            supp_column = f"{column}__supp"
            if supp_column not in merged.columns:
                continue
            if column not in merged.columns:
                merged[column] = merged[supp_column]
            else:
                merged[column] = merged[supp_column].where(merged[supp_column].notna(), merged[column])
            merged = merged.drop(columns=[supp_column])
    return merged.reset_index(drop=True)


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
        player_key = normalize_player_id(str(latest.get("Player", "")))
        if player_key:
            latest_player_rows[player_key] = latest

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
                player = normalize_player_id(str(row.get("Player", "")))
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
                player = normalize_player_id(str(row.get("Player", "")))
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
    player_context: dict[str, float] | None = None,
) -> tuple[float, float]:
    player_context = player_context or {}
    baseline = to_float(latest_row.get(spec.rolling_col))
    if baseline is None:
        baseline = to_float(latest_row.get(spec.lag1_col))
    if baseline is None:
        baseline = to_float(latest_row.get(spec.actual_col), 0.0)

    baseline = max(0.0, float(baseline))
    long_run_mean = to_float(player_context.get("target_mean"))
    recent_mean = to_float(player_context.get("target_recent_mean"))
    if long_run_mean is not None:
        baseline = (0.70 * max(0.0, long_run_mean)) + (0.30 * max(0.0, recent_mean if recent_mean is not None else long_run_mean))
    latest_pa_share = to_float(player_context.get("pa_share_mean"))
    if latest_pa_share is None:
        latest_pa_share = to_float(latest_row.get("Team_PA_share")) if to_float(latest_row.get("Team_PA_share")) is not None else 0.1
    park_factor = to_float(latest_row.get("Park_Factor")) if to_float(latest_row.get("Park_Factor")) is not None else 1.0
    temp_f = to_float(latest_row.get("Temp_F")) if to_float(latest_row.get("Temp_F")) is not None else 70.0
    woba = to_float(player_context.get("woba_mean"))
    if woba is None:
        woba = to_float(latest_row.get("wOBA")) if to_float(latest_row.get("wOBA")) is not None else 0.315
    iso = to_float(player_context.get("iso_mean"))
    if iso is None:
        iso = to_float(latest_row.get("ISO")) if to_float(latest_row.get("ISO")) is not None else 0.14
    barrel_pct = to_float(player_context.get("barrel_mean"))
    if barrel_pct is None:
        barrel_pct = to_float(latest_row.get("Barrel%")) if to_float(latest_row.get("Barrel%")) is not None else 7.0
    opp_pitcher_k9 = float(opponent_context.get("opp_pitcher_k9", to_float(latest_row.get("Opp_Pitcher_K9_3")) if to_float(latest_row.get("Opp_Pitcher_K9_3")) is not None else 8.2))
    opp_pitcher_era = float(opponent_context.get("opp_pitcher_era", to_float(latest_row.get("Opp_Pitcher_ERA_3")) if to_float(latest_row.get("Opp_Pitcher_ERA_3")) is not None else 4.1))
    opp_bullpen_era = float(opponent_context.get("opp_bullpen_era", to_float(latest_row.get("Opp_Bullpen_ERA_7")) if to_float(latest_row.get("Opp_Bullpen_ERA_7")) is not None else 4.0))
    opp_lineup_woba = float(opponent_context.get("opp_lineup_woba", to_float(latest_row.get("Opp_Lineup_wOBA_3")) if to_float(latest_row.get("Opp_Lineup_wOBA_3")) is not None else 0.315))
    opp_lineup_k_rate = float(opponent_context.get("opp_lineup_k_rate", to_float(latest_row.get("Opp_Lineup_K_rate_3")) if to_float(latest_row.get("Opp_Lineup_K_rate_3")) is not None else 0.225))
    lag_value = recent_mean if recent_mean is not None else baseline

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
            batting_order = to_float(player_context.get("batting_order_median"))
            if batting_order is None:
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
            legacy_prediction = (
                (0.72 * baseline)
                + (0.14 * lag_value)
                + (8.0 * (opp_lineup_k_rate - 0.20))
                + (0.10 * (park_factor - 1.0) * -4.0)
            )
            workload_prediction = to_float(player_context.get("workload_k_projection"))
            prediction = (
                (0.65 * workload_prediction) + (0.35 * legacy_prediction)
                if workload_prediction is not None
                else legacy_prediction
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

    if spec.role == "hitter":
        prediction += float(to_float(player_context.get("matchup_network_adjustment")) or 0.0)
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


def build_player_projection_context(history_frame: pd.DataFrame, spec: TargetSpec) -> dict[str, float]:
    recent = history_frame.tail(30)

    def mean_value(frame: pd.DataFrame, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    context: dict[str, float] = {}
    for key, frame, column in [
        ("target_mean", history_frame, spec.actual_col),
        ("target_recent_mean", recent, spec.actual_col),
        ("pa_share_mean", recent, "Team_PA_share"),
        ("woba_mean", recent, "wOBA"),
        ("iso_mean", recent, "ISO"),
        ("barrel_mean", recent, "Barrel%"),
    ]:
        value = mean_value(frame, column)
        if value is not None:
            context[key] = value
    if "Batting_Order" in recent.columns:
        batting_orders = pd.to_numeric(recent["Batting_Order"], errors="coerce").dropna()
        if not batting_orders.empty:
            context["batting_order_median"] = float(batting_orders.median())
    if spec.role == "pitcher":
        pitcher_history = history_frame.copy()
        innings = pd.to_numeric(pitcher_history.get("IP"), errors="coerce")
        pitches = pd.to_numeric(pitcher_history.get("Pitches"), errors="coerce")
        if "Was_Starter" in pitcher_history.columns:
            starter_flag = pd.to_numeric(pitcher_history["Was_Starter"], errors="coerce").fillna(0.0)
            starter_like = pitcher_history.loc[starter_flag.gt(0.0)].copy()
        else:
            starter_like = pitcher_history.loc[
                innings.ge(STARTER_LIKE_MIN_IP) | pitches.ge(STARTER_LIKE_MIN_PITCHES)
            ].copy()
        recent_starts = starter_like.tail(PITCHER_WORKLOAD_RECENT_STARTS)
        if not recent_starts.empty:
            recent_three = recent_starts.tail(3)
            recent_ip = pd.to_numeric(recent_starts.get("IP"), errors="coerce").dropna()
            recent_three_ip = pd.to_numeric(recent_three.get("IP"), errors="coerce").dropna()
            recent_pitches = pd.to_numeric(recent_starts.get("Pitches"), errors="coerce").dropna()
            recent_three_pitches = pd.to_numeric(recent_three.get("Pitches"), errors="coerce").dropna()
            recent_k = pd.to_numeric(recent_starts.get("K"), errors="coerce")
            projected_ip = (
                (0.60 * float(recent_three_ip.mean())) + (0.40 * float(recent_ip.mean()))
                if not recent_three_ip.empty and not recent_ip.empty
                else None
            )
            projected_pitches = (
                (0.60 * float(recent_three_pitches.mean())) + (0.40 * float(recent_pitches.mean()))
                if not recent_three_pitches.empty and not recent_pitches.empty
                else None
            )
            innings_sum = float(recent_ip.sum()) if not recent_ip.empty else 0.0
            strikeouts_sum = float(recent_k.fillna(0.0).sum())
            if projected_ip is not None:
                context["projected_ip"] = max(0.0, projected_ip)
            if projected_pitches is not None:
                context["projected_pitches"] = max(0.0, projected_pitches)
            if projected_ip is not None and innings_sum > 0.0:
                context["workload_k_projection"] = max(0.0, (strikeouts_sum / innings_sum) * projected_ip)
        context["starter_history_rows"] = float(len(starter_like))
    return context


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
        normalize_player_id(str(row.get("Player", ""))): row
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
        player_name = normalize_player_id(str(frame.iloc[0].get("Player", "")))
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
        home_team_id = to_int_string(home_meta.get("id"))
        away_team_id = to_int_string(away_meta.get("id"))
        probable_home = normalize_player_id((((game.get("teams") or {}).get("home") or {}).get("probablePitcher") or {}).get("fullName", ""))
        probable_away = normalize_player_id((((game.get("teams") or {}).get("away") or {}).get("probablePitcher") or {}).get("fullName", ""))

        for team, opponent, is_home, probable_pitcher_name, opp_probable_name in [
            (home_team, away_team, 1, probable_home, probable_away),
            (away_team, home_team, 0, probable_away, probable_home),
        ]:
            market_team_players = [
                player_name
                for player_name, row in market_by_player.items()
                if resolve_scheduled_player_team(
                    row,
                    latest_player_rows.get(player_name),
                    home_team=home_team,
                    home_team_id=home_team_id,
                    away_team=away_team,
                    away_team_id=away_team_id,
                ) == team
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
            opposing_pitcher_frame = frame_by_player.get(opp_probable_name)
            opposing_pitcher_id = 0
            opposing_pitcher_history = pd.DataFrame()
            if opposing_pitcher_frame is not None and not opposing_pitcher_frame.empty:
                opposing_pitcher_history = opposing_pitcher_frame.loc[
                    opposing_pitcher_frame["_game_date"] < requested_run_date
                ].copy()
                if not opposing_pitcher_history.empty:
                    opposing_pitcher_id = int(
                        to_float(opposing_pitcher_history.iloc[-1].get("Player_MLBAM_ID")) or 0.0
                    )

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
                matchup_signal = MatchupNetworkSignal.neutral()
                if player_type == "hitter" and opp_probable_name:
                    matchup_signal = build_matchup_network_signal(
                        history_frame,
                        opposing_pitcher_history,
                        opposing_pitcher_id=opposing_pitcher_id,
                        opposing_pitcher_name=opp_probable_name,
                    )
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

                    player_context = build_player_projection_context(history_frame, spec)
                    if player_type == "hitter":
                        player_context["matchup_network_adjustment"] = matchup_signal.adjustment[spec.target]
                    prediction, fallback_market_line = project_from_latest_row(
                        latest_row,
                        spec,
                        opponent_context=opponent_context,
                        player_context=player_context,
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
                            "Opposing_Pitcher_ID": opposing_pitcher_id if player_type == "hitter" else 0,
                            "Opposing_Pitcher": opp_probable_name.replace("_", " ") if player_type == "hitter" else "",
                            "Matchup_Network_Version": matchup_signal.version if player_type == "hitter" else "",
                            "Batter_Profile_Strength": matchup_signal.batter_strength.get(spec.target, 0.0),
                            "Pitcher_Profile_Vulnerability": matchup_signal.pitcher_vulnerability.get(spec.target, 0.0),
                            "Pitcher_Profile_Uncertainty": matchup_signal.pitcher_uncertainty if player_type == "hitter" else 0.0,
                            "Batter_Vs_Starter_Games": matchup_signal.direct_matchup_games if player_type == "hitter" else 0,
                            "Batter_Vs_Starter_Lift": matchup_signal.direct_matchup_lift.get(spec.target, 0.0),
                            "Matchup_Network_Score": matchup_signal.network_score.get(spec.target, 0.0),
                            "Matchup_Network_Confidence": matchup_signal.confidence if player_type == "hitter" else 0.0,
                            "Matchup_Network_Adjustment": matchup_signal.adjustment.get(spec.target, 0.0),
                            "Starter_Confirmed": int(
                                player_type == "pitcher" and player_name == probable_pitcher_name
                            ),
                            "Starter_History_Rows": int(player_context.get("starter_history_rows", 0.0)),
                            "Projected_IP": player_context.get("projected_ip"),
                            "Projected_Pitches": player_context.get("projected_pitches"),
                            "Team": team,
                            "Team_ID": home_team_id if is_home else away_team_id,
                            "Opponent": opponent,
                            "Opponent_ID": away_team_id if is_home else home_team_id,
                            "Is_Home": str(int(is_home)),
                            "Target": spec.target,
                            "Prediction": float(prediction),
                            "Baseline": float(baseline),
                            "Market_Line": float(market_line),
                            "Market_Source": market_source,
                            "Market_Books": int(to_float(market_row.get(f"Market_{spec.target}_books")) or 0.0) if market_row is not None else 0,
                            "Market_Book_Keys": str(market_row.get(f"Market_{spec.target}_book_keys", "") or "") if market_row is not None else "",
                            "Market_Common_Books": int(to_float(market_row.get(f"Market_{spec.target}_common_books")) or 0.0) if market_row is not None else 0,
                            "Market_Common_Book_Keys": str(market_row.get(f"Market_{spec.target}_common_book_keys", "") or "") if market_row is not None else "",
                            "Market_Line_Std": float(to_float(market_row.get(f"Market_{spec.target}_line_std")) or 0.0) if market_row is not None else 0.0,
                            "Market_Over_Price": to_float(market_row.get(f"Market_{spec.target}_over_price")) if market_row is not None else None,
                            "Market_Under_Price": to_float(market_row.get(f"Market_{spec.target}_under_price")) if market_row is not None else None,
                            "Market_Over_Book_Key": str(market_row.get(f"Market_{spec.target}_over_book_key", "") or "") if market_row is not None else "",
                            "Market_Over_Book": str(market_row.get(f"Market_{spec.target}_over_book", "") or "") if market_row is not None else "",
                            "Market_Under_Book_Key": str(market_row.get(f"Market_{spec.target}_under_book_key", "") or "") if market_row is not None else "",
                            "Market_Under_Book": str(market_row.get(f"Market_{spec.target}_under_book", "") or "") if market_row is not None else "",
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
    market_snapshot: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    status_code, status_detail = infer_status(selected_game_date, requested_run_date)
    market_by_player_id: dict[str, pd.Series] = {}
    if market_snapshot is not None and not market_snapshot.empty:
        for _, market_row in market_snapshot.iterrows():
            key = normalize_player_id(str(market_row.get("Player", "") or "").replace("_", " "))
            if key:
                market_by_player_id[key] = market_row

    for frame in frames:
        if frame.empty:
            continue

        current_rows = frame.loc[frame["_game_date"] == selected_game_date].copy()
        if current_rows.empty:
            continue

        player_name = str(current_rows.iloc[0].get("Player", "")).strip().replace("_", " ")
        player_id = normalize_player_id(player_name)
        player_type = str(current_rows.iloc[0].get("Player_Type", "")).strip().lower()
        market_row = market_by_player_id.get(player_id)
        specs = market_specs_for_role(player_type)
        if not specs:
            continue

        for _, current in current_rows.iterrows():
            history_frame = frame.loc[frame["_game_date"] < selected_game_date].copy()
            last_history_date = history_frame["_game_date"].max() if not history_frame.empty else pd.NaT

            for spec in specs:
                player_context = build_player_projection_context(history_frame, spec)
                processed_market_line = to_float(current.get(spec.market_col))
                if processed_market_line is None:
                    continue
                market_line = processed_market_line
                market_source = str(current.get(spec.market_source_col, "synthetic") or "synthetic")
                if market_row is not None:
                    selected_market_line = to_float(market_row.get(f"Market_{spec.target}"))
                    if selected_market_line is not None:
                        market_line = selected_market_line
                        market_source = str(market_row.get(f"Market_Source_{spec.target}", "real") or "real")

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
                prediction = float(processed_market_line + gap) if is_modeled else float(baseline)
                if spec.target == "K" and is_modeled:
                    workload_prediction = to_float(player_context.get("workload_k_projection"))
                    if workload_prediction is not None:
                        prediction = (0.65 * workload_prediction) + (0.35 * prediction)
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
                        "Opposing_Pitcher_ID": int(to_float(current.get("Opp_Starter_ID")) or 0.0),
                        "Opposing_Pitcher": str(current.get("Opp_Starter_Player", "") or "").replace("_", " "),
                        "Matchup_Network_Version": str(current.get("Matchup_Network_Version", "") or ""),
                        "Batter_Profile_Strength": to_float(current.get(f"Batter_Profile_{spec.target}_Strength")) or 0.0,
                        "Pitcher_Profile_Vulnerability": to_float(current.get(f"Pitcher_Profile_{spec.target}_Vulnerability")) or 0.0,
                        "Pitcher_Profile_Uncertainty": to_float(current.get("Pitcher_Profile_Uncertainty")) or 0.0,
                        "Batter_Vs_Starter_Games": int(to_float(current.get("Batter_Vs_Starter_Games")) or 0.0),
                        "Batter_Vs_Starter_Lift": to_float(current.get(f"Batter_Vs_Starter_{spec.target}_Lift")) or 0.0,
                        "Matchup_Network_Score": to_float(current.get(f"Matchup_Network_{spec.target}_Score")) or 0.0,
                        "Matchup_Network_Confidence": to_float(current.get("Matchup_Network_Confidence")) or 0.0,
                        "Matchup_Network_Adjustment": to_float(current.get(f"Matchup_Network_{spec.target}_Adjustment")) or 0.0,
                        "Starter_Confirmed": 0,
                        "Starter_History_Rows": int(player_context.get("starter_history_rows", 0.0)),
                        "Projected_IP": player_context.get("projected_ip"),
                        "Projected_Pitches": player_context.get("projected_pitches"),
                        "Team": str(current.get("Team", "") or ""),
                        "Team_ID": to_int_string(current.get("Team_ID")),
                        "Opponent": str(current.get("Opponent", "") or ""),
                        "Opponent_ID": to_int_string(current.get("Opponent_ID")),
                        "Is_Home": to_int_string(current.get("Is_Home")),
                        "Target": spec.target,
                        "Prediction": prediction,
                        "Baseline": float(baseline),
                        "Market_Line": float(market_line),
                        "Market_Source": market_source,
                        "Market_Books": int(to_float(market_row.get(f"Market_{spec.target}_books")) or 0.0) if market_row is not None else 0,
                        "Market_Book_Keys": str(market_row.get(f"Market_{spec.target}_book_keys", "") or "") if market_row is not None else "",
                        "Market_Common_Books": int(to_float(market_row.get(f"Market_{spec.target}_common_books")) or 0.0) if market_row is not None else 0,
                        "Market_Common_Book_Keys": str(market_row.get(f"Market_{spec.target}_common_book_keys", "") or "") if market_row is not None else "",
                        "Market_Line_Std": float(to_float(market_row.get(f"Market_{spec.target}_line_std")) or 0.0) if market_row is not None else 0.0,
                        "Market_Over_Price": to_float(market_row.get(f"Market_{spec.target}_over_price")) if market_row is not None else None,
                        "Market_Under_Price": to_float(market_row.get(f"Market_{spec.target}_under_price")) if market_row is not None else None,
                        "Market_Over_Book_Key": str(market_row.get(f"Market_{spec.target}_over_book_key", "") or "") if market_row is not None else "",
                        "Market_Over_Book": str(market_row.get(f"Market_{spec.target}_over_book", "") or "") if market_row is not None else "",
                        "Market_Under_Book_Key": str(market_row.get(f"Market_{spec.target}_under_book_key", "") or "") if market_row is not None else "",
                        "Market_Under_Book": str(market_row.get(f"Market_{spec.target}_under_book", "") or "") if market_row is not None else "",
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
        "Opposing_Pitcher_ID",
        "Opposing_Pitcher",
        "Matchup_Network_Version",
        "Batter_Profile_Strength",
        "Pitcher_Profile_Vulnerability",
        "Pitcher_Profile_Uncertainty",
        "Batter_Vs_Starter_Games",
        "Batter_Vs_Starter_Lift",
        "Matchup_Network_Score",
        "Matchup_Network_Confidence",
        "Matchup_Network_Adjustment",
        "Starter_Confirmed",
        "Starter_History_Rows",
        "Projected_IP",
        "Projected_Pitches",
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
        "Market_Books",
        "Market_Book_Keys",
        "Market_Common_Books",
        "Market_Common_Book_Keys",
        "Market_Line_Std",
        "Market_Over_Price",
        "Market_Under_Price",
        "Market_Over_Book_Key",
        "Market_Over_Book",
        "Market_Under_Book_Key",
        "Market_Under_Book",
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
    network_rows = [
        row for row in rows if str(row.get("Matchup_Network_Version", "")).strip()
    ]

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
        "matchup_network": {
            "version": str(network_rows[0].get("Matchup_Network_Version", "")) if network_rows else "",
            "rows": int(len(network_rows)),
            "linked_pitcher_rows": int(
                sum(1 for row in network_rows if str(row.get("Opposing_Pitcher", "")).strip())
            ),
            "direct_history_rows": int(
                sum(1 for row in network_rows if int(to_float(row.get("Batter_Vs_Starter_Games")) or 0.0) > 0)
            ),
            "avg_abs_adjustment": (
                float(
                    sum(abs(float(to_float(row.get("Matchup_Network_Adjustment")) or 0.0)) for row in network_rows)
                    / len(network_rows)
                )
                if network_rows
                else 0.0
            ),
        },
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
        market_snapshot = load_market_snapshot(args.market_root.resolve(), selected_game_date)

        rows = build_pool_rows(
            frames=frames,
            selected_game_date=selected_game_date,
            requested_run_date=requested_run_date,
            min_modeled_history_rows=int(args.min_modeled_history_rows),
            market_snapshot=market_snapshot,
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
