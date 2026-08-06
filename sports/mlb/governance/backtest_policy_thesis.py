#!/usr/bin/env python3
"""Retrospectively audit frozen MLB policy ideas on complete captured market dates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.scripts.backtest_prediction_method import wilson_interval
from sports.mlb.scripts.optimize_walk_forward_policy import load_universe
from sports.mlb.scripts.select_high_precision_predictions import estimate_count_hit_probabilities


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = REPO_ROOT / "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv"
DEFAULT_LINES = REPO_ROOT / "sports/mlb/data/raw/market_odds/mlb/odds_api_io/history_player_props_long.csv"
DEFAULT_OUTPUT = REPO_ROOT / "sports/mlb/data/predictions/backtests/policy_thesis"
DEFAULT_DAILY_RUNS = REPO_ROOT / "sports/mlb/data/predictions/daily_runs"
SUPPORTED_BOOKS = {"betmgm", "betrivers", "caesars", "draftkings", "fanduel"}
COMMON_BOOKS = {"caesars", "draftkings", "fanduel"}
BOOK_ALIASES = {"mgm": "betmgm"}
MARKET_TARGETS = {
    "batter_hits": "H",
    "batter_total_bases": "TB",
    "batter_runs_scored": "R",
    "batter_rbis": "RBI",
    "pitcher_strikeouts": "K",
}
EVIDENCE_LABEL = "RETROSPECTIVE_FULL_CANDIDATE_RECONSTRUCTION"
POLICY_VERSION = "MLB_OVER_POLICY_THESIS_AUDIT_V1"
PARLAY_VERSION = "MLB_TWO_LEG_OVER_PARLAY_THESIS_AUDIT_V1"
SELECTION_EXPORT_COLUMNS = [
    "date",
    "policy_version",
    "snapshot_id",
    "player",
    "player_id",
    "team",
    "opponent",
    "game_id",
    "provider_event_id",
    "target",
    "direction",
    "line",
    "prediction",
    "actual",
    "model_hit_probability",
    "probability",
    "push_probability",
    "abs_edge",
    "history_rows",
    "days_since_history",
    "supported_book_count",
    "common_book_count",
    "best_book",
    "best_american_price",
    "best_decimal_price",
    "best_price_observed_at_utc",
    "expected_return_model",
    "result",
    "unit_return",
    "policy_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--lines-csv", type=Path, default=DEFAULT_LINES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS)
    parser.add_argument("--development-dates", type=int, default=4)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matching_files_hash(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_player(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    return "_".join(re.findall(r"[a-z0-9]+", text))


def american_to_decimal(price: object) -> float | None:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or abs(value) < 100.0:
        return None
    return 1.0 + (value / 100.0 if value > 0.0 else 100.0 / abs(value))


def bounded_lcb(values: list[float], *, lower: float, upper: float, alpha: float) -> float | None:
    if not values:
        return None
    if any(value < lower - 1e-12 or value > upper + 1e-12 for value in values):
        raise ValueError("Observed return exceeds its predeclared bound.")
    radius = (upper - lower) * math.sqrt(math.log(1.0 / alpha) / (2.0 * len(values)))
    return max(lower, (sum(values) / len(values)) - radius)


def load_historical_schedule(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    source_files = 0
    for path in sorted(root.glob("*/daily_prediction_pool_*.csv")):
        header = pd.read_csv(path, nrows=0).columns
        required = {"Game_Date", "Commence_Time_UTC", "Game_ID", "Team", "Opponent", "Is_Home"}
        if not required.issubset(header):
            continue
        frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
        frame = frame.loc[frame["Commence_Time_UTC"].astype(str).str.strip().ne("")].copy()
        if frame.empty:
            continue
        is_home = frame["Is_Home"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        frame["date"] = pd.to_datetime(frame["Game_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["home_team"] = frame["Team"].where(is_home, frame["Opponent"]).astype(str).str.upper()
        frame["away_team"] = frame["Opponent"].where(is_home, frame["Team"]).astype(str).str.upper()
        frame["schedule_commence_time_utc"] = pd.to_datetime(frame["Commence_Time_UTC"], utc=True, errors="coerce")
        rows.append(frame[["date", "home_team", "away_team", "Game_ID", "schedule_commence_time_utc"]].drop_duplicates())
        source_files += 1
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "schedule_commence_time_utc"]), {
            "schedule_source_files": 0,
            "schedule_events": 0,
            "ambiguous_schedule_matchups_rejected": 0,
        }
    schedule = pd.concat(rows, ignore_index=True).dropna(subset=["date", "schedule_commence_time_utc"])
    key = ["date", "home_team", "away_team"]
    counts = schedule.groupby(key)["Game_ID"].nunique().rename("schedule_event_count")
    schedule = schedule.merge(counts, on=key, how="left")
    ambiguous = int(schedule.loc[schedule["schedule_event_count"] > 1, key].drop_duplicates().shape[0])
    schedule = schedule.loc[schedule["schedule_event_count"] == 1, key + ["schedule_commence_time_utc"]].drop_duplicates(key)
    return schedule, {
        "schedule_source_files": source_files,
        "schedule_events": int(len(schedule)),
        "ambiguous_schedule_matchups_rejected": ambiguous,
    }


def load_quotes(path: Path, schedule: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lines = pd.read_csv(path, low_memory=False)
    lines["fetched_at_utc"] = pd.to_datetime(lines["fetched_at_utc"], utc=True, errors="coerce")
    lines["commence_time_utc"] = pd.to_datetime(lines["commence_time_utc"], utc=True, errors="coerce")
    lines["date"] = pd.to_datetime(lines["event_date_et"], errors="coerce").dt.strftime("%Y-%m-%d")
    lines["player_id"] = lines["player_name_norm"].map(normalize_player)
    lines["target"] = lines["market_key"].map(MARKET_TARGETS)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce").round(4)
    lines["over_price"] = pd.to_numeric(lines["over_price"], errors="coerce")
    lines["book"] = lines["bookmaker_key"].astype(str).str.lower().replace(BOOK_ALIASES)
    for column in ("home_team", "away_team"):
        if column not in lines:
            lines[column] = ""
        lines[column] = lines[column].astype(str).str.upper()
    if schedule is not None and not schedule.empty:
        lines = lines.merge(schedule, on=["date", "home_team", "away_team"], how="left")
        lines["commence_time_utc"] = lines["commence_time_utc"].combine_first(lines["schedule_commence_time_utc"])
    rows_before = int(len(lines))
    valid = lines.loc[
        lines["fetched_at_utc"].notna()
        & lines["commence_time_utc"].notna()
        & (lines["fetched_at_utc"] < lines["commence_time_utc"])
        & lines["target"].notna()
        & lines["line"].notna()
        & lines["over_price"].map(lambda value: american_to_decimal(value) is not None)
        & lines["book"].isin(SUPPORTED_BOOKS)
    ].copy()
    selected_snapshots = valid.groupby("date")["fetched_at_utc"].max().rename("selected_snapshot_utc")
    lines = valid.merge(selected_snapshots, on="date", how="inner")
    lines = lines.loc[lines["fetched_at_utc"] == lines["selected_snapshot_utc"]].copy()
    lines["snapshot_id"] = lines["date"] + "|" + lines["selected_snapshot_utc"].map(lambda value: value.isoformat())
    quote_key = ["date", "event_id", "player_id", "target", "line", "book"]
    lines = lines.sort_values("fetched_at_utc").drop_duplicates(quote_key, keep="last")
    base_key = ["date", "player_id", "target", "line"]
    event_counts = lines.groupby(base_key, dropna=False)["event_id"].nunique().rename("event_count")
    lines = lines.merge(event_counts, on=base_key, how="left")
    ambiguous_rows = int((lines["event_count"] > 1).sum())
    lines = lines.loc[lines["event_count"] == 1].copy()
    lines["decimal_price"] = lines["over_price"].map(american_to_decimal)
    summaries: list[dict[str, Any]] = []
    for key, part in lines.groupby(base_key, sort=True):
        best = part.sort_values(["over_price", "book"], ascending=[False, True]).iloc[0]
        books = set(part["book"].astype(str))
        summaries.append(
            {
                "date": key[0],
                "player_id": key[1],
                "target": key[2],
                "line": key[3],
                "snapshot_id": str(best["snapshot_id"]),
                "provider_event_id": str(best["event_id"]),
                "supported_book_count": len(books),
                "common_book_count": len(books & COMMON_BOOKS),
                "best_book": str(best["book"]),
                "best_american_price": float(best["over_price"]),
                "best_decimal_price": float(best["decimal_price"]),
                "best_price_observed_at_utc": best["fetched_at_utc"].isoformat(),
            }
        )
    summary = pd.DataFrame(summaries)
    diagnostics = {
        "raw_rows": rows_before,
        "latest_valid_supported_pregame_quotes": int(len(lines)),
        "single_acquisition_snapshots": int(lines["snapshot_id"].nunique()),
        "exact_candidate_quote_groups": int(len(summary)),
        "ambiguous_doubleheader_quote_rows_rejected": ambiguous_rows,
        "captured_dates": sorted(str(value) for value in summary["date"].unique()),
    }
    return lines, summary, diagnostics


def prepare_candidates(universe: pd.DataFrame, quote_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    model = universe.copy()
    model["date"] = pd.to_datetime(model["Game_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    model["player_id"] = model["Player_ID"].where(model["Player_ID"].notna(), model["Player"]).map(normalize_player)
    model["target"] = model["Target"].astype(str).str.upper()
    model["prediction"] = pd.to_numeric(model["Prediction"], errors="coerce")
    model["actual"] = pd.to_numeric(model["Actual"], errors="coerce")
    model["history_rows"] = pd.to_numeric(model["History_Rows"], errors="coerce")
    last_history = pd.to_datetime(model["Last_History_Date"], errors="coerce")
    model["days_since_history"] = (pd.to_datetime(model["date"], errors="coerce") - last_history).dt.days
    model = model.loc[
        model["prediction"].notna()
        & model["actual"].notna()
        & model["target"].isin(set(MARKET_TARGETS.values()))
        & ~model["Model_Selected"].astype(str).str.lower().eq("baseline")
    ].copy()
    key = ["date", "player_id", "target"]
    model_counts = model.groupby(key)["Game_ID"].nunique().rename("model_event_count")
    model = model.merge(model_counts, on=key, how="left")
    ambiguous_model_rows = int((model["model_event_count"] > 1).sum())
    model = model.loc[model["model_event_count"] == 1].copy()
    model = model.sort_values(["history_rows", "Game_ID"], ascending=[False, True]).drop_duplicates(key)
    model = model.rename(
        columns={
            "Player": "player",
            "Team": "team",
            "Opponent": "opponent",
            "Game_ID": "game_id",
        }
    )
    frame = quote_summary.merge(model, on=["date", "player_id", "target"], how="inner")
    reconstructed_dates = sorted(str(value) for value in frame["date"].unique())
    reconstructed_rows = int(len(frame))
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce").round(4)
    probabilities = frame.apply(
        lambda row: estimate_count_hit_probabilities(float(row["prediction"]), float(row["line"]), "OVER"),
        axis=1,
    )
    frame["model_hit_probability"] = probabilities.map(lambda value: value[0])
    frame["push_probability"] = probabilities.map(lambda value: value[1])
    frame["probability"] = probabilities.map(lambda value: value[2])
    frame["abs_edge"] = frame["prediction"] - frame["line"]
    frame = frame.loc[frame["abs_edge"] > 0.0].copy()
    frame["selection_score"] = (
        (0.65 * frame["probability"])
        + (0.20 * frame["model_hit_probability"])
        + (0.15 * (frame["abs_edge"] / frame["abs_edge"].clip(lower=0.01).max()).fillna(0.0))
    )
    frame["direction"] = "OVER"
    for column in (
        "model_hit_probability", "probability", "push_probability", "abs_edge", "history_rows",
        "days_since_history", "selection_score", "actual", "best_american_price", "best_decimal_price",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["expected_return_model"] = (
        frame["model_hit_probability"] * frame["best_decimal_price"] - 1.0
    )
    frame["expected_return_calibrated"] = frame["probability"] * frame["best_decimal_price"] - 1.0
    frame["result"] = frame.apply(
        lambda row: "win" if row["actual"] > row["line"] else "push" if row["actual"] == row["line"] else "loss",
        axis=1,
    )
    frame["unit_return"] = frame.apply(
        lambda row: row["best_decimal_price"] - 1.0 if row["result"] == "win" else 0.0 if row["result"] == "push" else -1.0,
        axis=1,
    )
    return frame, {
        "ambiguous_model_event_rows_rejected": ambiguous_model_rows,
        "model_player_market_rows": int(len(model)),
        "exact_model_quote_groups": reconstructed_rows,
        "positive_edge_candidates": int(len(frame)),
        "reconstructed_candidate_dates": reconstructed_dates,
    }


def base_playable_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.loc[
        candidates["target"].isin({"H", "R", "TB"})
        & (candidates["supported_book_count"] >= 5)
        & (candidates["common_book_count"] >= 2)
        & (candidates["history_rows"] >= 35)
        & (candidates["days_since_history"] <= 4)
        & candidates["best_decimal_price"].between(1.4, 3.25, inclusive="both")
    ].copy()


def current_profile_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    base = base_playable_pool(candidates)
    optimized = base.loc[
        base["target"].isin({"R", "TB"})
        & (base["history_rows"] >= 55)
        & base["abs_edge"].between(0.15, 0.35, inclusive="both")
        & base["model_hit_probability"].between(0.45, 0.55, inclusive="both")
        & (base["expected_return_model"] >= 0.10)
        & (base["best_american_price"] <= 125.0)
    ]
    core_hits = base.loc[
        base["target"].eq("H")
        & (base["abs_edge"] >= 0.35)
        & (base["probability"] >= 0.75)
        & (base["push_probability"] <= 0.10)
        & base["best_american_price"].between(-250.0, -200.0, inclusive="both")
        & (base["expected_return_calibrated"] >= 0.0)
    ]
    result = pd.concat([optimized, core_hits], ignore_index=False).drop_duplicates()
    result["policy_score"] = (
        (0.55 * result["probability"])
        + (0.25 * result["model_hit_probability"])
        + (0.15 * (result["abs_edge"] / 0.35).clip(upper=1.0))
        + (0.05 * result["expected_return_model"].clip(lower=-1.0, upper=1.0))
    )
    return result


def baseline_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    result = base_playable_pool(candidates)
    result = result.loc[
        (result["abs_edge"] >= 0.10)
        & (result["model_hit_probability"] >= 0.45)
        & (result["expected_return_model"] >= 0.0)
    ].copy()
    result["policy_score"] = (
        (0.55 * result["probability"])
        + (0.25 * result["model_hit_probability"])
        + (0.20 * (result["abs_edge"] / result["abs_edge"].clip(lower=0.01).max()).fillna(0.0))
    )
    return result


def select_daily(pool: pd.DataFrame, *, maximum: int = 3) -> pd.DataFrame:
    chosen: list[int] = []
    for _, part in pool.sort_values(
        ["date", "policy_score", "best_decimal_price", "player_id"],
        ascending=[True, False, False, True],
        kind="stable",
    ).groupby("date", sort=True):
        players: set[str] = set()
        games: set[str] = set()
        buckets: Counter[str] = Counter()
        for index, row in part.iterrows():
            player = str(row["player_id"])
            game = str(row["game_id"])
            bucket = f"{row['target']}|{float(row['line']):g}"
            if player in players or game in games or buckets[bucket] >= 2:
                continue
            chosen.append(index)
            players.add(player)
            games.add(game)
            buckets[bucket] += 1
            if len(players) >= maximum:
                break
    return pool.loc[chosen].sort_values(["date", "policy_score"], ascending=[True, False]).copy()


def score_policy(rows: pd.DataFrame, eligible: pd.DataFrame, dates: list[str], *, alpha: float) -> dict[str, Any]:
    relevant = rows.loc[rows["date"].isin(dates)].copy()
    eligible_part = eligible.loc[eligible["date"].isin(dates)].copy()
    wins = int(relevant["result"].eq("win").sum())
    losses = int(relevant["result"].eq("loss").sum())
    pushes = int(relevant["result"].eq("push").sum())
    low, high = wilson_interval(wins, losses)
    daily_returns = []
    action_returns = []
    for date in dates:
        day = relevant.loc[relevant["date"] == date, "unit_return"]
        value = float(day.mean()) if len(day) else 0.0
        daily_returns.append(value)
        if len(day):
            action_returns.append(value)
    return {
        "eligible_slates": len(dates),
        "action_slates": len(action_returns),
        "slate_coverage": len(action_returns) / len(dates) if dates else 0.0,
        "eligible_candidates": int(len(eligible_part)),
        "selected_candidates": int(len(relevant)),
        "candidate_coverage": len(relevant) / len(eligible_part) if len(eligible_part) else 0.0,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": wins / (wins + losses) if wins + losses else None,
        "hit_rate_wilson_95_low": low,
        "hit_rate_wilson_95_high": high,
        "selection_return": float(relevant["unit_return"].mean()) if len(relevant) else None,
        "calendar_slate_return": sum(daily_returns) / len(daily_returns) if daily_returns else None,
        "action_day_return": sum(action_returns) / len(action_returns) if action_returns else None,
        "fixed_holdout_return_lcb": bounded_lcb(daily_returns, lower=-1.0, upper=2.25, alpha=alpha),
        "losing_action_slate_rate": sum(value < 0.0 for value in action_returns) / len(action_returns) if action_returns else None,
        "net_selection_units": float(relevant["unit_return"].sum()),
    }


def parlay_anchors(candidates: pd.DataFrame) -> pd.DataFrame:
    base = base_playable_pool(candidates)
    return base.loc[
        (base["abs_edge"] >= 0.10)
        & (base["probability"] >= 0.62)
        & (base["expected_return_calibrated"] >= 0.0)
    ].copy()


def settle_two_leg_parlay(
    left_result: str,
    left_decimal_price: float,
    right_result: str,
    right_decimal_price: float,
) -> tuple[str, float]:
    results = [left_result, right_result]
    if "loss" in results:
        return "loss", -1.0
    surviving_prices = [
        price
        for result, price in (
            (left_result, left_decimal_price),
            (right_result, right_decimal_price),
        )
        if result == "win"
    ]
    if not surviving_prices:
        return "push", 0.0
    payout = math.prod(surviving_prices)
    return "win", payout - 1.0


def build_parlays(anchors: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    quote_lookup = {
        (str(row.date), str(row.player_id), str(row.target), float(row.line), str(row.book)): row
        for row in quotes.itertuples()
    }
    tickets: list[dict[str, Any]] = []
    for date, part in anchors.groupby("date", sort=True):
        best: dict[str, Any] | None = None
        ranked = part.sort_values(["probability", "selection_score"], ascending=[False, False]).head(30)
        for (_, left), (_, right) in itertools.combinations(ranked.iterrows(), 2):
            if str(left["player_id"]) == str(right["player_id"]) or str(left["game_id"]) == str(right["game_id"]):
                continue
            left_books = set(
                quotes.loc[
                    (quotes["date"] == date)
                    & (quotes["player_id"] == left["player_id"])
                    & (quotes["target"] == left["target"])
                    & (quotes["line"] == left["line"]),
                    "book",
                ]
            )
            right_books = set(
                quotes.loc[
                    (quotes["date"] == date)
                    & (quotes["player_id"] == right["player_id"])
                    & (quotes["target"] == right["target"])
                    & (quotes["line"] == right["line"]),
                    "book",
                ]
            )
            for book in sorted(left_books & right_books):
                left_quote = quote_lookup[(date, left["player_id"], left["target"], float(left["line"]), book)]
                right_quote = quote_lookup[(date, right["player_id"], right["target"], float(right["line"]), book)]
                combined = float(left_quote.decimal_price) * float(right_quote.decimal_price)
                if not 1.8 <= combined <= 6.0:
                    continue
                probability = float(left["probability"]) * float(right["probability"]) * 0.95
                expected_return = probability * combined - 1.0
                if expected_return < 0.0:
                    continue
                result, unit_return = settle_two_leg_parlay(
                    str(left["result"]),
                    float(left_quote.decimal_price),
                    str(right["result"]),
                    float(right_quote.decimal_price),
                )
                candidate = {
                    "date": date,
                    "policy_version": PARLAY_VERSION,
                    "book": book,
                    "leg_count": 2,
                    "combined_decimal_price": combined,
                    "projected_probability": probability,
                    "expected_return": expected_return,
                    "result": result,
                    "unit_return": unit_return,
                    "leg_1": f"{left['player']} {left['target']} OVER {float(left['line']):g}",
                    "leg_2": f"{right['player']} {right['target']} OVER {float(right['line']):g}",
                }
                ranking = (probability, expected_return, combined)
                if best is None or ranking > best["_ranking"]:
                    best = {**candidate, "_ranking": ranking}
        if best is not None:
            best.pop("_ranking")
            tickets.append(best)
    return pd.DataFrame(tickets)


def score_parlays(tickets: pd.DataFrame, dates: list[str], *, alpha: float) -> dict[str, Any]:
    relevant = tickets.loc[tickets["date"].isin(dates)].copy() if not tickets.empty else tickets.copy()
    wins = int(relevant["result"].eq("win").sum()) if not relevant.empty else 0
    losses = int(relevant["result"].eq("loss").sum()) if not relevant.empty else 0
    pushes = int(relevant["result"].eq("push").sum()) if not relevant.empty else 0
    low, high = wilson_interval(wins, losses)
    if relevant.empty:
        daily = [0.0 for _ in dates]
        action: list[float] = []
    else:
        action_dates = set(relevant["date"])
        daily = [
            float(relevant.loc[relevant["date"] == date, "unit_return"].iloc[0]) if date in action_dates else 0.0
            for date in dates
        ]
        action = relevant["unit_return"].astype(float).tolist()
    return {
        "eligible_slates": len(dates),
        "action_slates": int(len(relevant)),
        "slate_coverage": len(relevant) / len(dates) if dates else 0.0,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": wins / (wins + losses) if wins + losses else None,
        "hit_rate_wilson_95_low": low,
        "hit_rate_wilson_95_high": high,
        "calendar_slate_return": sum(daily) / len(daily) if daily else None,
        "action_day_return": sum(action) / len(action) if action else None,
        "fixed_holdout_return_lcb": bounded_lcb(daily, lower=-1.0, upper=5.0, alpha=alpha),
        "losing_action_slate_rate": sum(value < 0.0 for value in action) / len(action) if action else None,
        "net_units": sum(action),
    }


def markdown_report(report: dict[str, Any]) -> str:
    development = report["results"]["development"]
    holdout = report["results"]["holdout"]
    singles = holdout["current_profile_singles"]
    baseline = holdout["playable_over_baseline"]
    parlay = holdout["two_leg_parlay"]
    format_rate = lambda value: "n/a" if value is None else f"{value:.1%}"
    return "\n".join(
        [
            "# MLB Policy Thesis Retrospective Audit",
            "",
            f"Generated: {report['generated_at_utc']}",
            f"Evidence: `{report['evidence']['label']}`",
            "",
            "## Verdict",
            "",
            f"**{report['verdict']['status']}**",
            "",
            *[f"- {reason}" for reason in report["verdict"]["reasons"]],
            "",
            "## Chronological Split",
            "",
            f"- Development audit dates: {', '.join(report['splits']['development'])}",
            f"- Retrospective holdout dates: {', '.join(report['splits']['holdout'])}",
            "- The holdout was not locked before earlier policy development and cannot certify production.",
            f"- Development current-profile result: {development['current_profile_singles']['wins']}-{development['current_profile_singles']['losses']}-{development['current_profile_singles']['pushes']} across {development['current_profile_singles']['selected_candidates']} picks.",
            "",
            "## Holdout Results",
            "",
            "| Policy | Picks/tickets | W-L-P | Hit rate | Calendar return | Return LCB | Coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Current-profile singles | {singles['selected_candidates']} | {singles['wins']}-{singles['losses']}-{singles['pushes']} | {format_rate(singles['hit_rate'])} | {format_rate(singles['calendar_slate_return'])} | {format_rate(singles['fixed_holdout_return_lcb'])} | {format_rate(singles['slate_coverage'])} |",
            f"| Playable-over baseline | {baseline['selected_candidates']} | {baseline['wins']}-{baseline['losses']}-{baseline['pushes']} | {format_rate(baseline['hit_rate'])} | {format_rate(baseline['calendar_slate_return'])} | {format_rate(baseline['fixed_holdout_return_lcb'])} | {format_rate(baseline['slate_coverage'])} |",
            f"| Two-leg parlay | {parlay['action_slates']} | {parlay['wins']}-{parlay['losses']}-{parlay['pushes']} | {format_rate(parlay['hit_rate'])} | {format_rate(parlay['calendar_slate_return'])} | {format_rate(parlay['fixed_holdout_return_lcb'])} | {format_rate(parlay['slate_coverage'])} |",
            "",
            "## Interpretation",
            "",
            "- Returns use the latest captured pregame price at the exact line and book.",
            "- Every positive-edge modeled OVER candidate with an exact historical quote was retained before policy filtering.",
            "- A positive point estimate is descriptive; the bounded held-out LCB is the decision criterion.",
            "- Pitcher workload fields and RBI lines were unavailable in the captured historical scope.",
            "- This audit does not create or activate a policy certificate.",
            "",
        ]
    )


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    universe_path = args.universe_csv.resolve()
    lines_path = args.lines_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(universe_path)
    daily_runs_root = args.daily_runs_root.resolve()
    schedule, schedule_diagnostics = load_historical_schedule(daily_runs_root)
    quotes, quote_summary, quote_diagnostics = load_quotes(lines_path, schedule)
    quote_diagnostics.update(schedule_diagnostics)
    candidates, candidate_diagnostics = prepare_candidates(universe, quote_summary)
    quote_diagnostics.update(candidate_diagnostics)
    captured_dates = list(candidate_diagnostics["reconstructed_candidate_dates"])
    development_count = int(args.development_dates)
    if len(captured_dates) <= development_count:
        raise ValueError("Not enough captured dates for the requested chronological split.")
    development_dates = captured_dates[:development_count]
    holdout_dates = captured_dates[development_count:]

    current_eligible = current_profile_pool(candidates)
    baseline_eligible = baseline_pool(candidates)
    current_rows = select_daily(current_eligible)
    baseline_rows = select_daily(baseline_eligible)
    tickets = build_parlays(parlay_anchors(candidates), quotes)
    family_alpha = 0.05 / 2.0
    development = {
        "current_profile_singles": score_policy(current_rows, current_eligible, development_dates, alpha=family_alpha),
        "playable_over_baseline": score_policy(baseline_rows, baseline_eligible, development_dates, alpha=family_alpha),
        "two_leg_parlay": score_parlays(tickets, development_dates, alpha=0.05),
    }
    holdout = {
        "current_profile_singles": score_policy(current_rows, current_eligible, holdout_dates, alpha=family_alpha),
        "playable_over_baseline": score_policy(baseline_rows, baseline_eligible, holdout_dates, alpha=family_alpha),
        "two_leg_parlay": score_parlays(tickets, holdout_dates, alpha=0.05),
    }
    current_holdout = holdout["current_profile_singles"]
    baseline_holdout = holdout["playable_over_baseline"]
    parlay_holdout = holdout["two_leg_parlay"]
    reasons: list[str] = []
    if (current_holdout["calendar_slate_return"] or 0.0) <= 0.0:
        reasons.append("Current-profile singles did not produce a positive holdout calendar return.")
    if (current_holdout["fixed_holdout_return_lcb"] or -1.0) <= 0.01:
        reasons.append("The multiplicity-adjusted bounded return LCB did not clear the 1% deployment margin.")
    if current_holdout["eligible_slates"] < 160 or current_holdout["action_slates"] < 100:
        reasons.append("Captured date and action-day volume is far below certificate requirements.")
    if (parlay_holdout["fixed_holdout_return_lcb"] or -1.0) <= 0.01:
        reasons.append("The parlay return LCB did not clear the deployment margin.")
    if (current_holdout["calendar_slate_return"] or -1.0) <= (baseline_holdout["calendar_slate_return"] or -1.0):
        reasons.append("The current profile did not outperform the broader playable-OVER baseline on calendar return.")
    status = (
        "CURRENT_PROFILE_REJECTED_BROADER_THEORY_UNPROVEN"
        if (current_holdout["calendar_slate_return"] or 0.0) <= 0.0
        else "THEORY_NOT_ESTABLISHED"
        if reasons
        else "RETROSPECTIVE_SUPPORT_ONLY"
    )
    if not reasons:
        reasons.append("Point estimates and bounded holdout gates were favorable, but the evidence remains retrospective.")

    report = {
        "schema_version": "MLB_POLICY_THESIS_BACKTEST_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "label": EVIDENCE_LABEL,
            "certificate_permitted": False,
            "locked_validation": False,
            "full_candidate_rows_with_exact_quotes": int(len(candidates)),
            "universe_sha256": file_hash(universe_path),
            "lines_sha256": file_hash(lines_path),
            "daily_run_schedule_sha256": matching_files_hash(
                daily_runs_root,
                "*/daily_prediction_pool_*.csv",
            ),
            "limitations": [
                "historical sportsbook captures exist on only 11 irregular dates",
                "the retrospective holdout was not hidden before prior policy development",
                "historical pitcher workload gates and RBI quote coverage are incomplete",
                "book availability and prices are historical snapshots, not guaranteed fills",
            ],
        },
        "policy_family": {
            "singles_policy_version": POLICY_VERSION,
            "parlay_policy_version": PARLAY_VERSION,
            "bounded_family_size": 2,
            "singles_family_alpha": family_alpha,
            "selection_cap": 3,
            "direction": "OVER_ONLY",
            "parlay_leg_count": 2,
            "staking_enabled": False,
        },
        "quote_diagnostics": quote_diagnostics,
        "filter_counts": {
            "positive_edge_exact_quote_candidates": int(len(candidates)),
            "base_playable_candidates": int(len(base_playable_pool(candidates))),
            "current_profile_candidates": int(len(current_eligible)),
            "playable_over_baseline_candidates": int(len(baseline_eligible)),
            "parlay_anchor_candidates": int(len(parlay_anchors(candidates))),
        },
        "splits": {"development": development_dates, "holdout": holdout_dates},
        "results": {"development": development, "holdout": holdout},
        "hypotheses": {
            "current_profile_calendar_return_exceeds_margin": {
                "margin": 0.01,
                "lcb": current_holdout["fixed_holdout_return_lcb"],
                "status": "REJECTED",
            },
            "playable_over_baseline_calendar_return_exceeds_margin": {
                "margin": 0.01,
                "lcb": baseline_holdout["fixed_holdout_return_lcb"],
                "status": "NOT_ESTABLISHED",
            },
            "two_leg_parlay_calendar_return_exceeds_margin": {
                "margin": 0.01,
                "lcb": parlay_holdout["fixed_holdout_return_lcb"],
                "status": "NOT_ESTABLISHED",
            },
        },
        "verdict": {"status": status, "reasons": reasons},
        "authorization": {
            "candidate_authorization_enabled": False,
            "staking_enabled": False,
            "certificate_status": "NO_ACTIVE_PROSPECTIVE_CERTIFICATE",
        },
    }
    report_path = output_dir / "mlb_policy_thesis_backtest.json"
    markdown_path = output_dir / "mlb_policy_thesis_backtest.md"
    selections_path = output_dir / "mlb_policy_thesis_holdout_selections.csv"
    parlays_path = output_dir / "mlb_policy_thesis_holdout_parlays.csv"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    holdout_selections = current_rows.loc[current_rows["date"].isin(holdout_dates)].copy()
    holdout_selections["policy_version"] = POLICY_VERSION
    holdout_selections.loc[:, SELECTION_EXPORT_COLUMNS].to_csv(selections_path, index=False)
    holdout_tickets = tickets.loc[tickets["date"].isin(holdout_dates)].copy() if not tickets.empty else tickets.copy()
    holdout_tickets.to_csv(parlays_path, index=False)
    return report


def main() -> None:
    report = run_backtest(parse_args())
    print(markdown_report(report))


if __name__ == "__main__":
    main()
