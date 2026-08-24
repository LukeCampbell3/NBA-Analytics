#!/usr/bin/env python3
"""
Backtest and optimize the MLB final prediction pool.

This script builds a larger historical sample from processed MLB player files,
joins any stored sportsbook prices we have, and evaluates selector configs on:

1. singles hit rate
2. singles unit profit / ROI on priced rows
3. parlay hit rate from the final selected pool

The output is a JSON report with a baseline config, leaderboard rows, and a
recommended balanced config.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.parlay_analysis import annotate_parlay_board
from sports.mlb.scripts.generate_daily_prediction_pool import (
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST,
    TARGET_SPECS,
    build_supplement_from_long,
    compute_walk_forward_metrics,
    discover_processed_files,
    normalize_player_id,
    to_float,
)
from sports.mlb.scripts.select_high_precision_predictions import (
    SUPPORTED_COUNT_TARGETS,
    build_candidate,
    default_bet_profile_cache_path,
    default_history_cache_path,
    filter_candidates,
    load_or_build_historical_bet_profile_priors,
    load_or_build_historical_bucket_priors,
    select_top_candidates,
)


# history_player_props_long.csv is the real, per-book, per-timestamp market
# archive (event_id, bookmaker_key, market_key, line, over_price,
# under_price, fetched_at_utc). It replaced history_player_props_wide.csv
# as this script's price source because the wide file's prices are a
# consensus AVERAGE across books (fetch_mlb_market_props.py's
# consensus_american_price()) with no book identity at all -- every row
# built from it therefore failed build_candidate()'s price_confirmed check,
# which requires a real Market_Over_Book_Key/Market_Under_Book_Key. Reusing
# build_supplement_from_long() gives this historical universe the exact
# same real, single-book, price-confirmed price the live daily board uses.
DEFAULT_MARKET_LONG = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io" / "history_player_props_long.csv"
DEFAULT_SAMPLE_CACHE = SPORT_ROOT / "data" / "predictions" / "calibration" / "historical_pool_universe_2026.csv"
DEFAULT_REPORT_JSON = SPORT_ROOT / "data" / "predictions" / "calibration" / "historical_pool_optimization_2026.json"


TARGET_TO_ACTUAL_COL = {
    "H": "H",
    "TB": "TB",
    "R": "R",
    "K": "K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the MLB final prediction pool on historical data.")
    parser.add_argument("--season", type=int, default=2026, help="Season year to backtest.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Processed MLB data root.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Processed MLB manifest path.")
    parser.add_argument(
        "--market-history-long",
        type=Path,
        default=DEFAULT_MARKET_LONG,
        help=(
            "Real, per-book, per-timestamp market-odds archive CSV or parquet "
            "(event_id, bookmaker_key, market_key, line, over_price, "
            "under_price, fetched_at_utc). NOT the _wide file -- its prices "
            "are a book-blind consensus average, not a real executable price."
        ),
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=DEFAULT_SAMPLE_CACHE,
        help="CSV cache for the expanded historical pool universe.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help="Output JSON summary for the optimization report.",
    )
    parser.add_argument(
        "--refresh-sample-cache",
        action="store_true",
        help="Rebuild the cached historical universe even if it already exists.",
    )
    parser.add_argument(
        "--min-modeled-history-rows",
        type=int,
        default=10,
        help="Minimum prior rows needed before keeping a modeled row as modeled.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Processed MLB history root used for empirical bucket priors.",
    )
    parser.add_argument(
        "--history-cache-json",
        type=Path,
        default=None,
        help="Optional cache JSON for empirical bucket priors.",
    )
    parser.add_argument(
        "--refresh-history-cache",
        action="store_true",
        help="Recompute historical priors even if the cache exists.",
    )
    parser.add_argument(
        "--bet-profile-cache-json",
        type=Path,
        default=None,
        help="Optional cache JSON for settled real-market bet-profile priors.",
    )
    parser.add_argument(
        "--refresh-bet-profile-cache",
        action="store_true",
        help="Recompute settled real-market bet-profile priors even if the cache exists.",
    )
    return parser.parse_args()


def build_price_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Builds the same (Market_Date, Player) -> price-row lookup
    build_historical_universe() joins against, but from the real
    per-book, per-timestamp long-format archive rather than the averaged
    wide file -- reusing build_supplement_from_long() so this historical
    universe and the live daily board always agree on what counts as a
    real, price-confirmed decision-time price."""
    if not path.exists():
        return {}
    if path.suffix.lower() == ".parquet":
        long_df = pd.read_parquet(path)
    else:
        long_df = pd.read_csv(path)
    if long_df.empty or "event_date_et" not in long_df.columns or "player_name_norm" not in long_df.columns:
        return {}

    long_df = long_df.copy()
    long_df["Market_Date"] = pd.to_datetime(long_df["event_date_et"], errors="coerce")
    long_df = long_df.loc[long_df["Market_Date"].notna()].copy()
    if long_df.empty:
        return {}

    frame = build_supplement_from_long(long_df)
    if frame.empty:
        return {}

    frame["Market_Date"] = pd.to_datetime(frame["Market_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["Player"] = frame["Player"].astype(str)
    price_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.get("Market_Date", "")), str(row.get("Player", "")))
        price_lookup[key] = row.to_dict()
    return price_lookup


def grade_result(actual: float, market_line: float, direction: str) -> str:
    if direction == "OVER":
        if actual > market_line:
            return "win"
        if actual == market_line:
            return "push"
        return "loss"
    if actual < market_line:
        return "win"
    if actual == market_line:
        return "push"
    return "loss"


def american_profit_per_unit(price: float | None) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not pd.notna(value) or abs(value) < 100.0:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


def settled_units(result: str, side_price: float | None) -> float | None:
    profit_if_win = american_profit_per_unit(side_price)
    if profit_if_win is None:
        return None
    if result == "win":
        return profit_if_win
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    return None


def selector_args(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "top_n": 10,
        "min_abs_edge": 0.45,
        "min_history_rows": 11,
        "min_prediction": 0.0,
        "min_hit_probability": 0.58,
        "min_graded_hit_rate": 0.60,
        "max_push_probability": 0.24,
        "max_days_since_history": 4,
        "max_per_player": 1,
        "max_per_game": 2,
        "max_per_team": 3,
        "max_per_market_bucket": 4,
        "min_market_books": 0,
        "max_market_line_std": 0.0,
        "min_expected_value": -1.0,
        "allow_unpriced_side": False,
        "allow_baseline": False,
        "require_real_market_source": False,
        "allow_synthetic_unders": False,
        "targets": sorted(SUPPORTED_COUNT_TARGETS),
        "history_season": 2026,
        "min_history_bucket_rows": 50,
        "max_history_prior_weight": 0.35,
        "history_prior_strength": 400.0,
        "disable_historical_calibration": False,
        "min_bet_profile_rows": 12,
        "max_bet_profile_prior_weight": 0.25,
        "bet_profile_prior_strength": 80.0,
        "min_market_availability_rows": 12,
        "disable_historical_bet_profiles": False,
        "min_historical_bet_profile_support": 0,
        "min_historical_bet_profile_win_rate": 0.0,
        "min_historical_market_availability_support": 0,
        "min_historical_market_availability_rate": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def build_historical_universe(
    *,
    season: int,
    data_dir: Path,
    manifest: Path,
    sample_cache: Path,
    refresh_sample_cache: bool,
    min_modeled_history_rows: int,
    price_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    if sample_cache.exists() and not refresh_sample_cache:
        return pd.read_csv(sample_cache)

    processed_files = discover_processed_files(data_dir.resolve(), manifest.resolve(), int(season))
    rows: list[dict[str, Any]] = []

    for path in processed_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns or "Player_Type" not in frame.columns:
            continue
        frame = frame.copy()
        frame["_game_date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["_game_date"].notna()].copy()
        if frame.empty:
            continue
        sort_cols = [column for column in ["_game_date", "Game_Index"] if column in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols).reset_index(drop=True)

        player_type = str(frame.iloc[0].get("Player_Type", "")).strip().lower()
        specs = [spec for spec in TARGET_SPECS if spec.role == player_type and spec.target in SUPPORTED_COUNT_TARGETS]
        if not specs:
            continue

        player_display = str(frame.iloc[0].get("Player", "")).strip().replace("_", " ")
        player_norm = str(frame.iloc[0].get("Player", "")).strip()
        player_id = normalize_player_id(player_display)

        for idx, current in frame.iterrows():
            game_date = pd.Timestamp(current["_game_date"]).strftime("%Y-%m-%d")
            history_frame = frame.iloc[:idx].copy()
            if history_frame.empty:
                continue
            last_history_date = pd.to_datetime(history_frame["_game_date"], errors="coerce").max()
            price_row = price_lookup.get((game_date, player_norm), {})

            for spec in specs:
                market_line = to_float(current.get(spec.market_col))
                actual = to_float(current.get(spec.actual_col))
                if market_line is None or actual is None:
                    continue

                history_values = pd.to_numeric(history_frame.get(spec.actual_col), errors="coerce").dropna()
                history_rows = int(len(history_values))
                if history_rows <= 0:
                    continue

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

                gap = to_float(current.get(spec.gap_col)) or 0.0
                direction = "OVER" if gap > 0 else "UNDER"
                is_modeled = abs(float(gap)) > 1e-9 and history_rows >= int(min_modeled_history_rows)
                prediction = float(market_line + gap) if is_modeled else float(baseline)
                prediction = max(0.0, prediction)
                edge = float(prediction - market_line)
                model_selected = "et" if is_modeled else "baseline"
                model_val_mae, model_val_rmse = compute_walk_forward_metrics(history_values)

                rows.append(
                    {
                        "Prediction_Run_Date": game_date,
                        "Game_Date": game_date,
                        "Commence_Time_UTC": str(current.get("Commence_Time_UTC", "") or ""),
                        "Game_ID": str(current.get("Game_ID", "") or ""),
                        "Game_Status_Code": "P",
                        "Game_Status_Detail": "Scheduled",
                        "Player": player_display,
                        "Player_ID": player_id,
                        "Player_Type": player_type,
                        "Team": str(current.get("Team", "") or ""),
                        "Opponent": str(current.get("Opponent", "") or ""),
                        "Is_Home": str(int(to_float(current.get("Is_Home")) or 0.0)),
                        "Target": spec.target,
                        "Prediction": prediction,
                        "Market_Line": float(market_line),
                        "Market_Source": str(current.get(spec.market_source_col, "synthetic") or "synthetic"),
                        "Market_Books": int(to_float(price_row.get(f"Market_{spec.target}_books")) or 0.0),
                        "Market_Line_Std": float(to_float(price_row.get(f"Market_{spec.target}_line_std")) or 0.0),
                        "Market_Over_Price": to_float(price_row.get(f"Market_{spec.target}_over_price")),
                        "Market_Under_Price": to_float(price_row.get(f"Market_{spec.target}_under_price")),
                        # Real book identity + decision-time timestamp for
                        # the two prices above -- carried straight through
                        # from build_supplement_from_long()'s real,
                        # single-book price selection. build_candidate()
                        # requires the book-key columns to be non-empty to
                        # mark a row price_confirmed; without them (as when
                        # this universe was built from the averaged wide
                        # file) every row was silently unconfirmable.
                        "Market_Over_Book_Key": str(price_row.get(f"Market_{spec.target}_over_book_key", "") or ""),
                        "Market_Under_Book_Key": str(price_row.get(f"Market_{spec.target}_under_book_key", "") or ""),
                        "Market_Over_Price_Time": str(price_row.get(f"Market_{spec.target}_over_price_time", "") or ""),
                        "Market_Under_Price_Time": str(price_row.get(f"Market_{spec.target}_under_price_time", "") or ""),
                        "Edge": edge,
                        "History_Rows": history_rows,
                        "Last_History_Date": pd.Timestamp(last_history_date).strftime("%Y-%m-%d") if pd.notna(last_history_date) else "",
                        "Model_Selected": model_selected,
                        "Model_Members": model_selected,
                        "Model_Val_MAE": float(model_val_mae),
                        "Model_Val_RMSE": float(model_val_rmse),
                        "Result": grade_result(float(actual), float(market_line), direction),
                        "Actual": float(actual),
                    }
                )

    universe = pd.DataFrame(rows)
    sample_cache.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(sample_cache, index=False)
    return universe


def score_config(
    *,
    name: str,
    universe: pd.DataFrame,
    args: SimpleNamespace,
    calibration: dict | None,
    bet_profile_priors: dict | None,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    parlay_outcomes: list[str] = []
    parlay_projected: list[float] = []
    priced_pool_units: list[float] = []
    selected_dates = 0

    for market_date, part in universe.groupby("Game_Date", dropna=False):
        candidates = []
        for row in part.to_dict(orient="records"):
            candidate = build_candidate(
                row,
                calibration=calibration,
                bet_profile_priors=bet_profile_priors,
                min_history_bucket_rows=int(args.min_history_bucket_rows),
                max_history_prior_weight=float(args.max_history_prior_weight),
                history_prior_strength=float(args.history_prior_strength),
                min_bet_profile_rows=int(args.min_bet_profile_rows),
                max_bet_profile_prior_weight=float(args.max_bet_profile_prior_weight),
                bet_profile_prior_strength=float(args.bet_profile_prior_strength),
                min_market_availability_rows=int(args.min_market_availability_rows),
            )
            if candidate is not None:
                candidates.append(candidate)

        eligible, _ = filter_candidates(candidates, args)
        selected = select_top_candidates(eligible, args)
        if not selected:
            continue
        selected_dates += 1

        day_units: list[float] = []
        parlay_plays: list[dict[str, Any]] = []
        for candidate in selected:
            result = str(candidate.raw.get("Result", ""))
            units = settled_units(result, candidate.selected_side_price)
            selected_rows.append(
                {
                    "date": market_date,
                    "player": candidate.player,
                    "target": candidate.target,
                    "direction": candidate.direction,
                    "result": result,
                    "units": units,
                    "priced": units is not None,
                    "real_market": bool(candidate.market_source == "real" and candidate.market_books > 0),
                    "price_confirmed": bool(candidate.price_confirmed),
                    "projected_probability": candidate.calibrated_graded_hit_rate,
                    "expected_value_per_unit": candidate.expected_value_per_unit,
                    "market_books": candidate.market_books,
                }
            )
            if units is not None:
                day_units.append(float(units))

            parlay_plays.append(
                {
                    "player": candidate.player,
                    "player_display_name": candidate.player,
                    "team": candidate.team,
                    "game_id": candidate.game_id,
                    "target": candidate.target,
                    "direction": candidate.direction,
                    "market_bucket": candidate.market_bucket,
                    "estimated_graded_hit_rate": candidate.calibrated_graded_hit_rate,
                    "final_pool_quality_score": max(0.0, min(1.0, candidate.precision_score / 1.15)),
                    "parlay_precision_eligible": True,
                    "result": result,
                }
            )

        if day_units:
            priced_pool_units.append(sum(day_units))

        parlay_payload = annotate_parlay_board(
            parlay_plays,
            sport="mlb",
            probability_field="estimated_graded_hit_rate",
            eligibility_field="parlay_precision_eligible",
        )
        for parlay in parlay_payload["pairs"]:
            leg_results = [str(parlay_plays[int(index)].get("result", "")) for index in parlay["leg_indices"]]
            if "loss" in leg_results:
                parlay_outcomes.append("miss")
            elif leg_results and all(value == "win" for value in leg_results):
                parlay_outcomes.append("hit")
            elif "push" in leg_results:
                parlay_outcomes.append("push")
            else:
                parlay_outcomes.append("unresolved")
            parlay_projected.append(float(parlay["projected_probability"]))

    selected_frame = pd.DataFrame(selected_rows)
    if selected_frame.empty:
        return {
            "name": name,
            "config": vars(args),
            "selected_dates": 0,
            "play_count": 0,
            "graded_play_count": 0,
            "hit_play_count": 0,
            "play_hit_rate": None,
            "priced_play_count": 0,
            "priced_roi": None,
            "avg_units_per_priced_play": None,
            "avg_units_per_priced_pool": None,
            "parlay_count": 0,
            "graded_parlay_count": 0,
            "parlay_hit_rate": None,
        }

    graded_frame = selected_frame.loc[selected_frame["result"].isin(["win", "loss"])].copy()
    priced_frame = selected_frame.loc[selected_frame["priced"]].copy()
    real_frame = selected_frame.loc[selected_frame["real_market"]].copy()
    confirmed_frame = selected_frame.loc[selected_frame["price_confirmed"]].copy()
    confirmed_graded_frame = confirmed_frame.loc[confirmed_frame["result"].isin(["win", "loss"])].copy()
    parlay_counts = Counter(parlay_outcomes)
    graded_parlays = int(parlay_counts.get("hit", 0) + parlay_counts.get("miss", 0))

    return {
        "name": name,
        "config": vars(args),
        "selected_dates": int(selected_dates),
        "play_count": int(len(selected_frame)),
        "graded_play_count": int(len(graded_frame)),
        "hit_play_count": int((graded_frame["result"] == "win").sum()),
        "play_hit_rate": float((graded_frame["result"] == "win").mean()) if not graded_frame.empty else None,
        "real_play_count": int(len(real_frame)),
        "price_confirmed_play_count": int(len(confirmed_frame)),
        "price_confirmed_rate": float(confirmed_frame.shape[0] / selected_frame.shape[0]) if not selected_frame.empty else None,
        "price_confirmed_hit_rate": float((confirmed_graded_frame["result"] == "win").mean()) if not confirmed_graded_frame.empty else None,
        "priced_play_count": int(len(priced_frame)),
        "priced_roi": float(priced_frame["units"].mean()) if not priced_frame.empty else None,
        "avg_units_per_priced_play": float(priced_frame["units"].mean()) if not priced_frame.empty else None,
        "avg_units_per_priced_pool": float(sum(priced_pool_units) / len(priced_pool_units)) if priced_pool_units else None,
        "avg_projected_play_hit_rate": float(selected_frame["projected_probability"].mean()) if not selected_frame.empty else None,
        "parlay_count": int(len(parlay_outcomes)),
        "graded_parlay_count": graded_parlays,
        "parlay_hit_rate": float(parlay_counts.get("hit", 0) / graded_parlays) if graded_parlays else None,
        "avg_projected_parlay_hit_rate": float(sum(parlay_projected) / len(parlay_projected)) if parlay_projected else None,
    }


def choose_recommended(results: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_hit = baseline.get("play_hit_rate") or 0.0
    baseline_confirmed_hit = baseline.get("price_confirmed_hit_rate") or baseline_hit
    baseline_roi = baseline.get("priced_roi") if baseline.get("priced_roi") is not None else -999.0
    hit_floor = baseline_hit - 0.01
    confirmed_hit_floor = baseline_confirmed_hit - 0.02

    eligible = [
        row
        for row in results
        if row.get("play_hit_rate") is not None
        and row.get("play_hit_rate", 0.0) >= hit_floor
        and row.get("graded_play_count", 0) >= max(18, int((baseline.get("graded_play_count") or 0) * 0.4))
        and row.get("price_confirmed_play_count", 0) >= 6
        and (row.get("price_confirmed_hit_rate") is None or row.get("price_confirmed_hit_rate", 0.0) >= confirmed_hit_floor)
        and (row.get("priced_roi") is None or row.get("priced_roi", -999.0) >= baseline_roi)
    ]
    if not eligible:
        eligible = [row for row in results if row.get("play_hit_rate") is not None]

    eligible.sort(
        key=lambda row: (
            row.get("avg_units_per_priced_pool") if row.get("avg_units_per_priced_pool") is not None else -999.0,
            row.get("price_confirmed_hit_rate") if row.get("price_confirmed_hit_rate") is not None else -999.0,
            row.get("price_confirmed_rate") if row.get("price_confirmed_rate") is not None else -999.0,
            row.get("parlay_hit_rate") if row.get("parlay_hit_rate") is not None else -999.0,
            row.get("play_hit_rate") if row.get("play_hit_rate") is not None else -999.0,
            row.get("priced_roi") if row.get("priced_roi") is not None else -999.0,
        ),
        reverse=True,
    )
    return eligible[0]


def main() -> None:
    args = parse_args()
    if args.history_cache_json is None:
        args.history_cache_json = default_history_cache_path(int(args.season))
    if args.bet_profile_cache_json is None:
        args.bet_profile_cache_json = default_bet_profile_cache_path(int(args.season))

    price_lookup = build_price_lookup(args.market_history_long.resolve())
    universe = build_historical_universe(
        season=int(args.season),
        data_dir=args.data_dir.resolve(),
        manifest=args.manifest.resolve(),
        sample_cache=args.sample_cache.resolve(),
        refresh_sample_cache=bool(args.refresh_sample_cache),
        min_modeled_history_rows=int(args.min_modeled_history_rows),
        price_lookup=price_lookup,
    )

    calibration = load_or_build_historical_bucket_priors(
        history_dir=args.history_dir.resolve(),
        season=int(args.season),
        cache_json=args.history_cache_json.resolve(),
        refresh=bool(args.refresh_history_cache),
    )
    bet_profile_priors = load_or_build_historical_bet_profile_priors(
        history_dir=args.history_dir.resolve(),
        season=int(args.season),
        cache_json=args.bet_profile_cache_json.resolve(),
        refresh=bool(args.refresh_bet_profile_cache),
    )

    configs: list[tuple[str, SimpleNamespace]] = []
    baseline_args = selector_args(history_season=int(args.season))
    configs.append(("baseline_current", baseline_args))

    idx = 1
    for top_n in [5, 6, 7]:
        for min_abs_edge in [0.45, 0.75]:
            for min_hit in [0.60, 0.64]:
                for min_graded in [0.68, 0.72]:
                    for max_market_bucket in [1, 2]:
                        for min_books in [2, 3]:
                            for min_ev in [-1.0, 0.0]:
                                config_name = f"grid_{idx:03d}"
                                idx += 1
                                configs.append(
                                    (
                                        config_name,
                                        selector_args(
                                            history_season=int(args.season),
                                            top_n=top_n,
                                            min_abs_edge=min_abs_edge,
                                            min_hit_probability=min_hit,
                                            min_graded_hit_rate=min_graded,
                                            max_per_market_bucket=max_market_bucket,
                                            min_market_books=min_books,
                                            min_expected_value=min_ev,
                                            require_real_market_source=True,
                                        ),
                                    )
                                )

    results = [
        score_config(
            name=name,
            universe=universe,
            args=config_args,
            calibration=calibration,
            bet_profile_priors=bet_profile_priors,
        )
        for name, config_args in configs
    ]
    baseline = next(row for row in results if row["name"] == "baseline_current")
    recommended = choose_recommended(results, baseline)

    leaderboard = sorted(
        results,
        key=lambda row: (
            row.get("avg_units_per_priced_pool") if row.get("avg_units_per_priced_pool") is not None else -999.0,
            row.get("play_hit_rate") if row.get("play_hit_rate") is not None else -999.0,
        ),
        reverse=True,
    )[:15]

    report = {
        "season": int(args.season),
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "sample_cache": str(args.sample_cache.resolve()),
        "history_rows": int(len(universe)),
        "sample_dates": int(pd.Series(universe["Game_Date"]).nunique()) if not universe.empty else 0,
        "priced_dates": int(pd.Series(universe.loc[universe["Market_Books"] > 0, "Game_Date"]).nunique()) if not universe.empty else 0,
        "priced_rows": int((pd.to_numeric(universe["Market_Books"], errors="coerce").fillna(0) > 0).sum()) if not universe.empty else 0,
        "baseline": baseline,
        "recommended": recommended,
        "leaderboard": leaderboard,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("MLB PREDICTION POOL OPTIMIZER")
    print("=" * 88)
    print(f"Historical rows:     {len(universe)}")
    print(f"Historical dates:    {report['sample_dates']}")
    print(f"Priced rows:         {report['priced_rows']}")
    print(f"Baseline hit rate:   {baseline.get('play_hit_rate')}")
    print(f"Baseline priced ROI: {baseline.get('priced_roi')}")
    print(f"Recommended config:  {recommended['name']}")
    print(f"Recommended hit:     {recommended.get('play_hit_rate')}")
    print(f"Recommended ROI:     {recommended.get('priced_roi')}")
    print(f"Recommended parlay:  {recommended.get('parlay_hit_rate')}")
    print(f"Report JSON:         {args.report_json}")


if __name__ == "__main__":
    main()
