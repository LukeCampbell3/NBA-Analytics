#!/usr/bin/env python3
"""Leakage-safe walk-forward backtest for sports/mlb/predictions/
team_win_model.py, in the same spirit as this repo's other walk-forward
backtests (e.g. optimize_walk_forward_policy.py): a chronological
train/holdout split, real closing market lines as the comparison
baseline, and an honest report -- this NEVER asserts the model is ready
for production; it reports real, backtested numbers so that judgment can
be made on real evidence, matching this whole project's discipline.

Real evaluation, not proxy: every graded game in the holdout has both a
real final score (from ESPN) and a real closing market line (moneyline
and/or run total) already recorded in mlb_team_game_history.csv, so
accuracy/calibration here is against real outcomes and real market
prices from the first run -- there is no analog of the MLB player-prop
system's earlier "proxy vs. real price" gap for this dataset, because
the historical fetcher only ever wrote rows with the real values it
found (see fetch_mlb_team_game_history.py's extract_team_game_row).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))
import team_win_model as model  # noqa: E402

DEFAULT_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_team_game_history.csv"
DEFAULT_OUTPUT = REPO_ROOT / "sports" / "mlb" / "data" / "evaluation" / "team_win_model_backtest.json"
MIN_GAMES_PLAYED_FOR_PREDICTION = 10  # both teams need at least this many real prior games this season


def load_games(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    games: list[dict[str, Any]] = []
    for row in rows:
        try:
            home_score = float(row["home_score"])
            away_score = float(row["away_score"])
        except (KeyError, ValueError):
            continue
        games.append(
            {
                "date": row["date"],
                "game_id": row.get("game_id", ""),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": home_score,
                "away_score": away_score,
                "market_home_moneyline": _to_float(row.get("market_home_moneyline")),
                "market_away_moneyline": _to_float(row.get("market_away_moneyline")),
                "market_run_total": _to_float(row.get("market_run_total")),
                "total_runs": home_score + away_score,
                # Real First 5 Innings runs (None for a real rain-shortened game
                # with fewer than 5 real innings played on that side) -- carried
                # through for game_simulation_model.py's real F5 calibration
                # check; unused by this file's own moneyline/run-total backtest.
                "home_innings_1_5": _to_float(row.get("home_innings_1_5")),
                "away_innings_1_5": _to_float(row.get("away_innings_1_5")),
            }
        )
    return sorted(games, key=lambda g: (g["date"], g["game_id"]))


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def american_to_probability(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return -price / (-price + 100.0)
    return None


def no_vig_home_probability(home_ml: Optional[float], away_ml: Optional[float]) -> Optional[float]:
    home_p = american_to_probability(home_ml)
    away_p = american_to_probability(away_ml)
    if home_p is None or away_p is None or (home_p + away_p) <= 0:
        return None
    return home_p / (home_p + away_p)


def brier_score(predicted_probs: list[float], actual_outcomes: list[int]) -> float:
    return sum((p - a) ** 2 for p, a in zip(predicted_probs, actual_outcomes)) / len(predicted_probs)


def log_loss(predicted_probs: list[float], actual_outcomes: list[int]) -> float:
    eps = 1e-9
    total = 0.0
    for p, a in zip(predicted_probs, actual_outcomes):
        clipped = min(max(p, eps), 1 - eps)
        total += -(a * math.log(clipped) + (1 - a) * math.log(1 - clipped))
    return total / len(predicted_probs)


def run_backtest(
    games: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.3,
) -> dict[str, Any]:
    if len(games) < 20:
        return {"status": "insufficient_real_data", "real_games_available": len(games)}

    split_index = int(len(games) * (1 - holdout_fraction))
    train_games = games[:split_index]
    holdout_games = games[split_index:]

    # Real, data-derived home-field advantage, computed ONLY from the
    # training split -- never from the holdout it will be applied to.
    home_field_advantage = model.compute_empirical_home_field_advantage(train_games)

    # Cumulative team stats must walk ALL games in real chronological
    # order (a team's true season-to-date record needs its full real
    # history, train and holdout together, to stay accurate) -- what
    # stays leakage-safe is that stats_as_of() only ever exposes a given
    # game's PRIOR real games, train or holdout, never that game itself
    # or anything after it.
    history = model.build_cumulative_team_stats(games)

    moneyline_predictions: list[dict[str, Any]] = []
    total_predictions: list[dict[str, Any]] = []

    for game in holdout_games:
        home_stats = model.stats_as_of(history[game["home_team"]], game["date"])
        away_stats = model.stats_as_of(history[game["away_team"]], game["date"])
        if (
            home_stats is None
            or away_stats is None
            or home_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
            or away_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
        ):
            continue

        model_prob = model.predict_moneyline_probability(home_stats, away_stats, home_field_advantage=home_field_advantage)
        actual_home_won = int(game["home_score"] > game["away_score"])
        market_prob = no_vig_home_probability(game["market_home_moneyline"], game["market_away_moneyline"])
        if model_prob is not None:
            moneyline_predictions.append(
                {
                    "game_id": game["game_id"],
                    "date": game["date"],
                    "model_home_win_probability": model_prob,
                    "market_no_vig_home_probability": market_prob,
                    "actual_home_won": actual_home_won,
                    "model_pick_correct": int((model_prob > 0.5) == bool(actual_home_won)),
                }
            )

        predicted_total = model.predict_run_total(home_stats, away_stats)
        if predicted_total is not None:
            total_predictions.append(
                {
                    "game_id": game["game_id"],
                    "date": game["date"],
                    "predicted_total": predicted_total,
                    "market_run_total": game["market_run_total"],
                    "actual_total": game["total_runs"],
                    "absolute_error": abs(predicted_total - game["total_runs"]),
                    "market_absolute_error": (
                        abs(game["market_run_total"] - game["total_runs"]) if game["market_run_total"] is not None else None
                    ),
                }
            )

    moneyline_report: dict[str, Any] = {"graded_games": len(moneyline_predictions)}
    if moneyline_predictions:
        probs = [p["model_home_win_probability"] for p in moneyline_predictions]
        outcomes = [p["actual_home_won"] for p in moneyline_predictions]
        moneyline_report.update(
            {
                "model_brier_score": brier_score(probs, outcomes),
                "model_log_loss": log_loss(probs, outcomes),
                "model_pick_accuracy": sum(p["model_pick_correct"] for p in moneyline_predictions) / len(moneyline_predictions),
            }
        )
        market_pairs = [(p["market_no_vig_home_probability"], p["actual_home_won"]) for p in moneyline_predictions if p["market_no_vig_home_probability"] is not None]
        if market_pairs:
            market_probs = [pair[0] for pair in market_pairs]
            market_outcomes = [pair[1] for pair in market_pairs]
            moneyline_report["market_brier_score"] = brier_score(market_probs, market_outcomes)
            moneyline_report["market_priced_games"] = len(market_pairs)

    total_report: dict[str, Any] = {"graded_games": len(total_predictions)}
    if total_predictions:
        total_report["model_mean_absolute_error"] = sum(p["absolute_error"] for p in total_predictions) / len(total_predictions)
        market_errors = [p["market_absolute_error"] for p in total_predictions if p["market_absolute_error"] is not None]
        if market_errors:
            total_report["market_mean_absolute_error"] = sum(market_errors) / len(market_errors)
            total_report["market_priced_games"] = len(market_errors)

    return {
        "status": "ok",
        "real_games_total": len(games),
        "train_games": len(train_games),
        "holdout_games": len(holdout_games),
        "home_field_advantage_used": home_field_advantage,
        "min_games_played_required": MIN_GAMES_PLAYED_FOR_PREDICTION,
        "moneyline": moneyline_report,
        "run_total": total_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    games = load_games(args.universe_csv)
    report = run_backtest(games, holdout_fraction=args.holdout_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
