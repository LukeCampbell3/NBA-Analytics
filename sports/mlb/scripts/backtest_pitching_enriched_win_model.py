#!/usr/bin/env python3
"""Honest walk-forward backtest comparing the real pitching-ENRICHED
model (pitching_enriched_win_model.py) against the real Pythagorean-only
BASELINE (team_win_model.py, task #23's already-committed result) and
the real market -- on the exact same chronological train/holdout split
and the exact same holdout games, so the comparison is apples-to-apples.
Never asserts the enriched model is ready for production; reports real,
backtested numbers, same discipline as backtest_team_win_model.py.

Real join: mlb_pitcher_game_data.csv (StatsAPI-keyed) is joined onto
mlb_team_game_history.csv (ESPN-keyed) by the real (date, home_team,
away_team) triple -- both sources' own game ids are source-specific, so
this is the same real, verifiable key NFL's half-score dataset uses for
its own (currently unresolved) join. A game with no real pitcher-data
match is not dropped or guessed -- both the base and enriched models
simply fall back to their team-level signal for it (enriched degrades
exactly to base when no real starter/bullpen data is available for that
game), so holdout coverage stays identical between the two models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
import pitcher_bullpen_model as pitching  # noqa: E402
import pitching_enriched_win_model as enriched_model  # noqa: E402
import team_win_model as base_model  # noqa: E402
from backtest_team_win_model import (  # noqa: E402
    american_to_probability,
    brier_score,
    load_games,
    log_loss,
    no_vig_home_probability,
)

DEFAULT_TEAM_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_team_game_history.csv"
DEFAULT_PITCHER_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_pitcher_game_data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "sports" / "mlb" / "data" / "evaluation" / "pitching_enriched_win_model_backtest.json"
MIN_GAMES_PLAYED_FOR_PREDICTION = 10


def load_pitcher_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_starter_id": int(row["home_starter_id"]),
                    "home_starter_name": row["home_starter_name"],
                    "home_starter_outs": int(row["home_starter_outs"]),
                    "home_starter_earned_runs": int(row["home_starter_earned_runs"]),
                    "home_bullpen_outs": int(row["home_bullpen_outs"]),
                    "home_bullpen_earned_runs": int(row["home_bullpen_earned_runs"]),
                    "away_starter_id": int(row["away_starter_id"]),
                    "away_starter_name": row["away_starter_name"],
                    "away_starter_outs": int(row["away_starter_outs"]),
                    "away_starter_earned_runs": int(row["away_starter_earned_runs"]),
                    "away_bullpen_outs": int(row["away_bullpen_outs"]),
                    "away_bullpen_earned_runs": int(row["away_bullpen_earned_runs"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return parsed


def _join_key(date_str: str, home_team: str, away_team: str) -> tuple[str, str, str]:
    return (date_str, home_team, away_team)


def flatten_starts_and_bullpen(pitcher_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts: list[dict[str, Any]] = []
    bullpen_appearances: list[dict[str, Any]] = []
    for row in pitcher_rows:
        for side in ("home", "away"):
            team = row[f"{side}_team"]
            starts.append(
                {
                    "date": row["date"],
                    "pitcher_id": row[f"{side}_starter_id"],
                    "name": row[f"{side}_starter_name"],
                    "outs": row[f"{side}_starter_outs"],
                    "earned_runs": row[f"{side}_starter_earned_runs"],
                }
            )
            bullpen_appearances.append(
                {
                    "date": row["date"],
                    "team": team,
                    "outs": row[f"{side}_bullpen_outs"],
                    "earned_runs": row[f"{side}_bullpen_earned_runs"],
                }
            )
    return starts, bullpen_appearances


def run_backtest(
    games: list[dict[str, Any]],
    pitcher_rows: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.3,
) -> dict[str, Any]:
    if len(games) < 20:
        return {"status": "insufficient_real_data", "real_games_available": len(games)}

    split_index = int(len(games) * (1 - holdout_fraction))
    train_games = games[:split_index]
    holdout_games = games[split_index:]
    train_keys = {_join_key(g["date"], g["home_team"], g["away_team"]) for g in train_games}

    home_field_advantage = base_model.compute_empirical_home_field_advantage(train_games)
    team_history = base_model.build_cumulative_team_stats(games)

    pitcher_by_key = {_join_key(r["date"], r["home_team"], r["away_team"]): r for r in pitcher_rows}
    train_pitcher_rows = [r for r in pitcher_rows if _join_key(r["date"], r["home_team"], r["away_team"]) in train_keys]
    # compute_empirical_starter_innings_share expects pooled starter_outs/
    # bullpen_outs -- pool both real sides of every real train-split game.
    pooled_train_innings = [
        {"starter_outs": r[f"{side}_starter_outs"], "bullpen_outs": r[f"{side}_bullpen_outs"]}
        for r in train_pitcher_rows
        for side in ("home", "away")
    ]
    starter_innings_share = pitching.compute_empirical_starter_innings_share(pooled_train_innings)

    all_starts, all_bullpen_appearances = flatten_starts_and_bullpen(pitcher_rows)
    pitcher_history = pitching.build_cumulative_pitcher_stats(all_starts)
    bullpen_history = pitching.build_cumulative_bullpen_stats(all_bullpen_appearances)

    matched_holdout_games = sum(1 for g in holdout_games if _join_key(g["date"], g["home_team"], g["away_team"]) in pitcher_by_key)

    ml_predictions: list[dict[str, Any]] = []
    total_predictions: list[dict[str, Any]] = []

    for game in holdout_games:
        home_team_stats = base_model.stats_as_of(team_history[game["home_team"]], game["date"])
        away_team_stats = base_model.stats_as_of(team_history[game["away_team"]], game["date"])
        if (
            home_team_stats is None
            or away_team_stats is None
            or home_team_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
            or away_team_stats.games_played < MIN_GAMES_PLAYED_FOR_PREDICTION
        ):
            continue

        key = _join_key(game["date"], game["home_team"], game["away_team"])
        pitcher_row = pitcher_by_key.get(key)
        home_starter_stats = away_starter_stats = home_bullpen_stats = away_bullpen_stats = None
        if pitcher_row is not None:
            home_starter_stats = pitching.stats_as_of(pitcher_history.get(pitcher_row["home_starter_id"], []), game["date"])
            away_starter_stats = pitching.stats_as_of(pitcher_history.get(pitcher_row["away_starter_id"], []), game["date"])
            home_bullpen_stats = pitching.stats_as_of(bullpen_history.get(game["home_team"], []), game["date"])
            away_bullpen_stats = pitching.stats_as_of(bullpen_history.get(game["away_team"], []), game["date"])

        base_prob = base_model.predict_moneyline_probability(home_team_stats, away_team_stats, home_field_advantage=home_field_advantage)
        enriched_prob = enriched_model.predict_moneyline_probability_enriched(
            home_team_stats, away_team_stats,
            home_starter_stats=home_starter_stats, home_bullpen_stats=home_bullpen_stats,
            away_starter_stats=away_starter_stats, away_bullpen_stats=away_bullpen_stats,
            starter_innings_share=starter_innings_share, home_field_advantage=home_field_advantage,
        )
        actual_home_won = int(game["home_score"] > game["away_score"])
        market_prob = no_vig_home_probability(game["market_home_moneyline"], game["market_away_moneyline"])
        if base_prob is not None and enriched_prob is not None:
            ml_predictions.append(
                {
                    "date": game["date"], "game_id": game["game_id"],
                    "base_prob": base_prob, "enriched_prob": enriched_prob,
                    "market_prob": market_prob, "actual_home_won": actual_home_won,
                    "has_real_pitching_data": pitcher_row is not None,
                }
            )

        base_total = base_model.predict_run_total(home_team_stats, away_team_stats)
        enriched_total = enriched_model.predict_run_total_enriched(
            home_team_stats, away_team_stats,
            home_starter_stats=home_starter_stats, home_bullpen_stats=home_bullpen_stats,
            away_starter_stats=away_starter_stats, away_bullpen_stats=away_bullpen_stats,
            starter_innings_share=starter_innings_share,
        )
        if base_total is not None and enriched_total is not None:
            total_predictions.append(
                {
                    "date": game["date"], "game_id": game["game_id"],
                    "base_total": base_total, "enriched_total": enriched_total,
                    "market_total": game["market_run_total"], "actual_total": game["total_runs"],
                }
            )

    moneyline_report: dict[str, Any] = {"graded_games": len(ml_predictions)}
    if ml_predictions:
        outcomes = [p["actual_home_won"] for p in ml_predictions]
        base_probs = [p["base_prob"] for p in ml_predictions]
        enriched_probs = [p["enriched_prob"] for p in ml_predictions]
        moneyline_report["base_brier_score"] = brier_score(base_probs, outcomes)
        moneyline_report["base_log_loss"] = log_loss(base_probs, outcomes)
        moneyline_report["base_pick_accuracy"] = sum(int((p > 0.5) == bool(a)) for p, a in zip(base_probs, outcomes)) / len(outcomes)
        moneyline_report["enriched_brier_score"] = brier_score(enriched_probs, outcomes)
        moneyline_report["enriched_log_loss"] = log_loss(enriched_probs, outcomes)
        moneyline_report["enriched_pick_accuracy"] = sum(int((p > 0.5) == bool(a)) for p, a in zip(enriched_probs, outcomes)) / len(outcomes)
        market_pairs = [(p["market_prob"], p["actual_home_won"]) for p in ml_predictions if p["market_prob"] is not None]
        if market_pairs:
            moneyline_report["market_brier_score"] = brier_score([p[0] for p in market_pairs], [p[1] for p in market_pairs])
            moneyline_report["market_priced_games"] = len(market_pairs)
        moneyline_report["games_with_real_pitching_data"] = sum(1 for p in ml_predictions if p["has_real_pitching_data"])

    total_report: dict[str, Any] = {"graded_games": len(total_predictions)}
    if total_predictions:
        total_report["base_mean_absolute_error"] = sum(abs(p["base_total"] - p["actual_total"]) for p in total_predictions) / len(total_predictions)
        total_report["enriched_mean_absolute_error"] = sum(abs(p["enriched_total"] - p["actual_total"]) for p in total_predictions) / len(total_predictions)
        market_errors = [abs(p["market_total"] - p["actual_total"]) for p in total_predictions if p["market_total"] is not None]
        if market_errors:
            total_report["market_mean_absolute_error"] = sum(market_errors) / len(market_errors)
            total_report["market_priced_games"] = len(market_errors)

    return {
        "status": "ok",
        "real_games_total": len(games),
        "train_games": len(train_games),
        "holdout_games": len(holdout_games),
        "holdout_games_with_real_pitching_data": matched_holdout_games,
        "home_field_advantage_used": home_field_advantage,
        "starter_innings_share_used": starter_innings_share,
        "min_games_played_required": MIN_GAMES_PLAYED_FOR_PREDICTION,
        "moneyline": moneyline_report,
        "run_total": total_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-universe-csv", type=Path, default=DEFAULT_TEAM_UNIVERSE)
    parser.add_argument("--pitcher-universe-csv", type=Path, default=DEFAULT_PITCHER_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    games = load_games(args.team_universe_csv)
    pitcher_rows = load_pitcher_rows(args.pitcher_universe_csv)
    report = run_backtest(games, pitcher_rows, holdout_fraction=args.holdout_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
