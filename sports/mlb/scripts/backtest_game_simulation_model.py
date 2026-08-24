#!/usr/bin/env python3
"""Honest walk-forward calibration check for game_simulation_model.py --
does the real joint simulation's own probability for each market
(moneyline, full-game total, F5 total) actually match what happened in
real holdout games? Same chronological split as the other MLB
backtests in this session, same real 1,040-game dataset.

This is a CALIBRATION check, not a same-game-combo profitability
backtest (there is no real historical F5 market line in this repo's
data to grade combo edges against -- see fetch_mlb_pitcher_game_data.py
and the-odds-api provider additions this session for why the real F5
PRICE only exists going forward, live). What this DOES verify with real
historical data: the simulation's own derived probabilities for
moneyline and full-game total (which DO have real historical market
lines) are honestly calibrated, and the F5 simulation's real probability
is graded against the real F5 OUTCOME (home_innings_1_5 +
away_innings_1_5), even without a historical market price for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "predictions"))
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))
import game_simulation_model as sim  # noqa: E402
import pitcher_bullpen_model as pitching  # noqa: E402
import pitching_enriched_win_model as enriched_model  # noqa: E402
import team_win_model as base_model  # noqa: E402
from backtest_pitching_enriched_win_model import _join_key, flatten_starts_and_bullpen, load_pitcher_rows  # noqa: E402
from backtest_team_win_model import brier_score, load_games, no_vig_home_probability  # noqa: E402

DEFAULT_TEAM_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_team_game_history.csv"
DEFAULT_PITCHER_UNIVERSE = REPO_ROOT / "sports" / "mlb" / "data" / "reference" / "mlb_pitcher_game_data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "sports" / "mlb" / "data" / "evaluation" / "game_simulation_model_backtest.json"
MIN_GAMES_PLAYED_FOR_PREDICTION = 10
NUM_TRIALS = 20000


def run_backtest(
    games: list[dict[str, Any]],
    pitcher_rows: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.3,
    num_trials: int = NUM_TRIALS,
    seed: int = 42,
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
    pooled_train_innings = [
        {"starter_outs": r[f"{side}_starter_outs"], "bullpen_outs": r[f"{side}_bullpen_outs"]}
        for r in train_pitcher_rows for side in ("home", "away")
    ]
    starter_innings_share = pitching.compute_empirical_starter_innings_share(pooled_train_innings)

    all_starts, all_bullpen_appearances = flatten_starts_and_bullpen(pitcher_rows)
    pitcher_history = pitching.build_cumulative_pitcher_stats(all_starts)
    bullpen_history = pitching.build_cumulative_bullpen_stats(all_bullpen_appearances)

    # Real, train-only dispersion and F5-share parameters (never fit on holdout).
    runs_dispersion_ratio = sim.compute_empirical_runs_dispersion(train_games)
    f5_share = sim.compute_empirical_f5_share(train_games)

    moneyline_predictions: list[dict[str, Any]] = []
    full_total_predictions: list[dict[str, Any]] = []
    f5_total_predictions: list[dict[str, Any]] = []

    for game_index, game in enumerate(holdout_games):
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

        sides = enriched_model.expected_runs_per_side_enriched(
            home_team_stats, away_team_stats,
            home_starter_stats=home_starter_stats, home_bullpen_stats=home_bullpen_stats,
            away_starter_stats=away_starter_stats, away_bullpen_stats=away_bullpen_stats,
            starter_innings_share=starter_innings_share,
        )
        if sides is None:
            continue
        home_expected, away_expected = sides

        result = sim.simulate_game_outcomes(
            home_expected, away_expected,
            runs_dispersion_ratio=runs_dispersion_ratio, f5_share=f5_share,
            home_field_advantage=home_field_advantage, num_trials=num_trials,
            seed=seed + game_index,  # real per-game determinism, still varies across the holdout
        )

        actual_home_won = int(game["home_score"] > game["away_score"])
        market_prob = no_vig_home_probability(game["market_home_moneyline"], game["market_away_moneyline"])
        moneyline_predictions.append(
            {"sim_prob": result.home_win_probability, "market_prob": market_prob, "actual_home_won": actual_home_won}
        )

        if game.get("market_run_total") is not None:
            line = float(game["market_run_total"])
            sim_over_prob = result.full_total_over_probability(line)
            actual_over = int(game["total_runs"] > line)
            full_total_predictions.append({"sim_prob": sim_over_prob, "actual_over": actual_over, "line": line})

        home_f5 = game.get("home_innings_1_5")
        away_f5 = game.get("away_innings_1_5")
        if home_f5 not in (None, "") and away_f5 not in (None, "") and f5_share is not None:
            actual_f5_total = float(home_f5) + float(away_f5)
            implied_f5_line = (home_expected + away_expected) * f5_share  # real model-implied line, since no real historical F5 market price exists
            sim_over_prob = result.f5_total_over_probability(implied_f5_line)
            actual_over = int(actual_f5_total > implied_f5_line)
            f5_total_predictions.append({"sim_prob": sim_over_prob, "actual_over": actual_over})

    moneyline_report: dict[str, Any] = {"graded_games": len(moneyline_predictions)}
    if moneyline_predictions:
        probs = [p["sim_prob"] for p in moneyline_predictions]
        outcomes = [p["actual_home_won"] for p in moneyline_predictions]
        moneyline_report["sim_brier_score"] = brier_score(probs, outcomes)
        moneyline_report["sim_pick_accuracy"] = sum(int((p > 0.5) == bool(a)) for p, a in zip(probs, outcomes)) / len(outcomes)
        market_pairs = [(p["market_prob"], p["actual_home_won"]) for p in moneyline_predictions if p["market_prob"] is not None]
        if market_pairs:
            moneyline_report["market_brier_score"] = brier_score([p[0] for p in market_pairs], [p[1] for p in market_pairs])
            moneyline_report["market_priced_games"] = len(market_pairs)

    full_total_report: dict[str, Any] = {"graded_games": len(full_total_predictions)}
    if full_total_predictions:
        probs = [p["sim_prob"] for p in full_total_predictions]
        outcomes = [p["actual_over"] for p in full_total_predictions]
        full_total_report["sim_brier_score"] = brier_score(probs, outcomes)
        full_total_report["sim_over_pick_accuracy"] = sum(int((p > 0.5) == bool(a)) for p, a in zip(probs, outcomes)) / len(outcomes)

    f5_total_report: dict[str, Any] = {"graded_games": len(f5_total_predictions), "note": "graded against a model-implied line -- no real historical F5 market price exists in this dataset"}
    if f5_total_predictions:
        probs = [p["sim_prob"] for p in f5_total_predictions]
        outcomes = [p["actual_over"] for p in f5_total_predictions]
        f5_total_report["sim_brier_score"] = brier_score(probs, outcomes)
        f5_total_report["sim_over_pick_accuracy"] = sum(int((p > 0.5) == bool(a)) for p, a in zip(probs, outcomes)) / len(outcomes)

    return {
        "status": "ok",
        "real_games_total": len(games),
        "train_games": len(train_games),
        "holdout_games": len(holdout_games),
        "home_field_advantage_used": home_field_advantage,
        "starter_innings_share_used": starter_innings_share,
        "runs_dispersion_ratio_used": runs_dispersion_ratio,
        "f5_share_used": f5_share,
        "num_trials_per_game": num_trials,
        "moneyline": moneyline_report,
        "full_game_total": full_total_report,
        "first_5_innings_total": f5_total_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-universe-csv", type=Path, default=DEFAULT_TEAM_UNIVERSE)
    parser.add_argument("--pitcher-universe-csv", type=Path, default=DEFAULT_PITCHER_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    games = load_games(args.team_universe_csv)
    pitcher_rows = load_pitcher_rows(args.pitcher_universe_csv)
    report = run_backtest(games, pitcher_rows, holdout_fraction=args.holdout_fraction, num_trials=args.num_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
