#!/usr/bin/env python3
"""Build a market-independent NFL weekly projection pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.latent_pipeline import predict_week_components_latent  # noqa: E402
from sports.nfl.predictions.pipeline import HISTORY_COLUMNS, load_weekly_stats  # noqa: E402
from sports.nfl.scripts.fetch_historical_nfl_props import SCHEDULE_URL, _kickoff_utc  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--stats", type=Path, default=NFL_ROOT / "data/raw/player_stats_deployment.parquet")
    parser.add_argument("--schedule", default=SCHEDULE_URL)
    parser.add_argument("--roster", type=Path, default=NFL_ROOT / "data/reference/current_skill_roster.csv")
    parser.add_argument("--depth-chart", type=Path, default=NFL_ROOT / "data/reference/current_depth_chart.csv")
    parser.add_argument("--artifact", type=Path, default=NFL_ROOT / "model/nfl_yardage_latent_hybrid.joblib")
    parser.add_argument("--backtest-rows", type=Path, default=NFL_ROOT / "data/evaluation/backtest_rows.csv")
    parser.add_argument("--backtest-report", type=Path, default=NFL_ROOT / "data/evaluation/backtest_report.json")
    parser.add_argument("--market-snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=NFL_ROOT / "web/data/week_1_pool.json")
    return parser.parse_args()


def _current_qb1(depth: pd.DataFrame) -> pd.DataFrame:
    latest = depth.loc[depth["dt"].astype(str).eq(str(depth["dt"].max()))].copy()
    return (
        latest.loc[latest["pos_abb"].eq("QB") & pd.to_numeric(latest["pos_rank"], errors="coerce").eq(1)]
        .sort_values(["team", "player_name"])
        .drop_duplicates("team", keep="first")
    )


def main() -> int:
    args = parse_args()
    stats = load_weekly_stats(args.stats, start_season=2018)
    roster = pd.read_csv(args.roster).rename(
        columns={"gsis_id": "player_id", "full_name": "player_display_name", "team": "recent_team"}
    )
    depth = pd.read_csv(args.depth_chart)
    schedule = pd.read_parquet(args.schedule)
    games = schedule.loc[
        schedule["season"].eq(args.season)
        & schedule["week"].eq(args.week)
        & schedule["game_type"].eq("REG")
    ].copy()
    if games.empty:
        raise ValueError(f"No regular-season schedule found for {args.season} Week {args.week}.")
    games["kickoff_utc"] = _kickoff_utc(games)
    opponents: dict[str, dict[str, object]] = {}
    for game in games.itertuples(index=False):
        common = {"game_id": str(game.game_id), "kickoff_utc": game.kickoff_utc}
        opponents[str(game.home_team)] = {**common, "opponent": str(game.away_team), "venue": "home"}
        opponents[str(game.away_team)] = {**common, "opponent": str(game.home_team), "venue": "away"}

    latest_history = stats.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last").copy()
    latest_history.index = latest_history["player_id"].astype(str)
    roster = roster.loc[pd.to_numeric(roster.get("season"), errors="coerce").eq(args.season)].copy()
    roster = roster.drop_duplicates("player_id", keep="last")
    roster.index = roster["player_id"].astype(str)
    starters = _current_qb1(depth)
    placeholders: list[pd.Series] = []
    starter_meta: dict[str, dict[str, object]] = {}
    for starter in starters.itertuples(index=False):
        player_id = str(starter.gsis_id)
        team = str(starter.team)
        matchup = opponents.get(team)
        if matchup is None or player_id not in latest_history.index:
            continue
        source = latest_history.loc[player_id].copy()
        if player_id in roster.index:
            source["player_display_name"] = roster.loc[player_id]["player_display_name"]
        source["recent_team"] = team
        source["opponent_team"] = matchup["opponent"]
        source["season"] = args.season
        source["week"] = args.week
        source["season_type"] = "REG"
        for column in HISTORY_COLUMNS:
            source[column] = 0.0
        placeholders.append(source[stats.columns])
        starter_meta[player_id] = {"team": team, "player": str(starter.player_name), **matchup}
    if not placeholders:
        raise ValueError("No current QB1 players could be joined to prior NFL history.")

    augmented = pd.concat([stats, pd.DataFrame(placeholders)], ignore_index=True)
    artifact = joblib.load(args.artifact)
    projected = predict_week_components_latent(
        augmented, artifact, season=args.season, week=args.week
    )
    projected = projected.loc[
        projected["target"].eq("passing")
        & projected["player_id"].astype(str).isin(starter_meta)
    ].copy()

    backtest = pd.read_csv(args.backtest_rows, low_memory=False)
    residuals = backtest.loc[backtest["target"].eq("passing"), "actual"].to_numpy(dtype=float) - backtest.loc[
        backtest["target"].eq("passing"), "prediction"
    ].to_numpy(dtype=float)
    residual_quantiles = np.quantile(residuals, [0.10, 0.50, 0.90])
    report = json.loads(args.backtest_report.read_text(encoding="utf-8"))
    passing_report = next(row for row in report["targets"] if row["target"] == "passing")
    pool: list[dict[str, object]] = []
    for row in projected.itertuples(index=False):
        meta = starter_meta[str(row.player_id)]
        mean = float(row.prediction)
        pool.append(
            {
                "player_id": str(row.player_id),
                "player": str(meta["player"]),
                "team": str(meta["team"]),
                "opponent": str(meta["opponent"]),
                "venue": str(meta["venue"]),
                "game_id": str(meta["game_id"]),
                "kickoff_utc": pd.Timestamp(meta["kickoff_utc"]).isoformat().replace("+00:00", "Z"),
                "depth_role": "QB1",
                "target": "passing_yards",
                "projection": round(mean, 1),
                "p10": round(max(0.0, mean + float(residual_quantiles[0])), 1),
                "median": round(max(0.0, mean + float(residual_quantiles[1])), 1),
                "p90": round(max(0.0, mean + float(residual_quantiles[2])), 1),
                "market_line": None,
                "market_status": "awaiting_two_sided_lines",
            }
        )
    pool.sort(key=lambda row: (-float(row["projection"]), str(row["player"])))
    for rank, row in enumerate(pool, start=1):
        row["projection_rank"] = rank

    market_observations = 0
    market_audit: dict[str, object] = {"status": "not_captured"}
    if args.market_snapshot and args.market_snapshot.is_file():
        snapshot = json.loads(args.market_snapshot.read_text(encoding="utf-8"))
        market_observations = len(snapshot.get("observations", []))
        market_audit = snapshot.get("provider_audit", snapshot.get("audit", market_audit))
    source_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            stats[["player_id", "season", "week", *HISTORY_COLUMNS]], index=False
        ).values.tobytes()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "league": "NFL",
        "season": args.season,
        "week": args.week,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "projection_pool_ready",
        "market_status": "lines_available" if market_observations else "awaiting_lines",
        "market_observations": market_observations,
        "market_audit": market_audit,
        "games": int(len(games)),
        "players": len(pool),
        "scope": "Current depth-chart QB1 passing-yard projections; not sportsbook picks.",
        "validation": {
            "holdout_season": report["evaluation_design"]["holdout_season"],
            "rows": passing_report["metrics"]["rows"],
            "mae": passing_report["metrics"]["mae"],
            "baseline_mae": passing_report["metrics"]["baseline_mae"],
            "mae_improvement_vs_baseline": passing_report["metrics"]["mae_improvement_vs_rolling_baseline"],
        },
        "model": {
            "name": report["architecture"]["name"],
            "selected_architecture": passing_report["model_selection"]["selected_architecture"],
            "source_history_sha256": source_hash,
            "interval_method": "Untouched 2025 passing residual P10/P50/P90 offsets.",
        },
        "pool": pool,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "games": len(games), "players": len(pool), "market_observations": market_observations}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
