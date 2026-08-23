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
ROLE_SPECS = {
    "QB": {"maximum_depth": 1, "target": "passing", "label": "Passing yards"},
    "RB": {"maximum_depth": 2, "target": "rushing", "label": "Rushing yards"},
    "WR": {"maximum_depth": 3, "target": "receiving", "label": "Receiving yards"},
    "TE": {"maximum_depth": 1, "target": "receiving", "label": "Receiving yards"},
}


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


def _current_skill_roles(depth: pd.DataFrame) -> pd.DataFrame:
    latest = depth.loc[depth["dt"].astype(str).eq(str(depth["dt"].max()))].copy()
    latest["pos_rank"] = pd.to_numeric(latest["pos_rank"], errors="coerce")
    eligible = pd.concat(
        [
            latest.loc[
                latest["pos_abb"].eq(position)
                & latest["pos_rank"].le(spec["maximum_depth"])
            ]
            for position, spec in ROLE_SPECS.items()
        ],
        ignore_index=True,
    )
    return eligible.sort_values(["team", "pos_abb", "pos_rank", "player_name"]).drop_duplicates(
        "gsis_id", keep="first"
    )


def _build_parlay_watchlists(pool: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create distinct-game projection templates; never manufacture prop lines."""

    def choose(positions: list[str]) -> list[dict[str, object]]:
        legs: list[dict[str, object]] = []
        used_games: set[str] = set()
        used_players: set[str] = set()
        for position in positions:
            candidates = sorted(
                (row for row in pool if row["position"] == position),
                key=lambda row: (-float(row["projection"]), str(row["player"])),
            )
            selected = next(
                (
                    row
                    for row in candidates
                    if str(row["game_id"]) not in used_games
                    and str(row["player_id"]) not in used_players
                ),
                None,
            )
            if selected is None:
                return []
            used_games.add(str(selected["game_id"]))
            used_players.add(str(selected["player_id"]))
            legs.append(
                {
                    "player": selected["player"],
                    "position": selected["position"],
                    "team": selected["team"],
                    "opponent": selected["opponent"],
                    "game_id": selected["game_id"],
                    "target": selected["target"],
                    "projection": selected["projection"],
                    "market_line": None,
                    "direction": None,
                    "status": "awaiting_two_sided_line",
                }
            )
        return legs

    definitions = [
        ("Air Volume", ["QB", "WR"], "Highest distinct-game QB and WR yardage projections."),
        ("Ground Volume", ["RB", "RB"], "Two highest projected rushers from different games."),
        ("Mixed Skill", ["RB", "WR", "TE"], "One projected volume leader at each non-QB skill position."),
        ("Four-Position", ["QB", "RB", "WR", "TE"], "One projected leader from each position, all in different games."),
    ]
    tickets = []
    for name, positions, note in definitions:
        legs = choose(positions)
        if not legs:
            continue
        tickets.append(
            {
                "name": name,
                "leg_count": len(legs),
                "status": "awaiting_lines",
                "candidate_authorized": False,
                "validation_status": "failed_locked_parlay_holdout",
                "note": note,
                "legs": legs,
            }
        )
    return tickets


def main() -> int:
    args = parse_args()
    if not args.stats.is_file():
        # sports/nfl/data/raw/player_stats_deployment.parquet is gitignored
        # and only ever produced by refresh_nfl_yardage_artifact.py (which
        # itself only runs once a week has real market coverage) or restored
        # from the runtime cache. A cold cache on a week with no completed
        # markets yet (preseason, bye weeks, or the first run on a given
        # branch's cache scope) leaves it genuinely absent -- this is a real
        # "not ready yet" state, not an error, and must not crash the whole
        # publication pipeline. Leave the previously published week_1_pool
        # payload in place rather than crash -- same withheld/not-ready
        # idiom already used elsewhere in this pipeline (e.g.
        # build_fantasy_draft_rankings.py, run_nfl_daily_predictions.py's
        # withheld_payload/`required` artifact check).
        print(json.dumps({
            "status": "withheld",
            "reason": f"No player history is available yet at {args.stats} (cache not warmed / no market week has completed).",
        }, indent=2))
        print(f"Week pool was not regenerated because no player history is available yet; leaving {args.output} unchanged.")
        return 0
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
    starters = _current_skill_roles(depth)
    placeholders: list[pd.Series] = []
    starter_meta: dict[str, dict[str, object]] = {}
    for starter in starters.itertuples(index=False):
        player_id = str(starter.gsis_id)
        team = str(starter.team)
        position = str(starter.pos_abb)
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
        starter_meta[player_id] = {
            "team": team,
            "player": str(starter.player_name),
            "position": position,
            "depth_rank": int(starter.pos_rank),
            "target": ROLE_SPECS[position]["target"],
            **matchup,
        }
    if not placeholders:
        raise ValueError("No current skill-position roles could be joined to prior NFL history.")

    augmented = pd.concat([stats, pd.DataFrame(placeholders)], ignore_index=True)
    artifact = joblib.load(args.artifact)
    projected = predict_week_components_latent(
        augmented, artifact, season=args.season, week=args.week
    )
    projected = projected.loc[projected["player_id"].astype(str).isin(starter_meta)].copy()
    projected = projected.loc[
        projected.apply(
            lambda row: str(row["target"]) == starter_meta[str(row["player_id"])]["target"],
            axis=1,
        )
    ].copy()

    backtest = pd.read_csv(args.backtest_rows, low_memory=False)
    residual_quantiles = {}
    for target in {spec["target"] for spec in ROLE_SPECS.values()}:
        target_rows = backtest.loc[backtest["target"].eq(target)]
        residuals = target_rows["actual"].to_numpy(dtype=float) - target_rows[
            "prediction"
        ].to_numpy(dtype=float)
        residual_quantiles[target] = np.quantile(residuals, [0.10, 0.50, 0.90])
    report = json.loads(args.backtest_report.read_text(encoding="utf-8"))
    target_reports = {row["target"]: row for row in report["targets"]}
    pool: list[dict[str, object]] = []
    for row in projected.itertuples(index=False):
        meta = starter_meta[str(row.player_id)]
        mean = float(row.prediction)
        target = str(row.target)
        quantiles = residual_quantiles[target]
        position = str(meta["position"])
        pool.append(
            {
                "player_id": str(row.player_id),
                "player": str(meta["player"]),
                "team": str(meta["team"]),
                "opponent": str(meta["opponent"]),
                "venue": str(meta["venue"]),
                "game_id": str(meta["game_id"]),
                "kickoff_utc": pd.Timestamp(meta["kickoff_utc"]).isoformat().replace("+00:00", "Z"),
                "position": position,
                "depth_rank": int(meta["depth_rank"]),
                "depth_role": f"{position}{int(meta['depth_rank'])}",
                "target": f"{target}_yards",
                "target_label": ROLE_SPECS[position]["label"],
                "projection": round(mean, 1),
                "p10": round(max(0.0, mean + float(quantiles[0])), 1),
                "median": round(max(0.0, mean + float(quantiles[1])), 1),
                "p90": round(max(0.0, mean + float(quantiles[2])), 1),
                "market_line": None,
                "market_status": "awaiting_two_sided_lines",
            }
        )
    pool.sort(key=lambda row: (str(row["position"]), -float(row["projection"]), str(row["player"])))
    position_ranks: dict[str, int] = {}
    for row in pool:
        position = str(row["position"])
        position_ranks[position] = position_ranks.get(position, 0) + 1
        row["projection_rank"] = position_ranks[position]

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
        "position_counts": {position: int(sum(row["position"] == position for row in pool)) for position in ROLE_SPECS},
        "scope": "Current depth-chart QB1, RB1-2, WR1-3, and TE1 primary-yardage projections; not sportsbook picks.",
        "validation": {
            "holdout_season": report["evaluation_design"]["holdout_season"],
            "targets": {
                target: {
                    "rows": target_reports[target]["metrics"]["rows"],
                    "mae": target_reports[target]["metrics"]["mae"],
                    "baseline_mae": target_reports[target]["metrics"]["baseline_mae"],
                    "mae_improvement_vs_baseline": target_reports[target]["metrics"]["mae_improvement_vs_rolling_baseline"],
                }
                for target in target_reports
            },
        },
        "model": {
            "name": report["architecture"]["name"],
            "selected_architectures": {
                target: values["model_selection"]["selected_architecture"]
                for target, values in target_reports.items()
            },
            "source_history_sha256": source_hash,
            "interval_method": "Position target-specific untouched 2025 residual P10/P50/P90 offsets.",
        },
        "parlay_watchlists": _build_parlay_watchlists(pool),
        "parlay_policy": {
            "status": "withheld",
            "candidate_authorized": False,
            "validation_status": "failed_locked_holdout",
            "reason": "No real Week 1 lines are available, and the deterministic two-leg parlay rule was 2-16 on its locked 2022 holdout.",
        },
        "pool": pool,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "games": len(games), "players": len(pool), "market_observations": market_observations}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
