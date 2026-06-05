#!/usr/bin/env python3
"""
Build leakage-safe historical teammate-in/out delta artifact tables.

The artifact estimates what has historically happened to a player's stat rates
when a regular teammate was absent during a shared team stint. It does not apply
same-game actual availability to model rows; live scoring must join these
artifacts to a pregame availability feed before using them as features.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"PTS": "PTS", "TRB": "REB", "AST": "AST"}
DELTA_COLUMNS = [
    "player_id",
    "player",
    "market",
    "teammate_id",
    "teammate",
    "team",
    "condition_type",
    "sample_size",
    "baseline_rate",
    "condition_rate",
    "baseline_per_minute",
    "condition_per_minute",
    "raw_delta",
    "raw_per_minute_delta",
    "shrunk_delta",
    "shrunk_per_minute_delta",
    "confidence",
    "first_shared_date",
    "last_shared_date",
    "last_seen_date",
]
REMOVED_COLUMNS = [
    "team",
    "game_id",
    "date",
    "teammate_id_out",
    "teammate",
    "team_usage_removed",
    "team_minutes_removed",
    "team_rebound_share_removed",
    "team_assist_share_removed",
    "rolling_sample_size",
]


def _resolve_path(path: Path) -> Path:
    if str(path).startswith("/workspace/"):
        path = ROOT.parent / str(path).replace("/workspace/", "", 1)
    if not path.is_absolute():
        path = (ROOT.parent / path).resolve()
    return path


def _load_rows(v9_manifest: Path) -> pd.DataFrame:
    manifest = json.loads(_resolve_path(v9_manifest).read_text(encoding="utf-8"))
    output = _resolve_path(Path(manifest["output"]))
    rows = pd.read_csv(output / "data" / "prop_training_rows.csv")
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    return rows.dropna(subset=["date"]).copy()


def _load_frame(path: Path) -> pd.DataFrame:
    path = _resolve_path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(frame: pd.DataFrame, path: Path) -> str:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return str(path.with_suffix(".parquet"))
    except Exception:
        frame.to_csv(path.with_suffix(".csv"), index=False)
        return str(path.with_suffix(".csv"))


def _normal_name(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def _coerce_logs(logs: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team",
        "GAME_ID": "game_id",
        "GAME_DATE": "date",
        "MIN": "MIN",
        "REB": "REB",
    }
    logs = logs.rename(columns={k: v for k, v in rename.items() if k in logs.columns}).copy()
    required = {"player_id", "player", "team", "game_id", "date", "MIN", "PTS", "REB", "AST"}
    missing = sorted(required - set(logs.columns))
    if missing:
        raise ValueError(f"game logs missing required columns: {missing}")

    logs["date"] = pd.to_datetime(logs["date"], errors="coerce")
    logs["player_id"] = logs["player_id"].astype(str)
    logs["player"] = logs["player"].map(_normal_name)
    logs["team"] = logs["team"].astype(str)
    logs["game_id"] = logs["game_id"].astype(str)
    for col in ["MIN", "PTS", "REB", "AST", "FGA", "FTA", "TOV"]:
        if col not in logs.columns:
            logs[col] = 0.0
        logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0.0)

    if "AVAILABLE_FLAG" in logs.columns:
        available = pd.to_numeric(logs["AVAILABLE_FLAG"], errors="coerce").fillna(0) > 0
    else:
        available = pd.Series(True, index=logs.index)
    logs["played"] = available & (logs["MIN"] > 0)
    return logs.dropna(subset=["date"]).sort_values(["team", "date", "game_id", "player_id"]).reset_index(drop=True)


def _shrink(raw_delta: float, n: int, k: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    weight = n / (n + k)
    return float(raw_delta * weight), float(weight)


def _safe_mean(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return math.nan
    return float(series.mean())


def _build_delta_tables(logs: pd.DataFrame, min_with_games: int, min_without_games: int, shrink_k: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    played = logs.loc[logs["played"]].copy()
    team_game_sets = played.groupby(["team", "game_id"])["player_id"].agg(lambda s: set(s.astype(str))).to_dict()
    name_map = played.drop_duplicates("player_id").set_index("player_id")["player"].to_dict()
    rows_out: list[dict] = []
    rows_in: list[dict] = []

    for (player_id, team), target_games in played.groupby(["player_id", "team"], sort=False):
        target_games = target_games.sort_values("date").copy()
        if len(target_games) < min_with_games + min_without_games:
            continue

        candidate_counts: dict[str, int] = {}
        for _, game in target_games.iterrows():
            teammates = team_game_sets.get((team, game["game_id"]), set()) - {player_id}
            for teammate_id in teammates:
                candidate_counts[teammate_id] = candidate_counts.get(teammate_id, 0) + 1

        for teammate_id, with_count in candidate_counts.items():
            if with_count < min_with_games:
                continue
            shared_mask = target_games.apply(
                lambda r: teammate_id in team_game_sets.get((team, r["game_id"]), set()),
                axis=1,
            )
            shared_dates = target_games.loc[shared_mask, "date"]
            if shared_dates.empty:
                continue

            stint_mask = target_games["date"].between(shared_dates.min(), shared_dates.max())
            stint_games = target_games.loc[stint_mask].copy()
            with_mask = stint_games.apply(
                lambda r: teammate_id in team_game_sets.get((team, r["game_id"]), set()),
                axis=1,
            )
            with_games = stint_games.loc[with_mask]
            without_games = stint_games.loc[~with_mask]
            if len(with_games) < min_with_games or len(without_games) < min_without_games:
                continue

            for market, stat_col in MARKETS.items():
                with_rate = _safe_mean(with_games[stat_col])
                without_rate = _safe_mean(without_games[stat_col])
                with_pm = _safe_mean((with_games[stat_col] / with_games["MIN"].clip(lower=1)).tolist())
                without_pm = _safe_mean((without_games[stat_col] / without_games["MIN"].clip(lower=1)).tolist())
                raw_delta = without_rate - with_rate
                raw_pm_delta = without_pm - with_pm
                shrunk, confidence = _shrink(raw_delta, len(without_games), shrink_k)
                shrunk_pm, _ = _shrink(raw_pm_delta, len(without_games), shrink_k)
                base = {
                    "player_id": player_id,
                    "player": name_map.get(player_id, player_id),
                    "market": market,
                    "teammate_id": teammate_id,
                    "teammate": name_map.get(teammate_id, teammate_id),
                    "team": team,
                    "sample_size": int(len(without_games)),
                    "baseline_rate": with_rate,
                    "condition_rate": without_rate,
                    "baseline_per_minute": with_pm,
                    "condition_per_minute": without_pm,
                    "raw_delta": float(raw_delta),
                    "raw_per_minute_delta": float(raw_pm_delta),
                    "shrunk_delta": shrunk,
                    "shrunk_per_minute_delta": shrunk_pm,
                    "confidence": confidence,
                    "first_shared_date": shared_dates.min().date().isoformat(),
                    "last_shared_date": shared_dates.max().date().isoformat(),
                    "last_seen_date": target_games["date"].max().date().isoformat(),
                }
                rows_out.append({**base, "condition_type": "teammate_out"})

                in_raw = with_rate - without_rate
                in_raw_pm = with_pm - without_pm
                in_shrunk, in_confidence = _shrink(in_raw, len(with_games), shrink_k)
                in_pm_shrunk, _ = _shrink(in_raw_pm, len(with_games), shrink_k)
                rows_in.append({
                    **base,
                    "condition_type": "teammate_in",
                    "sample_size": int(len(with_games)),
                    "baseline_rate": without_rate,
                    "condition_rate": with_rate,
                    "baseline_per_minute": without_pm,
                    "condition_per_minute": with_pm,
                    "raw_delta": float(in_raw),
                    "raw_per_minute_delta": float(in_raw_pm),
                    "shrunk_delta": in_shrunk,
                    "shrunk_per_minute_delta": in_pm_shrunk,
                    "confidence": in_confidence,
                })

    out_frame = pd.DataFrame(rows_out, columns=DELTA_COLUMNS)
    in_frame = pd.DataFrame(rows_in, columns=DELTA_COLUMNS)
    role_shift = out_frame.copy()
    if not role_shift.empty:
        role_shift["role_shift_score"] = (
            role_shift["shrunk_delta"].abs()
            * role_shift["confidence"]
            * role_shift["sample_size"].clip(lower=1).pow(0.25)
        )
        role_shift = role_shift.sort_values("role_shift_score", ascending=False)
    return out_frame, in_frame, role_shift


def _build_removed_tables(logs: pd.DataFrame, min_prior_games: int) -> pd.DataFrame:
    played = logs.loc[logs["played"]].copy().sort_values(["player_id", "date", "game_id"])
    for col in ["MIN", "FGA", "FTA", "TOV", "REB", "AST"]:
        played[f"prior_{col.lower()}"] = (
            played.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
            .fillna(0.0)
        )
    played["prior_games"] = played.groupby("player_id").cumcount()

    team_games = played[["team", "game_id", "date"]].drop_duplicates().sort_values(["team", "date", "game_id"])
    stint_rows: list[dict] = []
    for (player_id, team), player_team in played.groupby(["player_id", "team"], sort=False):
        if len(player_team) < min_prior_games:
            continue
        first_date = player_team["date"].min()
        last_date = player_team["date"].max()
        game_ids_played = set(player_team["game_id"])
        eligible_games = team_games.loc[
            (team_games["team"] == team)
            & team_games["date"].between(first_date, last_date)
            & ~team_games["game_id"].isin(game_ids_played)
        ]
        if eligible_games.empty:
            continue

        history = player_team.set_index("date").sort_index()
        for _, game in eligible_games.iterrows():
            prior = history.loc[history.index < game["date"]].tail(1)
            if prior.empty:
                continue
            row = prior.iloc[0]
            if int(row["prior_games"]) < min_prior_games:
                continue
            usage_removed = float(row["prior_fga"] + 0.44 * row["prior_fta"] + row["prior_tov"])
            stint_rows.append({
                "team": team,
                "game_id": str(game["game_id"]),
                "date": game["date"].date().isoformat(),
                "teammate_id_out": player_id,
                "teammate": row["player"],
                "team_usage_removed": usage_removed,
                "team_minutes_removed": float(row["prior_min"]),
                "team_rebound_share_removed": float(row["prior_reb"]),
                "team_assist_share_removed": float(row["prior_ast"]),
                "rolling_sample_size": int(min(row["prior_games"], 10)),
            })
    return pd.DataFrame(stint_rows, columns=REMOVED_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lineup delta artifacts")
    parser.add_argument("--v9-manifest", type=Path, default=ROOT / "model" / "props" / "v9_3" / "manifest.json")
    parser.add_argument("--game-logs", type=Path, default=ROOT / "data copy" / "raw" / "nba_enrichment" / "season=2026" / "player_game_logs.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_4" / "lineup_delta_artifacts")
    parser.add_argument("--shrink-k", type=float, default=20.0)
    parser.add_argument("--min-with-games", type=int, default=6)
    parser.add_argument("--min-without-games", type=int, default=2)
    parser.add_argument("--min-prior-games", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.v9_manifest)
    logs = _coerce_logs(_load_frame(args.game_logs))
    args.output = _resolve_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    out_deltas, in_deltas, role_shift = _build_delta_tables(
        logs=logs,
        min_with_games=args.min_with_games,
        min_without_games=args.min_without_games,
        shrink_k=args.shrink_k,
    )
    removed = _build_removed_tables(logs=logs, min_prior_games=args.min_prior_games)

    files = {
        "player_teammate_out_deltas": _write_table(out_deltas, args.output / "player_teammate_out_deltas"),
        "player_teammate_in_deltas": _write_table(in_deltas, args.output / "player_teammate_in_deltas"),
        "stat_specific_role_shift": _write_table(role_shift, args.output / "stat_specific_role_shift"),
        "team_usage_removed": _write_table(removed[["team", "game_id", "date", "teammate_id_out", "teammate", "team_usage_removed", "rolling_sample_size"]], args.output / "team_usage_removed"),
        "team_minutes_removed": _write_table(removed[["team", "game_id", "date", "teammate_id_out", "teammate", "team_minutes_removed", "rolling_sample_size"]], args.output / "team_minutes_removed"),
        "team_rebound_share_removed": _write_table(removed[["team", "game_id", "date", "teammate_id_out", "teammate", "team_rebound_share_removed", "rolling_sample_size"]], args.output / "team_rebound_share_removed"),
        "team_assist_share_removed": _write_table(removed[["team", "game_id", "date", "teammate_id_out", "teammate", "team_assist_share_removed", "rolling_sample_size"]], args.output / "team_assist_share_removed"),
    }

    report = {
        "status": "built_historical_lineup_delta_artifacts",
        "promotion_usage": "artifact_ready_requires_pregame_availability_feed_for_live_features",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_prop_rows": int(len(rows)),
        "game_log_rows": int(len(logs)),
        "played_game_log_rows": int(logs["played"].sum()),
        "players": int(logs["player_id"].nunique()),
        "teams": int(logs["team"].nunique()),
        "teammate_out_delta_rows": int(len(out_deltas)),
        "teammate_in_delta_rows": int(len(in_deltas)),
        "team_removed_rows": int(len(removed)),
        "markets": sorted(MARKETS),
        "shrinkage_formula": "shrunk_delta = raw_delta * n / (n + k)",
        "shrink_k": args.shrink_k,
        "min_with_games": args.min_with_games,
        "min_without_games": args.min_without_games,
        "leakage_guardrail": "historical actual absences are used only to estimate split artifacts; current-game actual availability is not joined to validation rows",
        "files": files,
    }
    (args.output / "lineup_delta_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
