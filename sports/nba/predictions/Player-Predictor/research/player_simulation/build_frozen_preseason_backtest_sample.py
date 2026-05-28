from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.player_simulation.simulate_next_season_player_states import load_player_logs


DEFAULT_OUTPUT_DIR = (
    PLAYER_PREDICTOR_ROOT.parents[1]
    / "validation"
    / "production_shadow"
    / "player_simulation"
    / "backtests"
    / "2025_preseason"
    / "frozen_sample"
)
STAT_COLUMNS = ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA", "MIN"]


def _season_bounds(season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=int(season), month=10, day=1), pd.Timestamp(year=int(season) + 1, month=10, day=1)


def _safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _safe_std(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.size < 2 else float(values.std(ddof=0))


def _safe_median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _safe_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").notna().sum())


def _player_id(frame: pd.DataFrame) -> str:
    for column in ["Player_ID", "player_id"]:
        if column in frame.columns and frame[column].notna().any():
            return str(frame[column].dropna().astype(str).iloc[-1])
    return ""


def _team(frame: pd.DataFrame) -> str:
    for column in ["Team", "team"]:
        if column in frame.columns and frame[column].notna().any():
            return str(frame[column].dropna().astype(str).iloc[-1])
    return ""


def _position(frame: pd.DataFrame) -> str:
    for column in ["Pos", "position"]:
        if column in frame.columns and frame[column].notna().any():
            return str(frame[column].dropna().astype(str).iloc[-1])
    return ""


def _archetype(prior: pd.DataFrame) -> str:
    pts = _safe_mean(prior, "PTS") or 0.0
    reb = _safe_mean(prior, "REB") or 0.0
    ast = _safe_mean(prior, "AST") or 0.0
    if pts >= 22 and ast >= 5:
        return "high_usage_creator"
    if reb >= 9:
        return "interior_rebounder"
    if ast >= 6:
        return "primary_facilitator"
    if pts >= 16:
        return "scoring_wing"
    return "rotation_role_player"


def _state_row(player: str, group: pd.DataFrame, *, cutoff: pd.Timestamp) -> dict[str, Any]:
    group = group.sort_values("Date").copy()
    recent = group.tail(min(25, len(group)))
    warnings: list[str] = []
    if len(group) < 8:
        warnings.append("low_preseason_history_sample")
    for column in ["MIN", "PTS", "REB", "AST"]:
        if _safe_count(group, column) < max(5, len(group) // 3):
            warnings.append(f"missing_or_sparse_{column.lower()}")
    minutes_mean = _safe_mean(recent, "MIN")
    minutes_std = _safe_std(recent, "MIN")
    usage_mean = _safe_mean(recent, "FGA")
    usage_std = _safe_std(recent, "FGA")
    row: dict[str, Any] = {
        "player_id": _player_id(group),
        "player": player,
        "team_prior": _team(group),
        "position": _position(group),
        "archetype": _archetype(group),
        "data_cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "max_source_date": group["Date"].max().strftime("%Y-%m-%d") if group["Date"].notna().any() else "",
        "seasons_available_before_cutoff": int(group["Date"].dt.year.nunique()),
        "games_available_before_cutoff": int(len(group)),
        "prior_season_games": int(len(group)),
        "prior_season_minutes": _safe_mean(group, "MIN"),
        "prior_season_pts": _safe_mean(group, "PTS"),
        "prior_season_reb": _safe_mean(group, "REB"),
        "prior_season_ast": _safe_mean(group, "AST"),
        "recent_minutes_mean": minutes_mean,
        "recent_minutes_median": _safe_median(recent, "MIN"),
        "recent_minutes_std": minutes_std,
        "recent_pts_mean": _safe_mean(recent, "PTS"),
        "recent_reb_mean": _safe_mean(recent, "REB"),
        "recent_ast_mean": _safe_mean(recent, "AST"),
        "minutes_mean": minutes_mean,
        "minutes_std": minutes_std,
        "minutes_cv": None if not minutes_mean else float((minutes_std or 0.0) / max(minutes_mean, 1.0)),
        "usage_fga_mean": usage_mean,
        "usage_fga_std": usage_std,
        "usage_fga_cv": None if not usage_mean else float((usage_std or 0.0) / max(usage_mean, 1.0)),
        "pts_mean": _safe_mean(recent, "PTS"),
        "pts_std": _safe_std(recent, "PTS"),
        "reb_mean": _safe_mean(recent, "REB"),
        "reb_std": _safe_std(recent, "REB"),
        "ast_mean": _safe_mean(recent, "AST"),
        "ast_std": _safe_std(recent, "AST"),
        "stl_mean": _safe_mean(recent, "STL"),
        "stl_std": _safe_std(recent, "STL"),
        "blk_mean": _safe_mean(recent, "BLK"),
        "blk_std": _safe_std(recent, "BLK"),
        "threepm_mean": _safe_mean(recent, "3PM"),
        "threepm_std": _safe_std(recent, "3PM"),
        "missing_feature_warnings": "|".join(sorted(set(warnings))),
        "simulation_eligible": bool(len(group) >= 8 and minutes_mean is not None and _safe_mean(recent, "PTS") is not None),
        "ineligible_reason": "" if len(group) >= 8 and minutes_mean is not None and _safe_mean(recent, "PTS") is not None else "insufficient_pre_cutoff_player_state",
    }
    return row


def _actual_row(player: str, group: pd.DataFrame, *, season: int) -> dict[str, Any]:
    games = int(len(group))
    return {
        "player_id": _player_id(group),
        "player": player,
        "evaluated_season": int(season),
        "actual_games_played": games,
        "actual_mpg": _safe_mean(group, "MIN"),
        "actual_pts": _safe_mean(group, "PTS"),
        "actual_reb": _safe_mean(group, "REB"),
        "actual_ast": _safe_mean(group, "AST"),
        "actual_pra": _safe_mean(group, "PRA"),
        "actual_pr": _safe_mean(group, "PR"),
        "actual_pa": _safe_mean(group, "PA"),
        "actual_ra": _safe_mean(group, "RA"),
        "actual_available": bool(games > 0),
        "outcome_source": "player_game_logs",
    }


def build_frozen_preseason_backtest_sample(
    *,
    data_proc_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluated_season: int = 2025,
    cutoff_date: str = "2025-10-01",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    season_start, season_end = _season_bounds(int(evaluated_season))
    logs, source_manifest = load_player_logs(data_proc_dir, cutoff_date=None)
    if not logs.empty:
        logs["Date"] = pd.to_datetime(logs["Date"], errors="coerce")
    pre = logs.loc[logs["Date"].lt(cutoff)].copy() if not logs.empty else pd.DataFrame()
    actuals = logs.loc[logs["Date"].ge(season_start) & logs["Date"].lt(season_end)].copy() if not logs.empty else pd.DataFrame()

    state_rows = [
        _state_row(player, group, cutoff=cutoff)
        for player, group in pre.groupby(pre["Player"].fillna("Unknown Player").astype(str), dropna=False)
    ] if not pre.empty else []
    outcome_rows = [
        _actual_row(player, group, season=int(evaluated_season))
        for player, group in actuals.groupby(actuals["Player"].fillna("Unknown Player").astype(str), dropna=False)
    ] if not actuals.empty else []

    state_df = pd.DataFrame(state_rows)
    outcome_df = pd.DataFrame(outcome_rows)
    state_csv = output_dir / "frozen_preseason_player_state_rows.csv"
    state_parquet = output_dir / "frozen_preseason_player_state_rows.parquet"
    actual_csv = output_dir / "frozen_preseason_actual_outcomes.csv"
    state_df.to_csv(state_csv, index=False)
    outcome_df.to_csv(actual_csv, index=False)
    try:
        state_df.to_parquet(state_parquet, index=False)
    except Exception:
        state_parquet.write_text("", encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_proc_dir": str(data_proc_dir),
        "evaluated_season": int(evaluated_season),
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "season_start": season_start.strftime("%Y-%m-%d"),
        "season_end_exclusive": season_end.strftime("%Y-%m-%d"),
        "input_rows": int(len(state_df)),
        "eligible_input_rows": int(state_df.get("simulation_eligible", pd.Series(dtype=bool)).fillna(False).sum()) if not state_df.empty else 0,
        "actual_outcome_rows": int(len(outcome_df)),
        "source_manifest": source_manifest,
        "production_behavior_changed": False,
        "promotion_ready": False,
    }
    quality = {
        **manifest,
        "sample_status": "READY" if manifest["eligible_input_rows"] > 0 and manifest["actual_outcome_rows"] > 0 else "INSUFFICIENT_FROZEN_SAMPLE",
        "missing_data_warnings": [
            "no_pre_cutoff_player_state_rows" if manifest["input_rows"] == 0 else "",
            "no_actual_outcome_rows" if manifest["actual_outcome_rows"] == 0 else "",
        ],
    }
    quality["missing_data_warnings"] = [warning for warning in quality["missing_data_warnings"] if warning]
    (output_dir / "frozen_preseason_sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "frozen_preseason_sample_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (output_dir / "frozen_preseason_sample_quality_report.md").write_text(_format_quality_md(quality), encoding="utf-8")
    return {
        "manifest": manifest,
        "quality": quality,
        "output_paths": {
            "state_csv": str(state_csv),
            "state_parquet": str(state_parquet),
            "actual_outcomes_csv": str(actual_csv),
            "manifest_json": str(output_dir / "frozen_preseason_sample_manifest.json"),
            "quality_json": str(output_dir / "frozen_preseason_sample_quality_report.json"),
            "quality_md": str(output_dir / "frozen_preseason_sample_quality_report.md"),
        },
    }


def _format_quality_md(quality: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen Preseason Sample Quality",
            "",
            f"- sample_status: {quality.get('sample_status')}",
            f"- cutoff_date: {quality.get('cutoff_date')}",
            f"- input_rows: {quality.get('input_rows')}",
            f"- eligible_input_rows: {quality.get('eligible_input_rows')}",
            f"- actual_outcome_rows: {quality.get('actual_outcome_rows')}",
            f"- production_behavior_changed: {quality.get('production_behavior_changed')}",
            f"- promotion_ready: {quality.get('promotion_ready')}",
            "",
            "Actual outcomes are evaluation-only and are written separately from frozen input rows.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen preseason player-state rows and isolated actual outcomes.")
    parser.add_argument("--data-proc-dir", type=Path, default=PLAYER_PREDICTOR_ROOT / "Data-Proc")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluated-season", type=int, default=2025)
    parser.add_argument("--cutoff-date", default="2025-10-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_frozen_preseason_backtest_sample(
        data_proc_dir=args.data_proc_dir,
        output_dir=args.output_dir,
        evaluated_season=int(args.evaluated_season),
        cutoff_date=str(args.cutoff_date),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
