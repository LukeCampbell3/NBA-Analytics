from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[1]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import build_candidate_id, safe_bool, safe_float, series_numeric, series_text, utc_now_iso, write_json
from research.failure_modes.attribute_pick_failures import attribute_pick_failures, summarize_failure_attribution
from research.failure_modes.discover_unknown_failures import discover_unknown_failures
from research.failure_modes.failure_mode_registry import FailureModeDefinition, load_failure_mode_registry
from research.failure_modes.failure_mode_scoreboard import build_failure_mode_scoreboard, summarize_failure_mode_scoreboard
from research.failure_modes.failure_subregions import build_failure_subregion_scoreboard, summarize_failure_subregion_scoreboard
from research.improvement_ledger.ledger import append_improvement_entry
from research.interventions.propose_interventions import propose_interventions, summarize_interventions
from research.market_quality.stale_price_dependency import annotate_stale_price_dependency_rows, summarize_stale_price_dependency


DEFAULT_TARGET_FAILURE_MODES = [
    "TEAM_OFFENSE_COLLAPSE",
    "LOW_TEAM_ASSIST_ENVIRONMENT",
    "USAGE_SUPPRESSION",
    "MINUTES_BAND_FAILURE",
    "MARKET_PRICE_MISPLACEMENT",
    "CALIBRATION_OVERCONFIDENCE",
]

FAMILY_PROXY_FEATURES: dict[str, list[str]] = {
    "TEAM_OFFENSE_COLLAPSE": [
        "projected_team_fg_pct",
        "line_decision_fragility_score",
        "line_decision_instability_score",
        "same_team_selected_over_count",
        "team_actual_points_vs_trailing",
        "team_actual_ast_vs_trailing",
        "team_actual_fg_pct_delta",
    ],
    "LOW_TEAM_ASSIST_ENVIRONMENT": [
        "projected_team_fg_pct",
        "line_decision_fragility_score",
        "expected_minutes_band_low",
        "team_actual_ast_vs_trailing",
        "team_actual_points_vs_trailing",
    ],
    "USAGE_SUPPRESSION": [
        "role_pathway_shift_score",
        "role_shift_risk",
        "volatility_score",
        "player_actual_fga_vs_trailing",
        "player_actual_usg_delta",
    ],
    "MINUTES_BAND_FAILURE": [
        "expected_minutes_band_low",
        "expected_minutes_band_high",
        "expected_minutes_band_width",
        "minutes_floor_recent",
        "bench_role_flag",
        "rotation_volatility_score",
        "blowout_minutes_sensitivity",
        "foul_rate_minutes_loss_risk",
        "actual_minutes",
    ],
    "MARKET_PRICE_MISPLACEMENT": [
        "market_side_price",
        "market_side_break_even",
        "stress_probability",
        "expected_win_rate",
        "odds_snapshot_time",
        "price_source",
        "book",
        "snapshot_market_side_price",
        "snapshot_over_price",
        "snapshot_under_price",
        "line_moved_since_prediction",
        "odds_moved_since_prediction",
        "stale_price_subregion",
        "would_change_decision",
    ],
    "CALIBRATION_OVERCONFIDENCE": [
        "predicted_probability",
        "stress_probability",
        "expected_win_rate",
        "belief_uncertainty",
        "posterior_variance",
        "calibration_bucket_rows",
    ],
}

SUBREGION_PROPOSAL_COLUMNS = [
    "intervention_id",
    "parent_failure_family",
    "subregion_id",
    "intervention_type",
    "target_markets",
    "trigger_condition",
    "required_features",
    "required_price_fields",
    "missing_features",
    "expected_loss_removal_rate",
    "expected_win_removal_rate",
    "expected_coverage_cost",
    "expected_non_target_damage",
    "overfit_risk",
    "validation_plan",
    "rollback_rule",
    "recommended_next_action",
]

FEATURE_GAP_META: dict[str, dict[str, Any]] = {
    "team_total": {
        "family": "TEAM_OFFENSE_COLLAPSE",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist pre-event team implied totals into selector daily runs.",
        "expected_value_of_adding_feature": "high",
    },
    "pace_proxy": {
        "family": "TEAM_OFFENSE_COLLAPSE",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist pace proxy or implied possessions into selector artifacts.",
        "expected_value_of_adding_feature": "medium",
    },
    "offensive_rating_proxy": {
        "family": "TEAM_OFFENSE_COLLAPSE",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist team offensive-rating proxy into selector artifacts.",
        "expected_value_of_adding_feature": "medium",
    },
    "teammate_return_risk": {
        "family": "USAGE_SUPPRESSION",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist teammate availability-return risk into pre-event feature set.",
        "expected_value_of_adding_feature": "high",
    },
    "projected_assist_conversion_proxy": {
        "family": "LOW_TEAM_ASSIST_ENVIRONMENT",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist assist-conversion proxy from team shooting support model.",
        "expected_value_of_adding_feature": "high",
    },
    "teammate_shooting_support": {
        "family": "LOW_TEAM_ASSIST_ENVIRONMENT",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist teammate shooting support metric into selector daily runs.",
        "expected_value_of_adding_feature": "high",
    },
    "usage_proxy": {
        "family": "USAGE_SUPPRESSION",
        "proxy_exists": True,
        "blocks_discovery": False,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist usage proxy directly instead of inferring from role-shift and volatility proxies.",
        "expected_value_of_adding_feature": "medium",
    },
    "market_side_price": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": False,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist side-specific entry odds into selector rows and board exports.",
        "expected_value_of_adding_feature": "high",
    },
    "market_side_break_even": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": True,
        "blocks_discovery": True,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist break-even probabilities derived from entry odds alongside selector rows.",
        "expected_value_of_adding_feature": "high",
    },
    "odds_snapshot_time": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": False,
        "blocks_discovery": False,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist snapshot fetch timestamps for the exact odds used at prediction time.",
        "expected_value_of_adding_feature": "high",
    },
    "price_source": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": False,
        "blocks_discovery": False,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist the odds source and provenance for every side-specific price field.",
        "expected_value_of_adding_feature": "medium",
    },
    "book": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": False,
        "blocks_discovery": False,
        "blocks_validation": True,
        "recommended_data_pipeline_addition": "Persist the source book or market aggregator identifier for each captured price.",
        "expected_value_of_adding_feature": "medium",
    },
    "side_specific_odds": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": True,
        "blocks_discovery": True,
        "blocks_validation": True,
        "present_if_any": ["snapshot_over_price", "snapshot_under_price", "snapshot_market_side_price"],
        "recommended_data_pipeline_addition": "Persist both over and under prices so no-vig and opposite-side checks are timestamp-safe.",
        "expected_value_of_adding_feature": "high",
    },
    "line_movement_fields": {
        "family": "MARKET_PRICE_MISPLACEMENT",
        "proxy_exists": False,
        "blocks_discovery": False,
        "blocks_validation": True,
        "present_if_any": ["line_moved_since_prediction", "odds_moved_since_prediction"],
        "recommended_data_pipeline_addition": "Persist odds and line movement relative to the prediction snapshot for each candidate.",
        "expected_value_of_adding_feature": "medium",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the failure-mode improvement discovery loop.")
    parser.add_argument("--selected-board-csv", type=Path, action="append", default=[])
    parser.add_argument("--candidate-pool-csv", type=Path, action="append", default=[])
    parser.add_argument("--daily-runs-dir", type=Path, action="append", default=[])
    parser.add_argument("--price-quality-rows", type=Path, default=None)
    parser.add_argument("--output-dir", "--outputs-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, default=None)
    parser.add_argument("--data-proc-root", type=Path, default=PLAYER_PREDICTOR_ROOT / "Data-Proc")
    parser.add_argument("--selected-variant", type=str, default="baseline_no_rebound_diagnostics")
    parser.add_argument("--exclude-failure-family", type=str, action="append", default=[])
    parser.add_argument("--target-failure-family", type=str, action="append", default=[])
    parser.add_argument("--target-subregion", type=str, action="append", default=[])
    parser.add_argument("--discover-subregions", action="store_true")
    parser.add_argument("--broad-walk-forward", action="store_true")
    parser.add_argument("--min-loss-count", type=int, default=3)
    parser.add_argument("--min-resolved-count", type=int, default=8)
    parser.add_argument("--min-pre-event-detectability", type=float, default=0.60)
    parser.add_argument("--max-coverage-cost", type=float, default=0.25)
    parser.add_argument("--max-non-target-damage", type=float, default=0.15)
    parser.add_argument("--max-win-removal-rate", type=float, default=0.35)
    parser.add_argument("--priority-floor", type=float, default=0.005)
    parser.add_argument("--min-cluster-losses", type=int, default=3)
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--shadow-only", action="store_true")
    return parser.parse_args()


def _normalize_date_value(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp) and len(text) == 8 and text.isdigit():
        timestamp = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(timestamp):
        return text[:10]
    return timestamp.strftime("%Y-%m-%d")


def _markdown_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "| none |\n| --- |\n| no rows |"
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    headers = subset.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in subset.iterrows():
        values = []
        for column in headers:
            value = row.get(column, "")
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _auto_discover_selected_board_paths() -> list[Path]:
    tmp_root = Path("tmp")
    if not tmp_root.exists():
        return []
    return sorted(tmp_root.rglob("rebound_validation_rows.csv"))


def _auto_discover_broad_walk_forward_selected_paths() -> list[Path]:
    candidates = [
        Path("tmp/profile_multiwindow/broad_production_board_objective_b12_rows.csv"),
        Path("tmp/rebound_backtest_upperband/mode_compare_abs_edge_board_objective_rows.csv"),
        Path("tmp/rebound_backtest_20260418_20260426/mode_compare_abs_edge_board_objective_rows.csv"),
    ]
    return [path.resolve() for path in candidates if path.exists()]


def _resolve_selected_board_paths(paths: list[Path], *, broad_walk_forward: bool = False) -> list[Path]:
    resolved = [path.resolve() for path in paths if path and path.exists()]
    if resolved:
        return resolved
    if broad_walk_forward:
        broad = _auto_discover_broad_walk_forward_selected_paths()
        if broad:
            return broad
    return [path.resolve() for path in _auto_discover_selected_board_paths()]


def _selected_pick_key(frame: pd.DataFrame) -> pd.Series:
    if "pick_key" in frame.columns:
        return frame["pick_key"].fillna("").astype(str)
    return build_candidate_id(frame).astype(str)


def _load_selected_rows(paths: list[Path], *, selected_variant: str, broad_walk_forward: bool = False) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    used_paths: list[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "variant" in frame.columns and selected_variant:
            frame = frame.loc[frame["variant"].astype(str) == str(selected_variant)].copy()
        if "mode" in frame.columns and broad_walk_forward:
            frame = frame.loc[frame["mode"].astype(str) == "board_objective"].copy()
        if frame.empty:
            continue
        frame["source_selected_board_csv"] = str(path)
        if "market_date" not in frame.columns:
            frame["market_date"] = frame.get("actual_matched_date", frame.get("run_date", ""))
        frame["market_date"] = frame["market_date"].map(_normalize_date_value)
        frame["run_date"] = frame.get("run_date", frame["market_date"]).map(_normalize_date_value)
        frame["actual_matched_date"] = frame.get("actual_matched_date", frame["market_date"]).map(_normalize_date_value)
        frame["selected_pick_key"] = _selected_pick_key(frame)
        frames.append(frame)
        used_paths.append(str(path))
    if not frames:
        return pd.DataFrame(), used_paths
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["run_date", "selected_pick_key"], keep="first").reset_index(drop=True)
    return out, used_paths


def _resolve_daily_runs_dirs(provided_dirs: list[Path], selected_board_paths: list[Path], *, broad_walk_forward: bool = False) -> list[Path]:
    resolved: list[Path] = []
    for directory in provided_dirs:
        if directory and directory.exists():
            resolved.append(directory.resolve())
    if resolved:
        if broad_walk_forward:
            analysis_daily_runs = (PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs").resolve()
            if analysis_daily_runs.exists():
                resolved.append(analysis_daily_runs)
        return sorted(set(resolved))
    inferred: list[Path] = []
    for path in selected_board_paths:
        candidate = path.resolve().parent / "daily_runs"
        if candidate.exists():
            inferred.append(candidate.resolve())
    if broad_walk_forward:
        analysis_daily_runs = (PLAYER_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs").resolve()
        if analysis_daily_runs.exists():
            inferred.append(analysis_daily_runs)
    return sorted(set(inferred))


def _load_candidate_pool_rows(
    *,
    selected_rows: pd.DataFrame,
    candidate_pool_paths: list[Path],
    daily_runs_dirs: list[Path],
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    used_paths: list[str] = []
    target_dates = {
        _normalize_date_value(value)
        for value in selected_rows.get("run_date", pd.Series(dtype="object")).tolist()
        if _normalize_date_value(value)
    }
    explicit_paths = [path.resolve() for path in candidate_pool_paths if path and path.exists()]
    discovered_paths = list(explicit_paths)
    if not discovered_paths:
        for directory in daily_runs_dirs:
            discovered_paths.extend(sorted(directory.rglob("upcoming_market_play_selector_*.csv")))
    for path in sorted(set(discovered_paths)):
        frame = pd.read_csv(path)
        if "market_date" in frame.columns:
            frame["market_date"] = frame["market_date"].map(_normalize_date_value)
            if target_dates:
                frame = frame.loc[frame["market_date"].isin(target_dates)].copy()
        else:
            date_from_name = _normalize_date_value(path.stem.split("_")[-1])
            if target_dates and date_from_name not in target_dates:
                continue
            frame["market_date"] = date_from_name
        if frame.empty:
            continue
        frame["source_candidate_pool_csv"] = str(path)
        frames.append(frame)
        used_paths.append(str(path))
    if not frames:
        return pd.DataFrame(), used_paths
    pool = pd.concat(frames, ignore_index=True)
    pool["candidate_id"] = build_candidate_id(pool)
    pool = pool.drop_duplicates(subset=["candidate_id"], keep="first").reset_index(drop=True)
    return pool, used_paths


def _load_price_quality_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    frame["record_scope"] = series_text(frame, "record_scope")
    frame["candidate_id"] = series_text(frame, "candidate_id")
    selected_on_board = series_text(frame, "selected_on_board").str.lower().isin(["true", "1", "yes"])
    selected_rows = frame.loc[frame["record_scope"].eq("selected") | selected_on_board].copy()
    if selected_rows.empty:
        selected_rows = frame.loc[series_text(frame, "result").str.lower().isin(["win", "loss", "push"])].copy()
    candidate_rows = frame.copy()
    return selected_rows.reset_index(drop=True), candidate_rows.reset_index(drop=True)


def expand_failure_mode_exclusions(
    tokens: list[str] | None,
    *,
    registry: dict[str, FailureModeDefinition] | None = None,
) -> tuple[set[str], set[str]]:
    active_registry = registry or load_failure_mode_registry()
    excluded_modes: set[str] = set()
    excluded_markets: set[str] = set()
    for token in tokens or []:
        text = str(token).strip()
        if not text:
            continue
        normalized = text.upper()
        if normalized == "REBOUND":
            for mode_id, definition in active_registry.items():
                if mode_id.startswith("REBOUND_"):
                    excluded_modes.add(mode_id)
                    excluded_markets.update(family for family in definition.market_families if family == "TRB")
            continue
        if normalized in active_registry:
            excluded_modes.add(normalized)
            excluded_markets.update(family for family in active_registry[normalized].market_families if family == "TRB")
    return excluded_modes, excluded_markets


def _normalize_target_failure_modes(values: list[str] | None) -> list[str]:
    normalized = [str(item).strip().upper() for item in (values or []) if str(item).strip()]
    return normalized or list(DEFAULT_TARGET_FAILURE_MODES)


def _player_data_path(data_proc_root: Path, player_name: str) -> Path:
    candidates = [player_name, player_name.replace(" ", "_")]
    for candidate in candidates:
        path = data_proc_root / candidate / "2026_processed_processed.csv"
        if path.exists():
            return path
    return data_proc_root / player_name / "2026_processed_processed.csv"


def _load_player_log(player_name: str, *, data_proc_root: Path, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if player_name in cache:
        return cache[player_name]
    path = _player_data_path(data_proc_root, player_name)
    if not path.exists():
        cache[player_name] = pd.DataFrame()
        return cache[player_name]
    usecols = [
        "Date",
        "Player",
        "PTS",
        "TRB",
        "AST",
        "FG%",
        "USG%",
        "Did_Not_Play",
        "Team_ID",
        "Opponent",
        "MP",
        "FGA",
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["Date"] = frame["Date"].map(_normalize_date_value)
    frame = frame.sort_values("Date").reset_index(drop=True)
    cache[player_name] = frame
    return frame


def _build_team_game_table(data_proc_root: Path) -> pd.DataFrame:
    if not data_proc_root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    usecols = ["Date", "Team_ID", "Opponent", "PTS", "AST", "FGA", "FG%", "Did_Not_Play", "MP"]
    for path in sorted(data_proc_root.glob("*/2026_processed_processed.csv")):
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except ValueError:
            continue
        frame["Date"] = frame["Date"].map(_normalize_date_value)
        frame["Did_Not_Play"] = frame.get("Did_Not_Play", False).map(lambda value: safe_bool(value, default=False))
        frame["MP"] = series_numeric(frame, "MP", default=0.0)
        frame["FGA"] = series_numeric(frame, "FGA", default=0.0)
        frame["FG%"] = series_numeric(frame, "FG%", default=np.nan)
        active = frame.loc[(~frame["Did_Not_Play"]) & (frame["MP"] > 0.0)].copy()
        if active.empty:
            continue
        active["fgm_est"] = active["FG%"].fillna(0.0) * active["FGA"].fillna(0.0)
        frames.append(active)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["Date", "Team_ID"], as_index=False)
        .agg(
            team_actual_points=("PTS", "sum"),
            team_actual_ast=("AST", "sum"),
            team_actual_fga=("FGA", "sum"),
            team_fgm_est=("fgm_est", "sum"),
            actual_opponent=("Opponent", "first"),
        )
        .sort_values(["Team_ID", "Date"])
        .reset_index(drop=True)
    )
    grouped["team_actual_fg_pct"] = grouped["team_fgm_est"] / grouped["team_actual_fga"].replace(0.0, np.nan)
    trailing_frames: list[pd.DataFrame] = []
    for _, group in grouped.groupby("Team_ID", sort=False):
        ordered = group.sort_values("Date").copy()
        ordered["team_points_trailing_median"] = ordered["team_actual_points"].shift(1).rolling(5, min_periods=3).median()
        ordered["team_ast_trailing_median"] = ordered["team_actual_ast"].shift(1).rolling(5, min_periods=3).median()
        ordered["team_fg_pct_trailing_median"] = ordered["team_actual_fg_pct"].shift(1).rolling(5, min_periods=3).median()
        trailing_frames.append(ordered)
    team_games = pd.concat(trailing_frames, ignore_index=True)
    team_games["team_actual_points_vs_trailing"] = (
        team_games["team_actual_points"] - team_games["team_points_trailing_median"]
    ) / team_games["team_points_trailing_median"].replace(0.0, np.nan)
    team_games["team_actual_ast_vs_trailing"] = (
        team_games["team_actual_ast"] - team_games["team_ast_trailing_median"]
    ) / team_games["team_ast_trailing_median"].replace(0.0, np.nan)
    team_games["team_actual_fg_pct_delta"] = team_games["team_actual_fg_pct"] - team_games["team_fg_pct_trailing_median"]
    team_games["actual_team"] = team_games["Team_ID"].astype(str)
    team_games["game_date"] = team_games["Date"].map(_normalize_date_value)
    return team_games


def _enrich_selected_with_game_context(
    selected_rows: pd.DataFrame,
    *,
    data_proc_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = selected_rows.copy()
    manifest: dict[str, Any] = {
        "data_proc_root": str(data_proc_root),
        "data_proc_available": bool(data_proc_root.exists()),
        "missing_player_logs": [],
        "team_game_table_available": False,
    }
    enriched["game_date"] = enriched.get("actual_matched_date", enriched.get("market_date", enriched.get("run_date", ""))).map(_normalize_date_value)
    if not data_proc_root.exists():
        return enriched, manifest

    cache: dict[str, pd.DataFrame] = {}
    player_context: list[dict[str, Any]] = []
    missing_players: set[str] = set()
    for idx, row in enriched.iterrows():
        player = str(row.get("player", row.get("market_player_raw", ""))).strip()
        game_date = _normalize_date_value(row.get("game_date", row.get("actual_matched_date", row.get("market_date", row.get("run_date", "")))))
        log = _load_player_log(player, data_proc_root=data_proc_root, cache=cache)
        if log.empty:
            missing_players.add(player)
            player_context.append({"row_index": idx})
            continue
        game_row = log.loc[log["Date"] == game_date].tail(1)
        if game_row.empty:
            missing_players.add(player)
            player_context.append({"row_index": idx})
            continue
        game = game_row.iloc[0]
        history = log.loc[log["Date"] < game_date].tail(10).copy()
        fga_trailing = safe_float(history.get("FGA", pd.Series(dtype="float64")).median(), default=np.nan)
        usg_trailing = safe_float(history.get("USG%", pd.Series(dtype="float64")).median(), default=np.nan)
        player_context.append(
            {
                "row_index": idx,
                "team": str(game.get("Team_ID", "")),
                "actual_team": str(game.get("Team_ID", "")),
                "opponent": str(game.get("Opponent", "")),
                "actual_opponent": str(game.get("Opponent", "")),
                "actual_minutes": safe_float(game.get("MP"), default=np.nan),
                "actual_fga": safe_float(game.get("FGA"), default=np.nan),
                "actual_usg": safe_float(game.get("USG%"), default=np.nan),
                "actual_PTS": safe_float(game.get("PTS"), default=np.nan),
                "actual_AST": safe_float(game.get("AST"), default=np.nan),
                "actual_TRB": safe_float(game.get("TRB"), default=np.nan),
                "player_actual_fga_vs_trailing": (
                    (safe_float(game.get("FGA"), default=np.nan) - fga_trailing) / fga_trailing
                    if not np.isnan(fga_trailing) and fga_trailing != 0.0
                    else np.nan
                ),
                "player_actual_usg_delta": (
                    safe_float(game.get("USG%"), default=np.nan) - usg_trailing
                    if not np.isnan(usg_trailing)
                    else np.nan
                ),
            }
        )
    if player_context:
        context_frame = pd.DataFrame(player_context)
        enriched = enriched.reset_index(drop=True).merge(context_frame, left_index=True, right_on="row_index", how="left").drop(columns=["row_index"])
    manifest["missing_player_logs"] = sorted(player for player in missing_players if player)

    team_games = _build_team_game_table(data_proc_root)
    if not team_games.empty and "actual_team" in enriched.columns:
        manifest["team_game_table_available"] = True
        merge_team = team_games.loc[
            :,
            [
                "game_date",
                "actual_team",
                "team_actual_points",
                "team_actual_ast",
                "team_actual_fg_pct",
                "team_actual_points_vs_trailing",
                "team_actual_ast_vs_trailing",
                "team_actual_fg_pct_delta",
            ],
        ].copy()
        enriched["actual_team"] = series_text(enriched, "actual_team")
        enriched = enriched.merge(merge_team, on=["game_date", "actual_team"], how="left")

    over_mask = series_text(enriched, "direction").str.upper().eq("OVER")
    if "actual_team" in enriched.columns:
        over_counts = (
            enriched.loc[over_mask]
            .groupby(["game_date", "actual_team"], dropna=False)
            .size()
            .rename("same_team_selected_over_count")
            .reset_index()
        )
        over_losses = (
            enriched.loc[over_mask & series_text(enriched, "result").str.lower().eq("loss")]
            .groupby(["game_date", "actual_team"], dropna=False)
            .size()
            .rename("same_team_selected_over_loss_count")
            .reset_index()
        )
        enriched = enriched.merge(over_counts, on=["game_date", "actual_team"], how="left")
        enriched = enriched.merge(over_losses, on=["game_date", "actual_team"], how="left")
        enriched["same_team_selected_over_count"] = series_numeric(enriched, "same_team_selected_over_count", default=0.0)
        enriched["same_team_selected_over_loss_count"] = series_numeric(enriched, "same_team_selected_over_loss_count", default=0.0)
    return enriched, manifest


def _build_feature_context(
    *,
    selected_rows: pd.DataFrame,
    candidate_pool_rows: pd.DataFrame | None,
    target_failure_modes: list[str],
    registry: dict[str, FailureModeDefinition],
) -> dict[str, dict[str, Any]]:
    available_columns = set(selected_rows.columns)
    if candidate_pool_rows is not None and not candidate_pool_rows.empty:
        available_columns.update(candidate_pool_rows.columns)
    context: dict[str, dict[str, Any]] = {}
    for failure_mode_id in target_failure_modes:
        definition = registry.get(failure_mode_id)
        proxy_features = FAMILY_PROXY_FEATURES.get(failure_mode_id, list(definition.required_pre_event_features) if definition else [])
        available_proxy = [column for column in proxy_features if column in available_columns]
        missing_features = [column for column in (definition.required_pre_event_features if definition else []) if column not in available_columns]
        evaluation_status = "blocked"
        if available_proxy:
            evaluation_status = "good" if not missing_features else "partial"
        context[failure_mode_id] = {
            "required_features": proxy_features,
            "available_features": available_proxy,
            "missing_features": missing_features,
            "evaluation_status": evaluation_status,
        }
    return context


def _overall_discovery_status(proposals: pd.DataFrame, clusters: pd.DataFrame) -> str:
    if not proposals.empty:
        actions = set(series_text(proposals, "recommended_next_action").tolist())
        if "VALIDATE_SHADOW" in actions:
            return "validate_shadow_next"
        if "FEATURE_GAP_BLOCKED" in actions:
            return "feature_gap_blocked"
        if "NEEDS_MORE_SAMPLE" in actions:
            return "needs_more_sample"
        if actions == {"REJECT_RANDOM"}:
            return "rejected_random"
    if not clusters.empty and series_text(clusters, "recommendation").eq("WATCHLIST").any():
        return "needs_more_sample"
    if not clusters.empty and series_text(clusters, "recommendation").eq("REJECT_RANDOM").all():
        return "rejected_random"
    return "discovery_only"


def _next_exact_action(overall_status: str, recommended_row: dict[str, Any]) -> str:
    failure_mode_id = str(recommended_row.get("failure_mode_id", "")).strip()
    subregion_id = str(recommended_row.get("subregion_id", "")).strip()
    intervention_id = str(recommended_row.get("intervention_id", "")).strip()
    if overall_status == "validate_shadow_next" and intervention_id:
        return f"validate shadow next for {intervention_id}"
    if overall_status == "feature_gap_blocked":
        missing = str(recommended_row.get("missing_features", "")).strip() or "missing features"
        return f"fill feature gaps for {subregion_id or failure_mode_id or 'the top failure mode'} before any shadow validation: {missing}"
    if overall_status == "needs_more_sample":
        return f"expand the settled replay sample for {subregion_id or failure_mode_id or 'the top failure mode'} before shadow validation"
    if overall_status == "broad_signal_unsafe_to_act":
        return f"reject the broad family as directly actionable and keep tracking the narrower candidate {subregion_id or failure_mode_id or 'under review'} for more sample"
    if overall_status == "no_actionable_subregion_found":
        return f"expand the walk-forward sample and fill missing features before revisiting {subregion_id or failure_mode_id or 'non-rebound discovery'}"
    if overall_status == "rejected_random":
        return "reject the current discovery region as noise and rerun discovery on a broader sample"
    return "review the discovery artifacts and choose the next failure family for shadow research"


def _select_recommended_validation_row(proposals: pd.DataFrame, top_scoreboard_row: dict[str, Any]) -> dict[str, Any]:
    if proposals.empty:
        return {
            "failure_mode_id": str(top_scoreboard_row.get("failure_mode_id", "")),
            "recommended_next_action": "NEEDS_MORE_SAMPLE",
        }
    work = proposals.copy()
    work["losses"] = series_numeric(work, "losses", default=0.0)
    work["resolved_count"] = series_numeric(work, "resolved_count", default=0.0)
    for status in ["VALIDATE_SHADOW", "FEATURE_GAP_BLOCKED", "NEEDS_MORE_SAMPLE", "REJECT_RANDOM"]:
        subset = work.loc[
            series_text(work, "recommended_next_action").eq(status)
            & (work["losses"] > 0.0)
            & (work["resolved_count"] > 0.0)
        ]
        if not subset.empty:
            return subset.iloc[0].to_dict()
    observed = work.loc[(work["losses"] > 0.0) & (work["resolved_count"] > 0.0)]
    if not observed.empty:
        return observed.iloc[0].to_dict()
    return work.iloc[0].to_dict()


def _build_feature_gap_report(feature_context: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_name, meta in FEATURE_GAP_META.items():
        present = False
        proxy_exists = bool(meta.get("proxy_exists", False))
        present_if_any = [str(item) for item in meta.get("present_if_any", [])]
        for family_context in feature_context.values():
            if feature_name in family_context.get("available_features", []):
                present = True
                break
            if present_if_any and any(column in family_context.get("available_features", []) for column in present_if_any):
                present = True
                break
        rows.append(
            {
                "feature_name": feature_name,
                "family": str(meta.get("family", "")),
                "present": present,
                "blocks_discovery": bool(meta.get("blocks_discovery", False)) and not present,
                "blocks_validation": bool(meta.get("blocks_validation", False)) and not present,
                "proxy_exists": proxy_exists,
                "recommended_data_pipeline_addition": str(meta.get("recommended_data_pipeline_addition", "")),
                "expected_value_of_adding_feature": str(meta.get("expected_value_of_adding_feature", "")),
            }
        )
    return rows


def _broad_signal_unsafe_families(
    family_scoreboard: pd.DataFrame,
    subregion_scoreboard: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if family_scoreboard.empty or subregion_scoreboard.empty:
        return rows
    for _, family_row in family_scoreboard.iterrows():
        family = str(family_row.get("failure_mode_id", "")).strip()
        if not family:
            continue
        if safe_float(family_row.get("losses"), default=0.0) <= 0.0:
            continue
        subset = subregion_scoreboard.loc[subregion_scoreboard["parent_failure_family"].astype(str) == family].copy()
        if subset.empty:
            continue
        if subset["recommended_next_action"].astype(str).eq("VALIDATE_SHADOW").any():
            continue
        rows.append(
            {
                "failure_family": family,
                "status": "BROAD_SIGNAL_UNSAFE_TO_ACT",
                "losses": int(safe_float(family_row.get("losses"), default=0.0)),
                "wins": int(safe_float(family_row.get("wins"), default=0.0)),
                "coverage_cost": float(safe_float(subset["coverage_cost"].max(), default=0.0)),
                "non_target_damage_risk": float(safe_float(subset["non_target_damage_risk"].max(), default=0.0)),
            }
        )
    return rows


def _overall_status_from_subregions(
    subregion_scoreboard: pd.DataFrame,
    *,
    broad_signal_unsafe: list[dict[str, Any]] | None = None,
) -> str:
    if subregion_scoreboard.empty:
        return "no_actionable_subregion_found"
    actions = set(series_text(subregion_scoreboard, "recommended_next_action").tolist())
    if "VALIDATE_SHADOW" in actions:
        return "validate_shadow_next"
    if broad_signal_unsafe:
        return "broad_signal_unsafe_to_act"
    if "FEATURE_GAP_BLOCKED" in actions:
        return "feature_gap_blocked"
    if "NEEDS_MORE_SAMPLE" in actions:
        return "needs_more_sample"
    if actions == {"REJECT_RANDOM"}:
        return "rejected_random"
    return "no_actionable_subregion_found"


def _propose_actionable_subregion_interventions(subregion_scoreboard: pd.DataFrame) -> pd.DataFrame:
    if subregion_scoreboard.empty:
        return pd.DataFrame(columns=SUBREGION_PROPOSAL_COLUMNS)
    actionable = subregion_scoreboard.loc[subregion_scoreboard["recommended_next_action"].astype(str) == "VALIDATE_SHADOW"].copy()
    if actionable.empty:
        return pd.DataFrame(columns=SUBREGION_PROPOSAL_COLUMNS)
    proposals: list[dict[str, Any]] = []
    for _, row in actionable.iterrows():
        subregion_id = str(row.get("subregion_id", "")).strip()
        parent_failure_family = str(row.get("parent_failure_family", ""))
        intervention_type = str(row.get("intervention_type", ""))
        required_features = str(row.get("required_features", ""))
        if parent_failure_family == "MARKET_PRICE_MISPLACEMENT":
            intervention_type = "price_dependent_tier"
        proposals.append(
            {
                "intervention_id": f"{subregion_id.lower()}__{intervention_type}",
                "parent_failure_family": parent_failure_family,
                "subregion_id": subregion_id,
                "intervention_type": intervention_type,
                "target_markets": str(row.get("target_markets", "")),
                "trigger_condition": str(row.get("trigger_condition", "")),
                "required_features": required_features,
                "required_price_fields": required_features if parent_failure_family == "MARKET_PRICE_MISPLACEMENT" else "",
                "missing_features": str(row.get("missing_features", "")),
                "expected_loss_removal_rate": safe_float(row.get("estimated_loss_removal_rate"), default=0.0),
                "expected_win_removal_rate": safe_float(row.get("estimated_win_removal_rate"), default=0.0),
                "expected_coverage_cost": safe_float(row.get("coverage_cost"), default=0.0),
                "expected_non_target_damage": safe_float(row.get("non_target_damage_risk"), default=0.0),
                "overfit_risk": "medium" if safe_float(row.get("sample_reliability_weight"), default=0.0) >= 0.75 else "high",
                "validation_plan": "paired_replay:no_op,active_risk,trained_bundle,broader_walk_forward",
                "rollback_rule": f"Disable shadow intervention for {subregion_id} if ROI, Brier, ECE, coverage, or non-target board integrity worsens in paired replay.",
                "recommended_next_action": "VALIDATE_SHADOW",
            }
        )
    return pd.DataFrame(proposals, columns=SUBREGION_PROPOSAL_COLUMNS)


def run_improvement_discovery(
    *,
    selected_rows: pd.DataFrame,
    candidate_pool_rows: pd.DataFrame | None,
    outputs_dir: Path,
    ledger_path: Path | None = None,
    input_manifest: dict[str, Any] | None = None,
    data_proc_root: Path | None = None,
    target_failure_modes: list[str] | None = None,
    target_subregions: list[str] | None = None,
    excluded_failure_modes: set[str] | None = None,
    excluded_market_families: set[str] | None = None,
    discover_subregions: bool = False,
    broad_walk_forward: bool = False,
    price_quality_mode: bool = False,
    min_loss_count: int = 3,
    min_resolved_count: int = 8,
    min_pre_event_detectability: float = 0.60,
    max_coverage_cost: float = 0.25,
    max_non_target_damage: float = 0.15,
    max_win_removal_rate: float = 0.35,
    priority_floor: float = 0.005,
    min_cluster_losses: int = 3,
    discovery_only: bool = True,
    shadow_only: bool = True,
) -> dict[str, Any]:
    outputs_dir.resolve().mkdir(parents=True, exist_ok=True)
    registry = load_failure_mode_registry()
    target_modes = _normalize_target_failure_modes(target_failure_modes)
    target_subregion_set = {str(item).strip() for item in (target_subregions or []) if str(item).strip()}
    excluded_modes = set(excluded_failure_modes or set())
    excluded_markets = set(excluded_market_families or set())

    enriched_selected = selected_rows.copy()
    working_candidate_pool = candidate_pool_rows.copy() if candidate_pool_rows is not None else pd.DataFrame()
    stale_price_summary: dict[str, Any] = {}
    if price_quality_mode:
        enriched_selected = annotate_stale_price_dependency_rows(enriched_selected)
        if working_candidate_pool is not None and not working_candidate_pool.empty:
            working_candidate_pool = annotate_stale_price_dependency_rows(working_candidate_pool)
            working_candidate_pool.to_csv(outputs_dir / "stale_price_dependency_rows.csv", index=False)
            stale_price_summary = summarize_stale_price_dependency(working_candidate_pool)
        else:
            enriched_selected.to_csv(outputs_dir / "stale_price_dependency_rows.csv", index=False)
            stale_price_summary = summarize_stale_price_dependency(enriched_selected)
        write_json(outputs_dir / "stale_price_dependency_summary.json", stale_price_summary)
    enrichment_manifest = {"data_proc_available": False, "missing_player_logs": [], "team_game_table_available": False}
    if data_proc_root is not None:
        enriched_selected, enrichment_manifest = _enrich_selected_with_game_context(enriched_selected, data_proc_root=data_proc_root)
    feature_context = _build_feature_context(
        selected_rows=enriched_selected,
        candidate_pool_rows=working_candidate_pool,
        target_failure_modes=target_modes,
        registry=registry,
    )

    attributed = attribute_pick_failures(
        enriched_selected,
        working_candidate_pool,
        registry=registry,
        allowed_failure_modes=set(target_modes),
        excluded_failure_modes=excluded_modes,
    )
    attributed_csv = outputs_dir / "failure_attribution_rows.csv"
    attributed.to_csv(attributed_csv, index=False)
    attribution_summary = summarize_failure_attribution(attributed)
    write_json(outputs_dir / "failure_attribution_summary.json", attribution_summary)

    scoreboard = build_failure_mode_scoreboard(
        attributed,
        registry=registry,
        candidate_pool_rows=working_candidate_pool,
        target_failure_modes=target_modes,
        excluded_failure_modes=excluded_modes,
    )
    scoreboard_csv = outputs_dir / "failure_mode_scoreboard.csv"
    scoreboard.to_csv(scoreboard_csv, index=False)
    scoreboard_summary = summarize_failure_mode_scoreboard(scoreboard)
    subregion_scoreboard = pd.DataFrame()
    subregion_summary: dict[str, Any] = {}
    if discover_subregions:
        subregion_scoreboard = build_failure_subregion_scoreboard(
            attributed,
            candidate_pool_rows=working_candidate_pool,
            target_failure_modes=target_modes,
            min_loss_count=int(min_loss_count),
            min_resolved_count=int(min_resolved_count),
            max_coverage_cost=float(max_coverage_cost),
            max_non_target_damage=float(max_non_target_damage),
            min_pre_event_detectability=float(min_pre_event_detectability),
            max_win_removal_rate=float(max_win_removal_rate),
        )
        if target_subregion_set:
            subregion_scoreboard = subregion_scoreboard.loc[
                subregion_scoreboard["subregion_id"].astype(str).isin(target_subregion_set)
            ].reset_index(drop=True)
        subregion_summary = summarize_failure_subregion_scoreboard(subregion_scoreboard)
    subregion_scoreboard_csv = outputs_dir / "failure_subregion_scoreboard.csv"
    if not subregion_scoreboard.empty:
        subregion_scoreboard.to_csv(subregion_scoreboard_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "parent_failure_family",
                "subregion_id",
                "description",
                "target_markets",
                "required_features",
                "missing_features",
                "proxy_features",
                "candidate_count",
                "selected_count",
                "resolved_count",
                "wins",
                "losses",
                "hit_rate",
                "profit_units",
                "ROI",
                "Brier",
                "ECE",
                "calibration_gap",
                "loss_concentration",
                "win_concentration",
                "loss_to_win_ratio",
                "estimated_loss_removal_rate",
                "estimated_win_removal_rate",
                "coverage_cost",
                "non_target_damage_risk",
                "pre_event_detectability_rate",
                "sample_reliability_weight",
                "subregion_priority_score",
                "run_date_count",
                "window_count",
                "player_count",
                "team_count",
                "market_count",
                "max_player_share",
                "max_team_share",
                "max_date_share",
                "postgame_only",
                "recommended_next_action",
                "intervention_type",
                "trigger_condition",
            ]
        ).to_csv(subregion_scoreboard_csv, index=False)

    if discover_subregions:
        proposals = _propose_actionable_subregion_interventions(subregion_scoreboard)
        intervention_summary = {
            "proposal_count": int(len(proposals)),
            "shadow_only": True,
            "failure_modes": sorted(proposals.get("parent_failure_family", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()) if not proposals.empty else [],
        }
    else:
        proposals = propose_interventions(
            scoreboard,
            registry=registry,
            priority_floor=priority_floor,
            min_resolved_count=max(2, int(min_loss_count)),
            min_loss_count=int(min_loss_count),
            pre_event_detectability_floor=float(min_pre_event_detectability),
            max_coverage_loss=float(max_coverage_cost),
            feature_context=feature_context,
        )
        intervention_summary = summarize_interventions(proposals)
    proposals_csv = outputs_dir / "intervention_candidates.csv"
    proposals.to_csv(proposals_csv, index=False)

    target_markets_for_unknown = sorted(
        {
            market
            for mode_id in target_modes
            for market in registry.get(mode_id, FailureModeDefinition("", (), (), (), (), "", (), (), (), (), (), ())).market_families
            if market and market != "TRB"
        }
    )
    unknown_clusters, unknown_report, yaml_payload = discover_unknown_failures(
        attributed,
        min_cluster_losses=int(min_cluster_losses),
        excluded_failure_modes=sorted(excluded_modes),
        excluded_market_families=sorted(excluded_markets),
        target_market_families=target_markets_for_unknown,
    )
    unknown_clusters_csv = outputs_dir / "unknown_failure_clusters.csv"
    unknown_clusters.to_csv(unknown_clusters_csv, index=False)
    write_json(outputs_dir / "unknown_failure_discovery_report.json", unknown_report)
    try:
        import yaml

        (outputs_dir / "new_failure_mode_candidates.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    except Exception:
        pass

    top_scoreboard_row = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    recommended_validation_row = _select_recommended_validation_row(
        proposals if not proposals.empty else subregion_scoreboard.rename(columns={"parent_failure_family": "failure_mode_id"}),
        top_scoreboard_row,
    )
    broad_signal_unsafe = [] if price_quality_mode else (_broad_signal_unsafe_families(scoreboard, subregion_scoreboard) if discover_subregions else [])
    actionable_subregions = subregion_scoreboard.loc[subregion_scoreboard["recommended_next_action"].astype(str) == "VALIDATE_SHADOW"].copy() if not subregion_scoreboard.empty else pd.DataFrame()
    if discover_subregions:
        if price_quality_mode and target_subregion_set:
            overall_status = _overall_status_from_subregions(subregion_scoreboard, broad_signal_unsafe=[])
        else:
            overall_status = _overall_status_from_subregions(subregion_scoreboard, broad_signal_unsafe=broad_signal_unsafe)
    else:
        overall_status = _overall_discovery_status(proposals, unknown_clusters)
    missing_feature_families = {
        mode_id: context
        for mode_id, context in feature_context.items()
        if context.get("evaluation_status") != "good"
    }
    feature_gap_report = _build_feature_gap_report(feature_context)
    input_manifest = dict(input_manifest or {})
    validation_mode_label = "artifact_free_or_replay_discovery_only"
    trained_bundle_available = bool((PLAYER_PREDICTOR_ROOT / "model" / "production_structured_lstm_stack.json").exists())
    total_wins = int(series_text(attributed, "result").str.lower().eq("win").sum())
    date_values = [value for value in selected_rows.get("run_date", pd.Series(dtype="object")).map(_normalize_date_value).tolist() if value]
    discovery_manifest = {
        "run_id": outputs_dir.resolve().name,
        "window_count": int(series_text(selected_rows, "source_selected_board_csv").replace("", np.nan).nunique(dropna=True)) if not selected_rows.empty else 0,
        "date_range_start": min(date_values) if date_values else "",
        "date_range_end": max(date_values) if date_values else "",
        "total_settled_picks": int(len(enriched_selected)),
        "total_losses": int(series_text(attributed, "result").str.lower().eq("loss").sum()),
        "total_wins": total_wins,
        "markets_included": sorted(series_text(attributed, "market_type").replace("", np.nan).dropna().unique().tolist()),
        "missing_input_families": sorted([row["feature_name"] for row in feature_gap_report if row["blocks_discovery"]]),
        "validation_mode": validation_mode_label,
        "trained_bundle_available": trained_bundle_available,
        "broad_walk_forward": bool(broad_walk_forward),
        "discover_subregions": bool(discover_subregions),
        "price_quality_mode": bool(price_quality_mode),
    }
    output_manifest = {
        "failure_mode_scoreboard_csv": str(scoreboard_csv),
        "failure_subregion_scoreboard_csv": str(subregion_scoreboard_csv),
        "unknown_failure_clusters_csv": str(unknown_clusters_csv),
        "intervention_candidates_csv": str(proposals_csv),
        "failure_attribution_rows_csv": str(attributed_csv),
        "discovery_manifest_json": str(outputs_dir / "discovery_manifest.json"),
        "improvement_discovery_report_json": str(outputs_dir / "improvement_discovery_report.json"),
        "improvement_discovery_report_md": str(outputs_dir / "improvement_discovery_report.md"),
    }
    if price_quality_mode:
        output_manifest["stale_price_dependency_rows_csv"] = str(outputs_dir / "stale_price_dependency_rows.csv")
        output_manifest["stale_price_dependency_summary_json"] = str(outputs_dir / "stale_price_dependency_summary.json")
    write_json(outputs_dir / "discovery_manifest.json", discovery_manifest)
    next_command = _next_exact_action(overall_status, recommended_validation_row)
    top_actionable_subregion = actionable_subregions.iloc[0].to_dict() if not actionable_subregions.empty else {}
    report_payload = {
        "run_id": outputs_dir.resolve().name,
        "created_at": utc_now_iso(),
        "mode": (
            "price_quality_discovery_only"
            if price_quality_mode
            else ("broader_walkforward_discovery_only" if broad_walk_forward else ("discovery_only" if discovery_only else "research"))
        ),
        "shadow_only": bool(shadow_only),
        "status_label": overall_status,
        "validation_mode": validation_mode_label,
        "trained_bundle_available": trained_bundle_available,
        "excluded_families": sorted(excluded_modes),
        "target_families": target_modes,
        "target_subregions": sorted(target_subregion_set),
        "input_paths": input_manifest,
        "output_paths": output_manifest,
        "discovery_manifest": discovery_manifest,
        "enrichment_manifest": enrichment_manifest,
        "feature_context": feature_context,
        "feature_gap_report": feature_gap_report,
        "summary": {
            "total_settled_picks_analyzed": int(len(enriched_selected)),
            "total_losses_analyzed": int(series_text(attributed, "result").str.lower().eq("loss").sum()),
            "total_wins_analyzed": total_wins,
            "rebound_failures_excluded_downweighted": sorted(excluded_modes),
            "top_non_rebound_failure_family": str(top_scoreboard_row.get("failure_mode_id", "")),
            "recommended_next_action": overall_status,
            "top_actionable_subregion": str(top_actionable_subregion.get("subregion_id", "")),
        },
        "failure_mode_scoreboard": scoreboard.to_dict(orient="records"),
        "failure_subregion_scoreboard": subregion_scoreboard.to_dict(orient="records"),
        "broad_signal_unsafe_families": broad_signal_unsafe,
        "unknown_failure_clusters": unknown_clusters.to_dict(orient="records"),
        "candidate_interventions": proposals.to_dict(orient="records"),
        "recommended_first_validation": {
            "failure_mode_id": str(recommended_validation_row.get("failure_mode_id", top_scoreboard_row.get("failure_mode_id", ""))),
            "subregion_id": str(recommended_validation_row.get("subregion_id", "")),
            "intervention_id": str(recommended_validation_row.get("intervention_id", "")),
            "intervention_type": str(recommended_validation_row.get("intervention_type", "")),
            "recommended_next_action": str(recommended_validation_row.get("recommended_next_action", overall_status)),
            "required_features": str(recommended_validation_row.get("required_features", "")),
            "missing_features": str(recommended_validation_row.get("missing_features", "")),
            "expected_coverage_cost": safe_float(recommended_validation_row.get("expected_coverage_cost"), default=np.nan),
            "expected_non_target_damage": safe_float(recommended_validation_row.get("expected_non_target_damage"), default=np.nan),
            "validation_windows": ["no_op_narrowness_window", "active_risk_window", "trained_bundle_replay", "broader_walk_forward"],
            "risk_notes": str(recommended_validation_row.get("overfit_risk", "")),
        },
        "guardrails": [
            "no production change",
            "no promotion claim",
            "trained-bundle validation required later",
            "broader walk-forward validation required later",
            "system remains shadow-only",
        ],
        "families_with_feature_gaps": missing_feature_families,
        "next_exact_command_or_action": next_command,
        "attribution_summary": attribution_summary,
        "scoreboard_summary": scoreboard_summary,
        "subregion_summary": subregion_summary,
        "intervention_summary": intervention_summary,
        "unknown_failure_summary": unknown_report,
        "stale_price_dependency_summary": stale_price_summary,
    }
    write_json(outputs_dir / "improvement_discovery_report.json", report_payload)

    scoreboard_table = _markdown_table(
        scoreboard.head(6),
        ["failure_mode_id", "losses", "wins", "pre_event_detectability_rate", "priority_score"],
    )
    unknown_table = _markdown_table(
        unknown_clusters.head(6),
        ["candidate_failure_mode_id", "loss_count", "win_count_nearby", "recommendation", "overfit_risk"],
    )
    subregion_table = _markdown_table(
        subregion_scoreboard.head(8),
        ["subregion_id", "parent_failure_family", "losses", "wins", "coverage_cost", "recommended_next_action"],
    )
    proposal_table = _markdown_table(
        proposals.head(6),
        ["intervention_id", "parent_failure_family", "subregion_id", "intervention_type", "recommended_next_action"],
    )
    feature_gap_table = _markdown_table(
        pd.DataFrame(feature_gap_report),
        ["feature_name", "family", "present", "blocks_discovery", "blocks_validation", "proxy_exists"],
    )
    stale_price_table = _markdown_table(
        pd.DataFrame(stale_price_summary.get("subregion_counts", [])),
        ["stale_price_subregion", "row_count", "selected_count", "resolved_count", "losses", "wins", "would_change_decision_count"],
    )
    markdown = "\n".join(
        [
            "# Non-Rebound Improvement Discovery Report",
            "",
            "## Executive Summary",
            f"- total settled picks analyzed: {report_payload['summary']['total_settled_picks_analyzed']}",
            f"- total losses analyzed: {report_payload['summary']['total_losses_analyzed']}",
            f"- total wins analyzed: {report_payload['summary']['total_wins_analyzed']}",
            f"- rebound failures excluded/downweighted: {', '.join(sorted(excluded_modes)) or 'none'}",
            f"- top non-rebound failure family: {report_payload['summary']['top_non_rebound_failure_family'] or 'none'}",
            f"- recommended next action: {report_payload['summary']['recommended_next_action']}",
            "",
            "## Failure Mode Scoreboard",
            scoreboard_table,
            "",
            "## Failure Subregion Scoreboard",
            subregion_table,
            "",
            "## Unknown Failure Clusters",
            unknown_table,
            "",
            "## Candidate Interventions",
            proposal_table,
            "",
            "## Stale Price Dependency",
            stale_price_table,
            "",
            "## Recommended First Validation",
            f"- failure mode: {report_payload['recommended_first_validation']['failure_mode_id'] or 'none'}",
            f"- subregion: {report_payload['recommended_first_validation']['subregion_id'] or 'none'}",
            f"- intervention: {report_payload['recommended_first_validation']['intervention_id'] or 'none'}",
            f"- candidate status: {report_payload['recommended_first_validation']['recommended_next_action'] or overall_status}",
            f"- expected coverage cost: {report_payload['recommended_first_validation']['expected_coverage_cost']}",
            f"- missing features: {report_payload['recommended_first_validation']['missing_features'] or 'none'}",
            "",
            "## Feature Gaps",
            feature_gap_table,
            "",
            "## Guardrails",
            _markdown_list(report_payload["guardrails"]),
            "",
            "## Next Exact Command Or Action",
            f"- {report_payload['next_exact_command_or_action']}",
        ]
    )
    (outputs_dir / "improvement_discovery_report.md").write_text(markdown, encoding="utf-8")

    run_id = str(outputs_dir.resolve().name)
    proposed_interventions = proposals.head(5).get("intervention_id", pd.Series(dtype="object")).astype(str).tolist() if not proposals.empty else []
    append_improvement_entry(
        {
            "improvement_id": run_id,
            "failure_mode_id": str(top_scoreboard_row.get("failure_mode_id", "")),
            "intervention_id": str(recommended_validation_row.get("intervention_id", "")),
            "author_or_run_id": "run_improvement_discovery",
            "hypothesis": (
                f"Discovery-only price-quality pass targeting {', '.join(sorted(target_subregion_set))}."
                if price_quality_mode
                else f"Discovery-only non-rebound failure-mode pass targeting {', '.join(target_modes)}."
            ),
            "implementation_files": [],
            "validation_windows": [],
            "metrics_before": {},
            "metrics_after": {},
            "segment_results": {},
            "promotion_status": "not_applicable",
            "blocked_reasons": ["discovery_only_not_validated"],
            "rollback_rule": "No production change was made; stop at discovery artifacts unless a later shadow validation explicitly passes.",
            "final_decision": "discovery_logged",
            "run_id": run_id,
            "mode": (
                "price_quality_discovery_only"
                if price_quality_mode
                else ("broader_walkforward_discovery_only" if broad_walk_forward else "discovery_only")
            ),
            "excluded_families": sorted(excluded_modes),
            "target_families": target_modes,
            "target_subregion": sorted(target_subregion_set)[0] if target_subregion_set else "",
            "input_paths": input_manifest,
            "output_paths": output_manifest,
            "top_failure_mode": str(top_scoreboard_row.get("failure_mode_id", "")),
            "top_actionable_subregion": str(top_actionable_subregion.get("subregion_id", "")),
            "proposed_interventions": proposed_interventions,
            "rejected_broad_signals": broad_signal_unsafe,
            "unknown_clusters": unknown_clusters.head(10).to_dict(orient="records") if not unknown_clusters.empty else [],
            "feature_gaps": feature_gap_report,
            "validation_status": "not_started",
            "next_action": next_command,
        },
        ledger_path=ledger_path,
    )
    return report_payload


def main() -> None:
    args = parse_args()
    registry = load_failure_mode_registry()
    price_quality_mode = bool(args.price_quality_rows)
    used_selected_paths: list[str] = []
    used_candidate_paths: list[str] = []
    daily_runs_dirs: list[Path] = []
    if price_quality_mode:
        selected_rows, candidate_pool_rows = _load_price_quality_rows(args.price_quality_rows)
        used_selected_paths = [str(args.price_quality_rows)]
        used_candidate_paths = [str(args.price_quality_rows)]
    else:
        selected_board_paths = _resolve_selected_board_paths(list(args.selected_board_csv), broad_walk_forward=bool(args.broad_walk_forward))
        selected_rows, used_selected_paths = _load_selected_rows(
            selected_board_paths,
            selected_variant=str(args.selected_variant),
            broad_walk_forward=bool(args.broad_walk_forward),
        )
        daily_runs_dirs = _resolve_daily_runs_dirs(list(args.daily_runs_dir), selected_board_paths, broad_walk_forward=bool(args.broad_walk_forward))
        candidate_pool_rows, used_candidate_paths = _load_candidate_pool_rows(
            selected_rows=selected_rows,
            candidate_pool_paths=list(args.candidate_pool_csv),
            daily_runs_dirs=daily_runs_dirs,
        )
    excluded_modes, excluded_markets = expand_failure_mode_exclusions(list(args.exclude_failure_family), registry=registry)
    report_payload = run_improvement_discovery(
        selected_rows=selected_rows,
        candidate_pool_rows=candidate_pool_rows,
        outputs_dir=args.output_dir,
        ledger_path=args.ledger_path,
        input_manifest={
            "selected_board_csvs": used_selected_paths,
            "candidate_pool_csvs": used_candidate_paths,
            "daily_runs_dirs": [str(path) for path in daily_runs_dirs],
            "price_quality_rows": str(args.price_quality_rows) if args.price_quality_rows else "",
            "data_proc_root": str(args.data_proc_root),
            "selected_variant": str(args.selected_variant),
            "discover_subregions": bool(args.discover_subregions),
            "broad_walk_forward": bool(args.broad_walk_forward),
        },
        data_proc_root=args.data_proc_root,
        target_failure_modes=_normalize_target_failure_modes(list(args.target_failure_family)),
        target_subregions=list(args.target_subregion),
        excluded_failure_modes=excluded_modes,
        excluded_market_families=excluded_markets,
        discover_subregions=bool(args.discover_subregions),
        broad_walk_forward=bool(args.broad_walk_forward),
        price_quality_mode=price_quality_mode,
        min_loss_count=int(args.min_loss_count),
        min_resolved_count=int(args.min_resolved_count),
        min_pre_event_detectability=float(args.min_pre_event_detectability),
        max_coverage_cost=float(args.max_coverage_cost),
        max_non_target_damage=float(args.max_non_target_damage),
        max_win_removal_rate=float(args.max_win_removal_rate),
        priority_floor=float(args.priority_floor),
        min_cluster_losses=int(args.min_cluster_losses),
        discovery_only=bool(args.discovery_only),
        shadow_only=bool(args.shadow_only),
    )
    print(json.dumps(report_payload.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
