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


DEFAULT_OUTPUT_DIR = PLAYER_PREDICTOR_ROOT.parents[1] / "validation" / "production_shadow" / "player_simulation"
STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "PRA", "PR", "PA", "RA", "MIN", "GP"]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _candidate_log_paths(data_proc_dir: Path) -> list[Path]:
    if not data_proc_dir.exists():
        return []
    return sorted(data_proc_dir.glob("*/**/*processed*.csv"))


def _normalize_logs(frame: pd.DataFrame, *, fallback_player: str = "") -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "Date" not in out.columns and "game_date" in out.columns:
        out["Date"] = out["game_date"]
    out["Date"] = pd.to_datetime(out.get("Date"), errors="coerce")
    if "Player" not in out.columns:
        out["Player"] = fallback_player
    if "MIN" not in out.columns:
        for candidate in ["MP", "minutes", "Minutes"]:
            if candidate in out.columns:
                out["MIN"] = out[candidate]
                break
    if "REB" not in out.columns and "TRB" in out.columns:
        out["REB"] = out["TRB"]
    if "3PM" not in out.columns:
        for candidate in ["FG3M", "3P", "3P Made", "FG3"]:
            if candidate in out.columns:
                out["3PM"] = out[candidate]
                break
    for stat in ["PTS", "REB", "AST", "STL", "BLK", "3PM", "MIN"]:
        if stat not in out.columns:
            out[stat] = np.nan
        out[stat] = pd.to_numeric(out[stat], errors="coerce")
    out["PRA"] = out["PTS"] + out["REB"] + out["AST"]
    out["PR"] = out["PTS"] + out["REB"]
    out["PA"] = out["PTS"] + out["AST"]
    out["RA"] = out["REB"] + out["AST"]
    return out


def load_player_logs(data_proc_dir: Path, *, cutoff_date: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = pd.to_datetime(cutoff_date, errors="coerce") if cutoff_date else pd.NaT
    frames: list[pd.DataFrame] = []
    used_paths: list[str] = []
    future_rows_removed = 0
    for path in _candidate_log_paths(data_proc_dir):
        frame = _read_csv(path)
        if frame.empty:
            continue
        fallback_player = path.parent.name.replace("_", " ")
        frame = _normalize_logs(frame, fallback_player=fallback_player)
        if frame.empty or frame["Date"].isna().all():
            continue
        if pd.notna(cutoff):
            future_rows_removed += int(frame["Date"].gt(cutoff).sum())
            frame = frame.loc[frame["Date"] <= cutoff].copy()
        if frame.empty:
            continue
        frames.append(frame)
        used_paths.append(str(path))
    logs = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    manifest = {
        "data_proc_dir": str(data_proc_dir),
        "cutoff_date": "" if pd.isna(cutoff) else cutoff.strftime("%Y-%m-%d"),
        "player_log_files_used": len(used_paths),
        "used_paths_sample": used_paths[:25],
        "future_rows_removed": int(future_rows_removed),
    }
    return logs, manifest


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ["mean", "median", "std", "p10", "p25", "p50", "p75", "p90", "floor_p10", "ceiling_p90"]}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "floor_p10": float(np.percentile(values, 10)),
        "ceiling_p90": float(np.percentile(values, 90)),
    }


def _confidence(sample_count: int, volatility: float, missing_count: int) -> str:
    if sample_count < 8:
        return "INSUFFICIENT_DATA"
    if missing_count >= 3 or volatility > 0.85:
        return "LOW_CONFIDENCE"
    if sample_count >= 35 and volatility <= 0.45 and missing_count == 0:
        return "HIGH_CONFIDENCE"
    return "MEDIUM_CONFIDENCE"


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, size: int, lower: float = 0.0) -> np.ndarray:
    draw = rng.normal(mean, max(std, 0.15), size=size)
    return np.clip(draw, lower, None)


def _simulate_player(player_logs: pd.DataFrame, *, simulation_count: int, rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs = player_logs.sort_values("Date").copy()
    player = str(logs["Player"].dropna().astype(str).iloc[-1]) if "Player" in logs.columns and logs["Player"].notna().any() else "Unknown Player"
    team = str(logs["Team"].dropna().astype(str).iloc[-1]) if "Team" in logs.columns and logs["Team"].notna().any() else ""
    player_id = str(logs["Player_ID"].dropna().astype(str).iloc[-1]) if "Player_ID" in logs.columns and logs["Player_ID"].notna().any() else ""
    sample_count = int(len(logs))
    recent = logs.tail(min(25, sample_count)).copy()
    missing_features: list[str] = []
    if logs["MIN"].notna().sum() < max(5, sample_count // 3):
        missing_features.append("minutes")
    for stat in ["PTS", "REB", "AST"]:
        if logs[stat].notna().sum() < max(5, sample_count // 3):
            missing_features.append(stat.lower())

    minutes_mean = float(recent["MIN"].dropna().mean()) if recent["MIN"].notna().any() else 18.0
    minutes_std = float(recent["MIN"].dropna().std(ddof=0)) if recent["MIN"].notna().sum() > 1 else 6.0
    games_baseline = min(82, max(1, sample_count))
    availability_rate = float(np.clip(sample_count / 82.0, 0.15, 0.98))
    games_played = rng.binomial(82, availability_rate, size=simulation_count)
    games_played = np.clip(games_played, 1, 82)
    minutes = _bounded_normal(rng, minutes_mean, minutes_std * 1.15, simulation_count)

    simulated_stats: dict[str, np.ndarray] = {"GP": games_played.astype(float), "MIN": minutes}
    stat_rows: list[dict[str, Any]] = []
    for stat in ["PTS", "REB", "AST", "STL", "BLK", "3PM"]:
        values = recent[stat].dropna().to_numpy(dtype="float64") if stat in recent.columns else np.array([])
        if values.size == 0:
            per_game = np.zeros(simulation_count)
            missing_features.append(stat.lower())
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0)) if values.size > 1 else max(1.0, mean * 0.25)
            minute_scale = np.where(minutes_mean > 0, minutes / minutes_mean, 1.0)
            role_shift = rng.normal(1.0, 0.10 if sample_count >= 25 else 0.18, size=simulation_count)
            per_game = _bounded_normal(rng, mean, std * 1.15, simulation_count) * minute_scale * role_shift
        simulated_stats[stat] = per_game
    simulated_stats["PRA"] = simulated_stats["PTS"] + simulated_stats["REB"] + simulated_stats["AST"]
    simulated_stats["PR"] = simulated_stats["PTS"] + simulated_stats["REB"]
    simulated_stats["PA"] = simulated_stats["PTS"] + simulated_stats["AST"]
    simulated_stats["RA"] = simulated_stats["REB"] + simulated_stats["AST"]

    minutes_cv = float(minutes_std / max(minutes_mean, 1.0))
    primary_vol = np.nanmean(
        [
            float(recent[stat].std(ddof=0) / max(recent[stat].mean(), 1.0))
            for stat in ["PTS", "REB", "AST"]
            if recent[stat].notna().sum() > 1 and pd.notna(recent[stat].mean())
        ]
        or [1.0]
    )
    volatility_score = float(np.clip(0.45 * minutes_cv + 0.55 * primary_vol, 0.0, 1.0))
    role_stability_score = float(np.clip(1.0 - minutes_cv, 0.0, 1.0))
    forecastability_score = float(np.clip(0.55 * role_stability_score + 0.45 * (1.0 - volatility_score), 0.0, 1.0))
    confidence = _confidence(sample_count, volatility_score, len(set(missing_features)))

    card_stats: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for stat in STATS:
        values = simulated_stats.get(stat)
        if values is None:
            continue
        q = _quantiles(values)
        last_avg = float(recent[stat].dropna().mean()) if stat in recent.columns and recent[stat].notna().any() else np.nan
        prob_above = None if not np.isfinite(last_avg) else float(np.mean(values > last_avg))
        prob_below = None if not np.isfinite(last_avg) else float(np.mean(values < last_avg))
        stat_vol = None if q["mean"] in {None, 0} else float((q["std"] or 0.0) / max(abs(q["mean"] or 0.0), 1.0))
        stat_payload = {
            "mean": q["mean"],
            "median": q["median"],
            "p10": q["p10"],
            "p25": q["p25"],
            "p50": q["p50"],
            "p75": q["p75"],
            "p90": q["p90"],
            "floor": q["floor_p10"],
            "ceiling": q["ceiling_p90"],
            "volatility": stat_vol,
            "confidence": confidence,
            "probability_above_last_season_avg": prob_above,
            "probability_below_last_season_avg": prob_below,
        }
        card_stats[stat.lower()] = stat_payload
        summary_rows.append(
            {
                "player_id": player_id,
                "player": player,
                "team": team,
                "stat": stat,
                "sample_count": sample_count,
                "simulation_count": simulation_count,
                "confidence_tier": confidence,
                **{f"{key}": value for key, value in stat_payload.items()},
            }
        )

    card = {
        "player_id": player_id,
        "player": player,
        "team": team,
        "position": str(logs["Pos"].dropna().astype(str).iloc[-1]) if "Pos" in logs.columns and logs["Pos"].notna().any() else "",
        "archetype": _infer_archetype(card_stats),
        "simulation_count": simulation_count,
        "confidence_tier": confidence,
        "forecastability_score": forecastability_score,
        "volatility_score": volatility_score,
        "role_stability_score": role_stability_score,
        "projected_games_played": card_stats["gp"]["median"],
        "projected_minutes_per_game": card_stats["min"]["median"],
        "pts": card_stats.get("pts", {}),
        "reb": card_stats.get("reb", {}),
        "ast": card_stats.get("ast", {}),
        "pra": card_stats.get("pra", {}),
        "best_projection_summary": _projection_summary(player, confidence, card_stats),
        "uncertainty_summary": _uncertainty_summary(confidence, volatility_score),
        "primary_upside_path": "Higher minutes stability and a stronger usage state push the upper p75-p90 range.",
        "primary_downside_path": "Lower availability, role compression, or shooting/usage volatility pulls outcomes toward the p10 floor.",
        "main_risk_factors": _risk_factors(volatility_score, missing_features),
        "missing_data_warnings": sorted(set(missing_features)),
        "credibility_notes": "Monte Carlo range is descriptive and uncertainty-aware; it is not a guarantee or betting recommendation.",
    }
    return card, summary_rows


def _simulate_frozen_player(row: pd.Series, *, simulation_count: int, rng: np.random.Generator) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    player = str(row.get("player", "Unknown Player"))
    player_id = str(row.get("player_id", ""))
    team = str(row.get("team_prior", ""))
    sample_count = int(pd.to_numeric(pd.Series([row.get("games_available_before_cutoff")]), errors="coerce").fillna(0).iloc[0])
    missing_features = [item for item in str(row.get("missing_feature_warnings", "")).split("|") if item]
    minutes_mean = _series_num(row, "minutes_mean", "recent_minutes_mean", default=18.0)
    minutes_std = _series_num(row, "minutes_std", "recent_minutes_std", default=6.0)
    availability_rate = float(np.clip(sample_count / 82.0, 0.15, 0.98))
    games_played = np.clip(rng.binomial(82, availability_rate, size=simulation_count), 1, 82).astype(float)
    minutes = _bounded_normal(rng, minutes_mean, minutes_std * 1.15, simulation_count)
    minute_scale = np.where(minutes_mean > 0, minutes / minutes_mean, 1.0)
    role_std = 0.10 if sample_count >= 25 else 0.18

    simulated_stats: dict[str, np.ndarray] = {"GP": games_played, "MIN": minutes}
    for stat, mean_col, std_col in [
        ("PTS", "pts_mean", "pts_std"),
        ("REB", "reb_mean", "reb_std"),
        ("AST", "ast_mean", "ast_std"),
        ("STL", "stl_mean", "stl_std"),
        ("BLK", "blk_mean", "blk_std"),
        ("3PM", "threepm_mean", "threepm_std"),
    ]:
        mean = _series_num(row, mean_col, default=np.nan)
        std = _series_num(row, std_col, default=np.nan)
        if not np.isfinite(mean):
            missing_features.append(stat.lower())
            per_game = np.zeros(simulation_count)
        else:
            std = std if np.isfinite(std) and std > 0 else max(1.0, mean * 0.25)
            role_shift = rng.normal(1.0, role_std, size=simulation_count)
            per_game = _bounded_normal(rng, mean, std * 1.15, simulation_count) * minute_scale * role_shift
        simulated_stats[stat] = per_game
    simulated_stats["PRA"] = simulated_stats["PTS"] + simulated_stats["REB"] + simulated_stats["AST"]
    simulated_stats["PR"] = simulated_stats["PTS"] + simulated_stats["REB"]
    simulated_stats["PA"] = simulated_stats["PTS"] + simulated_stats["AST"]
    simulated_stats["RA"] = simulated_stats["REB"] + simulated_stats["AST"]

    minutes_cv = float(minutes_std / max(minutes_mean, 1.0))
    primary_vol = np.nanmean(
        [
            float(np.std(simulated_stats[stat], ddof=0) / max(np.mean(simulated_stats[stat]), 1.0))
            for stat in ["PTS", "REB", "AST"]
        ]
    )
    volatility_score = float(np.clip(0.45 * minutes_cv + 0.55 * primary_vol, 0.0, 1.0))
    role_stability_score = float(np.clip(1.0 - minutes_cv, 0.0, 1.0))
    forecastability_score = float(np.clip(0.55 * role_stability_score + 0.45 * (1.0 - volatility_score), 0.0, 1.0))
    confidence = _confidence(sample_count, volatility_score, len(set(missing_features)))

    card_stats: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for stat in STATS:
        values = simulated_stats.get(stat)
        if values is None:
            continue
        q = _quantiles(values)
        prior_mean = _series_num(row, f"{stat.lower()}_mean", default=np.nan)
        prob_above = None if not np.isfinite(prior_mean) else float(np.mean(values > prior_mean))
        prob_below = None if not np.isfinite(prior_mean) else float(np.mean(values < prior_mean))
        stat_vol = None if q["mean"] in {None, 0} else float((q["std"] or 0.0) / max(abs(q["mean"] or 0.0), 1.0))
        stat_payload = {
            "mean": q["mean"],
            "median": q["median"],
            "p10": q["p10"],
            "p25": q["p25"],
            "p50": q["p50"],
            "p75": q["p75"],
            "p90": q["p90"],
            "floor": q["floor_p10"],
            "ceiling": q["ceiling_p90"],
            "volatility": stat_vol,
            "confidence": confidence,
            "probability_above_last_season_avg": prob_above,
            "probability_below_last_season_avg": prob_below,
        }
        card_stats[stat.lower()] = stat_payload
        summary_rows.append(
            {
                "player_id": player_id,
                "player": player,
                "team": team,
                "stat": stat,
                "sample_count": sample_count,
                "simulation_count": simulation_count,
                "confidence_tier": confidence,
                **stat_payload,
            }
        )

    card = {
        "player_id": player_id,
        "player": player,
        "team": team,
        "position": str(row.get("position", "")),
        "archetype": str(row.get("archetype", "")) or _infer_archetype(card_stats),
        "simulation_count": simulation_count,
        "confidence_tier": confidence,
        "forecastability_score": forecastability_score,
        "volatility_score": volatility_score,
        "role_stability_score": role_stability_score,
        "projected_games_played": card_stats["gp"]["median"],
        "projected_minutes_per_game": card_stats["min"]["median"],
        "pts": card_stats.get("pts", {}),
        "reb": card_stats.get("reb", {}),
        "ast": card_stats.get("ast", {}),
        "pra": card_stats.get("pra", {}),
        "best_projection_summary": _projection_summary(player, confidence, card_stats),
        "uncertainty_summary": _uncertainty_summary(confidence, volatility_score),
        "primary_upside_path": "Higher minutes stability and a stronger usage state push the upper p75-p90 range.",
        "primary_downside_path": "Lower availability, role compression, or shooting/usage volatility pulls outcomes toward the p10 floor.",
        "main_risk_factors": _risk_factors(volatility_score, missing_features),
        "missing_data_warnings": sorted(set(missing_features)),
        "credibility_notes": "Frozen preseason Monte Carlo range; actual outcomes are joined only after simulation for backtest evaluation.",
    }
    return card, summary_rows


def _series_num(row: pd.Series, *columns: str, default: float = 0.0) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return float(default)


def _infer_archetype(stats: dict[str, dict[str, Any]]) -> str:
    pts = float(stats.get("pts", {}).get("median") or 0.0)
    reb = float(stats.get("reb", {}).get("median") or 0.0)
    ast = float(stats.get("ast", {}).get("median") or 0.0)
    if pts >= 22 and ast >= 5:
        return "high_usage_creator"
    if reb >= 9:
        return "interior_rebounder"
    if ast >= 6:
        return "primary_facilitator"
    if pts >= 16:
        return "scoring_wing"
    return "rotation_role_player"


def _projection_summary(player: str, confidence: str, stats: dict[str, dict[str, Any]]) -> str:
    pts = stats.get("pts", {})
    return (
        f"{player} has a {confidence.lower().replace('_', '-')} projection with median PTS "
        f"{_fmt(pts.get('median'))} and p10-p90 range {_fmt(pts.get('p10'))}-{_fmt(pts.get('p90'))}."
    )


def _uncertainty_summary(confidence: str, volatility: float) -> str:
    if confidence == "INSUFFICIENT_DATA":
        return "Insufficient sample: ranges are intentionally wide and should be treated as exploratory."
    if volatility >= 0.65:
        return "High volatility: median is more reliable than ceiling."
    if volatility <= 0.35:
        return "Lower volatility: historical state is comparatively stable, but still uncertain."
    return "Moderate volatility: range should be read alongside role and availability assumptions."


def _risk_factors(volatility: float, missing: list[str]) -> list[str]:
    risks: list[str] = []
    if volatility >= 0.65:
        risks.append("high_stat_or_minutes_volatility")
    if missing:
        risks.append("missing_features_widen_uncertainty")
    if not risks:
        risks.append("normal_role_and_availability_uncertainty")
    return risks


def _fmt(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.1f}"
    except Exception:
        return "n/a"


def simulate_next_season_player_states(
    *,
    data_proc_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cutoff_date: str | None = None,
    cutoff_mode: str | None = None,
    input_frozen_sample: Path | None = None,
    actual_outcomes: Path | None = None,
    simulation_count: int = 10000,
    seed: int = 17,
    player_limit: int | None = None,
    backtest_season: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if backtest_season and cutoff_mode == "preseason" and not cutoff_date:
        cutoff_date = f"{int(backtest_season)}-10-01"
    frozen_sample_requested = bool(input_frozen_sample)
    frozen = _read_table(input_frozen_sample)
    actual_outcome_rows = _read_table(actual_outcomes)
    all_logs, all_logs_manifest = load_player_logs(data_proc_dir, cutoff_date=None) if backtest_season and actual_outcome_rows.empty else (pd.DataFrame(), {})
    logs, input_manifest = (pd.DataFrame(), {}) if frozen_sample_requested else load_player_logs(data_proc_dir, cutoff_date=cutoff_date)
    if frozen_sample_requested:
        input_manifest = {
            "input_frozen_sample": str(input_frozen_sample),
            "cutoff_date": cutoff_date or "",
            "frozen_sample_rows": int(len(frozen)),
            "eligible_frozen_sample_rows": int(frozen.get("simulation_eligible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
            if "simulation_eligible" in frozen.columns
            else int(len(frozen)),
        }
    rng = np.random.default_rng(seed)
    cards: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if frozen_sample_requested and not frozen.empty:
        eligible = frozen.copy()
        if "simulation_eligible" in eligible.columns:
            eligible = eligible.loc[eligible["simulation_eligible"].fillna(False).astype(bool)].copy()
        if player_limit:
            eligible = eligible.head(int(player_limit))
        for _, state_row in eligible.iterrows():
            card, summary_rows = _simulate_frozen_player(state_row, simulation_count=int(simulation_count), rng=rng)
            card["data_cutoff_date"] = cutoff_date or str(state_row.get("data_cutoff_date", ""))
            cards.append(card)
            rows.extend(summary_rows)
    elif not logs.empty:
        groups = list(logs.groupby(logs["Player"].fillna("Unknown Player").astype(str), dropna=False))
        if player_limit:
            groups = groups[: int(player_limit)]
        for _player, group in groups:
            if group.empty:
                continue
            card, summary_rows = _simulate_player(group, simulation_count=int(simulation_count), rng=rng)
            card["data_cutoff_date"] = cutoff_date or ""
            cards.append(card)
            rows.extend(summary_rows)

    summary = pd.DataFrame(rows)
    csv_path = output_dir / "next_season_player_simulations.csv"
    parquet_path = output_dir / "next_season_player_simulations.parquet"
    cards_path = output_dir / "player_simulation_cards.json"
    manifest_path = output_dir / "player_simulation_manifest.json"
    validation_path = output_dir / "player_simulation_validation_report.json"
    summary.to_csv(csv_path, index=False)
    try:
        summary.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path.write_text("", encoding="utf-8")
    cards_path.write_text(json.dumps(cards, indent=2), encoding="utf-8")

    validation = validate_player_simulation_cards(cards, cutoff_date=cutoff_date)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    backtest = (
        _run_backtest(
            summary,
            all_logs,
            int(backtest_season),
            output_dir=output_dir,
            cutoff_date=cutoff_date,
            cutoff_mode=cutoff_mode or "",
            simulation_count=int(simulation_count),
            actual_outcomes=actual_outcome_rows,
        )
        if backtest_season
        else {}
    )
    manifest = {
        "run_id": f"player_sim_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff_date": cutoff_date or "",
        "simulation_count": int(simulation_count),
        "seed": int(seed),
        "input_manifest": input_manifest,
        "all_logs_manifest": all_logs_manifest,
        "frozen_sample_mode": bool(frozen_sample_requested),
        "actual_outcomes_path": str(actual_outcomes) if actual_outcomes else "",
        "output_paths": {
            "parquet": str(parquet_path),
            "csv": str(csv_path),
            "cards_json": str(cards_path),
            "validation_report": str(validation_path),
        },
        "backtest": backtest,
        "production_behavior_changed": False,
        "promotion_ready": False,
        "shadow_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_player_simulation_cards(cards: list[dict[str, Any]], *, cutoff_date: str | None = None) -> dict[str, Any]:
    missing_range: list[str] = []
    missing_confidence: list[str] = []
    for card in cards:
        if not card.get("confidence_tier"):
            missing_confidence.append(str(card.get("player", "")))
        for stat in ["pts", "reb", "ast", "pra"]:
            payload = card.get(stat, {})
            if not all(key in payload and payload.get(key) is not None for key in ["p10", "p50", "p90"]):
                missing_range.append(f"{card.get('player', '')}:{stat}")
    return {
        "card_count": int(len(cards)),
        "data_cutoff_date": cutoff_date or "",
        "no_future_leakage_claim": True,
        "all_cards_include_uncertainty": not missing_range,
        "all_cards_include_confidence": not missing_confidence,
        "production_behavior_changed": False,
        "promotion_claim": False,
        "staking_field_enabled": False,
        "validation_passed": not missing_range and not missing_confidence,
        "issues": {
            "missing_range": missing_range[:50],
            "missing_confidence": missing_confidence[:50],
        },
    }


def _season_bounds(season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(season), month=10, day=1)
    end = pd.Timestamp(year=int(season) + 1, month=10, day=1)
    return start, end


def _serializable_rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return float(mask.mean())


def _run_backtest(
    summary: pd.DataFrame,
    logs: pd.DataFrame,
    season: int,
    *,
    output_dir: Path,
    cutoff_date: str | None,
    cutoff_mode: str,
    simulation_count: int,
    actual_outcomes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    rows_path = output_dir / f"simulation_backtest_{season}_preseason_rows.csv"
    report_json_path = output_dir / f"simulation_backtest_{season}_preseason_report.json"
    report_md_path = output_dir / f"simulation_backtest_{season}_preseason_report.md"

    start, end = _season_bounds(int(season))
    actual_outcomes = actual_outcomes if actual_outcomes is not None else pd.DataFrame()
    actuals = logs.copy()
    if not actual_outcomes.empty:
        actuals = actual_outcomes.copy()
    elif not actuals.empty:
        actuals["Date"] = pd.to_datetime(actuals.get("Date"), errors="coerce")
        actuals = actuals.loc[actuals["Date"].ge(start) & actuals["Date"].lt(end)].copy()

    records: list[dict[str, Any]] = []
    status = "BACKTEST_EVALUATED"
    blockers: list[str] = []
    if summary.empty:
        status = "BACKTEST_REQUIRED"
        blockers.append("no_preseason_simulation_rows_available")
    if actuals.empty:
        status = "BACKTEST_REQUIRED"
        blockers.append("no_actual_rows_for_backtest_season")

    if not summary.empty and not actuals.empty:
        for _, row in summary.iterrows():
            player = str(row.get("player", ""))
            stat = str(row.get("stat", "")).upper()
            actual_column = {"MIN": "actual_mpg", "GP": "actual_games_played"}.get(stat, f"actual_{stat.lower()}")
            player_column = "player" if "player" in actuals.columns else "Player" if "Player" in actuals.columns else ""
            if not player or not player_column:
                continue
            if actual_column in actuals.columns:
                matched = actuals.loc[actuals[player_column].astype(str).eq(player), actual_column]
            elif stat in actuals.columns:
                matched = actuals.loc[actuals[player_column].astype(str).eq(player), stat]
            else:
                continue
            player_actuals = pd.to_numeric(matched, errors="coerce").dropna()
            if player_actuals.empty:
                continue
            actual_mean = float(player_actuals.mean())
            p10 = pd.to_numeric(pd.Series([row.get("p10")]), errors="coerce").iloc[0]
            p25 = pd.to_numeric(pd.Series([row.get("p25")]), errors="coerce").iloc[0]
            p50 = pd.to_numeric(pd.Series([row.get("p50")]), errors="coerce").iloc[0]
            p75 = pd.to_numeric(pd.Series([row.get("p75")]), errors="coerce").iloc[0]
            p90 = pd.to_numeric(pd.Series([row.get("p90")]), errors="coerce").iloc[0]
            if pd.isna(p10) or pd.isna(p50) or pd.isna(p90):
                continue
            records.append(
                {
                    "player": player,
                    "stat": stat,
                    "confidence_tier": row.get("confidence_tier", ""),
                    "sample_count": row.get("sample_count"),
                    "actual_mean": actual_mean,
                    "p10": float(p10),
                    "p25": None if pd.isna(p25) else float(p25),
                    "p50": float(p50),
                    "p75": None if pd.isna(p75) else float(p75),
                    "p90": float(p90),
                    "within_p10_p90": bool(float(p10) <= actual_mean <= float(p90)),
                    "within_p25_p75": bool(pd.notna(p25) and pd.notna(p75) and float(p25) <= actual_mean <= float(p75)),
                    "absolute_error": abs(float(p50) - actual_mean),
                    "interval_miss_distance": max(float(p10) - actual_mean, actual_mean - float(p90), 0.0),
                }
            )

    result_rows = pd.DataFrame(records)
    result_rows.to_csv(rows_path, index=False)
    if result_rows.empty and not blockers:
        status = "BACKTEST_REQUIRED"
        blockers.append("no_joined_projection_actual_rows")

    calibration_by_stat: dict[str, Any] = {}
    confidence_tier_reliability: dict[str, Any] = {}
    if not result_rows.empty:
        for stat, group in result_rows.groupby("stat", dropna=False):
            calibration_by_stat[str(stat)] = {
                "rows": int(len(group)),
                "actual_within_p10_p90_rate": _serializable_rate(group["within_p10_p90"].astype(bool)),
                "actual_within_p25_p75_rate": _serializable_rate(group["within_p25_p75"].astype(bool)),
                "median_absolute_error": float(group["absolute_error"].median()),
                "mean_absolute_error": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(group["absolute_error"])))),
                "undercoverage_rate": float((~group["within_p10_p90"].astype(bool)).mean()),
            }
        for tier, group in result_rows.groupby("confidence_tier", dropna=False):
            confidence_tier_reliability[str(tier)] = {
                "rows": int(len(group)),
                "p10_p90_coverage": _serializable_rate(group["within_p10_p90"].astype(bool)),
                "median_absolute_error": float(group["absolute_error"].median()),
            }

    coverage = _serializable_rate(result_rows["within_p10_p90"].astype(bool)) if not result_rows.empty else None
    iqr_coverage = _serializable_rate(result_rows["within_p25_p75"].astype(bool)) if not result_rows.empty else None
    undercoverage_rate = None if coverage is None else float(1.0 - coverage)
    overconfidence_rate = None if undercoverage_rate is None else max(0.0, undercoverage_rate - 0.20)
    worst_misses = []
    if not result_rows.empty:
        worst_misses = (
            result_rows.sort_values("interval_miss_distance", ascending=False)
            .head(25)
            .to_dict(orient="records")
        )

    report = {
        "status": status,
        "season": int(season),
        "cutoff_mode": cutoff_mode,
        "cutoff_date": cutoff_date,
        "simulation_count": int(simulation_count),
        "projection_rows": int(len(summary)),
        "actual_rows_available": int(len(actuals)),
        "joined_rows": int(len(result_rows)),
        "actual_within_p10_p90_rate": coverage,
        "actual_within_p25_p75_rate": iqr_coverage,
        "median_absolute_error_by_stat": {
            stat: payload.get("median_absolute_error") for stat, payload in calibration_by_stat.items()
        },
        "mean_absolute_error_by_stat": {
            stat: payload.get("mean_absolute_error") for stat, payload in calibration_by_stat.items()
        },
        "rmse_by_stat": {
            stat: payload.get("rmse") for stat, payload in calibration_by_stat.items()
        },
        "calibration_by_stat": calibration_by_stat,
        "undercoverage_rate": undercoverage_rate,
        "overconfidence_rate": overconfidence_rate,
        "confidence_tier_reliability": confidence_tier_reliability,
        "volatility_tier_reliability": {},
        "player_archetype_error": {},
        "missing_data_impact": {
            "blockers": blockers,
            "note": "Frozen preseason calibration requires pre-cutoff player-state rows and settled in-season actuals.",
        },
        "worst_miss_examples": worst_misses,
        "credible_calibration_claim_allowed": bool(status == "BACKTEST_EVALUATED" and coverage is not None and coverage >= 0.72),
    }
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(_format_backtest_markdown(report), encoding="utf-8")
    return {
        **report,
        "output_paths": {
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
            "rows_csv": str(rows_path),
        },
    }


def _format_backtest_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Simulation Backtest {report.get('season')} Preseason",
        "",
        f"- status: {report.get('status')}",
        f"- cutoff_mode: {report.get('cutoff_mode')}",
        f"- cutoff_date: {report.get('cutoff_date')}",
        f"- projection_rows: {report.get('projection_rows')}",
        f"- actual_rows_available: {report.get('actual_rows_available')}",
        f"- joined_rows: {report.get('joined_rows')}",
        f"- p10_p90_coverage: {report.get('actual_within_p10_p90_rate')}",
        f"- p25_p75_coverage: {report.get('actual_within_p25_p75_rate')}",
        f"- undercoverage_rate: {report.get('undercoverage_rate')}",
        f"- overconfidence_rate: {report.get('overconfidence_rate')}",
        "",
        "## Credibility",
        "",
        "Simulation cards remain research projections until frozen preseason coverage and confidence-tier reliability pass.",
    ]
    blockers = report.get("missing_data_impact", {}).get("blockers", [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate next-season NBA player state distributions.")
    parser.add_argument("--data-proc-dir", type=Path, default=PLAYER_PREDICTOR_ROOT / "Data-Proc")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date")
    parser.add_argument("--cutoff-mode", choices=["preseason", "explicit"], default=None)
    parser.add_argument("--input-frozen-sample", type=Path)
    parser.add_argument("--actual-outcomes", type=Path)
    parser.add_argument("--simulation-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--player-limit", type=int)
    parser.add_argument("--backtest-season", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = simulate_next_season_player_states(
        data_proc_dir=args.data_proc_dir,
        output_dir=args.output_dir,
        cutoff_date=args.cutoff_date,
        cutoff_mode=args.cutoff_mode,
        input_frozen_sample=args.input_frozen_sample,
        actual_outcomes=args.actual_outcomes,
        simulation_count=args.simulation_count,
        seed=args.seed,
        player_limit=args.player_limit,
        backtest_season=args.backtest_season,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
