#!/usr/bin/env python3
"""
Build an upcoming NBA market slate by pairing future market lines with the current
production predictor.

This intentionally does not merge future market rows into historical training data.
Instead it:
- loads a normalized market snapshot (wide format)
- finds each player's processed history
- runs inference on the history only
- writes a slate table with prediction, market, and model-vs-market edge columns
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference"))

from research.market_quality.event_time_resolver import resolve_event_times  # noqa: E402
from research.market_quality.price_provenance_schema import derive_snapshot_id, load_market_snapshot_manifest  # noqa: E402


DATA_DIR = REPO_ROOT / "Data-Proc"
MODEL_DIR = REPO_ROOT / "model"
DEFAULT_MARKET_WIDE = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "latest_player_props_wide.parquet"
DEFAULT_TARGET_PREDICTION_CALIBRATOR = REPO_ROOT / "model" / "analysis" / "calibration" / "short_term_target_prediction_calibrator.json"
TARGETS = ["PTS", "TRB", "AST"]
REBOUND_ENV_COLUMNS = [
    "Date",
    "TRB",
    "AST",
    "MP",
    "FGA",
    "FG%",
    "FTA",
    "FT%",
    "Did_Not_Play",
    "MATCHUP",
    "Opponent",
]
REBOUND_RECENT_WINDOW = 12

# Covers feed uses first-initial names (e.g., J_Brunson); these disambiguate
# rare collisions where multiple players share the same initial + last name.
AMBIGUOUS_PLAYER_OVERRIDES = {
    "M_Bridges": "Mikal_Bridges",
}


def _make_structured_stack_inference(*, model_dir: str, manifest_path: Path):
    from structured_stack_inference import StructuredStackInference

    return StructuredStackInference(model_dir=model_dir, manifest_path=manifest_path)


def build_heuristic_explanation(history_df: pd.DataFrame, failure_reason: str | None = None) -> dict:
    active = history_df.copy()
    if "Did_Not_Play" in active.columns:
        active = active.loc[pd.to_numeric(active["Did_Not_Play"], errors="coerce").fillna(0.0) < 0.5].copy()
    if active.empty:
        active = history_df.copy()

    predicted: dict[str, float] = {}
    baseline: dict[str, float] = {}
    target_factors: dict[str, dict] = {}
    sigma_values: list[float] = []

    for target in TARGETS:
        values = pd.to_numeric(active.get(target), errors="coerce").dropna()
        if values.empty:
            values = pd.to_numeric(history_df.get(target), errors="coerce").dropna()

        base_col = f"{target}_rolling_avg"
        baseline_series = pd.to_numeric(history_df.get(base_col), errors="coerce").dropna()
        baseline_value = float(baseline_series.iloc[-1]) if not baseline_series.empty else float(values.mean()) if not values.empty else 0.0

        if values.empty:
            pred_value = max(0.0, baseline_value)
            sigma = 0.0
            spike_prob = 0.10
        else:
            recent = values.tail(12)
            weights = np.linspace(1.0, 2.2, len(recent))
            recency_mean = float(np.average(recent.to_numpy(dtype=float), weights=weights))
            season_mean = float(values.mean())
            trend = float(recent.tail(min(3, len(recent))).mean() - recent.head(min(3, len(recent))).mean())

            pred_value = 0.55 * recency_mean + 0.30 * season_mean + 0.15 * (baseline_value + 0.35 * trend)
            pred_value = float(max(0.0, pred_value))
            sigma = float(np.std(recent.to_numpy(dtype=float), ddof=0)) if len(recent) > 1 else 0.0
            if len(recent) > 1:
                recent_std = float(np.std(recent.to_numpy(dtype=float), ddof=0)) + 1e-6
                z_score = float((recent.iloc[-1] - recent.mean()) / recent_std)
                spike_prob = float(np.clip(0.50 + 0.20 * z_score, 0.05, 0.95))
            else:
                spike_prob = 0.10

        predicted[target] = pred_value
        baseline[target] = float(max(0.0, baseline_value))
        sigma_values.append(sigma)
        target_factors[target] = {
            "uncertainty_sigma": sigma,
            "spike_probability": spike_prob,
        }

    avg_prediction = float(np.mean(list(predicted.values()))) if predicted else 0.0
    avg_sigma = float(np.mean(sigma_values)) if sigma_values else 0.0
    sigma_ratio = avg_sigma / max(1.0, avg_prediction)
    belief_uncertainty = float(np.clip(0.20 + 0.80 * sigma_ratio, 0.05, 0.95))
    mp_series = pd.to_numeric(active.get("MP"), errors="coerce").dropna()
    if mp_series.empty:
        feasibility = 0.70
    else:
        feasibility = float(np.clip(mp_series.tail(10).mean() / 34.0, 0.25, 0.98))

    fallback_reasons = ["heuristic_player_history"]
    if failure_reason:
        fallback_reasons.append(f"model_error:{failure_reason}")

    return {
        "predicted": predicted,
        "baseline": baseline,
        "data_quality": {
            "fallback_blend": 1.0,
            "fallback_reasons": fallback_reasons,
        },
        "latent_environment": {
            "belief_uncertainty": belief_uncertainty,
            "feasibility": feasibility,
            "role_shift_risk": 0.35,
            "volatility_regime_risk": float(np.clip(sigma_ratio, 0.05, 0.95)),
            "context_pressure_risk": 0.30,
        },
        "target_factors": target_factors,
    }


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def team_abbr_from_matchup(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"\s*([A-Z]{2,3})\b", str(value))
    return match.group(1) if match else None


def latest_player_id(history_df: pd.DataFrame) -> float:
    for column in ["Player_ID", "PLAYER_ID", "player_id"]:
        if column not in history_df.columns:
            continue
        numeric = pd.to_numeric(history_df[column], errors="coerce").dropna()
        if not numeric.empty:
            return float(numeric.iloc[-1])
    return np.nan


def _coerce_rate(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not out.dropna().empty and float((out.dropna() > 1.0).mean()) >= 0.50:
        out = out / 100.0
    return out.clip(lower=0.0, upper=1.0)


def _weighted_recent_mean(values: pd.Series, window: int = REBOUND_RECENT_WINDOW) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if cleaned.empty:
        return np.nan
    tail = cleaned.tail(min(int(window), len(cleaned)))
    weights = np.linspace(1.0, 2.0, len(tail), dtype="float64")
    return float(np.average(tail.to_numpy(dtype="float64"), weights=weights))


def _score_from_scale(value: float, lower: float, upper: float, default: float = 0.50) -> float:
    if not np.isfinite(value) or not np.isfinite(lower) or not np.isfinite(upper):
        return float(default)
    span = float(upper) - float(lower)
    if span <= 1e-9:
        return float(default)
    return float(np.clip((float(value) - float(lower)) / span, 0.0, 1.0))


@lru_cache(maxsize=4)
def load_rebound_environment_context(season: int) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for player_dir in DATA_DIR.iterdir():
        if not player_dir.is_dir():
            continue
        csv_path = player_dir / f"{season}_processed_processed.csv"
        if not csv_path.exists():
            continue
        try:
            frame = pd.read_csv(csv_path, usecols=lambda name: name in REBOUND_ENV_COLUMNS)
        except ValueError:
            frame = pd.read_csv(csv_path)
            keep = [name for name in REBOUND_ENV_COLUMNS if name in frame.columns]
            if not keep:
                continue
            frame = frame[keep].copy()
        if frame.empty or "Date" not in frame.columns or "MATCHUP" not in frame.columns:
            continue
        frame["source_player"] = player_dir.name
        frames.append(frame)

    if not frames:
        return {
            "team_games": pd.DataFrame(),
            "team_recent_lookup": {},
            "missed_fga_scale": (np.nan, np.nan),
            "rebound_event_scale": (np.nan, np.nan),
            "team_leakage_default": 0.50,
        }

    env = pd.concat(frames, ignore_index=True, sort=False)
    env["Date"] = pd.to_datetime(env["Date"], errors="coerce")
    env["team_abbr"] = env.get("MATCHUP", pd.Series("", index=env.index)).map(team_abbr_from_matchup)
    env["opponent_abbr"] = (
        env.get("Opponent", pd.Series("", index=env.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )
    did_not_play = pd.to_numeric(env.get("Did_Not_Play"), errors="coerce").fillna(0.0)
    env = env.loc[env["Date"].notna() & env["team_abbr"].notna() & (did_not_play < 0.5)].copy()
    if env.empty:
        return {
            "team_games": pd.DataFrame(),
            "team_recent_lookup": {},
            "missed_fga_scale": (np.nan, np.nan),
            "rebound_event_scale": (np.nan, np.nan),
            "team_leakage_default": 0.50,
        }

    env["TRB"] = pd.to_numeric(env.get("TRB"), errors="coerce").fillna(0.0).clip(lower=0.0)
    env["AST"] = pd.to_numeric(env.get("AST"), errors="coerce").fillna(0.0).clip(lower=0.0)
    env["FGA"] = pd.to_numeric(env.get("FGA"), errors="coerce").fillna(0.0).clip(lower=0.0)
    env["FTA"] = pd.to_numeric(env.get("FTA"), errors="coerce").fillna(0.0).clip(lower=0.0)
    env["FG%"] = _coerce_rate(env.get("FG%", pd.Series(np.nan, index=env.index)))
    env["FT%"] = _coerce_rate(env.get("FT%", pd.Series(np.nan, index=env.index)))
    env["MP"] = pd.to_numeric(env.get("MP"), errors="coerce").fillna(0.0).clip(lower=0.0)
    env["missed_fga"] = (env["FGA"] * (1.0 - env["FG%"])).clip(lower=0.0)
    env["missed_fta"] = (env["FTA"] * (1.0 - env["FT%"])).clip(lower=0.0)
    env["rebound_events_proxy"] = (env["missed_fga"] + 0.44 * env["missed_fta"]).clip(lower=0.0)

    team_games = (
        env.groupby(["Date", "team_abbr"], as_index=False)
        .agg(
            team_total_trb=("TRB", "sum"),
            team_total_ast=("AST", "sum"),
            team_total_minutes=("MP", "sum"),
            team_total_missed_fga=("missed_fga", "sum"),
            team_total_missed_fta=("missed_fta", "sum"),
            team_total_rebound_events=("rebound_events_proxy", "sum"),
            team_fg_pct=("FG%", "mean"),
        )
        .sort_values(["team_abbr", "Date"])
        .reset_index(drop=True)
    )
    opponent_map = (
        env.groupby(["Date", "team_abbr"])["opponent_abbr"]
        .agg(lambda values: next((str(item) for item in values if str(item).strip()), ""))
        .reset_index(name="opponent_abbr")
    )
    team_games = team_games.merge(opponent_map, on=["Date", "team_abbr"], how="left")

    player_team_games = env[["Date", "team_abbr", "source_player", "TRB"]].copy()
    player_team_games = player_team_games.merge(
        team_games[["Date", "team_abbr", "team_total_trb"]],
        on=["Date", "team_abbr"],
        how="left",
    )
    player_team_games["rebound_share"] = np.where(
        player_team_games["team_total_trb"] > 0.0,
        player_team_games["TRB"] / player_team_games["team_total_trb"],
        np.nan,
    )

    leakage_rows: list[dict[str, Any]] = []
    for (date_value, team_abbr), group in player_team_games.groupby(["Date", "team_abbr"], sort=False):
        shares = pd.to_numeric(group["rebound_share"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        shares = np.sort(shares)[::-1]
        top_two_share = float(shares[:2].sum()) if shares.size else 0.0
        leakage_rows.append(
            {
                "Date": date_value,
                "team_abbr": team_abbr,
                "wing_rebound_leakage_score": float(np.clip(1.0 - top_two_share, 0.0, 1.0)),
            }
        )
    leakage_df = pd.DataFrame.from_records(leakage_rows)
    if not leakage_df.empty:
        team_games = team_games.merge(leakage_df, on=["Date", "team_abbr"], how="left")
    else:
        team_games["wing_rebound_leakage_score"] = 0.50
    team_games["wing_rebound_leakage_score"] = pd.to_numeric(
        team_games.get("wing_rebound_leakage_score"),
        errors="coerce",
    ).fillna(0.50)

    recent_rows: list[dict[str, Any]] = []
    for team_abbr, group in team_games.groupby("team_abbr", sort=False):
        ordered = group.sort_values("Date")
        recent_rows.append(
            {
                "team_abbr": team_abbr,
                "projected_missed_fga": _weighted_recent_mean(ordered["team_total_missed_fga"]),
                "projected_missed_fta": _weighted_recent_mean(ordered["team_total_missed_fta"]),
                "projected_rebound_events": _weighted_recent_mean(ordered["team_total_rebound_events"]),
                "projected_wing_rebound_leakage": _weighted_recent_mean(ordered["wing_rebound_leakage_score"]),
                "projected_fg_pct": _weighted_recent_mean(ordered["team_fg_pct"]),
            }
        )
    recent_df = pd.DataFrame.from_records(recent_rows)
    if recent_df.empty:
        return {
            "team_games": team_games,
            "team_recent_lookup": {},
            "missed_fga_scale": (np.nan, np.nan),
            "rebound_event_scale": (np.nan, np.nan),
            "team_leakage_default": 0.50,
        }

    missed_low = float(recent_df["projected_missed_fga"].quantile(0.15))
    missed_high = float(recent_df["projected_missed_fga"].quantile(0.85))
    rebound_low = float(recent_df["projected_rebound_events"].quantile(0.15))
    rebound_high = float(recent_df["projected_rebound_events"].quantile(0.85))
    leakage_default = float(
        pd.to_numeric(recent_df.get("projected_wing_rebound_leakage"), errors="coerce").fillna(0.50).median()
    )
    return {
        "team_games": team_games,
        "team_recent_lookup": recent_df.set_index("team_abbr").to_dict(orient="index"),
        "missed_fga_scale": (missed_low, missed_high),
        "rebound_event_scale": (rebound_low, rebound_high),
        "team_leakage_default": leakage_default,
    }


def build_rebound_diagnostics(
    history_df: pd.DataFrame,
    env_context: dict[str, Any],
    market_home_team: str | None,
    market_away_team: str | None,
) -> dict[str, Any]:
    default = {
        "projected_team_missed_fga": np.nan,
        "projected_opponent_missed_fga": np.nan,
        "projected_team_missed_fta": np.nan,
        "projected_opponent_missed_fta": np.nan,
        "projected_missed_fga_total": np.nan,
        "projected_missed_fta_total": np.nan,
        "projected_available_rebound_events": np.nan,
        "expected_rebound_chances": np.nan,
        "team_rebound_pool_size": np.nan,
        "pace_rebound_environment": 0.50,
        "long_rebound_profile": 0.50,
        "free_throw_rebound_suppression": 0.0,
        "rebound_supply_score": 0.50,
        "rebound_share_estimate": np.nan,
        "rebound_share_stability": 0.50,
        "rebound_share_stability_score": 0.50,
        "player_team_rebound_share_recent": np.nan,
        "player_rebound_share_std": np.nan,
        "teammate_rebound_competition": 0.50,
        "teammate_rebound_competition_score": 0.50,
        "center_rebound_share_pressure": 0.50,
        "frontcourt_rebound_overlap_score": 0.50,
        "team_shooting_efficiency_stress": 0.50,
        "opponent_shooting_efficiency_stress": 0.50,
        "projected_team_fg_pct": np.nan,
        "projected_opponent_fg_pct": np.nan,
        "wing_rebound_leakage_score": float(env_context.get("team_leakage_default", 0.50)),
        "recent_games_count": 0,
        "trb_median_recent": np.nan,
        "trb_q75_recent": np.nan,
        "trb_q90_recent": np.nan,
        "minutes_floor_recent": np.nan,
        "minutes_p25_recent": np.nan,
        "minutes_median_recent": np.nan,
        "minutes_range_recent": np.nan,
        "expected_minutes_band_low": np.nan,
        "expected_minutes_band_high": np.nan,
        "expected_minutes_band_width": np.nan,
        "bench_role_flag": False,
        "starter_status_recent": np.nan,
        "starter_status_change_count": 0,
        "rotation_volatility_score": 0.50,
        "blowout_minutes_sensitivity": 0.50,
        "foul_rate_minutes_loss_risk": np.nan,
        "coach_trust_score": np.nan,
    }

    if history_df.empty:
        return default

    active = history_df.copy()
    if "Date" in active.columns:
        active["Date"] = pd.to_datetime(active["Date"], errors="coerce")
        active = active.loc[active["Date"].notna()].copy()
    if "Did_Not_Play" in active.columns:
        active = active.loc[pd.to_numeric(active["Did_Not_Play"], errors="coerce").fillna(0.0) < 0.5].copy()
    if active.empty:
        return default

    active["team_abbr"] = active.get("MATCHUP", pd.Series("", index=active.index)).map(team_abbr_from_matchup)
    active["TRB"] = pd.to_numeric(active.get("TRB"), errors="coerce")
    active = active.loc[active["team_abbr"].notna()].copy()
    if active.empty:
        return default

    latest_row = active.sort_values("Date").iloc[-1]
    player_team = latest_row.get("team_abbr")
    opponent_team = str(latest_row.get("Opponent", "")).upper().strip() if pd.notna(latest_row.get("Opponent")) else ""
    if not player_team:
        market_teams = [str(team).upper().strip() for team in [market_home_team, market_away_team] if team]
        if len(market_teams) == 2 and opponent_team in market_teams:
            player_team = market_teams[0] if market_teams[1] == opponent_team else market_teams[1]
    if not opponent_team:
        market_teams = [str(team).upper().strip() for team in [market_home_team, market_away_team] if team]
        if len(market_teams) == 2 and player_team in market_teams:
            opponent_team = market_teams[0] if market_teams[1] == player_team else market_teams[1]

    trb_values = pd.to_numeric(active["TRB"], errors="coerce").dropna()
    if not trb_values.empty:
        recent_trb = trb_values.tail(min(REBOUND_RECENT_WINDOW, len(trb_values)))
        default["recent_games_count"] = int(len(recent_trb))
        default["trb_median_recent"] = float(recent_trb.quantile(0.50))
        default["trb_q75_recent"] = float(recent_trb.quantile(0.75))
        default["trb_q90_recent"] = float(recent_trb.quantile(0.90))

    minutes_values = pd.to_numeric(active.get("MP"), errors="coerce").dropna()
    if not minutes_values.empty:
        recent_minutes = minutes_values.tail(min(REBOUND_RECENT_WINDOW, len(minutes_values)))
        minutes_floor = float(recent_minutes.min())
        minutes_p25 = float(recent_minutes.quantile(0.25))
        minutes_median = float(recent_minutes.quantile(0.50))
        minutes_p75 = float(recent_minutes.quantile(0.75))
        minutes_range = float(recent_minutes.max() - recent_minutes.min())
        starter_like = (recent_minutes >= 24.0).astype(int)
        starter_changes = int((starter_like.diff().fillna(0).abs() > 0).sum())
        starter_recent = float(starter_like.tail(min(5, len(starter_like))).mean()) if len(starter_like) else np.nan
        change_rate = float(starter_changes / max(len(recent_minutes) - 1, 1))
        rotation_volatility = float(
            np.clip(
                0.40 * (max(0.0, minutes_p75 - minutes_p25) / 10.0)
                + 0.35 * (minutes_range / 18.0)
                + 0.25 * change_rate,
                0.0,
                1.0,
            )
        )
        blowout_sensitivity = float(np.clip(max(0.0, minutes_median - minutes_floor) / max(10.0, minutes_median), 0.0, 1.0))
        low_minutes_share = float((recent_minutes <= max(16.0, minutes_median - 8.0)).mean())
        coach_trust = float(np.clip((minutes_median / 34.0) * (1.0 - 0.45 * rotation_volatility), 0.0, 1.0))
        default.update(
            {
                "minutes_floor_recent": minutes_floor,
                "minutes_p25_recent": minutes_p25,
                "minutes_median_recent": minutes_median,
                "minutes_range_recent": minutes_range,
                "expected_minutes_band_low": minutes_p25,
                "expected_minutes_band_high": minutes_p75,
                "expected_minutes_band_width": float(max(0.0, minutes_p75 - minutes_p25)),
                "bench_role_flag": bool(minutes_median < 24.0 or minutes_p75 < 26.0),
                "starter_status_recent": starter_recent,
                "starter_status_change_count": starter_changes,
                "rotation_volatility_score": rotation_volatility,
                "blowout_minutes_sensitivity": blowout_sensitivity,
                "foul_rate_minutes_loss_risk": float(np.clip(0.60 * low_minutes_share + 0.40 * rotation_volatility, 0.0, 1.0)),
                "coach_trust_score": coach_trust,
            }
        )

    team_games = env_context.get("team_games")
    if isinstance(team_games, pd.DataFrame) and not team_games.empty:
        merged = active.merge(
            team_games[["Date", "team_abbr", "team_total_trb"]],
            on=["Date", "team_abbr"],
            how="left",
        )
        merged["rebound_share"] = np.where(
            pd.to_numeric(merged["team_total_trb"], errors="coerce").fillna(0.0) > 0.0,
            pd.to_numeric(merged["TRB"], errors="coerce").fillna(0.0) / pd.to_numeric(merged["team_total_trb"], errors="coerce").fillna(np.nan),
            np.nan,
        )
        rebound_share = pd.to_numeric(merged.get("rebound_share"), errors="coerce").dropna()
        if not rebound_share.empty:
            share_recent = rebound_share.tail(min(REBOUND_RECENT_WINDOW, len(rebound_share)))
            share_estimate = _weighted_recent_mean(share_recent)
            share_std = float(share_recent.std(ddof=0)) if len(share_recent) > 1 else 0.0
            default["rebound_share_estimate"] = share_estimate
            default["rebound_share_stability"] = float(np.clip(1.0 - (share_std / 0.12), 0.0, 1.0))
            default["rebound_share_stability_score"] = default["rebound_share_stability"]
            default["player_team_rebound_share_recent"] = share_estimate
            default["player_rebound_share_std"] = share_std
            if np.isfinite(share_estimate):
                default["teammate_rebound_competition"] = float(np.clip(1.0 - share_estimate, 0.0, 1.0))
                default["teammate_rebound_competition_score"] = default["teammate_rebound_competition"]

    team_recent_lookup = env_context.get("team_recent_lookup", {})
    team_projection = team_recent_lookup.get(player_team, {}) if isinstance(team_recent_lookup, dict) else {}
    opponent_projection = team_recent_lookup.get(opponent_team, {}) if isinstance(team_recent_lookup, dict) else {}

    projected_team_missed = float(team_projection.get("projected_missed_fga", np.nan))
    projected_opp_missed = float(opponent_projection.get("projected_missed_fga", np.nan))
    projected_team_missed_fta = float(team_projection.get("projected_missed_fta", np.nan))
    projected_opp_missed_fta = float(opponent_projection.get("projected_missed_fta", np.nan))
    projected_team_events = float(team_projection.get("projected_rebound_events", np.nan))
    projected_opp_events = float(opponent_projection.get("projected_rebound_events", np.nan))
    if np.isfinite(projected_team_events) and np.isfinite(projected_opp_events):
        default["projected_available_rebound_events"] = float(projected_team_events + projected_opp_events)
    elif np.isfinite(projected_team_events):
        default["projected_available_rebound_events"] = projected_team_events
    elif np.isfinite(projected_opp_events):
        default["projected_available_rebound_events"] = projected_opp_events

    default["projected_team_missed_fga"] = projected_team_missed
    default["projected_opponent_missed_fga"] = projected_opp_missed
    default["projected_team_missed_fta"] = projected_team_missed_fta
    default["projected_opponent_missed_fta"] = projected_opp_missed_fta
    default["projected_team_fg_pct"] = float(team_projection.get("projected_fg_pct", np.nan))
    default["projected_opponent_fg_pct"] = float(opponent_projection.get("projected_fg_pct", np.nan))

    if np.isfinite(projected_team_missed) or np.isfinite(projected_opp_missed):
        default["projected_missed_fga_total"] = float(
            np.nansum([projected_team_missed if np.isfinite(projected_team_missed) else np.nan, projected_opp_missed if np.isfinite(projected_opp_missed) else np.nan])
        )
    if np.isfinite(projected_team_missed_fta) or np.isfinite(projected_opp_missed_fta):
        default["projected_missed_fta_total"] = float(
            np.nansum(
                [
                    projected_team_missed_fta if np.isfinite(projected_team_missed_fta) else np.nan,
                    projected_opp_missed_fta if np.isfinite(projected_opp_missed_fta) else np.nan,
                ]
            )
        )

    missed_low, missed_high = env_context.get("missed_fga_scale", (np.nan, np.nan))
    rebound_low, rebound_high = env_context.get("rebound_event_scale", (np.nan, np.nan))
    team_miss_score = _score_from_scale(projected_team_missed, missed_low, missed_high)
    opp_miss_score = _score_from_scale(projected_opp_missed, missed_low, missed_high)
    default["team_shooting_efficiency_stress"] = float(np.clip(1.0 - team_miss_score, 0.0, 1.0))
    default["opponent_shooting_efficiency_stress"] = float(np.clip(1.0 - opp_miss_score, 0.0, 1.0))

    if np.isfinite(default["projected_available_rebound_events"]):
        default["rebound_supply_score"] = _score_from_scale(
            float(default["projected_available_rebound_events"]) / 2.0,
            rebound_low,
            rebound_high,
        )
    else:
        default["rebound_supply_score"] = float(np.clip(0.50 * team_miss_score + 0.50 * opp_miss_score, 0.0, 1.0))
    default["expected_rebound_chances"] = default["projected_available_rebound_events"]
    default["pace_rebound_environment"] = default["rebound_supply_score"]

    leakage_default = float(env_context.get("team_leakage_default", 0.50))
    default["wing_rebound_leakage_score"] = float(
        pd.to_numeric(pd.Series([team_projection.get("projected_wing_rebound_leakage", leakage_default)]), errors="coerce")
        .fillna(leakage_default)
        .iloc[0]
    )
    opponent_leakage = float(
        pd.to_numeric(pd.Series([opponent_projection.get("projected_wing_rebound_leakage", leakage_default)]), errors="coerce")
        .fillna(leakage_default)
        .iloc[0]
    )
    default["long_rebound_profile"] = float(np.clip(0.55 * default["wing_rebound_leakage_score"] + 0.45 * opponent_leakage, 0.0, 1.0))
    default["center_rebound_share_pressure"] = float(
        np.clip(default["teammate_rebound_competition"] * (1.0 - default["wing_rebound_leakage_score"]), 0.0, 1.0)
    )
    default["frontcourt_rebound_overlap_score"] = float(
        np.clip(0.55 * default["teammate_rebound_competition"] + 0.45 * default["center_rebound_share_pressure"], 0.0, 1.0)
    )
    if np.isfinite(projected_opp_events) or np.isfinite(projected_team_events):
        defensive_pool = projected_opp_events if np.isfinite(projected_opp_events) else 0.0
        offensive_pool = 0.35 * projected_team_events if np.isfinite(projected_team_events) else 0.0
        default["team_rebound_pool_size"] = float(defensive_pool + offensive_pool)
    total_miss_proxy = float(
        np.nansum(
            [
                default["projected_missed_fga_total"] if np.isfinite(default["projected_missed_fga_total"]) else np.nan,
                default["projected_missed_fta_total"] if np.isfinite(default["projected_missed_fta_total"]) else np.nan,
            ]
        )
    )
    if np.isfinite(total_miss_proxy) and total_miss_proxy > 0.0 and np.isfinite(default["projected_available_rebound_events"]):
        default["free_throw_rebound_suppression"] = float(
            np.clip(1.0 - (float(default["projected_available_rebound_events"]) / total_miss_proxy), 0.0, 1.0)
        )
    return default


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def load_market_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market snapshot not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "Player" not in df.columns:
        raise ValueError("Market snapshot must include a Player column")
    if "Market_Date" not in df.columns:
        raise ValueError("Market snapshot must include a Market_Date column")
    df = df.copy()
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce")
    df = resolve_event_times(df)
    return df


def load_target_prediction_calibrator(path: Path | None) -> dict[str, dict] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    target_block = payload.get("targets", payload) if isinstance(payload, dict) else {}
    if not isinstance(target_block, dict):
        return None
    out: dict[str, dict] = {}
    for key, value in target_block.items():
        if str(key).upper().strip() not in TARGETS or not isinstance(value, dict):
            continue
        out[str(key).upper().strip()] = dict(value)
    return out or None


def apply_target_prediction_calibration(
    target: str,
    raw_prediction: float,
    market_line: float,
    calibrator: dict[str, dict] | None,
) -> tuple[float, dict[str, Any]]:
    meta = {
        "applied": False,
        "source": "identity",
        "edge_multiplier": 1.0,
        "edge_bias": 0.0,
        "support_rows": 0,
    }
    if calibrator is None:
        return float(raw_prediction), meta
    payload = calibrator.get(str(target).upper().strip())
    if not payload or not bool(payload.get("enabled", True)):
        meta["source"] = "disabled"
        return float(raw_prediction), meta
    raw_prediction = float(raw_prediction)
    market_line = float(market_line)
    if not np.isfinite(raw_prediction) or not np.isfinite(market_line):
        meta["source"] = "non_finite_input"
        return float(raw_prediction), meta

    edge_multiplier = float(np.clip(float(payload.get("edge_multiplier", 1.0)), 0.0, 1.25))
    edge_bias = float(payload.get("edge_bias", 0.0))
    tuned_prediction = float(market_line + edge_bias + edge_multiplier * (raw_prediction - market_line))
    max_adjustment_abs = float(payload.get("max_adjustment_abs", {"PTS": 3.0, "TRB": 2.0, "AST": 1.5}.get(str(target).upper().strip(), 2.0)))
    if np.isfinite(max_adjustment_abs) and max_adjustment_abs > 0.0:
        tuned_prediction = float(raw_prediction + np.clip(tuned_prediction - raw_prediction, -max_adjustment_abs, max_adjustment_abs))
    meta.update(
        {
            "applied": True,
            "source": str(payload.get("source", "target_prediction_calibrator")),
            "edge_multiplier": edge_multiplier,
            "edge_bias": edge_bias,
            "support_rows": int(payload.get("support_rows", 0)),
        }
    )
    return tuned_prediction, meta


def _player_aliases(player_dir_name: str) -> set[str]:
    normalized = normalize_name(player_dir_name)
    aliases = {normalized}
    parts = [part for part in normalized.split("_") if part]
    if len(parts) >= 2:
        aliases.add(f"{parts[0][0]}_{'_'.join(parts[1:])}")
        aliases.add(f"{parts[0][0]}_{parts[-1]}")
    return aliases


@lru_cache(maxsize=16)
def build_player_csv_index(season: int) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for player_dir in DATA_DIR.iterdir():
        if not player_dir.is_dir():
            continue
        csv_path = player_dir / f"{season}_processed_processed.csv"
        if not csv_path.exists():
            continue
        for alias in _player_aliases(player_dir.name):
            index.setdefault(alias, []).append(csv_path)
    return index


def infer_player_csv(player_name: str, season: int, player_csv_index: dict[str, list[Path]] | None = None) -> Path | None:
    candidate = DATA_DIR / player_name / f"{season}_processed_processed.csv"
    if candidate.exists():
        return candidate

    lookup = player_csv_index if player_csv_index is not None else build_player_csv_index(season)
    key = normalize_name(player_name)
    matches = lookup.get(key, [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    override_name = AMBIGUOUS_PLAYER_OVERRIDES.get(player_name) or AMBIGUOUS_PLAYER_OVERRIDES.get(key)
    if override_name:
        override_path = DATA_DIR / override_name / f"{season}_processed_processed.csv"
        if override_path.exists():
            return override_path

    # Skip unresolved ambiguous aliases instead of forcing a potentially wrong
    # mapping that can contaminate predictions and downstream validation.
    return None


def build_records(
    predictor: StructuredStackInference | None,
    market_df: pd.DataFrame,
    season: int,
    target_prediction_calibrator_path: Path | None = None,
    market_provenance_manifest: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []
    player_csv_index = build_player_csv_index(season)
    target_prediction_calibrator = load_target_prediction_calibrator(target_prediction_calibrator_path)
    rebound_env_context = load_rebound_environment_context(season)
    market_manifest = dict(market_provenance_manifest or {})
    market_provider = str(market_manifest.get("provider", "snapshot")).strip() or "snapshot"

    for _, market_row in market_df.iterrows():
        player = str(market_row["Player"])
        csv_path = infer_player_csv(player, season, player_csv_index=player_csv_index)
        if csv_path is None:
            skipped.append({"player": player, "reason": f"missing processed csv for season {season}"})
            continue

        history_df = pd.read_csv(csv_path)
        if history_df.empty:
            skipped.append({"player": player, "reason": "empty processed csv"})
            continue

        if "Date" in history_df.columns:
            history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
            history_df = history_df.loc[history_df["Date"].notna()].copy()
            history_df = history_df.loc[history_df["Date"] < market_row["Market_Date"]].copy()
        min_history_rows = max(5, int(getattr(predictor, "seq_len", 5)))
        if len(history_df) < min_history_rows:
            skipped.append({"player": player, "reason": f"insufficient history rows ({len(history_df)})"})
            continue

        explanation = None
        if predictor is not None:
            try:
                market_context = {
                    "player": player,
                    "market_date": str(pd.to_datetime(market_row["Market_Date"], errors="coerce").date()) if pd.notna(market_row["Market_Date"]) else None,
                    "market_home_team": market_row.get("Market_Home_Team"),
                    "market_away_team": market_row.get("Market_Away_Team"),
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    explanation = predictor.predict(history_df, assume_prepared=True, market_context=market_context)
            except Exception as exc:
                explanation = build_heuristic_explanation(history_df, failure_reason=f"{type(exc).__name__}")
        if explanation is None:
            explanation = build_heuristic_explanation(history_df)

        latest_row = history_df.iloc[-1]
        player_team = team_abbr_from_matchup(latest_row.get("MATCHUP")) if "MATCHUP" in latest_row.index else None
        market_home_team = market_row.get("Market_Home_Team")
        market_away_team = market_row.get("Market_Away_Team")
        market_teams = {str(item) for item in [market_home_team, market_away_team] if pd.notna(item) and str(item)}
        if market_teams and player_team is not None and player_team not in market_teams:
            skipped.append(
                {
                    "player": player,
                    "reason": f"market_team_mismatch:{player_team} not in {sorted(market_teams)}",
                }
            )
            continue
        market_fetched_at_utc = market_row.get("Market_Fetched_At_UTC")
        market_row_provider = str(market_row.get("Market_Provider", "")).strip() or market_provider
        market_row_book = str(market_row.get("Market_Book", "")).strip() or "aggregate_market_snapshot"
        market_price_source = str(market_row.get("Market_Price_Source", "")).strip() or "normalized_market_snapshot"
        market_price_source_type = str(market_row.get("Market_Price_Source_Type", "")).strip() or (
            "ARCHIVED_ENTRY" if pd.notna(market_fetched_at_utc) else "UNKNOWN"
        )
        market_snapshot_id = str(market_row.get("Market_Snapshot_ID", "")).strip() or derive_snapshot_id(
            provider=market_row_provider,
            odds_snapshot_time=market_fetched_at_utc,
            fallback_label=str(market_row.get("Market_Event_ID", "")),
        )
        record = {
            "player": player,
            "player_id": latest_player_id(history_df),
            "team": player_team,
            "opponent": str(latest_row.get("Opponent", "")).strip() or None,
            "market_date": str(market_row["Market_Date"].date()) if pd.notna(market_row["Market_Date"]) else None,
            "market_player_raw": market_row.get("Market_Player_Raw"),
            "market_event_id": market_row.get("Market_Event_ID"),
            "market_commence_time_utc": market_row.get("Market_Commence_Time_UTC"),
            "event_time_source": market_row.get("event_time_source"),
            "event_time_confidence": market_row.get("event_time_confidence"),
            "event_time_resolution_reason": market_row.get("event_time_resolution_reason"),
            "event_time_resolution_warning": market_row.get("event_time_resolution_warning"),
            "market_home_team": market_home_team if pd.notna(market_home_team) else None,
            "market_away_team": market_away_team if pd.notna(market_away_team) else None,
            "market_provider": market_row_provider,
            "market_book": market_row_book,
            "market_price_source": market_price_source,
            "market_price_source_type": market_price_source_type,
            "market_price_source_hint": market_price_source_type,
            "market_snapshot_id": market_snapshot_id,
            "market_fetched_at_utc": market_fetched_at_utc,
            "history_rows": int(len(history_df)),
            "last_history_date": str(pd.to_datetime(latest_row["Date"]).date()) if "Date" in latest_row.index and pd.notna(latest_row["Date"]) else None,
            "csv": str(csv_path),
            "belief_uncertainty": float(explanation["latent_environment"].get("belief_uncertainty", 0.0)),
            "feasibility": float(explanation["latent_environment"].get("feasibility", 0.0)),
            "role_shift_risk": float(explanation["latent_environment"].get("role_shift_risk", 0.0)),
            "volatility_regime_risk": float(explanation["latent_environment"].get("volatility_regime_risk", 0.0)),
            "context_pressure_risk": float(explanation["latent_environment"].get("context_pressure_risk", 0.0)),
            "fallback_blend": float(explanation.get("data_quality", {}).get("fallback_blend", 0.0)),
            "fallback_reasons": ",".join(explanation.get("data_quality", {}).get("fallback_reasons", [])),
        }
        record.update(
            build_rebound_diagnostics(
                history_df,
                rebound_env_context,
                market_home_team if pd.notna(market_home_team) else None,
                market_away_team if pd.notna(market_away_team) else None,
            )
        )
        for target in TARGETS:
            raw_pred_value = float(explanation["predicted"][target])
            baseline_value = float(explanation["baseline"][target])
            market_value = market_row.get(f"Market_{target}", np.nan)
            market_value = float(market_value) if pd.notna(market_value) else np.nan
            pred_value = raw_pred_value
            calibration_meta = {
                "applied": False,
                "source": "identity",
                "edge_multiplier": 1.0,
                "edge_bias": 0.0,
                "support_rows": 0,
            }
            if pd.notna(market_value):
                pred_value, calibration_meta = apply_target_prediction_calibration(
                    target,
                    raw_prediction=raw_pred_value,
                    market_line=market_value,
                    calibrator=target_prediction_calibrator,
                )
            record[f"pred_{target}"] = pred_value
            record[f"raw_pred_{target}"] = raw_pred_value
            record[f"baseline_{target}"] = baseline_value
            record[f"market_{target}"] = market_value
            record[f"edge_{target}"] = pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"raw_edge_{target}"] = raw_pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"baseline_edge_{target}"] = baseline_value - market_value if pd.notna(market_value) else np.nan
            record[f"{target}_uncertainty_sigma"] = float(explanation["target_factors"][target].get("uncertainty_sigma", 0.0))
            record[f"{target}_spike_probability"] = float(explanation["target_factors"][target].get("spike_probability", 0.0))
            record[f"market_books_{target}"] = float(market_row.get(f"Market_{target}_books", np.nan)) if pd.notna(market_row.get(f"Market_{target}_books", np.nan)) else np.nan
            record[f"market_over_price_{target}"] = float(market_row.get(f"Market_{target}_over_price", np.nan)) if pd.notna(market_row.get(f"Market_{target}_over_price", np.nan)) else np.nan
            record[f"market_under_price_{target}"] = float(market_row.get(f"Market_{target}_under_price", np.nan)) if pd.notna(market_row.get(f"Market_{target}_under_price", np.nan)) else np.nan
            record[f"market_line_std_{target}"] = float(market_row.get(f"Market_{target}_line_std", np.nan)) if pd.notna(market_row.get(f"Market_{target}_line_std", np.nan)) else np.nan
            record[f"prediction_calibration_applied_{target}"] = bool(calibration_meta["applied"])
            record[f"prediction_calibration_source_{target}"] = str(calibration_meta["source"])
            record[f"prediction_calibration_multiplier_{target}"] = float(calibration_meta["edge_multiplier"])
            record[f"prediction_calibration_bias_{target}"] = float(calibration_meta["edge_bias"])
            record[f"prediction_calibration_support_rows_{target}"] = int(calibration_meta["support_rows"])
        records.append(record)

    return records, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an upcoming market slate with model-vs-market edges.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026 for 2025-26.")
    parser.add_argument("--market-wide-path", type=Path, default=DEFAULT_MARKET_WIDE, help="Normalized wide market snapshot.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow slate build to continue with heuristic-only predictions when model load fails.",
    )
    parser.add_argument(
        "--target-prediction-calibrator-json",
        type=Path,
        default=DEFAULT_TARGET_PREDICTION_CALIBRATOR,
        help="Optional target-level short-term prediction calibrator JSON.",
    )
    parser.add_argument(
        "--disable-target-prediction-calibration",
        action="store_true",
        help="Disable target-level prediction calibration and keep raw model outputs.",
    )
    parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.csv", help="Output CSV path.")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor: StructuredStackInference | None = None
    predictor_error = None
    try:
        predictor = _make_structured_stack_inference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    except Exception as exc:
        predictor_error = f"{type(exc).__name__}: {exc}"
        if not args.allow_heuristic_fallback:
            raise RuntimeError(
                "Model inference failed while heuristic fallback is disabled. "
                "Pass --allow-heuristic-fallback to continue anyway. "
                f"Root cause: {predictor_error}"
            ) from exc
        print(f"Warning: model inference unavailable, using heuristic fallback only ({predictor_error})")
    market_df = load_market_wide(args.market_wide_path)
    market_provenance_manifest = load_market_snapshot_manifest(args.market_wide_path.resolve())
    calibrator_path = None if args.disable_target_prediction_calibration else args.target_prediction_calibrator_json
    records, skipped = build_records(
        predictor,
        market_df,
        args.season,
        target_prediction_calibrator_path=calibrator_path,
        market_provenance_manifest=market_provenance_manifest,
    )

    if not records:
        raise RuntimeError(f"No upcoming slate rows built. Skipped={len(skipped)} sample={skipped[:5]}")

    results_df = pd.DataFrame.from_records(records).sort_values(["market_date", "player"]).reset_index(drop=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.csv_out, index=False)
    payload = {
        "manifest_path": str(manifest_path),
        "run_id": predictor.metadata.get("run_id") if predictor is not None else None,
        "predictor_error": predictor_error,
        "market_snapshot": str(args.market_wide_path),
        "market_snapshot_provider": str(market_provenance_manifest.get("provider", "")),
        "season": args.season,
        "rows": int(len(results_df)),
        "skipped": skipped,
        "target_prediction_calibrator_json": str(calibrator_path) if calibrator_path is not None else None,
        "target_prediction_calibration_enabled": bool(not args.disable_target_prediction_calibration),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("UPCOMING MARKET SLATE BUILT")
    print("=" * 80)
    print(f"Rows:     {len(results_df)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"CSV:      {args.csv_out}")
    print(f"JSON:     {args.json_out}")
    print("\nSample:")
    sample_cols = [
        "player",
        "market_date",
        "pred_PTS",
        "market_PTS",
        "edge_PTS",
        "pred_TRB",
        "market_TRB",
        "edge_TRB",
        "pred_AST",
        "market_AST",
        "edge_AST",
    ]
    print(results_df[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
