#!/usr/bin/env python3
"""
Generate NBA 2026-27 Simulated Season Metrics

Uses latest player history from Data-Proc to produce Monte Carlo
simulation cards for the frontend. Each player gets:
  - Per-stat distributions (PTS, REB, AST, PRA)
  - Confidence tiers
  - Forecastability / volatility / stability scores
  - Narrative summaries

Output: sports/nba/web/data/player_simulation_cards.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[3]
DATA_DIR = WORKSPACE / "Player-Predictor" / "Data-Proc"
OUTPUT_PATH = WORKSPACE / "sports" / "nba" / "web" / "data" / "player_simulation_cards.json"
SEASON = 2026
SIM_COUNT = 10000
MIN_GAMES = 10
STATS = ["PTS", "TRB", "AST"]


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def load_player_history(player_dir: Path) -> pd.DataFrame:
    csv_path = player_dir / f"{SEASON}_processed_processed.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        if "Did_Not_Play" in df.columns:
            df = df[df["Did_Not_Play"] != 1]
        return df
    except Exception:
        return pd.DataFrame()


def compute_stat_distribution(values: np.ndarray) -> Dict[str, Any]:
    """Run Monte Carlo simulation on a stat's recent values."""
    if len(values) < MIN_GAMES:
        return None

    # Exponential weighted parameters
    weights = np.exp(np.linspace(-1, 0, len(values)))
    weights /= weights.sum()
    mean = float(np.average(values, weights=weights))
    std = float(np.std(values))
    if std < 0.5:
        std = max(0.5, mean * 0.15)

    # Monte Carlo: sample from normal distribution
    rng = np.random.default_rng(seed=hash(f"{mean:.3f}_{std:.3f}") % (2**31))
    samples = rng.normal(loc=mean, scale=std, size=SIM_COUNT)
    samples = np.maximum(samples, 0)  # Floor at 0

    p10, p25, p50, p75, p90 = np.percentile(samples, [10, 25, 50, 75, 90])
    volatility = std / mean if mean > 0 else 1.0

    # Probability comparisons
    last_season_avg = float(np.mean(values))
    p_above = float(np.mean(samples > last_season_avg))
    p_below = 1.0 - p_above

    return {
        "mean": float(np.mean(samples)),
        "median": float(p50),
        "p10": float(p10),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "floor": float(p10),
        "ceiling": float(p90),
        "volatility": float(min(volatility, 1.0)),
        "confidence": _confidence_from_volatility(volatility, len(values)),
        "probability_above_last_season_avg": round(p_above, 4),
        "probability_below_last_season_avg": round(p_below, 4),
    }


def _confidence_from_volatility(vol: float, n_games: int) -> str:
    if n_games >= 50 and vol < 0.4:
        return "HIGH_CONFIDENCE"
    elif n_games >= 25 and vol < 0.6:
        return "MEDIUM_CONFIDENCE"
    else:
        return "LOW_CONFIDENCE"


def determine_archetype(pts_avg: float, reb_avg: float, ast_avg: float, mp_avg: float) -> str:
    if mp_avg < 15:
        return "bench_depth"
    if pts_avg >= 25:
        return "primary_scorer"
    if ast_avg >= 7:
        return "primary_playmaker"
    if reb_avg >= 10:
        return "paint_anchor"
    if pts_avg >= 18:
        return "secondary_scorer"
    if mp_avg >= 28:
        return "versatile_starter"
    if mp_avg >= 20:
        return "rotation_role_player"
    return "rotation_role_player"


def generate_narrative(player: str, pts_dist: Dict, reb_dist: Dict, ast_dist: Dict, confidence: str) -> Dict[str, str]:
    pts_mean = pts_dist["mean"] if pts_dist else 0
    reb_mean = reb_dist["mean"] if reb_dist else 0
    ast_mean = ast_dist["mean"] if ast_dist else 0

    name = player.replace("_", " ")
    best = f"{name} projects to {pts_mean:.1f} PPG / {reb_mean:.1f} RPG / {ast_mean:.1f} APG for 2026-27."
    uncertainty = f"Volatility range: {pts_dist['p10']:.0f}-{pts_dist['p90']:.0f} PTS per game." if pts_dist else "Insufficient data."
    upside = f"Ceiling scenario: {pts_dist['ceiling']:.0f} PTS if role expands." if pts_dist else ""
    downside = f"Floor scenario: {pts_dist['floor']:.0f} PTS if minutes decline." if pts_dist else ""
    risk = "Projection quality: " + confidence.replace("_", " ").lower() + "."

    return {
        "best_projection_summary": best,
        "uncertainty_summary": uncertainty,
        "primary_upside_path": upside,
        "primary_downside_path": downside,
        "main_risk_factors": risk,
    }


def generate_all_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    player_dirs = sorted(DATA_DIR.iterdir())

    for player_dir in player_dirs:
        if not player_dir.is_dir():
            continue

        df = load_player_history(player_dir)
        if df.empty or len(df) < MIN_GAMES:
            continue

        player_name = player_dir.name
        player_id = str(df["Player_ID"].iloc[0]) if "Player_ID" in df.columns else player_name
        team = str(df["Team_ID"].iloc[-1]) if "Team_ID" in df.columns else ""

        # Get stat arrays
        pts_vals = df["PTS"].dropna().values if "PTS" in df.columns else np.array([])
        reb_vals = df["TRB"].dropna().values if "TRB" in df.columns else np.array([])
        ast_vals = df["AST"].dropna().values if "AST" in df.columns else np.array([])
        mp_vals = df["MP"].dropna().values if "MP" in df.columns else np.array([])

        if len(pts_vals) < MIN_GAMES:
            continue

        # Compute distributions
        pts_dist = compute_stat_distribution(pts_vals)
        reb_dist = compute_stat_distribution(reb_vals)
        ast_dist = compute_stat_distribution(ast_vals)

        # PRA combined
        pra_vals = pts_vals[:min(len(pts_vals), len(reb_vals), len(ast_vals))]
        if len(reb_vals) >= len(pra_vals) and len(ast_vals) >= len(pra_vals):
            pra_vals = pts_vals[:len(pra_vals)] + reb_vals[:len(pra_vals)] + ast_vals[:len(pra_vals)]
            pra_dist = compute_stat_distribution(pra_vals)
        else:
            pra_dist = None

        # Scores
        pts_avg = float(np.mean(pts_vals))
        reb_avg = float(np.mean(reb_vals)) if len(reb_vals) > 0 else 0
        ast_avg = float(np.mean(ast_vals)) if len(ast_vals) > 0 else 0
        mp_avg = float(np.mean(mp_vals)) if len(mp_vals) > 0 else 0

        vol_score = float(np.std(pts_vals) / max(pts_avg, 1))
        n_games = len(df)
        forecastability = min(1.0, n_games / 70) * (1 - min(vol_score, 0.8))
        role_stability = min(1.0, n_games / 60) * (1 - abs(np.std(mp_vals) / max(mp_avg, 1)) if mp_avg > 0 else 0.5)

        confidence = _confidence_from_volatility(vol_score, n_games)
        archetype = determine_archetype(pts_avg, reb_avg, ast_avg, mp_avg)
        narrative = generate_narrative(player_name, pts_dist, reb_dist, ast_dist, confidence)

        card = {
            "player_id": player_id,
            "player": player_name,
            "team": team,
            "position": "",
            "archetype": archetype,
            "simulation_count": SIM_COUNT,
            "confidence_tier": confidence,
            "forecastability_score": forecastability,
            "volatility_score": vol_score,
            "role_stability_score": role_stability,
            "projected_games_played": min(82, int(n_games * 82 / max(n_games, 30))),
            "projected_minutes_per_game": round(mp_avg, 1),
            "pts": pts_dist,
            "reb": reb_dist,
            "ast": ast_dist,
            "pra": pra_dist,
            **narrative,
            "missing_data_warnings": [],
            "credibility_notes": f"Based on {n_games} games from 2025-26 season data.",
            "data_cutoff_date": "2026-06-05",
        }
        cards.append(card)

    return cards


def main():
    print("Generating NBA 2026-27 simulation cards...")
    cards = generate_all_cards()
    print(f"Generated {len(cards)} player cards")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cards, indent=2, default=_json_default), encoding="utf-8")
    print(f"Written to: {OUTPUT_PATH}")

    # Update credibility gate
    gate_path = OUTPUT_PATH.parent / "simulation_credibility_gate.json"
    gate = {
        "status": "PUBLISH_RESEARCH_ONLY",
        "labels": {
            "pipeline": "SIMULATION_PIPELINE_READY",
            "credibility": "PUBLISH_RESEARCH_ONLY",
            "frontend_label": "research projection (2026-27 season)"
        },
        "publish_as_calibrated": False,
        "player_count": len(cards),
        "simulation_count": SIM_COUNT,
        "data_source": "Player-Predictor/Data-Proc (2025-26 season)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Credibility gate updated: {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
