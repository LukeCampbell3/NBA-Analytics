"""
Build League Vectors

Builds capability vectors for all players in Data-Proc,
computes league-wide percentiles, and outputs player cards.

Output:
  sports/nba/analytics/output/player_vectors.json
  sports/nba/analytics/output/league_percentiles.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "sports" / "nba"))

from analytics.schema import CAPABILITY_DIMENSIONS, PlayerCapabilityVector
from analytics.features.capability_vectors import build_capability_vector, DATA_PROC_DIR
from analytics.features.percentiles import compute_percentile, reliability_shrinkage
from analytics.features.spacing_features import compute_player_spacing
from analytics.roles.role_distribution import generate_role_distribution

OUTPUT_DIR = WORKSPACE / "sports" / "nba" / "analytics" / "output"


def discover_players() -> List[str]:
    """Find all players with data in Data-Proc."""
    if not DATA_PROC_DIR.exists():
        return []
    players = []
    for d in sorted(DATA_PROC_DIR.iterdir()):
        if d.is_dir() and any(d.glob("*processed*.csv")):
            players.append(d.name.replace("_", " "))
    return players


def build_all_vectors(max_players: int = 0) -> List[PlayerCapabilityVector]:
    """Build capability vectors for all available players."""
    players = discover_players()
    if max_players > 0:
        players = players[:max_players]

    vectors = []
    for name in players:
        v = build_capability_vector(name)
        if v.confidence_summary()["observed_dimensions"] > 0:
            vectors.append(v)
    return vectors


def compute_league_percentiles(vectors: List[PlayerCapabilityVector]) -> Dict[str, np.ndarray]:
    """Compute league-wide percentile populations for each dimension."""
    populations = {}
    for dim in CAPABILITY_DIMENSIONS:
        vals = [v.get(dim).raw_value for v in vectors if v.get(dim).raw_value is not None]
        populations[dim] = np.array(vals) if vals else np.array([])
    return populations


def assign_percentiles(vectors: List[PlayerCapabilityVector], populations: Dict[str, np.ndarray]) -> None:
    """Assign league-wide percentiles to each vector in-place."""
    for v in vectors:
        for dim in CAPABILITY_DIMENSIONS:
            d = v.get(dim)
            if d.raw_value is not None and len(populations.get(dim, [])) > 0:
                raw_pct = compute_percentile(d.raw_value, populations[dim])
                adj_pct = reliability_shrinkage(raw_pct, d.sample_size)
                v.set_dimension(dim, raw_percentile=raw_pct, reliability_adjusted_percentile=adj_pct)


def build_spacing_profiles(vectors: List[PlayerCapabilityVector]) -> Dict[str, Any]:
    """Build spacing profiles for all players."""
    profiles = {}
    for v in vectors:
        three_pct = (v.get("shooting_gravity").raw_value or 0) / 100 * 0.42  # Approximate
        three_freq = (v.get("spacing_gravity").raw_value or 0) / 100 * 0.4
        prof = compute_player_spacing(
            three_pct=min(three_pct, 0.50),
            three_pa_rate=min(three_freq, 0.6),
            three_pa_per_game=three_freq * 15,
            ft_pct=0.78,  # League average fallback
            assisted_rate=0.5,
            rim_frequency=(v.get("rim_pressure").raw_value or 30) / 100 * 0.5,
            usage_rate=0.2,
            games_played=v.metadata.get("games_played", 30),
            player_id=v.player_id,
            player_name=v.player_name,
        )
        profiles[v.player_name] = prof.to_dict()
    return profiles


def run_full_build(max_players: int = 0) -> Dict[str, Any]:
    """Run full league build pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering players...")
    vectors = build_all_vectors(max_players)
    print(f"Built {len(vectors)} player vectors")

    print("Computing league percentiles...")
    populations = compute_league_percentiles(vectors)
    assign_percentiles(vectors, populations)

    print("Generating role distributions...")
    roles = {}
    for v in vectors:
        rd = generate_role_distribution(v)
        roles[v.player_name] = rd.to_dict()

    print("Computing spacing profiles...")
    spacing = build_spacing_profiles(vectors)

    # Output
    vector_output = [v.to_dict() for v in vectors]
    output_path = OUTPUT_DIR / "player_vectors.json"
    output_path.write_text(json.dumps(vector_output, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(vector_output)} vectors to {output_path}")

    roles_path = OUTPUT_DIR / "role_distributions.json"
    roles_path.write_text(json.dumps(roles, indent=2, default=str), encoding="utf-8")
    print(f"Wrote roles to {roles_path}")

    spacing_path = OUTPUT_DIR / "spacing_profiles.json"
    spacing_path.write_text(json.dumps(spacing, indent=2, default=str), encoding="utf-8")
    print(f"Wrote spacing to {spacing_path}")

    # Summary
    summary = {
        "players_processed": len(vectors),
        "dimensions": len(CAPABILITY_DIMENSIONS),
        "avg_confidence": round(np.mean([v.confidence_summary()["average_confidence"] for v in vectors]), 3),
        "avg_observed_dims": round(np.mean([v.confidence_summary()["observed_dimensions"] for v in vectors]), 1),
    }
    print(f"\nSummary: {summary}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-players", type=int, default=0, help="Limit players (0=all)")
    args = parser.parse_args()
    run_full_build(args.max_players)
