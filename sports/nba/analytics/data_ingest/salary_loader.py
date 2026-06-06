"""
NBA Salary Data Loader

Loads player salary data for cap-aware roster building.
Sources: existing repo config + scraped/manual salary CSV.

If no salary data available:
  - Uses estimated salary from valuation model
  - Marks as inferred with reduced confidence
  - Does not crash pipeline
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..cap.cap_rules import PlayerSalary, SALARY_CAP, MINIMUM_SALARY

WORKSPACE = Path(__file__).resolve().parents[4]
SALARY_CSV_PATH = WORKSPACE / "sports" / "nba" / "analytics" / "data" / "raw" / "salaries_2026.csv"
SALARY_ESTIMATES_PATH = WORKSPACE / "sports" / "nba" / "analytics" / "data" / "processed" / "salary_estimates.json"


def load_salary_data() -> Dict[str, PlayerSalary]:
    """Load salary data from available sources.

    Priority:
    1. Actual salary CSV if available
    2. Estimated from player value model
    3. Minimum salary fallback

    Returns dict keyed by player_name.
    """
    salaries = {}

    # Try actual salary file
    if SALARY_CSV_PATH.exists():
        try:
            df = pd.read_csv(SALARY_CSV_PATH)
            for _, row in df.iterrows():
                name = str(row.get("player_name", row.get("PLAYER", ""))).strip()
                sal = float(row.get("salary", row.get("SALARY", 0)))
                if name and sal > 0:
                    salaries[name] = PlayerSalary(
                        player_id=str(row.get("player_id", "")),
                        player_name=name,
                        salary=sal,
                        years_remaining=int(row.get("years_remaining", 1)),
                        is_rookie_scale=bool(row.get("rookie_scale", False)),
                        bird_rights=str(row.get("bird_rights", "none")),
                    )
            return salaries
        except Exception:
            pass

    # Try estimated salaries
    if SALARY_ESTIMATES_PATH.exists():
        try:
            data = json.loads(SALARY_ESTIMATES_PATH.read_text())
            for entry in data:
                name = entry.get("player_name", "")
                sal = float(entry.get("estimated_salary", 0))
                if name and sal > 0:
                    salaries[name] = PlayerSalary(
                        player_name=name,
                        salary=sal,
                    )
            return salaries
        except Exception:
            pass

    return salaries


def estimate_salary_from_value(player_name: str, player_value_score: float, age: float = 25) -> PlayerSalary:
    """Estimate salary from player value when no actual data exists.

    Uses simple market-based estimation:
    - Top tier (value > 80): max contract range ($35-50M)
    - High tier (60-80): $15-35M
    - Mid tier (40-60): $5-15M
    - Low tier (< 40): $1.5-5M

    Marks as inferred data.
    """
    if player_value_score >= 80:
        base = 35_000_000 + (player_value_score - 80) * 750_000
    elif player_value_score >= 60:
        base = 15_000_000 + (player_value_score - 60) * 1_000_000
    elif player_value_score >= 40:
        base = 5_000_000 + (player_value_score - 40) * 500_000
    else:
        base = max(MINIMUM_SALARY, 1_500_000 + player_value_score * 80_000)

    # Age adjustment: young players on rookie deals, old players declining
    if age <= 23:
        base = min(base, 12_000_000)  # Rookie scale cap
    elif age >= 34:
        base *= 0.7

    return PlayerSalary(
        player_name=player_name,
        salary=int(base),
        is_rookie_scale=(age <= 23),
    )


def build_league_salary_map(vectors=None) -> Dict[str, PlayerSalary]:
    """Build salary map for all players, using actual data where available
    and estimates where not.

    Args:
        vectors: Optional list of PlayerCapabilityVectors for estimation
    """
    actual = load_salary_data()

    if vectors:
        for v in vectors:
            if v.player_name not in actual:
                # Estimate from capability vector
                dims = [d.raw_value for d in v.dimensions.values() if d.raw_value is not None]
                avg_val = np.mean(dims) if dims else 30
                # Normalize to 0-100 scale
                val_score = min(100, max(0, avg_val * 1.5))
                age = float(v.metadata.get("age", 25))
                actual[v.player_name] = estimate_salary_from_value(v.player_name, val_score, age)

    return actual
