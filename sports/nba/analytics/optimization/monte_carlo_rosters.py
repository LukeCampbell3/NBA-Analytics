"""
Monte Carlo Roster Generation

Generates random cap-legal rosters and scores them to find
high-quality roster configurations through random search.

Complements greedy_builder by exploring roster space stochastically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..schema import PlayerCapabilityVector
from ..cap.cap_rules import CapConstraints, PlayerSalary, validate_roster_legality
from ..team_building.roster_score import score_roster, RosterScoreResult


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo roster search."""
    iterations: int = 0
    valid_rosters_found: int = 0
    best_score: float = 0.0
    best_roster: List[str] = field(default_factory=list)
    best_result: Optional[RosterScoreResult] = None
    top_10: List[Dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "valid_rosters_found": self.valid_rosters_found,
            "best_score": round(self.best_score, 1),
            "best_roster": self.best_roster,
            "best_breakdown": self.best_result.to_dict() if self.best_result else None,
            "top_10_scores": [r["score"] for r in self.top_10],
            "stopped_reason": self.stopped_reason,
        }


def monte_carlo_search(
    pool: List[Tuple[PlayerCapabilityVector, PlayerSalary]],
    constraints: CapConstraints = None,
    iterations: int = 1000,
    roster_size: int = 15,
    seed: int = 42,
) -> MonteCarloResult:
    """Generate random cap-legal rosters and find the best.

    Args:
        pool: Available players with salaries
        constraints: Cap constraints
        iterations: Number of random rosters to try
        roster_size: Target roster size
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with best roster found
    """
    if constraints is None:
        constraints = CapConstraints()

    rng = random.Random(seed)
    result = MonteCarloResult()
    max_salary = constraints.max_salary()
    top_rosters: List[Tuple[float, List[str], RosterScoreResult]] = []

    for i in range(iterations):
        # Random selection
        shuffled = list(range(len(pool)))
        rng.shuffle(shuffled)

        selected_vectors = []
        selected_salaries = []
        used_ids = set()
        total_salary = 0.0

        for idx in shuffled:
            if len(selected_vectors) >= roster_size:
                break
            vector, salary = pool[idx]
            if vector.player_id in used_ids:
                continue
            if total_salary + salary.cap_hit > max_salary:
                continue

            selected_vectors.append(vector)
            selected_salaries.append(salary)
            used_ids.add(vector.player_id)
            total_salary += salary.cap_hit

        # Validate
        if len(selected_vectors) < constraints.min_roster_size:
            continue

        legality = validate_roster_legality(selected_salaries, constraints)
        if not legality["legal"]:
            continue

        result.valid_rosters_found += 1

        # Score
        roster_result = score_roster(selected_vectors)
        score = roster_result.roster_score

        if score > result.best_score:
            result.best_score = score
            result.best_roster = [v.player_name for v in selected_vectors]
            result.best_result = roster_result

        # Track top 10
        top_rosters.append((score, [v.player_name for v in selected_vectors], roster_result))
        top_rosters.sort(key=lambda x: -x[0])
        top_rosters = top_rosters[:10]

    result.iterations = iterations
    result.top_10 = [{"score": round(s, 1), "players": p[:5]} for s, p, _ in top_rosters]
    result.stopped_reason = f"completed_{iterations}_iterations"
    return result
