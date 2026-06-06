"""
Greedy Roster Builder

Initial roster optimization using greedy selection + local swap search.

Objective: Maximize RosterScore subject to cap/legality constraints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..schema import PlayerCapabilityVector
from ..cap.cap_rules import CapConstraints, PlayerSalary, validate_roster_legality
from ..team_building.roster_score import RosterScoreResult, score_roster


@dataclass
class RosterCandidate:
    """A candidate roster for optimization."""
    players: List[PlayerCapabilityVector]
    salaries: List[PlayerSalary]
    score: Optional[RosterScoreResult] = None

    def total_salary(self) -> float:
        return sum(s.cap_hit for s in self.salaries)

    def player_ids(self) -> set:
        return {v.player_id for v in self.players}


from dataclasses import dataclass


def greedy_build(
    pool: List[Tuple[PlayerCapabilityVector, PlayerSalary]],
    constraints: CapConstraints = None,
    target_size: int = 15,
) -> RosterCandidate:
    """Build a roster greedily by adding highest-value players under cap.

    Strategy:
    1. Sort pool by estimated value (avg dimension score / salary)
    2. Add players one at a time if they fit under cap
    3. Stop at target_size or when cap is exhausted

    Args:
        pool: List of (vector, salary) tuples for available players
        constraints: Cap constraints
        target_size: Target roster size

    Returns:
        RosterCandidate with selected players and scores
    """
    if constraints is None:
        constraints = CapConstraints()

    # Score each player by average capability / salary efficiency
    def _player_value(pair):
        v, s = pair
        dims = [d.raw_value for d in v.dimensions.values() if d.raw_value is not None]
        avg_cap = np.mean(dims) if dims else 0
        salary_m = max(s.cap_hit / 1_000_000, 1.0)
        return avg_cap / salary_m

    sorted_pool = sorted(pool, key=_player_value, reverse=True)

    selected_vectors = []
    selected_salaries = []
    used_ids = set()
    total = 0.0
    max_sal = constraints.max_salary()

    for vector, salary in sorted_pool:
        if len(selected_vectors) >= target_size:
            break
        if vector.player_id in used_ids:
            continue
        if total + salary.cap_hit > max_sal:
            continue

        selected_vectors.append(vector)
        selected_salaries.append(salary)
        used_ids.add(vector.player_id)
        total += salary.cap_hit

    # Score the result
    result_score = score_roster(selected_vectors)
    candidate = RosterCandidate(
        players=selected_vectors,
        salaries=selected_salaries,
        score=result_score,
    )
    return candidate


def local_swap_search(
    roster: RosterCandidate,
    pool: List[Tuple[PlayerCapabilityVector, PlayerSalary]],
    constraints: CapConstraints = None,
    max_iterations: int = 100,
) -> RosterCandidate:
    """Improve roster via local swap search.

    For each player on the roster, try swapping with each pool player.
    Accept swaps that improve roster score while staying cap-legal.
    """
    if constraints is None:
        constraints = CapConstraints()

    best = roster
    best_total = best.score.roster_score if best.score else 0
    iterations = 0
    improved = True

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for i in range(len(best.players)):
            for vector, salary in pool:
                if vector.player_id in best.player_ids():
                    continue

                # Try swap
                new_vectors = best.players[:i] + [vector] + best.players[i+1:]
                new_salaries = best.salaries[:i] + [salary] + best.salaries[i+1:]

                # Check legality
                legality = validate_roster_legality(new_salaries, constraints)
                if not legality["legal"]:
                    continue

                # Score new roster
                new_score = score_roster(new_vectors)
                if new_score.roster_score > best_total:
                    best = RosterCandidate(
                        players=new_vectors,
                        salaries=new_salaries,
                        score=new_score,
                    )
                    best_total = new_score.roster_score
                    improved = True
                    break  # Restart from beginning after improvement

            if improved:
                break

    return best
