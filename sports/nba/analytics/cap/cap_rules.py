"""
NBA Salary Cap Rules (2025-26 Season)

Supports:
  salary loading, roster size constraints, cap limit,
  luxury tax placeholder, first/second apron placeholders,
  rookie scale placeholder, minimum contract placeholder,
  Bird rights as future extension hook
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


# 2025-26 NBA Cap figures (approximate)
SALARY_CAP = 141_000_000
LUXURY_TAX = 172_000_000
FIRST_APRON = 179_000_000
SECOND_APRON = 189_000_000
MINIMUM_SALARY = 1_200_000
ROOKIE_SCALE_MIN = 1_200_000
ROOKIE_SCALE_MAX = 12_000_000

MIN_ROSTER_SIZE = 14
MAX_ROSTER_SIZE = 15
STARTING_LINEUP_SIZE = 5


@dataclass
class CapConstraints:
    """Cap constraints for roster building."""
    cap_limit: float = SALARY_CAP
    luxury_tax_line: float = LUXURY_TAX
    first_apron: float = FIRST_APRON
    second_apron: float = SECOND_APRON
    min_roster_size: int = MIN_ROSTER_SIZE
    max_roster_size: int = MAX_ROSTER_SIZE
    allow_over_cap: bool = False
    allow_luxury_tax: bool = True
    hard_cap_mode: str = "soft"  # "soft", "first_apron", "second_apron"

    def max_salary(self) -> float:
        if self.hard_cap_mode == "second_apron":
            return self.second_apron
        if self.hard_cap_mode == "first_apron":
            return self.first_apron
        if self.allow_luxury_tax:
            return self.luxury_tax_line * 1.5  # Practical limit
        return self.cap_limit


@dataclass
class PlayerSalary:
    """Salary information for a player."""
    player_id: str = ""
    player_name: str = ""
    salary: float = 0.0
    years_remaining: int = 1
    is_rookie_scale: bool = False
    is_minimum: bool = False
    cap_hold: float = 0.0
    bird_rights: str = "none"  # "none", "early_bird", "bird", "non_bird"

    @property
    def cap_hit(self) -> float:
        return self.salary if self.salary > 0 else self.cap_hold


def validate_roster_legality(
    salaries: List[PlayerSalary],
    constraints: CapConstraints = None,
) -> Dict[str, Any]:
    """Validate a roster against cap rules.

    Returns validation result with pass/fail and reasons.
    """
    if constraints is None:
        constraints = CapConstraints()

    total_salary = sum(p.cap_hit for p in salaries)
    roster_size = len(salaries)
    max_sal = constraints.max_salary()

    violations = []
    warnings = []

    if roster_size < constraints.min_roster_size:
        violations.append(f"roster_too_small: {roster_size} < {constraints.min_roster_size}")
    if roster_size > constraints.max_roster_size:
        violations.append(f"roster_too_large: {roster_size} > {constraints.max_roster_size}")
    if total_salary > max_sal:
        violations.append(f"over_cap_limit: ${total_salary:,.0f} > ${max_sal:,.0f}")
    if total_salary > constraints.luxury_tax_line:
        warnings.append(f"luxury_tax_triggered: ${total_salary:,.0f} > ${constraints.luxury_tax_line:,.0f}")
    if total_salary > constraints.first_apron:
        warnings.append(f"first_apron_exceeded: restricted from certain transactions")

    # Check for duplicate players
    ids = [p.player_id for p in salaries if p.player_id]
    if len(ids) != len(set(ids)):
        violations.append("duplicate_players_detected")

    return {
        "legal": len(violations) == 0,
        "total_salary": total_salary,
        "cap_space": max(0, constraints.cap_limit - total_salary),
        "over_cap": max(0, total_salary - constraints.cap_limit),
        "roster_size": roster_size,
        "violations": violations,
        "warnings": warnings,
    }
