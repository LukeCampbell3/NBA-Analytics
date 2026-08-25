"""Provisional/permanent structure lifecycle (spec section 29) + explicit
complexity budget (spec section 61: DRM must not recursively grow the
network without bound)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StructuralMutationRecord:
    mutation_id: str
    tier: str
    description: str
    residual_motivating_it: dict
    before_metrics: dict
    after_metrics: dict
    param_delta: int
    compute_delta: float
    status: str  # "PROVISIONAL" | "PERMANENT" | "REJECTED"
    decision_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class DRMBudget:
    """Bounded mutation vocabulary enforcement (spec section 61): freezes
    max experts, max mutation attempts per cycle, and total active-param
    growth allowed across the whole DRM development stage."""

    def __init__(self, max_mutation_attempts_per_cycle: int = 3, max_total_param_growth: int = 2_000_000):
        self.max_mutation_attempts_per_cycle = max_mutation_attempts_per_cycle
        self.max_total_param_growth = max_total_param_growth
        self.total_param_growth = 0
        self.history: list[StructuralMutationRecord] = []

    def can_attempt(self, attempts_this_cycle: int) -> bool:
        return attempts_this_cycle < self.max_mutation_attempts_per_cycle

    def can_afford(self, param_delta: int) -> bool:
        return self.total_param_growth + max(param_delta, 0) <= self.max_total_param_growth

    def record(self, record: StructuralMutationRecord) -> None:
        self.history.append(record)
        if record.status == "PERMANENT":
            self.total_param_growth += max(record.param_delta, 0)

    def to_report(self) -> dict:
        return {
            "max_mutation_attempts_per_cycle": self.max_mutation_attempts_per_cycle,
            "max_total_param_growth": self.max_total_param_growth,
            "total_param_growth_used": self.total_param_growth,
            "mutations": [r.to_dict() for r in self.history],
            "committed_count": sum(1 for r in self.history if r.status == "PERMANENT"),
            "rejected_count": sum(1 for r in self.history if r.status == "REJECTED"),
        }
