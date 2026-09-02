from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import mean


class ReplayEvidence(StrEnum):
    EXACT_POINT_IN_TIME = "EXACT_POINT_IN_TIME"
    RECONSTRUCTED_WITH_VALID_PRIOR_DATA = "RECONSTRUCTED_WITH_VALID_PRIOR_DATA"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class GradedBet:
    probability: float
    decimal_price: float
    result: str
    decision_timestamp_utc: str
    event_start_utc: str
    evidence: ReplayEvidence


def wilson_interval(wins: int, trials: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = wins / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def evaluate(records: list[GradedBet]) -> dict:
    eligible = [record for record in records if record.evidence != ReplayEvidence.UNAVAILABLE and record.result in {"WIN", "LOSS"}]
    wins = sum(record.result == "WIN" for record in eligible)
    losses = len(eligible) - wins
    returns = [(record.decimal_price - 1.0) if record.result == "WIN" else -1.0 for record in eligible]
    probabilities = [min(max(record.probability, 1e-12), 1 - 1e-12) for record in eligible]
    outcomes = [1 if record.result == "WIN" else 0 for record in eligible]
    brier = mean((p - y) ** 2 for p, y in zip(probabilities, outcomes)) if eligible else None
    log_loss = mean(-(y * math.log(p) + (1-y) * math.log(1-p)) for p, y in zip(probabilities, outcomes)) if eligible else None
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in returns:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    low, high = wilson_interval(wins, len(eligible))
    return {"bets": len(eligible), "wins": wins, "losses": losses, "hit_rate": wins/len(eligible) if eligible else None,
            "wilson_95": [low, high], "units": sum(returns), "roi": mean(returns) if eligible else None,
            "max_drawdown_units": max_drawdown, "brier": brier, "log_loss": log_loss,
            "evidence_counts": {state.value: sum(record.evidence == state for record in records) for state in ReplayEvidence}}


def bankroll_paths(records: list[GradedBet], starting: float = 100.0) -> dict[str, float]:
    strategies = {"flat_1": lambda b: 1.0, "flat_5": lambda b: 5.0, "flat_10": lambda b: 10.0, "bankroll_1pct": lambda b: .01*b, "bankroll_2pct": lambda b: .02*b}
    balances = {}
    eligible = [r for r in records if r.evidence != ReplayEvidence.UNAVAILABLE and r.result in {"WIN", "LOSS"}]
    for name, stake_fn in strategies.items():
        balance = starting
        for record in eligible:
            stake = min(balance, stake_fn(balance))
            balance += stake * (record.decimal_price - 1.0) if record.result == "WIN" else -stake
        balances[name] = balance
    return balances


def assert_point_in_time(record: GradedBet) -> None:
    if record.decision_timestamp_utc >= record.event_start_utc:
        raise ValueError("TARGET_DATE_OUTCOME_OR_POST_START_STATE_UNAVAILABLE")
