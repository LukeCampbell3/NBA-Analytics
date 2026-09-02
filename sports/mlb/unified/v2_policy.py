from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


SUPPORTED_V2_CAPABILITIES = frozenset({"batter_hits", "batter_total_bases"})


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if -100.0 < odds < 100.0:
        raise ValueError("invalid American odds")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def implied_probability(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def remove_two_way_vig(side_odds: float, opposite_odds: float) -> tuple[float, float]:
    first, second = implied_probability(side_odds), implied_probability(opposite_odds)
    total = first + second
    if total <= 0:
        raise ValueError("invalid two-way market")
    return first / total, second / total


def discrete_over_probability(pmf: dict[int, float], line: float) -> float:
    """Exact probability of sportsbook OVER settlement for a discrete statistic."""
    if not pmf or any(value < 0 for value in pmf.values()):
        raise ValueError("invalid probability mass")
    total = sum(pmf.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("probability mass must sum to one")
    return sum(probability for value, probability in pmf.items() if value > line)


def discrete_settlement(value: int, line: float, side: str) -> str:
    side = side.upper()
    if value == line:
        return "push"
    won = value > line if side == "OVER" else value < line
    return "won" if won else "lost"


def poisson_binomial_cdf(probabilities: Iterable[float], observed_wins: int) -> float:
    probabilities = list(probabilities)
    distribution = [1.0] + [0.0] * len(probabilities)
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Bernoulli probability outside [0, 1]")
        for wins in range(len(probabilities), 0, -1):
            distribution[wins] = distribution[wins] * (1.0 - probability) + distribution[wins - 1] * probability
        distribution[0] *= 1.0 - probability
    return sum(distribution[: max(0, observed_wins) + 1])


@dataclass(frozen=True)
class UnifiedPolicyV2:
    name: str = "UNIFIED_POLICY_V2"
    evidence_state: str = "DEVELOPMENT"
    minimum_usable_probability: float = 0.60
    minimum_probability_edge: float = 0.01
    minimum_conservative_ev: float = 0.0
    maximum_uncertainty: float = 0.10
    minimum_support: int = 50
    maximum_quote_age_seconds: int = 600
    require_two_sided_no_vig_market: bool = True
    require_exact_selection_ids: bool = True
    out_of_support_fails_closed: bool = True
    production_authorized: bool = False

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def evaluate_v2_candidate(candidate: dict[str, Any], policy: UnifiedPolicyV2 = UnifiedPolicyV2()) -> dict[str, Any]:
    """Fail-closed admissibility first; ranking is deliberately separate."""
    reasons: list[str] = []
    capability = str(candidate.get("capability") or "")
    if capability not in SUPPORTED_V2_CAPABILITIES:
        reasons.append("CAPABILITY_NOT_SUPPORTED")
    if candidate.get("identity_status") != "CONFIRMED":
        reasons.append("IDENTITY_INVALID")
    if candidate.get("lineup_status") != "CONFIRMED":
        reasons.append("LINEUP_INVALID")
    if candidate.get("player_status") not in {"CONFIRMED", "ACTIVE"}:
        reasons.append("PLAYER_STATUS_INVALID")
    support = candidate.get("support_size")
    if support is None or int(support) < policy.minimum_support:
        reasons.append("SUPPORT_INVALID")
    if candidate.get("support_status") == "OUT_OF_SUPPORT":
        reasons.append("OUT_OF_SUPPORT")
    quote_time = _parse_timestamp(candidate.get("quote_timestamp"))
    decision_time = _parse_timestamp(candidate.get("decision_timestamp"))
    if quote_time is None or decision_time is None:
        reasons.append("QUOTE_FRESHNESS_UNPROVABLE")
    elif not 0 <= (decision_time - quote_time).total_seconds() <= policy.maximum_quote_age_seconds:
        reasons.append("QUOTE_STALE")
    if policy.require_exact_selection_ids and not (candidate.get("market_id") and candidate.get("selection_id")):
        reasons.append("EXACT_SELECTION_UNAVAILABLE")
    probability = candidate.get("usable_probability")
    uncertainty = candidate.get("uncertainty")
    odds = candidate.get("quoted_odds")
    if probability is None:
        reasons.append("PROBABILITY_UNAVAILABLE")
    if uncertainty is None or float(uncertainty) > policy.maximum_uncertainty:
        reasons.append("UNCERTAINTY_INVALID")
    if odds is None:
        reasons.append("PRICE_UNAVAILABLE")
    edge = ev = None
    if probability is not None and odds is not None:
        decimal = american_to_decimal(float(odds))
        edge = float(probability) - 1.0 / decimal
        ev = float(probability) * decimal - 1.0
        if float(probability) < policy.minimum_usable_probability:
            reasons.append("USABLE_PROBABILITY_BELOW_FLOOR")
        if edge < policy.minimum_probability_edge:
            reasons.append("PROBABILITY_EDGE_BELOW_FLOOR")
        if ev <= policy.minimum_conservative_ev:
            reasons.append("NON_POSITIVE_CONSERVATIVE_EV")
    return {**candidate, "probability_edge": edge, "conservative_expected_value": ev,
            "admissible": not reasons, "rejection_reasons": sorted(set(reasons)),
            "policy_hash": policy.policy_hash}


def rank_admissible(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    admissible = [candidate for candidate in candidates if candidate.get("admissible")]
    return sorted(admissible, key=lambda row: (
        float(row.get("conservative_expected_value") or -math.inf),
        float(row.get("probability_edge") or -math.inf),
        float(row.get("usable_probability") or -math.inf),
    ), reverse=True)
