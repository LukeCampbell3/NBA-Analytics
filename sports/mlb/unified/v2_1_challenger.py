from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .v2_policy import UnifiedPolicyV2, american_to_decimal, evaluate_v2_candidate


BASELINE_POLICY_HASH = "52deb038a076b39a1bc840b77ae26648d9e4ffa20194135e7d48b9761edbc611"


@dataclass(frozen=True)
class UncertaintyComponents:
    model: float
    calibration: float
    market_disagreement: float
    player_role: float
    lineup: float
    opportunity: float
    data_support: float
    distribution_shift: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} uncertainty outside [0, 1]")

    @property
    def total(self) -> float:
        """Conservative root-sum-square; components are retained for diagnosis."""
        self.validate()
        return min(1.0, math.sqrt(sum(float(value) ** 2 for value in asdict(self).values())))


@dataclass(frozen=True)
class UnifiedPolicyV21:
    name: str = "UNIFIED_MLB_V2_1"
    evidence_state: str = "PROSPECTIVE_SHADOW"
    minimum_usable_probability: float = .60
    minimum_probability_lcb: float = .55
    minimum_edge_lcb: float = 0.0
    minimum_conservative_ev: float = 0.0
    maximum_uncertainty: float = .10
    minimum_support: int = 50
    maximum_quote_age_seconds: int = 600
    lcb_multiplier: float = 1.0
    top_k: int = 2
    one_market_per_player: bool = True
    pareto_filter: bool = True
    uncertainty_rule: str = "ROOT_SUM_SQUARE_EIGHT_COMPONENTS_V1"
    probability_lcb_rule: str = "P_USABLE_MINUS_1X_U_TOTAL"
    ranking_rule: str = "PARETO_THEN_LEXICOGRAPHIC_PLCB_EV_EDGE_SUPPORT_NEG_U"
    duplicate_exposure_rule: str = "ONE_MARKET_PER_PLAYER"
    hits_event_model: str = "PA_MIXTURE_BINOMIAL_V1"
    total_bases_event_model: str = "PA_MIXTURE_DISCRETE_CONVOLUTION_V1"
    price_sensitivity_rule: str = "LIVE_PRICE_MUST_PRESERVE_POSITIVE_LCB_EV"
    production_authorized: bool = False

    @property
    def policy_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def lower_bound_probability(probability: float, uncertainty: float, multiplier: float = 1.0) -> float:
    if not 0 <= probability <= 1 or not 0 <= uncertainty <= 1 or multiplier < 0:
        raise ValueError("invalid lower-bound inputs")
    return max(0.0, probability - multiplier * uncertainty)


def maximum_acceptable_negative_price(probability: float) -> float:
    """Break-even American price; live odds must be no more expensive."""
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly inside (0, 1)")
    if probability >= .5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def hits_distribution(pa_distribution: dict[int, float], hit_probability_per_pa: float) -> dict[int, float]:
    """Opportunity-conditioned hit distribution as a mixture over projected PA."""
    if not math.isclose(sum(pa_distribution.values()), 1.0, abs_tol=1e-9):
        raise ValueError("PA distribution must sum to one")
    if not 0 <= hit_probability_per_pa <= 1:
        raise ValueError("invalid per-PA hit probability")
    result: dict[int, float] = {}
    for pa, pa_probability in pa_distribution.items():
        if pa < 0 or pa_probability < 0:
            raise ValueError("invalid PA distribution")
        for hits in range(pa + 1):
            probability = (math.comb(pa, hits) * hit_probability_per_pa**hits
                           * (1-hit_probability_per_pa)**(pa-hits))
            result[hits] = result.get(hits, 0.0) + pa_probability * probability
    return result


def total_bases_distribution(pa_distribution: dict[int, float], per_pa_tb: dict[int, float]) -> dict[int, float]:
    """Exact convolution of 0/1/2/3/4+ TB mass over opportunity uncertainty."""
    if not math.isclose(sum(pa_distribution.values()), 1.0, abs_tol=1e-9):
        raise ValueError("PA distribution must sum to one")
    if not math.isclose(sum(per_pa_tb.values()), 1.0, abs_tol=1e-9) or any(value < 0 for value in per_pa_tb.values()):
        raise ValueError("per-PA TB distribution must sum to one")
    result: dict[int, float] = {}
    for pa, pa_probability in pa_distribution.items():
        distribution = {0: 1.0}
        for _ in range(pa):
            next_distribution: dict[int, float] = {}
            for total, total_probability in distribution.items():
                for bases, event_probability in per_pa_tb.items():
                    next_distribution[total+bases] = next_distribution.get(total+bases, 0.0) + total_probability*event_probability
            distribution = next_distribution
        for total, probability in distribution.items():
            result[total] = result.get(total, 0.0) + pa_probability*probability
    return result


def line_sensitivity(distribution: dict[int, float], offered: Iterable[dict[str, float]], uncertainty: float) -> list[dict[str, float]]:
    """Evaluate only observed price/line pairs; never synthesize wagers."""
    output = []
    for quote in offered:
        line, odds = float(quote["line"]), float(quote["odds"])
        probability = sum(mass for value, mass in distribution.items() if value > line)
        probability_lcb = lower_bound_probability(probability, uncertainty)
        decimal = american_to_decimal(odds)
        output.append({"line": line, "odds": odds, "probability": probability,
                       "probability_lcb": probability_lcb,
                       "conservative_ev": probability_lcb*decimal-1})
    return sorted(output, key=lambda row: (row["conservative_ev"], row["probability_lcb"]), reverse=True)


def _score_tuple(row: dict[str, Any]) -> tuple[float, ...]:
    # Lexicographic ordering preserves distinct objectives rather than hiding
    # them in an unsupported weighted scalar.
    return (
        float(row.get("probability_lcb") or -math.inf),
        float(row.get("conservative_expected_value") or -math.inf),
        float(row.get("edge_lcb") or -math.inf),
        float(row.get("support_score") or -math.inf),
        -float(row.get("uncertainty") or math.inf),
        str(row.get("candidate_id") or ""),
    )


def pareto_frontier(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(candidates)
    frontier = []
    for row in rows:
        probability, ev = row.get("probability_lcb"), row.get("conservative_expected_value")
        if probability is None or ev is None:
            continue
        dominated = any(
            other is not row
            and float(other.get("probability_lcb", -math.inf)) >= float(probability)
            and float(other.get("conservative_expected_value", -math.inf)) >= float(ev)
            and (float(other.get("probability_lcb", -math.inf)) > float(probability)
                 or float(other.get("conservative_expected_value", -math.inf)) > float(ev))
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return frontier


def evaluate_challenger_candidate(candidate: dict[str, Any], policy: UnifiedPolicyV21 = UnifiedPolicyV21()) -> dict[str, Any]:
    components = candidate.get("uncertainty_components")
    if isinstance(components, UncertaintyComponents):
        components = asdict(components)
    reasons: list[str] = []
    if not isinstance(components, dict) or set(components) != set(UncertaintyComponents.__annotations__):
        uncertainty = None
        reasons.append("UNCERTAINTY_COMPONENTS_UNAVAILABLE")
    else:
        try:
            uncertainty = UncertaintyComponents(**components).total
        except (TypeError, ValueError):
            uncertainty = None
            reasons.append("UNCERTAINTY_COMPONENTS_INVALID")
    baseline_input = {
        **candidate, "uncertainty": uncertainty,
        "support_size": candidate.get("support_score"),
    }
    universal = evaluate_v2_candidate(baseline_input, UnifiedPolicyV2(
        minimum_usable_probability=policy.minimum_usable_probability,
        minimum_probability_edge=0.0,
        minimum_conservative_ev=policy.minimum_conservative_ev,
        maximum_uncertainty=policy.maximum_uncertainty,
        minimum_support=policy.minimum_support,
        maximum_quote_age_seconds=policy.maximum_quote_age_seconds,
    ))
    reasons.extend(universal["rejection_reasons"])
    probability = candidate.get("usable_probability")
    odds = candidate.get("quoted_odds")
    probability_lcb = edge_lcb = conservative_ev = min_price = None
    if probability is not None and uncertainty is not None:
        probability_lcb = lower_bound_probability(float(probability), uncertainty, policy.lcb_multiplier)
        if probability_lcb < policy.minimum_probability_lcb:
            reasons.append("PROBABILITY_LCB_BELOW_FLOOR")
        min_price = maximum_acceptable_negative_price(probability_lcb) if 0 < probability_lcb < 1 else None
        if odds is not None:
            decimal = american_to_decimal(float(odds))
            market_probability = 1.0 / decimal
            edge_lcb = probability_lcb - market_probability
            conservative_ev = probability_lcb * decimal - 1.0
            if edge_lcb <= policy.minimum_edge_lcb:
                reasons.append("EDGE_LCB_NOT_POSITIVE")
            if conservative_ev <= policy.minimum_conservative_ev:
                reasons.append("NON_POSITIVE_LCB_EV")
    evaluated = {
        **candidate, "uncertainty_components": components, "uncertainty": uncertainty,
        "probability_lcb": probability_lcb, "edge_lcb": edge_lcb,
        "conservative_expected_value": conservative_ev,
        "minimum_acceptable_price": min_price,
        "admissible": not reasons, "rejection_reasons": sorted(set(reasons)),
        "policy_hash": policy.policy_hash, "baseline_policy_hash": BASELINE_POLICY_HASH,
    }
    evaluated["ranking_key"] = list(_score_tuple(evaluated))
    return evaluated


def select_challenger(candidates: Iterable[dict[str, Any]], policy: UnifiedPolicyV21 = UnifiedPolicyV21()) -> dict[str, Any]:
    evaluated = [evaluate_challenger_candidate(candidate, policy) for candidate in candidates]
    admissible = [candidate for candidate in evaluated if candidate["admissible"]]
    frontier = pareto_frontier(admissible) if policy.pareto_filter else admissible
    ranked = sorted(frontier, key=_score_tuple, reverse=True)
    if policy.one_market_per_player:
        deduplicated, used = [], set()
        for row in ranked:
            player = row.get("player_id") or row.get("subject_id")
            if player in used:
                row["selection_status"] = "ADMISSIBLE_NOT_SELECTED_CORRELATED_PLAYER"
                continue
            used.add(player)
            deduplicated.append(row)
        ranked = deduplicated
    for position, row in enumerate(ranked, 1):
        row["ranking_position"] = position
        row["selection_status"] = "SELECTED" if position <= policy.top_k else "ADMISSIBLE_NOT_SELECTED_TOP_K"
    selected = ranked[:policy.top_k]
    return {"evaluated": evaluated, "admissible": admissible, "pareto_frontier": frontier,
            "ranked": ranked, "selected": selected, "policy_hash": policy.policy_hash}
