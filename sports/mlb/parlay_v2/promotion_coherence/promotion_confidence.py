"""Coherent promotion decision + explainable `promotion_confidence` margin.

Both live inside the shadow subpackage and never mutate any payload. A
`decide_coherent_promotion(payload, thresholds=..., penalties=...)` call
returns a `CoherentPromotionDecision` describing what should have been
published if the conservative quality overlay were the final publication
authority.

`promotion_confidence` is not a competing model. It is a bookkeeping
identity:

    promotion_margin = calibrated_joint_probability
                       - uncertainty_deduction
                       - market_disagreement_deduction
                       - shared_failure_deduction
                       - fragility_deduction
                       - break_even_probability

`calibrated_joint_probability` is the payload's own conservative joint
probability (the value the existing `public_quality_overlay` already
computes; this module never recomputes the underlying model). The
deductions are seeded from concrete signals already present on the
payload where possible, and default to 0.0 elsewhere so a caller can tune
them in shadow mode without silent behavior. `break_even_probability` is
derived from the combined decimal price the overlay already reports.

The `authorize` verdict requires all of:

    1. `parlays.eligible` is True
    2. `parlays.public_quality_overlay.action == "ACT"`
    3. every leg probability >= `min_leg_probability`
    4. `joint_probability >= min_joint_probability`
    5. `promotion_margin >= min_promotion_margin`

Any one failure returns ABSTAIN together with the specific blocking rule
and the raw components, so a downstream report can explain exactly why a
pick was not promoted rather than opaquely dropping it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional


# --- price / probability helpers ----------------------------------------

def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _break_even_probability(decimal_price: Any) -> Optional[float]:
    price = _finite(decimal_price)
    if price is None or price <= 1.0:
        return None
    return 1.0 / price


# --- thresholds & penalties ---------------------------------------------

@dataclass(frozen=True)
class PromotionThresholds:
    """Numerical bar every coherent-ACT pick must clear.

    Defaults mirror the values the existing `public_quality_overlay`
    already carries on real payloads: 0.70 per-leg, 0.50 joint, edge and
    EV floors kept for parity. `min_promotion_margin` starts at 0.0 --
    the minimally-coherent rule "the calibrated joint must at least cover
    the break-even price after deductions" -- and is tunable in shadow.
    """

    min_leg_probability: float = 0.70
    min_joint_probability: float = 0.50
    min_probability_edge: float = 0.0
    min_expected_value_per_unit: float = 0.0
    min_promotion_margin: float = 0.0


def default_thresholds() -> PromotionThresholds:
    return PromotionThresholds()


@dataclass(frozen=True)
class PromotionPenalties:
    """Explicit deductions applied to the calibrated joint probability.

    Every field defaults to 0.0 so this module never invents a penalty
    silently. A tuner passes concrete values (or a subclass returns
    payload-derived numbers) once shadow results support it. Values are
    absolute probability points, capped [0.0, 1.0] individually.
    """

    uncertainty_deduction: float = 0.0
    market_disagreement_deduction: float = 0.0
    shared_failure_deduction: float = 0.0
    fragility_deduction: float = 0.0


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# --- component extraction ------------------------------------------------

@dataclass
class PromotionConfidenceComponents:
    """Fully-explainable breakdown of the promotion margin.

    Every field is either a concrete number pulled from the payload or an
    explicit deduction supplied by the caller. Nothing here is inferred
    from a model this module owns; the audit trail is total.
    """

    calibrated_joint_probability: Optional[float]
    break_even_probability: Optional[float]
    leg_probabilities: list[float] = field(default_factory=list)
    combined_decimal_price: Optional[float] = None
    expected_value_per_unit: Optional[float] = None
    probability_edge: Optional[float] = None
    uncertainty_deduction: float = 0.0
    market_disagreement_deduction: float = 0.0
    shared_failure_deduction: float = 0.0
    fragility_deduction: float = 0.0
    total_deductions: float = 0.0
    promotion_margin: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_overlay(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    parlays = payload.get("parlays") or {}
    overlay = parlays.get("public_quality_overlay") or {}
    return overlay


def promotion_confidence_components(
    payload: Mapping[str, Any],
    *,
    penalties: PromotionPenalties | None = None,
) -> PromotionConfidenceComponents:
    """Compute the explainable promotion-confidence components for a payload.

    Reads only fields the existing overlay already exposes. Returns a
    populated `PromotionConfidenceComponents`; the `promotion_margin` is
    None only when the payload lacks either a calibrated joint
    probability or a combined decimal price -- both required for a
    price-vs-probability comparison to even be defined.
    """
    penalties = penalties or PromotionPenalties()
    overlay = _extract_overlay(payload)

    joint = _finite(overlay.get("joint_probability"))
    price = _finite(overlay.get("combined_decimal_price"))
    legs = [
        _finite(p)
        for p in (overlay.get("leg_probabilities") or [])
    ]
    legs = [p for p in legs if p is not None]
    edge = _finite(overlay.get("probability_edge") or overlay.get("edge"))
    ev = _finite(overlay.get("expected_value_per_unit"))
    break_even = _break_even_probability(price)

    total_deductions = _clip01(penalties.uncertainty_deduction) \
        + _clip01(penalties.market_disagreement_deduction) \
        + _clip01(penalties.shared_failure_deduction) \
        + _clip01(penalties.fragility_deduction)

    margin: Optional[float]
    if joint is None or break_even is None:
        margin = None
    else:
        margin = joint - total_deductions - break_even

    return PromotionConfidenceComponents(
        calibrated_joint_probability=joint,
        break_even_probability=break_even,
        leg_probabilities=legs,
        combined_decimal_price=price,
        expected_value_per_unit=ev,
        probability_edge=edge,
        uncertainty_deduction=_clip01(penalties.uncertainty_deduction),
        market_disagreement_deduction=_clip01(penalties.market_disagreement_deduction),
        shared_failure_deduction=_clip01(penalties.shared_failure_deduction),
        fragility_deduction=_clip01(penalties.fragility_deduction),
        total_deductions=total_deductions,
        promotion_margin=margin,
    )


# --- coherent decision --------------------------------------------------

@dataclass
class CoherentPromotionDecision:
    """Shadow decision alongside the payload's live `parlays.action`.

    `action` is "ACT" only when every coherence rule passes; otherwise
    "ABSTAIN". `blocking_reasons` names every failed rule (order is stable
    and matches the check order below), so a report can explain exactly
    why the coherent gate declined. `live_action_agrees` records whether
    the payload's live action matches this shadow verdict, which is the
    quantity a shadow replay is measuring.
    """

    action: str  # "ACT" | "ABSTAIN"
    live_action: str
    live_action_agrees: bool
    blocking_reasons: list[str]
    components: PromotionConfidenceComponents
    thresholds: PromotionThresholds
    overlay_action: Optional[str]
    slate_date: Optional[str]
    candidate_id: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "live_action": self.live_action,
            "live_action_agrees": self.live_action_agrees,
            "blocking_reasons": list(self.blocking_reasons),
            "components": self.components.to_dict(),
            "thresholds": asdict(self.thresholds),
            "overlay_action": self.overlay_action,
            "slate_date": self.slate_date,
            "candidate_id": self.candidate_id,
        }


def _leg_index_reasons(legs: Iterable[float], floor: float) -> list[str]:
    reasons: list[str] = []
    for idx, prob in enumerate(legs, start=1):
        if prob is None or prob < floor:
            reasons.append(f"leg_{idx}_probability_below_{int(floor * 100)}pct")
    return reasons


def decide_coherent_promotion(
    payload: Mapping[str, Any],
    *,
    thresholds: PromotionThresholds | None = None,
    penalties: PromotionPenalties | None = None,
) -> CoherentPromotionDecision:
    """Return the coherent shadow decision for a persisted payload.

    Rule order is stable so downstream diffs and reports read the same
    way every time:

        1. parlay-eligibility (payload says the slate itself is
           actionable)
        2. quality-overlay concurrence (the payload's own conservative
           overlay is ACT)
        3. per-leg probability floor
        4. joint probability floor
        5. promotion-margin floor (calibrated joint minus deductions
           minus break-even >= min_promotion_margin)

    Every failed rule appears in `blocking_reasons`; the caller sees the
    full picture, not just the first failure.
    """
    thresholds = thresholds or default_thresholds()
    penalties = penalties or PromotionPenalties()

    parlays = payload.get("parlays") or {}
    overlay = _extract_overlay(payload)
    live_action = str(parlays.get("action") or "ABSTAIN").upper()
    overlay_action_raw = overlay.get("action")
    overlay_action = str(overlay_action_raw).upper() if overlay_action_raw else None
    slate_date = payload.get("run_date") or payload.get("slate_date")
    selected = parlays.get("selected_parlay") or {}
    candidate_id = selected.get("candidate_id")

    components = promotion_confidence_components(payload, penalties=penalties)

    blocking: list[str] = []

    if not bool(parlays.get("eligible", True)):
        blocking.append("slate_not_eligible")

    if overlay_action != "ACT":
        blocking.append("quality_overlay_not_act")

    blocking.extend(_leg_index_reasons(components.leg_probabilities, thresholds.min_leg_probability))

    if (
        components.calibrated_joint_probability is None
        or components.calibrated_joint_probability < thresholds.min_joint_probability
    ):
        blocking.append(f"joint_probability_below_{int(thresholds.min_joint_probability * 100)}pct")

    if components.promotion_margin is None:
        blocking.append("promotion_margin_undefined")
    elif components.promotion_margin < thresholds.min_promotion_margin:
        blocking.append("promotion_margin_below_floor")

    if (
        components.probability_edge is not None
        and components.probability_edge < thresholds.min_probability_edge
    ):
        blocking.append("probability_edge_below_floor")

    if (
        components.expected_value_per_unit is not None
        and components.expected_value_per_unit < thresholds.min_expected_value_per_unit
    ):
        blocking.append("expected_value_below_floor")

    action = "ACT" if not blocking else "ABSTAIN"
    live_agrees = live_action == action

    return CoherentPromotionDecision(
        action=action,
        live_action=live_action,
        live_action_agrees=live_agrees,
        blocking_reasons=blocking,
        components=components,
        thresholds=thresholds,
        overlay_action=overlay_action,
        slate_date=str(slate_date) if slate_date else None,
        candidate_id=candidate_id,
    )
