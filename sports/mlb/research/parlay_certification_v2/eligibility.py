from __future__ import annotations

"""OPERATIONAL ELIGIBILITY (mission section 2) -- external, immutable, and
structurally incapable of depending on model output.

E_t = 1 iff PREDECLARED OPERATIONAL eligibility holds. Eligibility MUST
NOT depend on model probability/score/edge/confidence, the existence of a
qualifying candidate/pair, or any later result. `EligibilityInputs` below
enforces this by construction: it is a fixed dataclass of purely
operational booleans (feed reachability, slate existence, system-component
health, cutoff timing). There is no field through which a model score or
pair could be passed in, so it is impossible to compute E from them.

"Slate exists but no pair passes" / "prices exist but none are
attractive" / "model has low confidence" / "all candidates fail policy
thresholds" are ABSTENTIONS (E=1, A=0), decided entirely downstream in
policy.py -- this module has no visibility into policy/action outcomes at
all, so it cannot be made to depend on them even by mistake.
"""

from dataclasses import dataclass

ELIGIBILITY_VERSION = "ELIGIBILITY_V1"


@dataclass(frozen=True)
class EligibilityInputs:
    """The COMPLETE set of operational fields eligibility may read. If a
    future need arises to gate on something new, add a field here
    explicitly -- do not smuggle a model/decision signal into an existing
    boolean's meaning."""

    date: str
    required_feed_available: bool
    slate_has_mlb_games: bool
    required_system_component_available: bool
    decision_cutoff_met: bool


@dataclass(frozen=True)
class EligibilityDecision:
    date: str
    eligible: bool
    reason: str
    eligibility_version: str


def evaluate_eligibility(inputs: EligibilityInputs) -> EligibilityDecision:
    if not inputs.required_feed_available:
        return EligibilityDecision(inputs.date, False, "required_feed_unavailable", ELIGIBILITY_VERSION)
    if not inputs.slate_has_mlb_games:
        return EligibilityDecision(inputs.date, False, "no_mlb_slate", ELIGIBILITY_VERSION)
    if not inputs.required_system_component_available:
        return EligibilityDecision(inputs.date, False, "required_system_component_unavailable", ELIGIBILITY_VERSION)
    if not inputs.decision_cutoff_met:
        return EligibilityDecision(inputs.date, False, "decision_cutoff_not_met", ELIGIBILITY_VERSION)
    return EligibilityDecision(inputs.date, True, "operationally_eligible", ELIGIBILITY_VERSION)
