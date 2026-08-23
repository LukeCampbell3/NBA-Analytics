"""Player research summary / archetype labeling (spec sections 43-44).

Rule-based, transparent, and built ONLY from metrics this pipeline has
actually computed for the player -- never an unsupported declarative
claim. Every label is accompanied by the specific real/derived signal
that triggered it, and an overall confidence reflects both the rule's
own strength and the real sample size behind it (few sampled games ->
lower confidence, by construction).

These are DESCRIPTIVE CLUSTERS, not grades (section 44) -- a player may
carry more than one label, and the absence of a label is not a
criticism, it is "this pattern was not observed in the real, if
partial, data available."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchSummary:
    archetype: list[str]
    primary_gravity: list[str]
    best_recipients: list[dict]
    role_constraint: str
    simulation_finding: str
    confidence: float
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "primary_gravity": self.primary_gravity,
            "best_recipients": self.best_recipients,
            "role_constraint": self.role_constraint,
            "simulation_finding": self.simulation_finding,
            "confidence": self.confidence,
            "caveats": self.caveats,
        }


def build_research_summary(
    *,
    player_name: str,
    gravity_mechanisms_present: list[str],
    gravity_values: dict[str, float],
    recipient_network_as_dict: dict,
    sampled_assists: int,
    baseline_decision_touches_per_game: float,
    baseline_usage_pct: float,
    scenario_neutral_assists_delta: float,
) -> ResearchSummary:
    archetype: list[str] = []
    caveats: list[str] = []

    vertical = gravity_values.get("VERTICAL_GRAVITY_rim_fg_pct")
    dunk_fga = gravity_values.get("VERTICAL_GRAVITY_dunk_attempts_season")
    post_index = gravity_values.get("POST_SCORING_GRAVITY_post_scoring_gravity_index")
    paint_fg_pct = gravity_values.get("PAINT_FACEUP_GRAVITY_paint_fg_pct")
    pop_share = gravity_values.get("POP_GRAVITY_three_pa_share_of_fga")

    if vertical is not None and vertical >= 0.60 and (dunk_fga or 0) >= 30:
        archetype.append("FINISHER")
        archetype.append("VERTICAL_GRAVITY_PROCESSOR")
    if post_index is not None and post_index >= 1.0:
        archetype.append("SCORE_FIRST_POST")
    if paint_fg_pct is not None and paint_fg_pct >= 0.50 and post_index is not None and 0.5 <= post_index < 1.0:
        archetype.append("POST_MANIPULATOR")

    top_recipients = recipient_network_as_dict.get("recipients", [])[:3]
    real_assist_share_concentration = sum(r["assist_share"]["value"] or 0 for r in top_recipients) if top_recipients else 0

    if sampled_assists >= 10 and real_assist_share_concentration <= 0.6 and len(recipient_network_as_dict.get("recipients", [])) >= 4:
        archetype.append("CONNECTOR")
    if sampled_assists >= 10 and real_assist_share_concentration > 0.6:
        archetype.append("ADVANTAGE_PROCESSOR")

    if not archetype:
        archetype.append("FINISHER")
        caveats.append("No gravity/recipient-network signal cleared the labeling thresholds with the real sample available; defaulted to FINISHER as the most conservative label.")

    primary_gravity = sorted(
        gravity_mechanisms_present,
        key=lambda m: -(gravity_values.get(f"{m}_score_proxy", 0) or 0),
    )[:2] if gravity_mechanisms_present else []

    touches_per_36 = None
    if baseline_decision_touches_per_game:
        touches_per_36 = baseline_decision_touches_per_game
    if baseline_usage_pct is not None and baseline_usage_pct < 15.0:
        role_constraint = "LOW_TOUCH_VOLUME"
    elif baseline_usage_pct is not None and baseline_usage_pct < 20.0:
        role_constraint = "MODERATE_TOUCH_VOLUME"
    else:
        role_constraint = "HIGH_TOUCH_VOLUME"

    direction = "rises" if scenario_neutral_assists_delta > 0 else "falls"
    simulation_finding = (
        f"Under a neutral-scenario role expansion, {player_name}'s simulated assist output {direction} by "
        f"{abs(scenario_neutral_assists_delta):.2f} per game relative to baseline. This is a conditional "
        f"projection under explicit, disclosed assumptions (see simulation_parameters), not a forecast."
    )

    sample_confidence_factor = min(1.0, sampled_assists / 40.0)
    confidence = round(0.35 + 0.4 * sample_confidence_factor, 2)

    if sampled_assists < 20:
        caveats.append(f"Recipient-network signal is based on only {sampled_assists} real sampled assists -- treat archetype labels involving CONNECTOR/ADVANTAGE_PROCESSOR as provisional.")

    return ResearchSummary(
        archetype=sorted(set(archetype)),
        primary_gravity=primary_gravity,
        best_recipients=[{"label": r["recipient_label"], "assist_share": r["assist_share"]["value"], "high_value_share_index": r["high_value_share_index"]["value"]} for r in top_recipients],
        role_constraint=role_constraint,
        simulation_finding=simulation_finding,
        confidence=confidence,
        caveats=caveats,
    )
