"""Role/usage simulator (spec sections 16-20). USAGE IS NOT PASSING --
this module keeps target usage, decision-touch growth, and pass
tendency as separate, explicit inputs; nothing here ever implies
"higher USG% -> automatically more assists."

DATA-HONESTY NOTE ON THE "PASSES" BASELINE: the spec's model scales a
real ``baseline_passes`` quantity. This pipeline has no real total-pass
count (see routing/recipients.py's module docstring -- only assists are
observable via play-by-play). The baseline this module actually scales
is ``baseline_decision_touches`` = real, DERIVED
(FGA + AST + TOV) per game -- a standard "how often did this player make
a scoring-relevant decision" proxy, NOT a claim about true touches or
true pass volume. Every simulated output is explicitly SIMULATED
regardless of how good this proxy is; the point of labeling it here is
that the *baseline* feeding the simulation stays honestly DERIVED, not
OBSERVED, and the simulation is conditional on this specific proxy
choice -- documented in docs/advantage-routing.md.

The formulas, verbatim from the spec:

    H = 1 + e * ((target_usage / current_usage) - 1)
    simulated_decision_touches = baseline_decision_touches * H
    simulated_passes = simulated_decision_touches * (1 + pass_tendency_change)
    simulated_assists = simulated_passes * baseline_AST_per_pass * efficiency_retention
    simulated_receiver_makes = simulated_passes * baseline_makes_per_pass * shot_generation_retention

Role saturation (section 20): retention = exp(-k * max(0, H - 1)) by
default, applied to efficiency/shot-generation retention unless the
caller supplies an explicit retention. Turnover growth moves the other
way -- see ``simulate_scenario``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.schemas import Metric
from .saturation import saturation_retention, turnover_growth_from_saturation


@dataclass(frozen=True)
class SimulationBaseline:
    """Every field here is a real, DERIVED per-game rate -- never itself
    SIMULATED. See module docstring for the decision-touch proxy."""

    baseline_decision_touches_per_game: float  # DERIVED: mean(FGA + AST + TOV)
    baseline_ast_per_game: float  # OBSERVED: mean(AST)
    baseline_tov_per_game: float  # OBSERVED: mean(TOV)
    baseline_ast_per_touch: float  # DERIVED: ast_per_game / decision_touches_per_game
    baseline_tov_per_touch: float  # DERIVED: tov_per_game / decision_touches_per_game
    baseline_makes_per_touch: float  # DERIVED: real sampled-assist-implied made-shot rate per touch (see build/build_player.py)
    current_usage_pct: float  # OBSERVED: real season-average USG%


@dataclass(frozen=True)
class ScenarioParameters:
    """The interactive sliders (sections 17-18, 28-29)."""

    target_usage_pct: float
    touch_elasticity: float = 0.6  # "e" -- fraction of role change that becomes decision-touch growth
    pass_tendency_change: float = 0.0  # e.g. +0.10 = 10% more pass-tendency
    efficiency_retention: Optional[float] = None  # None -> use saturation curve
    turnover_growth: Optional[float] = None  # None -> use saturation-linked default
    gravity_growth_pct: float = 0.0  # reserved for a future gravity-linked adjustment; see docs
    saturation_k: float = 0.55


@dataclass
class ScenarioResult:
    scenario_name: str
    touch_multiplier_h: Metric
    simulated_decision_touches: Metric
    simulated_passes: Metric
    simulated_assists: Metric
    simulated_receiver_makes: Metric
    simulated_turnovers: Metric
    efficiency_retention_used: Metric
    turnover_growth_used: Metric
    assumptions_explanation: str

    def as_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "touch_multiplier_h": self.touch_multiplier_h.as_dict(),
            "simulated_decision_touches": self.simulated_decision_touches.as_dict(),
            "simulated_passes": self.simulated_passes.as_dict(),
            "simulated_assists": self.simulated_assists.as_dict(),
            "simulated_receiver_makes": self.simulated_receiver_makes.as_dict(),
            "simulated_turnovers": self.simulated_turnovers.as_dict(),
            "efficiency_retention_used": self.efficiency_retention_used.as_dict(),
            "turnover_growth_used": self.turnover_growth_used.as_dict(),
            "assumptions_explanation": self.assumptions_explanation,
        }


def touch_multiplier(target_usage_pct: float, current_usage_pct: float, elasticity: float) -> float:
    if current_usage_pct <= 0:
        return 1.0
    return 1.0 + elasticity * ((target_usage_pct / current_usage_pct) - 1.0)


def simulate_scenario(baseline: SimulationBaseline, params: ScenarioParameters, *, scenario_name: str = "scenario") -> ScenarioResult:
    h = touch_multiplier(params.target_usage_pct, baseline.current_usage_pct, params.touch_elasticity)
    saturation = saturation_retention(h, params.saturation_k)

    efficiency_retention = params.efficiency_retention if params.efficiency_retention is not None else saturation
    # Turnover growth moves the OPPOSITE way from efficiency retention by
    # default -- more role, more saturation loss, more turnover risk.
    turnover_growth = params.turnover_growth if params.turnover_growth is not None else turnover_growth_from_saturation(saturation)

    simulated_decision_touches = baseline.baseline_decision_touches_per_game * h
    simulated_passes = simulated_decision_touches * (1.0 + params.pass_tendency_change)
    simulated_assists = simulated_passes * baseline.baseline_ast_per_touch * efficiency_retention
    simulated_receiver_makes = simulated_passes * baseline.baseline_makes_per_touch * efficiency_retention
    simulated_turnovers = simulated_passes * baseline.baseline_tov_per_touch * (1.0 + turnover_growth)

    explanation = (
        f"Target usage {params.target_usage_pct:.1f}% vs. current {baseline.current_usage_pct:.1f}% "
        f"(ratio {params.target_usage_pct / baseline.current_usage_pct:.2f}x, if current usage is positive). "
        f"With touch elasticity e={params.touch_elasticity:.2f}, {params.touch_elasticity*100:.0f}% of the "
        f"proportional role change is assumed to become additional decision touches, giving a touch multiplier "
        f"H={h:.2f}. Pass tendency is then changed by {params.pass_tendency_change*100:+.0f}%. "
        f"Role saturation (k={params.saturation_k:.2f}) retains {efficiency_retention*100:.0f}% of baseline "
        f"efficiency at this touch multiplier, and turnover risk grows by {turnover_growth*100:+.0f}% "
        f"relative to baseline. All of this is SIMULATED -- a conditional projection under these explicit "
        f"assumptions, never a forecast."
    )

    return ScenarioResult(
        scenario_name=scenario_name,
        touch_multiplier_h=Metric.simulated("touch_multiplier_h", h, method="H = 1 + e*((target_usage/current_usage)-1)"),
        simulated_decision_touches=Metric.simulated("simulated_decision_touches", simulated_decision_touches, method="baseline_decision_touches * H"),
        simulated_passes=Metric.simulated("simulated_passes", simulated_passes, method="simulated_decision_touches * (1 + pass_tendency_change)"),
        simulated_assists=Metric.simulated("simulated_assists", simulated_assists, method="simulated_passes * baseline_ast_per_touch * efficiency_retention"),
        simulated_receiver_makes=Metric.simulated("simulated_receiver_makes", simulated_receiver_makes, method="simulated_passes * baseline_makes_per_touch * efficiency_retention"),
        simulated_turnovers=Metric.simulated("simulated_turnovers", simulated_turnovers, method="simulated_passes * baseline_tov_per_touch * (1 + turnover_growth)"),
        efficiency_retention_used=Metric.simulated("efficiency_retention_used", efficiency_retention, method="exp(-k * max(0, H-1)) unless explicitly overridden"),
        turnover_growth_used=Metric.simulated("turnover_growth_used", turnover_growth, method="1/saturation - 1 unless explicitly overridden"),
        assumptions_explanation=explanation,
    )


def standard_scenarios(baseline: SimulationBaseline, target_usage_pct: float, pass_tendency_change: float = 0.0) -> dict[str, ScenarioResult]:
    """Section 20's three scenarios -- differ by efficiency retention and
    turnover growth, both explicit rather than re-derived from the same
    saturation curve, so OPTIMISTIC really is a distinct assumption."""
    scenarios = {
        "OPTIMISTIC": ScenarioParameters(target_usage_pct=target_usage_pct, pass_tendency_change=pass_tendency_change, efficiency_retention=0.97, turnover_growth=0.05, saturation_k=0.35),
        "NEUTRAL": ScenarioParameters(target_usage_pct=target_usage_pct, pass_tendency_change=pass_tendency_change, efficiency_retention=None, turnover_growth=None, saturation_k=0.55),
        "CONSERVATIVE": ScenarioParameters(target_usage_pct=target_usage_pct, pass_tendency_change=pass_tendency_change, efficiency_retention=0.80, turnover_growth=0.25, saturation_k=0.85),
    }
    return {name: simulate_scenario(baseline, params, scenario_name=name) for name, params in scenarios.items()}
