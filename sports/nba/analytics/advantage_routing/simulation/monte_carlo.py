"""Monte Carlo scenario uncertainty (spec section 21). Simulations are
never presented as deterministic forecasts -- this module draws the
underlying per-touch rates from their Beta posteriors (same prior
convention as stats/shrinkage.py) and propagates each draw through
simulation/usage.py's scenario formula, reporting median/P10/P25/P75/P90
rather than a single number.

Reproducible: every call takes an explicit integer seed and uses a
dedicated numpy Generator (never the global numpy random state), so the
same inputs always produce byte-identical output distributions --
tested directly in tests/test_monte_carlo.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .saturation import saturation_retention, turnover_growth_from_saturation
from .usage import ScenarioParameters, SimulationBaseline, touch_multiplier

DEFAULT_SEED = 20260823
DEFAULT_DRAWS = 4000


@dataclass(frozen=True)
class RateObservation:
    """successes/trials for one per-touch rate, used to build its Beta
    posterior for Monte Carlo draws -- mirrors
    stats.shrinkage.beta_binomial_shrink's prior convention."""

    successes: int
    trials: int
    prior_mean: float
    prior_strength: float = 8.0

    def posterior_alpha_beta(self) -> tuple[float, float]:
        alpha0 = self.prior_mean * self.prior_strength
        beta0 = (1.0 - self.prior_mean) * self.prior_strength
        return alpha0 + self.successes, beta0 + max(0, self.trials - self.successes)


@dataclass(frozen=True)
class MonteCarloInputs:
    decision_touches: RateObservation
    ast_per_touch: RateObservation
    makes_per_touch: RateObservation
    tov_per_touch: RateObservation
    baseline_decision_touches_per_game: float
    current_usage_pct: float


@dataclass
class PercentileSummary:
    median: float
    p10: float
    p25: float
    p75: float
    p90: float

    def as_dict(self) -> dict:
        return {"median": self.median, "p10": self.p10, "p25": self.p25, "p75": self.p75, "p90": self.p90}


def _percentiles(draws: np.ndarray) -> PercentileSummary:
    q = np.percentile(draws, [10, 25, 50, 75, 90])
    return PercentileSummary(median=float(q[2]), p10=float(q[0]), p25=float(q[1]), p75=float(q[3]), p90=float(q[4]))


@dataclass
class MonteCarloResult:
    scenario_name: str
    n_draws: int
    seed: int
    assists: PercentileSummary
    turnovers: PercentileSummary
    receiver_makes: PercentileSummary
    decision_touches: PercentileSummary

    def as_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "n_draws": self.n_draws,
            "seed": self.seed,
            "status": "SIMULATED",
            "method": "Beta-posterior draws of per-touch rates, propagated through the scenario formula; reproducible via the fixed seed",
            "assists": self.assists.as_dict(),
            "turnovers": self.turnovers.as_dict(),
            "receiver_makes": self.receiver_makes.as_dict(),
            "decision_touches": self.decision_touches.as_dict(),
        }


def run_monte_carlo(
    inputs: MonteCarloInputs,
    params: ScenarioParameters,
    *,
    scenario_name: str = "scenario",
    n_draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)

    ast_alpha, ast_beta = inputs.ast_per_touch.posterior_alpha_beta()
    makes_alpha, makes_beta = inputs.makes_per_touch.posterior_alpha_beta()
    tov_alpha, tov_beta = inputs.tov_per_touch.posterior_alpha_beta()

    ast_draws = rng.beta(ast_alpha, ast_beta, size=n_draws)
    makes_draws = rng.beta(makes_alpha, makes_beta, size=n_draws)
    tov_draws = rng.beta(tov_alpha, tov_beta, size=n_draws)

    h = touch_multiplier(params.target_usage_pct, inputs.current_usage_pct, params.touch_elasticity)
    saturation = saturation_retention(h, params.saturation_k)
    efficiency_retention = params.efficiency_retention if params.efficiency_retention is not None else saturation
    turnover_growth = params.turnover_growth if params.turnover_growth is not None else turnover_growth_from_saturation(saturation)

    simulated_decision_touches = inputs.baseline_decision_touches_per_game * h
    simulated_passes = simulated_decision_touches * (1.0 + params.pass_tendency_change)

    assist_draws = simulated_passes * ast_draws * efficiency_retention
    makes_draws_out = simulated_passes * makes_draws * efficiency_retention
    tov_draws_out = simulated_passes * tov_draws * (1.0 + turnover_growth)
    touches_draws = np.full(n_draws, simulated_decision_touches)

    return MonteCarloResult(
        scenario_name=scenario_name, n_draws=n_draws, seed=seed,
        assists=_percentiles(assist_draws),
        turnovers=_percentiles(tov_draws_out),
        receiver_makes=_percentiles(makes_draws_out),
        decision_touches=_percentiles(touches_draws),
    )
