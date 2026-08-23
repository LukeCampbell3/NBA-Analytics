"""Sample-size shrinkage (spec section 14). Low-usage players are a
primary target of this whole system, and their real sample sizes are
small (a handful of sampled games, sometimes single-digit assists to a
given recipient) -- ranking or trusting raw rates naively would be
exactly the kind of false precision this project's own non-negotiable
principle forbids.

Two shrinkage tools, both textbook, both fully transparent:

  beta_binomial_shrink  -- for a single rate (successes / trials),
                            shrinks toward a prior mean using a
                            Beta(alpha, beta) prior; returns the
                            posterior mean and an equal-tailed credible
                            interval via the Beta quantile function.
  dirichlet_shrink       -- for a whole probability vector (e.g. a
                            recipient's share of assists across
                            teammates) that must sum to 1; shrinks
                            toward a prior vector with a Dirichlet
                            concentration.

Both ALWAYS return the raw (unshrunk) value alongside the shrunk one --
"never hide the raw value" (section 14) -- and the sample size and
method are always recorded.

Priors: this pipeline does not have a separately-built positional/role
prior population (that would itself require the same touch-tracking
data this whole project cannot currently reach). The default prior is
therefore the simplest defensible, fully-disclosed choice: for a single
rate, the prior mean defaults to the empirical mean across the
observed population passed in by the caller (e.g. "this player's own
overall high_value_assist_rate" for a recipient-level shrink); for a
vector, the default prior is uniform across the observed categories.
Callers may pass an explicit prior when a better one is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from scipy import stats as scipy_stats

from ..models.schemas import ShrunkRate


def beta_binomial_shrink(
    successes: int,
    trials: int,
    *,
    prior_mean: float,
    prior_strength: float = 8.0,
    credible_level: float = 0.80,
    method_note: str = "beta_binomial",
) -> ShrunkRate:
    """prior_strength is the prior's effective sample size (alpha+beta);
    prior_mean must be in (0, 1). Larger prior_strength shrinks harder
    toward prior_mean -- 8.0 is a moderate default (roughly "trust the
    prior as much as 8 real observations")."""
    if not (0.0 < prior_mean < 1.0):
        raise ValueError("prior_mean must be strictly between 0 and 1")
    alpha0 = prior_mean * prior_strength
    beta0 = (1.0 - prior_mean) * prior_strength

    raw_rate = (successes / trials) if trials > 0 else None
    alpha_post = alpha0 + successes
    beta_post = beta0 + max(0, trials - successes)
    shrunk_rate = alpha_post / (alpha_post + beta_post)

    lower_q = (1.0 - credible_level) / 2.0
    upper_q = 1.0 - lower_q
    ci_low = float(scipy_stats.beta.ppf(lower_q, alpha_post, beta_post))
    ci_high = float(scipy_stats.beta.ppf(upper_q, alpha_post, beta_post))

    return ShrunkRate(
        raw_rate=raw_rate, shrunk_rate=shrunk_rate, sample_size=trials,
        credible_interval_low=ci_low, credible_interval_high=ci_high,
        method=f"{method_note}: Beta({alpha0:.2f},{beta0:.2f}) prior (mean={prior_mean:.3f}, strength={prior_strength}), {int(credible_level*100)}% equal-tailed credible interval",
    )


@dataclass(frozen=True)
class DirichletShrinkResult:
    category: str
    raw_share: float
    shrunk_share: float
    count: int


def dirichlet_shrink(
    counts: dict[str, int],
    *,
    prior: dict[str, float] | None = None,
    prior_strength: float = 6.0,
) -> list[DirichletShrinkResult]:
    """Shrinks a multinomial count vector toward `prior` (a probability
    vector over the same categories; defaults to uniform over the
    observed categories if not given). prior_strength is the Dirichlet
    concentration's effective sample size."""
    total = sum(counts.values())
    categories = list(counts.keys())
    if not categories:
        return []
    if prior is None:
        prior = {c: 1.0 / len(categories) for c in categories}
    prior_sum = sum(prior.get(c, 0.0) for c in categories)
    if prior_sum <= 0:
        prior = {c: 1.0 / len(categories) for c in categories}
        prior_sum = 1.0

    results = []
    for c in categories:
        n_c = counts[c]
        raw_share = (n_c / total) if total else 0.0
        alpha_c = (prior.get(c, 0.0) / prior_sum) * prior_strength + n_c
        alpha_total = prior_strength + total
        shrunk_share = alpha_c / alpha_total if alpha_total else 0.0
        results.append(DirichletShrinkResult(category=c, raw_share=raw_share, shrunk_share=shrunk_share, count=n_c))
    return results
