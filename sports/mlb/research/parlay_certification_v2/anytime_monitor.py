from __future__ import annotations

"""PARLAY_REFERENCE_ANYTIME_CERT_V1 (mission section 7/8) -- a simple,
auditable, conservative reference sequential monitor for the three bounded
predictable-mean processes G_C, G_L, G_V. Inference lives here, explicitly
and inspectably -- not hidden inside the predictive/world model.

ASSUMPTIONS (do not call this a generic "confidence sequence" without
them): G_C,t / G_L,t / G_V,t are bounded in known, fixed ranges (below),
evaluated only at ELIGIBLE times t=1..n under the sequential/predictable-
mean interpretation this project uses for MLB (section 1/10) rather than
an i.i.d./stationarity assumption. Per-step significance is allocated as
    alpha_{j,t} = alpha_j / [t(t+1)],  since sum_{t>=1} 1/[t(t+1)] = 1,
a transparent union bound over all t simultaneously -- this is what makes
the procedure anytime-valid IN THE UNION-BOUND SENSE used here (protects
any single look at any t, jointly across all t, at overall level alpha_j),
via a per-step Hoeffding-style concentration argument for the bounded
increment at that t. A sharper betting/e-process procedure may replace
this later ONLY if independently tested against this reference and its own
assumptions documented (section 7) -- this reference stays authoritative
until then.
"""

import math
from dataclasses import dataclass

REFERENCE_MONITOR_VERSION = "PARLAY_REFERENCE_ANYTIME_CERT_V1"


@dataclass(frozen=True)
class ProcessBounds:
    low: float
    high: float

    @property
    def width(self) -> float:
        return float(self.high - self.low)


def g_c_bounds(c: float) -> ProcessBounds:
    return ProcessBounds(low=-c, high=1.0 - c)


def g_l_bounds(r: float) -> ProcessBounds:
    return ProcessBounds(low=-r, high=1.0 - r)


def g_v_bounds(delta: float, r_max: float) -> ProcessBounds:
    return ProcessBounds(low=min(0.0, -1.0 - delta), high=max(0.0, r_max - delta))


def g_c_value(a_t: int, c: float) -> float:
    return float(a_t) - c


def g_l_value(a_t: int, ell_t: int, r: float) -> float:
    return float(a_t) * (float(ell_t) - r)


def g_v_value(a_t: int, r_t: float, delta: float) -> float:
    return float(a_t) * (float(r_t) - delta)


@dataclass(frozen=True)
class AnytimeBoundResult:
    t: int
    cumulative_mean: float
    alpha_t: float
    radius: float
    lcb: float
    ucb: float


def anytime_bound(values: list[float], bounds: ProcessBounds, alpha_total_process: float) -> AnytimeBoundResult:
    """values: G_j,1..G_j,t in eligible-time order, one per eligible day
    INCLUDING abstentions (abstention days still contribute a value under
    each process's own definition -- e.g. G_C=-c, G_L=0, G_V=0 -- callers
    must never skip them). Returns the bound at horizon t=len(values)."""
    t = len(values)
    if t < 1:
        raise ValueError("anytime_bound requires at least one eligible-time observation")
    if not (0.0 < alpha_total_process < 1.0):
        raise ValueError("alpha_total_process must be in (0, 1)")
    w = bounds.width
    if w <= 0:
        raise ValueError("process width must be > 0")
    mean = float(sum(values)) / t
    alpha_t = alpha_total_process / (t * (t + 1))
    radius = w * math.sqrt(math.log(1.0 / alpha_t) / (2.0 * t))
    return AnytimeBoundResult(t=t, cumulative_mean=mean, alpha_t=alpha_t, radius=radius, lcb=mean - radius, ucb=mean + radius)


@dataclass(frozen=True)
class AlphaAllocation:
    alpha_total: float
    alpha_c: float
    alpha_l: float
    alpha_v: float

    def __post_init__(self) -> None:
        allocated = self.alpha_c + self.alpha_l + self.alpha_v
        if allocated > self.alpha_total + 1e-12:
            raise ValueError(
                f"alpha allocation {allocated} exceeds alpha_total {self.alpha_total}: "
                "reporting this as a joint certificate would misstate three independent "
                "endpoint statements as a simultaneous one (section 8)"
            )


def default_equal_split(alpha_total: float) -> AlphaAllocation:
    third = alpha_total / 3.0
    return AlphaAllocation(alpha_total=alpha_total, alpha_c=third, alpha_l=third, alpha_v=third)


@dataclass(frozen=True)
class SimultaneousCertificate:
    t: int
    coverage_bound: AnytimeBoundResult
    loss_bound: AnytimeBoundResult
    value_bound: AnytimeBoundResult
    coverage_supported: bool  # LCB(mean G_C) >= 0
    loss_supported: bool  # UCB(mean G_L) <= 0
    value_supported: bool  # LCB(mean G_V) >= 0
    fully_supported: bool  # all three simultaneously -- the ONLY basis for full support
    alpha_allocation: AlphaAllocation
    version: str = REFERENCE_MONITOR_VERSION


def evaluate_simultaneous_certificate(
    g_c_values: list[float],
    g_l_values: list[float],
    g_v_values: list[float],
    *,
    c: float,
    r: float,
    delta: float,
    r_max: float,
    alpha_allocation: AlphaAllocation,
) -> SimultaneousCertificate:
    t = len(g_c_values)
    if not (len(g_l_values) == t and len(g_v_values) == t):
        raise ValueError("G_C, G_L, G_V must carry one value per eligible time step")
    cb = anytime_bound(g_c_values, g_c_bounds(c), alpha_allocation.alpha_c)
    lb = anytime_bound(g_l_values, g_l_bounds(r), alpha_allocation.alpha_l)
    vb = anytime_bound(g_v_values, g_v_bounds(delta, r_max), alpha_allocation.alpha_v)
    coverage_supported = cb.lcb >= 0.0
    loss_supported = lb.ucb <= 0.0
    value_supported = vb.lcb >= 0.0
    return SimultaneousCertificate(
        t=t,
        coverage_bound=cb,
        loss_bound=lb,
        value_bound=vb,
        coverage_supported=coverage_supported,
        loss_supported=loss_supported,
        value_supported=value_supported,
        fully_supported=bool(coverage_supported and loss_supported and value_supported),
        alpha_allocation=alpha_allocation,
    )
