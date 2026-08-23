"""Role-saturation / diminishing-returns curve (spec section 20).

    retention = exp(-k * max(0, H - 1))

Transparent and bounded: retention == 1.0 whenever the touch multiplier
H <= 1 (a flat or shrinking role never "loses" efficiency by this
curve's own construction), and decays smoothly toward 0 as H grows
without bound. k controls how fast a role expansion saturates -- a
larger k means a given role increase costs more efficiency.
"""

from __future__ import annotations

import math


def saturation_retention(h: float, k: float) -> float:
    return math.exp(-k * max(0.0, h - 1.0))


def turnover_growth_from_saturation(saturation: float) -> float:
    """The default (non-overridden) turnover-growth assumption: turnover
    risk grows as the inverse of retained efficiency, i.e. the same
    saturation loss that erodes AST/pass and shot generation also
    inflates turnover risk, by construction of this simple first model
    (section 20: "Higher role can cause... higher turnovers")."""
    if saturation <= 0:
        return 0.0
    return 1.0 / saturation - 1.0
