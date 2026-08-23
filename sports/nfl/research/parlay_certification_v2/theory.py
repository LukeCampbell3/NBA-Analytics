from __future__ import annotations

"""Documentation-only module (mission section 10) -- do not import this
for a runtime decision; the reference monitor (anytime_monitor.py) uses the
sequential/predictable-mean interpretation, not the stationary oracle
below, unless a project decision explicitly freezes a stationarity
assumption (not currently the case anywhere in this repo).

STATIONARY ORACLE (retained for documentation only):

    c*(r, delta) = sup over policies pi {
        coverage(pi) :
        loss_risk(pi) <= r,
        value(pi) >= delta
    }

A certified policy under a STABLE target distribution constructively
proves c*(r, delta) >= c. The implemented monitor never assumes
stationarity, so it never computes or claims a value for c*(r, delta) --
only the sequential simultaneous certificate in anytime_monitor.py.

Do NOT claim global information-theoretic feasibility/infeasibility from a
sequential policy's failure to certify. The only terminal research labels
this system may emit are the three below; a fourth,
INFORMATION_THEORETICALLY_INFEASIBLE, is reserved for an actual proven
upper-bound/impossibility theorem and is never emitted by anything in this
package.
"""

FROZEN_POLICY_PROSPECTIVELY_SUPPORTED = "FROZEN_POLICY_PROSPECTIVELY_SUPPORTED"
POLICY_NOT_SUPPORTED_WITHIN_PREDECLARED_CLASS = "POLICY_NOT_SUPPORTED_WITHIN_PREDECLARED_CLASS"
EVIDENCE_INCONCLUSIVE = "EVIDENCE_INCONCLUSIVE"

# Reserved, never emitted by this package -- requires an actual proven
# upper-bound/impossibility theorem, which no code here constructs.
INFORMATION_THEORETICALLY_INFEASIBLE = "INFORMATION_THEORETICALLY_INFEASIBLE"

TERMINAL_LABELS = (
    FROZEN_POLICY_PROSPECTIVELY_SUPPORTED,
    POLICY_NOT_SUPPORTED_WITHIN_PREDECLARED_CLASS,
    EVIDENCE_INCONCLUSIVE,
)
