from __future__ import annotations

"""FROZEN status for JOINT_POSITION_BUILDER_V2.

production_authorized is unconditionally False. If/when a selective
empirical-risk certificate becomes available on real, non-retired data
(see risk_gate.SelectiveRiskCertificate), the status may advance to
SHADOW_ONLY -- never further without a separate, explicit decision this
file does not make for you.
"""

PRODUCTION_AUTHORIZED = False

VERSION = "JOINT_POSITION_BUILDER_V2"

# See REPORT.md for the full reasoning. Short version: real per-leg H-target
# market prices are entirely absent from DEVELOPMENT_STAMPS (0 of 4081 rows);
# every real price observed anywhere in this repo (1503 of 2227 rows) falls
# inside the retired TEST_STAMPS window, which this package's development
# code never reads (enforced by test_joint_position_builder_v2.py and the
# AST-based guard pattern already used by sports/mlb/research/h_over_ranker).
# The 2x2 real-data ablation (A/B/C/D) therefore has zero evaluable action-
# universe pairs in every cell: the EV-admission theory is architecturally
# implemented and mechanically verified (see the theorem tests), but is
# UNTESTABLE against real settled MLB outcomes with data currently available
# outside the retired TEST block.
STATUS = "INSUFFICIENT_EVIDENCE_NO_REAL_PRICE_COVERAGE_IN_DEVELOPMENT_WINDOW"

CONCLUSION = "INSUFFICIENT_EVIDENCE"

CONCLUSION_REASONING = """
Not MARGINAL_EV_GATE_SUPPORTED or MARGINAL_EV_GATE_REJECTED: neither can be
claimed, because the empirical comparison the task asks for (does allowing
EV_i<0 legs produce real +EV joint positions more/less often than the
individual-EV gate) requires a real quoted parlay price for at least one
side of the comparison, and DEVELOPMENT_STAMPS has zero rows with a real
H-target market price. This is a data-coverage gap, not a finding about the
theory's correctness.

What WAS validated on real, non-retired data (see REPORT.md):
  - The joint-probability mechanism itself (independence via
    outcome_worlds.build_world_distribution, unmodified) is well-calibrated
    in "narrow" state (mean predicted 0.380 vs. actual 0.376, n=1470 pairs,
    14 days) but meaningfully overconfident in "broad" state (mean predicted
    0.573 vs. actual 0.430) -- broadening state inputs is not free with the
    CURRENT frozen marginal model; it measurably degrades joint calibration.
  - Same-game pairs in narrow state showed a small NEGATIVE calibration gap
    under a pure independence assumption (predicted 0.382 vs. actual 0.392,
    n=288) -- a mild hint of real positive same-game dependence, consistent
    with baseball intuition, but not large enough or rigorously enough
    isolated to build a correlation model on (out of scope here regardless
    -- see the task's own instruction not to fit a correlation model in the
    same pass as the parlay policy).

What WAS validated mechanically (theorem tests, synthetic, data-independent):
  - A pair with one individually -EV leg can be genuinely +EV under
    independence (theorem 1: pA=.80/dA=1.40, pB=.70/dB=1.40 -> pair EV
    +9.76% despite leg B individually -2%).
  - A pair with BOTH legs individually -EV can become +EV under a positive
    dependence structure that plain independence cannot produce (theorem 2).
  - The pair-specific compatible-worlds certificate (build_pair_certificate)
    agrees exactly with the pre-existing, unmodified generic N-candidate
    certificate logic in outcome_worlds.py (theorem 3) -- confirms no
    divergent reimplementation crept in.
  - Chronology/no-leakage and the "never auto-authorizes" invariant hold
    (theorems 4-5).

So: the CORE CORRECTION under test (individual EV_i>0 is not necessary for
pair admission) is proven as a real, non-degenerate mathematical
possibility, and the guardrails around it (conservative LCB, certificate,
selective-risk gate, one-pair-per-day cap) are implemented and mechanically
correct. Whether it produces real economic value on real MLB games remains
untested for lack of real combo-pricing data in the window this task is
scoped to use.
""".strip()

NEXT_PROSPECTIVE_PROTOCOL = """
1. Do not touch TEST_STAMPS. A genuine test of this system needs NEW real
   days beyond both DEVELOPMENT_STAMPS and TEST_STAMPS to accumulate, the
   same position h_over_ranker's manifest already takes.
2. Specifically watch for real H-target Market_Over_Price/Market_Under_Price
   coverage in future archived daily_prediction_pool_*.csv files -- it did
   not exist at all before the current TEST_STAMPS window and only started
   appearing there. Confirm it persists going forward before relying on it.
3. Once >=20 fresh, real-price-covered day-folds exist beyond the retired
   blocks (matching this repo's existing minimum_calibration_slates
   convention), re-run ablation.run_all_variants() unchanged and report
   real pair-class (++/+-/--) breakdowns, joint_EV/joint_EV_LCB
   distributions, and the selective-risk certificate's validation-block
   result -- the machinery is already built and does not need modification
   to consume that data once it exists.
4. Do not lower RISK_TARGET, JOINT_EV_LCB_MARGIN, or MIN_SUPPORT_HISTORY_ROWS
   in ablation.py to manufacture actionable pairs before there is real
   price coverage to evaluate them against.
5. Only after this system's own selective-risk certificate is
   SELECTIVE_RISK_BOUND_SUPPORTED_ON_VALIDATION on real fresh data does
   STATUS advance past SHADOW_ONLY -- PRODUCTION_AUTHORIZED stays False in
   this file regardless; production authorization is never a code change
   here.
""".strip()
