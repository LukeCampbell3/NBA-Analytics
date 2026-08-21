from __future__ import annotations

"""FROZEN status for JOINT_POSITION_BUILDER_V2.

production_authorized is unconditionally False. If/when a selective
empirical-risk certificate becomes available on real, non-retired data
(see risk_gate.SelectiveRiskCertificate), the status may advance to
SHADOW_ONLY -- never further without a separate, explicit decision this
file does not make for you.

REVISION NOTE (see STATE.md for full detail, not a retroactive rewrite of
the original evidence below -- the original H-only ablation results and
their reasoning are left intact because they were correctly derived from
what was tested; this note narrows the SCOPE of what "no real price
coverage" was ever true of): the original STATUS string below was accurate
for the H target specifically (which the original 2x2 ablation A/B/C/D was
scoped to) but was mistakenly read as "no real price coverage exists in
DEVELOPMENT_STAMPS at all". It does not -- R (57%), TB (58%), and HR (53%)
all have substantial real price coverage in the same window. See
multi_target_universe.py / multi_target_backtest.py / STATE.md for the
generalized, real-price multi-target pass this produced. That pass's own
conclusion is ALSO insufficient evidence, but for a different, better-
grounded reason (marginal-model overconfidence outside the narrow H-OVER
slice it was frozen on, plus too little data -- 12-16 days -- to resolve a
day-clustered CI either way), not absence of priced markets.
"""

PRODUCTION_AUTHORIZED = False

# SUPERSEDED: the certification/risk-gating machinery this package used
# (legacy/risk_gate_v1_ARCHIVED.py, formerly risk_gate.py) is archived and
# no longer authoritative for any new decision -- see
# sports/mlb/research/parlay_certification_v2/ and its MIGRATION.md. This
# package's own pair-proposal machinery (pairs.py, observation_universe.py,
# multi_target_universe.py) remains a valid PREDICTIVE/WORLD-MODEL input to
# V2's policy layer; it just no longer certifies/authorizes anything itself.
CERTIFICATION_AUTHORITY_SUPERSEDED_BY = "PARLAY_CERTIFICATION_V2"

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
#
# NOTE: this STATUS string described the H-only ablation, not the full
# picture -- see the REVISION NOTE above and STATE.md. Left as originally
# written (not rewritten) because it accurately describes what that specific
# ablation found; MULTI_TARGET_STATUS below is the up-to-date, broader
# conclusion.
STATUS = "INSUFFICIENT_EVIDENCE_NO_REAL_H_PRICE_COVERAGE_IN_DEVELOPMENT_WINDOW"

# Up-to-date conclusion after generalizing beyond H (see STATE.md). Also
# INSUFFICIENT_EVIDENCE, now for a MARGINAL MODEL / DATA reason rather than
# an ACTION COVERAGE reason.
MULTI_TARGET_STATUS = "INSUFFICIENT_EVIDENCE_MARGINAL_MODEL_UNCALIBRATED_OUTSIDE_H_OVER_AND_TOO_FEW_DAYS"

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

MULTI_TARGET_CONCLUSION_REASONING = """
Full detail in STATE.md; summary here for anyone reading only manifest.py.
Numbers below are from the full, uncapped-universe backtest
(reports/multi_target_broad_summary.json, 619,191 evaluated priced pairs,
11 days) -- a smaller top-25-legs/day exploratory pass done first is
superseded by this and explicitly not the evidence of record; see STATE.md
for why the two differ (restricting to the model's highest-probability
legs concentrates its worst overconfidence -- itself a finding, see below).

Real R/TB/HR (broad, both-direction) price coverage exists in
DEVELOPMENT_STAMPS (3623 action-eligible rows, 12 days) -- the ACTION
COVERAGE gap that blocked the H-only pass is closed.

  - The joint-probability mechanism's own confidence is overconfident vs.
    both actual outcomes and the market-implied price, worst in the ++
    class (mean p_joint=0.391 vs. actual both-win rate 0.284 vs.
    market-implied 0.277, gap +0.106) and present but smaller overall
    (+0.046). This isolates the problem to the MARGINAL MODEL
    (probability_score, frozen on narrow H-OVER data) being overconfident
    on broad-mode R/TB/HR data it was never tuned on -- not to the
    joint/pair mechanism, which is unmodified and was previously shown
    well-calibrated in narrow H state.
  - mean_joint_ev (the model's own belief in its edge, +0.674 for ++
    pairs) is a model-confidence figure, not a realized return. The
    REALIZED backtest return for the same pairs is much smaller and, at
    11 days, not statistically distinguishable from zero: ++ class +6.1%
    (day-clustered 90% CI [-0.016, +0.147]), overall -1.0% (CI [-0.051,
    +0.045]). Only the `--` class (both legs individually -EV) reaches
    significance, and in the expected negative direction (-13.2%, CI
    [-0.199, -0.038]) -- a validity check on the pipeline, not a proposed
    trade.
  - A "value subset" filter (pairs where model p_joint > market-implied
    price) shows a promising-looking +4.9% (CI [-0.018, +0.130], still
    crossing zero) but was constructed post-hoc during this exact analysis
    pass, so it is a hypothesis for a future frozen rule, not confirmed
    evidence.
  - Concrete evidence against a "rank candidates by raw model confidence"
    selective-action policy (one of the mission's explicitly-flagged
    assumptions not to make): the model's overconfidence is worse, not
    better, among its own highest-probability legs -- selecting for raw
    confidence selects for the bottleneck.

Conclusion: still INSUFFICIENT_EVIDENCE, but the bottleneck has moved from
"no data" to "the frozen marginal model needs its own recalibration pass
for the broad multi-target state, and even the best-looking slices don't
clear a day-clustered significance bar at only 11-16 real days" -- a
MARGINAL MODEL + DATA volume problem, not a mechanism-correctness problem.
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
