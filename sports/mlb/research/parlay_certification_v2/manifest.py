from __future__ import annotations

"""FROZEN manifest for PARLAY_CERTIFICATION_V2 -- the sole authoritative
certification/decision layer for the MLB parlay research system.

Supersedes sports/mlb/research/joint_position_builder_v2/legacy/
risk_gate_v1_ARCHIVED.py (a single-endpoint empirical-risk bound with a
vacuous-set gap in its world-certificate check) as the authority for any
NEW certification decision. See MIGRATION.md for the full audit and call
graph.

production_authorized stays False here unconditionally, always. Only the
outer V2 prospective certificate (state_machine.PolicyStatus reaching
SUPPORTED_CURRENT) may ever be the basis for advancing it, and even then
this file does not flip it automatically -- see CONCLUSION_REASONING.
"""

VERSION = "PARLAY_CERTIFICATION_V2"
PRODUCTION_AUTHORIZED = False  # never set programmatically -- section 12/16

# Frozen sub-versions -- section 14. A change to ANY of these is a NEW
# policy version with its own prospective evidence stream (evidence_store
# already enforces this structurally via one file per policy_version).
ELIGIBILITY_VERSION = "ELIGIBILITY_V1"
POLICY_VERSION = "PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION"
SETTLEMENT_VERSION = "SETTLEMENT_V1"
WORLD_CERTIFICATE_VERSION = "NONVACUOUS_WORLD_CERTIFICATE_V1"
EVIDENCE_STORE_VERSION = "EVIDENCE_STORE_V1"
REFERENCE_MONITOR_VERSION = "PARLAY_REFERENCE_ANYTIME_CERT_V1"
STATE_MACHINE_VERSION = "PARLAY_CERTIFICATION_STATE_V1"

MAX_ACTIONS_PER_ELIGIBLE_SLATE = 1
TWO_LEG_PARLAYS_ONLY = True

# Predeclared targets -- frozen BEFORE any prospective evaluation (section
# 14). No prior value existed anywhere in this repo for c/delta/R_max (the
# archived risk_gate had only a single RISK_TARGET=0.30, reused for r
# below); these are conservative placeholders and MUST NOT be tuned
# post-hoc based on prospective results -- a change requires a new policy
# version, never an edit to these constants after real evidence exists.
C_MIN_COVERAGE = 0.50  # c: minimum fraction of eligible days the policy must act on
R_MAX_LOSS_RISK = 0.30  # r: maximum acceptable selective loss risk (matches legacy risk_gate.RISK_TARGET)
DELTA_MIN_RETURN = 0.0  # delta: minimum expected return per action (breakeven -- no positive-edge claim a priori)
R_MAX_ACCEPTED = 25.0  # R_max: predeclared maximum accepted parlay-price-derived return bound

ALPHA_TOTAL = 0.05
ALPHA_C = ALPHA_TOTAL / 3.0
ALPHA_L = ALPHA_TOTAL / 3.0
ALPHA_V = ALPHA_TOTAL / 3.0

# PolicyStatus value. DEVELOPMENT until a deliberate freeze action moves it
# to FROZEN_PROSPECTIVE_INCONCLUSIVE -- see CONCLUSION_REASONING; this file
# existing does not itself constitute that freeze.
STATUS = "DEVELOPMENT"

CONCLUSION_REASONING = """
This manifest freezes PARLAY_CERTIFICATION_V2 as the sole authoritative
certification/decision layer for this research system, but records ZERO
real prospective evidence so far. STATUS is DEVELOPMENT, not
FROZEN_PROSPECTIVE_INCONCLUSIVE: the c/r/delta/R_max/alpha values above
are provisional defaults carried over from the old single-endpoint
risk_gate convention (RISK_TARGET->r) or set conservatively where no prior
frozen value existed (c, delta, R_max) -- they have not yet been through a
deliberate freeze review for a real prospective run. Advancing STATUS to
FROZEN_PROSPECTIVE_INCONCLUSIVE is a deliberate action (see MIGRATION.md's
NEXT PROSPECTIVE STEP), not an automatic consequence of this file
existing or of the implementation being validated.

Per section 16: implementation validation
(PARLAY_CERTIFICATION_V2_IMPLEMENTATION_VALIDATED, established by
sports/mlb/tests/test_parlay_certification_v2.py) is a claim about the
CERTIFICATION MACHINERY, not about the MLB policy's profitability. The
policy itself must remain FROZEN_PROSPECTIVE_INCONCLUSIVE (once frozen)
until real, untouched prospective evidence on eligible slate days
satisfies all three simultaneous bounds (coverage >= c, loss risk <= r,
return >= delta) at some horizon t, per anytime_monitor.py.
""".strip()

MIGRATION_SUMMARY = """
Audited and replaced: sports/mlb/research/joint_position_builder_v2/
risk_gate.py (moved to legacy/risk_gate_v1_ARCHIVED.py; its
build_selective_risk_certificate/gate_and_rank_day are a single-endpoint
empirical-risk bound, not a simultaneous coverage/loss/return certificate,
and pairs.py's build_pair_certificate lacked an explicit positive-
retained-mass check alongside its nonempty check). Neither was ever wired
into CI, run_daily_predictions.py, or any production/shadow execution
path -- both were manual/local research scripts only; this migration
removes them from the set of things that could ever be mistaken for a
live authority going forward, without breaking the historical regression
tests that characterize their archived behavior.

Explicitly audited and left OUT OF SCOPE: sports/parlay_analysis.py and
sports/mlb/scripts/select_daily_parlay.py (CONTROL). These implement a
structurally unrelated architecture (ML probability-threshold gates --
min_leg_probability, hit-survival consensus thresholds -- with no
outcome_worlds/world-certificate concept at all) and every authorization
string they emit is already hard-labeled "shadow_only"/"diagnostic_only"
in the payload itself, so nothing downstream can mistake their output for
a live authorization. This research thread has never touched CONTROL (see
joint_position_builder_v2/REPORT.md); rewriting a large, live,
FanDuel-integrated production script's authorization logic as a side
effect of this migration was judged out of scope and higher-risk than
the mission's stated goal (upgrading the parlay CERTIFICATION theory this
session's own research built) justifies. See MIGRATION.md for detail.
""".strip()
