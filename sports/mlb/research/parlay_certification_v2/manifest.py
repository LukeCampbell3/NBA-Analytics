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

# CANDIDATE-SET SIZE BOUND (mission: fixing the circular support-gate bug
# made this a LIVE concern for the first time -- before the fix,
# support_is_structurally_unreachable() always short-circuited before any
# candidate ever reached select_action_for_day, so an unbounded C(n,2)
# candidate set was moot. Once real REQUIRED support accumulates broadly
# across a mature ledger (observed in testing: 203/420 legs individually
# support-passing on one real slate -> 27,279 candidate pairs reaching
# certification), evaluating every one of them daily is a real runtime
# risk. This bound truncates the SUPPORT-PASSING candidate set
# deterministically -- sorted by candidate_id, a purely structural key
# with NO relationship to predicted probability, price, or any other
# quality signal -- before it reaches select_action_for_day. It never
# changes WHICH candidate gets selected among those considered (the
# frozen tie-breaker in policy.py is untouched), only how many are
# considered; like MAX_ELIGIBLE_LEGS_PER_DAY in
# calibration/pair_ingest.py, this is a volume bound, never a quality
# gate, and it does not touch the G_C/G_L/G_V certificate math at all.
MAX_CANDIDATES_PER_SLATE = 500

# Predeclared targets -- frozen BEFORE any prospective evaluation (section
# 14). No prior value existed anywhere in this repo for c/delta/R_max (the
# archived risk_gate had only a single RISK_TARGET=0.30, reused for r
# below); these are conservative placeholders and MUST NOT be tuned
# post-hoc based on prospective results -- a change requires a new policy
# version, never an edit to these constants after real evidence exists.
C_MIN_COVERAGE = 0.50  # c: minimum fraction of eligible days the policy must act on
R_MAX_LOSS_RISK = 0.30  # r: maximum acceptable selective loss risk (matches legacy risk_gate.RISK_TARGET)
DELTA_MIN_RETURN = 0.0  # delta: minimum expected return per action (breakeven -- no positive-edge claim a priori)

# ECONOMIC BOUNDS (mission section 12) -- D_max is the actual frozen
# config; R_MAX_ACCEPTED is ALWAYS derived from it, never configured
# independently (a free-standing R_max invites picking it just to make
# the anytime process converge faster, which section 12 explicitly
# forbids). D_max=6.0 reflects a desired 2-leg parlay PRODUCT profile --
# generous enough to cover realistic combined prices for two modestly-
# favored props (each leg commonly in the ~1.4-2.2 decimal range, so
# combined ~2-5 is typical), tight enough to exclude longshot-stacked
# combinations this product is not aiming to certify. Not chosen to
# shrink anytime_monitor's radius (a materially wider D_max, e.g. the
# previous placeholder R_MAX_ACCEPTED=25.0 this replaces, would have made
# convergence slower, not faster -- the direction of that old placeholder
# was actually conservative, not a shortcut, but it was also never tied
# to an actual product decision, which this D_max is).
D_MAX = 6.0
R_MAX_ACCEPTED = D_MAX - 1.0  # derived -- never edit this independently of D_MAX

ALPHA_TOTAL = 0.05
ALPHA_C = ALPHA_TOTAL / 3.0
ALPHA_L = ALPHA_TOTAL / 3.0
ALPHA_V = ALPHA_TOTAL / 3.0

# PROGRAM-LEVEL MULTIPLICITY (mission section 13). ALPHA_TOTAL above is
# THIS policy version's within-policy budget (alpha_policy_k); it must
# itself be drawn from a program-level ledger shared across all policy
# versions ever frozen for prospective confirmation, so that repeatedly
# freezing new versions after a failure/demotion cannot silently reset to
# a fresh alpha. See sports/mlb/parlay_v2/program_alpha.py --
# ProgramAlphaLedger.spend(...) is the only way ALPHA_TOTAL above is
# actually authorized to be spent; this constant alone does not spend it.
ALPHA_PROGRAM = 0.05
PROGRAM_ALPHA_LEDGER_PATH = "sports/mlb/research/parlay_certification_v2/reports/program_alpha_ledger.json"

# CURRENT-DAY FREEZE BOUNDARY (mission section 11). The real boundary
# timestamp lives in prospective_boundary.py's one-way marker file, NOT
# as an editable constant in this frozen manifest (editing a source
# constant to "set" a timestamp would blur the same freeze discipline
# this section exists to protect). See
# prospective_boundary.read_prospective_start_timestamp(). Until that
# marker is set, EVERY evaluated slate -- including any already inspected
# while building this integration -- counts only as DEVELOPMENT/SHADOW,
# never as confirmatory prospective evidence.

# PROSPECTIVE POLICY IDENTIFIER (mission: "Resolve the PARLAY_V2
# perpetual-abstention problem WITHOUT weakening the theorem-grade outer
# certification system"). The ACTION RULE changed materially -- support
# evaluation moved from "all five dimensions (including two permanently
# UNESTABLISHED ones) must PASS" to gate-mode-aware evaluation where only
# the three REQUIRED, real, implemented dimensions can block (see
# calibration/support.py). Per this program's own version-isolation
# discipline, a materially different action rule is a NEW policy version,
# never a silent redefinition of POLICY_VERSION above (POLICY_VERSION
# names the *structural shape* -- two-leg, single-action -- which is
# unchanged; PROSPECTIVE_POLICY_ID names this specific frozen attempt at
# proving it). "...001" was already used as a label on a never-activated
# freeze-readiness dry-run artifact from an earlier session, so this
# increments to "002" rather than reusing it for different semantics.
PROSPECTIVE_POLICY_ID = "PARLAY_POLICY_V2_PROSPECTIVE_002"

# FROZEN GATE-MODE CONFIGURATION for PROSPECTIVE_POLICY_ID above (mission
# section 4). REQUIRED dimensions block action when not PASS; OBSERVE_ONLY
# dimensions are computed and exposed for research but can NEVER block,
# regardless of status (including UNESTABLISHED) -- see
# calibration/support.py's GateMode/SupportDimension for the authoritative
# implementation this dict is the frozen manifest record of. Promoting
# joint_support or shift_status to REQUIRED requires an independently
# validated, non-arbitrary threshold AND a new PROSPECTIVE_POLICY_ID --
# never an edit to this dict in place once frozen.
SUPPORT_GATE_MODES = {
    "market_support": "REQUIRED",
    "line_support": "REQUIRED",
    "state_support": "REQUIRED",
    "joint_support": "OBSERVE_ONLY",
    "shift_status": "OBSERVE_ONLY",
}

# ============================================================
# CANDIDATE NEXT POLICY (mission: "Resolve the remaining PARLAY_V2 APS /
# counterexample admission bottleneck"). world_gate_research.py's
# DEVELOPMENT-only research (DERIVE then SELECT, chronological, day-
# clustered) found:
#   - HARD_ZERO (world_gate_mode=REQUIRED, the PROSPECTIVE_002 config):
#     0.000000 nonvacuous-certificate rate at FROZEN_APS_THRESHOLD=1.0
#     across every real DEVELOPMENT pair sampled -> operationally
#     degenerate, confirmed empirically, not merely suspected.
#   - counterexample_mass at that same frozen threshold is EXACTLY
#     1 - predicted_joint_probability (proven identity, verified to
#     float precision on real data) -- the world-set machinery, as
#     currently implemented (independence-only, no fitted dependence
#     model), carries NO information beyond the raw joint-probability
#     baseline. A BOUNDED_RISK gate on it would therefore be gating on
#     the same baseline under a different name; the DERIVE threshold
#     sweep's admissible-vs-inadmissible loss-rate split was additionally
#     noisy/non-monotone across the frozen grid (n days too small: 4
#     usable DERIVE days), so BOUNDED_RISK is NOT supported by this
#     research pass.
#   - The underlying continuous quantity (1 - predicted_joint_probability,
#     equivalently counterexample_mass at full retention) DOES predict
#     realized pair loss: Spearman rho=0.111 (DERIVE, 4 days, 95% day-
#     clustered bootstrap CI (0.028, 0.176)) replicating to rho=0.229
#     (SELECT, 8 days, CI (0.191, 0.258)) on the SAME predeclared bin
#     definitions, both chronologically forward, monotone across quintile
#     bins on both partitions. This justifies keeping it as a RANKING
#     diagnostic, not a gate.
# Conclusion: WORLD_GATE_OBSERVE_ONLY_SUPPORTED. See
# world_gate_research.py and the mission's own required report tables for
# the full analysis this constant reflects.
#
# NOT YET FROZEN FOR REAL PROSPECTIVE CONFIRMATION -- see ALPHA BUDGET
# AUDIT below. This id/config is fully implemented and tested
# (policy.select_action_for_day's world_gate_mode parameter,
# run_parlay_v2.build_slate_payload's world_gate_mode parameter) but
# freeze_prospective.py has NOT been run with --confirm for it, no
# program alpha has been spent for it, and it is NOT wired into the
# actual daily production/CI invocation of run_parlay_v2.py -- a human
# must resolve the alpha-budget question below before that happens.
PROSPECTIVE_POLICY_ID_CANDIDATE = "PARLAY_POLICY_V2_PROSPECTIVE_003"
WORLD_GATE_MODE_CANDIDATE = "OBSERVE_ONLY"
WORLD_RISK_THRESHOLD_CANDIDATE = None  # OBSERVE_ONLY never gates -- no threshold needed

# ALPHA BUDGET AUDIT (mission section 20), performed before considering
# PROSPECTIVE_003 for a real freeze -- see program_alpha_ledger.json
# (checked directly, not assumed): PROSPECTIVE_002 already recorded a
# 0.05 spend against ALPHA_PROGRAM=0.05, i.e. the ENTIRE program budget,
# and ProgramAlphaLedger.spend() mechanically REFUSES a second 0.05 spend
# for PROSPECTIVE_003 (verified directly: raises ValueError, "would bring
# total spend to 0.1 > alpha_program=0.05"). However, the EvidenceStore
# for manifest.POLICY_VERSION contains ZERO rows and the
# DecisionRecordStore contains ZERO rows (both checked directly on disk)
# -- meaning PROSPECTIVE_002's alpha was spent at freeze time but NO
# actual G_C/G_L/G_V hypothesis evaluation has ever been performed under
# it (no real day has been decided since the freeze). Whether that means
# the spend can be legitimately retired under the frozen multiple-testing
# rules, or whether ANY frozen spend is permanent regardless of whether
# it was ever exercised (the conservative reading -- ProgramAlphaLedger's
# own docstring: "Never resets total_spent() on a demotion/failure"),
# or whether a program-level correction (raising ALPHA_PROGRAM) is the
# right fix, is a multiple-testing-methodology decision this pass does
# NOT make unilaterally -- doing so under mission pressure is exactly the
# "hidden repeated-testing inflation" section 20 forbids. PROSPECTIVE_003
# stays un-frozen (STATUS below still describes PROSPECTIVE_002 only)
# until a human resolves this.
ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003 = True

# PolicyStatus value. Advanced to FROZEN_PROSPECTIVE_INCONCLUSIVE as a
# deliberate freeze action for PROSPECTIVE_POLICY_ID above -- see
# CONCLUSION_REASONING for the three-step freeze this accompanies
# (alpha-ledger spend + prospective_start boundary, both performed
# alongside this edit, never automatically).
STATUS = "FROZEN_PROSPECTIVE_INCONCLUSIVE"

CONCLUSION_REASONING = """
This manifest freezes PARLAY_CERTIFICATION_V2 as the sole authoritative
certification/decision layer for this research system. STATUS advanced
DEVELOPMENT -> FROZEN_PROSPECTIVE_INCONCLUSIVE for PROSPECTIVE_POLICY_ID
= "PARLAY_POLICY_V2_PROSPECTIVE_002" as the deliberate freeze action for
the mission that fixed the circular support-gate bug (see
calibration/support.py's module docstring): joint_support/shift_status
moved from wrongly-REQUIRED-forever to correctly OBSERVE_ONLY, making
real selection reachable for the first time. This freeze records ZERO
real prospective evidence so far -- STATUS documents that a policy
version is now open to accumulate it, not that any exists yet. The
c/r/delta/alpha values are still the provisional defaults carried over
from the old single-endpoint risk_gate convention (RISK_TARGET->r) or set
conservatively where no prior frozen value existed (c, delta); D_MAX=6.0
is a real product-profile decision (see its comment above), not a
placeholder. All three freeze prerequisites were completed alongside this
STATUS edit, in order: (1) this final review of every constant in this
file (SUPPORT_GATE_MODES/MAX_CANDIDATES_PER_SLATE added, all other
values left untouched), (2) recording this policy version's alpha spend
in the program alpha ledger
(sports/mlb/parlay_v2/program_alpha.ProgramAlphaLedger.spend), (3)
setting the one-way prospective_start boundary
(parlay_certification_v2.prospective_boundary.set_prospective_start_timestamp)
for PROSPECTIVE_POLICY_ID via
sports/mlb/parlay_v2/freeze_prospective.py --confirm.

IMPORTANT -- this freeze is NOT a claim of profitability or of certified
production-readiness. A SEPARATE, deliberately conservative bottleneck
(FROZEN_APS_THRESHOLD=1.0, retain-all, in run_parlay_v2.py/
candidate_adapter.py -- untouched by this mission, since it is part of
the G_C/G_L/G_V world-certificate machinery this mission was explicitly
required not to weaken) means the world certificate cannot yet certify
any candidate built from a non-deterministic prediction: any nonzero
probability mass on a losing world violates zero_loss_counterexamples.
Real ACT selections -- and therefore real coverage/loss/return evidence
-- will not begin accumulating until that SEPARATE APS-calibration
research thread (re-deriving a shrinking APS threshold from real settled
prospective days, already flagged in this file's own FROZEN_APS_THRESHOLD
comment) is done. This freeze only removes the circular block that made
selection permanently impossible regardless of that; it does not by
itself make ACT happen today.

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
