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
# proving it).
#
# PROSPECTIVE_POLICY_ID names the CURRENTLY FROZEN attempt -- the one
# freeze_prospective.py has activated a one-way boundary for, and the one
# a real production/CI invocation of run_parlay_v2.py actually uses
# (main() defaults world_gate_mode/world_risk_threshold to WORLD_GATE_MODE
# /WORLD_RISK_THRESHOLD below). PRIOR_PROSPECTIVE_POLICY_IDS preserves the
# audit trail of every earlier frozen attempt -- each one's own freeze
# artifacts (readiness json, boundary marker) remain on disk, permanently,
# unmodified; this tuple exists only so that history is discoverable from
# manifest.py itself without having to know filenames in advance.
PRIOR_PROSPECTIVE_POLICY_IDS: tuple[str, ...] = ("PARLAY_POLICY_V2_PROSPECTIVE_002",)
# "...001" was already used as a label on a never-activated freeze-
# readiness dry-run artifact from an earlier session, so "002" (the first
# real freeze, see PRIOR_PROSPECTIVE_POLICY_IDS) never reused it.
# "002" -> "003" is the second real freeze, performed by the mission that
# resolved the APS/counterexample admission bottleneck below: fixing a
# materially different world-gate rule is, per this file's own version-
# isolation discipline, a new policy version, never an in-place edit to
# 002 (which remains frozen and immutable -- see world_gate_research.py).
PROSPECTIVE_POLICY_ID = "PARLAY_POLICY_V2_PROSPECTIVE_003"

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
# WORLD-GATE CONFIGURATION for PROSPECTIVE_POLICY_ID above (mission:
# "Resolve the remaining PARLAY_V2 APS / counterexample admission
# bottleneck"). world_gate_research.py's DEVELOPMENT-only research
# (DERIVE then SELECT, chronological, day-clustered) found:
#   - HARD_ZERO (world_gate_mode=REQUIRED, PROSPECTIVE_002's config):
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
# Conclusion: WORLD_GATE_OBSERVE_ONLY_SUPPORTED. See world_gate_research.py
# and its own required report tables for the full analysis this reflects.
#
# FROZEN for PROSPECTIVE_POLICY_ID (see ALPHA BUDGET AUDIT below for how
# this became possible) -- run_parlay_v2.main() defaults to these values
# for every real invocation, including the daily production/CI run
# (sports/site/pipeline/run_daily_predictions.py never overrides them).
WORLD_GATE_MODE = "OBSERVE_ONLY"
WORLD_RISK_THRESHOLD = None  # OBSERVE_ONLY never gates -- no threshold needed

# ALPHA BUDGET AUDIT (mission section 20), performed before freezing
# PROSPECTIVE_003 -- see program_alpha_ledger.json, the actual source of
# truth (checked directly, never assumed): PROSPECTIVE_002 had recorded a
# 0.05 spend against ALPHA_PROGRAM=0.05, i.e. the ENTIRE program budget,
# and ProgramAlphaLedger.spend() mechanically REFUSED a second 0.05 spend
# for PROSPECTIVE_003 (verified directly: raised ValueError, "would bring
# total spend to 0.1 > alpha_program=0.05"). The EvidenceStore for
# manifest.POLICY_VERSION and the DecisionRecordStore both contained ZERO
# rows (checked directly on disk before any action was taken) -- meaning
# PROSPECTIVE_002's alpha was spent at freeze time but NO actual
# G_C/G_L/G_V hypothesis evaluation was EVER performed under it (no real
# day was ever decided under that spend).
#
# RESOLUTION (human-authorized, not made unilaterally): a person was
# presented exactly three options -- (1) retire PROSPECTIVE_002's spend
# given the verified zero-evidence precondition, then spend fresh for
# PROSPECTIVE_003; (2) raise ALPHA_PROGRAM instead, leaving 002's spend
# permanent; (3) leave the conflict unresolved and keep PROSPECTIVE_003
# un-frozen -- and chose (1). Executed via
# program_alpha.ProgramAlphaLedger.retire_untested_spend (a narrow,
# auditable, append-only exception -- it hard-refuses if the real,
# freshly-counted evidence row count is ever nonzero; it is NOT the
# "reset on demotion/failure" case that method's docstring keeps
# permanently forbidden) followed by a normal `spend` for
# PROSPECTIVE_003. The ledger file itself carries the full history: the
# original 002 spend row, the retirement row (negative, offsetting,
# reason="retired_zero_evidence_observed..."), and the 003 spend row --
# nothing was edited or deleted. ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003 is
# False because this resolution actually happened, not because the
# underlying constraint stopped mattering -- see the ledger file for the
# real, checkable record.
ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003 = False

# PolicyStatus value. Advanced to FROZEN_PROSPECTIVE_INCONCLUSIVE as a
# deliberate freeze action for PROSPECTIVE_POLICY_ID above -- see
# CONCLUSION_REASONING for the three-step freeze this accompanies
# (alpha-ledger spend + prospective_start boundary, both performed
# alongside this edit, never automatically).
STATUS = "FROZEN_PROSPECTIVE_INCONCLUSIVE"

CONCLUSION_REASONING = """
This manifest freezes PARLAY_CERTIFICATION_V2 as the sole authoritative
certification/decision layer for this research system. STATUS has now
been advanced twice, for two successive frozen attempts:

(1) DEVELOPMENT -> FROZEN_PROSPECTIVE_INCONCLUSIVE for
PARLAY_POLICY_V2_PROSPECTIVE_002 -- the deliberate freeze action for the
mission that fixed the circular support-gate bug (see
calibration/support.py's module docstring): joint_support/shift_status
moved from wrongly-REQUIRED-forever to correctly OBSERVE_ONLY. This
freeze recorded ZERO real prospective evidence: a SEPARATE, deliberately
conservative bottleneck (FROZEN_APS_THRESHOLD=1.0, retain-all, inside the
G_C/G_L/G_V world-certificate machinery that mission was explicitly
required not to weaken) meant the world certificate could not certify any
candidate built from a non-deterministic prediction -- so PROSPECTIVE_002
never produced a single real ACT day, and consequently zero real
G_C/G_L/G_V hypothesis evaluations were ever performed under it.
PROSPECTIVE_002 remains frozen and immutable; its own artifacts (freeze
readiness json, prospective boundary marker) are untouched.

(2) FROZEN_PROSPECTIVE_INCONCLUSIVE re-affirmed for the current
PROSPECTIVE_POLICY_ID = "PARLAY_POLICY_V2_PROSPECTIVE_003" -- the
deliberate freeze action for the mission that resolved (1)'s exact
bottleneck: world_gate_research.py's DEVELOPMENT-only research
(DERIVE->SELECT, chronological, day-clustered) found HARD_ZERO
operationally degenerate (0% pass rate, confirmed empirically) and
BOUNDED_RISK not supported (no incremental value over the raw joint-
probability baseline, unstable threshold behavior); WORLD_GATE_MODE=
"OBSERVE_ONLY" above is the result -- world/counterexample diagnostics
are recorded but can never block admission; a DEVELOPMENT-validated
continuous ranking diagnostic (ascending world_risk_rho) replaces the
previously-constant retained_probability_mass tie-breaker. This is the
result of REAL falsification-driven research, not a threshold lowered
until something passed.

Because PROSPECTIVE_002 never produced any real evidence, resolving the
alpha-budget conflict this created (PROSPECTIVE_002's 0.05 spend already
consumed the entire 0.05 ALPHA_PROGRAM budget) required a human decision
-- see the ALPHA BUDGET AUDIT comment above for the three options
presented and the retirement actually executed. All three freeze
prerequisites were completed for PROSPECTIVE_003, in order: (1) a final
review of every constant in this file (WORLD_GATE_MODE/
WORLD_RISK_THRESHOLD added; SUPPORT_GATE_MODES/MAX_CANDIDATES_PER_SLATE,
frozen for PROSPECTIVE_002, carried over unchanged -- the support-gate
fix they record still applies), (2) recording PROSPECTIVE_003's alpha
spend in the program alpha ledger (after the human-authorized retirement
of PROSPECTIVE_002's unused spend --
sports/mlb/parlay_v2/program_alpha.ProgramAlphaLedger.retire_untested_spend
then .spend), (3) setting the one-way prospective_start boundary for
PROSPECTIVE_003 via sports/mlb/parlay_v2/freeze_prospective.py --confirm.

IMPORTANT -- this freeze is STILL NOT a claim of profitability or of
certified production-readiness. It removes the SPECIFIC bottleneck that
made PROSPECTIVE_002 structurally unable to ever act; it does not by
itself prove PARLAY_POLICY_V2_PROSPECTIVE_003 will earn positive real
return. Whatever real coverage/loss/return evidence PROSPECTIVE_003 now
accumulates is graded exclusively by the unchanged, authoritative
G_C/G_L/G_V simultaneous certificate (anytime_monitor.py/state_machine.py)
-- never by world-gate diagnostics, which inform candidate SELECTION
only, per this mission's own explicit constraint.

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
