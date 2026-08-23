from __future__ import annotations

"""FROZEN manifest for NFL PARLAY_CERTIFICATION_V2 -- the sole
authoritative certification/decision layer for the NFL parlay research
system. Ported from sports/mlb/research/parlay_certification_v2/manifest.py,
adapted for NFL's own first prospective policy attempt.

Supersedes nothing existing: NFL's only prior parlay logic
(sports/nfl/predictions/daily_policy.build_shadow_parlay) already
self-disables (status="withheld", candidate_authorized=False) after a real
locked-2022-holdout backtest went 2-16, and is explicitly left untouched,
diagnostic-only CONTROL -- see legacy_control.py/comparison.py, and the
MIGRATION note at the bottom of this file.

production_authorized stays False here unconditionally, always. Only the
outer V2 prospective certificate (state_machine.PolicyStatus reaching
SUPPORTED_CURRENT) may ever be the basis for advancing it, and even then
this file does not flip it automatically -- see CONCLUSION_REASONING.
"""

VERSION = "NFL_PARLAY_CERTIFICATION_V2"
PRODUCTION_AUTHORIZED = False  # never set programmatically

# Frozen sub-versions. A change to ANY of these is a NEW policy version
# with its own prospective evidence stream (evidence_store already
# enforces this structurally via one file per policy_version). Values
# match the ported certification-core modules' own internal version
# constants exactly (settlement.py/world_certificate.py/evidence_store.py/
# anytime_monitor.py/state_machine.py) -- these are sport-agnostic labels,
# unchanged from MLB's manifest, because the ported code itself is
# unchanged.
ELIGIBILITY_VERSION = "ELIGIBILITY_V1"
POLICY_VERSION = "NFL_PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION"
SETTLEMENT_VERSION = "SETTLEMENT_V1"
WORLD_CERTIFICATE_VERSION = "NONVACUOUS_WORLD_CERTIFICATE_V1"
EVIDENCE_STORE_VERSION = "EVIDENCE_STORE_V1"
REFERENCE_MONITOR_VERSION = "PARLAY_REFERENCE_ANYTIME_CERT_V1"
STATE_MACHINE_VERSION = "PARLAY_CERTIFICATION_STATE_V1"

MAX_ACTIONS_PER_ELIGIBLE_SLATE = 1
TWO_LEG_PARLAYS_ONLY = True

# CANDIDATE-SET SIZE BOUND. NFL's own MAXIMUM_WEEKLY_PICKS=6
# (daily_policy.py) already caps a week's action-eligible play list to at
# most 6, so C(6,2)=15 candidate pairs at most per week -- this bound is
# reached nowhere near as often as MLB's much larger daily pool, but it is
# kept at the same value for structural parity and as a harmless ceiling
# should NFL's own pick volume ever grow. Deterministic, quality-blind
# truncation (sorted by candidate_id) if it is ever reached -- never a
# quality gate.
MAX_CANDIDATES_PER_SLATE = 500

# Predeclared targets -- frozen BEFORE any prospective evaluation, reusing
# MLB's own conservative placeholders unchanged (they were never derived
# from MLB-specific evidence either -- see MLB manifest.py's own comment).
# MUST NOT be tuned post-hoc based on prospective results -- a change
# requires a new policy version, never an edit to these constants after
# real evidence exists.
C_MIN_COVERAGE = 0.50  # c: minimum fraction of eligible weeks the policy must act on
R_MAX_LOSS_RISK = 0.30  # r: maximum acceptable selective loss risk
DELTA_MIN_RETURN = 0.0  # delta: minimum expected return per action (breakeven -- no positive-edge claim a priori)

# ECONOMIC BOUNDS. D_max is the actual frozen config; R_MAX_ACCEPTED is
# ALWAYS derived from it, never configured independently. D_max=6.0 is
# carried over from MLB's manifest unchanged, and checked (not assumed)
# against NFL's own real, frozen price bounds
# (daily_policy.MINIMUM_AMERICAN_PRICE=-150.0, MAXIMUM_AMERICAN_PRICE=130.0):
# each leg's decimal price is bounded to roughly [1.667, 2.3], so a 2-leg
# product is bounded to roughly [2.78, 5.29] -- comfortably inside D_max=6.0
# with no widening needed.
D_MAX = 6.0
R_MAX_ACCEPTED = D_MAX - 1.0  # derived -- never edit this independently of D_MAX

ALPHA_TOTAL = 0.05
ALPHA_C = ALPHA_TOTAL / 3.0
ALPHA_L = ALPHA_TOTAL / 3.0
ALPHA_V = ALPHA_TOTAL / 3.0

# PROGRAM-LEVEL MULTIPLICITY. ALPHA_TOTAL above is THIS policy version's
# within-policy budget (alpha_policy_k); it must itself be drawn from a
# program-level ledger shared across all NFL policy versions ever frozen
# for prospective confirmation, so that repeatedly freezing new versions
# after a failure/demotion cannot silently reset to a fresh alpha. See
# sports/nfl/parlay_v2/program_alpha.py -- ProgramAlphaLedger.spend(...)
# is the only way ALPHA_TOTAL above is actually authorized to be spent;
# this constant alone does not spend it. This is a SEPARATE ledger from
# MLB's own program_alpha_ledger.json -- the two programs' alpha budgets
# are never shared or comingled.
ALPHA_PROGRAM = 0.05
PROGRAM_ALPHA_LEDGER_PATH = "sports/nfl/research/parlay_certification_v2/reports/program_alpha_ledger.json"

# CURRENT-WEEK FREEZE BOUNDARY. The real boundary timestamp lives in
# prospective_boundary.py's one-way marker file, NOT as an editable
# constant in this frozen manifest. See
# prospective_boundary.read_prospective_start_timestamp(). Until that
# marker is set, EVERY evaluated week -- including any already inspected
# while building this integration -- counts only as DEVELOPMENT/SHADOW,
# never as confirmatory prospective evidence.

# PROSPECTIVE POLICY IDENTIFIER. This is NFL's FIRST frozen prospective
# attempt -- there is no prior NFL PARLAY_V2 policy version, so
# PRIOR_PROSPECTIVE_POLICY_IDS is empty (unlike MLB's manifest, which
# carries PROSPECTIVE_002's audit trail). PROSPECTIVE_POLICY_ID names the
# CURRENTLY FROZEN attempt -- the one freeze_prospective.py has activated
# a one-way boundary for, and the one a real production/CI invocation of
# run_parlay_v2.py actually uses (main() defaults world_gate_mode/
# world_risk_threshold to WORLD_GATE_MODE/WORLD_RISK_THRESHOLD below).
PRIOR_PROSPECTIVE_POLICY_IDS: tuple[str, ...] = ()
PROSPECTIVE_POLICY_ID = "NFL_PARLAY_POLICY_V2_PROSPECTIVE_001"

# FROZEN GATE-MODE CONFIGURATION for PROSPECTIVE_POLICY_ID above. REQUIRED
# dimensions block action when not PASS; OBSERVE_ONLY dimensions are
# computed and exposed for research but can NEVER block, regardless of
# status (including UNESTABLISHED) -- see calibration/support.py's
# GateMode/SupportDimension for the authoritative implementation this
# dict is the frozen manifest record of. This gate-mode split (three real,
# implemented REQUIRED dimensions; two permanently-unestablished
# OBSERVE_ONLY dimensions) is a sport-agnostic design decision, carried
# over unchanged from MLB's own (already-corrected) manifest -- it was
# never MLB-specific to begin with. Promoting joint_support or
# shift_status to REQUIRED requires an independently validated,
# non-arbitrary threshold AND a new PROSPECTIVE_POLICY_ID -- never an edit
# to this dict in place once frozen.
SUPPORT_GATE_MODES = {
    "market_support": "REQUIRED",
    "line_support": "REQUIRED",
    "state_support": "REQUIRED",
    "joint_support": "OBSERVE_ONLY",
    "shift_status": "OBSERVE_ONLY",
}

# ============================================================
# WORLD-GATE CONFIGURATION for PROSPECTIVE_POLICY_ID above.
#
# NFL starts DIRECTLY at world_gate_mode=OBSERVE_ONLY -- it does NOT
# repeat MLB's PROSPECTIVE_002 mistake of freezing at REQUIRED first. This
# is a DELIBERATE, DISCLOSED INFERENCE from MLB's already-completed
# research (sports/mlb/research/parlay_certification_v2/world_gate_research.py
# and CONCLUSION_REASONING in MLB's own manifest.py), not a fresh
# NFL-specific empirical finding -- no NFL-specific DERIVE->SELECT
# research pass has been run. The inference is sound because the specific
# thing MLB's research found degenerate is a STRUCTURAL/MATHEMATICAL
# property of the generic world-certificate machinery itself
# (world_certificate.py/outcome_worlds.py, both ported byte-for-byte into
# sports/nfl/...), not an MLB-specific empirical fact:
#   - counterexample_mass at FROZEN_APS_THRESHOLD=1.0 (retain-all) is
#     PROVABLY IDENTICAL to 1 - predicted_joint_probability whenever the
#     joint model is the independence-only construction
#     build_world_distribution implements (no fitted dependence model
#     exists in this codebase, for MLB or NFL) -- this is an identity of
#     the math, true for any two-leg independence-model pair regardless
#     of sport, not a property MLB's real data happened to exhibit.
#   - Consequently world_gate_mode=REQUIRED (gating on
#     `certified`/`world_risk_rho` at that threshold) is operationally
#     degenerate FOR THE SAME STRUCTURAL REASON in NFL as in MLB: it adds
#     no gating information beyond the raw joint-probability baseline,
#     which is separately, honestly evaluated by calibration/support.py's
#     REQUIRED market/line/state dimensions regardless of world_gate_mode.
# OBSERVE_ONLY keeps world/counterexample diagnostics recorded (never
# hidden) and available as a ranking diagnostic (ascending world_risk_rho,
# inherited unchanged) without ever letting them block real admission --
# exactly the same non-gating role MLB's research justified.
#
# If NFL's own real prospective evidence later suggests this inference
# doesn't hold (e.g. a materially different price/probability
# distribution than MLB's), that is a question for a NEW NFL-specific
# research pass and a NEW PROSPECTIVE_POLICY_ID -- never a silent edit
# here.
WORLD_GATE_MODE = "OBSERVE_ONLY"
WORLD_RISK_THRESHOLD = None  # OBSERVE_ONLY never gates -- no threshold needed

# ALPHA BUDGET AUDIT. Unlike MLB, there is no PRIOR spend to audit or
# retire -- NFL_PARLAY_POLICY_V2_PROSPECTIVE_001 is this program's first
# ever alpha spend, so this is always False and there is no conflict to
# resolve.
ALPHA_BUDGET_BLOCKS_PROSPECTIVE_001 = False

# PolicyStatus value (state_machine.PolicyStatus). Advanced to
# FROZEN_PROSPECTIVE_INCONCLUSIVE as a deliberate freeze action for
# PROSPECTIVE_POLICY_ID above -- performed, in order: (1) this manifest's
# constants reviewed and finalized, (2) alpha spend recorded in
# sports/nfl/research/parlay_certification_v2/reports/program_alpha_ledger.json
# via program_alpha.ProgramAlphaLedger.spend, (3) the one-way
# prospective_start boundary activated via
# sports/nfl/parlay_v2/freeze_prospective.py --confirm (see
# sports/nfl/research/parlay_certification_v2/reports/prospective_boundary/
# NFL_PARLAY_POLICY_V2_PROSPECTIVE_001_prospective_start.json for the real,
# checkable timestamp). See CONCLUSION_REASONING below.
STATUS = "FROZEN_PROSPECTIVE_INCONCLUSIVE"

CONCLUSION_REASONING = """
This manifest establishes PARLAY_CERTIFICATION_V2 as the sole
authoritative certification/decision layer for the NFL parlay research
system, ported from MLB's own (real, production-verified)
implementation. Unlike MLB's manifest, this is NFL's first attempt: there
is no PROSPECTIVE_002-style prior failure to audit, and no alpha-budget
conflict to resolve -- NFL_PARLAY_POLICY_V2_PROSPECTIVE_001 spent its own
fresh 0.05 alpha_program with nothing to retire.

The one deliberate, disclosed choice this manifest makes up front is
starting world_gate_mode at OBSERVE_ONLY rather than REQUIRED -- see the
WORLD-GATE CONFIGURATION comment above for the full reasoning. This is an
inference from MLB's completed research about the world-certificate
machinery's STRUCTURAL behavior (an identity of the independence-only
joint model, true regardless of sport), not a claim that NFL-specific
research has been performed. It is disclosed here precisely so it is
never mistaken for that.

STATUS was advanced to FROZEN_PROSPECTIVE_INCONCLUSIVE on 2026-08-23,
before the 2026 NFL regular season had begun -- deliberately, so that
real prospective evidence begins accumulating from the season's first
eligible week rather than losing early weeks to a later freeze. As of
this freeze, zero real weeks have been evaluated under this policy (the
season had not started); the frozen boundary and this policy's config are
locked in now precisely so that changes later, once evidence exists,
would be visible as exactly that.

IMPORTANT -- this freeze is STILL NOT a claim of profitability or of
certified production-readiness. Whatever real coverage/loss/return
evidence PROSPECTIVE_POLICY_ID now accumulates is graded exclusively by
the unchanged, authoritative G_C/G_L/G_V simultaneous certificate
(anytime_monitor.py/state_machine.py) -- never by world-gate diagnostics,
which inform candidate SELECTION only.

Per MLB's precedent: implementation validation
(PARLAY_CERTIFICATION_V2_IMPLEMENTATION_VALIDATED, established by
sports/nfl/tests/test_parlay_certification_v2.py) is a claim about the
CERTIFICATION MACHINERY, not about the NFL policy's profitability. The
policy itself must remain FROZEN_PROSPECTIVE_INCONCLUSIVE (once frozen)
until real, untouched prospective evidence on eligible weeks satisfies
all three simultaneous bounds (coverage >= c, loss risk <= r, return >=
delta) at some horizon t, per anytime_monitor.py.
""".strip()

MIGRATION_SUMMARY = """
Nothing is superseded or replaced: sports/nfl/predictions/daily_policy.py
(POLICY_VERSION="nfl_passing_loss_aware_meta_policy_v2", including its own
PARLAY_POLICY_VERSION="nfl_distinct_game_parlay_v1" and
build_shadow_parlay) is left completely untouched, exactly as it was
before this system existed. It already self-labels its own parlay
ticket "shadow-only"/status="withheld"/candidate_authorized=False in its
own output, with an honest, already-recorded reason (2-16 on the locked
2022 holdout) -- so nothing downstream could ever mistake it for a live
authorization, and this migration does not need to change that.
sports/nfl/parlay_v2/legacy_control.py/comparison.py exist to read that
old system's output for CONTROL/comparison purposes only -- never to
feed anything back into it, and never the reverse.

This is an ADDITIVE product path, wired into
.github/workflows/nfl-predictions.yml alongside (never in place of) the
existing NFL prediction/export steps, mirroring exactly how MLB's
PARLAY_V2 was wired in additively next to select_daily_parlay.py.
""".strip()
