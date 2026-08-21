from __future__ import annotations

"""CLI entry point: candidate adapter -> frozen V2 policy ->
PARLAY_CERTIFICATION_V2 -> JSON, for exactly one MLB slate. Mirrors
sports/mlb/scripts/select_daily_parlay.py's CLI shape (pool-csv in,
--out-json out) for pipeline symmetry, but is otherwise fully independent
-- it never imports from or writes to the old parlay subsystem.

Pipeline (mission section 1B):
    multi-target predictive observations (candidate_adapter.build_pregame_action_rows)
        -> parlay candidate adapter (candidate_adapter.build_candidates_for_day)
        -> joint/state analysis (already inside build_candidates_for_day)
        -> bridge to CandidateWager (_to_candidate_wager, below -- policy
           independently recomputes its own world certificate; it does not
           trust the adapter's pre-computed diagnostics)
        -> frozen parlay decision policy (parlay_certification_v2.policy.select_action_for_day)
        -> PARLAY_CERTIFICATION_V2 output

D_S convention for the bridge (matches
joint_position_builder_v2/pairs.py's documented rule, reused here by
citation rather than import to keep parlay_v2 independent of the archived
research package): for a cross-game pair, the accepted combined price is
the product of each leg's own real decimal price -- the standard
sportsbook convention for a straight, non-SGP parlay. For a same-game
pair, no real quote exists (no SGP price/dependence model here), so the
combined price is None and the pair is never economically actionable --
consistent with mission section 6 ("initially prefer cross-game pairs
unless a real same-game quote and a supported dependence model exist").
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sports.mlb.conditional_chain.outcome_worlds import build_binary_outcome_set, build_world_distribution
from sports.mlb.research.parlay_certification_v2 import manifest
from sports.mlb.research.parlay_certification_v2.decision_record_store import DecisionRecordStore
from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
from sports.mlb.research.parlay_certification_v2.policy import CandidateWager, build_decision_record, select_action_for_day

from .calibration.snapshot import assert_snapshot_precedes_decision, build_snapshot
from .calibration.store import CalibrationStore
from .calibration.support import CandidateSupport, evaluate_support
from .candidate_adapter import Leg, PairCandidate, build_candidates_for_day, build_pregame_action_rows

# APS threshold for the PAIR world-model diagnostics (unrelated to the
# calibration LEDGER above -- see world_certificate.py / outcome_worlds.py).
# Not yet re-derived from real settled prospective days (see MIGRATION
# notes); frozen at 1.0 (retain-all) until that exists, exactly like
# h_over_ranker's frozen H_BIAS was frozen once from DEVELOPMENT_STAMPS.
FROZEN_APS_THRESHOLD = 1.0
FROZEN_CALIBRATION_SLATES = 0


def _leg_buckets(leg: Leg, *, predictive_version: str, state_version: str) -> tuple[str, str, str]:
    market_bucket = leg.target
    line_bucket = f"{leg.target}|{leg.side}|{leg.line}"
    state_bucket = f"{predictive_version}|{state_version}"
    return market_bucket, line_bucket, state_bucket


# Priority order for turning REQUIRED-dimension blocks into ONE specific
# abstain reason instead of an opaque catch-all (mission: "specific,
# non-generic abstain reason"). OBSERVE_ONLY dimensions (joint_support,
# shift_status) never appear here -- see calibration/support.py, they can
# never block by construction, so they can never be "the reason" either.
_BLOCKING_REASON_PRIORITY = (
    ("state_support", "NO_STATE_SUPPORT"),
    ("market_support", "NO_LEG_MARKET_SUPPORT"),
    ("line_support", "NO_LEG_LINE_SUPPORT"),
)


def _pair_support(
    candidate: PairCandidate,
    *,
    snapshot_rows: list[dict],
    independent_slate_count: int,
    predictive_version: str,
    state_version: str,
) -> tuple[CandidateSupport, CandidateSupport]:
    """Evaluates REQUIRED-dimension support on the REAL forward-only
    calibration ledger for both legs of a candidate (mission section 2/5),
    replacing the earlier placeholder that hardcoded
    FROZEN_CALIBRATION_SLATES=0. `snapshot_rows`/`independent_slate_count`
    come from a SINGLE snapshot built once per slate (see
    build_slate_payload) -- not rebuilt per candidate. Returns both legs'
    CandidateSupport (never just a bool) so callers can report a specific
    blocking dimension rather than a generic reason."""
    results = []
    for leg in (candidate.leg_1, candidate.leg_2):
        market_bucket, line_bucket, state_bucket = _leg_buckets(leg, predictive_version=predictive_version, state_version=state_version)
        results.append(
            evaluate_support(
                snapshot_rows,
                market_bucket=market_bucket,
                line_bucket=line_bucket,
                state_bucket=state_bucket,
                independent_slate_count=independent_slate_count,
            )
        )
    return results[0], results[1]


def _pair_in_support(support_pair: tuple[CandidateSupport, CandidateSupport]) -> bool:
    return support_pair[0].in_support and support_pair[1].in_support


def _aggregate_blocking_reason(support_pairs: list[tuple[CandidateSupport, CandidateSupport]]) -> str:
    """One specific abstain reason for a whole slate, derived honestly from
    every REQUIRED dimension that blocked at least one candidate leg --
    never a generic catch-all. Priority order matches
    _BLOCKING_REASON_PRIORITY (state > market > line): state_support is
    checked first because it reflects overall ledger maturity (independent
    slate count), which is the most informative single fact for an
    operator deciding whether to just wait longer."""
    blocking: set[str] = set()
    for support_i, support_j in support_pairs:
        blocking.update(support_i.blocking_dimensions)
        blocking.update(support_j.blocking_dimensions)
    for dim_name, reason in _BLOCKING_REASON_PRIORITY:
        if dim_name in blocking:
            return reason
    return "NO_PAIR_IN_SUPPORT"  # defensive fallback; unreachable if blocking is non-empty


def _candidate_priced(candidate: PairCandidate) -> bool:
    return candidate.leg_1.decimal_price is not None and candidate.leg_2.decimal_price is not None


def _best_shadow_candidate(candidates: list[PairCandidate]) -> PairCandidate | None:
    """Best-available, really-priced, cross-game candidate, ranked by
    joint_probability_estimate -- NOT `joint_score` (retained_probability_mass),
    which is uninformative while the frozen APS threshold retains every
    world (aps_threshold=1.0, see FROZEN_APS_THRESHOLD).

    This is a DISPLAY-ONLY ranking, not a certification, and it never
    influences `action`/`selected_parlay`: the Parlays tab always has
    something concrete to show on an abstain day, per the product spec
    ('Do not hide useful candidate diagnostics merely because policy
    support is not yet proven'). It is never gated on calibration
    support -- an uncertified shadow pick is exactly what it is
    regardless of ledger state.

    Known caveat, worth being honest about rather than hiding: ranking by
    raw joint probability reproduces the exact selection bias this
    program's own research already found (see joint_position_builder_v2/
    STATE.md -- restricting to a pool's highest-probability legs
    concentrates the frozen marginal model's worst overconfidence). That
    finding is exactly why this value is never used to gate `action` --
    it is fine as a display convenience for an explicitly uncertified
    example, but must never be mistaken for a reliability signal, which
    is also why the frontend surfaces it as a bare 'shadow candidate',
    with no probability or score shown next to it."""
    priced = [c for c in candidates if c.leg_1.game_id != c.leg_2.game_id and _candidate_priced(c)]
    if not priced:
        return None
    return max(priced, key=lambda c: c.joint_probability_estimate)


def _to_candidate_wager(candidate: PairCandidate) -> CandidateWager:
    leg_i, leg_j = candidate.leg_1, candidate.leg_2
    same_game = leg_i.game_id == leg_j.game_id
    world_id_i, world_id_j = "leg_1", "leg_2"
    clipped = np.clip([leg_i.model_probability_estimate, leg_j.model_probability_estimate], 1e-4, 1 - 1e-4)
    distribution = build_world_distribution([world_id_i, world_id_j], clipped)
    outcome_set = build_binary_outcome_set(distribution, aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=FROZEN_CALIBRATION_SLATES)

    decimal_price = None
    if not same_game and leg_i.decimal_price is not None and leg_j.decimal_price is not None:
        decimal_price = float(leg_i.decimal_price) * float(leg_j.decimal_price)

    losing_world_ids = np.array([w for w in range(4) if w != 3])  # world 3 = both legs win
    return CandidateWager(
        wager_id=candidate.candidate_id,
        decimal_price=decimal_price,
        retained_world_ids=outcome_set.world_ids,
        world_probabilities=distribution.probabilities,
        losing_world_ids=losing_world_ids,
        book=(leg_i.book if leg_i.book == leg_j.book else "mixed"),
    )


def build_slate_payload(
    *,
    pool_csv: pd.DataFrame,
    slate_id: str,
    eligibility_inputs: EligibilityInputs,
    predictive_version: str,
    state_version: str,
    mode: str = "broad",
    calibration_store: CalibrationStore | None = None,
    decision_record_store: DecisionRecordStore | None = None,
    world_gate_mode: str = "REQUIRED",
    world_risk_threshold: float | None = None,
) -> dict:
    """Enforces the required daily ordering (mission section 2):
    eligibility -> capture immutable pregame candidate universe -> load
    calibration snapshot (PRIOR settled slates only, cutoff = the
    calibration_as_of timestamp captured HERE, before any candidate work)
    -> build candidates -> compute support from that snapshot only -> V2
    select_action_for_day -> freeze ACT/ABSTAIN. Today's own outcomes are
    never read (this function never touches settlement data at all), and
    cannot enter the calibration ledger until a SEPARATE, later admission
    step runs after settlement is final (see calibration/store.py's
    forward-only invariant -- enforced there, not here).

    world_gate_mode/world_risk_threshold default to REQUIRED/None --
    byte-identical to this function's behavior before the "APS /
    counterexample admission bottleneck" research pass, and what every
    REAL production/CI invocation still uses (PARLAY_POLICY_V2_
    PROSPECTIVE_002's frozen config). OBSERVE_ONLY/BOUNDED_RISK are
    implemented and tested (see policy.select_action_for_day and
    world_gate_research.py) but are NOT wired into any real invocation
    yet -- see manifest.ALPHA_BUDGET_BLOCKS_PROSPECTIVE_003."""
    eligibility = evaluate_eligibility(eligibility_inputs)
    # Captured BEFORE any candidate/support work -- this is the cutoff
    # every support calculation below is pinned to, strictly before
    # decision_frozen_at (captured later, at the bottom of this function).
    calibration_as_of = datetime.now(timezone.utc).isoformat()

    payload = {
        "system": "PARLAY_POLICY_V2",
        "policy_version": manifest.POLICY_VERSION,
        "policy_status": manifest.STATUS,
        "world_gate_mode": world_gate_mode,
        "eligible": eligibility.eligible,
        "eligibility_reason": eligibility.reason,
        # `action`/`selected_parlay` kept for frontend backward-compat.
        # `selection_status`/`shadow_execution_status`/`staking_authorized`
        # are the new three-way-separated status fields (mission sections
        # 14-17): POLICY SELECTION (this function, allowed while unproven)
        # vs PRODUCTION/STAKING AUTHORIZATION (staking_authorized -- never
        # settable True here, only ever by the outer certificate reaching
        # SUPPORTED_CURRENT) vs POLICY CERTIFICATION (policy_status, driven
        # solely by PARLAY_CERTIFICATION_V2's own state machine).
        "action": "ABSTAIN",
        "selected_parlay": None,
        "selection_status": "ABSTAIN",
        "shadow_execution_status": "NOT_EXECUTED",
        "staking_authorized": False,
        "shadow_candidate": None,
        "shadow_candidate_note": (
            "Best available candidate by descriptive joint score -- NOT a certified pick. "
            "The frozen policy has not authorized action on it; see policy_status."
        ),
        "evidence_status": {
            "state_machine_status": manifest.STATUS,
            "note": "STATUS reflects the FROZEN POLICY's evidence state, never this slate's own candidates -- see manifest.CONCLUSION_REASONING.",
        },
        "calibration_as_of": calibration_as_of,
    }
    if not eligibility.eligible:
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "OPERATIONALLY_INELIGIBLE"
        return payload

    action_rows = build_pregame_action_rows(pool_csv, stamp=slate_id, mode=mode)
    if action_rows.empty:
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "NO_REAL_QUOTE"
        return payload

    # Candidates are built unconditionally (not only when calibration
    # support might pass) so the Parlays tab always has a real, priced
    # shadow candidate to show on an abstain day -- see
    # _best_shadow_candidate. This is diagnostic-only and never gated on
    # calibration support; it never sets `action`/`selected_parlay`, and
    # never feeds back into which candidate today's REAL selection below
    # considers (certification independence -- no feedback from
    # policy_status into candidate identity).
    candidates = build_candidates_for_day(
        action_rows,
        slate_id=slate_id,
        aps_threshold=FROZEN_APS_THRESHOLD,
        calibration_slates=FROZEN_CALIBRATION_SLATES,
        predictive_version=predictive_version,
        state_version=state_version,
    )
    shadow_candidate = _best_shadow_candidate(candidates)
    if shadow_candidate is not None:
        payload["shadow_candidate"] = shadow_candidate.as_dict()

    # POLICY_NOT_FROZEN guard: real ACT selection only counts as
    # confirmatory prospective evidence once this policy_version's
    # freeze boundary has been deliberately set (prospective_boundary.py)
    # -- selecting before that point would just be more DEVELOPMENT/SHADOW
    # noise mistaken for prospective evidence. This is NOT the old
    # circular block: it depends only on manifest.STATUS (a deliberate,
    # one-time human freeze action), never on joint_support/shift_status
    # or any other per-candidate support dimension, and it never applies
    # to the shadow_candidate display above.
    if manifest.STATUS == "DEVELOPMENT":
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "POLICY_NOT_FROZEN"
        return payload

    # calibration_store is None (e.g. no --calibration-ledger passed) is
    # treated as an honest ZERO accumulated observations, not a special
    # circular early-return: state_support/market_support/line_support
    # will correctly report FAIL for lack of data, which the aggregation
    # below surfaces as a real, specific reason -- exactly what would
    # happen with a real-but-empty ledger. There is no separate code path.
    calibration_snapshot = build_snapshot(calibration_store, as_of=calibration_as_of) if calibration_store is not None else None
    calibration_rows = calibration_store.observations_as_of(calibration_as_of) if calibration_store is not None else []
    independent_slate_count = calibration_snapshot.independent_slate_count if calibration_snapshot is not None else 0

    # Section 6: prefer cross-game pairs -- same-game pairs carry no real
    # quote by construction (see _to_candidate_wager), so they would never
    # be economically actionable anyway; excluding them here is purely an
    # efficiency filter, not a separate authority decision.
    cross_game_candidates = [c for c in candidates if c.leg_1.game_id != c.leg_2.game_id]
    decision_frozen_at = datetime.now(timezone.utc).isoformat()
    payload["decision_timestamp_utc"] = decision_frozen_at
    if calibration_snapshot is not None:
        # Mission section 4, enforced with a strict comparison, not merely
        # documented: refuse to proceed if the snapshot this decision
        # relied on doesn't strictly precede the decision itself.
        assert_snapshot_precedes_decision(calibration_snapshot, decision_frozen_at)
        payload["calibration_snapshot_id"] = calibration_snapshot.calibration_snapshot_id
        payload["calibration_snapshot_sha256"] = calibration_snapshot.calibration_snapshot_sha256

    if not cross_game_candidates:
        payload["abstain_reason"] = "NO_CANDIDATES"
        return payload

    # Support gating comes from the REAL forward-only calibration ledger
    # (calibration_as_of, captured above, before any of this work), using
    # ONLY the REQUIRED dimensions -- market_support/line_support/
    # state_support -- via calibration/support.py's GateMode. This is the
    # fix for the circular dependency this module's docstring describes:
    # joint_support/shift_status are OBSERVE_ONLY and can never block here,
    # so selection genuinely becomes possible once real REQUIRED evidence
    # accumulates, instead of being permanently unreachable.
    support_pairs = [
        _pair_support(c, snapshot_rows=calibration_rows, independent_slate_count=independent_slate_count, predictive_version=predictive_version, state_version=state_version)
        for c in cross_game_candidates
    ]
    supported_candidates = [c for c, sp in zip(cross_game_candidates, support_pairs) if _pair_in_support(sp)]
    if not supported_candidates:
        payload["abstain_reason"] = _aggregate_blocking_reason(support_pairs)
        return payload

    # Deterministic, quality-blind truncation -- see
    # manifest.MAX_CANDIDATES_PER_SLATE's docstring. Sorted by
    # candidate_id (a structural identity key), never by predicted
    # probability/price/joint score/etc.
    if len(supported_candidates) > manifest.MAX_CANDIDATES_PER_SLATE:
        supported_candidates = sorted(supported_candidates, key=lambda c: c.candidate_id)[: manifest.MAX_CANDIDATES_PER_SLATE]

    wagers = [_to_candidate_wager(c) for c in supported_candidates]
    by_wager_id = {c.candidate_id: c for c in supported_candidates}

    selection = select_action_for_day(
        eligibility, wagers, r_max=manifest.R_MAX_ACCEPTED,
        world_gate_mode=world_gate_mode, world_risk_threshold=world_risk_threshold,
    )
    decision_record = build_decision_record(
        date=slate_id,
        eligibility=eligibility,
        decision_timestamp_utc=decision_frozen_at,
        predictive_model_version=predictive_version,
        candidate_universe_size=len(supported_candidates),
        action_selection=selection,
        c=manifest.C_MIN_COVERAGE,
        r=manifest.R_MAX_LOSS_RISK,
        delta=manifest.DELTA_MIN_RETURN,
        r_max=manifest.R_MAX_ACCEPTED,
        world_gate_mode=world_gate_mode,
    )
    payload["decision_record"] = {
        "eligible": decision_record.eligible,
        "policy_version": decision_record.policy_version,
        "candidate_universe_size": decision_record.candidate_universe_size,
        "action": decision_record.action,
        "world_certificate_diagnostics": decision_record.world_certificate_diagnostics,
    }
    # Durable persistence (mission section 19/27): the ephemeral per-run
    # JSON this payload becomes does not survive past this CI run (see
    # decision_record_store.py's module docstring) -- this is the ONLY
    # place a DecisionRecord is admitted, and it is admitted exactly once
    # per date, idempotently, regardless of ACT/ABSTAIN outcome, so
    # settle_evidence.py has something real to grade later. This never
    # revises an already-frozen decision.
    if decision_record_store is not None:
        decision_record_store.admit(decision_record)

    if selection.action == 0:
        # Support-passing candidates existed, but none was certified by
        # the frozen G_C/G_L/G_V machinery (no real quote, price outside
        # D_MAX, or a vacuous world certificate -- see policy.py's own
        # select_action_for_day, untouched by this change).
        payload["abstain_reason"] = "NO_PAIR_PASSES_FROZEN_POLICY"
        return payload

    selected_candidate = by_wager_id[selection.selected.wager_id]
    payload["action"] = "ACT"
    payload["selected_parlay"] = selected_candidate.as_dict()
    payload["selection_status"] = "SELECTED"
    # Shadow action semantics (mission): A_t=1 means the frozen policy
    # selected one exact wager at a real frozen quote, regardless of
    # whether real money is staked. staking_authorized stays False --
    # selection alone never authorizes production/real-money staking; only
    # the outer PARLAY_CERTIFICATION_V2 state machine reaching
    # SUPPORTED_CURRENT may ever change that, and never from this module.
    payload["shadow_execution_status"] = "EXECUTED_SHADOW"
    payload["staking_authorized"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PARLAY_POLICY_V2 for one MLB slate (Parlays tab source).")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--slate-id", type=str, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--mode", choices=["narrow", "broad"], default="broad")
    parser.add_argument("--predictive-version", type=str, default="H_OVER_RANKER_V1+MULTI_TARGET")
    parser.add_argument("--state-version", type=str, default="MULTI_TARGET_BROAD_V1")
    parser.add_argument("--calibration-ledger", type=Path, default=None, help="Path to the forward-only calibration ledger JSONL (sports/mlb/parlay_v2/calibration/store.py). Omit to run with no calibration support (always abstains).")
    parser.add_argument("--decision-record-ledger", type=Path, default=None, help="Path to the durable per-day DecisionRecord ledger (decision_record_store.py). Omit to skip persistence (e.g. for local/manual runs).")
    parser.add_argument(
        "--world-gate-mode", choices=["REQUIRED", "BOUNDED_RISK", "OBSERVE_ONLY"], default=manifest.WORLD_GATE_MODE,
        help=(
            "How world/counterexample information participates in admission (policy.select_action_for_day). "
            f"Defaults to manifest.WORLD_GATE_MODE ({manifest.WORLD_GATE_MODE!r}) -- the CURRENTLY FROZEN policy's "
            "config, i.e. what a real production/CI run actually uses with no extra flags. Pass explicitly to "
            "replay/audit an earlier frozen policy version's exact behavior (e.g. --world-gate-mode REQUIRED for "
            "PARLAY_POLICY_V2_PROSPECTIVE_002)."
        ),
    )
    parser.add_argument(
        "--world-risk-threshold", type=float, default=manifest.WORLD_RISK_THRESHOLD,
        help="Required only for --world-gate-mode BOUNDED_RISK. Defaults to manifest.WORLD_RISK_THRESHOLD.",
    )
    args = parser.parse_args()

    pool_exists = args.pool_csv.exists()
    pool_csv = pd.read_csv(args.pool_csv, low_memory=False) if pool_exists else pd.DataFrame()
    eligibility_inputs = EligibilityInputs(
        date=args.slate_id,
        required_feed_available=pool_exists,
        slate_has_mlb_games=bool(pool_exists and len(pool_csv) > 0),
        required_system_component_available=True,
        decision_cutoff_met=True,
    )
    calibration_store = CalibrationStore(args.calibration_ledger) if args.calibration_ledger else None
    decision_record_store = DecisionRecordStore(args.decision_record_ledger) if args.decision_record_ledger else None
    payload = build_slate_payload(
        pool_csv=pool_csv,
        slate_id=args.slate_id,
        eligibility_inputs=eligibility_inputs,
        predictive_version=args.predictive_version,
        state_version=args.state_version,
        mode=args.mode,
        calibration_store=calibration_store,
        decision_record_store=decision_record_store,
        world_gate_mode=args.world_gate_mode,
        world_risk_threshold=args.world_risk_threshold,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {args.out_json} action={payload['action']}")


if __name__ == "__main__":
    main()
