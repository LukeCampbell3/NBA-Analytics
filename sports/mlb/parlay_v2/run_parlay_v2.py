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
from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
from sports.mlb.research.parlay_certification_v2.policy import CandidateWager, build_decision_record, select_action_for_day

from .calibration.snapshot import assert_snapshot_precedes_decision, build_snapshot
from .calibration.store import CalibrationStore
from .calibration.support import evaluate_support, support_is_structurally_unreachable
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


def _pair_in_support(
    candidate: PairCandidate,
    *,
    snapshot_rows: list[dict],
    independent_slate_count: int,
    predictive_version: str,
    state_version: str,
) -> bool:
    """Gates on the REAL forward-only calibration ledger (mission section
    2/5), replacing the earlier placeholder that hardcoded
    FROZEN_CALIBRATION_SLATES=0. `snapshot_rows`/`independent_slate_count`
    come from a SINGLE snapshot built once per slate (see
    build_slate_payload) -- not rebuilt per candidate."""
    for leg in (candidate.leg_1, candidate.leg_2):
        market_bucket, line_bucket, state_bucket = _leg_buckets(leg, predictive_version=predictive_version, state_version=state_version)
        support = evaluate_support(
            snapshot_rows,
            market_bucket=market_bucket,
            line_bucket=line_bucket,
            state_bucket=state_bucket,
            independent_slate_count=independent_slate_count,
        )
        if not support.in_support:
            return False
    return True


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
    forward-only invariant -- enforced there, not here)."""
    eligibility = evaluate_eligibility(eligibility_inputs)
    # Captured BEFORE any candidate/support work -- this is the cutoff
    # every support calculation below is pinned to, strictly before
    # decision_frozen_at (captured later, at the bottom of this function).
    calibration_as_of = datetime.now(timezone.utc).isoformat()

    payload = {
        "system": "PARLAY_POLICY_V2",
        "policy_version": manifest.POLICY_VERSION,
        "policy_status": manifest.STATUS,
        "eligible": eligibility.eligible,
        "eligibility_reason": eligibility.reason,
        "action": "ABSTAIN",
        "selected_parlay": None,
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

    if calibration_store is None:
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "CERTIFICATION_STREAM_NOT_READY"
        return payload

    # Cheap, honest short-circuit: today, support can NEVER pass for any
    # candidate regardless of ledger content (joint_support/shift_status
    # are still UNESTABLISHED -- see calibration/support.py). Skip the
    # expensive O(n^2) pair enumeration entirely rather than building tens
    # of thousands of certificates only to discard them all downstream.
    if support_is_structurally_unreachable():
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "NO_PAIR_IN_SUPPORT"
        return payload

    # ONE snapshot per slate (mission section 4: "Every V2 decision must
    # reference an immutable calibration snapshot"), built once and reused
    # for every candidate -- not rebuilt per pair.
    calibration_snapshot = build_snapshot(calibration_store, as_of=calibration_as_of)
    calibration_rows = calibration_store.observations_as_of(calibration_as_of)

    candidates = build_candidates_for_day(
        action_rows,
        slate_id=slate_id,
        aps_threshold=FROZEN_APS_THRESHOLD,
        calibration_slates=FROZEN_CALIBRATION_SLATES,
        predictive_version=predictive_version,
        state_version=state_version,
    )
    # Section 6: prefer cross-game pairs -- same-game pairs carry no real
    # quote by construction (see _to_candidate_wager), so they would never
    # be economically actionable anyway; excluding them here is purely an
    # efficiency filter, not a separate authority decision.
    #
    # Support gating now comes from the REAL forward-only calibration
    # ledger (calibration_as_of, captured above, before any of this
    # work) -- not the earlier placeholder. See _pair_in_support /
    # calibration/support.py: with the joint_support and shift_status
    # dimensions still UNESTABLISHED (no validated research exists for
    # either), every candidate correctly reports not-in-support right
    # now, regardless of ledger size -- this is an honest limitation, not
    # a bug (see calibration/support.py's module docstring).
    cross_game_candidates = [
        c for c in candidates
        if c.leg_1.game_id != c.leg_2.game_id
        and _pair_in_support(c, snapshot_rows=calibration_rows, independent_slate_count=calibration_snapshot.independent_slate_count, predictive_version=predictive_version, state_version=state_version)
    ]
    decision_frozen_at = datetime.now(timezone.utc).isoformat()
    payload["decision_timestamp_utc"] = decision_frozen_at
    # Mission section 4, enforced with a strict comparison, not merely
    # documented: refuse to proceed if the snapshot this decision relied
    # on doesn't strictly precede the decision itself.
    assert_snapshot_precedes_decision(calibration_snapshot, decision_frozen_at)
    payload["calibration_snapshot_id"] = calibration_snapshot.calibration_snapshot_id
    payload["calibration_snapshot_sha256"] = calibration_snapshot.calibration_snapshot_sha256
    if not cross_game_candidates:
        payload["abstain_reason"] = "NO_PAIR_IN_SUPPORT"
        return payload

    wagers = [_to_candidate_wager(c) for c in cross_game_candidates]
    by_wager_id = {c.candidate_id: c for c in cross_game_candidates}

    selection = select_action_for_day(eligibility, wagers, r_max=manifest.R_MAX_ACCEPTED)
    decision_record = build_decision_record(
        date=slate_id,
        eligibility=eligibility,
        decision_timestamp_utc=decision_frozen_at,
        predictive_model_version=predictive_version,
        candidate_universe_size=len(cross_game_candidates),
        action_selection=selection,
        c=manifest.C_MIN_COVERAGE,
        r=manifest.R_MAX_LOSS_RISK,
        delta=manifest.DELTA_MIN_RETURN,
        r_max=manifest.R_MAX_ACCEPTED,
    )
    payload["decision_record"] = {
        "eligible": decision_record.eligible,
        "policy_version": decision_record.policy_version,
        "candidate_universe_size": decision_record.candidate_universe_size,
        "action": decision_record.action,
        "world_certificate_diagnostics": decision_record.world_certificate_diagnostics,
    }

    if selection.action == 0:
        payload["abstain_reason"] = "NO_PAIR_PASSES_FROZEN_POLICY"
        return payload

    selected_candidate = by_wager_id[selection.selected.wager_id]
    payload["action"] = "ACT"
    payload["selected_parlay"] = selected_candidate.as_dict()
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
    payload = build_slate_payload(
        pool_csv=pool_csv,
        slate_id=args.slate_id,
        eligibility_inputs=eligibility_inputs,
        predictive_version=args.predictive_version,
        state_version=args.state_version,
        mode=args.mode,
        calibration_store=calibration_store,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {args.out_json} action={payload['action']}")


if __name__ == "__main__":
    main()
