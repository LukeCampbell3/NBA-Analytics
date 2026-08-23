from __future__ import annotations

"""CLI entry point: candidate adapter -> frozen V2 policy ->
PARLAY_CERTIFICATION_V2 -> JSON, for exactly one NFL week. Ported from
sports/mlb/parlay_v2/run_parlay_v2.py, replacing MLB's pool-CSV input with
NFL's already-published weekly plays JSON (sports/nfl/web/data/
daily_predictions.json or an archived sports/nfl/data/production/snapshots/
.../*.json) -- it never imports from or writes to
sports/nfl/predictions/daily_policy.py's old shadow-parlay path.

Pipeline:
    this week's action-eligible plays (candidate_adapter.build_week_action_plays)
        -> parlay candidate adapter (candidate_adapter.build_candidates_for_week)
        -> joint/state analysis (already inside build_candidates_for_week)
        -> bridge to CandidateWager (_to_candidate_wager, below -- policy
           independently recomputes its own world certificate; it does not
           trust the adapter's pre-computed diagnostics)
        -> frozen parlay decision policy (parlay_certification_v2.policy.select_action_for_day)
        -> PARLAY_CERTIFICATION_V2 output

Pricing convention (matches MLB's run_parlay_v2.py, itself citing
joint_position_builder_v2/pairs.py's documented D_S rule by citation
rather than import): the accepted combined price is the product of each
leg's own real decimal price -- the standard sportsbook convention for a
straight, non-SGP parlay. Every candidate here is already cross-event AND
cross-player by construction (candidate_adapter.py's DISTINCTNESS RULE),
so unlike MLB there is no same-game case to special-case: a real price
always exists whenever both legs have one.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sports.nfl.conditional_chain.outcome_worlds import build_binary_outcome_set, build_world_distribution
from sports.nfl.research.parlay_certification_v2 import manifest
from sports.nfl.research.parlay_certification_v2.decision_record_store import DecisionRecordStore
from sports.nfl.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
from sports.nfl.research.parlay_certification_v2.policy import CandidateWager, build_decision_record, select_action_for_day

from .calibration.snapshot import assert_snapshot_precedes_decision, build_snapshot
from .calibration.store import CalibrationStore
from .calibration.support import N_STATE, CandidateSupport, evaluate_support
from .candidate_adapter import Leg, PairCandidate, build_candidates_for_week, build_week_action_plays

# APS threshold for the PAIR world-model diagnostics (unrelated to the
# calibration LEDGER above -- see world_certificate.py / outcome_worlds.py).
# Frozen at 1.0 (retain-all) -- matches MLB's identical frozen value; see
# manifest.py's WORLD-GATE CONFIGURATION comment for why NFL starts at
# world_gate_mode=OBSERVE_ONLY rather than re-deriving this from scratch.
FROZEN_APS_THRESHOLD = 1.0
FROZEN_CALIBRATION_SLATES = 0


def _leg_buckets(leg: Leg, *, predictive_version: str, state_version: str) -> tuple[str, str, str]:
    market_bucket = leg.target
    line_bucket = f"{leg.target}|{leg.side}|{leg.line}"
    state_bucket = f"{predictive_version}|{state_version}"
    return market_bucket, line_bucket, state_bucket


# Priority order for turning REQUIRED-dimension blocks into ONE specific
# abstain reason instead of an opaque catch-all. OBSERVE_ONLY dimensions
# (joint_support, shift_status) never appear here -- see calibration/
# support.py, they can never block by construction.
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
    calibration ledger for both legs of a candidate. `snapshot_rows`/
    `independent_slate_count` come from a SINGLE snapshot built once per
    week (see build_week_payload) -- not rebuilt per candidate."""
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
    """Best-available, really-priced candidate, ranked by
    joint_probability_estimate -- NOT `joint_score`
    (retained_probability_mass), which is uninformative while the frozen
    APS threshold retains every world (aps_threshold=1.0). DISPLAY-ONLY;
    never gates `action`/`selected_parlay`. Every candidate here is already
    cross-event/cross-player by construction, so (unlike MLB) there is no
    same-game filter to apply here -- just the pricing check."""
    priced = [c for c in candidates if _candidate_priced(c)]
    if not priced:
        return None
    return max(priced, key=lambda c: c.joint_probability_estimate)


def _to_candidate_wager(candidate: PairCandidate) -> CandidateWager:
    leg_i, leg_j = candidate.leg_1, candidate.leg_2
    world_id_i, world_id_j = "leg_1", "leg_2"
    clipped = np.clip([leg_i.model_probability_estimate, leg_j.model_probability_estimate], 1e-4, 1 - 1e-4)
    distribution = build_world_distribution([world_id_i, world_id_j], clipped)
    outcome_set = build_binary_outcome_set(distribution, aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=FROZEN_CALIBRATION_SLATES)

    decimal_price = None
    if leg_i.decimal_price is not None and leg_j.decimal_price is not None:
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


def build_week_payload(
    *,
    plays: list[dict],
    week_id: str,
    eligibility_inputs: EligibilityInputs,
    predictive_version: str,
    state_version: str,
    calibration_store: CalibrationStore | None = None,
    decision_record_store: DecisionRecordStore | None = None,
    world_gate_mode: str = "REQUIRED",
    world_risk_threshold: float | None = None,
) -> dict:
    """Enforces the required weekly ordering: eligibility -> capture
    immutable pregame candidate universe -> load calibration snapshot
    (PRIOR settled weeks only, cutoff = the calibration_as_of timestamp
    captured HERE, before any candidate work) -> build candidates ->
    compute support from that snapshot only -> V2 select_action_for_day
    -> freeze ACT/ABSTAIN. This week's own outcomes are never read (this
    function never touches settlement data at all), and cannot enter the
    calibration ledger until a SEPARATE, later admission step runs after
    settlement is final (see calibration/store.py's forward-only invariant
    -- enforced there, not here).

    world_gate_mode/world_risk_threshold default to REQUIRED/None for the
    function signature's own safety, but every real production/CI
    invocation (main(), below) defaults to manifest.WORLD_GATE_MODE
    (OBSERVE_ONLY) -- see manifest.py's WORLD-GATE CONFIGURATION comment."""
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
            "note": "STATUS reflects the FROZEN POLICY's evidence state, never this week's own candidates -- see manifest.CONCLUSION_REASONING.",
        },
        "calibration_as_of": calibration_as_of,
    }
    if not eligibility.eligible:
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "OPERATIONALLY_INELIGIBLE"
        return payload

    if not plays:
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "NO_REAL_QUOTE"
        return payload

    # Candidates are built unconditionally (not only when calibration
    # support might pass) so the Parlays tab always has a real, priced
    # shadow candidate to show on an abstain week -- see
    # _best_shadow_candidate. This is diagnostic-only and never gated on
    # calibration support.
    candidates = build_candidates_for_week(
        plays,
        week_id=week_id,
        aps_threshold=FROZEN_APS_THRESHOLD,
        calibration_slates=FROZEN_CALIBRATION_SLATES,
        predictive_version=predictive_version,
        state_version=state_version,
    )
    shadow_candidate = _best_shadow_candidate(candidates)
    if shadow_candidate is not None:
        payload["shadow_candidate"] = shadow_candidate.as_dict()

    # POLICY_NOT_FROZEN guard: real ACT selection only counts as
    # confirmatory prospective evidence once this policy_version's freeze
    # boundary has been deliberately set (prospective_boundary.py).
    if manifest.STATUS == "DEVELOPMENT":
        payload["decision_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        payload["abstain_reason"] = "POLICY_NOT_FROZEN"
        return payload

    calibration_snapshot = build_snapshot(calibration_store, as_of=calibration_as_of) if calibration_store is not None else None
    calibration_rows = calibration_store.observations_as_of(calibration_as_of) if calibration_store is not None else []
    independent_slate_count = calibration_snapshot.independent_slate_count if calibration_snapshot is not None else 0

    decision_frozen_at = datetime.now(timezone.utc).isoformat()
    payload["decision_timestamp_utc"] = decision_frozen_at
    if calibration_snapshot is not None:
        assert_snapshot_precedes_decision(calibration_snapshot, decision_frozen_at)
        payload["calibration_snapshot_id"] = calibration_snapshot.calibration_snapshot_id
        payload["calibration_snapshot_sha256"] = calibration_snapshot.calibration_snapshot_sha256
    # Exposed unconditionally (0 when calibration_store is None) so the
    # frontend can show real progress toward N_STATE instead of a bare
    # "not enough yet".
    payload["independent_slate_count"] = independent_slate_count
    payload["independent_slate_count_required"] = N_STATE

    if not candidates:
        payload["abstain_reason"] = "NO_CANDIDATES"
        return payload

    support_pairs = [
        _pair_support(c, snapshot_rows=calibration_rows, independent_slate_count=independent_slate_count, predictive_version=predictive_version, state_version=state_version)
        for c in candidates
    ]
    supported_candidates = [c for c, sp in zip(candidates, support_pairs) if _pair_in_support(sp)]
    if not supported_candidates:
        payload["abstain_reason"] = _aggregate_blocking_reason(support_pairs)
        return payload

    # Deterministic, quality-blind truncation.
    if len(supported_candidates) > manifest.MAX_CANDIDATES_PER_SLATE:
        supported_candidates = sorted(supported_candidates, key=lambda c: c.candidate_id)[: manifest.MAX_CANDIDATES_PER_SLATE]

    wagers = [_to_candidate_wager(c) for c in supported_candidates]
    by_wager_id = {c.candidate_id: c for c in supported_candidates}

    selection = select_action_for_day(
        eligibility, wagers, r_max=manifest.R_MAX_ACCEPTED,
        world_gate_mode=world_gate_mode, world_risk_threshold=world_risk_threshold,
    )
    decision_record = build_decision_record(
        date=week_id,
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
        "world_gate_mode": decision_record.world_gate_mode,
        "world_certificate_diagnostics": decision_record.world_certificate_diagnostics,
    }
    if decision_record_store is not None:
        decision_record_store.admit(decision_record)

    if selection.action == 0:
        payload["abstain_reason"] = "NO_PAIR_PASSES_FROZEN_POLICY"
        return payload

    selected_candidate = by_wager_id[selection.selected.wager_id]
    payload["action"] = "ACT"
    payload["selected_parlay"] = selected_candidate.as_dict()
    payload["selection_status"] = "SELECTED"
    payload["shadow_execution_status"] = "EXECUTED_SHADOW"
    payload["staking_authorized"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PARLAY_POLICY_V2 for one NFL week (Parlays tab source).")
    parser.add_argument("--plays-json", type=Path, required=True, help="Path to a JSON payload with a top-level 'plays' key (sports/nfl/web/data/daily_predictions.json or an archived sports/nfl/data/production/snapshots/.../*.json).")
    parser.add_argument("--week-id", type=str, required=True, help="e.g. 2026-W03")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--predictive-version", type=str, default="NFL_PASSING_LOSS_AWARE_META_POLICY_V2")
    parser.add_argument("--state-version", type=str, default="NFL_WEEKLY_BROAD_V1")
    parser.add_argument("--calibration-ledger", type=Path, default=None, help="Path to the forward-only calibration ledger JSONL (sports/nfl/parlay_v2/calibration/store.py). Omit to run with no calibration support (always abstains).")
    parser.add_argument("--decision-record-ledger", type=Path, default=None, help="Path to the durable per-week DecisionRecord ledger (decision_record_store.py). Omit to skip persistence (e.g. for local/manual runs).")
    parser.add_argument(
        "--world-gate-mode", choices=["REQUIRED", "BOUNDED_RISK", "OBSERVE_ONLY"], default=manifest.WORLD_GATE_MODE,
        help=(
            "How world/counterexample information participates in admission (policy.select_action_for_day). "
            f"Defaults to manifest.WORLD_GATE_MODE ({manifest.WORLD_GATE_MODE!r}) -- the CURRENTLY FROZEN policy's "
            "config, i.e. what a real production/CI run actually uses with no extra flags."
        ),
    )
    parser.add_argument(
        "--world-risk-threshold", type=float, default=manifest.WORLD_RISK_THRESHOLD,
        help="Required only for --world-gate-mode BOUNDED_RISK. Defaults to manifest.WORLD_RISK_THRESHOLD.",
    )
    args = parser.parse_args()

    plays_exist = args.plays_json.exists()
    payload_in = {}
    if plays_exist:
        with open(args.plays_json, encoding="utf-8") as f:
            payload_in = json.load(f)
    plays = build_week_action_plays(payload_in)
    eligibility_inputs = EligibilityInputs(
        date=args.week_id,
        required_feed_available=plays_exist,
        week_has_games=bool(plays_exist and len(plays) > 0),
        required_system_component_available=True,
        decision_cutoff_met=True,
    )
    calibration_store = CalibrationStore(args.calibration_ledger) if args.calibration_ledger else None
    decision_record_store = DecisionRecordStore(args.decision_record_ledger) if args.decision_record_ledger else None
    payload = build_week_payload(
        plays=plays,
        week_id=args.week_id,
        eligibility_inputs=eligibility_inputs,
        predictive_version=args.predictive_version,
        state_version=args.state_version,
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
