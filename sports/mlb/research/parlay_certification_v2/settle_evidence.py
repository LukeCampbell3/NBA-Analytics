from __future__ import annotations

"""Settlement -> policy evidence ingestion (mission section 19/27) -- the
ONLY writer of FinalEvidenceRecord rows into EvidenceStore.

For one settled day whose DecisionRecord was already durably persisted
(decision_record_store.py, written once per day AT DECISION TIME by
run_parlay_v2.build_slate_payload -- never revised here), this
regenerates the frozen candidate universe deterministically
(candidate_adapter.build_day_action_universe +
candidate_adapter.build_candidates_for_day, the same pure function of
that day's frozen archived data parlay_v2/calibration/pair_ingest.py
already relies on for the identical reason) ONLY to look up the real
settled win/loss for the two legs of the ALREADY-DECIDED selected_wager.
This never re-runs policy.select_action_for_day and never second-guesses
which wager was actually chosen live -- selecting a DIFFERENT wager
post-hoc, even a "better" one, would be exactly the kind of prospective-
evidence contamination this program's version-isolation/forward-only
discipline exists to prevent.

Grading:
    action == 0 (the day was decided as an abstention) -> loss=0,
        realized_return=0.0 (no action was taken -- see settlement.py's
        own -1<=R<=r_max contract; an untaken action has no realized R).
    action == 1 -> resolve WIN/LOSS from both legs' real settled outcomes
        via settlement.SettlementInput/resolve_return/is_loss (UNTOUCHED
        certified machinery). Push/void is not modeled here -- the
        underlying multi_target_universe grades every action-eligible row
        strictly win/loss, exactly like calibration/ingest.py and
        pair_ingest.py's own grading; a real push/void would need its own
        SettlementStatus wiring and is out of scope for this ingestion
        path for the same reason it already is for those two.

Idempotent by source_id = f"{policy_version}|{date}" (EvidenceStore's own
idempotency key) -- one real-world settlement event per decided day,
re-running this script is always safe.

IMPORTANT identifier note: `--policy-version` here must be
manifest.POLICY_VERSION (the STRUCTURAL policy shape,
"PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION" -- what policy.build_decision_record
actually stamps onto every DecisionRecord, unchanged by this mission),
NOT manifest.PROSPECTIVE_POLICY_ID (this mission's new identifier for one
specific frozen gate-mode config, used only by prospective_boundary.py and
program_alpha.py). EvidenceStore's one-file-per-policy_version isolation
is keyed by the former -- that granularity is part of the existing,
untouched certified machinery, not something this mission changes.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from sports.mlb.parlay_v2.candidate_adapter import build_candidates_for_day, build_day_action_universe

from .decision_record_store import DecisionRecordStore
from .evidence_store import DecisionRecord, EvidenceStore, FinalEvidenceRecord
from .settlement import SettlementInput, SettlementStatus, is_loss, resolve_return

SETTLE_EVIDENCE_VERSION = "SETTLE_EVIDENCE_V1"

# APS threshold/calibration-slates for the POST-HOC candidate regeneration
# below -- frozen, matches run_parlay_v2.FROZEN_APS_THRESHOLD /
# FROZEN_CALIBRATION_SLATES. Neither actually affects candidate_id
# identity (candidate_id is built only from slate_id/player/target/side/
# line/game_id -- see candidate_adapter.build_candidates_for_day), so this
# regeneration reproduces the exact same wager_id space the live decision
# used regardless of these values; they are here only because the
# function signature requires them.
FROZEN_APS_THRESHOLD = 1.0
FROZEN_CALIBRATION_SLATES = 0
# state_version is likewise not part of candidate_id identity and is not
# persisted on DecisionRecord (a genuinely frozen, certified schema this
# module must not alter) -- this placeholder never affects matching.
_REGEN_STATE_VERSION = "SETTLEMENT_REGEN"


def _leg_key(leg) -> tuple:
    return (leg.player_id, leg.game_id, leg.target, leg.side, leg.line)


def _row_key(row) -> tuple:
    return (str(row["player_key"]), str(row["game_id"]), str(row["target"]), str(row["direction"]), float(row["market_line"]))


def settle_decided_day(
    date: str,
    *,
    decision_record_store: DecisionRecordStore,
    evidence_store: EvidenceStore,
    mode: str = "broad",
) -> dict:
    record_row = decision_record_store.record_for_date(date)
    if record_row is None:
        return {"date": date, "status": "no_decision_record"}
    decision_record = DecisionRecord(**record_row)

    if decision_record.policy_version != evidence_store.policy_version:
        return {"date": date, "status": "policy_version_mismatch"}

    source_id = f"{decision_record.policy_version}|{date}"
    if source_id in evidence_store.existing_source_ids():
        return {"date": date, "status": "already_settled"}

    now = datetime.now(timezone.utc).isoformat()

    if decision_record.action == 0 or decision_record.selected_wager is None:
        final = FinalEvidenceRecord(
            date=date, policy_version=decision_record.policy_version,
            eligible=int(decision_record.eligible), action=0, loss=0, realized_return=0.0,
            settlement_status="not_actioned", settlement_timestamp_utc=now,
            source_id=source_id, decision_record=decision_record,
        )
        evidence_store.append_final_settlement(final)
        return {"date": date, "status": "settled_abstain"}

    # action == 1: regenerate the frozen candidate universe post-hoc and
    # locate the EXACT wager already decided live -- never a different one.
    action_rows = build_day_action_universe((date,), date, mode=mode)
    if action_rows.empty:
        return {"date": date, "status": "not_yet_settled"}  # outcomes not graded upstream yet -- retry later, like ingest.py

    candidates = build_candidates_for_day(
        action_rows, slate_id=date,
        aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=FROZEN_CALIBRATION_SLATES,
        predictive_version=decision_record.predictive_model_version, state_version=_REGEN_STATE_VERSION,
    )
    matched = next((c for c in candidates if c.candidate_id == decision_record.selected_wager), None)
    if matched is None:
        return {"date": date, "status": "wager_not_found_yet"}  # candidate universe not fully regradeable yet -- retry later

    row_by_key = {_row_key(row): bool(row["win"]) for _, row in action_rows.iterrows()}
    win_1 = row_by_key.get(_leg_key(matched.leg_1))
    win_2 = row_by_key.get(_leg_key(matched.leg_2))
    if win_1 is None or win_2 is None:
        return {"date": date, "status": "leg_outcome_missing"}  # one leg not yet graded upstream -- retry later

    both_win = bool(win_1 and win_2)
    settlement_status = SettlementStatus.WIN if both_win else SettlementStatus.LOSS
    settlement_input = SettlementInput(status=settlement_status, accepted_decimal_price=decision_record.accepted_decimal_price)
    r = resolve_return(settlement_input, r_max=decision_record.r_max)
    loss = int(is_loss(r))

    final = FinalEvidenceRecord(
        date=date, policy_version=decision_record.policy_version,
        eligible=int(decision_record.eligible), action=1, loss=loss, realized_return=r,
        settlement_status=settlement_status.value, settlement_timestamp_utc=now,
        source_id=source_id, decision_record=decision_record,
    )
    evidence_store.append_final_settlement(final)
    return {"date": date, "status": "settled_act", "loss": loss, "realized_return": r}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade one settled MLB slate's already-frozen PARLAY_POLICY_V2 decision and admit its FinalEvidenceRecord.")
    parser.add_argument("--date", required=True, help="Settled slate date stamp, e.g. 20260820")
    parser.add_argument("--decision-record-ledger", type=Path, required=True)
    parser.add_argument("--evidence-store-root", type=Path, required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--mode", choices=["narrow", "broad"], default="broad")
    args = parser.parse_args()

    decision_record_store = DecisionRecordStore(args.decision_record_ledger)
    evidence_store = EvidenceStore(args.evidence_store_root, args.policy_version)
    result = settle_decided_day(args.date, decision_record_store=decision_record_store, evidence_store=evidence_store, mode=args.mode)
    print(result)


if __name__ == "__main__":
    main()
