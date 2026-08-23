from __future__ import annotations

"""Settlement -> policy evidence ingestion -- the ONLY writer of
FinalEvidenceRecord rows into EvidenceStore. Ported from
sports/mlb/research/parlay_certification_v2/settle_evidence.py, replacing
MLB's multi_target_universe-based regeneration with NFL's archived weekly
snapshot + settlement_source.grade_play.

For one settled week whose DecisionRecord was already durably persisted
(decision_record_store.py, written once per week AT DECISION TIME by
run_parlay_v2.build_week_payload -- never revised here), this regenerates
the frozen candidate universe deterministically
(candidate_adapter.build_week_action_plays + build_candidates_for_week,
the same pure function of that week's frozen archived snapshot
parlay_v2/calibration/pair_ingest.py already relies on for the identical
reason) ONLY to look up the real settled win/loss for the two legs of the
ALREADY-DECIDED selected_wager. This never re-runs
policy.select_action_for_day and never second-guesses which wager was
actually chosen live.

Grading:
    action == 0 (the week was decided as an abstention) -> loss=0,
        realized_return=0.0 (no action was taken).
    action == 1 -> resolve WIN/LOSS from both legs' real settled outcomes
        via settlement.SettlementInput/resolve_return/is_loss (UNTOUCHED
        certified machinery). Push/void is not modeled here, exactly like
        calibration/ingest.py and pair_ingest.py's own grading (see
        settlement_source.grade_play).

Idempotent by source_id = f"{policy_version}|{week_id}" (EvidenceStore's
own idempotency key) -- one real-world settlement event per decided week,
re-running this script is always safe.

IMPORTANT identifier note: `--policy-version` here must be
manifest.POLICY_VERSION (the STRUCTURAL policy shape,
"NFL_PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION"), NOT
manifest.PROSPECTIVE_POLICY_ID (this program's identifier for one specific
frozen gate-mode config, used only by prospective_boundary.py and
program_alpha.py). EvidenceStore's one-file-per-policy_version isolation
is keyed by the former.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sports.nfl.parlay_v2.calibration.settlement_source import grade_play, load_season_actuals
from sports.nfl.parlay_v2.candidate_adapter import build_candidates_for_week, build_week_action_plays

from .decision_record_store import DecisionRecordStore
from .evidence_store import DecisionRecord, EvidenceStore, FinalEvidenceRecord
from .settlement import SettlementInput, SettlementStatus, is_loss, resolve_return

SETTLE_EVIDENCE_VERSION = "NFL_SETTLE_EVIDENCE_V1"

# APS threshold/calibration-slates for the POST-HOC candidate regeneration
# below -- frozen, matches run_parlay_v2.FROZEN_APS_THRESHOLD /
# FROZEN_CALIBRATION_SLATES. Neither actually affects candidate_id
# identity (candidate_id is built only from week_id/player/target/side/
# line/event_id -- see candidate_adapter.build_candidates_for_week), so
# this regeneration reproduces the exact same wager_id space the live
# decision used regardless of these values; they are here only because
# the function signature requires them.
FROZEN_APS_THRESHOLD = 1.0
FROZEN_CALIBRATION_SLATES = 0
_REGEN_STATE_VERSION = "SETTLEMENT_REGEN"


def _leg_key(leg) -> tuple:
    return (leg.player_id, leg.event_id, leg.target, leg.side, leg.line)


def settle_decided_week(
    week_id: str,
    *,
    snapshot_path: Path,
    season: int,
    week: int,
    decision_record_store: DecisionRecordStore,
    evidence_store: EvidenceStore,
    actuals_cache_path: Path | None = None,
) -> dict:
    record_row = decision_record_store.record_for_date(week_id)
    if record_row is None:
        return {"week_id": week_id, "status": "no_decision_record"}
    decision_record = DecisionRecord(**record_row)

    if decision_record.policy_version != evidence_store.policy_version:
        return {"week_id": week_id, "status": "policy_version_mismatch"}

    source_id = f"{decision_record.policy_version}|{week_id}"
    if source_id in evidence_store.existing_source_ids():
        return {"week_id": week_id, "status": "already_settled"}

    now = datetime.now(timezone.utc).isoformat()

    if decision_record.action == 0 or decision_record.selected_wager is None:
        final = FinalEvidenceRecord(
            date=week_id, policy_version=decision_record.policy_version,
            eligible=int(decision_record.eligible), action=0, loss=0, realized_return=0.0,
            settlement_status="not_actioned", settlement_timestamp_utc=now,
            source_id=source_id, decision_record=decision_record,
        )
        evidence_store.append_final_settlement(final)
        return {"week_id": week_id, "status": "settled_abstain"}

    # action == 1: regenerate the frozen candidate universe post-hoc and
    # locate the EXACT wager already decided live -- never a different one.
    if not Path(snapshot_path).is_file():
        return {"week_id": week_id, "status": "not_yet_settled"}
    with open(snapshot_path, encoding="utf-8") as f:
        payload = json.load(f)
    plays = build_week_action_plays(payload)
    if not plays:
        return {"week_id": week_id, "status": "not_yet_settled"}

    candidates = build_candidates_for_week(
        plays, week_id=week_id,
        aps_threshold=FROZEN_APS_THRESHOLD, calibration_slates=FROZEN_CALIBRATION_SLATES,
        predictive_version=decision_record.predictive_model_version, state_version=_REGEN_STATE_VERSION,
    )
    matched = next((c for c in candidates if c.candidate_id == decision_record.selected_wager), None)
    if matched is None:
        return {"week_id": week_id, "status": "wager_not_found_yet"}

    actuals = load_season_actuals(season, cache_path=actuals_cache_path)
    play_by_key = {
        (str(p.get("player_id")), str(p.get("event_id")), str(p.get("target") or p.get("market")), str(p.get("direction")), float(p.get("line"))): p
        for p in plays
    }
    win_1 = grade_play(play_by_key.get(_leg_key(matched.leg_1), {}), actuals, season=season, week=week) if _leg_key(matched.leg_1) in play_by_key else None
    win_2 = grade_play(play_by_key.get(_leg_key(matched.leg_2), {}), actuals, season=season, week=week) if _leg_key(matched.leg_2) in play_by_key else None
    if win_1 is None or win_2 is None:
        return {"week_id": week_id, "status": "leg_outcome_missing"}  # one leg not yet graded upstream -- retry later

    both_win = bool(win_1 and win_2)
    settlement_status = SettlementStatus.WIN if both_win else SettlementStatus.LOSS
    settlement_input = SettlementInput(status=settlement_status, accepted_decimal_price=decision_record.accepted_decimal_price)
    r = resolve_return(settlement_input, r_max=decision_record.r_max)
    loss = int(is_loss(r))

    final = FinalEvidenceRecord(
        date=week_id, policy_version=decision_record.policy_version,
        eligible=int(decision_record.eligible), action=1, loss=loss, realized_return=r,
        settlement_status=settlement_status.value, settlement_timestamp_utc=now,
        source_id=source_id, decision_record=decision_record,
    )
    evidence_store.append_final_settlement(final)
    return {"week_id": week_id, "status": "settled_act", "loss": loss, "realized_return": r}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade one settled NFL week's already-frozen PARLAY_POLICY_V2 decision and admit its FinalEvidenceRecord.")
    parser.add_argument("--week-id", required=True, help="Settled week identifier, e.g. 2026-W03 -- must match the week_id used at decision time.")
    parser.add_argument("--snapshot", type=Path, required=True, help="Path to the archived weekly plays snapshot JSON for this week.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--decision-record-ledger", type=Path, required=True)
    parser.add_argument("--evidence-store-root", type=Path, required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--actuals-cache", type=Path, default=None)
    args = parser.parse_args()

    decision_record_store = DecisionRecordStore(args.decision_record_ledger)
    evidence_store = EvidenceStore(args.evidence_store_root, args.policy_version)
    result = settle_decided_week(
        args.week_id, snapshot_path=args.snapshot, season=args.season, week=args.week,
        decision_record_store=decision_record_store, evidence_store=evidence_store,
        actuals_cache_path=args.actuals_cache,
    )
    print(result)


if __name__ == "__main__":
    main()
