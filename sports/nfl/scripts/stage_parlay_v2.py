"""Archives this week's published plays for PARLAY_V2 calibration
history, runs the frozen PARLAY_POLICY_V2 decision, and embeds the result
into the already-published daily_predictions.json payload.

ADDITIVE ONLY -- mirrors how sports/site/pipeline/run_daily_predictions.py
wires sports/mlb/parlay_v2 in alongside (never in place of) MLB's old
parlay path. This script never touches sports/nfl/predictions/daily_policy.py
or its "daily_parlay" key; frontend_payload.embed_parlays_v2 only ever adds
the new "parlays" top-level key.

Run AFTER "Build current NFL slate" has already written
sports/nfl/web/data/daily_predictions.json, and BEFORE
export_nfl_validation_web.py / build_static_site.py so the "parlays" key
is present when the static site is built.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from sports.nfl.parlay_v2 import frontend_payload, run_parlay_v2
from sports.nfl.parlay_v2.calibration.store import CalibrationStore
from sports.nfl.parlay_v2.candidate_adapter import build_week_action_plays
from sports.nfl.research.parlay_certification_v2 import manifest
from sports.nfl.research.parlay_certification_v2.decision_record_store import DecisionRecordStore
from sports.nfl.research.parlay_certification_v2.eligibility import EligibilityInputs

NFL_ROOT = Path(__file__).resolve().parents[1]
WEEKLY_PLAYS_ROOT = NFL_ROOT / "parlay_v2" / "reports" / "weekly_plays"
CALIBRATION_LEDGER = NFL_ROOT / "parlay_v2" / "calibration" / "reports" / "calibration_ledger.jsonl"
DECISION_RECORD_LEDGER = NFL_ROOT / "research" / "parlay_certification_v2" / "reports" / "decision_record_ledger.jsonl"
PARLAY_V2_OUT = NFL_ROOT / "parlay_v2" / "reports" / "latest_parlay_v2.json"
PREDICTIVE_VERSION = "NFL_PASSING_LOSS_AWARE_META_POLICY_V2"
STATE_VERSION = "NFL_WEEKLY_BROAD_V1"


def iso_week_id(run_date: date) -> str:
    """A stable, sortable weekly partition key derived from the ISO
    calendar week of run_date -- NOT the NFL's own season/week numbering
    (which requires a real schedule lookup; see settlement_source.py /
    calibration/ingest.py's --season/--week flags, supplied separately at
    settlement time). Used only as this week's candidate/ledger identity
    (week_id), matching MLB's own use of a date stamp as slate_id."""
    year, week, _ = run_date.isocalendar()
    return f"{year}-W{week:02d}"


def run(run_date: str, *, published_payload_path: Path) -> dict:
    run_day = date.fromisoformat(run_date)
    week_id = iso_week_id(run_day)

    if not published_payload_path.is_file():
        return {"status": "no_published_payload", "week_id": week_id}
    with open(published_payload_path, encoding="utf-8") as f:
        payload = json.load(f)
    plays = build_week_action_plays(payload)

    if plays:
        WEEKLY_PLAYS_ROOT.mkdir(parents=True, exist_ok=True)
        archive_path = WEEKLY_PLAYS_ROOT / f"{week_id}.json"
        with open(archive_path, "w") as f:
            json.dump({"week_id": week_id, "run_date": run_date, "plays": plays}, f, indent=2, sort_keys=True)

    eligibility_inputs = EligibilityInputs(
        date=week_id, required_feed_available=True, week_has_games=bool(plays),
        required_system_component_available=True, decision_cutoff_met=True,
    )
    calibration_store = CalibrationStore(CALIBRATION_LEDGER)
    decision_record_store = DecisionRecordStore(DECISION_RECORD_LEDGER)
    parlay_payload = run_parlay_v2.build_week_payload(
        plays=plays, week_id=week_id, eligibility_inputs=eligibility_inputs,
        predictive_version=PREDICTIVE_VERSION, state_version=STATE_VERSION,
        calibration_store=calibration_store, decision_record_store=decision_record_store,
        world_gate_mode=manifest.WORLD_GATE_MODE, world_risk_threshold=manifest.WORLD_RISK_THRESHOLD,
    )
    PARLAY_V2_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PARLAY_V2_OUT, "w") as f:
        json.dump(parlay_payload, f, indent=2, sort_keys=True, default=str)

    updated = frontend_payload.embed_parlays_v2(payload, PARLAY_V2_OUT)
    with open(published_payload_path, "w") as f:
        json.dump(updated, f, indent=2, sort_keys=True)
    return {"status": "staged", "week_id": week_id, "action": parlay_payload["action"], "abstain_reason": parlay_payload.get("abstain_reason")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--published-payload", type=Path, default=NFL_ROOT / "web" / "data" / "daily_predictions.json")
    args = parser.parse_args()
    result = run(args.run_date, published_payload_path=args.published_payload)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
