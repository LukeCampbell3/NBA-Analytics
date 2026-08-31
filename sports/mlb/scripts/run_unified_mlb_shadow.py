#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.evidence_ledger import append_generation
from sports.mlb.unified.pipeline import export_payload, run, write_payload
from sports.mlb.unified.production_state import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed unified MLB shadow artifact")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "sports/mlb/web/data")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "sports/mlb/web/data/unified_predictions.json")
    parser.add_argument("--run-date")
    parser.add_argument("--engine-manifest", type=Path, default=REPO_ROOT / "artifacts/mlb_engine_manifest.json")
    parser.add_argument("--manifest-output", type=Path, default=REPO_ROOT / "sports/mlb/web/data/mlb_engine_manifest.json")
    parser.add_argument("--evidence-ledger", type=Path, default=REPO_ROOT / "sports/mlb/data/predictions/unified/evidence.jsonl")
    args = parser.parse_args()
    run_date = args.run_date
    if not run_date:
        daily_path = args.data_dir / "daily_predictions.json"
        if daily_path.exists():
            run_date = json.loads(daily_path.read_text(encoding="utf-8")).get("run_date")
    result = run(args.data_dir)
    manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    payload = export_payload(result, run_date=run_date, repo_root=REPO_ROOT, engine_state=manifest["production_state"])
    if payload["policy_hash"] != manifest["policy_hash"]:
        raise ValueError("artifact policy hash does not match engine manifest")
    write_payload(payload, args.output)
    atomic_write_json(args.manifest_output, manifest)
    append_generation(args.evidence_ledger, {
        "generation_id": payload["generation_id"],
        "generated_at_utc": payload["generated_at_utc"],
        "run_date": payload["run_date"],
        "policy_hash": payload["policy_hash"],
        "engine_state": payload["engine_state"],
        "source_status": result.source_status,
        "candidate_count": len(result.candidates),
        "accepted_single_ids": [candidate.candidate_id for candidate in result.singles],
        "rejected": [{"candidate_id": candidate.candidate_id, "reasons": candidate.rejection_reasons} for candidate in result.rejected],
        "tickets": {str(count): [ticket.ticket_id for ticket in tickets] for count, (tickets, _) in result.tickets.items()},
        "settlement": None,
        "revision": 1,
    })
    print(f"unified shadow: candidates={len(result.candidates)} singles={len(result.singles)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
