#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.evidence_ledger import append_generation
from sports.mlb.unified.pipeline import export_payload, run, write_payload
from sports.mlb.unified.policy_manifest import FROZEN_POLICY_COMMIT
from sports.mlb.unified.production_state import atomic_write_json


def _current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _runtime_manifest(manifest: dict[str, Any], runtime_policy_hash: str) -> dict[str, Any]:
    """Reconcile implementation-hash drift without granting new authority.

    The authoritative manifest can be older than plumbing fixes to adapters or
    data contracts. A stale hash must be visible, but it should not make a
    shadow-only compatibility artifact impossible to build. This function is
    intentionally fail-closed for authority: only a legacy-active, non-active
    production state may be reconciled automatically.
    """
    result = dict(manifest)
    prior_hash = str(result.get("policy_hash") or "")
    if prior_hash == runtime_policy_hash:
        result["governance_drift_detected"] = False
        return result

    if result.get("active_engine") != "legacy" or result.get("fallback_engine") != "legacy":
        raise ValueError("refusing runtime manifest reconciliation unless legacy remains active/fallback")
    state = str(result.get("production_state") or "")
    if state in {"PRODUCTION_ACTIVE", "PROSPECTIVE_CANARY", "PRODUCTION_CANDIDATE"}:
        raise ValueError(f"refusing runtime hash reconciliation in authority-bearing state {state}")
    if result.get("unified_policy_commit") != FROZEN_POLICY_COMMIT:
        raise ValueError("frozen unified policy commit changed unexpectedly")
    if result.get("certified_capabilities"):
        raise ValueError("refusing runtime hash reconciliation with certified capabilities")

    result["prior_policy_hash"] = prior_hash
    result["policy_hash"] = runtime_policy_hash
    result["governance_drift_detected"] = True
    result["governance_drift_reason"] = (
        "runtime compatibility/data-contract implementation differs from the older engine-manifest snapshot; "
        "decision authority remains legacy and validation-only"
    )
    result["source_engine_manifest_generated_at"] = manifest.get("generated_at")
    result["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    current = _current_commit()
    if current:
        result["implementation_commit"] = current
    # Explicitly preserve the locked authority posture.
    result["active_engine"] = "legacy"
    result["fallback_engine"] = "legacy"
    result["certified_capabilities"] = []
    return result


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
    source_manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    payload = export_payload(result, run_date=run_date, repo_root=REPO_ROOT, engine_state=source_manifest["production_state"])
    runtime_manifest = _runtime_manifest(source_manifest, str(payload["policy_hash"]))
    if payload["policy_hash"] != runtime_manifest["policy_hash"]:
        raise ValueError("runtime manifest reconciliation failed")
    write_payload(payload, args.output)
    atomic_write_json(args.manifest_output, runtime_manifest)
    append_generation(args.evidence_ledger, {
        "generation_id": payload["generation_id"],
        "generated_at_utc": payload["generated_at_utc"],
        "run_date": payload["run_date"],
        "policy_hash": payload["policy_hash"],
        "prior_policy_hash": runtime_manifest.get("prior_policy_hash"),
        "governance_drift_detected": runtime_manifest.get("governance_drift_detected", False),
        "engine_state": payload["engine_state"],
        "source_status": result.source_status,
        "candidate_count": len(result.candidates),
        "candidates": [candidate.to_dict() for candidate in result.candidates],
        "accepted_single_ids": [candidate.candidate_id for candidate in result.singles],
        "rejected": [{"candidate_id": candidate.candidate_id, "reasons": candidate.rejection_reasons} for candidate in result.rejected],
        "tickets": {str(count): [ticket.to_dict() for ticket in tickets] for count, (tickets, _) in result.tickets.items()},
        "settlement": None,
        "revision": 1,
    })
    print(
        f"unified shadow: candidates={len(result.candidates)} singles={len(result.singles)} "
        f"governance_drift={runtime_manifest.get('governance_drift_detected', False)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
