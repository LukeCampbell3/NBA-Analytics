#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.production_state import atomic_write_json
from sports.mlb.unified.slate_audit import audit_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--challenger", type=Path, default=REPO_ROOT / "artifacts/mlb_v2_daily_evidence.json")
    parser.add_argument("--evidence-ledger", type=Path, default=REPO_ROOT / "sports/mlb/data/predictions/unified/v2_1_evidence.jsonl")
    parser.add_argument("--fail-on-integrity", action="store_true")
    args = parser.parse_args()
    candidates = []
    if args.evidence_ledger.exists():
        for line in args.evidence_ledger.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("run_date") == args.run_date:
                candidates.extend(record.get("challenger_candidates") or record.get("candidates") or [])
    report = audit_candidates(candidates, run_date=args.run_date)
    atomic_write_json(REPO_ROOT / "artifacts/mlb_slate_integrity_audit.json", report)
    lines = ["# MLB slate integrity audit", "", f"Run date: `{args.run_date}`", "",
             f"Publication integrity: **{report['publication_integrity']}**", "", "## Issues", ""]
    lines += [f"- `{key}`: {value}" for key, value in report["issue_counts"].items()] or ["- None"]
    lines += ["", "## Canonical statuses", ""]
    lines += [f"- `{key}`: {value}" for key, value in report["candidate_status_counts"].items()] or ["- No candidates"]
    lines += ["", "Research-only, support-blocked, identity-invalid, stale, or non-positive-EV rows have no execution authority.", ""]
    (REPO_ROOT / "docs/mlb_slate_integrity_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("run_date", "publication_integrity", "fatal_issues")}, indent=2))
    return int(args.fail_on_integrity and report["publication_integrity"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
