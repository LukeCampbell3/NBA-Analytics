"""Reconciliation tests (spec section 36) -- run against already-built
player JSON artifacts. Never silently patches a discrepancy: every
check either passes, or is reported as a named failure in the
reconciliation report.

Because the drive/post routing-STATE vectors are UNAVAILABLE in this
data environment (see routing/states.py), the state-count reconciliation
checks the spec describes (sum(state pass counts) == eligible pass
count, state probabilities sum to 1) are checked against the ONE real
multinomial vector this pipeline actually produces: the recipient
network's real assist_share vector. The routing-state checks are still
run, and correctly report "N/A -- UNAVAILABLE" rather than a fabricated
pass.

    python -m sports.nba.analytics.advantage_routing.build.validate --season 2025-26
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .build_player import OUTPUT_ROOT

FLOAT_TOLERANCE = 1e-6


@dataclass
class ReconciliationCheck:
    name: str
    status: str  # "PASS" | "FAIL" | "N/A"
    detail: str


@dataclass
class ReconciliationReport:
    player_name: str
    checks: list[ReconciliationCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "all_passed": self.all_passed,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks],
        }


def reconcile_player_artifact(artifact: dict) -> ReconciliationReport:
    player_name = artifact.get("player", {}).get("name", "UNKNOWN")
    report = ReconciliationReport(player_name=player_name)

    recipients = artifact.get("recipients", {}).get("recipients", [])
    sample_size = artifact.get("recipients", {}).get("sample_size", 0)
    sum_recipient_assists = sum(r["assists"]["value"] or 0 for r in recipients)
    report.checks.append(ReconciliationCheck(
        name="recipient_assists_sum_equals_sample_size",
        status="PASS" if sum_recipient_assists == sample_size else "FAIL",
        detail=f"sum(recipient.assists)={sum_recipient_assists} vs sample_size={sample_size}",
    ))

    sum_assist_share = sum(r["assist_share"]["value"] or 0 for r in recipients)
    if sample_size > 0:
        status = "PASS" if abs(sum_assist_share - 1.0) < FLOAT_TOLERANCE else "FAIL"
        detail = f"sum(assist_share)={sum_assist_share:.9f} (expected 1.0 +/- {FLOAT_TOLERANCE})"
    else:
        status, detail = "N/A", "sample_size is 0 (no real sampled assists for this player) -- vector is trivially empty, not a reconciliation failure"
    report.checks.append(ReconciliationCheck(name="recipient_assist_share_sums_to_one", status=status, detail=detail))

    for recipient in recipients:
        n = recipient["assists"]["value"]
        zone_total = sum(recipient.get("zone_breakdown", {}).values())
        status = "PASS" if zone_total == n else "FAIL"
        report.checks.append(ReconciliationCheck(
            name=f"zone_breakdown_sums_to_assists[{recipient['recipient_label']}]",
            status=status, detail=f"sum(zone_breakdown)={zone_total} vs assists={n}",
        ))

    for mode, path in (("drive", ["drive"]), ("post_strict", ["post"]), ("post_interior", ["interior_hub"])):
        node = artifact
        for key in path:
            node = node.get(key, {})
        routing_vector = node.get("routing_vector", {})
        status_field = routing_vector.get("status")
        report.checks.append(ReconciliationCheck(
            name=f"routing_state_probabilities_sum_to_one[{mode}]",
            status="N/A" if status_field == "UNAVAILABLE" else "FAIL",
            detail=f"routing_vector.status={status_field!r} -- routing-state classification is honestly UNAVAILABLE in this data environment, not silently assumed normalized",
        ))

    shot_outcomes_ok = "expected_points_by_zone" in artifact.get("shot_outcomes", {})
    report.checks.append(ReconciliationCheck(
        name="shot_outcomes_present",
        status="PASS" if shot_outcomes_ok else "FAIL",
        detail="shot_outcomes.expected_points_by_zone key present" if shot_outcomes_ok else "missing expected_points_by_zone",
    ))

    return report


def validate_all(output_root: Path = OUTPUT_ROOT) -> list[ReconciliationReport]:
    reports = []
    for path in sorted(output_root.glob("*.json")):
        if path.name == "players.json":
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        reports.append(reconcile_player_artifact(artifact))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    reports = validate_all(args.output_root)
    any_failed = False
    for report in reports:
        print(f"=== {report.player_name} -- {'PASS' if report.all_passed else 'FAIL'} ===")
        for check in report.checks:
            print(f"  [{check.status}] {check.name}: {check.detail}")
        any_failed = any_failed or not report.all_passed
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
