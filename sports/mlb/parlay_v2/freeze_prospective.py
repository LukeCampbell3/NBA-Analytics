from __future__ import annotations

"""Freeze-readiness / prospective-freeze command (mission section 13/28).

    python -m sports.mlb.parlay_v2.freeze_prospective --policy PARLAY_POLICY_V2_PROSPECTIVE_001 [--confirm]

Without --confirm: DRY RUN. Verifies readiness, writes a FREEZE_READY
artifact, and prints exactly what would be frozen. Never touches
prospective_start_timestamp.

With --confirm: verifies readiness (same checks), then calls
prospective_boundary.set_prospective_start_timestamp (one-way; refuses if
already set for this policy_version). Only run this once a human has
reviewed the readiness artifact and deliberately decided to start the
prospective stream -- see mission section 28: "Do not guess."
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sports.mlb.research.parlay_certification_v2 import manifest, prospective_boundary
from sports.mlb.research.parlay_certification_v2.anytime_monitor import REFERENCE_MONITOR_VERSION
from sports.mlb.research.parlay_certification_v2.evidence_store import EVIDENCE_STORE_VERSION
from sports.mlb.research.parlay_certification_v2.settlement import SETTLEMENT_VERSION
from sports.mlb.research.parlay_certification_v2.state_machine import STATE_MACHINE_VERSION
from sports.mlb.research.parlay_certification_v2.world_certificate import WORLD_CERTIFICATE_VERSION

from .calibration.versioning import CALIBRATION_VERSION, SUPPORT_RULE_VERSION
from .candidate_adapter import ADAPTER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
BOUNDARY_ROOT = REPO_ROOT / "sports" / "mlb" / "research" / "parlay_certification_v2" / "reports" / "prospective_boundary"
READINESS_ROOT = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "reports"

# Source paths considered "relevant" for the uncommitted-changes check --
# a dirty tree ANYWHERE under these means the frozen artifact would not
# reproducibly describe what's actually checked in.
RELEVANT_SOURCE_PATHS = (
    "sports/mlb/parlay_v2",
    "sports/mlb/research/parlay_certification_v2",
    "sports/mlb/research/joint_position_builder_v2",
)


def _git_dirty_relevant_paths() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", *RELEVANT_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _code_hash() -> str:
    hasher = hashlib.sha256()
    files: list[Path] = []
    for rel in RELEVANT_SOURCE_PATHS:
        files.extend(sorted((REPO_ROOT / rel).rglob("*.py")))
    for f in sorted(files):
        hasher.update(str(f.relative_to(REPO_ROOT)).encode())
        hasher.update(f.read_bytes())
    return hasher.hexdigest()


def _config_hash() -> str:
    config = {
        "eligibility_version": manifest.ELIGIBILITY_VERSION,
        "policy_version": manifest.POLICY_VERSION,
        "prospective_policy_id": manifest.PROSPECTIVE_POLICY_ID,
        "support_gate_modes": manifest.SUPPORT_GATE_MODES,
        "max_candidates_per_slate": manifest.MAX_CANDIDATES_PER_SLATE,
        "settlement_version": SETTLEMENT_VERSION,
        "world_certificate_version": WORLD_CERTIFICATE_VERSION,
        "evidence_store_version": EVIDENCE_STORE_VERSION,
        "reference_monitor_version": REFERENCE_MONITOR_VERSION,
        "state_machine_version": STATE_MACHINE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "support_rule_version": SUPPORT_RULE_VERSION,
        "candidate_adapter_version": ADAPTER_VERSION,
        "c": manifest.C_MIN_COVERAGE,
        "r": manifest.R_MAX_LOSS_RISK,
        "delta": manifest.DELTA_MIN_RETURN,
        "d_max": manifest.D_MAX,
        "r_max_accepted": manifest.R_MAX_ACCEPTED,
        "alpha_program": manifest.ALPHA_PROGRAM,
        "alpha_total": manifest.ALPHA_TOTAL,
        "alpha_c": manifest.ALPHA_C,
        "alpha_l": manifest.ALPHA_L,
        "alpha_v": manifest.ALPHA_V,
        "max_actions_per_eligible_slate": manifest.MAX_ACTIONS_PER_ELIGIBLE_SLATE,
        "two_leg_parlays_only": manifest.TWO_LEG_PARLAYS_ONLY,
    }
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest(), config


def build_readiness_artifact(policy_id: str) -> dict:
    dirty = _git_dirty_relevant_paths()
    code_hash = _code_hash()
    config_hash, config = _config_hash()
    now = datetime.now(timezone.utc).isoformat()

    # joint_support/shift_status are OBSERVE_ONLY (calibration/support.py's
    # GateMode) -- they are informational forever, until independently
    # researched and promoted to REQUIRED in a NEW policy version. Freeze
    # readiness does NOT wait on them: that was the exact circularity this
    # mission fixed (see support.py's module docstring). The three
    # REQUIRED dimensions -- market_support/line_support/state_support --
    # are real, implemented, and frozen with real thresholds (N_MARKET/
    # N_LINE/N_STATE) already; whether any given candidate currently PASSES
    # them is a per-slate ledger-content question, not a freeze-readiness
    # (code/config) question, so it is correctly not checked here either.
    observe_only_support_dimensions = ["joint_support", "shift_status"]

    readiness = {
        "policy_id": policy_id,
        "checked_at_utc": now,
        "code_hash": code_hash,
        "config_hash": config_hash,
        "frozen_config": config,
        "uncommitted_relevant_changes": dirty,
        "git_clean": len(dirty) == 0,
        "observe_only_support_dimensions": observe_only_support_dimensions,
        "required_support_dimensions": ["market_support", "line_support", "state_support"],
        "candidate_enumeration": "all cross-game 2-leg pairs from the pregame action-eligible universe (candidate_adapter.build_candidates_for_day)",
        "tie_breaker": "highest retained_probability_mass, then lexicographic wager_id (policy.select_action_for_day)",
        "max_actions_per_eligible_slate": manifest.MAX_ACTIONS_PER_ELIGIBLE_SLATE,
        "leg_count": 2,
        "supported_books": "real market_source rows only (no synthetic/fabricated prices, enforced at ingestion)",
    }
    # FREEZE_READY requires only a clean git tree -- the frozen code/config
    # this artifact hashes is what's being locked in, not today's ledger
    # content. This does NOT authorize production/staking (see manifest.py:
    # PRODUCTION_AUTHORIZED stays False unconditionally, independent of
    # freeze_ready) -- it only lets real prospective evidence begin
    # accumulating from real, non-circular selection.
    readiness["freeze_ready"] = readiness["git_clean"]
    readiness["freeze_ready_reason"] = "clean" if readiness["freeze_ready"] else "uncommitted relevant changes"
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze-readiness check / one-way prospective boundary activation for a PARLAY_POLICY_V2 stream.")
    parser.add_argument("--policy", required=True, help="Policy id, e.g. PARLAY_POLICY_V2_PROSPECTIVE_001")
    parser.add_argument("--confirm", action="store_true", help="Actually write the one-way prospective_start_timestamp boundary. Omit for a dry run.")
    args = parser.parse_args()

    readiness = build_readiness_artifact(args.policy)
    READINESS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = READINESS_ROOT / f"{args.policy}_freeze_readiness.json"
    with open(out_path, "w") as f:
        json.dump(readiness, f, indent=2, sort_keys=True)
    print(json.dumps(readiness, indent=2, sort_keys=True))

    if not args.confirm:
        print(f"\nDRY RUN ONLY -- wrote {out_path}. Boundary NOT touched. Re-run with --confirm to activate (only after human review).")
        return

    if not readiness["freeze_ready"]:
        raise SystemExit(f"Refusing to activate boundary: freeze_ready=False ({readiness['freeze_ready_reason']})")

    ok = prospective_boundary.set_prospective_start_timestamp(
        BOUNDARY_ROOT, args.policy, readiness["checked_at_utc"], note=f"config_hash={readiness['config_hash']} code_hash={readiness['code_hash']}"
    )
    if not ok:
        raise SystemExit(f"Boundary already set for {args.policy} -- one-way, refusing to overwrite.")
    print(f"Prospective boundary ACTIVATED for {args.policy} at {readiness['checked_at_utc']}")


if __name__ == "__main__":
    main()
