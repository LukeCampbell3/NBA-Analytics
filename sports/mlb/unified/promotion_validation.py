from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import adapt_legacy_play
from .decision import DecisionPolicy, decide
from .market_registry import CAPABILITIES
from .policy_manifest import FROZEN_POLICY_COMMIT, build_policy_manifest
from .production_state import CapabilityAuthority, EngineState, atomic_write_json, build_engine_manifest


@dataclass(frozen=True)
class SnapshotCandidate:
    commit: str
    commit_time: str
    run_date: str
    generated_at: str
    play: dict[str, Any]
    fidelity: str
    reason: str


REQUIRED_FROZEN_FIELDS = {
    "game_id", "player", "target", "direction", "market_line",
    "final_hit_probability", "selected_side_price", "lineup_status",
    "historical_bucket_support", "commence_time_utc",
}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL)


def committed_daily_snapshots(repo_root: Path) -> list[SnapshotCandidate]:
    path = "sports/mlb/web/data/daily_predictions.json"
    commits = _git(repo_root, "log", "--all", "--format=%H|%cI", "--", path).splitlines()
    candidates: list[SnapshotCandidate] = []
    for item in commits:
        commit, commit_time = item.split("|", 1)
        try:
            payload = json.loads(_git(repo_root, "show", f"{commit}:{path}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        run_date = str(payload.get("run_date") or "")
        generated_at = str(payload.get("generated_at_utc") or "")
        for play in payload.get("plays", []):
            missing = sorted(field for field in REQUIRED_FROZEN_FIELDS if play.get(field) is None)
            generated = _parse_time(generated_at)
            start = _parse_time(play.get("commence_time_utc"))
            if missing:
                fidelity, reason = "RECONSTRUCTED_WEAK", f"MISSING_FROZEN_FIELDS:{','.join(missing)}"
            elif generated is None or start is None:
                fidelity, reason = "RECONSTRUCTED_WEAK", "PREDICTION_OR_START_TIMESTAMP_UNVERIFIABLE"
            elif generated >= start:
                fidelity, reason = "UNUSABLE", "SNAPSHOT_NOT_PREGAME"
            else:
                fidelity, reason = "EXACT", "COMMITTED_PREGAME_FROZEN_INPUTS"
            candidates.append(SnapshotCandidate(commit, commit_time, run_date, generated_at, play, fidelity, reason))
    # The earliest exact committed version of a semantic decision is the
    # immutable prediction-time observation. Later regenerations cannot add
    # independent evidence.
    deduped: dict[tuple, SnapshotCandidate] = {}
    for row in sorted(candidates, key=lambda value: value.generated_at):
        play = row.play
        key = (row.run_date, play.get("game_id"), play.get("player_id") or play.get("player"), play.get("target"), play.get("direction"), play.get("market_line"))
        prior = deduped.get(key)
        if prior is None or (prior.fidelity != "EXACT" and row.fidelity == "EXACT"):
            deduped[key] = row
    return list(deduped.values())


def historical_inventory(repo_root: Path) -> dict[str, Any]:
    universe_path = repo_root / "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv"
    frame = pd.read_csv(universe_path, low_memory=False)
    fields = {}
    for column in frame.columns:
        present = int(frame[column].notna().sum())
        fields[column] = {"present": present, "total": int(len(frame)), "fraction": present / len(frame) if len(frame) else 0.0}
    snapshot_rows = committed_daily_snapshots(repo_root)
    fidelity = Counter(row.fidelity for row in snapshot_rows)
    settlement_present = sum(bool(row.play.get("settlement") or row.play.get("result")) for row in snapshot_rows if row.fidelity == "EXACT")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": {
            str(universe_path.relative_to(repo_root)): {
                "rows": int(len(frame)),
                "independent_slates": int(frame["Prediction_Run_Date"].nunique()),
                "date_min": str(frame["Prediction_Run_Date"].min()),
                "date_max": str(frame["Prediction_Run_Date"].max()),
                "field_availability": fields,
                "fidelity": "RECONSTRUCTED_WEAK",
                "reason": "Settled predictions and sparse quote timestamps exist, but frozen final/usable probability, lineup, role, calibration and uncertainty state do not.",
            },
            "git_history:sports/mlb/web/data/daily_predictions.json": {
                "deduplicated_candidates": len(snapshot_rows),
                "fidelity_counts": dict(fidelity),
                "exact_settled_candidates": settlement_present,
            },
        },
        "frozen_replay_requirements": sorted(REQUIRED_FROZEN_FIELDS | {"settlement"}),
        "certification_eligible_rows": settlement_present,
    }


def build_corpus(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for snapshot in committed_daily_snapshots(repo_root):
        play = snapshot.play
        settlement = play.get("settlement") or play.get("result")
        record = {
            "source_commit": snapshot.commit,
            "source_commit_time": snapshot.commit_time,
            "event_date": snapshot.run_date,
            "generation_timestamp": snapshot.generated_at,
            "game_id": play.get("game_id"),
            "player_id": play.get("player_id") or play.get("player"),
            "market": play.get("target"),
            "line": play.get("market_line"),
            "book": play.get("selected_sportsbook_key") or play.get("sportsbook"),
            "quoted_odds": play.get("selected_side_price"),
            "raw_probability": play.get("estimated_hit_probability"),
            "calibrated_probability": play.get("final_hit_probability"),
            "settlement": settlement,
            "fidelity": snapshot.fidelity,
            "fidelity_reason": snapshot.reason,
        }
        if snapshot.fidelity != "EXACT":
            record["exclusion_reason"] = snapshot.reason
            exclusions.append(record)
            continue
        if not settlement:
            record["exclusion_reason"] = "SETTLEMENT_UNAVAILABLE_IN_PRESERVED_ARTIFACT"
            exclusions.append(record)
            continue
        candidate = decide(adapt_legacy_play(play), DecisionPolicy())
        record.update({
            "usable_probability": candidate.usable_probability,
            "uncertainty": candidate.uncertainty,
            "probability_edge": candidate.probability_edge,
            "conservative_expected_value": candidate.conservative_expected_value,
            "eligible": not candidate.rejection_reasons,
            "rejection_reasons": candidate.rejection_reasons,
        })
        eligible.append(record)
    return eligible, exclusions


def capability_states() -> dict[str, str]:
    states = {}
    for name, capability in CAPABILITIES.items():
        if capability.status.value in {"MODEL_REQUIRED", "EVENT_MODEL_REQUIRED", "EVENT_IDENTITY_UNAVAILABLE", "BLOCKED", "DATA_REQUIRED"}:
            states[name] = CapabilityAuthority.BLOCKED.value
        else:
            states[name] = CapabilityAuthority.VALIDATION_ONLY.value
    states.update({"parlay_2_leg": "VALIDATION_ONLY", "parlay_3_leg": "VALIDATION_ONLY", "parlay_4_leg": "VALIDATION_ONLY", "same_game_parlays": "SHADOW"})
    return states


def certification(repo_root: Path, promotion_policy: dict[str, Any], eligible: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    slates = len({row["event_date"] for row in eligible})
    selections = [row for row in eligible if row.get("eligible")]
    required = promotion_policy["sample_sufficiency"]
    failures = []
    if slates < required["minimum_independent_slates"]:
        failures.append(f"INDEPENDENT_SLATES:{slates}<{required['minimum_independent_slates']}")
    if len(selections) < required["minimum_selected_singles_per_capability"]:
        failures.append(f"SELECTED_SINGLES:{len(selections)}<{required['minimum_selected_singles_per_capability']}")
    failures.append("FROZEN_USABLE_PROBABILITY_HISTORY_UNAVAILABLE" if not eligible else "CAPABILITY_LEVEL_SAMPLE_REQUIREMENTS_NOT_MET")
    return {
        "schema_version": 1,
        "status": "HISTORICAL_VALIDATION_FAIL" if failures else "HISTORICAL_VALIDATION_PASS",
        "evidence_state": "LOCKED_HISTORICAL_VALIDATION",
        "policy_commit": FROZEN_POLICY_COMMIT,
        "policy_hash": build_policy_manifest(repo_root)["policy_hash"],
        "eligible_slates": slates,
        "eligible_candidates": len(eligible),
        "selected_singles": len(selections),
        "excluded_candidates": len(exclusions),
        "metrics": {
            "hit_rate": None, "roi": None, "brier": None, "log_loss": None,
            "ece": None, "calibration_intercept": None, "calibration_slope": None,
            "max_drawdown_units": None,
        },
        "capabilities": capability_states(),
        "failures": sorted(set(failures)),
        "thresholds_modified_after_lock": False,
    }


def run_promotion_validation(repo_root: Path) -> dict[str, Any]:
    artifacts = repo_root / "artifacts"
    docs = repo_root / "docs"
    artifacts.mkdir(exist_ok=True)
    docs.mkdir(exist_ok=True)
    policy = json.loads((repo_root / "config/mlb_unified_promotion_policy.json").read_text())
    policy_manifest = build_policy_manifest(repo_root)
    inventory = historical_inventory(repo_root)
    eligible, exclusions = build_corpus(repo_root)
    result = certification(repo_root, policy, eligible, exclusions)
    atomic_write_json(artifacts / "mlb_unified_policy_manifest.json", policy_manifest)
    atomic_write_json(artifacts / "mlb_unified_historical_inventory.json", inventory)
    atomic_write_json(artifacts / "mlb_unified_historical_exclusions.json", {"rows": exclusions})
    corpus_path = artifacts / "mlb_unified_historical_corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in eligible), encoding="utf-8")
    atomic_write_json(artifacts / "mlb_unified_historical_certification.json", result)
    implementation = _git(repo_root, "rev-parse", "HEAD").strip()
    engine_manifest = build_engine_manifest(
        policy_hash=policy_manifest["policy_hash"], implementation_commit=implementation,
        state=EngineState.LOCKED_HISTORICAL_VALIDATION_FAILED,
        capabilities=result["capabilities"], rollback_reference=None,
    )
    atomic_write_json(artifacts / "mlb_engine_manifest.json", engine_manifest)
    production_status = {
        "engine": "unified_mlb",
        "implementation_commit": implementation,
        "policy_commit": FROZEN_POLICY_COMMIT,
        "policy_hash": policy_manifest["policy_hash"],
        "state": "UNIFIED_MLB_SHADOW_ONLY",
        "active_engine": "legacy",
        "historical_validation": {
            "status": result["status"], "eligible_slates": result["eligible_slates"],
            "selected_singles": result["selected_singles"], "roi": None,
            "hit_rate": None, "max_drawdown_units": None,
        },
        "live_canary": {
            "status": "NOT_AUTHORIZED_AFTER_HISTORICAL_GATE_FAILURE",
            "generation_id": None, "artifact_valid": False,
            "frontend_valid": False, "distribution_valid": False,
        },
        "capabilities": result["capabilities"],
        "production_authorized": False,
        "rollback_available": True,
        "rollback_reference": "static-deployment@c42bf2c1579d140b72efc5597fb9d074834ddfb4",
    }
    atomic_write_json(artifacts / "mlb_unified_production_status.json", production_status)
    return {"inventory": inventory, "certification": result, "production_status": production_status}
