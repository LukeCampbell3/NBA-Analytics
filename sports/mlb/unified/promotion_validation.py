from __future__ import annotations

import json
import math
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
    universe_frame = pd.read_csv(universe_path, low_memory=False)
    fields = {}
    for column in universe_frame.columns:
        present = int(universe_frame[column].notna().sum())
        fields[column] = {"present": present, "total": int(len(universe_frame)), "fraction": present / len(universe_frame) if len(universe_frame) else 0.0}
    snapshot_rows = committed_daily_snapshots(repo_root)
    fidelity = Counter(row.fidelity for row in snapshot_rows)
    settlement_present = sum(bool(row.play.get("settlement") or row.play.get("result")) for row in snapshot_rows if row.fidelity == "EXACT")
    daily_root = repo_root / "sports/mlb/data/predictions/daily_runs"
    archived_rows = archived_priced = archived_final_probability = 0
    archived_slate_dates: set[str] = set()
    for path in daily_root.glob("*/daily_prediction_pool_*_high_precision_predictions.csv"):
        archived = pd.read_csv(path, low_memory=False)
        if archived.empty:
            continue
        archived_rows += len(archived)
        if "Prediction_Run_Date" in archived:
            archived_slate_dates.update(str(value) for value in archived["Prediction_Run_Date"].dropna().unique())
        if "Selected_Side_Price" in archived:
            archived_priced += int(archived["Selected_Side_Price"].notna().sum())
        if "Final_Hit_Probability" in archived:
            archived_final_probability += int(archived["Final_Hit_Probability"].notna().sum())
    governance_slates: set[str] = set()
    governance_rows = governance_settled = governance_lineup_confirmed = 0
    for path in daily_root.glob("*/governance/slates/*/*/candidate_universe.csv.gz"):
        governance_frame = pd.read_csv(path, low_memory=False)
        if governance_frame.empty:
            continue
        governance_rows += len(governance_frame)
        governance_slates.update(str(value) for value in governance_frame.get("slate_id", []))
        if "settlement" in governance_frame:
            governance_settled += int((~governance_frame["settlement"].astype(str).str.upper().isin({"PENDING", "NAN", "NONE", ""})).sum())
        if "lineup_state" in governance_frame:
            governance_lineup_confirmed += int(governance_frame["lineup_state"].astype(str).str.upper().eq("CONFIRMED").sum())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": {
            str(universe_path.relative_to(repo_root)): {
                "rows": int(len(universe_frame)),
                "independent_slates": int(universe_frame["Prediction_Run_Date"].nunique()),
                "date_min": str(universe_frame["Prediction_Run_Date"].min()),
                "date_max": str(universe_frame["Prediction_Run_Date"].max()),
                "field_availability": fields,
                "fidelity": "RECONSTRUCTED_WEAK",
                "reason": "Settled predictions and sparse quote timestamps exist, but frozen final/usable probability, lineup, role, calibration and uncertainty state do not.",
            },
            "git_history:sports/mlb/web/data/daily_predictions.json": {
                "deduplicated_candidates": len(snapshot_rows),
                "fidelity_counts": dict(fidelity),
                "exact_settled_candidates": settlement_present,
            },
            "daily_runs:high_precision_prediction_artifacts": {
                "rows": archived_rows, "independent_slates": len(archived_slate_dates),
                "priced_rows": archived_priced, "rows_with_frozen_final_probability": archived_final_probability,
                "fidelity": "RECONSTRUCTED_WEAK",
                "reason": "Selected-only artifacts do not preserve the full rejected control pool, prediction-time lineup state, and frozen final probability on enough rows.",
            },
            "daily_runs:immutable_governance_candidate_universes": {
                "rows": governance_rows, "independent_slates": len(governance_slates),
                "settled_rows": governance_settled, "lineup_confirmed_rows": governance_lineup_confirmed,
                "fidelity": "PARTIAL",
                "reason": "Full timestamped candidate universes exist, but only across eight slates and their committed rows remain unsettled with lineup state unavailable.",
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


TARGET_CAPABILITY = {
    "H": "batter_hits", "TB": "batter_total_bases", "R": "batter_runs_scored",
    "RBI": "batter_rbis", "HR": "batter_home_runs", "K": "pitcher_strikeouts",
    "pitcher_outs": "pitcher_outs", "moneyline": "moneyline", "game_total": "game_total",
    "first_5_innings_total": "first_5_innings_total", "team_total": "team_total",
}


def _win(row: dict[str, Any]) -> int | None:
    value = str(row.get("settlement") or "").lower()
    if value in {"win", "won"}:
        return 1
    if value in {"loss", "lost"}:
        return 0
    return None


def _decimal_price(american: Any) -> float | None:
    try:
        value = float(american)
    except (TypeError, ValueError):
        return None
    return 1.0 + (100.0 / abs(value) if value < 0 else value / 100.0)


def _realized_return(row: dict[str, Any]) -> float | None:
    outcome = _win(row)
    decimal = _decimal_price(row.get("quoted_odds"))
    if outcome is None or decimal is None:
        return None
    return decimal - 1.0 if outcome == 1 else -1.0


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if _win(row) is not None]
    if not graded:
        return {"bets": 0, "hit_rate": None, "roi": None, "brier": None, "log_loss": None,
                "ece": None, "max_drawdown_units": None, "slate_clustered_roi_lcb": None}
    probabilities = [float(row["usable_probability"]) for row in graded]
    outcomes = [_win(row) for row in graded]
    returns: list[float] = []
    market_probabilities: list[float] = []
    equity = peak = drawdown = 0.0
    slate_returns: dict[str, list[float]] = {}
    for row, outcome in zip(graded, outcomes):
        decimal = _decimal_price(row.get("quoted_odds"))
        if decimal is None:
            continue
        market_probabilities.append(1.0 / decimal)
        realized = decimal - 1.0 if outcome == 1 else -1.0
        returns.append(realized)
        equity += realized
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
        slate_returns.setdefault(str(row["event_date"]), []).append(realized)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(graded)
    log_loss = -sum(y * math.log(max(p, 1e-12)) + (1-y) * math.log(max(1-p, 1e-12)) for p, y in zip(probabilities, outcomes)) / len(graded)
    market_brier = sum((p - y) ** 2 for p, y in zip(market_probabilities, outcomes)) / len(graded)
    market_log_loss = -sum(y * math.log(max(p, 1e-12)) + (1-y) * math.log(max(1-p, 1e-12)) for p, y in zip(market_probabilities, outcomes)) / len(graded)
    bins = [[] for _ in range(10)]
    for p, y in zip(probabilities, outcomes):
        bins[min(9, int(p * 10))].append((p, y))
    ece = sum(len(part) / len(graded) * abs(sum(p for p, _ in part)/len(part) - sum(y for _, y in part)/len(part)) for part in bins if part)
    cluster = [sum(values) / len(values) for values in slate_returns.values()]
    mean_cluster = sum(cluster) / len(cluster)
    if len(cluster) > 1:
        variance = sum((value - mean_cluster) ** 2 for value in cluster) / (len(cluster)-1)
        cluster_lcb = mean_cluster - 1.645 * math.sqrt(variance / len(cluster))
    else:
        cluster_lcb = None
    return {"bets": len(graded), "hit_rate": sum(outcomes)/len(graded), "roi": sum(returns)/len(returns),
            "brier": brier, "market_brier": market_brier, "brier_increase_vs_market": brier-market_brier,
            "log_loss": log_loss, "market_log_loss": market_log_loss, "log_loss_increase_vs_market": log_loss-market_log_loss,
            "ece": ece, "max_drawdown_units": drawdown,
            "slate_clustered_roi_lcb": cluster_lcb}


def certification(repo_root: Path, promotion_policy: dict[str, Any], eligible: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> dict[str, Any]:
    slates = len({row["event_date"] for row in eligible})
    selections = [row for row in eligible if row.get("eligible")]
    required = promotion_policy["sample_sufficiency"]
    failures: list[str] = []
    if slates < required["minimum_independent_slates"]:
        failures.append(f"INDEPENDENT_SLATES:{slates}<{required['minimum_independent_slates']}")
    if len(selections) < required["minimum_selected_singles_per_capability"]:
        failures.append(f"SELECTED_SINGLES:{len(selections)}<{required['minimum_selected_singles_per_capability']}")
    if not eligible:
        failures.append("FROZEN_USABLE_PROBABILITY_HISTORY_UNAVAILABLE")
    metrics = _metrics(selections)
    capability_results: dict[str, dict[str, Any]] = {}
    states = capability_states()
    for capability in sorted(set(TARGET_CAPABILITY.values())):
        rows = [row for row in selections if TARGET_CAPABILITY.get(str(row.get("market"))) == capability]
        cap_slates = len({row["event_date"] for row in rows})
        cap_metrics = _metrics(rows)
        cap_failures = []
        if cap_slates < required["minimum_independent_slates"]:
            cap_failures.append(f"INDEPENDENT_SLATES:{cap_slates}<{required['minimum_independent_slates']}")
        if len(rows) < required["minimum_selected_singles_per_capability"]:
            cap_failures.append(f"SELECTED_SINGLES:{len(rows)}<{required['minimum_selected_singles_per_capability']}")
        if rows:
            calibration = promotion_policy["calibration_requirements"]
            economic = promotion_policy["economic_requirements"]
            if cap_metrics["ece"] is None or cap_metrics["ece"] > calibration["maximum_ece"]:
                cap_failures.append("ECE_REQUIREMENT_FAILED")
            if cap_metrics["brier_increase_vs_market"] > calibration["maximum_brier_increase_vs_market"]:
                cap_failures.append("BRIER_VS_MARKET_REQUIREMENT_FAILED")
            if cap_metrics["log_loss_increase_vs_market"] > calibration["maximum_log_loss_increase_vs_market"]:
                cap_failures.append("LOG_LOSS_VS_MARKET_REQUIREMENT_FAILED")
            if cap_metrics["roi"] is None or cap_metrics["roi"] < economic["minimum_realized_roi"]:
                cap_failures.append("ROI_REQUIREMENT_FAILED")
            if cap_metrics["slate_clustered_roi_lcb"] is None or cap_metrics["slate_clustered_roi_lcb"] < economic["minimum_slate_clustered_roi_lcb"]:
                cap_failures.append("SLATE_CLUSTERED_ROI_LCB_FAILED")
            if cap_metrics["max_drawdown_units"] is None or abs(cap_metrics["max_drawdown_units"]) > economic["maximum_drawdown_units"]:
                cap_failures.append("DRAWDOWN_REQUIREMENT_FAILED")
            by_source: dict[str, float] = {}
            for row in rows:
                realized = _realized_return(row)
                if realized is not None:
                    by_source[str(row.get("book") or "UNKNOWN")] = by_source.get(str(row.get("book") or "UNKNOWN"), 0.0) + realized
            positive_total = sum(max(0.0, value) for value in by_source.values())
            source_share = max((max(0.0, value) / positive_total for value in by_source.values()), default=1.0) if positive_total else 1.0
            cap_metrics["maximum_single_source_return_share"] = source_share
            if source_share > economic["maximum_single_source_return_share"]:
                cap_failures.append("SOURCE_CONCENTRATION_REQUIREMENT_FAILED")
            controls = [row for row in eligible if not row.get("eligible") and TARGET_CAPABILITY.get(str(row.get("market"))) == capability and _realized_return(row) is not None]
            minimum_controls = promotion_policy["selector_discrimination"]["minimum_boundary_comparison_count"]
            if len(controls) < minimum_controls:
                cap_failures.append(f"REJECTED_CONTROL_COUNT:{len(controls)}<{minimum_controls}")
            elif promotion_policy["selector_discrimination"]["require_accepted_return_not_worse_than_rejected"]:
                accepted_return = sum(_realized_return(row) or 0.0 for row in rows) / len(rows)
                rejected_return = sum(_realized_return(row) or 0.0 for row in controls) / len(controls)
                cap_metrics["accepted_mean_return"] = accepted_return
                cap_metrics["rejected_control_mean_return"] = rejected_return
                if accepted_return < rejected_return:
                    cap_failures.append("SELECTOR_DISCRIMINATION_REQUIREMENT_FAILED")
        if not cap_failures and rows:
            states[capability] = CapabilityAuthority.CERTIFIED.value
        capability_results[capability] = {"eligible_slates": cap_slates, "selected_singles": len(rows), "metrics": cap_metrics, "failures": cap_failures}
    certified = [name for name, value in states.items() if value == CapabilityAuthority.CERTIFIED.value]
    if eligible and not certified:
        failures.append("CAPABILITY_LEVEL_SAMPLE_REQUIREMENTS_NOT_MET")
    status = "HISTORICAL_VALIDATION_PARTIAL" if certified else "HISTORICAL_VALIDATION_FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "evidence_state": "LOCKED_HISTORICAL_VALIDATION",
        "policy_commit": FROZEN_POLICY_COMMIT,
        "policy_hash": build_policy_manifest(repo_root)["policy_hash"],
        "eligible_slates": slates,
        "eligible_candidates": len(eligible),
        "selected_singles": len(selections),
        "excluded_candidates": len(exclusions),
        "metrics": metrics,
        "capability_results": capability_results,
        "capabilities": states,
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
    certified = [name for name, value in result["capabilities"].items() if value == CapabilityAuthority.CERTIFIED.value]
    validation_state = EngineState.PRODUCTION_CANDIDATE if certified else EngineState.LOCKED_HISTORICAL_VALIDATION_FAILED
    engine_manifest = build_engine_manifest(
        policy_hash=policy_manifest["policy_hash"], implementation_commit=implementation,
        state=validation_state,
        capabilities=result["capabilities"], rollback_reference=None,
    )
    atomic_write_json(artifacts / "mlb_engine_manifest.json", engine_manifest)
    production_status = {
        "engine": "unified_mlb",
        "implementation_commit": implementation,
        "policy_commit": FROZEN_POLICY_COMMIT,
        "policy_hash": policy_manifest["policy_hash"],
        "state": "PRODUCTION_CANDIDATE" if certified else "UNIFIED_MLB_SHADOW_ONLY",
        "active_engine": "legacy",
        "historical_validation": {
            "status": result["status"], "eligible_slates": result["eligible_slates"],
            "selected_singles": result["selected_singles"], "roi": None,
            "hit_rate": None, "max_drawdown_units": None,
        },
        "live_canary": {
            "status": "PENDING_DARK_DEPLOYMENT" if certified else "NOT_AUTHORIZED_AFTER_HISTORICAL_GATE_FAILURE",
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
