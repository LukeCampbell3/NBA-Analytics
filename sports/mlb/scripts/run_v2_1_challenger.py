#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.adapters import TARGET_MARKETS
from sports.mlb.unified.pipeline import run as run_unified
from sports.mlb.unified.production_state import atomic_write_json
from sports.mlb.unified.v2_1_challenger import (
    BASELINE_POLICY_HASH, UnifiedPolicyV21, select_challenger,
)
from sports.mlb.unified.v2_evidence import capture_policy_generation, canonical_hash
from sports.mlb.unified.candidate_contract import from_bet_candidate, terminal_decision


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(valid) / len(valid) if valid else None


def normalize(candidate: Any, *, slate_id: str, prediction_time: str) -> dict[str, Any]:
    source = dict(candidate.source_payload or {})
    contract = from_bet_candidate(candidate, slate_id=slate_id, prediction_time=prediction_time)
    row = contract.to_dict()
    raw_probability = contract.raw_structural_probability
    calibrated = contract.calibrated_probability
    odds = contract.quoted_odds
    decimal = candidate.decimal_price
    return {
        **row, "event_id": contract.game_id, "capability": contract.market_type,
        "quote_timestamp": contract.odds_snapshot_time,
        "prediction_timestamp": prediction_time, "decision_timestamp": prediction_time,
        "model_version": source.get("model_version") or "compatibility_models_at_source_artifact",
        "calibrator_version": source.get("calibrator_version") or "legacy_source_calibrator",
        "raw_probability": raw_probability, "uncertainty": candidate.uncertainty,
        "edge": candidate.probability_edge,
        "raw_ev": raw_probability * decimal - 1 if raw_probability is not None and decimal else None,
        "calibrated_ev": calibrated * decimal - 1 if calibrated is not None and decimal else None,
        "conservative_ev": candidate.conservative_expected_value,
    }


def _baseline_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        reasons = []
        if row["quote_timestamp"] is None: reasons.append("QUOTE_FRESHNESS_UNPROVABLE")
        if row["player_status"] is None: reasons.append("PLAYER_STATUS_INVALID")
        if row["support_score"] is None: reasons.append("SUPPORT_INVALID")
        if row["ood_status"] != "IN_SUPPORT": reasons.append("OOD_UNMEASURED")
        if row["uncertainty"] is None: reasons.append("UNCERTAINTY_INVALID")
        if not row["market_id"] or not row["selection_id"]: reasons.append("EXACT_SELECTION_UNAVAILABLE")
        result.append({**row, "policy_hash": BASELINE_POLICY_HASH, "admissible": not reasons,
                       "rejection_reasons": reasons, "ranking_position": None,
                       "final_selection_decision": False})
    return result


def _evidence_candidate(row: dict[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    return {
        **row, "final_selection_decision": row["candidate_id"] in selected_ids,
        "ranking_position": row.get("ranking_position"),
        "rejection_reasons": list(row.get("rejection_reasons") or []),
        "admissible": bool(row.get("admissible")),
    }


def run(data_dir: Path, run_date: str) -> dict[str, Any]:
    unified = run_unified(data_dir)
    source_candidates, source_status = unified.candidates, unified.source_status
    prediction_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    normalized = [normalize(candidate, slate_id=f"MLB_{run_date.replace('-', '')}", prediction_time=prediction_time)
                  for candidate in source_candidates]
    baseline = _baseline_view(normalized)
    policy = UnifiedPolicyV21()
    challenger = select_challenger(normalized, policy)
    selected_ids = {row["candidate_id"] for row in challenger["selected"]}
    challenger_rows = [_evidence_candidate(row, selected_ids) for row in challenger["evaluated"]]
    baseline_ids = {row["candidate_id"] for row in baseline if row["final_selection_decision"]}
    disagreements = []
    for candidate_id in sorted({row["candidate_id"] for row in baseline + challenger_rows}):
        base = next((row for row in baseline if row["candidate_id"] == candidate_id), {})
        chall = next((row for row in challenger_rows if row["candidate_id"] == candidate_id), {})
        if (base.get("final_selection_decision") != chall.get("final_selection_decision")
                or base.get("rejection_reasons") != chall.get("rejection_reasons")):
            disagreements.append({"candidate_id": candidate_id,
                                  "baseline_selected": bool(base.get("final_selection_decision")),
                                  "challenger_selected": bool(chall.get("final_selection_decision")),
                                  "baseline_reasons": base.get("rejection_reasons", []),
                                  "challenger_reasons": chall.get("rejection_reasons", [])})
    rejection_counts = Counter(reason for row in challenger_rows for reason in row["rejection_reasons"])
    selected = [row for row in challenger_rows if row["final_selection_decision"]]
    daily = {
        "schema_version": 1, "slate_id": f"MLB_{run_date.replace('-', '')}", "run_date": run_date,
        "generated_at_utc": prediction_time, "source_status": source_status,
        "normalized_candidates": len(normalized),
        "fully_valid_candidates": sum(not any(reason in row["rejection_reasons"] for reason in
                                               ("QUOTE_FRESHNESS_UNPROVABLE", "PLAYER_STATUS_INVALID", "EXACT_SELECTION_UNAVAILABLE"))
                                      for row in challenger_rows),
        "admissible_candidates": sum(row["admissible"] for row in challenger_rows),
        "selected_bets": len(selected), "abstention_count": int(not selected),
        "rejection_count_by_reason": dict(sorted(rejection_counts.items())),
        "means": {
            "raw_probability": _mean([row["raw_probability"] for row in normalized]),
            "usable_probability": _mean([row["usable_probability"] for row in normalized]),
            "market_probability": _mean([row["market_implied_probability"] for row in normalized]),
            "uncertainty": _mean([row.get("uncertainty") for row in challenger_rows]),
            "edge": _mean([row["edge"] for row in normalized]),
            "conservative_ev": _mean([row.get("conservative_expected_value") for row in challenger_rows]),
            "support": _mean([row["support_score"] for row in normalized]),
        },
        "ood_count": sum(row["ood_status"] != "IN_SUPPORT" for row in normalized),
        "capabilities": {capability: {"exact_candidates": 0, "exact_qualified_selections": 0,
                                      "independent_slates": 0, "wins": 0, "losses": 0,
                                      "expected_wins": 0.0, "actual_wins": 0}
                         for capability in sorted(set(row["capability"] for row in normalized))},
    }
    status = {
        "policy": policy.name, "state": "PROSPECTIVE_SHADOW",
        "baseline_policy_hash": BASELINE_POLICY_HASH, "challenger_policy_hash": policy.policy_hash,
        "production_authorized": False, "certification_started": False,
        "normalized_candidates": len(normalized), "admissible": len(challenger["admissible"]),
        "selected": len(selected), "disagreements": len(disagreements),
        "blockers": sorted(rejection_counts), "decision": terminal_decision(challenger_rows),
    }
    return {"daily": daily, "baseline": baseline, "challenger": challenger_rows,
            "disagreements": disagreements, "status": status}


def _write_markdown(path: Path, title: str, result: dict[str, Any]) -> None:
    daily, status = result["daily"], result["status"]
    lines = [f"# {title}", "", f"Slate: `{daily['slate_id']}`", "",
             f"Baseline: `{status['baseline_policy_hash']}`", "",
             f"Challenger: `{status['challenger_policy_hash']}`", "",
             "| Population | Count |", "|---|---:|",
             f"| Normalized | {daily['normalized_candidates']} |",
             f"| Fully valid | {daily['fully_valid_candidates']} |",
             f"| Admissible | {daily['admissible_candidates']} |",
             f"| Selected | {daily['selected_bets']} |", "", "## Rejections", ""]
    lines.extend(f"- `{key}`: {value}" for key, value in daily["rejection_count_by_reason"].items())
    lines += ["", "## Scientific status", "",
              "No historical outcome was used to tune V2.1. Current inputs do not preserve quote time, independent player status, measured uncertainty components, or OOD state, so the challenger abstains. Coverage-risk, rank, Top-K, boundary, and uncertainty-discrimination claims remain `INSUFFICIENT_PROSPECTIVE_EVIDENCE` until settled all-candidate slates accumulate.", "",
              "Parlays remain shadow-only.", "", "## Final decision", "", status["decision"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_high_efficiency(path: Path, result: dict[str, Any]) -> None:
    status = result["status"]
    questions = [
        ("Does usable probability rank winners?", "INSUFFICIENT_PROSPECTIVE_EVIDENCE"),
        ("Does measured uncertainty predict error?", "Not yet; current live inputs do not contain measured components."),
        ("Does lower-bound probability improve selection?", "UNTESTED_PROSPECTIVELY"),
        ("Does edge predict realized value?", "INSUFFICIENT_PROSPECTIVE_EVIDENCE"),
        ("Does conservative EV predict realized value?", "INSUFFICIENT_PROSPECTIVE_EVIDENCE"),
        ("Does support predict reliability?", "INSUFFICIENT_PROSPECTIVE_EVIDENCE"),
        ("Does OOD rejection improve behavior?", "UNTESTED; OOD state is currently unmeasured."),
        ("Does market disagreement contain incremental signal?", "NO RELIABLE INCREMENTAL SIGNAL ESTABLISHED"),
        ("Does opportunity modeling improve Hits?", "Framework implemented; no leakage-safe fitted inputs or locked comparison yet."),
        ("Does discrete outcome modeling improve Total Bases?", "Exact convolution implemented; no locked comparison yet."),
        ("Does ranking improve over all-admissible selection?", "UNTESTED because there are no fully admissible exact candidates."),
        ("What Top-K policy is most efficient?", "No Top-K policy is supported; Top-1/2/3 diagnostics are armed."),
        ("Are near-threshold bets materially weaker?", "INSUFFICIENT PROSPECTIVE BOUNDARY COUNTS"),
        ("Which rejection gates are useful?", "Integrity gates are operationally necessary; outcome discrimination is not yet estimable."),
        ("Which gates are redundant?", "None can be declared redundant from current evidence."),
        ("Does V2.1 outperform V2 without increasing fragility?", "NOT ESTABLISHED"),
        ("Is improvement stable across independent slates?", "NOT ESTABLISHED; certification has not started."),
    ]
    lines = ["# MLB V2 high-efficiency selection report", "",
             f"V2 baseline hash: `{status['baseline_policy_hash']}`", "",
             f"V2.1 challenger hash: `{status['challenger_policy_hash']}`", "",
             "V2.1 is a fail-closed prospective shadow challenger. It does not use the eight consumed V1 outcomes for fitting.", ""]
    for index, (question, answer) in enumerate(questions, 1):
        lines += [f"## {index}. {question}", "", answer, ""]
    lines += ["## Final decision", "", status["decision"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "sports/mlb/web/data")
    parser.add_argument("--run-date")
    parser.add_argument("--evidence-ledger", type=Path,
                        default=REPO_ROOT / "sports/mlb/data/predictions/unified/v2_1_evidence.jsonl")
    args = parser.parse_args()
    run_date = args.run_date or json.loads((args.data_dir / "daily_predictions.json").read_text())["run_date"]
    result = run(args.data_dir, run_date)
    generated_at = result["daily"]["generated_at_utc"]
    generation_id = canonical_hash({"run_date": run_date, "generated_at": generated_at,
                                    "challenger": result["status"]["challenger_policy_hash"]})[:24]
    capture_policy_generation(
        args.evidence_ledger, generation_id=generation_id, generated_at_utc=generated_at,
        run_date=run_date, baseline_policy_hash=BASELINE_POLICY_HASH,
        challenger_policy_hash=result["status"]["challenger_policy_hash"],
        baseline_candidates=result["baseline"], challenger_candidates=result["challenger"],
        disagreements=result["disagreements"],
    )
    artifacts, docs = REPO_ROOT / "artifacts", REPO_ROOT / "docs"
    atomic_write_json(artifacts / "mlb_v2_daily_evidence.json", result["daily"])
    atomic_write_json(artifacts / "mlb_v2_challenger_status.json", result["status"])
    policy = UnifiedPolicyV21()
    atomic_write_json(artifacts / "mlb_v2_1_policy_manifest.json", {
        "schema_version": 1, "policy": policy.name, "policy_hash": policy.policy_hash,
        "baseline_policy_hash": BASELINE_POLICY_HASH, "configuration": asdict(policy),
        "model_hashes": [], "calibrator_hashes": [],
        "feature_version": "point_in_time_candidate_v1",
        "uncertainty_configuration": "eight_component_root_sum_square",
        "ranking_configuration": "pareto_then_lexicographic_lcb_ev_edge_support_uncertainty",
        "selection_limit": policy.top_k, "certification_started": False,
    })
    empty_diagnostics = {"state": "INSUFFICIENT_PROSPECTIVE_EVIDENCE", "settled_candidates": 0,
                         "coverage_risk": [], "rank_performance": [], "uncertainty_quantiles": [],
                         "top_k": [], "market_comparison": [], "concentration": {}}
    atomic_write_json(artifacts / "mlb_v2_selector_diagnostics.json", empty_diagnostics)
    atomic_write_json(artifacts / "mlb_v2_boundary_diagnostics.json",
                      {"state": "INSUFFICIENT_PROSPECTIVE_EVIDENCE", "boundaries": {key: [] for key in
                       ("probability", "edge", "uncertainty", "conservative_ev", "support")}})
    _write_markdown(docs / "mlb_v2_daily_evidence.md", "MLB V2 daily evidence", result)
    _write_markdown(docs / "mlb_v2_selector_efficiency_report.md", "MLB V2 selector efficiency", result)
    _write_markdown(docs / "mlb_v2_challenger_report.md", "MLB V2.1 challenger", result)
    _write_high_efficiency(docs / "mlb_v2_high_efficiency_selection_report.md", result)
    print(json.dumps(result["status"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
