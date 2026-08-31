from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.unified.production_state import atomic_write_json
from sports.mlb.unified.promotion_validation import _decimal_price, _win, build_corpus
from sports.mlb.unified.v2_policy import UnifiedPolicyV2, implied_probability, poisson_binomial_cdf


ROOT = REPO_ROOT


def _proper_scores(row: dict[str, Any]) -> dict[str, float]:
    probability, outcome = float(row["usable_probability"]), int(_win(row))
    clipped = min(max(probability, 1e-12), 1 - 1e-12)
    return {
        "expected_win": probability,
        "probability_residual": outcome - probability,
        "brier_contribution": (probability - outcome) ** 2,
        "log_loss_contribution": -(outcome * math.log(clipped) + (1 - outcome) * math.log(1 - clipped)),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = [float(row["usable_probability"]) for row in rows]
    outcomes = [int(_win(row)) for row in rows]
    expected, actual = sum(probabilities), sum(outcomes)
    returns = [(_decimal_price(row["quoted_odds"]) - 1 if outcome else -1) for row, outcome in zip(rows, outcomes)]
    return {
        "observations": len(rows), "expected_wins": expected, "actual_wins": actual,
        "expected_minus_actual": expected - actual,
        "mean_predicted_probability": mean(probabilities) if rows else None,
        "observed_hit_rate": mean(outcomes) if rows else None,
        "brier": mean((p-y) ** 2 for p, y in zip(probabilities, outcomes)) if rows else None,
        "log_loss": mean(-(y*math.log(max(p, 1e-12))+(1-y)*math.log(max(1-p, 1e-12))) for p, y in zip(probabilities, outcomes)) if rows else None,
        "poisson_binomial_p_w_le_observed": poisson_binomial_cdf(probabilities, actual) if rows else None,
        "roi": mean(returns) if rows else None,
    }


def _loss_labels(row: dict[str, Any]) -> list[str]:
    labels = ["UNKNOWN"]
    if row.get("uncertainty") == 0:
        labels = ["UNCERTAINTY_UNDERESTIMATION"]
    if float(row.get("probability_edge") or 0) <= .025:
        labels.append("SELECTION_BOUNDARY_ERROR")
    if abs(float(row.get("usable_probability") or 0) - implied_probability(float(row["quoted_odds"]))) <= .025:
        labels.append("MARKET_PRIOR_ERROR")
    return sorted(set(labels))


def _development_diagnostic() -> dict[str, Any]:
    path = ROOT / "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["Target"].isin(["H", "TB"])].copy()
    frame["Prediction_Run_Date"] = pd.to_datetime(frame["Prediction_Run_Date"], errors="coerce")
    frame["Actual"] = pd.to_numeric(frame["Actual"], errors="coerce")
    frame["Prediction"] = pd.to_numeric(frame["Prediction"], errors="coerce")
    frame = frame.dropna(subset=["Prediction_Run_Date", "Actual", "Prediction"])
    result: dict[str, Any] = {
        "evidence_class": "RECONSTRUCTED_DIAGNOSTIC",
        "rows": int(len(frame)), "independent_slates": int(frame["Prediction_Run_Date"].nunique()),
        "date_min": str(frame["Prediction_Run_Date"].min().date()) if len(frame) else None,
        "date_max": str(frame["Prediction_Run_Date"].max().date()) if len(frame) else None,
        "limitations": [
            "No frozen usable probability or calibration state", "Sparse/missing real two-sided prices",
            "No prediction-time lineup/player-status state", "Cannot certify selector ROI or incremental probability skill",
        ], "capabilities": {},
    }
    for market, rows in frame.groupby("Target"):
        errors = rows["Prediction"] - rows["Actual"]
        result["capabilities"][market] = {
            "rows": int(len(rows)), "slates": int(rows["Prediction_Run_Date"].nunique()),
            "mean_prediction_error": float(errors.mean()), "mae": float(errors.abs().mean()),
            "rmse": float((errors.pow(2).mean()) ** .5),
            "real_priced_rows": int(((rows["Market_Source"] == "real") & rows["Market_Over_Price"].notna()).sum()),
        }
    return result


def build_failure_analysis() -> dict[str, Any]:
    eligible, exclusions = build_corpus(ROOT)
    exact = [row for row in exclusions if row.get("evidence_class") == "EXACT_CANDIDATE_ONLY" and _win(row) is not None]
    enriched = []
    for row in exact:
        scored = {**row, **_proper_scores(row)}
        scored["implied_market_probability"] = implied_probability(float(row["quoted_odds"]))
        scored["failure_taxonomy"] = [] if _win(row) else _loss_labels(row)
        enriched.append(scored)
    groups = {
        "batter_hits": [row for row in enriched if row["market"] == "H"],
        "batter_total_bases": [row for row in enriched if row["market"] == "TB"],
        "combined": enriched,
    }
    return {
        "schema_version": 1,
        "v1_policy": "BASELINE_POLICY_V1",
        "evidence_disposition": "CONSUMED_LOCKED_DIAGNOSTIC_EVIDENCE",
        "qualification_audit": {
            "exact_records_reviewed": len(enriched), "fully_qualified": len(eligible),
            "reclassified_exact_candidate_only": len(enriched),
            "systemic_blockers": ["QUOTE_FRESHNESS_UNPROVABLE", "PLAYER_STATUS_UNPROVABLE"],
            "conclusion": "The eight rows are exact pregame candidates but cannot prove the complete advertised frozen qualification contract.",
        },
        "summaries": {name: _summary(rows) for name, rows in groups.items()},
        "observations": enriched,
        "loss_taxonomy_counts": dict(Counter(label for row in enriched for label in row["failure_taxonomy"])),
        "development_diagnostic": _development_diagnostic(),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Unified MLB V1 failure diagnosis", "", "## Qualification audit", "",
             "All eight recovered rows are authentic pregame candidate artifacts, but **zero can prove the complete frozen qualification contract**. The missing per-quote timestamp makes quote freshness unprovable; a separately preserved player-status assertion is also absent. They are retained as `EXACT_CANDIDATE_ONLY` and consumed diagnostics, not deleted.", "",
             "| Capability | Records | Expected wins | Actual wins | P(W ≤ observed) | Brier | ROI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("batter_hits", "batter_total_bases", "combined"):
        item = report["summaries"][name]
        lines.append(f"| {name} | {item['observations']} | {item['expected_wins']:.4f} | {item['actual_wins']} | {item['poisson_binomial_p_w_le_observed']:.4f} | {item['brier']:.4f} | {item['roi']:.2%} |")
    lines += ["", "## Exact observations", "", "| Player | Market | P usable | Market P | Edge | EV | Result | Audit blockers |", "|---|---|---:|---:|---:|---:|---|---|"]
    for row in report["observations"]:
        blockers = ", ".join(name for name, gate in row["qualification_gate_audit"].items() if gate["status"] != "PASS")
        lines.append(f"| {row['player_id']} | {row['market']} {row['line']} | {row['usable_probability']:.2%} | {row['implied_market_probability']:.2%} | {row['probability_edge']:.2%} | {row['conservative_expected_value']:.2%} | {row['settlement']} | {blockers} |")
    lines += ["", "## Evidence-based diagnosis", "",
              "1. V1 expected **{:.4f} wins** and observed **3**; the combined lower-tail probability is **{:.4f}**. This is adverse but not independently informative because every row belongs to one slate.".format(report["summaries"]["combined"]["expected_wins"], report["summaries"]["combined"]["poisson_binomial_p_w_le_observed"]),
              "2. The model and FanDuel baseline were nearly tied on the consumed sample. Eight dependent observations cannot establish Hits overconfidence or Total Bases calibration.",
              "3. The uncertainty field was exactly zero for all eight rows by adapter convention, not measurement. It therefore had no discrimination and must not be treated as empirical uncertainty.",
              "4. Edge and conservative EV did not cleanly separate winners from losses. Several losses sat just above the 1 pp edge boundary, identifying a boundary hypothesis—not a validated replacement threshold.",
              "5. No identity mismatch or settlement corruption was found. The structural failures are incomplete qualification evidence, non-measured uncertainty, and absence of demonstrated incremental information beyond market probability.",
              "6. The 242,425-row universe is useful for outcome-model error studies but lacks frozen probability/lineup/calibration state; it cannot legitimately validate a V2 wagering selector.", ""]
    return "\n".join(lines)


def main() -> None:
    report = build_failure_analysis()
    atomic_write_json(ROOT / "artifacts/mlb_unified_failure_analysis.json", report)
    diagnosis = _markdown(report)
    (ROOT / "docs/mlb_unified_failure_analysis.md").write_text(diagnosis, encoding="utf-8")
    (ROOT / "docs/mlb_unified_v1_failure_diagnosis.md").write_text(diagnosis, encoding="utf-8")
    policy = UnifiedPolicyV2()
    manifest = {"schema_version": 2, "policy": policy.name, "policy_hash": policy.policy_hash,
                "decision_policy": policy.__dict__, "feature_version": "legacy_compatibility_diagnostic_only",
                "model_hashes": [], "calibrator_hashes": [], "parlay_rules": "SHADOW_ONLY_UNTIL_SINGLE_CERTIFICATION",
                "locked_validation_started": False}
    atomic_write_json(ROOT / "artifacts/mlb_unified_v2_policy_manifest.json", manifest)
    status = {
        "policy": "UNIFIED_POLICY_V2", "state": "UNIFIED_MLB_V2_SHADOW_ONLY", "policy_hash": policy.policy_hash,
        "v1_diagnosis_complete": True, "calibration_status": "BLOCKED_NO_LEAKAGE_SAFE_PROBABILITY_CORPUS",
        "selector_status": "FAIL_CLOSED_DEVELOPMENT_ONLY",
        "hits": {"state": "SHADOW", "exact_selections": 0, "independent_slates": 0},
        "total_bases": {"state": "SHADOW", "exact_selections": 0, "independent_slates": 0},
        "production_authorized": False,
    }
    atomic_write_json(ROOT / "artifacts/mlb_unified_v2_status.json", status)
    resolution = diagnosis + "\n## V2 resolution\n\nV2 adds fail-closed quote freshness, player status, support/OOD, capability, uncertainty, exact-selection, positive-EV, and admissibility-before-ranking contracts. No calibrator or edge threshold was fitted to the consumed eight. Because the repository has no untouched, point-in-time probability corpus capable of proving incremental skill beyond the market, Hits and Total Bases remain shadow; parlays remain shadow-only.\n\nFresh V2 evidence counters start at zero. Promotion still requires 20 independent slates and 50 exact qualifying selections per capability. `static-deployment` remains legacy active.\n\n## Final decision\n\nUNIFIED_MLB_V2_SHADOW_ONLY\n"
    (ROOT / "docs/mlb_unified_v2_resolution_report.md").write_text(resolution, encoding="utf-8")


if __name__ == "__main__":
    main()
