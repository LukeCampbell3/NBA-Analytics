from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import as_string_list, safe_float, safe_int, write_json
from research.failure_modes.failure_mode_registry import get_failure_mode, load_failure_mode_registry

PROPOSAL_COLUMNS = [
    "intervention_id",
    "failure_mode_id",
    "intervention_type",
    "target_markets",
    "trigger_condition",
    "expected_benefit",
    "expected_coverage_cost",
    "expected_non_target_damage",
    "required_features",
    "missing_features",
    "overfit_risk",
    "validation_plan",
    "rollback_rule",
    "recommended_next_action",
    "shadow_only",
    "implementation_status",
    "ablation_flag",
    "expected_loss_removal_rate",
    "expected_win_removal_rate",
    "losses",
    "wins",
    "resolved_count",
    "description",
]

ACTION_RANK = {
    "VALIDATE_SHADOW": 0,
    "FEATURE_GAP_BLOCKED": 1,
    "NEEDS_MORE_SAMPLE": 2,
    "REGISTER_UNKNOWN_FIRST": 3,
    "REJECT_RANDOM": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose shadow-only interventions for recurring failure modes.")
    parser.add_argument("--failure-mode-scoreboard-csv", type=Path, required=True)
    parser.add_argument("--intervention-candidates-csv-out", type=Path, required=True)
    parser.add_argument("--intervention-summary-json-out", type=Path, required=True)
    parser.add_argument("--priority-floor", type=float, default=0.005)
    parser.add_argument("--min-resolved-count", type=int, default=4)
    parser.add_argument("--min-loss-count", type=int, default=3)
    parser.add_argument("--pre-event-detectability-floor", type=float, default=0.50)
    parser.add_argument("--max-coverage-loss", type=float, default=0.20)
    return parser.parse_args()


def _overfit_risk(
    row: pd.Series,
    *,
    missing_features: list[str],
) -> str:
    resolved = safe_int(row.get("resolved_count"), default=0)
    coverage_loss = safe_float(row.get("estimated_coverage_cost", row.get("coverage_loss_if_gated")), default=0.0)
    priority = safe_float(row.get("priority_score"), default=0.0)
    if missing_features:
        return "high"
    if resolved < 8 or coverage_loss > 0.18:
        return "high"
    if priority < 0.02:
        return "medium"
    return "low"


def _recommended_next_action(
    row: pd.Series,
    *,
    failure_mode_id: str,
    missing_features: list[str],
    evaluation_status: str,
    priority_floor: float,
    min_resolved_count: int,
    min_loss_count: int,
    pre_event_detectability_floor: float,
    max_coverage_loss: float,
) -> str:
    resolved_count = safe_int(row.get("resolved_count"), default=0)
    losses = safe_int(row.get("losses"), default=0)
    wins = safe_int(row.get("wins"), default=0)
    priority = safe_float(row.get("priority_score"), default=0.0)
    detectability = safe_float(row.get("pre_event_detectability_rate"), default=0.0)
    coverage_loss = safe_float(row.get("estimated_coverage_cost", row.get("coverage_loss_if_gated")), default=0.0)
    loss_removal = safe_float(row.get("estimated_loss_removal_rate"), default=0.0)
    win_removal = safe_float(row.get("estimated_win_removal_rate"), default=0.0)
    non_target_damage = safe_float(row.get("non_target_damage_risk"), default=0.0)
    if failure_mode_id.startswith("UNKNOWN_"):
        return "REGISTER_UNKNOWN_FIRST"
    if evaluation_status == "blocked":
        return "FEATURE_GAP_BLOCKED"
    if resolved_count < int(min_resolved_count) or losses < int(min_loss_count):
        return "NEEDS_MORE_SAMPLE"
    if priority < float(priority_floor) or detectability < float(pre_event_detectability_floor):
        return "NEEDS_MORE_SAMPLE"
    if coverage_loss > float(max_coverage_loss) or non_target_damage > 0.15:
        return "REJECT_RANDOM"
    if win_removal >= loss_removal or losses <= wins:
        return "REJECT_RANDOM"
    if missing_features and evaluation_status != "good":
        return "FEATURE_GAP_BLOCKED"
    return "VALIDATE_SHADOW"


def propose_interventions(
    scoreboard: pd.DataFrame,
    *,
    registry: dict[str, Any] | None = None,
    priority_floor: float = 0.005,
    min_resolved_count: int = 4,
    min_loss_count: int = 3,
    pre_event_detectability_floor: float = 0.50,
    max_coverage_loss: float = 0.20,
    feature_context: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    active_registry = registry or load_failure_mode_registry()
    if scoreboard.empty:
        return pd.DataFrame()
    proposals: list[dict[str, Any]] = []
    for _, row in scoreboard.iterrows():
        failure_mode_id = str(row.get("failure_mode_id", "")).strip()
        if not failure_mode_id:
            continue
        definition = get_failure_mode(failure_mode_id, active_registry)
        if definition is None:
            continue
        resolved_count = safe_int(row.get("resolved_count"), default=0)
        losses = safe_int(row.get("losses"), default=0)
        wins = safe_int(row.get("wins"), default=0)
        candidate_count = safe_int(row.get("candidate_count"), default=0)
        intervention_available = bool(row.get("intervention_available", True))
        if not intervention_available or (resolved_count <= 0 and losses <= 0 and wins <= 0):
            continue
        context = dict((feature_context or {}).get(failure_mode_id, {}))
        required_features = as_string_list(context.get("required_features")) or list(definition.required_pre_event_features)
        missing_features = as_string_list(context.get("missing_features"))
        evaluation_status = str(context.get("evaluation_status", "good" if not missing_features else "partial")).strip().lower() or "partial"
        recommended_next_action = _recommended_next_action(
            row,
            failure_mode_id=failure_mode_id,
            missing_features=missing_features,
            evaluation_status=evaluation_status,
            priority_floor=priority_floor,
            min_resolved_count=min_resolved_count,
            min_loss_count=min_loss_count,
            pre_event_detectability_floor=pre_event_detectability_floor,
            max_coverage_loss=max_coverage_loss,
        )
        for intervention in definition.candidate_interventions:
            intervention_type = str(intervention.get("intervention_type", "")).strip()
            if not intervention_type:
                continue
            intervention_id = f"{failure_mode_id.lower()}__{intervention_type}"
            proposals.append(
                {
                    "intervention_id": intervention_id,
                    "failure_mode_id": failure_mode_id,
                    "intervention_type": intervention_type,
                    "target_markets": "|".join(definition.market_families),
                    "trigger_condition": "|".join(definition.candidate_symptoms),
                    "expected_benefit": safe_float(row.get("expected_improvement_if_gated", row.get("estimated_loss_removal_rate")), default=0.0),
                    "expected_coverage_cost": safe_float(row.get("estimated_coverage_cost", row.get("coverage_loss_if_gated")), default=0.0),
                    "expected_non_target_damage": safe_float(row.get("non_target_damage_risk"), default=0.0),
                    "required_features": "|".join(required_features),
                    "missing_features": "|".join(missing_features),
                    "overfit_risk": _overfit_risk(row, missing_features=missing_features),
                    "validation_plan": "paired_replay:no_op,active_risk,trained_bundle,broader_walk_forward",
                    "rollback_rule": f"Disable {intervention_id} if ROI, Brier, ECE, calibration, coverage, or non-target board integrity worsens in paired replay.",
                    "recommended_next_action": recommended_next_action,
                    "shadow_only": True,
                    "implementation_status": "discovery_only",
                    "ablation_flag": intervention_id,
                    "expected_loss_removal_rate": safe_float(row.get("estimated_loss_removal_rate"), default=0.0),
                    "expected_win_removal_rate": safe_float(row.get("estimated_win_removal_rate"), default=0.0),
                    "losses": losses,
                    "wins": wins,
                    "resolved_count": resolved_count,
                    "description": str(intervention.get("description", "")).strip(),
                }
            )
    if not proposals:
        return pd.DataFrame(columns=PROPOSAL_COLUMNS)
    out = pd.DataFrame(proposals)
    out["_action_rank"] = out["recommended_next_action"].map(lambda value: ACTION_RANK.get(str(value), 99))
    out = out.sort_values(
        ["_action_rank", "expected_benefit", "resolved_count", "expected_coverage_cost", "overfit_risk"],
        ascending=[True, False, False, True, True],
    ).drop(columns=["_action_rank"]).reset_index(drop=True)
    return out


def summarize_interventions(proposals: pd.DataFrame) -> dict[str, Any]:
    if proposals.empty:
        return {
            "proposal_count": 0,
            "shadow_only": True,
            "failure_modes": [],
            "recommended_next_action_counts": {},
        }
    return {
        "proposal_count": int(len(proposals)),
        "shadow_only": bool(proposals.get("shadow_only", pd.Series(dtype=bool)).astype(bool).all()),
        "failure_modes": sorted(proposals.get("failure_mode_id", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()),
        "recommended_next_action_counts": proposals.get("recommended_next_action", pd.Series(dtype="object")).replace("", "UNKNOWN").value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    args = parse_args()
    scoreboard = pd.read_csv(args.failure_mode_scoreboard_csv)
    proposals = propose_interventions(
        scoreboard,
        priority_floor=float(args.priority_floor),
        min_resolved_count=int(args.min_resolved_count),
        min_loss_count=int(args.min_loss_count),
        pre_event_detectability_floor=float(args.pre_event_detectability_floor),
        max_coverage_loss=float(args.max_coverage_loss),
    )
    args.intervention_candidates_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(args.intervention_candidates_csv_out, index=False)
    write_json(args.intervention_summary_json_out, summarize_interventions(proposals))


if __name__ == "__main__":
    main()
