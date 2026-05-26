from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.common import build_candidate_id, safe_float
from research.improvement_ledger.ledger import append_improvement_entry


FAILURE_MODE_SIGNAL_MAP = {
    "REBOUND_UPPER_BAND_SUPPLY_RISK": ("upper_band_line_penalty", "consider", False),
    "REBOUND_LOW_LINE_ROLE_VOLATILITY": ("low_line_role_volatility_penalty", "consider", False),
    "REBOUND_SHARE_COMPETITION": ("rebound_share_competition_penalty", "consider", False),
    "REBOUND_SUPPLY_COLLAPSE": ("rebound_supply_penalty", "consider", False),
    "MINUTES_BAND_FAILURE": ("low_line_role_volatility_penalty", "pass", True),
    "MARKET_PRICE_MISPLACEMENT": ("risk_penalty", "consider", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a validated intervention into a generic adjustment sidecar.")
    parser.add_argument("--intervention-candidates-csv", type=Path, required=True)
    parser.add_argument("--candidate-pool-csv", type=Path, required=True)
    parser.add_argument("--intervention-id", type=str, required=True)
    parser.add_argument("--failure-mode-adjustments-csv-out", type=Path, required=True)
    parser.add_argument("--intervention-config-yaml-out", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, default=None)
    return parser.parse_args()


def materialize_intervention(
    intervention_row: dict[str, Any] | pd.Series,
    candidate_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidate_pool.empty:
        return pd.DataFrame(columns=["candidate_id"]), {
            "enabled": True,
            "row_count": 0,
            "rollback_plan": "Delete the generated failure_mode_adjustments.csv and disable the intervention ablation flag.",
        }
    row = dict(intervention_row)
    failure_mode_id = str(row.get("failure_mode_id", "")).strip()
    intervention_type = str(row.get("intervention_type", "")).strip()
    column_name, downgrade_tier, veto_default = FAILURE_MODE_SIGNAL_MAP.get(
        failure_mode_id,
        ("risk_penalty", "consider", False),
    )
    work = candidate_pool.copy()
    work["candidate_id"] = build_candidate_id(work)
    signal = pd.to_numeric(work.get(column_name), errors="coerce").fillna(0.0).clip(lower=0.0)
    if signal.empty:
        return pd.DataFrame(columns=["candidate_id"]), {
            "enabled": True,
            "row_count": 0,
            "rollback_plan": "Delete the generated failure_mode_adjustments.csv and disable the intervention ablation flag.",
        }
    active_mask = signal > 0.0
    if intervention_type == "hard_gate":
        active_mask = active_mask & (signal >= max(float(signal.median()), 0.08))
    adjustments = pd.DataFrame(
        {
            "candidate_id": work.loc[active_mask, "candidate_id"].astype(str),
            "failure_mode_id": failure_mode_id,
            "penalty": signal.loc[active_mask].astype("float64"),
            "downgrade_tier": downgrade_tier if intervention_type != "hard_gate" else "pass",
            "veto_flag": bool(veto_default or intervention_type == "hard_gate"),
            "opposite_side_candidate_flag": intervention_type == "opposite_side_discovery",
            "alt_line_candidate_flag": intervention_type == "alternate_line_discovery",
            "explanation": f"{failure_mode_id}:{column_name}",
        }
    )
    config = {
        "intervention_id": str(row.get("intervention_id", "")).strip(),
        "failure_mode_id": failure_mode_id,
        "intervention_type": intervention_type,
        "shadow_only": True,
        "ablation_flag": str(row.get("ablation_flag", row.get("intervention_id", ""))).strip(),
        "signal_column": column_name,
        "row_count": int(len(adjustments)),
        "rollback_plan": str(
            row.get(
                "rollback_rule",
                "Delete the generated failure_mode_adjustments.csv and disable the intervention ablation flag.",
            )
        ),
    }
    return adjustments, config


def main() -> None:
    args = parse_args()
    proposals = pd.read_csv(args.intervention_candidates_csv)
    matches = proposals.loc[proposals["intervention_id"].astype(str) == str(args.intervention_id)].copy()
    if matches.empty:
        raise ValueError(f"Intervention not found: {args.intervention_id}")
    candidate_pool = pd.read_csv(args.candidate_pool_csv)
    adjustments, config = materialize_intervention(matches.iloc[0].to_dict(), candidate_pool)
    args.failure_mode_adjustments_csv_out.resolve().parent.mkdir(parents=True, exist_ok=True)
    adjustments.to_csv(args.failure_mode_adjustments_csv_out, index=False)
    args.intervention_config_yaml_out.resolve().write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    append_improvement_entry(
        {
            "improvement_id": str(config.get("intervention_id", "")),
            "failure_mode_id": str(config.get("failure_mode_id", "")),
            "intervention_id": str(config.get("intervention_id", "")),
            "author_or_run_id": "materialize_intervention",
            "hypothesis": f"Materialized {config.get('failure_mode_id')} as a shadow sidecar adjustment.",
            "implementation_files": [str(args.failure_mode_adjustments_csv_out), str(args.intervention_config_yaml_out)],
            "validation_windows": [],
            "metrics_before": {},
            "metrics_after": {},
            "segment_results": {},
            "promotion_status": "shadow_only_candidate",
            "blocked_reasons": ["promotion_gate_pending"],
            "rollback_rule": str(config.get("rollback_plan", "")),
            "final_decision": "shadow_materialized",
        },
        ledger_path=args.ledger_path,
    )


if __name__ == "__main__":
    main()
