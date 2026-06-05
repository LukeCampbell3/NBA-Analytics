#!/usr/bin/env python3
"""Freeze v9.1 as the honest distribution-led shadow baseline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze v9.1 honest baseline manifest")
    parser.add_argument("--v9-manifest", type=Path, default=ROOT / "model" / "props" / "v9" / "manifest.json")
    parser.add_argument("--v9-audit", type=Path, default=ROOT / "model" / "props" / "v9" / "validation" / "paired_shadow_report.json")
    parser.add_argument("--v10-audit", type=Path, default=ROOT / "model" / "props" / "v10" / "validation" / "audit_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_1" / "manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v9_manifest = _read_json(args.v9_manifest)
    v9_audit = _read_json(args.v9_audit)
    v10_audit = _read_json(args.v10_audit)

    comparison = v9_audit.get("comparison", {})
    if not comparison and v10_audit:
        comparison = {
            key: value
            for key, value in v10_audit.get("comparison", {}).items()
            if key.startswith("v9") or key in {"current_market_no_vig", "side_prior"}
        }
    elif comparison and v10_audit:
        comparison = dict(comparison)
        if "side_prior" not in comparison and "side_prior" in v10_audit.get("comparison", {}):
            comparison["side_prior"] = v10_audit["comparison"]["side_prior"]

    manifest = {
        "model_version": "prop_engine_v9_1_honest_distribution_baseline",
        "status": "shadow_only",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_manifest": str(args.v9_manifest),
        "source_v9_artifacts": v9_manifest.get("artifacts", {}),
        "honest_baselines": {
            "neutral_brier": comparison.get("current_market_no_vig", {}).get("brier", 0.25),
            "side_prior_brier": comparison.get("side_prior", {}).get("brier"),
            "v9_raw_brier": comparison.get("v9_raw", {}).get("brier"),
            "v9_calibrated_brier": comparison.get("v9_calibrated", {}).get("brier"),
            "v9_gated_brier": comparison.get("v9_calibrated_gate", {}).get("brier"),
            "v9_calibrated_ece": comparison.get("v9_calibrated", {}).get("ece"),
        },
        "required_validation": {
            "leakage_audit": True,
            "label_shuffle": True,
            "component_cutoff_safety": True,
            "walk_forward": True,
            "true_market_odds_for_promotion": True,
            "closing_line_clv_for_promotion": True,
        },
        "forbidden_features": [
            "actual_minutes",
            "actual_usage",
            "actual_fga",
            "actual_player_stat",
            "residual",
            "abs_residual",
            "postgame_box_score",
            "closing_line_before_close",
        ],
        "next_required_signal_sources": [
            "pregame_minutes_projection",
            "teammate_out_delta_artifacts",
            "true_odds_snapshots",
            "closing_line_clv_fields",
        ],
        "verdict": "best_honest_stack_distribution_led_shadow_baseline",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest["honest_baselines"], indent=2, default=str))
    print(f"\nWrote v9.1 baseline manifest to {args.output}")


if __name__ == "__main__":
    main()
