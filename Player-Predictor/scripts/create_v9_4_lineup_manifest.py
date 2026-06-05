#!/usr/bin/env python3
"""Package v9.4 lineup-state artifacts around the honest v9.3 probability stack."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _resolve(path: Path) -> Path:
    if str(path).startswith("/workspace/"):
        return REPO_ROOT / str(path).replace("/workspace/", "", 1)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _copy_tree_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create v9.4 manifest with lineup delta artifacts")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "model" / "props" / "v9_3" / "manifest.json")
    parser.add_argument("--lineup-artifacts", type=Path, default=ROOT / "model" / "props" / "v9_4" / "lineup_delta_artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v9_4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest_path = _resolve(args.source_manifest)
    output = _resolve(args.output)
    lineup_artifacts = _resolve(args.lineup_artifacts)
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_output = _resolve(Path(source_manifest["output"]))
    if not (source_output / "data" / "prop_training_rows.csv").exists():
        raise FileNotFoundError(source_output / "data" / "prop_training_rows.csv")
    if not (lineup_artifacts / "lineup_delta_report.json").exists():
        raise FileNotFoundError(lineup_artifacts / "lineup_delta_report.json")

    _copy_tree_if_exists(source_output / "data", output / "data")
    _copy_tree_if_exists(source_output / "calibration", output / "calibration")
    _copy_tree_if_exists(source_output / "validation", output / "validation")

    lineup_report = json.loads((lineup_artifacts / "lineup_delta_report.json").read_text(encoding="utf-8"))
    manifest = {
        "model_version": "prop_engine_v9_4_lineup_delta_ready_distribution",
        "status": "controlled_shadow_lineup_artifacts_ready",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_3_manifest": str(source_manifest_path.relative_to(REPO_ROOT)) if source_manifest_path.is_relative_to(REPO_ROOT) else str(source_manifest_path),
        "output": str(output.relative_to(REPO_ROOT)) if output.is_relative_to(REPO_ROOT) else str(output),
        "rows": source_manifest.get("rows"),
        "players": source_manifest.get("players"),
        "date_min": source_manifest.get("date_min"),
        "date_max": source_manifest.get("date_max"),
        "artifacts": {
            "data": "data/prop_training_rows.csv",
            "calibration": "calibration",
            "market_odds_schema": source_manifest.get("artifacts", {}).get("market_odds_schema"),
            "lineup_delta_artifacts": str(lineup_artifacts.relative_to(output)) if lineup_artifacts.is_relative_to(output) else str(lineup_artifacts),
        },
        "lineup_delta_artifacts": {
            "status": lineup_report.get("status"),
            "promotion_usage": lineup_report.get("promotion_usage"),
            "teammate_out_delta_rows": lineup_report.get("teammate_out_delta_rows"),
            "teammate_in_delta_rows": lineup_report.get("teammate_in_delta_rows"),
            "team_removed_rows": lineup_report.get("team_removed_rows"),
            "shrink_k": lineup_report.get("shrink_k"),
            "min_with_games": lineup_report.get("min_with_games"),
            "min_without_games": lineup_report.get("min_without_games"),
            "leakage_guardrail": lineup_report.get("leakage_guardrail"),
            "files": lineup_report.get("files"),
        },
        "live_promotion_blockers": [
            "Pregame teammate availability feed must be joined before lineup deltas affect probabilities.",
            "Lineup-adjusted probabilities must beat v9.3 out of sample after availability feed integration.",
            "CLV must remain positive and correlated with model edge out of sample.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
