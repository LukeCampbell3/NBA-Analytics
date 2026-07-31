#!/usr/bin/env python3
"""Execute NFL selector training and production replay as one fail-closed pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats", type=Path, default=NFL_ROOT / "data" / "raw" / "player_stats.parquet"
    )
    parser.add_argument(
        "--development-market-rows",
        type=Path,
        default=NFL_ROOT / "tmp" / "2021" / "market_rows_edge0.csv",
    )
    parser.add_argument(
        "--final-market-rows",
        type=Path,
        default=NFL_ROOT / "tmp" / "2022" / "market_rows_edge0.csv",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=NFL_ROOT / "tmp" / "production_pipeline_validation"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_pipeline_validation_report.json",
    )
    return parser.parse_args()


def _run(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2_000:],
        "stderr_tail": completed.stderr[-2_000:],
    }


def main() -> int:
    args = parse_args()
    required = [args.stats, args.development_market_rows, args.final_market_rows]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Production validation inputs are missing: {missing}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    selector_report_path = args.work_dir / "market_selector_report.json"
    selector_artifact_path = args.work_dir / "nfl_market_selector.joblib"
    development_pool_path = args.work_dir / "candidate_pool_2021.csv"
    final_pool_path = args.work_dir / "candidate_pool_2022.csv"
    final_audit_rows_path = args.work_dir / "final_audit_rows.csv"

    selector_run = _run(
        [
            sys.executable,
            str(NFL_ROOT / "scripts" / "train_nfl_market_selector.py"),
            "--stats",
            str(args.stats),
            "--development-market-rows",
            str(args.development_market_rows),
            "--final-market-rows",
            str(args.final_market_rows),
            "--report",
            str(selector_report_path),
            "--artifact",
            str(selector_artifact_path),
            "--development-pool",
            str(development_pool_path),
            "--final-pool",
            str(final_pool_path),
            "--rows",
            str(final_audit_rows_path),
        ]
    )
    if selector_run["returncode"] != 0 or not selector_report_path.is_file():
        output = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed_closed",
            "failed_stage": "selector_training",
            "selector_run": selector_run,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
        return 2

    selector_report = json.loads(selector_report_path.read_text(encoding="utf-8"))
    validated_final_pool = selector_report_path.with_name("market_selector_validated_pool_2022.csv")
    replay_report_path = args.work_dir / "production_replay_report.json"
    replay_picks_path = args.work_dir / "production_replay_picks.csv"
    replay_weekly_path = args.work_dir / "production_replay_weekly.csv"
    replay_run = _run(
        [
            sys.executable,
            str(NFL_ROOT / "scripts" / "run_nfl_production_replay.py"),
            "--pool",
            str(validated_final_pool),
            "--policy-report",
            str(selector_report_path),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--report",
            str(replay_report_path),
            "--picks",
            str(replay_picks_path),
            "--weekly-ledger",
            str(replay_weekly_path),
        ]
    )
    replay_report = (
        json.loads(replay_report_path.read_text(encoding="utf-8"))
        if replay_report_path.is_file()
        else {}
    )
    checks = {
        "selector_completed": selector_run["returncode"] == 0,
        "passing_is_only_validated_target": selector_report.get("validated_targets")
        == ["passing"],
        "weekly_cap_locked_at_12": selector_report.get("weekly_cap_policy", {}).get(
            "selected_top_n"
        )
        == 12,
        "selector_artifact_written": selector_artifact_path.is_file()
        and selector_artifact_path.stat().st_size > 0,
        "replay_completed": replay_run["returncode"] == 0,
        "replay_contract_passed": replay_report.get("contract_gate", {}).get("status")
        == "passed",
        "replay_operational_gate_passed": replay_report.get("operational_gate", {}).get(
            "status"
        )
        == "passed",
        "replay_effectiveness_passed": replay_report.get("effectiveness", {}).get("status")
        == "passed",
        "replay_stability_passed": replay_report.get("stability_gate", {}).get("status")
        == "passed",
        "expected_final_decisions": replay_report.get("effectiveness", {})
        .get("result", {})
        .get("graded_decisions")
        == 210,
        "expected_final_wins": replay_report.get("effectiveness", {})
        .get("result", {})
        .get("wins")
        == 127,
        "source_provenance_correctly_blocks_deployment": replay_report.get(
            "deployment_gate", {}
        ).get("status")
        == "blocked",
    }
    passed = all(checks.values())
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_research_source_blocked" if passed else "failed_closed",
        "checks": checks,
        "selector_run": selector_run,
        "replay_run": replay_run,
        "selector_summary": {
            "validated_targets": selector_report.get("validated_targets"),
            "weekly_top_n": selector_report.get("weekly_cap_policy", {}).get("selected_top_n"),
            "final_test": selector_report.get("weekly_cap_policy", {}).get("final_test_result"),
        },
        "replay_summary": replay_report.get("effectiveness"),
        "deployment_gate": replay_report.get("deployment_gate"),
        "work_dir": str(args.work_dir.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checks": checks}, indent=2))
    print(f"Report: {args.report}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
