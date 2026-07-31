#!/usr/bin/env python3
"""Run the frozen NFL betting policy as a production-style historical replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.production_replay import run_production_replay  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_selector_validated_pool_2022.csv",
    )
    parser.add_argument(
        "--policy-report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "market_selector_report.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_replay_report.json",
    )
    parser.add_argument(
        "--picks",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_replay_picks.csv",
    )
    parser.add_argument(
        "--weekly-ledger",
        type=Path,
        default=NFL_ROOT / "data" / "evaluation" / "production_replay_weekly.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = pd.read_csv(args.pool, low_memory=False)
    policy_report = json.loads(args.policy_report.read_text(encoding="utf-8"))
    report, picks, weekly = run_production_replay(
        pool,
        policy_report,
        bootstrap_samples=args.bootstrap_samples,
        random_state=args.random_state,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.picks.parent.mkdir(parents=True, exist_ok=True)
    args.weekly_ledger.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    picks.to_csv(args.picks, index=False)
    weekly.to_csv(args.weekly_ledger, index=False)
    print(json.dumps({"status": report["status"], "deployment": report.get("deployment_gate")}, indent=2))
    print(f"Report: {args.report}")
    return 0 if report["status"] in {"production_ready", "effectiveness_proven_source_blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
