from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sports.nfl.predictions.production_replay import (
    apply_locked_policy,
    grade_sides,
    run_production_replay,
    selection_fingerprint,
    validate_pool_contract,
)


NFL_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = NFL_ROOT / "data" / "evaluation" / "market_selector_validated_pool_2022.csv"
POLICY_PATH = NFL_ROOT / "data" / "evaluation" / "market_selector_report.json"


def _inputs() -> tuple[pd.DataFrame, dict]:
    return (
        pd.read_csv(POOL_PATH, low_memory=False),
        json.loads(POLICY_PATH.read_text(encoding="utf-8")),
    )


def test_production_replay_passes_operational_effectiveness_and_stability_gates() -> None:
    pool, policy = _inputs()
    report, picks, weekly = run_production_replay(
        pool, policy, bootstrap_samples=1_000, random_state=42
    )
    assert report["status"] == "effectiveness_proven_source_blocked"
    assert report["contract_gate"]["status"] == "passed"
    assert report["operational_gate"]["status"] == "passed"
    assert report["effectiveness"]["status"] == "passed"
    assert report["stability_gate"]["status"] == "passed"
    assert report["source_provenance_gate"]["status"] == "failed"
    assert report["effectiveness"]["result"]["graded_decisions"] == 210
    assert report["effectiveness"]["result"]["wins"] == 127
    assert report["effectiveness"]["result"]["hit_rate"] == 0.6048
    assert len(picks) == 210
    assert len(weekly) == 18


def test_selection_is_deterministic_and_does_not_read_outcomes() -> None:
    pool, policy = _inputs()
    expected = apply_locked_policy(pool, policy)
    changed = pool.sample(frac=1.0, random_state=17).copy()
    changed["actual"] = -999.0
    changed["result"] = "corrupted"
    changed["profit_units"] = 999.0
    selected = apply_locked_policy(changed, policy)
    assert selection_fingerprint(selected) == selection_fingerprint(expected)


def test_contract_fails_closed_on_schema_duplicate_threshold_and_target_errors() -> None:
    pool, policy = _inputs()
    assert validate_pool_contract(pool.iloc[0:0], policy)["errors"] == ["empty_pool"]
    missing = validate_pool_contract(pool.drop(columns=["line"]), policy)
    assert missing["status"] == "failed"
    assert "missing_columns:line" in missing["errors"]

    duplicate = pd.concat([pool, pool.iloc[[0]]], ignore_index=True)
    assert validate_pool_contract(duplicate, policy)["status"] == "failed"

    below_threshold = pool.copy()
    below_threshold.loc[below_threshold.index[0], "estimated_side_probability"] = 0.51
    assert "side_probability_below_policy" in validate_pool_contract(
        below_threshold, policy
    )["errors"]

    wrong_target = pool.copy()
    wrong_target.loc[wrong_target.index[0], "target"] = "receiving"
    assert "unvalidated_targets:receiving" in validate_pool_contract(
        wrong_target, policy
    )["errors"]

    wrong_price = pool.copy()
    wrong_price.loc[wrong_price.index[0], "selected_price"] = 999
    assert "selected_price_side_mismatch" in validate_pool_contract(
        wrong_price, policy
    )["errors"]

    wrong_architecture = pool.copy()
    wrong_architecture.loc[wrong_architecture.index[0], "selected_architecture"] = "leaky"
    assert "selected_architecture_mismatch" in validate_pool_contract(
        wrong_architecture, policy
    )["errors"]


def test_replay_regrades_instead_of_trusting_stored_results() -> None:
    pool, policy = _inputs()
    selected = apply_locked_policy(pool, policy).copy()
    selected["result"] = "win"
    selected["profit_units"] = 99.0
    regraded = grade_sides(selected, selected["side"])
    assert int(regraded["result"].eq("win").sum()) == 127
    assert round(float(regraded["profit_units"].sum()), 4) == 27.2981


def test_production_replay_cli_writes_report_picks_and_weekly_ledger(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    picks_path = tmp_path / "picks.csv"
    weekly_path = tmp_path / "weekly.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(NFL_ROOT / "scripts" / "run_nfl_production_replay.py"),
            "--pool",
            str(POOL_PATH),
            "--policy-report",
            str(POLICY_PATH),
            "--bootstrap-samples",
            "500",
            "--report",
            str(report_path),
            "--picks",
            str(picks_path),
            "--weekly-ledger",
            str(weekly_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["operational_gate"][
        "status"
    ] == "passed"
    assert len(pd.read_csv(picks_path)) == 210
    assert len(pd.read_csv(weekly_path)) == 18
