from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_ownership_reports_and_gate_are_written(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "run_algorithmic_benchmarks.py"),
        "--mode",
        "benchmark-lite",
        "--scale",
        "small",
        "--sample-limit",
        "32",
        "--train-steps",
        "1",
        "--device",
        "cpu",
        "--models",
        "pvr_ec_deploy_top1,pvr_ec_ownership_top1_disabled,pvr_ec_ownership_top1_shadow,pvr_ec_ownership_top1_frozen_candidate,pvr_ec_ownership_top1_forced_action_eval",
        "--enable-ownership-map",
        "--ownership-map-mode",
        "canary",
        "--run-real-ownership-action",
        "--run-real-counterfactual-owner-eval",
        "--run-capacity-ladder",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)

    action = json.loads((tmp_path / "real_ownership_action_report.json").read_text(encoding="utf-8"))
    assert action["metric_source"] == "real_forward_trace"
    assert "owner_change_count" in action
    if action["owner_change_count"] == 0:
        assert action["owner_changed_success_rate"] is None

    counterfactual = json.loads((tmp_path / "real_counterfactual_owner_report.json").read_text(encoding="utf-8"))
    assert counterfactual["metric_source"] == "real_counterfactual_trace"
    assert "candidate_loss_delta_mean" in counterfactual

    sweep = json.loads((tmp_path / "ownership_action_sweep_report.json").read_text(encoding="utf-8"))
    assert sweep["rows"]
    assert "owner_change_rate" in sweep["rows"][0]

    capacity = json.loads((tmp_path / "ownership_capacity_ladder_report.json").read_text(encoding="utf-8"))
    assert capacity["rows"]
    assert {row["expert_variant"] for row in capacity["rows"]} >= {
        "pvr_ec_deploy_top1_delta_small",
        "pvr_ec_ownership_top1_delta_large",
    }

    gate = json.loads((tmp_path / "ownership_promotion_gate_report.json").read_text(encoding="utf-8"))
    assert gate["promotion_status"] == "PVR_EC_DO_NOT_PROMOTE"
    assert gate["blocked_reasons"]


def test_fixture_or_non_gpu_metrics_cannot_promote(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "run_algorithmic_benchmarks.py"),
        "--sample-limit",
        "16",
        "--train-steps",
        "1",
        "--device",
        "cpu",
        "--models",
        "pvr_ec_deploy_top1,pvr_ec_ownership_top1_frozen_candidate",
        "--enable-ownership-map",
        "--ownership-map-mode",
        "canary",
        "--run-real-ownership-action",
        "--output-dir",
        str(tmp_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    gate = json.loads((tmp_path / "ownership_promotion_gate_report.json").read_text(encoding="utf-8"))
    assert gate["promotion_ready"] is False
    assert "NO_REAL_TRACE_METRICS" in gate["blocked_reasons"]
