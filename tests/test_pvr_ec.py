from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sparse_loop_moe" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sparse_loop_moe.models.pvr_ec import (  # noqa: E402
    ExpertOwnershipMap,
    OwnershipBalanceController,
    OwnershipMapVersionManager,
    OwnershipRoutingConfig,
    compute_top1_oracle_gap,
    generate_balanced_assignment_targets,
    minimum_sample_protection,
    non_uniform_target_owner_share,
    route_ownership_top1,
    run_offline_ownership_replay,
    write_ownership_reports,
)


def _route_inputs(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = torch.tensor(
        [
            [1.0, 0.3, -0.2, -0.4],
            [0.1, 1.2, 0.5, -0.3],
            [0.0, 0.2, 1.0, -0.5],
            [0.2, -0.1, 0.0, 1.1],
        ],
        device=device,
    )
    prototype_bias = torch.zeros_like(router_logits)
    compatible_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, True, True, False],
            [False, True, True, True],
            [True, False, True, True],
        ],
        device=device,
    )
    proto_ids = torch.tensor([0, 1, 2, 3], device=device)
    return router_logits, prototype_bias, compatible_mask, proto_ids


def test_ownership_map_tensors_initialize_and_lookup_shape() -> None:
    ownership_map = ExpertOwnershipMap.zeros(5, 4)
    proto_ids = torch.tensor([0, 1, 2])
    candidate_experts = torch.tensor([[0, 1], [1, 2], [2, 3]])
    bias = ownership_map.get_bias(proto_ids, candidate_experts)
    assert ownership_map.ownership_reliability_bias.shape == (5, 4)
    assert bias["reliability"].shape == (3, 2)
    assert bias["failure"].shape == (3, 2)
    assert bias["monopoly"].shape == (3, 2)
    assert bias["stale"].shape == (3, 2)
    assert bias["balance"].shape == (3, 2)


def test_ownership_bias_is_clipped_and_respects_compatible_mask() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    reliability = torch.zeros(4, 4)
    reliability[0, 2] = 100.0
    reliability[0, 1] = 100.0
    ownership_map = ExpertOwnershipMap(4, 4, ownership_reliability_bias=reliability, map_mode="frozen")
    cfg = OwnershipRoutingConfig(ownership_weight=1.0, ownership_bias_cap=0.25, ownership_map_mode="frozen")
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, cfg)
    assert result.owner[0].item() == 0
    assert torch.isneginf(result.effective_score[0, 2]) or result.effective_score[0, 2] < -1e30
    assert result.ownership_bias_clipped[0, 1]


def test_balance_bias_is_clipped_and_margin_guard_logs_override_attempt() -> None:
    router_logits = torch.tensor([[5.0, 1.0, 0.0, -1.0]])
    prototype_bias = torch.zeros_like(router_logits)
    compatible_mask = torch.ones_like(router_logits, dtype=torch.bool)
    proto_ids = torch.tensor([0])
    ownership_map = ExpertOwnershipMap(
        1,
        4,
        balance_bias=torch.tensor([-10.0, 10.0, 0.0, 0.0]),
        map_mode="frozen",
    )
    cfg = OwnershipRoutingConfig(
        balance_weight=1.0,
        balance_bias_cap=10.0,
        semantic_margin_guard=0.1,
        ownership_map_mode="frozen",
    )
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, cfg)
    assert result.owner.item() == 0
    assert result.balance_bias_override_attempt.item()
    assert result.metrics["balance_bias_override_attempt_rate"].item() == pytest.approx(1.0)


def test_disabled_mode_ignores_ownership_and_balance_bias() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    ownership_map = ExpertOwnershipMap(
        4,
        4,
        ownership_reliability_bias=torch.full((4, 4), 100.0),
        balance_bias=torch.tensor([10.0, -10.0, 10.0, -10.0]),
        map_mode="disabled",
    )
    cfg = OwnershipRoutingConfig(ownership_weight=1.0, balance_weight=1.0, ownership_map_mode="disabled")
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, cfg)
    baseline = (router_logits + prototype_bias).masked_fill(~compatible_mask, torch.finfo(router_logits.dtype).min).argmax(dim=-1)
    assert torch.equal(result.owner, baseline)


def test_shadow_update_records_without_changing_route() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    ownership_map = ExpertOwnershipMap(
        4,
        4,
        ownership_reliability_bias=torch.full((4, 4), 100.0),
        balance_bias=torch.tensor([-10.0, 10.0, -10.0, 10.0]),
        map_mode="shadow_update",
    )
    cfg = OwnershipRoutingConfig(ownership_weight=1.0, balance_weight=1.0, ownership_map_mode="shadow_update")
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, cfg)
    baseline = router_logits.masked_fill(~compatible_mask, torch.finfo(router_logits.dtype).min).argmax(dim=-1)
    assert torch.equal(result.owner, baseline)


def test_frozen_mode_does_not_mutate_map_or_balance_bias() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    ownership_map = ExpertOwnershipMap(4, 4, balance_bias=torch.tensor([0.1, 0.0, 0.0, 0.0]), map_mode="frozen")
    before = ownership_map.export_bias_tensors()
    route_ownership_top1(
        router_logits,
        prototype_bias,
        compatible_mask,
        proto_ids,
        ownership_map,
        OwnershipRoutingConfig(balance_weight=1.0, ownership_map_mode="frozen"),
    )
    after = ownership_map.export_bias_tensors()
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_canary_mode_applies_candidate_map_only_in_controlled_path() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    reliability = torch.zeros(4, 4)
    reliability[0, 1] = 10.0
    ownership_map = ExpertOwnershipMap(4, 4, ownership_reliability_bias=reliability, map_mode="canary")
    result = route_ownership_top1(
        router_logits,
        prototype_bias,
        compatible_mask,
        proto_ids,
        ownership_map,
        OwnershipRoutingConfig(ownership_weight=1.0, ownership_bias_cap=5.0, ownership_map_mode="canary"),
    )
    assert result.owner[0].item() == 1


def test_each_state_gets_exactly_one_owner_and_score_challenger_not_executed() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    result = route_ownership_top1(
        router_logits,
        prototype_bias,
        compatible_mask,
        proto_ids,
        ExpertOwnershipMap.zeros(4, 4),
        OwnershipRoutingConfig(),
    )
    assert result.owner.shape == (4,)
    assert result.score_challenger.shape == (4,)
    assert torch.all(result.score_challenger.ne(result.owner))


def test_low_confidence_route_does_not_execute_top2_or_top4() -> None:
    router_logits = torch.zeros(3, 4)
    prototype_bias = torch.zeros_like(router_logits)
    compatible_mask = torch.ones_like(router_logits, dtype=torch.bool)
    proto_ids = torch.tensor([0, 1, 2])
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ExpertOwnershipMap.zeros(3, 4))
    assert result.owner.shape == (3,)
    assert result.metrics["balance_bias_changed_owner_rate"].numel() == 1


def test_route_rejects_zero_compatible_owner() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs()
    compatible_mask[0, :] = False
    with pytest.raises(ValueError, match="at least one compatible"):
        route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ExpertOwnershipMap.zeros(4, 4))


def test_map_compatibility_check_rejects_mismatched_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ownership_map_current.pt"
    ownership_map = ExpertOwnershipMap.zeros(2, 3)
    ownership_map.save_versioned(path, {"router_config_hash": "a", "prototype_table_hash": "p", "compatible_mask_hash": "c"})
    with pytest.raises(ValueError, match="PVR_EC_OWNERSHIP_MAP_COMPATIBILITY_FAILED"):
        ExpertOwnershipMap.load_versioned(path, "frozen", expected_metadata={"router_config_hash": "b"})


def test_version_manager_candidate_cannot_replace_current_without_promotion(tmp_path: Path) -> None:
    manager = OwnershipMapVersionManager(tmp_path)
    current = ExpertOwnershipMap.zeros(2, 2)
    current.save_versioned(manager.current_path, {"map_version": "current"})
    candidate = ExpertOwnershipMap(2, 2, balance_bias=torch.tensor([0.1, -0.1]))
    manager.save_candidate(candidate, {"map_version": "candidate"})
    loaded = ExpertOwnershipMap.load_versioned(manager.current_path, "frozen")
    assert torch.equal(loaded.balance_bias, torch.zeros(2))


def test_rollback_restores_previous_map(tmp_path: Path) -> None:
    manager = OwnershipMapVersionManager(tmp_path)
    previous = ExpertOwnershipMap(2, 2, balance_bias=torch.tensor([0.1, -0.1]))
    current = ExpertOwnershipMap(2, 2, balance_bias=torch.tensor([0.5, -0.5]))
    previous.save_versioned(manager.previous_path, {"map_version": "previous"})
    current.save_versioned(manager.current_path, {"map_version": "current"})
    report = manager.rollback()
    restored = ExpertOwnershipMap.load_versioned(manager.current_path, "frozen")
    assert report["rollback_completed"] is True
    assert torch.allclose(restored.balance_bias, previous.balance_bias)


def test_balance_controller_update_clips_and_non_uniform_targets() -> None:
    controller = OwnershipBalanceController(4, beta=0.0, eta=10.0, balance_bias_cap=0.1)
    target = non_uniform_target_owner_share(torch.ones(4), specialization_weight=torch.tensor([2.0, 1.0, 1.0, 0.5]))
    bias = controller.update_from_owner_share(torch.tensor([0.9, 0.05, 0.03, 0.02]), target)
    assert bias.abs().max().item() <= 0.1 + 1e-6
    assert target[0] > target[-1]


def test_offline_replay_computes_top1_oracle_gap_and_reports(tmp_path: Path) -> None:
    owner = torch.tensor([0, 1, 2])
    losses = torch.tensor([[0.4, 0.2, 0.8], [0.5, 0.3, 0.1], [0.2, 0.4, 0.6]])
    compatible = torch.ones_like(losses, dtype=torch.bool)
    gap, best = compute_top1_oracle_gap(owner, losses, compatible)
    assert torch.allclose(gap, torch.tensor([0.2, 0.2, 0.4]))
    assert torch.equal(best, torch.tensor([1, 2, 0]))
    replay = run_offline_ownership_replay(
        torch.tensor([0, 1, 1]),
        owner,
        losses,
        compatible,
        num_prototypes=2,
        num_experts=3,
        sample_count=torch.zeros(2, 3),
        output_dir=tmp_path,
    )
    assert "PVR_EC_OWNERSHIP_LOW_SAMPLE_REGION" in replay.statuses
    assert (tmp_path / "ownership_oracle_gap_report.json").exists()
    assert (tmp_path / "prototype_local_monopoly_report.json").exists()
    assert (tmp_path / "ownership_confidence_calibration_report.json").exists()


def test_offline_balanced_assignment_is_target_generation_only() -> None:
    losses = torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]])
    compatible = torch.ones_like(losses, dtype=torch.bool)
    result = generate_balanced_assignment_targets(losses, compatible, torch.tensor([0.2, 0.3, 0.5]))
    assert result["status"] == "PVR_EC_OFFLINE_BALANCED_ASSIGNMENT_ONLY"
    assert result["target_owner"].shape == (2,)


def test_minimum_sample_protection_blocks_low_sample_promotion() -> None:
    sample_count = torch.tensor([[100, 1], [100, 100]])
    allowed = minimum_sample_protection(sample_count, torch.tensor([0, 0]), torch.tensor([1, 1]), min_ownership_samples=32)
    assert torch.equal(allowed, torch.tensor([False, True]))


def test_reports_write_required_json_files(tmp_path: Path) -> None:
    write_ownership_reports(tmp_path, {"top1_oracle_gap": 0.1, "prototype_local_monopoly_rate": 0.2})
    required = [
        "ownership_map_report.json",
        "ownership_oracle_gap_report.json",
        "ownership_confidence_calibration_report.json",
        "ownership_shadow_challenger_report.json",
        "ownership_bias_diagnostics_report.json",
        "ownership_balance_bias_report.json",
        "ownership_drift_report.json",
        "ownership_replay_refresh_report.json",
        "ownership_map_canary_report.json",
        "ownership_map_rollback_report.json",
        "prototype_local_monopoly_report.json",
    ]
    for name in required:
        assert (tmp_path / name).exists()
        json.loads((tmp_path / name).read_text(encoding="utf-8"))


def test_benchmark_cli_writes_ownership_artifacts(tmp_path: Path) -> None:
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
        "fixed_moe_vectorized,pvr_ec_deploy_top1,pvr_ec_ownership_top1",
        "--enable-ownership-map",
        "--ownership-map-mode",
        "shadow_update",
        "--run-ownership-replay",
        "--ownership-probe-sample-limit",
        "8",
        "--output-dir",
        str(tmp_path),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "PVR_EC_OWNERSHIP_MAP_SHADOW_READY" in completed.stdout
    report = json.loads((tmp_path / "pvr_ec_ownership_benchmark_report.json").read_text(encoding="utf-8"))
    assert report["pvr_ec_ownership_top1"]["single_owner"] is True
    assert report["pvr_ec_ownership_top1"]["top2_executed"] is False
    assert report["pvr_ec_ownership_top1"]["top4_executed"] is False
    comparison = json.loads((tmp_path / "pvr_ec_model_comparison_metrics.json").read_text(encoding="utf-8"))
    comparison_models = {row["model"] for row in comparison}
    assert {
        "fixed_moe_vectorized",
        "pvr_ec_deploy_top1",
        "pvr_ec_ownership_top1",
    }.issubset(comparison_models)
    assert (tmp_path / "pvr_ec_model_comparison_metrics.csv").exists()
    assert (tmp_path / "pvr_ec_ownership_benchmark_report.md").read_text(encoding="utf-8").count("|") > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_route_runs_on_gpu_when_available() -> None:
    router_logits, prototype_bias, compatible_mask, proto_ids = _route_inputs("cuda")
    ownership_map = ExpertOwnershipMap.zeros(4, 4, device="cuda")
    result = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map)
    assert result.owner.device.type == "cuda"
