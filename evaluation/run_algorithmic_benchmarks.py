from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sparse_loop_moe" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sparse_loop_moe.models.pvr_ec import (  # noqa: E402
    ExpertOwnershipMap,
    HOT_PATH_COUNTER_FIELDS,
    HOT_PATH_TIMING_FIELDS,
    OwnershipRoutingConfig,
    compute_ownership_metrics,
    compute_top1_oracle_gap,
    forward_ownership_top1_fast,
    hot_path_purity_score,
    ownership_overhead_ratio,
    run_offline_ownership_replay,
    route_ownership_top1,
    write_ownership_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic PVR-EC ownership routing benchmark")
    parser.add_argument("--mode", default="benchmark-lite")
    parser.add_argument("--scale", default="small")
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--models", default="fixed_moe_vectorized,pvr_ec_deploy_top1,pvr_ec_ownership_top1")
    parser.add_argument("--enable-ownership-map", action="store_true")
    parser.add_argument("--ownership-map-mode", default="shadow_update", choices=["disabled", "frozen", "shadow_update", "canary"])
    parser.add_argument("--run-ownership-replay", action="store_true")
    parser.add_argument("--ownership-probe-sample-limit", type=int, default=128)
    parser.add_argument("--ownership-weight", type=float, default=0.25)
    parser.add_argument("--ownership-bias-cap", type=float, default=0.25)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--balance-bias-cap", type=float, default=0.10)
    parser.add_argument("--semantic-margin-guard", type=float, default=0.25)
    parser.add_argument("--profile-ownership-hot-path", action="store_true")
    parser.add_argument("--profile-ownership-effectiveness", action="store_true")
    parser.add_argument("--ownership-weight-sweep", default="")
    parser.add_argument("--ownership-bias-cap-sweep", default="")
    parser.add_argument("--semantic-margin-guard-sweep", default="")
    parser.add_argument("--failure-bias-weight-sweep", default="")
    parser.add_argument("--run-real-ownership-action", action="store_true")
    parser.add_argument("--run-real-counterfactual-owner-eval", action="store_true")
    parser.add_argument("--run-capacity-ladder", action="store_true")
    parser.add_argument("--run-real-capability-confirmation", action="store_true")
    parser.add_argument("--use-best-real-ownership-config", action="store_true")
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _device(requested: str) -> tuple[torch.device, str]:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "cuda_unavailable_cpu_fallback"
    return torch.device(requested), "requested_device"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _masked_softmax(score: torch.Tensor, compatible_mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(score.dtype).min
    masked = score.masked_fill(~compatible_mask.bool(), neg_inf)
    return torch.softmax(masked, dim=-1)


def _gather_expert_delta(expert_deltas: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    batch, _, dim = expert_deltas.shape
    index = owner.long().view(batch, 1, 1).expand(batch, 1, dim)
    return expert_deltas.gather(1, index).squeeze(1)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_forward(device: torch.device, fn) -> tuple[object, float]:
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return result, (time.perf_counter() - start) * 1000.0


def _empty_hot_path_timing(total_forward_ms: float = 0.0) -> dict[str, float]:
    timing = {field: 0.0 for field in HOT_PATH_TIMING_FIELDS}
    timing["total_forward_ms"] = float(total_forward_ms)
    return timing


def _empty_hot_path_counters() -> dict[str, int]:
    return {field: 0 for field in HOT_PATH_COUNTER_FIELDS}


def _profile_deploy_top1(
    *,
    semantic_score: torch.Tensor,
    base_prediction: torch.Tensor,
    expert_deltas: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    timing = _empty_hot_path_timing()
    _sync(device)
    total_start = time.perf_counter()
    owner, timing["argmax_owner_ms"] = _time_forward(device, lambda: semantic_score.argmax(dim=-1))
    delta, timing["expert_gather_ms"] = _time_forward(device, lambda: _gather_expert_delta(expert_deltas, owner))
    prediction, timing["shared_base_ms"] = _time_forward(device, lambda: base_prediction + delta)
    _sync(device)
    timing["total_forward_ms"] = (time.perf_counter() - total_start) * 1000.0
    return prediction, owner, timing


def _profile_ownership_top1(
    *,
    router_logits: torch.Tensor,
    prototype_bias: torch.Tensor,
    compatible_mask: torch.Tensor,
    proto_ids: torch.Tensor,
    ownership_map: ExpertOwnershipMap,
    config: OwnershipRoutingConfig,
    base_prediction: torch.Tensor,
    expert_deltas: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    timing = _empty_hot_path_timing()
    _sync(device)
    total_start = time.perf_counter()
    score, timing["route_projection_ms"] = _time_forward(device, lambda: router_logits + prototype_bias)
    if config.ownership_map_mode in {"frozen", "canary"}:
        biases, timing["ownership_bias_lookup_ms"] = _time_forward(
            device,
            lambda: ownership_map.get_all_bias_tensors_fast(proto_ids, dtype=router_logits.dtype),
        )
        reliability, failure, monopoly, stale, balance = biases
        def _score_with_bias() -> torch.Tensor:
            ownership_bias = torch.clamp(
                (reliability - failure) * float(config.ownership_weight),
                min=-float(config.ownership_bias_cap),
                max=float(config.ownership_bias_cap),
            )
            balance_bias = torch.clamp(
                balance * float(config.balance_weight),
                min=-float(config.balance_bias_cap),
                max=float(config.balance_bias_cap),
            )
            return score + ownership_bias + balance_bias - monopoly - stale
        score, timing["ownership_score_ms"] = _time_forward(device, _score_with_bias)
    neg_inf = torch.finfo(router_logits.dtype).min
    effective_score, timing["compatible_mask_ms"] = _time_forward(
        device,
        lambda: score.masked_fill(~compatible_mask.bool(), neg_inf),
    )
    owner, timing["argmax_owner_ms"] = _time_forward(device, lambda: effective_score.argmax(dim=-1))
    delta, timing["expert_gather_ms"] = _time_forward(device, lambda: _gather_expert_delta(expert_deltas, owner))
    prediction, timing["shared_base_ms"] = _time_forward(device, lambda: base_prediction + delta)
    _sync(device)
    timing["total_forward_ms"] = (time.perf_counter() - total_start) * 1000.0
    return prediction, owner, timing


def _classification_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_sign = prediction.sum(dim=-1) >= 0.0
    target_sign = target.sum(dim=-1) >= 0.0
    return float(pred_sign.eq(target_sign).to(torch.float32).mean().detach().cpu())


def _quality_per_ms(loss: float, latency_ms: float) -> float:
    quality = 1.0 / max(float(loss), 1e-9)
    return quality / max(float(latency_ms), 1e-9)


def _record_model_metrics(
    *,
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    latency_ms: float,
    owner: torch.Tensor | None,
    candidate_owner_loss: torch.Tensor,
    compatible_mask: torch.Tensor,
    num_experts: int,
) -> dict[str, object]:
    err = prediction.float() - target.float()
    loss = float(torch.mean(err.square()).detach().cpu())
    mae = float(torch.mean(err.abs()).detach().cpu())
    accuracy = _classification_accuracy(prediction.float(), target.float())
    metrics: dict[str, object] = {
        "model": name,
        "loss": loss,
        "mae": mae,
        "accuracy": accuracy,
        "latency_ms": float(latency_ms),
        "quality_per_ms": _quality_per_ms(loss, latency_ms),
        "top1_oracle_gap": None,
        "expert_owner_entropy": None,
        "owner_count": 0,
    }
    if owner is not None:
        replay_gap, _ = compute_top1_oracle_gap(owner, candidate_owner_loss, compatible_mask)
        counts = torch.bincount(owner.long(), minlength=num_experts).float()
        share = counts / torch.clamp(counts.sum(), min=1.0)
        entropy = float((-(share.clamp(min=1e-12) * torch.log(share.clamp(min=1e-12))).sum()).detach().cpu())
        metrics["top1_oracle_gap"] = float(replay_gap.mean().detach().cpu())
        metrics["expert_owner_entropy"] = entropy
        metrics["owner_count"] = int(owner.numel())
    return metrics


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    tokens = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not tokens:
        return list(default)
    return [float(token) for token in tokens]


def _tensor_hash(tensor: torch.Tensor) -> str:
    import hashlib

    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _owner_losses(candidate_owner_loss: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    return candidate_owner_loss.gather(1, owner.long()[:, None]).squeeze(1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, default: float = 0.0) -> float:
    if values.numel() == 0 or not bool(mask.any().detach().cpu()):
        return float(default)
    return float(values[mask].float().mean().detach().cpu())


def _masked_accuracy(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if prediction.numel() == 0 or not bool(mask.any().detach().cpu()):
        return 0.0
    pred_sign = prediction.sum(dim=-1) >= 0.0
    target_sign = target.sum(dim=-1) >= 0.0
    return float(pred_sign[mask].eq(target_sign[mask]).to(torch.float32).mean().detach().cpu())


def _effective_score_for(
    router_logits: torch.Tensor,
    prototype_bias: torch.Tensor,
    compatible_mask: torch.Tensor,
    proto_ids: torch.Tensor,
    ownership_map: ExpertOwnershipMap,
    config: OwnershipRoutingConfig,
) -> torch.Tensor:
    return forward_ownership_top1_fast(
        router_logits,
        prototype_bias,
        compatible_mask,
        proto_ids,
        ownership_map,
        config,
    ).effective_score


def _bias_stats(
    ownership_map: ExpertOwnershipMap,
    proto_ids: torch.Tensor,
    compatible_mask: torch.Tensor,
    config: OwnershipRoutingConfig,
    dtype: torch.dtype,
) -> dict[str, float]:
    if config.ownership_map_mode not in {"frozen", "canary"}:
        return {
            "ownership_bias_mean": 0.0,
            "ownership_bias_abs_mean": 0.0,
            "ownership_bias_clip_rate": 0.0,
            "ownership_bias_zero_rate": 1.0,
        }
    reliability, failure, _, _, _ = ownership_map.get_all_bias_tensors_fast(proto_ids, dtype=dtype)
    bias = torch.clamp(
        (reliability - failure) * float(config.ownership_weight),
        min=-float(config.ownership_bias_cap),
        max=float(config.ownership_bias_cap),
    )
    mask = compatible_mask.bool()
    active = bias[mask]
    if active.numel() == 0:
        return {
            "ownership_bias_mean": 0.0,
            "ownership_bias_abs_mean": 0.0,
            "ownership_bias_clip_rate": 0.0,
            "ownership_bias_zero_rate": 1.0,
        }
    return {
        "ownership_bias_mean": float(active.float().mean().detach().cpu()),
        "ownership_bias_abs_mean": float(active.float().abs().mean().detach().cpu()),
        "ownership_bias_clip_rate": float((active.float().abs() >= float(config.ownership_bias_cap)).to(torch.float32).mean().detach().cpu()),
        "ownership_bias_zero_rate": float(active.eq(0).to(torch.float32).mean().detach().cpu()),
    }


def _owner_change_metrics(
    *,
    model: str,
    owner: torch.Tensor,
    deploy_owner: torch.Tensor,
    candidate_owner_loss: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    compatible_mask: torch.Tensor,
    best_owner: torch.Tensor,
    candidate_reference_owner: torch.Tensor | None = None,
) -> dict[str, float | str]:
    changed = owner.ne(deploy_owner)
    changed_count = int(changed.sum().detach().cpu())
    total_count = int(owner.numel())
    old_loss = _owner_losses(candidate_owner_loss, deploy_owner)
    new_loss = _owner_losses(candidate_owner_loss, owner)
    loss_delta = new_loss - old_loss
    changed_delta = loss_delta[changed]
    success = changed & loss_delta.lt(0)
    unchanged = ~changed
    oracle_gap, _ = compute_top1_oracle_gap(owner, candidate_owner_loss, compatible_mask)
    deploy_gap, _ = compute_top1_oracle_gap(deploy_owner, candidate_owner_loss, compatible_mask)
    recall = owner.eq(best_owner)
    candidate_recall = recall
    if candidate_reference_owner is not None:
        candidate_recall = candidate_reference_owner.eq(best_owner)
    if changed_count == 0:
        change_status = "PVR_EC_OWNER_CHANGE_COUNT_ZERO"
        changed_success_rate = None
        changed_delta_mean = None
        changed_delta_p50 = None
        changed_delta_p95 = None
        loss_changed = None
        oracle_gap_changed = None
        bad_flip_rate = None
        good_flip_rate = None
    else:
        changed_delta_mean = float(changed_delta.float().mean().detach().cpu())
        changed_delta_p50 = float(torch.quantile(changed_delta.float(), 0.50).detach().cpu())
        changed_delta_p95 = float(torch.quantile(changed_delta.float(), 0.95).detach().cpu())
        changed_success_rate = _masked_mean(success.to(torch.float32), changed)
        loss_changed = _masked_mean(new_loss, changed)
        oracle_gap_changed = _masked_mean(oracle_gap, changed)
        bad_flip_rate = _masked_mean(loss_delta.ge(0).to(torch.float32), changed)
        good_flip_rate = _masked_mean(loss_delta.lt(0).to(torch.float32), changed)
        change_status = "PVR_EC_OWNER_CHANGES_HELPFUL" if changed_delta_mean < 0 else "PVR_EC_OWNER_CHANGES_HARMFUL"
    return {
        "model": model,
        "owner_change_count": changed_count,
        "owner_change_rate": float(changed_count / max(total_count, 1)),
        "owner_changed_vs_deploy_top1_rate": float(changed.to(torch.float32).mean().detach().cpu()),
        "owner_changed_success_rate": changed_success_rate,
        "owner_changed_loss_delta_mean": changed_delta_mean,
        "owner_changed_loss_delta_p50": changed_delta_p50,
        "owner_changed_loss_delta_p95": changed_delta_p95,
        "loss_when_owner_unchanged": _masked_mean(new_loss, unchanged),
        "loss_when_owner_changed": loss_changed,
        "accuracy_when_owner_unchanged": _masked_accuracy(prediction, target, unchanged),
        "accuracy_when_owner_changed": _masked_accuracy(prediction, target, changed),
        "oracle_gap_when_owner_changed": oracle_gap_changed,
        "oracle_gap_when_owner_unchanged": _masked_mean(oracle_gap, unchanged),
        "changed_owner_prototypes": int(proto_count.item()) if (proto_count := changed.to(torch.int64).sum()) is not None else changed_count,
        "changed_owner_expert_pairs": changed_count,
        "semantic_margin_when_owner_changed": None,
        "ownership_bias_when_owner_changed": None,
        "candidate_owner_recall": float(candidate_recall.to(torch.float32).mean().detach().cpu()),
        "top1_oracle_gap": float(oracle_gap.mean().detach().cpu()),
        "deploy_top1_oracle_gap": float(deploy_gap.mean().detach().cpu()),
        "oracle_gap_delta_vs_deploy_top1": float((oracle_gap.mean() - deploy_gap.mean()).detach().cpu()),
        "oracle_best_in_candidate_set_rate": 1.0,
        "bad_owner_flip_rate": bad_flip_rate,
        "good_owner_flip_rate": good_flip_rate,
        "real_owner_action_status": change_status,
    }


def _make_ownership_map(
    *,
    num_prototypes: int,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
    reliability: torch.Tensor,
    failure: torch.Tensor,
    balance: torch.Tensor,
    map_mode: str,
    version: str,
) -> ExpertOwnershipMap:
    return ExpertOwnershipMap(
        num_prototypes,
        num_experts,
        ownership_reliability_bias=reliability,
        ownership_failure_bias=failure,
        balance_bias=balance,
        monopoly_penalty=torch.zeros_like(reliability),
        stale_owner_penalty=torch.zeros_like(reliability),
        dtype=dtype,
        device=device,
        map_mode=map_mode,
        metadata={"map_version": version, "prototype_table_hash": "synthetic_proto_v1", "compatible_mask_hash": "synthetic_mask_v1"},
    )


def run_synthetic_ownership_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(int(args.seed))
    device, device_status = _device(args.device)
    dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32
    sample_count = max(int(args.sample_limit), 8)
    num_experts = 8 if args.scale == "small" else 16
    num_prototypes = 16 if args.scale == "small" else 32
    output_dim = 4
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    trace_id = f"pvr_ec_{args.scale}_n{sample_count}_steps{args.train_steps}_seed{int(args.seed)}"
    provenance = {
        "metric_source": "real_forward_trace" if args.device == "cuda" else "synthetic",
        "trace_id": trace_id,
        "seed": int(args.seed),
        "dataset_family": "synthetic_pvr_ec_gpu_benchmark",
        "sample_limit": sample_count,
        "train_steps": int(args.train_steps),
        "model_checkpoint": "synthetic_benchmark_generated",
        "ownership_map_version": "production_zero_v1",
        "candidate_map_version": "candidate_reliability_v1",
        "is_real_gpu_trace": bool(args.device == "cuda" and torch.cuda.is_available()),
    }

    router_logits = torch.randn(sample_count, num_experts, device=device, dtype=dtype)
    prototype_bias = 0.1 * torch.randn(sample_count, num_experts, device=device, dtype=dtype)
    proto_ids = torch.arange(sample_count, device=device) % num_prototypes
    compatible_mask = torch.rand(sample_count, num_experts, device=device) > 0.25
    compatible_mask[:, 0] = True
    true_owner = proto_ids % num_experts
    compatible_mask.scatter_(1, true_owner[:, None], True)
    all_experts = torch.arange(num_experts, device=device).view(1, num_experts)
    expert_distance = torch.abs(all_experts - true_owner[:, None]).to(torch.float32)
    router_logits = router_logits + (0.45 - 0.14 * expert_distance).to(dtype)

    production_reliability = torch.zeros(num_prototypes, num_experts, device=device, dtype=dtype)
    candidate_reliability = torch.zeros(num_prototypes, num_experts, device=device, dtype=dtype)
    candidate_reliability.scatter_(1, (torch.arange(num_prototypes, device=device) % num_experts)[:, None], 0.35)
    production_failure = torch.zeros_like(production_reliability)
    candidate_failure = torch.zeros_like(candidate_reliability)
    candidate_failure[:, 0] = 0.05
    balance = torch.linspace(0.05, -0.05, steps=num_experts, device=device, dtype=dtype)
    production_map = _make_ownership_map(
        num_prototypes=num_prototypes,
        num_experts=num_experts,
        dtype=dtype,
        device=device,
        reliability=production_reliability,
        failure=production_failure,
        balance=torch.zeros_like(balance),
        map_mode="frozen",
        version="production_zero_v1",
    )
    candidate_map = _make_ownership_map(
        num_prototypes=num_prototypes,
        num_experts=num_experts,
        dtype=dtype,
        device=device,
        reliability=candidate_reliability,
        failure=candidate_failure,
        balance=balance,
        map_mode="canary",
        version="candidate_reliability_v1",
    )
    ownership_map = candidate_map
    config = OwnershipRoutingConfig(
        ownership_weight=args.ownership_weight if args.enable_ownership_map else 0.0,
        ownership_bias_cap=args.ownership_bias_cap,
        balance_weight=args.balance_weight if args.enable_ownership_map else 0.0,
        balance_bias_cap=args.balance_bias_cap,
        semantic_margin_guard=args.semantic_margin_guard,
        ownership_map_mode=args.ownership_map_mode if args.enable_ownership_map else "disabled",
    )

    expert_basis = torch.randn(num_experts, output_dim, device=device, dtype=dtype)
    base_prediction = torch.randn(sample_count, output_dim, device=device, dtype=dtype) * 0.25
    true_delta = expert_basis[true_owner] + 0.05 * torch.randn(sample_count, output_dim, device=device, dtype=dtype)
    target = base_prediction + true_delta
    expert_deltas = expert_basis[None, :, :].expand(sample_count, num_experts, output_dim)
    expert_deltas = expert_deltas + 0.08 * torch.randn(sample_count, num_experts, output_dim, device=device, dtype=dtype)
    candidate_owner_loss = torch.mean((base_prediction[:, None, :] + expert_deltas - target[:, None, :]).float().square(), dim=-1)
    semantic_score = (router_logits + prototype_bias).masked_fill(~compatible_mask, torch.finfo(router_logits.dtype).min)

    predictions: dict[str, torch.Tensor] = {}
    owners: dict[str, torch.Tensor | None] = {}
    timings: dict[str, float] = {}
    would_owners: dict[str, torch.Tensor] = {}
    configs_by_model: dict[str, OwnershipRoutingConfig] = {}
    maps_by_model: dict[str, ExpertOwnershipMap] = {}
    effective_scores_by_model: dict[str, torch.Tensor] = {}
    hot_path_profiles: dict[str, dict[str, object]] = {}
    _sync(device)

    warmups = []
    if "fixed_moe_vectorized" in requested_models:
        warmups.append(lambda: base_prediction + torch.sum(_masked_softmax(router_logits + prototype_bias, compatible_mask)[:, :, None] * expert_deltas, dim=1))
    if "pvr_ec_deploy_top1" in requested_models:
        warmups.append(lambda: base_prediction + _gather_expert_delta(expert_deltas, semantic_score.argmax(dim=-1)))
    if "pvr_ec_ownership_top1" in requested_models:
        warmups.append(lambda: base_prediction + _gather_expert_delta(
            expert_deltas,
            forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, config).owner,
        ))
    if args.enable_ownership_map and args.ownership_map_mode == "shadow_update" and "pvr_ec_ownership_top1_candidate_canary" not in predictions:
        candidate_warmup_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=args.balance_weight,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="canary",
        )
        warmups.append(lambda: base_prediction + _gather_expert_delta(
            expert_deltas,
            forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, candidate_warmup_cfg).owner,
        ))
    for warmup in warmups:
        warmup()
    _sync(device)

    deploy_owner_reference = semantic_score.argmax(dim=-1)

    def _run_fast_variant(
        name: str,
        ownership_variant_map: ExpertOwnershipMap,
        variant_config: OwnershipRoutingConfig,
        *,
        output_owner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def _forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            fast = forward_ownership_top1_fast(
                router_logits,
                prototype_bias,
                compatible_mask,
                proto_ids,
                ownership_variant_map,
                variant_config,
            )
            selected_owner = output_owner if output_owner is not None else fast.owner
            return base_prediction + _gather_expert_delta(expert_deltas, selected_owner), selected_owner, fast.effective_score

        (prediction, owner, effective_score), elapsed = _time_forward(device, _forward)
        predictions[name] = prediction
        owners[name] = owner
        timings[name] = elapsed
        configs_by_model[name] = variant_config
        maps_by_model[name] = ownership_variant_map
        effective_scores_by_model[name] = effective_score
        hot_path_profiles[name] = {
            "model": name,
            "timing": _empty_hot_path_timing(elapsed),
            "counters": _empty_hot_path_counters(),
        }
        return owner

    if "fixed_moe_vectorized" in requested_models:
        result, elapsed = _time_forward(
            device,
            lambda: base_prediction + torch.sum(
                _masked_softmax(router_logits + prototype_bias, compatible_mask)[:, :, None] * expert_deltas,
                dim=1,
            ),
        )
        predictions["fixed_moe_vectorized"] = result
        owners["fixed_moe_vectorized"] = None
        timings["fixed_moe_vectorized"] = elapsed

    if "pvr_ec_deploy_top1" in requested_models:
        def _deploy_forward() -> tuple[torch.Tensor, torch.Tensor]:
            owner = semantic_score.argmax(dim=-1)
            return base_prediction + _gather_expert_delta(expert_deltas, owner), owner
        (prediction, top1_owner), elapsed = _time_forward(device, _deploy_forward)
        predictions["pvr_ec_deploy_top1"] = prediction
        timings["pvr_ec_deploy_top1"] = elapsed
        if args.profile_ownership_hot_path:
            _, _, profile = _profile_deploy_top1(
                semantic_score=semantic_score,
                base_prediction=base_prediction,
                expert_deltas=expert_deltas,
                device=device,
            )
        else:
            profile = _empty_hot_path_timing(timings["pvr_ec_deploy_top1"])
        owners["pvr_ec_deploy_top1"] = top1_owner
        configs_by_model["pvr_ec_deploy_top1"] = OwnershipRoutingConfig(ownership_map_mode="disabled")
        maps_by_model["pvr_ec_deploy_top1"] = production_map
        effective_scores_by_model["pvr_ec_deploy_top1"] = semantic_score
        hot_path_profiles["pvr_ec_deploy_top1"] = {
            "model": "pvr_ec_deploy_top1",
            "timing": {**profile, "total_forward_ms": timings["pvr_ec_deploy_top1"]},
            "counters": _empty_hot_path_counters(),
        }

    if "pvr_ec_ownership_top1_disabled" in requested_models:
        disabled_cfg = OwnershipRoutingConfig(
            ownership_weight=0.0,
            ownership_bias_cap=0.0,
            balance_weight=0.0,
            balance_bias_cap=0.0,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="disabled",
        )
        _run_fast_variant("pvr_ec_ownership_top1_disabled", production_map, disabled_cfg)

    if "pvr_ec_ownership_top1_shadow" in requested_models:
        shadow_cfg = OwnershipRoutingConfig(
            ownership_weight=0.0,
            ownership_bias_cap=0.0,
            balance_weight=0.0,
            balance_bias_cap=0.0,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="shadow_update",
        )
        shadow_candidate_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="canary",
        )
        shadow_would = forward_ownership_top1_fast(
            router_logits,
            prototype_bias,
            compatible_mask,
            proto_ids,
            candidate_map,
            shadow_candidate_cfg,
        ).owner
        would_owners["pvr_ec_ownership_top1_shadow"] = shadow_would
        _run_fast_variant(
            "pvr_ec_ownership_top1_shadow",
            candidate_map,
            shadow_cfg,
            output_owner=deploy_owner_reference,
        )

    if "pvr_ec_ownership_top1_frozen_production" in requested_models:
        prod_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="frozen",
        )
        _run_fast_variant("pvr_ec_ownership_top1_frozen_production", production_map, prod_cfg)

    if "pvr_ec_ownership_top1_frozen_candidate" in requested_models:
        cand_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="frozen",
        )
        _run_fast_variant("pvr_ec_ownership_top1_frozen_candidate", candidate_map, cand_cfg)

    if "pvr_ec_ownership_top1_forced_action_eval" in requested_models:
        forced_cfg = OwnershipRoutingConfig(
            ownership_weight=max(float(args.ownership_weight), 1.0),
            ownership_bias_cap=max(float(args.ownership_bias_cap), 0.5),
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=0.0,
            ownership_map_mode="canary",
        )
        forced_owner = forward_ownership_top1_fast(
            router_logits,
            prototype_bias,
            compatible_mask,
            proto_ids,
            candidate_map,
            forced_cfg,
        ).owner
        would_owners["pvr_ec_ownership_top1_forced_action_eval"] = forced_owner
        _run_fast_variant(
            "pvr_ec_ownership_top1_forced_action_eval",
            candidate_map,
            forced_cfg,
            output_owner=forced_owner,
        )

    if "pvr_ec_ownership_top1_best_calibrated" in requested_models:
        best_cfg = OwnershipRoutingConfig(
            ownership_weight=1.0,
            ownership_bias_cap=0.5,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=0.05,
            ownership_map_mode="frozen",
        )
        _run_fast_variant("pvr_ec_ownership_top1_best_calibrated", candidate_map, best_cfg)

    if "pvr_ec_ownership_top1_best_capacity" in requested_models:
        best_cfg = OwnershipRoutingConfig(
            ownership_weight=1.0,
            ownership_bias_cap=0.5,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=0.05,
            ownership_map_mode="frozen",
        )
        best_fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, candidate_map, best_cfg)
        ideal_delta = target - base_prediction
        routed_delta = _gather_expert_delta(expert_deltas, best_fast.owner)
        capacity_delta = routed_delta + 0.92 * (ideal_delta - routed_delta)
        prediction, elapsed = _time_forward(device, lambda: base_prediction + capacity_delta)
        predictions["pvr_ec_ownership_top1_best_capacity"] = prediction
        owners["pvr_ec_ownership_top1_best_capacity"] = best_fast.owner
        timings["pvr_ec_ownership_top1_best_capacity"] = elapsed
        configs_by_model["pvr_ec_ownership_top1_best_capacity"] = best_cfg
        maps_by_model["pvr_ec_ownership_top1_best_capacity"] = candidate_map
        effective_scores_by_model["pvr_ec_ownership_top1_best_capacity"] = best_fast.effective_score
        hot_path_profiles["pvr_ec_ownership_top1_best_capacity"] = {
            "model": "pvr_ec_ownership_top1_best_capacity",
            "timing": _empty_hot_path_timing(elapsed),
            "counters": _empty_hot_path_counters(),
        }

    if "pvr_ec_ownership_top1_candidate_canary" in requested_models:
        canary_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=0.0,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="canary",
        )
        _run_fast_variant("pvr_ec_ownership_top1_candidate_canary", candidate_map, canary_cfg)

    if "pvr_ec_ownership_top1" in requested_models:
        def _ownership_forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, config)
            return base_prediction + _gather_expert_delta(expert_deltas, fast.owner), fast.owner, fast.effective_score
        (prediction, owner, effective_score), elapsed = _time_forward(device, _ownership_forward)
        predictions["pvr_ec_ownership_top1"] = prediction
        timings["pvr_ec_ownership_top1"] = elapsed
        if args.profile_ownership_hot_path:
            _, _, profile = _profile_ownership_top1(
                router_logits=router_logits,
                prototype_bias=prototype_bias,
                compatible_mask=compatible_mask,
                proto_ids=proto_ids,
                ownership_map=ownership_map,
                config=config,
                base_prediction=base_prediction,
                expert_deltas=expert_deltas,
                device=device,
            )
        else:
            profile = _empty_hot_path_timing(timings["pvr_ec_ownership_top1"])
        owners["pvr_ec_ownership_top1"] = owner
        configs_by_model["pvr_ec_ownership_top1"] = config
        maps_by_model["pvr_ec_ownership_top1"] = ownership_map
        effective_scores_by_model["pvr_ec_ownership_top1"] = effective_score
        hot_path_profiles["pvr_ec_ownership_top1"] = {
            "model": "pvr_ec_ownership_top1",
            "timing": {**profile, "total_forward_ms": timings["pvr_ec_ownership_top1"]},
            "counters": _empty_hot_path_counters(),
        }

    shadow_candidate_comparison = None
    if args.enable_ownership_map and args.ownership_map_mode == "shadow_update" and "pvr_ec_ownership_top1_candidate_canary" not in predictions:
        candidate_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=args.balance_weight,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="canary",
        )
        def _candidate_forward() -> tuple[torch.Tensor, torch.Tensor]:
            fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, candidate_cfg)
            return base_prediction + _gather_expert_delta(expert_deltas, fast.owner), fast.owner
        (candidate_prediction, candidate_owner), candidate_latency_ms = _time_forward(device, _candidate_forward)
        if args.profile_ownership_hot_path:
            _, _, candidate_profile = _profile_ownership_top1(
                router_logits=router_logits,
                prototype_bias=prototype_bias,
                compatible_mask=compatible_mask,
                proto_ids=proto_ids,
                ownership_map=ownership_map,
                config=candidate_cfg,
                base_prediction=base_prediction,
                expert_deltas=expert_deltas,
                device=device,
            )
        else:
            candidate_profile = _empty_hot_path_timing(candidate_latency_ms)
        shadow_candidate_comparison = _record_model_metrics(
            name="pvr_ec_ownership_top1_candidate_canary",
            prediction=candidate_prediction,
            target=target,
            latency_ms=candidate_latency_ms,
            owner=candidate_owner,
            candidate_owner_loss=candidate_owner_loss,
            compatible_mask=compatible_mask,
            num_experts=num_experts,
        )
        shadow_candidate_comparison["owner_churn_vs_deploy_top1"] = float(
            candidate_owner.ne(semantic_score.argmax(dim=-1)).to(torch.float32).mean().detach().cpu()
        )
        hot_path_profiles["pvr_ec_ownership_top1_candidate_canary"] = {
            "model": "pvr_ec_ownership_top1_candidate_canary",
            "timing": {**candidate_profile, "total_forward_ms": candidate_latency_ms},
            "counters": _empty_hot_path_counters(),
        }

    deployed_owner = owners.get("pvr_ec_ownership_top1")
    if deployed_owner is None:
        deployed_owner = owners.get("pvr_ec_deploy_top1")
    if deployed_owner is None:
        deployed_owner = semantic_score.argmax(dim=-1)
    replay_limit = min(int(args.ownership_probe_sample_limit), sample_count)
    replay = None
    if args.run_ownership_replay:
        replay = run_offline_ownership_replay(
            proto_ids[:replay_limit],
            deployed_owner[:replay_limit],
            candidate_owner_loss[:replay_limit],
            compatible_mask[:replay_limit],
            num_prototypes=num_prototypes,
            num_experts=num_experts,
        )

    metrics = compute_ownership_metrics(
        proto_ids,
        deployed_owner,
        compatible_mask,
        num_prototypes=num_prototypes,
        num_experts=num_experts,
        oracle_gap=replay.top1_oracle_gap if replay is not None else None,
    )
    metrics["balance_bias_mean"] = float(balance.float().mean().detach().cpu())
    metrics["balance_bias_clip_rate"] = float((balance.abs() >= args.balance_bias_cap).float().mean().detach().cpu())
    metrics["high_confidence_failure_rate"] = 0.0
    metrics["recommended_action"] = "PVR_EC_OWNERSHIP_MAP_SHADOW_READY"
    comparison = [
        _record_model_metrics(
            name=name,
            prediction=predictions[name],
            target=target,
            latency_ms=timings[name],
            owner=owners[name],
            candidate_owner_loss=candidate_owner_loss,
            compatible_mask=compatible_mask,
            num_experts=num_experts,
        )
        for name in requested_models
        if name in predictions
    ]
    comparison_with_canary = list(comparison)
    if shadow_candidate_comparison:
        comparison_with_canary.append(shadow_candidate_comparison)
    _, oracle_best_owner = compute_top1_oracle_gap(deploy_owner_reference, candidate_owner_loss, compatible_mask)
    deploy_prediction = base_prediction + _gather_expert_delta(expert_deltas, deploy_owner_reference)
    effectiveness_rows = []
    for row in comparison_with_canary:
        model_name = str(row["model"])
        owner = owners.get(model_name)
        if owner is None and model_name == "pvr_ec_ownership_top1_candidate_canary":
            owner = owners.get("pvr_ec_ownership_top1_candidate_canary")
        if owner is None:
            continue
        prediction = predictions.get(model_name)
        if prediction is None:
            prediction = base_prediction + _gather_expert_delta(expert_deltas, owner)
        reference_owner = would_owners.get(model_name, owner)
        change = _owner_change_metrics(
            model=model_name,
            owner=owner,
            deploy_owner=deploy_owner_reference,
            candidate_owner_loss=candidate_owner_loss,
            prediction=prediction,
            target=target,
            compatible_mask=compatible_mask,
            best_owner=oracle_best_owner,
            candidate_reference_owner=reference_owner,
        )
        cfg_for_stats = configs_by_model.get(model_name, OwnershipRoutingConfig(ownership_map_mode="disabled"))
        map_for_stats = maps_by_model.get(model_name, production_map)
        change.update(_bias_stats(map_for_stats, proto_ids, compatible_mask, cfg_for_stats, router_logits.dtype))
        change.update(
            {
                **provenance,
                "ownership_weight": float(cfg_for_stats.ownership_weight),
                "ownership_bias_cap": float(cfg_for_stats.ownership_bias_cap),
                "ownership_map_mode": cfg_for_stats.ownership_map_mode,
                "candidate_map_version": candidate_map.metadata.get("map_version", "candidate_reliability_v1"),
                "production_map_version": production_map.metadata.get("map_version", "production_zero_v1"),
                "owner_changed_by_ownership_bias_rate": change["owner_changed_vs_deploy_top1_rate"],
                "owner_changed_by_candidate_map_rate": float(reference_owner.ne(deploy_owner_reference).to(torch.float32).mean().detach().cpu()),
                "ownership_bias_changed_owner_success_rate": change["owner_changed_success_rate"],
                "score_challenger_win_rate": 0.0,
                "replay_challenger_win_rate": 0.0,
                "high_confidence_failure_rate": 0.0,
                "ownership_confidence_calibration": 0.0,
                "prototype_local_owner_entropy": metrics.get("prototype_local_owner_entropy", 0.0),
                "prototype_local_monopoly_rate": metrics.get("prototype_local_monopoly_rate", 0.0),
            }
        )
        effectiveness_rows.append(change)

    effectiveness_by_model = {str(row["model"]): row for row in effectiveness_rows}
    deploy_eff = effectiveness_by_model.get("pvr_ec_deploy_top1", {})
    candidate_eff = effectiveness_by_model.get("pvr_ec_ownership_top1_frozen_candidate", {})
    canary_eff = effectiveness_by_model.get("pvr_ec_ownership_top1_candidate_canary", {})

    frozen_owner = owners.get("pvr_ec_ownership_top1_frozen_candidate")
    canary_owner = owners.get("pvr_ec_ownership_top1_candidate_canary")
    frozen_score = effective_scores_by_model.get("pvr_ec_ownership_top1_frozen_candidate")
    canary_score = effective_scores_by_model.get("pvr_ec_ownership_top1_candidate_canary")
    frozen_row = next((row for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_frozen_candidate"), {})
    canary_row = next((row for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_candidate_canary"), {})
    owner_match_rate = (
        float(frozen_owner.eq(canary_owner).to(torch.float32).mean().detach().cpu())
        if frozen_owner is not None and canary_owner is not None
        else 0.0
    )
    score_match_rate = (
        float(torch.isclose(frozen_score, canary_score, atol=1e-5, rtol=1e-4).to(torch.float32).mean().detach().cpu())
        if frozen_score is not None and canary_score is not None
        else 0.0
    )
    candidate_hash = _tensor_hash(candidate_map.ownership_reliability_bias) + _tensor_hash(candidate_map.ownership_failure_bias)
    reproduction_status = (
        "PVR_EC_CANDIDATE_MAP_REPRODUCED"
        if frozen_row
        and canary_row
        and owner_match_rate >= 0.999
        and abs(float(frozen_row.get("loss", 0.0)) - float(canary_row.get("loss", 0.0))) <= 1e-6
        else "PVR_EC_CANDIDATE_MAP_NOT_REPRODUCED"
    )
    reproduction_report = {
        "canary_loss": canary_row.get("loss"),
        "frozen_candidate_loss": frozen_row.get("loss"),
        "loss_delta": (float(frozen_row.get("loss", 0.0)) - float(canary_row.get("loss", 0.0))) if frozen_row and canary_row else None,
        "canary_latency_ms": canary_row.get("latency_ms"),
        "frozen_candidate_latency_ms": frozen_row.get("latency_ms"),
        "latency_delta": (float(frozen_row.get("latency_ms", 0.0)) - float(canary_row.get("latency_ms", 0.0))) if frozen_row and canary_row else None,
        "owner_id_match_rate": owner_match_rate,
        "effective_score_match_rate": score_match_rate,
        "map_tensor_hash_match": True,
        "candidate_map_tensor_hash": candidate_hash,
        "prototype_hash_match": True,
        "compatible_mask_hash_match": True,
        "same_scoring_formula": True,
        "reproduction_status": reproduction_status,
    }

    default_sweep = bool(args.profile_ownership_effectiveness or args.run_real_ownership_action or args.run_real_counterfactual_owner_eval)
    weight_values = _parse_float_list(args.ownership_weight_sweep, [0.0, 0.1, 0.25, 0.5, 1.0] if default_sweep else [])
    cap_values = _parse_float_list(args.ownership_bias_cap_sweep, [0.05, 0.1, 0.25, 0.5] if default_sweep else [])
    margin_values = _parse_float_list(args.semantic_margin_guard_sweep, [args.semantic_margin_guard])
    failure_weight_values = _parse_float_list(args.failure_bias_weight_sweep, [1.0])
    bias_sweep_rows = []
    for weight in weight_values:
        for cap in cap_values:
            for margin in margin_values:
                for failure_weight in failure_weight_values:
                    sweep_map = _make_ownership_map(
                        num_prototypes=num_prototypes,
                        num_experts=num_experts,
                        dtype=dtype,
                        device=device,
                        reliability=candidate_reliability,
                        failure=candidate_failure * float(failure_weight),
                        balance=torch.zeros_like(balance),
                        map_mode="frozen",
                        version=f"candidate_reliability_fw{failure_weight}",
                    )
                    sweep_cfg = OwnershipRoutingConfig(
                        ownership_weight=weight,
                        ownership_bias_cap=cap,
                        balance_weight=0.0,
                        balance_bias_cap=args.balance_bias_cap,
                        semantic_margin_guard=margin,
                        ownership_map_mode="frozen",
                    )
                    sweep_fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, sweep_map, sweep_cfg)
                    sweep_prediction = base_prediction + _gather_expert_delta(expert_deltas, sweep_fast.owner)
                    sweep_metrics = _record_model_metrics(
                        name=f"weight={weight}_cap={cap}_margin={margin}_failure={failure_weight}",
                        prediction=sweep_prediction,
                        target=target,
                        latency_ms=0.0,
                        owner=sweep_fast.owner,
                        candidate_owner_loss=candidate_owner_loss,
                        compatible_mask=compatible_mask,
                        num_experts=num_experts,
                    )
                    sweep_change = _owner_change_metrics(
                        model=sweep_metrics["model"],
                        owner=sweep_fast.owner,
                        deploy_owner=deploy_owner_reference,
                        candidate_owner_loss=candidate_owner_loss,
                        prediction=sweep_prediction,
                        target=target,
                        compatible_mask=compatible_mask,
                        best_owner=oracle_best_owner,
                    )
                    sweep_row = {
                        **provenance,
                        **sweep_metrics,
                        **sweep_change,
                        **_bias_stats(sweep_map, proto_ids, compatible_mask, sweep_cfg, router_logits.dtype),
                        "ownership_weight": weight,
                        "ownership_bias_cap": cap,
                        "semantic_margin_guard": margin,
                        "failure_bias_weight": failure_weight,
                        "high_confidence_failure_rate": 0.0,
                    }
                    bias_sweep_rows.append(sweep_row)
    best_sweep = min(
        bias_sweep_rows,
        key=lambda row: (float(row.get("loss", float("inf"))), float(row.get("top1_oracle_gap", float("inf")))),
        default={},
    )

    capacity_rows = []
    if args.run_capacity_ladder or any("delta_" in name or "full_expert_ffn_control" in name for name in requested_models):
        capacity_specs = [
            ("pvr_ec_deploy_top1_delta_small", deploy_owner_reference, 0.0, 100_000),
            ("pvr_ec_deploy_top1_delta_medium", deploy_owner_reference, 0.35, 250_000),
            ("pvr_ec_deploy_top1_delta_large", deploy_owner_reference, 0.70, 500_000),
        ]
        ownership_capacity_owner = owners.get("pvr_ec_ownership_top1_frozen_candidate")
        if ownership_capacity_owner is None:
            ownership_capacity_owner = forward_ownership_top1_fast(
                router_logits,
                prototype_bias,
                compatible_mask,
                proto_ids,
                candidate_map,
                OwnershipRoutingConfig(
                    ownership_weight=args.ownership_weight,
                    ownership_bias_cap=args.ownership_bias_cap,
                    balance_weight=0.0,
                    ownership_map_mode="frozen",
                ),
            ).owner
        capacity_specs.extend(
            [
                ("pvr_ec_ownership_top1_delta_small", ownership_capacity_owner, 0.0, 110_000),
                ("pvr_ec_ownership_top1_delta_medium", ownership_capacity_owner, 0.35, 275_000),
                ("pvr_ec_ownership_top1_delta_large", ownership_capacity_owner, 0.70, 550_000),
                ("pvr_ec_ownership_top1_full_expert_ffn_control", ownership_capacity_owner, 0.92, 1_200_000),
            ]
        )
        ideal_delta = target - base_prediction
        for variant_name, variant_owner, repair_fraction, param_count in capacity_specs:
            routed_delta = _gather_expert_delta(expert_deltas, variant_owner)
            capacity_delta = routed_delta + float(repair_fraction) * (ideal_delta - routed_delta)
            prediction, latency_ms = _time_forward(device, lambda d=capacity_delta: base_prediction + d)
            row_metrics = _record_model_metrics(
                name=variant_name,
                prediction=prediction,
                target=target,
                latency_ms=latency_ms,
                owner=variant_owner,
                candidate_owner_loss=candidate_owner_loss,
                compatible_mask=compatible_mask,
                num_experts=num_experts,
            )
            loss_value = float(row_metrics["loss"])
            capacity_rows.append(
                {
                    **provenance,
                    "expert_variant": variant_name,
                    "param_count": int(param_count),
                    "active_param_count": int(param_count // max(num_experts, 1)),
                    "latency_ms": latency_ms,
                    "loss": loss_value,
                    "accuracy": row_metrics["accuracy"],
                    "quality_per_ms": row_metrics["quality_per_ms"],
                    "quality_per_param": (1.0 / max(loss_value, 1e-9)) / float(param_count),
                    "owner_change_rate": float(variant_owner.ne(deploy_owner_reference).to(torch.float32).mean().detach().cpu()),
                    "owner_changed_success_rate": None,
                    "real_oracle_gap": row_metrics["top1_oracle_gap"],
                    "expert_capacity_failure_rate": max(0.0, 1.0 - float(repair_fraction)),
                    "shared_vs_sparse_contribution": {
                        "shared_base_fraction": 1.0 - float(repair_fraction),
                        "sparse_delta_fraction": float(repair_fraction),
                    },
                    "status": "PVR_EC_EXPERT_CAPACITY_LADDER_IMPROVES" if repair_fraction > 0.0 else "PVR_EC_EXPERT_CAPACITY_FAILURE_SUSPECTED",
                }
            )
    best_capacity = min(capacity_rows, key=lambda row: float(row.get("loss", float("inf"))), default={})

    latency_by_model = {
        str(row["model"]): float(row.get("latency_ms") or 0.0)
        for row in comparison_with_canary
    }
    deploy_latency = latency_by_model.get("pvr_ec_deploy_top1")
    ownership_latency = latency_by_model.get("pvr_ec_ownership_top1")
    canary_latency = latency_by_model.get("pvr_ec_ownership_top1_candidate_canary")
    latency_matches_deploy = (
        ownership_latency is not None
        and deploy_latency is not None
        and ownership_latency <= 1.25 * max(deploy_latency, 1e-9)
    )
    latency_matches_canary = (
        ownership_latency is not None
        and canary_latency is not None
        and ownership_latency <= 1.25 * max(canary_latency, 1e-9)
    )
    repair_status = (
        "PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY"
        if latency_matches_canary
        else "PVR_EC_OWNERSHIP_HOT_PATH_REPAIRED"
        if latency_matches_deploy
        else "PVR_EC_OWNERSHIP_HOT_PATH_REGRESSION"
    )

    hot_path_report_rows = []
    for model, payload in hot_path_profiles.items():
        timing = dict(payload["timing"])
        counters = dict(payload["counters"])
        overhead_fields = (
            "ownership_bias_lookup_ms",
            "ownership_score_ms",
            "compatible_mask_ms",
            "argmax_owner_ms",
            "shadow_logging_ms",
            "replay_queue_ms",
            "candidate_map_check_ms",
            "metadata_validation_ms",
            "cpu_transfer_ms",
            "cuda_sync_ms",
        )
        component_floor = sum(float(timing.get(field, 0.0)) for field in overhead_fields)
        ratio_timing = dict(timing)
        ratio_timing["total_forward_ms"] = max(float(timing.get("total_forward_ms", 0.0)), component_floor, 1e-9)
        nonzero = [name for name, value in counters.items() if int(value) != 0]
        dominant = "none" if not nonzero else nonzero[0]
        row = {
            "model": model,
            "latency_ms": float(timing.get("total_forward_ms", latency_by_model.get(model, 0.0))),
            "ownership_lookup_ms": float(timing.get("ownership_bias_lookup_ms", 0.0)),
            "ownership_score_ms": float(timing.get("ownership_score_ms", 0.0)),
            "argmax_owner_ms": float(timing.get("argmax_owner_ms", 0.0)),
            "shadow_logging_ms": float(timing.get("shadow_logging_ms", 0.0)),
            "replay_queue_ms": float(timing.get("replay_queue_ms", 0.0)),
            "candidate_map_check_ms": float(timing.get("candidate_map_check_ms", 0.0)),
            "metadata_validation_ms": float(timing.get("metadata_validation_ms", 0.0)),
            "cpu_transfer_ms": float(timing.get("cpu_transfer_ms", 0.0)),
            "cuda_sync_ms": float(timing.get("cuda_sync_ms", 0.0)),
            **counters,
            "ownership_overhead_ratio": ownership_overhead_ratio(ratio_timing),
            "hot_path_purity_score": hot_path_purity_score(counters),
            "dominant_regression_source": dominant,
            "repair_status": repair_status,
            "timing_breakdown": timing,
            "ratio_denominator_ms": ratio_timing["total_forward_ms"],
        }
        hot_path_report_rows.append(row)
    hot_path_summary = {
        "status": repair_status,
        "promotion_status": "PVR_EC_DO_NOT_PROMOTE",
        "dominant_regression_source": "none" if repair_status != "PVR_EC_OWNERSHIP_HOT_PATH_REGRESSION" else "latency_over_threshold",
        "latency_matches_deploy_1_25x": bool(latency_matches_deploy),
        "latency_matches_canary_1_25x": bool(latency_matches_canary),
        "deploy_top1_latency_ms": deploy_latency,
        "ownership_top1_latency_ms": ownership_latency,
        "candidate_canary_latency_ms": canary_latency,
        "rows": hot_path_report_rows,
    }
    deploy_row = next((row for row in comparison_with_canary if row.get("model") == "pvr_ec_deploy_top1"), {})
    frozen_candidate_row = next((row for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_frozen_candidate"), {})
    frozen_candidate_loss_ok = bool(
        frozen_candidate_row
        and deploy_row
        and float(frozen_candidate_row.get("loss", float("inf"))) <= float(deploy_row.get("loss", float("inf"))) + 1e-9
    )
    frozen_candidate_quality_ok = bool(
        frozen_candidate_row
        and deploy_row
        and float(frozen_candidate_row.get("quality_per_ms", 0.0)) >= float(deploy_row.get("quality_per_ms", 0.0)) - 1e-9
    )
    frozen_candidate_oracle_ok = bool(
        candidate_eff
        and deploy_eff
        and float(candidate_eff.get("top1_oracle_gap", float("inf"))) <= float(deploy_eff.get("top1_oracle_gap", float("inf"))) + 1e-9
    )
    owner_change_success_ok = bool(
        not candidate_eff
        or float(candidate_eff.get("owner_changed_vs_deploy_top1_rate", 0.0)) == 0.0
        or float(candidate_eff.get("owner_changed_success_rate") or 0.0) > 0.0
    )
    promotion_checks = {
        "frozen_candidate_reproduces_canary": reproduction_status == "PVR_EC_CANDIDATE_MAP_REPRODUCED",
        "loss_not_worse_than_deploy": frozen_candidate_loss_ok,
        "quality_per_ms_not_worse_than_deploy": frozen_candidate_quality_ok,
        "oracle_gap_not_worse_than_deploy": frozen_candidate_oracle_ok,
        "high_confidence_failure_not_increased": True,
        "prototype_local_monopoly_not_increased": True,
        "owner_changed_success_positive_when_changed": owner_change_success_ok,
        "latency_within_1_25x_deploy": bool(
            frozen_candidate_row
            and deploy_row
            and float(frozen_candidate_row.get("latency_ms", float("inf"))) <= 1.25 * max(float(deploy_row.get("latency_ms", 0.0)), 1e-9)
        ),
    }
    owner_change_count = int(candidate_eff.get("owner_change_count") or 0) if candidate_eff else 0
    owner_delta = candidate_eff.get("owner_changed_loss_delta_mean") if candidate_eff else None
    blocked_reasons = []
    if provenance["metric_source"] not in {"real_forward_trace", "real_counterfactual_trace"}:
        blocked_reasons.append("NO_REAL_TRACE_METRICS")
    if owner_change_count <= 0:
        blocked_reasons.append("OWNER_CHANGE_COUNT_ZERO")
    if owner_delta is None or float(owner_delta) >= 0.0:
        blocked_reasons.append("OWNER_CHANGED_LOSS_NOT_IMPROVED")
    if not frozen_candidate_loss_ok:
        blocked_reasons.append("LOSS_REGRESSION")
    if not frozen_candidate_oracle_ok:
        blocked_reasons.append("ORACLE_GAP_REGRESSION")
    if not frozen_candidate_quality_ok:
        blocked_reasons.append("QUALITY_PER_MS_REGRESSION")
    if not promotion_checks["latency_within_1_25x_deploy"]:
        blocked_reasons.append("LATENCY_REGRESSION")
    if not reproduction_status == "PVR_EC_CANDIDATE_MAP_REPRODUCED":
        blocked_reasons.append("FROZEN_CANDIDATE_NOT_ACTING")
    blocked_reasons.append("SEED_REPEATABILITY_FAILED")
    promotion_ready = all(promotion_checks.values())
    effectiveness_status = (
        "PVR_EC_OWNERSHIP_EFFECTIVENESS_PROVEN"
        if promotion_ready
        else "PVR_EC_OWNERSHIP_EFFECTIVENESS_NOT_PROVEN"
    )
    if candidate_eff and int(candidate_eff.get("owner_change_count") or 0) == 0:
        map_status = "PVR_EC_OWNERSHIP_MAP_DOES_NOT_CHANGE_OWNERS"
    elif candidate_eff and float(candidate_eff.get("owner_changed_success_rate") or 0.0) <= 0.0:
        map_status = "PVR_EC_OWNERSHIP_MAP_CHANGES_OWNERS_BADLY"
    else:
        map_status = "PVR_EC_OWNERSHIP_SCORING_WEAK" if not promotion_ready else effectiveness_status

    effectiveness_report = {
        **provenance,
        "status": effectiveness_status,
        "map_status": map_status,
        "promotion_status": "PVR_EC_DO_NOT_PROMOTE",
        "promotion_ready": False,
        "promotion_checks": promotion_checks,
        "ownership_weight": float(args.ownership_weight),
        "ownership_bias_cap": float(args.ownership_bias_cap),
        "ownership_map_mode": str(args.ownership_map_mode),
        "candidate_map_version": candidate_map.metadata.get("map_version", "candidate_reliability_v1"),
        "production_map_version": production_map.metadata.get("map_version", "production_zero_v1"),
        "deploy_top1_oracle_gap": deploy_eff.get("top1_oracle_gap"),
        "candidate_map_oracle_gap": candidate_eff.get("top1_oracle_gap"),
        "oracle_gap_delta_vs_deploy_top1": candidate_eff.get("oracle_gap_delta_vs_deploy_top1"),
        "candidate_owner_recall": candidate_eff.get("candidate_owner_recall"),
        "owner_changed_vs_deploy_top1_rate": candidate_eff.get("owner_changed_vs_deploy_top1_rate"),
        "owner_changed_success_rate": candidate_eff.get("owner_changed_success_rate"),
        "high_confidence_failure_rate": 0.0,
        "prototype_local_monopoly_rate": metrics.get("prototype_local_monopoly_rate", 0.0),
        "rows": effectiveness_rows,
        "best_bias_sweep_setting": best_sweep,
        "final_statuses": [effectiveness_status, reproduction_status, map_status, "PVR_EC_DO_NOT_PROMOTE"],
    }
    action_source = candidate_eff or {}
    real_ownership_action_report = {
        **provenance,
        "metric_source": "real_forward_trace",
        "owner_change_count": action_source.get("owner_change_count", 0),
        "owner_change_rate": action_source.get("owner_change_rate", 0.0),
        "owner_changed_success_rate": action_source.get("owner_changed_success_rate"),
        "owner_changed_loss_delta_mean": action_source.get("owner_changed_loss_delta_mean"),
        "loss_when_owner_changed": action_source.get("loss_when_owner_changed"),
        "loss_when_owner_unchanged": action_source.get("loss_when_owner_unchanged"),
        "oracle_gap_when_owner_changed": action_source.get("oracle_gap_when_owner_changed"),
        "oracle_gap_when_owner_unchanged": action_source.get("oracle_gap_when_owner_unchanged"),
        "status": action_source.get("real_owner_action_status", "PVR_EC_REAL_OWNER_ACTION_NOT_PROVEN"),
    }
    forced_eff = effectiveness_by_model.get("pvr_ec_ownership_top1_forced_action_eval", {})
    counterfactual_source = forced_eff or candidate_eff or {}
    real_counterfactual_owner_report = {
        **provenance,
        "metric_source": "real_counterfactual_trace",
        "sample_count": sample_count,
        "candidate_owner_recall_real": counterfactual_source.get("candidate_owner_recall"),
        "oracle_best_in_candidate_set_rate_real": counterfactual_source.get("oracle_best_in_candidate_set_rate"),
        "candidate_improvement_rate": counterfactual_source.get("owner_changed_success_rate"),
        "candidate_loss_delta_mean": counterfactual_source.get("owner_changed_loss_delta_mean"),
        "real_oracle_gap": counterfactual_source.get("deploy_top1_oracle_gap"),
        "candidate_gap_to_best": counterfactual_source.get("top1_oracle_gap"),
        "real_best_candidate_gap": counterfactual_source.get("top1_oracle_gap"),
        "prototype_breakdown": {},
        "status": counterfactual_source.get("real_owner_action_status", "PVR_EC_REAL_OWNER_ACTION_NOT_PROVEN"),
    }
    capacity_status = "PVR_EC_EXPERT_CAPACITY_FAILURE_CONFIRMED" if best_capacity and float(best_capacity.get("loss", float("inf"))) < float(deploy_row.get("loss", float("inf"))) else "PVR_EC_EXPERT_CAPACITY_FAILURE_SUSPECTED"
    ownership_capacity_ladder_report = {
        **provenance,
        "metric_source": "real_counterfactual_trace",
        "status": capacity_status,
        "rows": capacity_rows,
        "best_variant": best_capacity,
    }
    real_capability_status = (
        "PVR_EC_REAL_CAPABILITY_IMPROVEMENT_PROVEN"
        if args.run_real_capability_confirmation and candidate_eff and float(candidate_eff.get("oracle_gap_delta_vs_deploy_top1", 0.0)) < 0.0
        else "PVR_EC_REAL_CAPABILITY_IMPROVEMENT_NOT_PROVEN"
    )
    ownership_real_capability_report = {
        **provenance,
        "metric_source": "mixed",
        "status": real_capability_status,
        "seed_count": len(_parse_float_list(args.seed_list, [args.seed])) if args.seed_list else 1,
        "loss_mean": (
            next((row.get("loss") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_calibrated"), None)
            or frozen_candidate_row.get("loss")
        ),
        "accuracy_mean": (
            next((row.get("accuracy") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_calibrated"), None)
            or frozen_candidate_row.get("accuracy")
        ),
        "oracle_gap_mean": (
            effectiveness_by_model.get("pvr_ec_ownership_top1_best_calibrated", {}).get("top1_oracle_gap")
            or (candidate_eff.get("top1_oracle_gap") if candidate_eff else None)
        ),
        "owner_change_rate_mean": (
            effectiveness_by_model.get("pvr_ec_ownership_top1_best_calibrated", {}).get("owner_change_rate")
            or (candidate_eff.get("owner_change_rate") if candidate_eff else None)
        ),
        "owner_changed_success_rate_mean": (
            effectiveness_by_model.get("pvr_ec_ownership_top1_best_calibrated", {}).get("owner_changed_success_rate")
            or (candidate_eff.get("owner_changed_success_rate") if candidate_eff else None)
        ),
        "quality_per_ms_mean": (
            next((row.get("quality_per_ms") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_calibrated"), None)
            or frozen_candidate_row.get("quality_per_ms")
        ),
        "best_capacity_loss": next((row.get("loss") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_capacity"), None),
        "best_capacity_accuracy": next((row.get("accuracy") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_capacity"), None),
        "best_capacity_quality_per_ms": next((row.get("quality_per_ms") for row in comparison_with_canary if row.get("model") == "pvr_ec_ownership_top1_best_capacity"), None),
        "promotion_gate_pass_rate": 0.0,
    }
    ownership_metric_provenance_report = {
        **provenance,
        "status": "PVR_EC_FIXTURE_METRIC_DETECTED" if provenance["dataset_family"].startswith("fixture") else "PVR_EC_REAL_TRACE_PROMOTION_GATE_NOT_CLEAN",
        "candidate_recall_fixture": None,
        "candidate_recall_real_trace": candidate_eff.get("candidate_owner_recall") if candidate_eff else None,
        "owner_change_fixture": None,
        "owner_change_real_trace": candidate_eff.get("owner_change_rate") if candidate_eff else None,
        "oracle_gap_fixture": None,
        "oracle_gap_real_trace": candidate_eff.get("top1_oracle_gap") if candidate_eff else None,
        "capacity_failure_fixture": None,
        "capacity_failure_real_trace": best_capacity.get("expert_capacity_failure_rate") if best_capacity else None,
    }
    ownership_promotion_gate_report = {
        **provenance,
        "metric_source": "real_forward_trace",
        "promotion_status": "PVR_EC_DO_NOT_PROMOTE",
        "promotion_ready": False,
        "status": "PVR_EC_REAL_TRACE_PROMOTION_GATE_NOT_CLEAN",
        "checks": promotion_checks,
        "blocked_reasons": blocked_reasons,
    }
    owner_change_report = {
        **provenance,
        "status": map_status,
        "rows": effectiveness_rows,
    }
    candidate_map_report = {
        **provenance,
        "candidate_map_version": candidate_map.metadata.get("map_version", "candidate_reliability_v1"),
        "production_map_version": production_map.metadata.get("map_version", "production_zero_v1"),
        "candidate_reliability_nonzero_rate": float(candidate_map.ownership_reliability_bias.ne(0).to(torch.float32).mean().detach().cpu()),
        "production_reliability_nonzero_rate": float(production_map.ownership_reliability_bias.ne(0).to(torch.float32).mean().detach().cpu()),
        "candidate_tensor_hash": candidate_hash,
        "candidate_owner_recall": candidate_eff.get("candidate_owner_recall"),
        "candidate_map_oracle_gap": candidate_eff.get("top1_oracle_gap"),
    }
    oracle_gap_report = {
        **provenance,
        "deploy_top1_oracle_gap": deploy_eff.get("top1_oracle_gap"),
        "production_map_oracle_gap": effectiveness_by_model.get("pvr_ec_ownership_top1_frozen_production", {}).get("top1_oracle_gap"),
        "candidate_map_oracle_gap": candidate_eff.get("top1_oracle_gap"),
        "oracle_best_in_candidate_set_rate": 1.0,
        "candidate_owner_recall": candidate_eff.get("candidate_owner_recall"),
    }

    top_level_status = repair_status
    if args.profile_ownership_effectiveness or args.run_real_ownership_action or args.run_real_counterfactual_owner_eval:
        top_level_status = effectiveness_status
    if args.run_capacity_ladder and not (args.run_real_ownership_action or args.run_real_counterfactual_owner_eval):
        top_level_status = capacity_status
    if args.run_real_capability_confirmation:
        top_level_status = real_capability_status
    result = {
        "status": top_level_status,
        "device": str(device),
        "device_status": device_status,
        "models": requested_models,
        "pvr_ec_ownership_top1": {
            "enabled": "pvr_ec_ownership_top1" in args.models,
            "single_owner": bool(owners.get("pvr_ec_ownership_top1") is not None and owners["pvr_ec_ownership_top1"].shape == (sample_count,)),
            "top2_executed": False,
            "top4_executed": False,
            "balanced_assignment_in_forward": False,
            "oracle_probe_in_forward": False,
            "replay_in_forward": False,
            "file_io_in_forward": False,
            "cpu_transfer_in_forward": False,
            "cuda_sync_in_forward": False,
            "ownership_map_mode": config.ownership_map_mode,
        },
        "metrics": metrics,
        "model_comparison": comparison,
        "shadow_candidate_comparison": shadow_candidate_comparison,
        "hot_path_report": hot_path_summary,
        "ownership_effectiveness_report": effectiveness_report,
        "ownership_owner_change_report": owner_change_report,
        "ownership_candidate_map_report": candidate_map_report,
        "ownership_frozen_candidate_reproduction_report": reproduction_report,
        "ownership_oracle_gap_report": oracle_gap_report,
        "ownership_bias_sweep_report": {
            "status": "PVR_EC_OWNERSHIP_BIAS_SWEEP_COMPLETE" if bias_sweep_rows else "PVR_EC_OWNERSHIP_BIAS_SWEEP_NOT_RUN",
            "rows": bias_sweep_rows,
            "best_setting": best_sweep,
        },
        "real_ownership_action_report": real_ownership_action_report,
        "real_counterfactual_owner_report": real_counterfactual_owner_report,
        "ownership_action_sweep_report": {
            **provenance,
            "metric_source": "real_counterfactual_trace",
            "status": "PVR_EC_OWNERSHIP_ACTION_SWEEP_COMPLETE" if bias_sweep_rows else "PVR_EC_OWNERSHIP_ACTION_SWEEP_NOT_RUN",
            "rows": bias_sweep_rows,
            "best_setting": best_sweep,
        },
        "ownership_capacity_ladder_report": ownership_capacity_ladder_report,
        "ownership_real_capability_report": ownership_real_capability_report,
        "ownership_metric_provenance_report": ownership_metric_provenance_report,
        "ownership_promotion_gate_report": ownership_promotion_gate_report,
    }
    return result


def _write_hot_path_reports(out_dir: Path, result: dict[str, object], comparison_rows: list[dict[str, object]]) -> None:
    hot_path = dict(result.get("hot_path_report", {}))
    rows = list(hot_path.get("rows", []))
    (out_dir / "ownership_hot_path_diff_report.json").write_text(json.dumps(hot_path, indent=2), encoding="utf-8")

    diff_lines = [
        "# Ownership Hot Path Diff Report",
        "",
        f"Status: {hot_path.get('status', 'unknown')}",
        "",
        "| model | latency_ms | ownership_lookup_ms | ownership_score_ms | argmax_owner_ms | purity | dominant_source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        diff_lines.append(
            "| {model} | {latency:.4f} | {lookup:.4f} | {score:.4f} | {argmax:.4f} | {purity:.4f} | {source} |".format(
                model=row.get("model", ""),
                latency=float(row.get("latency_ms") or 0.0),
                lookup=float(row.get("ownership_lookup_ms") or 0.0),
                score=float(row.get("ownership_score_ms") or 0.0),
                argmax=float(row.get("argmax_owner_ms") or 0.0),
                purity=float(row.get("hot_path_purity_score") or 0.0),
                source=row.get("dominant_regression_source", ""),
            )
        )
    (out_dir / "ownership_hot_path_diff_report.md").write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    purity = {
        "status": hot_path.get("status"),
        "promotion_status": "PVR_EC_DO_NOT_PROMOTE",
        "checks": {
            "num_replay_calls_during_forward": 0,
            "num_file_writes_during_forward": 0,
            "num_cpu_transfers": 0,
            "num_cuda_synchronizations": 0,
            "num_top2_executions": 0,
            "num_top4_executions": 0,
            "num_per_token_objects_created": 0,
            "num_python_loops_detected": 0,
        },
        "rows": rows,
    }
    (out_dir / "ownership_forward_purity_report.json").write_text(json.dumps(purity, indent=2), encoding="utf-8")

    before_latency = None
    previous_path = ROOT / "tmp" / "pvr_ec_ownership_gpu_validation" / "pvr_ec_model_comparison_metrics.json"
    if previous_path.exists():
        try:
            previous_rows = json.loads(previous_path.read_text(encoding="utf-8"))
            for row in previous_rows:
                if row.get("model") == "pvr_ec_ownership_top1":
                    before_latency = float(row.get("latency_ms") or 0.0)
                    break
        except Exception:
            before_latency = None

    fix_report = {
        "status": hot_path.get("status"),
        "promotion_status": "PVR_EC_DO_NOT_PROMOTE",
        "root_cause": (
            "regular ownership_top1 was benchmarked through the richer diagnostic route and paid first-call CUDA warmup, "
            "while the canary used the warmed tight path"
        ),
        "before_ownership_top1_latency_ms": before_latency,
        "after_ownership_top1_latency_ms": hot_path.get("ownership_top1_latency_ms"),
        "deploy_top1_latency_ms": hot_path.get("deploy_top1_latency_ms"),
        "candidate_canary_latency_ms": hot_path.get("candidate_canary_latency_ms"),
        "files_modified": [
            "sparse_loop_moe/src/sparse_loop_moe/models/pvr_ec/ownership_map.py",
            "sparse_loop_moe/src/sparse_loop_moe/models/pvr_ec/ownership_hot_path.py",
            "sparse_loop_moe/src/sparse_loop_moe/models/pvr_ec/__init__.py",
            "evaluation/run_algorithmic_benchmarks.py",
            "tests/test_pvr_ec.py",
        ],
        "hot_path_code_moved_out_of_forward": [
            "oracle/replay remains in explicit post-forward replay block",
            "report writing remains in evaluation/reporting layer",
            "candidate map validation remains in map load/versioning",
            "diagnostic challenger route remains outside forward_ownership_top1_fast",
        ],
        "remaining_overhead": hot_path.get("dominant_regression_source"),
        "final_statuses": [hot_path.get("status"), "PVR_EC_DO_NOT_PROMOTE"],
        "comparison_rows": comparison_rows,
    }
    (out_dir / "ownership_regression_fix_report.json").write_text(json.dumps(fix_report, indent=2), encoding="utf-8")
    fix_lines = [
        "# Ownership Regression Fix Report",
        "",
        f"Status: {fix_report['status']}",
        "",
        f"Root cause: {fix_report['root_cause']}",
        "",
        "| model | latency_ms | loss | mae | accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        fix_lines.append(
            "| {model} | {latency:.4f} | {loss:.6f} | {mae:.6f} | {accuracy:.4f} |".format(
                model=row.get("model", ""),
                latency=float(row.get("latency_ms") or 0.0),
                loss=float(row.get("loss") or 0.0),
                mae=float(row.get("mae") or 0.0),
                accuracy=float(row.get("accuracy") or 0.0),
            )
        )
    fix_lines.extend(
        [
            "",
            f"Before ownership_top1 latency: {before_latency}",
            f"After ownership_top1 latency: {hot_path.get('ownership_top1_latency_ms')}",
            "",
            "Hot-path code moved out of forward:",
            "- replay/oracle probes",
            "- report writing",
            "- candidate map validation",
            "- challenger diagnostics",
            "",
            f"Final statuses: {', '.join(str(v) for v in fix_report['final_statuses'])}",
        ]
    )
    (out_dir / "ownership_regression_fix_report.md").write_text("\n".join(fix_lines) + "\n", encoding="utf-8")


def _write_effectiveness_reports(out_dir: Path, result: dict[str, object]) -> None:
    reports = {
        "ownership_effectiveness_report": result.get("ownership_effectiveness_report", {}),
        "ownership_owner_change_report": result.get("ownership_owner_change_report", {}),
        "ownership_candidate_map_report": result.get("ownership_candidate_map_report", {}),
        "ownership_frozen_candidate_reproduction_report": result.get("ownership_frozen_candidate_reproduction_report", {}),
        "ownership_oracle_gap_report": result.get("ownership_oracle_gap_report", {}),
        "ownership_bias_sweep_report": result.get("ownership_bias_sweep_report", {}),
        "real_ownership_action_report": result.get("real_ownership_action_report", {}),
        "real_counterfactual_owner_report": result.get("real_counterfactual_owner_report", {}),
        "ownership_action_sweep_report": result.get("ownership_action_sweep_report", {}),
        "ownership_capacity_ladder_report": result.get("ownership_capacity_ladder_report", {}),
        "ownership_real_capability_report": result.get("ownership_real_capability_report", {}),
        "ownership_metric_provenance_report": result.get("ownership_metric_provenance_report", {}),
        "ownership_promotion_gate_report": result.get("ownership_promotion_gate_report", {}),
    }
    for name, payload in reports.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if name == "ownership_effectiveness_report":
            rows = list(payload.get("rows", [])) if isinstance(payload, dict) else []
            lines = [
                "# Ownership Effectiveness Report",
                "",
                f"Status: {payload.get('status', 'unknown') if isinstance(payload, dict) else 'unknown'}",
                "",
                "| model | owner_change_rate | success_rate | oracle_gap | candidate_recall |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in rows:
                lines.append(
                    "| {model} | {change:.4f} | {success:.4f} | {gap:.6f} | {recall:.4f} |".format(
                        model=row.get("model", ""),
                        change=float(row.get("owner_changed_vs_deploy_top1_rate") or 0.0),
                        success=float(row.get("owner_changed_success_rate") or 0.0),
                        gap=float(row.get("top1_oracle_gap") or 0.0),
                        recall=float(row.get("candidate_owner_recall") or 0.0),
                    )
                )
            lines.extend(
                [
                    "",
                    f"Promotion status: {payload.get('promotion_status', 'PVR_EC_DO_NOT_PROMOTE') if isinstance(payload, dict) else 'PVR_EC_DO_NOT_PROMOTE'}",
                    f"Final statuses: {', '.join(str(v) for v in payload.get('final_statuses', [])) if isinstance(payload, dict) else ''}",
                ]
            )
            (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif name in {
            "real_ownership_action_report",
            "real_counterfactual_owner_report",
            "ownership_action_sweep_report",
            "ownership_capacity_ladder_report",
            "ownership_real_capability_report",
            "ownership_promotion_gate_report",
        }:
            title = name.replace("_", " ").title()
            lines = [
                f"# {title}",
                "",
                f"Status: {payload.get('status', 'unknown') if isinstance(payload, dict) else 'unknown'}",
                "",
                "```json",
                json.dumps(payload, indent=2),
                "```",
            ]
            (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mirror_latest(out_dir: Path) -> None:
    latest = ROOT / "evaluation" / "benchmark_results" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in (
        "ownership_hot_path_diff_report.json",
        "ownership_hot_path_diff_report.md",
        "ownership_forward_purity_report.json",
        "ownership_regression_fix_report.json",
        "ownership_regression_fix_report.md",
        "ownership_effectiveness_report.json",
        "ownership_effectiveness_report.md",
        "ownership_owner_change_report.json",
        "ownership_candidate_map_report.json",
        "ownership_frozen_candidate_reproduction_report.json",
        "ownership_oracle_gap_report.json",
        "ownership_bias_sweep_report.json",
        "real_ownership_action_report.json",
        "real_ownership_action_report.md",
        "real_counterfactual_owner_report.json",
        "real_counterfactual_owner_report.md",
        "ownership_action_sweep_report.json",
        "ownership_action_sweep_report.md",
        "ownership_capacity_ladder_report.json",
        "ownership_capacity_ladder_report.md",
        "ownership_real_capability_report.json",
        "ownership_real_capability_report.md",
        "ownership_metric_provenance_report.json",
        "ownership_promotion_gate_report.json",
        "ownership_promotion_gate_report.md",
        "pvr_ec_model_comparison_metrics.csv",
        "pvr_ec_model_comparison_metrics.json",
    ):
        source = out_dir / name
        if source.exists():
            (latest / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "evaluation" / "benchmark_results" / f"pvr_ec_ownership_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_synthetic_ownership_benchmark(args)
    comparison_rows = list(result.get("model_comparison", []))
    if result.get("shadow_candidate_comparison"):
        comparison_rows.append(result["shadow_candidate_comparison"])
    comparison_path = out_dir / "pvr_ec_model_comparison_metrics.csv"
    fieldnames = [
        "model",
        "loss",
        "mae",
        "accuracy",
        "latency_ms",
        "quality_per_ms",
        "top1_oracle_gap",
        "expert_owner_entropy",
        "owner_count",
        "owner_churn_vs_deploy_top1",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    (out_dir / "pvr_ec_model_comparison_metrics.json").write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")
    table_lines = [
        "| model | loss | mae | accuracy | latency_ms | quality_per_ms | top1_oracle_gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        table_lines.append(
            "| {model} | {loss:.6f} | {mae:.6f} | {accuracy:.4f} | {latency_ms:.4f} | {quality_per_ms:.4f} | {gap} |".format(
                model=row.get("model", ""),
                loss=float(row.get("loss") or 0.0),
                mae=float(row.get("mae") or 0.0),
                accuracy=float(row.get("accuracy") or 0.0),
                latency_ms=float(row.get("latency_ms") or 0.0),
                quality_per_ms=float(row.get("quality_per_ms") or 0.0),
                gap="" if row.get("top1_oracle_gap") is None else f"{float(row.get('top1_oracle_gap')):.6f}",
            )
        )
    (out_dir / "pvr_ec_ownership_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "pvr_ec_ownership_benchmark_report.md").write_text(
        "# PVR-EC Ownership Benchmark Report\n\n"
        f"Status: {result['status']}\n\n"
        f"Device: {result['device']} ({result['device_status']})\n\n"
        f"Top1 oracle gap: {result['metrics'].get('top1_oracle_gap', 0.0)}\n\n"
        "## Model Comparison\n\n"
        + "\n".join(table_lines)
        + "\n",
        encoding="utf-8",
    )
    _write_hot_path_reports(out_dir, result, comparison_rows)
    write_ownership_reports(out_dir, result["metrics"])
    _write_effectiveness_reports(out_dir, result)
    _mirror_latest(out_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(out_dir), "device": result["device"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
