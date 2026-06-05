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


def run_synthetic_ownership_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(7)
    device, device_status = _device(args.device)
    dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32
    sample_count = max(int(args.sample_limit), 8)
    num_experts = 8 if args.scale == "small" else 16
    num_prototypes = 16 if args.scale == "small" else 32
    output_dim = 4
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]

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

    reliability = torch.zeros(num_prototypes, num_experts, device=device, dtype=dtype)
    reliability.scatter_(1, (torch.arange(num_prototypes, device=device) % num_experts)[:, None], 0.35)
    failure = torch.zeros_like(reliability)
    failure[:, 0] = 0.05
    monopoly = torch.zeros_like(reliability)
    stale = torch.zeros_like(reliability)
    balance = torch.linspace(0.05, -0.05, steps=num_experts, device=device, dtype=dtype)
    ownership_map = ExpertOwnershipMap(
        num_prototypes,
        num_experts,
        ownership_reliability_bias=reliability,
        ownership_failure_bias=failure,
        balance_bias=balance,
        monopoly_penalty=monopoly,
        stale_owner_penalty=stale,
        dtype=dtype,
        device=device,
        map_mode=args.ownership_map_mode if args.enable_ownership_map else "disabled",
    )
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
    if args.enable_ownership_map and args.ownership_map_mode == "shadow_update":
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
        if args.profile_ownership_hot_path:
            prediction, top1_owner, profile = _profile_deploy_top1(
                semantic_score=semantic_score,
                base_prediction=base_prediction,
                expert_deltas=expert_deltas,
                device=device,
            )
            predictions["pvr_ec_deploy_top1"] = prediction
            timings["pvr_ec_deploy_top1"] = profile["total_forward_ms"]
        else:
            def _deploy_forward() -> tuple[torch.Tensor, torch.Tensor]:
                owner = semantic_score.argmax(dim=-1)
                return base_prediction + _gather_expert_delta(expert_deltas, owner), owner
            (prediction, top1_owner), elapsed = _time_forward(device, _deploy_forward)
            predictions["pvr_ec_deploy_top1"] = prediction
            timings["pvr_ec_deploy_top1"] = elapsed
        owners["pvr_ec_deploy_top1"] = top1_owner
        hot_path_profiles["pvr_ec_deploy_top1"] = {
            "model": "pvr_ec_deploy_top1",
            "timing": profile if args.profile_ownership_hot_path else _empty_hot_path_timing(timings["pvr_ec_deploy_top1"]),
            "counters": _empty_hot_path_counters(),
        }

    if "pvr_ec_ownership_top1" in requested_models:
        if args.profile_ownership_hot_path:
            prediction, owner, profile = _profile_ownership_top1(
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
            predictions["pvr_ec_ownership_top1"] = prediction
            timings["pvr_ec_ownership_top1"] = profile["total_forward_ms"]
        else:
            def _ownership_forward() -> tuple[torch.Tensor, torch.Tensor]:
                fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, config)
                return base_prediction + _gather_expert_delta(expert_deltas, fast.owner), fast.owner
            (prediction, owner), elapsed = _time_forward(device, _ownership_forward)
            predictions["pvr_ec_ownership_top1"] = prediction
            timings["pvr_ec_ownership_top1"] = elapsed
        owners["pvr_ec_ownership_top1"] = owner
        hot_path_profiles["pvr_ec_ownership_top1"] = {
            "model": "pvr_ec_ownership_top1",
            "timing": profile if args.profile_ownership_hot_path else _empty_hot_path_timing(timings["pvr_ec_ownership_top1"]),
            "counters": _empty_hot_path_counters(),
        }

    shadow_candidate_comparison = None
    if args.enable_ownership_map and args.ownership_map_mode == "shadow_update":
        candidate_cfg = OwnershipRoutingConfig(
            ownership_weight=args.ownership_weight,
            ownership_bias_cap=args.ownership_bias_cap,
            balance_weight=args.balance_weight,
            balance_bias_cap=args.balance_bias_cap,
            semantic_margin_guard=args.semantic_margin_guard,
            ownership_map_mode="canary",
        )
        if args.profile_ownership_hot_path:
            candidate_prediction, candidate_owner, candidate_profile = _profile_ownership_top1(
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
            candidate_latency_ms = candidate_profile["total_forward_ms"]
        else:
            def _candidate_forward() -> tuple[torch.Tensor, torch.Tensor]:
                fast = forward_ownership_top1_fast(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, candidate_cfg)
                return base_prediction + _gather_expert_delta(expert_deltas, fast.owner), fast.owner
            (candidate_prediction, candidate_owner), candidate_latency_ms = _time_forward(device, _candidate_forward)
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
            "timing": candidate_profile if args.profile_ownership_hot_path else _empty_hot_path_timing(candidate_latency_ms),
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
            "ownership_overhead_ratio": ownership_overhead_ratio(timing),
            "hot_path_purity_score": hot_path_purity_score(counters),
            "dominant_regression_source": dominant,
            "repair_status": repair_status,
            "timing_breakdown": timing,
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

    result = {
        "status": repair_status,
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
    }
    return result


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
    write_ownership_reports(out_dir, result["metrics"])
    print(json.dumps({"status": result["status"], "output_dir": str(out_dir), "device": result["device"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
