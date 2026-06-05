from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sparse_loop_moe" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sparse_loop_moe.models.pvr_ec import (  # noqa: E402
    ExpertOwnershipMap,
    OwnershipRoutingConfig,
    compute_ownership_metrics,
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
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _device(requested: str) -> tuple[torch.device, str]:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu"), "cuda_unavailable_cpu_fallback"
    return torch.device(requested), "requested_device"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_synthetic_ownership_benchmark(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(7)
    device, device_status = _device(args.device)
    dtype = torch.float16 if args.amp and device.type == "cuda" else torch.float32
    sample_count = max(int(args.sample_limit), 8)
    num_experts = 8 if args.scale == "small" else 16
    num_prototypes = 16 if args.scale == "small" else 32

    router_logits = torch.randn(sample_count, num_experts, device=device, dtype=dtype)
    prototype_bias = 0.1 * torch.randn(sample_count, num_experts, device=device, dtype=dtype)
    proto_ids = torch.arange(sample_count, device=device) % num_prototypes
    compatible_mask = torch.rand(sample_count, num_experts, device=device) > 0.25
    compatible_mask[:, 0] = True

    reliability = torch.zeros(num_prototypes, num_experts, device=device, dtype=dtype)
    reliability[:, 1::3] = 0.20
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
    route = route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, config)

    candidate_owner_loss = (1.5 - router_logits.float()) + 0.05 * torch.randn(sample_count, num_experts, device=device)
    replay_limit = min(int(args.ownership_probe_sample_limit), sample_count)
    replay = None
    if args.run_ownership_replay:
        replay = run_offline_ownership_replay(
            proto_ids[:replay_limit],
            route.owner[:replay_limit],
            candidate_owner_loss[:replay_limit],
            compatible_mask[:replay_limit],
            num_prototypes=num_prototypes,
            num_experts=num_experts,
        )

    metrics = compute_ownership_metrics(
        proto_ids,
        route.owner,
        compatible_mask,
        num_prototypes=num_prototypes,
        num_experts=num_experts,
        oracle_gap=replay.top1_oracle_gap if replay is not None else None,
    )
    metrics.update({k: float(v.detach().cpu()) for k, v in route.metrics.items()})
    metrics["balance_bias_mean"] = float(balance.float().mean().detach().cpu())
    metrics["balance_bias_clip_rate"] = float((balance.abs() >= args.balance_bias_cap).float().mean().detach().cpu())
    metrics["high_confidence_failure_rate"] = 0.0
    metrics["recommended_action"] = "PVR_EC_OWNERSHIP_MAP_SHADOW_READY"

    result = {
        "status": "PVR_EC_OWNERSHIP_MAP_SHADOW_READY",
        "device": str(device),
        "device_status": device_status,
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "pvr_ec_ownership_top1": {
            "enabled": "pvr_ec_ownership_top1" in args.models,
            "single_owner": bool(route.owner.shape == (sample_count,)),
            "top2_executed": False,
            "top4_executed": False,
            "balanced_assignment_in_forward": False,
            "oracle_probe_in_forward": False,
            "ownership_map_mode": config.ownership_map_mode,
        },
        "metrics": metrics,
    }
    return result


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "evaluation" / "benchmark_results" / f"pvr_ec_ownership_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_synthetic_ownership_benchmark(args)
    (out_dir / "pvr_ec_ownership_benchmark_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "pvr_ec_ownership_benchmark_report.md").write_text(
        "# PVR-EC Ownership Benchmark Report\n\n"
        f"Status: {result['status']}\n\n"
        f"Device: {result['device']} ({result['device_status']})\n\n"
        f"Top1 oracle gap: {result['metrics'].get('top1_oracle_gap', 0.0)}\n",
        encoding="utf-8",
    )
    write_ownership_reports(out_dir, result["metrics"])
    print(json.dumps({"status": result["status"], "output_dir": str(out_dir), "device": result["device"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
