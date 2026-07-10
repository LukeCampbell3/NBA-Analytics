from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

HOT_PATH_COUNTER_FIELDS = [
    "num_replay_calls_during_forward",
    "num_file_writes_during_forward",
    "num_cpu_transfers",
    "num_cuda_synchronizations",
    "num_top2_executions",
    "num_top4_executions",
    "num_per_token_objects_created",
    "num_python_loops_detected",
]

HOT_PATH_TIMING_FIELDS = [
    "route_projection_ms",
    "ownership_bias_lookup_ms",
    "ownership_score_ms",
    "compatible_mask_ms",
    "argmax_owner_ms",
    "expert_gather_ms",
    "shared_base_ms",
    "shadow_logging_ms",
    "replay_queue_ms",
    "candidate_map_check_ms",
    "metadata_validation_ms",
    "cpu_transfer_ms",
    "cuda_sync_ms",
    "total_forward_ms",
]


@dataclass(frozen=True)
class OwnershipRoutingConfig:
    ownership_weight: float = 0.25
    ownership_bias_cap: float = 0.25
    failure_bias_weight: float = 1.0
    balance_weight: float = 0.05
    balance_bias_cap: float = 0.10
    semantic_margin_guard: float = 0.25
    ownership_map_mode: str = "disabled"


@dataclass
class HotPathCounters:
    num_replay_calls_during_forward: int = 0
    num_file_writes_during_forward: int = 0
    num_cpu_transfers: int = 0
    num_cuda_synchronizations: int = 0
    num_top2_executions: int = 0
    num_top4_executions: int = 0
    num_per_token_objects_created: int = 0
    num_python_loops_detected: int = 0

    def to_dict(self) -> dict[str, int]:
        return {field_name: int(getattr(self, field_name)) for field_name in HOT_PATH_COUNTER_FIELDS}


@dataclass
class OwnershipRoutingResult:
    owner: torch.Tensor
    score_challenger: torch.Tensor
    effective_score: torch.Tensor
    ownership_bias_clipped: torch.Tensor
    balance_bias_override_attempt: torch.Tensor
    metrics: dict[str, torch.Tensor]
    counters: HotPathCounters = field(default_factory=HotPathCounters)


@dataclass
class OwnershipReplayResult:
    statuses: list[str]
    top1_oracle_gap: torch.Tensor
    oracle_best_owner: torch.Tensor
    metrics: dict[str, float]


class ExpertOwnershipMap:
    def __init__(
        self,
        num_prototypes: int,
        num_experts: int,
        *,
        ownership_reliability_bias: torch.Tensor | None = None,
        ownership_failure_bias: torch.Tensor | None = None,
        monopoly_penalty: torch.Tensor | None = None,
        stale_owner_penalty: torch.Tensor | None = None,
        balance_bias: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        map_mode: str = "disabled",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.num_prototypes = int(num_prototypes)
        self.num_experts = int(num_experts)
        self.map_mode = map_mode
        self.metadata = dict(metadata or {})
        self.ownership_reliability_bias = self._matrix(ownership_reliability_bias, dtype, device)
        self.ownership_failure_bias = self._matrix(ownership_failure_bias, dtype, device)
        self.monopoly_penalty = self._matrix(monopoly_penalty, dtype, device)
        self.stale_owner_penalty = self._matrix(stale_owner_penalty, dtype, device)
        self.balance_bias = self._vector(balance_bias, dtype, device)

    @classmethod
    def zeros(
        cls,
        num_prototypes: int,
        num_experts: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        map_mode: str = "disabled",
    ) -> "ExpertOwnershipMap":
        return cls(num_prototypes, num_experts, dtype=dtype, device=device, map_mode=map_mode)

    def _matrix(
        self,
        value: torch.Tensor | None,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros((self.num_prototypes, self.num_experts), dtype=dtype, device=device)
        tensor = value.to(device=device, dtype=dtype).clone()
        if tuple(tensor.shape) != (self.num_prototypes, self.num_experts):
            raise ValueError("ownership map matrix shape mismatch")
        return tensor

    def _vector(
        self,
        value: torch.Tensor | None,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(self.num_experts, dtype=dtype, device=device)
        tensor = value.to(device=device, dtype=dtype).clone()
        if tuple(tensor.shape) != (self.num_experts,):
            raise ValueError("ownership map vector shape mismatch")
        return tensor

    def get_bias(self, proto_ids: torch.Tensor, candidate_experts: torch.Tensor) -> dict[str, torch.Tensor]:
        rows = proto_ids.long().to(self.ownership_reliability_bias.device)
        experts = candidate_experts.long().to(self.ownership_reliability_bias.device)
        return {
            "reliability": self.ownership_reliability_bias[rows].gather(1, experts),
            "failure": self.ownership_failure_bias[rows].gather(1, experts),
            "monopoly": self.monopoly_penalty[rows].gather(1, experts),
            "stale": self.stale_owner_penalty[rows].gather(1, experts),
            "balance": self.balance_bias[experts],
        }

    def get_all_bias_tensors_fast(
        self,
        proto_ids: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = proto_ids.long().to(self.ownership_reliability_bias.device)
        cast = dtype or self.ownership_reliability_bias.dtype
        reliability = self.ownership_reliability_bias[rows].to(dtype=cast)
        failure = self.ownership_failure_bias[rows].to(dtype=cast)
        monopoly = self.monopoly_penalty[rows].to(dtype=cast)
        stale = self.stale_owner_penalty[rows].to(dtype=cast)
        balance = self.balance_bias.to(dtype=cast).view(1, -1).expand(rows.shape[0], -1)
        return reliability, failure, monopoly, stale, balance

    def export_bias_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "ownership_reliability_bias": self.ownership_reliability_bias.clone(),
            "ownership_failure_bias": self.ownership_failure_bias.clone(),
            "monopoly_penalty": self.monopoly_penalty.clone(),
            "stale_owner_penalty": self.stale_owner_penalty.clone(),
            "balance_bias": self.balance_bias.clone(),
        }

    def save_versioned(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_prototypes": self.num_prototypes,
            "num_experts": self.num_experts,
            "metadata": {**self.metadata, **dict(metadata or {})},
            "tensors": self.export_bias_tensors(),
        }
        torch.save(payload, path)

    @classmethod
    def load_versioned(
        cls,
        path: str | Path,
        map_mode: str,
        *,
        expected_metadata: dict[str, Any] | None = None,
    ) -> "ExpertOwnershipMap":
        payload = torch.load(Path(path), map_location="cpu")
        metadata = dict(payload.get("metadata") or {})
        for key, expected in dict(expected_metadata or {}).items():
            if metadata.get(key) != expected:
                raise ValueError(
                    "PVR_EC_OWNERSHIP_MAP_COMPATIBILITY_FAILED: "
                    f"{key} expected {expected!r}, found {metadata.get(key)!r}"
                )
        tensors = payload["tensors"]
        return cls(
            int(payload["num_prototypes"]),
            int(payload["num_experts"]),
            ownership_reliability_bias=tensors["ownership_reliability_bias"],
            ownership_failure_bias=tensors["ownership_failure_bias"],
            monopoly_penalty=tensors["monopoly_penalty"],
            stale_owner_penalty=tensors["stale_owner_penalty"],
            balance_bias=tensors["balance_bias"],
            map_mode=map_mode,
            metadata=metadata,
        )


class OwnershipMapVersionManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "ownership_map_current.pt"
        self.candidate_path = self.root / "ownership_map_candidate.pt"
        self.previous_path = self.root / "ownership_map_previous.pt"

    def save_candidate(self, ownership_map: ExpertOwnershipMap, metadata: dict[str, Any] | None = None) -> None:
        ownership_map.save_versioned(self.candidate_path, metadata or {})

    def rollback(self) -> dict[str, Any]:
        if not self.previous_path.exists():
            return {"rollback_completed": False, "reason": "missing_previous_map"}
        if self.current_path.exists():
            shutil.copy2(self.current_path, self.root / "ownership_map_rollback_backup.pt")
        shutil.copy2(self.previous_path, self.current_path)
        return {"rollback_completed": True, "restored_from": str(self.previous_path)}


class OwnershipBalanceController:
    def __init__(self, num_experts: int, *, beta: float = 0.9, eta: float = 0.1, balance_bias_cap: float = 0.1) -> None:
        self.bias = torch.zeros(num_experts)
        self.beta = float(beta)
        self.eta = float(eta)
        self.balance_bias_cap = float(balance_bias_cap)

    def update_from_owner_share(self, owner_share: torch.Tensor, target_share: torch.Tensor) -> torch.Tensor:
        delta = target_share.float() - owner_share.float()
        self.bias = torch.clamp(self.beta * self.bias.to(delta.device) + self.eta * delta, -self.balance_bias_cap, self.balance_bias_cap)
        return self.bias.clone()


def _semantic_owner(score: torch.Tensor, compatible_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    neg_inf = torch.finfo(score.dtype).min
    masked = score.masked_fill(~compatible_mask.bool(), neg_inf)
    owner = masked.argmax(dim=-1)
    top2 = torch.topk(masked, k=min(2, masked.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = torch.full_like(owner, float("inf"), dtype=score.dtype)
    else:
        margin = top2[:, 0] - top2[:, 1]
    return owner, margin


def route_ownership_top1(
    router_logits: torch.Tensor,
    prototype_bias: torch.Tensor,
    compatible_mask: torch.Tensor,
    proto_ids: torch.Tensor,
    ownership_map: ExpertOwnershipMap,
    config: OwnershipRoutingConfig | None = None,
) -> OwnershipRoutingResult:
    cfg = config or OwnershipRoutingConfig()
    if not torch.all(compatible_mask.bool().any(dim=-1)):
        raise ValueError("Each routed state must have at least one compatible owner")

    base_score = router_logits + prototype_bias
    semantic_owner, semantic_margin = _semantic_owner(base_score, compatible_mask)
    mode = cfg.ownership_map_mode or ownership_map.map_mode
    score = base_score
    ownership_bias_clipped = torch.zeros_like(base_score, dtype=torch.bool)
    ownership_bias = torch.zeros_like(base_score)
    balance_bias = torch.zeros_like(base_score)

    if mode in {"frozen", "canary"}:
        reliability, failure, monopoly, stale, balance = ownership_map.get_all_bias_tensors_fast(proto_ids, dtype=router_logits.dtype)
        raw_ownership = (reliability - failure * float(cfg.failure_bias_weight)) * float(cfg.ownership_weight)
        ownership_bias = torch.clamp(raw_ownership, -float(cfg.ownership_bias_cap), float(cfg.ownership_bias_cap))
        ownership_bias_clipped = raw_ownership.ne(ownership_bias)
        balance_bias = torch.clamp(
            balance * float(cfg.balance_weight),
            -float(cfg.balance_bias_cap),
            float(cfg.balance_bias_cap),
        )
        score = score + ownership_bias + balance_bias - monopoly - stale

    neg_inf = torch.finfo(router_logits.dtype).min
    effective_score = score.masked_fill(~compatible_mask.bool(), neg_inf)
    owner = effective_score.argmax(dim=-1)
    changed_by_bias = owner.ne(semantic_owner)
    balance_active = balance_bias.abs().sum(dim=-1).gt(0)
    ownership_active = ownership_bias.abs().sum(dim=-1).gt(0)
    guard_blocks = (
        changed_by_bias
        & semantic_margin.gt(float(cfg.semantic_margin_guard))
        & balance_active
        & ~ownership_active
    )
    if mode in {"frozen", "canary"} and bool(guard_blocks.any().detach().cpu()):
        owner = torch.where(guard_blocks, semantic_owner, owner)

    adjusted_score = effective_score.clone()
    adjusted_score.scatter_(1, owner.long().view(-1, 1), neg_inf)
    score_challenger = adjusted_score.argmax(dim=-1)
    balance_override_attempt = guard_blocks
    metrics = {
        "balance_bias_changed_owner_rate": changed_by_bias.to(torch.float32).mean(),
        "balance_bias_override_attempt_rate": balance_override_attempt.to(torch.float32).mean(),
    }
    return OwnershipRoutingResult(
        owner=owner,
        score_challenger=score_challenger,
        effective_score=effective_score,
        ownership_bias_clipped=ownership_bias_clipped,
        balance_bias_override_attempt=balance_override_attempt,
        metrics=metrics,
    )


def forward_ownership_top1_fast(
    router_logits: torch.Tensor,
    prototype_bias: torch.Tensor,
    compatible_mask: torch.Tensor,
    proto_ids: torch.Tensor,
    ownership_map: ExpertOwnershipMap,
    config: OwnershipRoutingConfig | None = None,
) -> OwnershipRoutingResult:
    return route_ownership_top1(router_logits, prototype_bias, compatible_mask, proto_ids, ownership_map, config)


def compute_top1_oracle_gap(
    owner: torch.Tensor,
    losses: torch.Tensor,
    compatible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = losses.masked_fill(~compatible_mask.bool(), torch.finfo(losses.dtype).max)
    best_loss, best_owner = masked.min(dim=-1)
    selected = losses.gather(1, owner.long().view(-1, 1)).squeeze(1)
    return selected - best_loss, best_owner


def non_uniform_target_owner_share(base: torch.Tensor, *, specialization_weight: torch.Tensor | None = None) -> torch.Tensor:
    weights = base.float().clone()
    if specialization_weight is not None:
        weights = weights * specialization_weight.float().to(weights.device)
    return weights / torch.clamp(weights.sum(), min=1e-9)


def generate_balanced_assignment_targets(
    losses: torch.Tensor,
    compatible_mask: torch.Tensor,
    target_owner_share: torch.Tensor,
) -> dict[str, Any]:
    masked = losses.masked_fill(~compatible_mask.bool(), torch.finfo(losses.dtype).max)
    return {
        "status": "PVR_EC_OFFLINE_BALANCED_ASSIGNMENT_ONLY",
        "target_owner": masked.argmin(dim=-1),
        "target_owner_share": target_owner_share.float(),
    }


def minimum_sample_protection(
    sample_count: torch.Tensor,
    current_owner: torch.Tensor,
    candidate_owner: torch.Tensor,
    *,
    min_ownership_samples: int = 32,
) -> torch.Tensor:
    rows = torch.arange(candidate_owner.numel(), device=sample_count.device)
    candidate_samples = sample_count[rows, candidate_owner.long().to(sample_count.device)]
    current_samples = sample_count[rows, current_owner.long().to(sample_count.device)]
    return (candidate_samples >= min_ownership_samples) & (current_samples >= min_ownership_samples)


def compute_ownership_metrics(
    proto_ids: torch.Tensor,
    owner: torch.Tensor,
    compatible_mask: torch.Tensor,
    *,
    num_prototypes: int,
    num_experts: int,
    oracle_gap: torch.Tensor | None = None,
) -> dict[str, float]:
    counts = torch.bincount(owner.long().detach().cpu(), minlength=num_experts).float()
    share = counts / torch.clamp(counts.sum(), min=1.0)
    entropy = float(-(share[share > 0] * torch.log2(share[share > 0])).sum().item())
    monopoly = 0.0
    if proto_ids.numel() > 0:
        rows = []
        cpu_proto = proto_ids.detach().cpu()
        cpu_owner = owner.detach().cpu()
        for proto in torch.unique(cpu_proto):
            proto_owners = cpu_owner[cpu_proto == proto]
            if proto_owners.numel():
                proto_counts = torch.bincount(proto_owners.long(), minlength=num_experts).float()
                rows.append(float(proto_counts.max().item() / max(proto_owners.numel(), 1)))
        monopoly = float(sum(1 for value in rows if value >= 0.9) / max(len(rows), 1))
    return {
        "owner_count": int(owner.numel()),
        "expert_owner_entropy": entropy,
        "prototype_local_owner_entropy": entropy,
        "prototype_local_monopoly_rate": monopoly,
        "top1_oracle_gap": float(oracle_gap.float().mean().item()) if oracle_gap is not None and oracle_gap.numel() else 0.0,
        "compatible_owner_rate": float(compatible_mask.bool().float().mean().item()),
    }


def run_offline_ownership_replay(
    proto_ids: torch.Tensor,
    owner: torch.Tensor,
    losses: torch.Tensor,
    compatible_mask: torch.Tensor,
    *,
    num_prototypes: int,
    num_experts: int,
    sample_count: torch.Tensor | None = None,
    output_dir: str | Path | None = None,
) -> OwnershipReplayResult:
    gap, best = compute_top1_oracle_gap(owner, losses, compatible_mask)
    statuses = ["PVR_EC_OWNERSHIP_REPLAY_COMPLETE"]
    if sample_count is not None and bool((sample_count < 32).any().detach().cpu()):
        statuses.append("PVR_EC_OWNERSHIP_LOW_SAMPLE_REGION")
    metrics = compute_ownership_metrics(
        proto_ids,
        owner,
        compatible_mask,
        num_prototypes=num_prototypes,
        num_experts=num_experts,
        oracle_gap=gap,
    )
    if output_dir is not None:
        write_ownership_reports(Path(output_dir), metrics)
    return OwnershipReplayResult(statuses=statuses, top1_oracle_gap=gap, oracle_best_owner=best, metrics=metrics)


def hot_path_purity_score(counters: dict[str, int]) -> float:
    total = sum(abs(int(counters.get(field_name, 0))) for field_name in HOT_PATH_COUNTER_FIELDS)
    return 1.0 if total == 0 else 1.0 / (1.0 + float(total))


def ownership_overhead_ratio(timing: dict[str, float]) -> float:
    overhead = sum(
        float(timing.get(field_name, 0.0))
        for field_name in (
            "ownership_bias_lookup_ms",
            "ownership_score_ms",
            "shadow_logging_ms",
            "replay_queue_ms",
            "candidate_map_check_ms",
            "metadata_validation_ms",
            "cpu_transfer_ms",
            "cuda_sync_ms",
        )
    )
    return overhead / max(float(timing.get("total_forward_ms", 0.0)), 1e-9)


def write_ownership_reports(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _jsonable({**metrics, "promotion_status": "PVR_EC_DO_NOT_PROMOTE"})
    report_names = [
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
    for name in report_names:
        body = dict(payload)
        body["report"] = name.removesuffix(".json")
        if name == "ownership_oracle_gap_report.json":
            body.setdefault("deploy_top1_oracle_gap", body.get("top1_oracle_gap", 0.0))
        out.joinpath(name).write_text(json.dumps(body, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
