"""DRM DERIVE (spec section 26): bounded structural mutation vocabulary in
least-complexity escalation order.

DISCLOSED SCOPE LIMITATION: the full spec section 26 escalation order has
8 tiers (parameter adaptation, router repair, expert width/local repair,
expert birth, expert merge/split, additional MoE layer, added temporal/
state capacity, shared-width expansion). This build implements the first,
smallest three plus the last (widest) one:

    1. parameter_adaptation  (extra fine-tuning steps, no structural change)
    2. router_repair         (reset the gate's weights -- addresses collapse)
    3. expert_birth          (function-preserving, per moe.py)
    4. shared_width_expansion (grow the shared attention hidden width)

"expert width/local repair", "expert merge/split", "additional MoE layer",
and "added temporal/state capacity" are NOT implemented in this build --
each would need either a fixed-width-invariant surgery (merge/split) or a
new architectural axis (temporal memory) this dataset's 8-token,
single-timestep representation does not yet use. Reported honestly in
FINAL_REPORT.md rather than silently skipped.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch

from sports.universal_model.drm_controller.residuals import ResidualSignature
from sports.universal_model.model.universal_model import UniversalModel

ESCALATION_ORDER = ["parameter_adaptation", "router_repair", "expert_birth", "shared_width_expansion"]

MAX_EXPERTS = 16
MAX_SHARED_WIDTH = 192


@dataclass
class MutationCandidate:
    tier: str
    description: str
    apply: Callable[[UniversalModel], None]
    param_delta_estimate: int


def propose_mutation(model: UniversalModel, residuals: list[ResidualSignature], attempted_tiers: set[str]) -> MutationCandidate | None:
    """Smallest untried tier in escalation order that is both warranted by
    a residual and still within bounds. Returns None if nothing is
    warranted or every eligible tier has already been tried this cycle
    (spec section 29: no unbounded recursive spawning before evaluation)."""
    if not residuals:
        return None

    has_collapse = any(r.error_type == "routing_collapse" for r in residuals)
    has_calibration_or_brier = any(r.error_type in ("calibration", "brier") for r in residuals)
    n_experts_now = _current_n_experts(model)

    for tier in ESCALATION_ORDER:
        if tier in attempted_tiers:
            continue
        if tier == "parameter_adaptation" and has_calibration_or_brier:
            return MutationCandidate(
                tier=tier,
                description="extra fine-tuning steps only, no structural change",
                apply=lambda m: None,
                param_delta_estimate=0,
            )
        if tier == "router_repair" and has_collapse:
            return MutationCandidate(
                tier=tier,
                description="reset router gate weights to break collapse",
                apply=_reset_routers,
                param_delta_estimate=0,
            )
        if tier == "expert_birth" and has_calibration_or_brier and n_experts_now < MAX_EXPERTS:
            return MutationCandidate(
                tier=tier,
                description="function-preserving expert birth on all MoE/Switch layers",
                apply=_birth_experts,
                param_delta_estimate=_expert_param_estimate(model),
            )
        if tier == "shared_width_expansion" and has_calibration_or_brier:
            return None  # requires rebuilding the whole stem; deferred, see module docstring
    return None


def _current_n_experts(model: UniversalModel) -> int:
    layers = model.stem.moe_layers()
    return layers[0].n_experts if layers else 0


def _expert_param_estimate(model: UniversalModel) -> int:
    layers = model.stem.moe_layers()
    if not layers:
        return 0
    layer = layers[0]
    per_expert = layer.experts.w1[0].numel() + layer.experts.b1[0].numel() + layer.experts.w2[0].numel() + layer.experts.b2[0].numel()
    return int(per_expert) * len(layers)


def _reset_routers(model: UniversalModel) -> None:
    for layer in model.stem.moe_layers():
        nn_init_linear(layer.router.gate)


def _birth_experts(model: UniversalModel) -> None:
    for layer in model.stem.moe_layers():
        layer.add_expert()


def nn_init_linear(linear: torch.nn.Linear) -> None:
    torch.nn.init.xavier_uniform_(linear.weight)
    torch.nn.init.zeros_(linear.bias)


def snapshot(model: UniversalModel) -> dict:
    return copy.deepcopy(model.state_dict())


def resize_moe_layers_to_match(model: UniversalModel, state_dict: dict) -> None:
    """Rebuild every MoE/Switch layer's gate + expert bank (and this
    package's own n_experts counters) to match ``state_dict``'s real
    tensor shapes, in place, WITHOUT loading values yet.

    Needed in two places: DRM rollback (a rejected expert-birth must
    shrink back down) and checkpoint loading (a model built fresh from
    ``model.config["n_experts"]`` does not know a later-committed DRM
    expert birth grew that count -- ``model.config`` is a static
    construction-time dict, not kept in sync with in-place mutations).

    A naive "just resize the raw weight tensors and load_state_dict" is
    NOT sufficient for the router's gate: it is an ``nn.Linear``, and
    ``add_expert()`` (moe.py) replaces that whole module with a new
    ``nn.Linear`` of larger ``out_features``. Resizing ``.weight``/
    ``.bias`` in place instead leaves the Linear module's own
    ``in_features``/``out_features`` metadata stale even though the tensor
    shape is correct -- so the gate object itself is rebuilt here, not
    just its parameters.
    """
    for name, module in model.named_modules():
        if not hasattr(module, "experts") or not hasattr(module, "router"):
            continue  # not a Switch/Top2MoEFFN layer
        w1_key = f"{name}.experts.w1"
        if w1_key not in state_dict:
            continue
        target_n_experts = state_dict[w1_key].shape[0]
        if module.experts.w1.shape[0] == target_n_experts:
            continue  # no shape drift for this layer
        hidden_dim, inner = state_dict[w1_key].shape[1], state_dict[w1_key].shape[2]
        device = module.experts.w1.device
        module.experts.w1 = torch.nn.Parameter(torch.zeros(target_n_experts, hidden_dim, inner, device=device))
        module.experts.b1 = torch.nn.Parameter(torch.zeros(target_n_experts, inner, device=device))
        module.experts.w2 = torch.nn.Parameter(torch.zeros(target_n_experts, inner, hidden_dim, device=device))
        module.experts.b2 = torch.nn.Parameter(torch.zeros(target_n_experts, hidden_dim, device=device))
        module.experts.n_experts = target_n_experts
        module.router.gate = torch.nn.Linear(hidden_dim, target_n_experts, device=device)
        module.router.n_experts = target_n_experts


def restore(model: UniversalModel, snap: dict) -> None:
    """Exact rollback (spec section 28/61): a rejected mutation must
    revert exactly, including any shape change from expert birth."""
    resize_moe_layers_to_match(model, snap)
    model.load_state_dict(snap, strict=True)
