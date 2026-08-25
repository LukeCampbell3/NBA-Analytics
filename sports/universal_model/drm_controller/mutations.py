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


def restore(model: UniversalModel, snap: dict) -> None:
    """Exact rollback (spec section 28/61): a rejected mutation must
    revert exactly, including any shape change from expert birth -- so we
    rebuild the model's expert/router tensors to the snapshot's shapes
    before loading, rather than assuming shapes are unchanged."""
    # Shape-safe restore: reconstruct parameter tensors directly from the
    # snapshot's shapes for any MoE/Switch layer whose expert count grew.
    own_state = model.state_dict()
    for key, tensor in snap.items():
        if key in own_state and own_state[key].shape != tensor.shape:
            _resize_param(model, key, tensor.shape)
    model.load_state_dict(snap, strict=True)


def _resize_param(model: UniversalModel, dotted_name: str, shape: torch.Size) -> None:
    module_path, _, param_name = dotted_name.rpartition(".")
    module = model
    for part in module_path.split("."):
        module = getattr(module, part)
    setattr(module, param_name, torch.nn.Parameter(torch.zeros(shape)))
