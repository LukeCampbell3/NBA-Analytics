"""DRM development loop (spec sections 24, 30): OBSERVE -> DERIVE ->
bounded SELECT-gated experiment -> COMMIT/ROLLBACK, run strictly outside
the hot training loop (called between checkpoints, not per-minibatch).

Never touches TEST. Every mutation attempt -- committed or rejected -- is
recorded in the returned ``DRMBudget.history`` (spec section 48: "No
hidden mutations").
"""
from __future__ import annotations

import itertools
import uuid

import torch
from torch.utils.data import DataLoader

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.drm_controller.evaluator import ComplexityWeights, compute_J, decide_commit
from sports.universal_model.drm_controller.mutations import propose_mutation, restore, snapshot
from sports.universal_model.drm_controller.provisional import DRMBudget, StructuralMutationRecord
from sports.universal_model.drm_controller.residuals import observe_residuals
from sports.universal_model.model.universal_model import UniversalModel
from sports.universal_model.train.losses import compute_losses
from sports.universal_model.train.sampler import build_temperature_sampler
from sports.universal_model.train.trainer import evaluate


def _collect_routing_diagnostics(model: UniversalModel, dataset: UniversalDataset, batch_size: int = 256) -> tuple[float | None, list[float] | None]:
    layers = model.stem.moe_layers()
    if not layers:
        return None, None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch)
    moe_diags = [d for d in out["block_diagnostics"] if "routing_entropy" in d]
    if not moe_diags:
        return None, None
    entropy = float(sum(d["routing_entropy"].item() for d in moe_diags) / len(moe_diags))
    tokens_per_expert = moe_diags[-1]["tokens_per_expert"].tolist()
    return entropy, tokens_per_expert


def _finetune(model: UniversalModel, derive: UniversalDataset, steps: int, lr: float, batch_size: int, alpha: float, config) -> None:
    sampler = build_temperature_sampler(derive, alpha=alpha)
    loader = DataLoader(derive, batch_size=batch_size, sampler=sampler)
    it = itertools.cycle(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        batch = next(it)
        out = model(batch)
        losses = compute_losses(out, batch, config)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()


def run_drm_development(
    model: UniversalModel,
    config,
    derive: UniversalDataset,
    select: UniversalDataset,
    n_cycles: int = 3,
    max_mutation_attempts_per_cycle: int = 3,
    finetune_steps: int = 300,
    weights: ComplexityWeights = ComplexityWeights(),
) -> tuple[UniversalModel, DRMBudget]:
    """Returns ``(final_model, budget)``. ``model`` itself is mutated
    in-place for router_repair/expert_birth (snapshot/restore via
    state_dict works normally there), but shared_width_expansion replaces
    the model object entirely (growing hidden_dim changes nearly every
    parameter's shape) -- callers MUST use the returned model, not assume
    the passed-in ``model`` reference reflects a committed width growth."""
    budget = DRMBudget(max_mutation_attempts_per_cycle=max_mutation_attempts_per_cycle)
    persistence_tracker: dict[str, int] = {}
    baseline_active_params = model.active_parameters_per_token()

    for cycle in range(n_cycles):
        attempts = 0
        attempted_tiers: set[str] = set()
        while budget.can_attempt(attempts):
            eval_before = evaluate(model, select)
            entropy, tokens_per_expert = _collect_routing_diagnostics(model, select)
            residuals = observe_residuals(eval_before["macro_by_sport"], entropy, tokens_per_expert, persistence_tracker)
            candidate = propose_mutation(model, residuals, attempted_tiers)
            if candidate is None:
                break
            attempted_tiers.add(candidate.tier)
            attempts += 1

            if not budget.can_afford(candidate.param_delta_estimate):
                budget.record(
                    StructuralMutationRecord(
                        mutation_id=str(uuid.uuid4())[:8],
                        tier=candidate.tier,
                        description=candidate.description,
                        residual_motivating_it=residuals[0].to_dict() if residuals else {},
                        before_metrics=eval_before["micro_classification"],
                        after_metrics=eval_before["micro_classification"],
                        param_delta=0,
                        compute_delta=0.0,
                        status="REJECTED",
                        decision_reason="exceeds complexity budget (max_total_param_growth)",
                    )
                )
                continue

            # Two mutation kinds: in-place (apply() mutates `model` and
            # returns None -- snapshot/restore via state_dict handles
            # rollback) and replacing (apply() returns a NEW model object,
            # e.g. shared_width_expansion -- rollback there is simply
            # "keep using the pre-mutation model", no state_dict surgery
            # needed since the old object was never touched).
            snap = None
            pre_mutation_model = model
            replacement = candidate.apply(model)
            working_model = replacement if replacement is not None else model
            if replacement is None:
                snap = snapshot(model)

            _finetune(working_model, derive, steps=finetune_steps, lr=1e-4, batch_size=config.batch_size, alpha=config.alpha, config=config)
            eval_after = evaluate(working_model, select)
            active_after = working_model.active_parameters_per_token()

            loss_before = eval_before["micro_classification"]["log_loss"] or 1.0
            loss_after = eval_after["micro_classification"]["log_loss"] or 1.0
            j_before = compute_J(loss_before, baseline_active_params, baseline_active_params, weights)
            j_after = compute_J(loss_after, active_after, baseline_active_params, weights)
            commit, reason = decide_commit(j_before, j_after)

            record = StructuralMutationRecord(
                mutation_id=str(uuid.uuid4())[:8],
                tier=candidate.tier,
                description=candidate.description,
                residual_motivating_it=residuals[0].to_dict() if residuals else {},
                before_metrics=eval_before["micro_classification"],
                after_metrics=eval_after["micro_classification"],
                param_delta=active_after - baseline_active_params,
                compute_delta=(active_after - baseline_active_params) / max(baseline_active_params, 1),
                status="PERMANENT" if commit else "REJECTED",
                decision_reason=f"J_before={j_before:.6f} J_after={j_after:.6f}: {reason}",
            )
            if commit:
                model = working_model
                baseline_active_params = active_after
            elif snap is not None:
                restore(pre_mutation_model, snap)
                model = pre_mutation_model
            else:
                model = pre_mutation_model  # replacing mutation rejected: discard `working_model`, keep the old one untouched
            budget.record(record)

    return model, budget
