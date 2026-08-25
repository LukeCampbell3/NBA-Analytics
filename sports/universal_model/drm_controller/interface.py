"""Public DRM controller contract (spec section 24).

The controller manages architectural coordinates only -- it never touches
predictions, never sees TEST, and never influences the outer betting-
certification system (spec section 31: PolicyStatus/G_C/G_L/G_V live in
sports/mlb/research/parlay_certification_v2/ and this package imports
nothing from there and exports nothing to it).

Allowed architectural coordinates in this build: number_of_experts
(via expert_birth), router_family_structure (via router_repair),
extra optimization steps (via parameter_adaptation). NOT implemented in
this build: MoE_layer_count, adapter_capacity, temporal_memory_capacity
(see mutations.py module docstring for why).

Entry point: ``controller.run_drm_development(model, config, derive,
select, ...) -> DRMBudget`` -- see controller.py.
"""
from sports.universal_model.drm_controller.controller import run_drm_development
from sports.universal_model.drm_controller.provisional import DRMBudget, StructuralMutationRecord
from sports.universal_model.drm_controller.residuals import ResidualSignature

__all__ = ["run_drm_development", "DRMBudget", "StructuralMutationRecord", "ResidualSignature"]
