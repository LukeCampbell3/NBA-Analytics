"""Shared-width growth (spec section 24's `shared_hidden_width` coordinate;
the "shared_width_expansion" DRM tier that was previously stubbed out as
deferred in `drm_controller/mutations.py`).

Grows every module in a ``UniversalModel`` whose parameters depend on
``hidden_dim`` from ``H_old`` to ``H_new = H_old + delta``, transplanting
old weights into the new, larger tensors rather than reinitializing from
scratch. This is APPROXIMATELY, not exactly, function-preserving:

- Embeddings, numeric projections, DenseFFN/ExpertBank second layers, and
  attention (``nn.MultiheadAttention``, handled block-aware below so its
  stacked Q/K/V layout doesn't get corrupted by a naive copy) ARE exactly
  function-preserving: every new dimension is wired to contribute exactly
  zero until trained.
- ``nn.LayerNorm`` is NOT exactly function-preserving: it normalizes over
  the full (now longer, zero-padded-on-the-new-dims) vector, which shifts
  the computed mean/variance for the original dimensions by a small
  amount. This is disclosed rather than hidden -- unlike function-
  preserving expert birth (spec section 27), this mutation is gated purely
  by the same bounded SELECT-evaluated commit/rollback machinery
  (`drm_controller/evaluator.py`) as a real experiment, not guaranteed
  identity at the instant of growth.

``delta`` must be a multiple of ``n_heads`` (attention requires
``hidden_dim % n_heads == 0``).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from sports.universal_model.model.universal_model import UniversalModel


def _grow_embedding(old: nn.Embedding, new: nn.Embedding) -> None:
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.weight.shape[1]] = old.weight


def _grow_linear_out(old: nn.Linear, new: nn.Linear) -> None:
    """Input dim unchanged, output dim grows (e.g. numeric projections
    H_old -> H_new): new output rows start at exact zero."""
    with torch.no_grad():
        new.weight.zero_()
        new.bias.zero_()
        new.weight[: old.weight.shape[0], :] = old.weight
        new.bias[: old.bias.shape[0]] = old.bias


def _grow_linear_in_out(old: nn.Linear, new: nn.Linear, zero_new_in_cols: bool = True) -> None:
    """Both input and output dims grow (FFN first layer: H -> mult*H)."""
    with torch.no_grad():
        if zero_new_in_cols:
            new.weight.zero_()
        new.bias.zero_()
        oh, ow = old.weight.shape
        new.weight[:oh, :ow] = old.weight
        new.bias[:oh] = old.bias


def _grow_linear_in(old: nn.Linear, new: nn.Linear) -> None:
    """Input dim grows, output dim unchanged (FFN second layer:
    mult*H -> H): new input columns are zeroed so whatever the
    corresponding (untrained) new inner units compute contributes
    nothing downstream."""
    with torch.no_grad():
        new.weight.zero_()
        new.bias.zero_()
        oh, ow = old.weight.shape
        new.weight[:oh, :ow] = old.weight
        new.bias[:oh] = old.bias


def _grow_layernorm(old: nn.LayerNorm, new: nn.LayerNorm) -> None:
    with torch.no_grad():
        new.weight.fill_(1.0)
        new.bias.zero_()
        new.weight[: old.weight.shape[0]] = old.weight
        new.bias[: old.bias.shape[0]] = old.bias


def _grow_attention(old: nn.MultiheadAttention, new: nn.MultiheadAttention, h_old: int, h_new: int) -> None:
    with torch.no_grad():
        new.in_proj_weight.zero_()
        new.in_proj_bias.zero_()
        q_old, k_old, v_old = old.in_proj_weight.chunk(3, dim=0)
        qb_old, kb_old, vb_old = old.in_proj_bias.chunk(3, dim=0)
        new.in_proj_weight[0:h_old, 0:h_old] = q_old
        new.in_proj_weight[h_new: h_new + h_old, 0:h_old] = k_old
        new.in_proj_weight[2 * h_new: 2 * h_new + h_old, 0:h_old] = v_old
        new.in_proj_bias[0:h_old] = qb_old
        new.in_proj_bias[h_new: h_new + h_old] = kb_old
        new.in_proj_bias[2 * h_new: 2 * h_new + h_old] = vb_old

        new.out_proj.weight.zero_()
        new.out_proj.bias.zero_()
        new.out_proj.weight[0:h_old, 0:h_old] = old.out_proj.weight
        new.out_proj.bias[0:h_old] = old.out_proj.bias


def _grow_expert_bank(old_experts, new_experts) -> None:
    with torch.no_grad():
        new_experts.w1.zero_()
        new_experts.b1.zero_()
        new_experts.w2.zero_()
        new_experts.b2.zero_()
        e, h_old, inner_old = old_experts.w1.shape
        new_experts.w1[:, :h_old, :inner_old] = old_experts.w1
        new_experts.b1[:, :inner_old] = old_experts.b1
        new_experts.w2[:, :inner_old, :h_old] = old_experts.w2
        new_experts.b2[:, :h_old] = old_experts.b2


def grow_hidden_dim(model: UniversalModel, delta: int) -> UniversalModel:
    n_heads = model.config["n_heads"]
    if delta % n_heads != 0:
        raise ValueError(f"delta={delta} must be a multiple of n_heads={n_heads}")
    h_old = model.config["hidden_dim"]
    h_new = h_old + delta
    new_config = dict(model.config)
    new_config["hidden_dim"] = h_new
    new_model = UniversalModel(**new_config)

    # model.config["n_experts"] is a static construction-time value that a
    # prior DRM expert-birth commit (within the same session) can have
    # already outgrown -- grow new_model's MoE/Switch layers to match the
    # OLD model's REAL per-layer expert count (from its actual tensors)
    # BEFORE copying weights in. Unlike drm_controller.mutations's
    # resize_moe_layers_to_match (used for rollback/checkpoint-loading,
    # where hidden_dim never changes), this must keep h_new -- not reuse
    # that helper, which would also overwrite the new hidden_dim.
    for layer_old, layer_new in zip(model.stem.moe_layers(), new_model.stem.moe_layers()):
        n_old = layer_old.experts.w1.shape[0]
        if layer_new.experts.w1.shape[0] != n_old:
            inner = layer_new.experts.w1.shape[2]
            device = layer_new.experts.w1.device
            layer_new.experts.w1 = nn.Parameter(torch.zeros(n_old, h_new, inner, device=device))
            layer_new.experts.b1 = nn.Parameter(torch.zeros(n_old, inner, device=device))
            layer_new.experts.w2 = nn.Parameter(torch.zeros(n_old, inner, h_new, device=device))
            layer_new.experts.b2 = nn.Parameter(torch.zeros(n_old, h_new, device=device))
            layer_new.experts.n_experts = n_old
            layer_new.router.gate = nn.Linear(h_new, n_old, device=device)
            layer_new.router.n_experts = n_old
            layer_new.n_experts = n_old

    tok_old, tok_new = model.tokenizer, new_model.tokenizer
    _grow_embedding(tok_old.sport_emb, tok_new.sport_emb)
    _grow_embedding(tok_old.entity_emb, tok_new.entity_emb)
    _grow_embedding(tok_old.role_emb, tok_new.role_emb)
    _grow_embedding(tok_old.home_emb, tok_new.home_emb)
    _grow_embedding(tok_old.target_emb, tok_new.target_emb)
    _grow_embedding(tok_old.missing_mask_emb, tok_new.missing_mask_emb)
    _grow_embedding(tok_old.token_type_emb, tok_new.token_type_emb)
    for name in ("opportunity_proj", "temporal_proj", "market_proj", "uncertainty_proj"):
        _grow_linear_out(getattr(tok_old, name), getattr(tok_new, name))

    for block_old, block_new in zip(model.stem.blocks, new_model.stem.blocks):
        _grow_attention(block_old.attn, block_new.attn, h_old, h_new)
        _grow_layernorm(block_old.ln1, block_new.ln1)
        _grow_layernorm(block_old.ln2, block_new.ln2)
        ffn_old, ffn_new = block_old.ffn, block_new.ffn
        if hasattr(ffn_old, "net"):  # DenseFFN
            _grow_linear_in_out(ffn_old.net[0], ffn_new.net[0])
            _grow_linear_in(ffn_old.net[3], ffn_new.net[3])
        else:  # Switch/Top2MoEFFN
            _grow_expert_bank(ffn_old.experts, ffn_new.experts)
            # Router gate: input dim grows (H_old -> H_new), output (n_experts) unchanged.
            with torch.no_grad():
                ffn_new.router.gate.weight.zero_()
                ffn_new.router.gate.bias.zero_()
                ffn_new.router.gate.weight[:, :h_old] = ffn_old.router.gate.weight
                ffn_new.router.gate.bias[:] = ffn_old.router.gate.bias
    _grow_layernorm(model.stem.final_norm, new_model.stem.final_norm)

    _grow_linear_in_out(model.heads.prob_over_head[0], new_model.heads.prob_over_head[0])
    _grow_linear_in(model.heads.prob_over_head[3], new_model.heads.prob_over_head[3])
    _grow_linear_in_out(model.heads.z_head[0], new_model.heads.z_head[0])
    _grow_linear_in(model.heads.z_head[3], new_model.heads.z_head[3])

    return new_model
