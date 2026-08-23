from __future__ import annotations

"""SETTLEMENT (mission section 4) -- actual unit net return under frozen
sportsbook settlement rules. Never universally assumes R = D*W - 1 in the
empirical layer; that formula applies only to the WIN/LOSS cases below,
one case among several a real settlement can take.

Boundedness (-1 <= R <= R_max) is part of the inference contract: an
action must be REJECTED before acceptance if its price would violate the
configured bound (`reject_if_price_exceeds_bound`), and settlement itself
re-validates the bound as a belt-and-suspenders check.
"""

from dataclasses import dataclass
from enum import Enum

SETTLEMENT_VERSION = "SETTLEMENT_V1"


class SettlementStatus(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"
    CANCELED = "canceled"
    REPRICED_VOID = "repriced_void"  # one leg voided; sportsbook reprices the remaining leg(s)


_ZERO_RETURN_STATUSES = (SettlementStatus.PUSH, SettlementStatus.VOID, SettlementStatus.CANCELED)


@dataclass(frozen=True)
class SettlementInput:
    status: SettlementStatus
    accepted_decimal_price: float | None = None  # required for WIN
    repriced_decimal_price: float | None = None  # required for REPRICED_VOID


def reject_if_price_exceeds_bound(decimal_price: float, *, r_max: float) -> None:
    """Pre-acceptance guard: an action must never be accepted if its price
    would produce R > r_max. Raises ValueError; callers must treat this as
    'reject this candidate', not 'the day is ineligible'."""
    implied_r = decimal_price - 1.0
    if implied_r > r_max + 1e-12:
        raise ValueError(f"accepted price implies R={implied_r} > R_max={r_max}; action must be rejected")


def resolve_return(inp: SettlementInput, *, r_max: float) -> float:
    """R_t: actual unit net return under frozen settlement rules. Raises
    ValueError if the resolved return would violate -1 <= R <= R_max
    (boundedness is part of the inference contract, not an afterthought)."""
    if inp.status == SettlementStatus.WIN:
        if inp.accepted_decimal_price is None:
            raise ValueError("WIN settlement requires accepted_decimal_price")
        r = inp.accepted_decimal_price - 1.0
    elif inp.status == SettlementStatus.LOSS:
        r = -1.0
    elif inp.status in _ZERO_RETURN_STATUSES:
        r = 0.0
    elif inp.status == SettlementStatus.REPRICED_VOID:
        if inp.repriced_decimal_price is None:
            raise ValueError("REPRICED_VOID settlement requires repriced_decimal_price")
        r = inp.repriced_decimal_price - 1.0
    else:
        raise ValueError(f"unknown settlement status {inp.status!r}")

    if not (-1.0 - 1e-9 <= r <= r_max + 1e-9):
        raise ValueError(f"resolved settlement return {r} violates bound [-1, {r_max}]")
    return float(min(max(r, -1.0), r_max))


def is_loss(r: float) -> bool:
    """ell_t = 1{R_t < 0} -- push/void/refund with R=0 is NOT a loss."""
    return bool(r < 0.0)
