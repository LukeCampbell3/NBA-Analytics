from __future__ import annotations

"""Multidimensional CandidateSupport with EXPLICIT GATE MODES (mission:
"Resolve the PARLAY_V2 perpetual-abstention problem"). Support is not
reduced to one arbitrary number, and -- the fix this module makes --
support dimensions with no validated research behind them no longer
silently block every action forever.

THE BUG THIS REPLACES: the previous version required ALL FIVE dimensions
(including joint_support and shift_status, both permanently
"UNESTABLISHED" because no validated research established them) to pass
before ANY candidate could be selected. That made policy selection
circularly dependent on research that itself depends on having selected
candidates to observe -- selection could never happen, so evidence could
never accumulate, so the research could never be done. See
run_parlay_v2.py's module docstring for the full call-graph writeup.

THE FIX: each dimension declares a GateMode.
    REQUIRED     -- must PASS or the dimension blocks action.
    OBSERVE_ONLY -- computed and exposed for research, NEVER blocks action,
                    regardless of its status (including UNESTABLISHED).
    DISABLED     -- not evaluated for this policy version at all.

Frozen support-gate configuration (SUPPORT_GATE_MODES in manifest.py --
carried unchanged from PARLAY_POLICY_V2_PROSPECTIVE_002 into
PARLAY_POLICY_V2_PROSPECTIVE_003; the two differ only in world_gate_mode,
never in this):
    market_support  REQUIRED   (>= N_MARKET prior settled observations, real, implemented)
    line_support    REQUIRED   (>= N_LINE prior settled observations, real, implemented)
    state_support   REQUIRED   (>= N_STATE independent prior slates, real, implemented --
                                 reuses the already-established
                                 MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT=20 convention)
    joint_support   OBSERVE_ONLY, status=UNESTABLISHED, used_for_action=False
    shift_status    OBSERVE_ONLY, status=UNESTABLISHED, used_for_action=False

UNESTABLISHED is never reinterpreted as PASS. It is a real status value.
What changed is that an OBSERVE_ONLY dimension's status -- PASS, FAIL, or
UNESTABLISHED -- has NO effect on `in_support`, by construction (see
`blocks_action` below): only `gate_mode == REQUIRED and status != "PASS"`
blocks. A future policy version may promote joint_support or shift_status
to REQUIRED only once each has an independently validated, non-arbitrary
threshold -- see calibration/README or joint_position_builder_v2/STATE.md
for why no such threshold exists yet. Nothing in this module invents one.
"""

from dataclasses import dataclass
from enum import Enum

from .versioning import SUPPORT_RULE_VERSION

N_MARKET = 20
N_LINE = 20
N_STATE = 20  # matches candidate_adapter.MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT


class GateMode(str, Enum):
    REQUIRED = "REQUIRED"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    DISABLED = "DISABLED"


class SupportStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNESTABLISHED = "UNESTABLISHED"


@dataclass(frozen=True)
class SupportDimension:
    name: str
    value: object  # raw count (int) or None for status-only dimensions
    status: SupportStatus
    gate_mode: GateMode

    @property
    def used_for_action(self) -> bool:
        """Whether this dimension's status is actually consulted when
        deciding in_support. False for OBSERVE_ONLY/DISABLED dimensions
        -- their status is exposed for research, never for gating."""
        return self.gate_mode == GateMode.REQUIRED

    @property
    def blocks_action(self) -> bool:
        """The ONLY predicate `in_support` is built from. A dimension
        blocks iff it is REQUIRED and did not PASS -- an OBSERVE_ONLY or
        DISABLED dimension can never block, no matter its status."""
        return self.gate_mode == GateMode.REQUIRED and self.status != SupportStatus.PASS

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status.value,
            "gate_mode": self.gate_mode.value,
            "used_for_action": self.used_for_action,
        }


@dataclass(frozen=True)
class CandidateSupport:
    market_support: SupportDimension
    line_support: SupportDimension
    state_support: SupportDimension
    joint_support: SupportDimension
    shift_status: SupportDimension
    recent_support: int  # descriptive only, never gated -- see module docstring
    calibration_error: float | None  # descriptive only, never gated
    in_support: bool
    support_rule_version: str = SUPPORT_RULE_VERSION

    def as_dict(self) -> dict:
        return {
            "market_support": self.market_support.as_dict(),
            "line_support": self.line_support.as_dict(),
            "state_support": self.state_support.as_dict(),
            "joint_support": self.joint_support.as_dict(),
            "shift_status": self.shift_status.as_dict(),
            "recent_support": self.recent_support,
            "calibration_error": self.calibration_error,
            "in_support": self.in_support,
            "support_rule_version": self.support_rule_version,
        }

    @property
    def blocking_dimensions(self) -> list[str]:
        """Names of every REQUIRED dimension currently blocking action --
        empty iff in_support is True. Used to produce a specific,
        non-generic abstain reason instead of one opaque catch-all."""
        return [d.name for d in (self.market_support, self.line_support, self.state_support) if d.blocks_action]


def _mean_abs_calibration_gap(rows: list[dict]) -> float | None:
    diffs = [
        abs(float(r["predictive_probability_if_available"]) - float(r["actual_outcome"]))
        for r in rows
        if r.get("predictive_probability_if_available") is not None and r.get("actual_outcome") is not None
    ]
    return float(sum(diffs) / len(diffs)) if diffs else None


def evaluate_support(
    snapshot_rows: list[dict],
    *,
    market_bucket: str,
    line_bucket: str,
    state_bucket: str,
    independent_slate_count: int,
    recent_window: int = 20,
) -> CandidateSupport:
    """snapshot_rows: the observation rows behind a CalibrationSnapshot
    (already forward-only filtered -- see snapshot.build_snapshot). This
    function performs no additional time filtering itself; it only
    aggregates what it's given."""
    market_rows = [r for r in snapshot_rows if r.get("market_bucket") == market_bucket]
    line_rows = [r for r in snapshot_rows if r.get("line_bucket") == line_bucket]

    market_count = len(market_rows)
    line_count = len(line_rows)

    market_support = SupportDimension(
        "market_support", market_count,
        SupportStatus.PASS if market_count >= N_MARKET else SupportStatus.FAIL,
        GateMode.REQUIRED,
    )
    line_support = SupportDimension(
        "line_support", line_count,
        SupportStatus.PASS if line_count >= N_LINE else SupportStatus.FAIL,
        GateMode.REQUIRED,
    )
    state_support = SupportDimension(
        "state_support", independent_slate_count,
        SupportStatus.PASS if independent_slate_count >= N_STATE else SupportStatus.FAIL,
        GateMode.REQUIRED,
    )
    # OBSERVE_ONLY: never blocks, regardless of status. No arbitrary
    # threshold is invented here -- see module docstring.
    joint_support = SupportDimension("joint_support", None, SupportStatus.UNESTABLISHED, GateMode.OBSERVE_ONLY)
    shift_status = SupportDimension("shift_status", None, SupportStatus.UNESTABLISHED, GateMode.OBSERVE_ONLY)

    recent_rows = sorted(snapshot_rows, key=lambda r: str(r.get("calibration_admitted_at", "")))[-recent_window:]
    recent_support = sum(1 for r in recent_rows if r.get("market_bucket") == market_bucket)
    calibration_error = _mean_abs_calibration_gap(market_rows)

    dimensions = (market_support, line_support, state_support, joint_support, shift_status)
    in_support = not any(d.blocks_action for d in dimensions)

    return CandidateSupport(
        market_support=market_support,
        line_support=line_support,
        state_support=state_support,
        joint_support=joint_support,
        shift_status=shift_status,
        recent_support=recent_support,
        calibration_error=calibration_error,
        in_support=in_support,
    )
