from __future__ import annotations

"""Multidimensional CandidateSupport (mission section 5). Support is NOT
reduced to one arbitrary `state_support >= N` number.

Frozen rule (deliberately simple, per section 5's instruction, using only
already-validated information):

    market_support  >= N_MARKET   (prior settled observations sharing this candidate's market/target bucket)
    line_support    >= N_LINE     (prior settled observations sharing this EXACT line bucket)
    state_support   >= N_STATE    (independent PRIOR SLATES admitted -- reuses the
                                    already-established MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT=20
                                    convention from candidate_adapter.py)
    joint_support   -- UNESTABLISHED: no validated joint/pair-level calibration
                        threshold exists in this repo (the prior turn's
                        multi-target pair backtest specifically found
                        pair-level APS calibration unreliable at this
                        data volume -- see joint_position_builder_v2/STATE.md).
                        Exposed, never silently passed.
    recent_support  -- UNESTABLISHED: no validated recency-window threshold exists.
                        The count is exposed as a real diagnostic; it does
                        not gate in_support on its own.
    calibration_error -- UNESTABLISHED: no frozen epsilon exists for this
                        bucket-level metric. The raw value (mean |predicted
                        probability - realized outcome| in-bucket) is
                        exposed when computable; it never gates in_support
                        on its own.
    shift_status    -- UNESTABLISHED: no distribution-shift detector has
                        been built or validated in this repo. Always
                        reports UNESTABLISHED; allowed_states never
                        contains it, so this dimension can never pass.

in_support = market_support_ok AND line_support_ok AND state_support_ok
             AND joint_support_established AND shift_status in ALLOWED_SHIFT_STATES

Because joint_support and shift_status are UNESTABLISHED by honest
necessity (not a placeholder to be quietly loosened later), in_support is
currently always False. This is deliberate and documented, not a bug --
see mission section 5: "If current research does not justify one of these
dimensions: expose the dimension, mark it UNESTABLISHED, let the frozen
policy abstain. Do not silently collapse missing support into PASS."
"""

from dataclasses import dataclass

N_MARKET = 20
N_LINE = 20
N_STATE = 20  # matches candidate_adapter.MIN_CALIBRATION_SLATES_FOR_STATE_SUPPORT

ALLOWED_SHIFT_STATES = frozenset({"STABLE"})  # UNESTABLISHED is never a member -- see module docstring

SUPPORT_RULE_VERSION = "SUPPORT_RULE_V1"


@dataclass(frozen=True)
class CandidateSupport:
    market_support: int
    line_support: int
    state_support: int
    joint_support: str  # "UNESTABLISHED" always, currently -- see module docstring
    recent_support: int
    calibration_error: float | None  # None when not computable (insufficient bucket data)
    shift_status: str  # "UNESTABLISHED" always, currently
    in_support: bool
    support_rule_version: str = SUPPORT_RULE_VERSION


def _mean_abs_calibration_gap(rows: list[dict]) -> float | None:
    diffs = [
        abs(float(r["predictive_probability_if_available"]) - float(r["actual_outcome"]))
        for r in rows
        if r.get("predictive_probability_if_available") is not None and r.get("actual_outcome") is not None
    ]
    return float(sum(diffs) / len(diffs)) if diffs else None


def support_is_structurally_unreachable() -> bool:
    """True iff evaluate_support can NEVER return in_support=True right
    now, regardless of ledger content -- because at least one dimension
    (joint_support, shift_status) is a hard-coded UNESTABLISHED with no
    passing state reachable. Callers may use this as a cheap, honest
    short-circuit (skip expensive candidate enumeration, abstain
    immediately) -- NOT as a way to silently skip computing support
    per-candidate once these dimensions ARE established."""
    return True  # both joint_support and shift_status are UNESTABLISHED below -- update this the day either is retired


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

    market_support = len(market_rows)
    line_support = len(line_rows)
    state_support = independent_slate_count

    # recent_support: same-market rows among the most-recently-admitted
    # `recent_window` observations overall (a real, computable diagnostic;
    # does not gate in_support -- see module docstring).
    recent_rows = sorted(snapshot_rows, key=lambda r: str(r.get("calibration_admitted_at", "")))[-recent_window:]
    recent_support = sum(1 for r in recent_rows if r.get("market_bucket") == market_bucket)

    calibration_error = _mean_abs_calibration_gap(market_rows)

    market_support_ok = market_support >= N_MARKET
    line_support_ok = line_support >= N_LINE
    state_support_ok = state_support >= N_STATE
    joint_support = "UNESTABLISHED"
    shift_status = "UNESTABLISHED"

    in_support = bool(
        market_support_ok
        and line_support_ok
        and state_support_ok
        and joint_support != "UNESTABLISHED"  # currently always blocks -- see module docstring
        and shift_status in ALLOWED_SHIFT_STATES  # currently always blocks -- see module docstring
    )

    return CandidateSupport(
        market_support=market_support,
        line_support=line_support,
        state_support=state_support,
        joint_support=joint_support,
        recent_support=recent_support,
        calibration_error=calibration_error,
        shift_status=shift_status,
        in_support=in_support,
    )
