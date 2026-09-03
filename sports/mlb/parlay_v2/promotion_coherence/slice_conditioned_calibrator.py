"""Slice-conditioned joint-probability calibrator.

The negative-slope finding from `pair_ledger_calibration.py` says the
raw joint model's confidence is anti-informative on the aggregated
ledger. Slicing by market pair type (see `discriminative_power.py`)
splits the picture: TB-heavy slices are strongly inverted (AUC 0.27-
0.41), R|R is flat (AUC 0.53), no slice is clearly-positive. A single
global calibrator either inverts everywhere or ignores everywhere; a
slice-conditioned calibrator fits a per-slice sub-calibrator and
falls back to the global fit for keys with too little data.

This module is used exactly like `BetaCalibrator`. Its `calibrate(p)`
method requires a row, not a bare float, so it can pick the right
slice; there is a companion `calibrate_raw(p, row)` that a caller
with a row hands in. For call sites that only carry a bare
probability (e.g. today's normal-parlay coherence gate that reads
the payload's overlay), the classic `BetaCalibrator` still applies.

Read-only. No live-selector import.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from .pair_ledger_calibration import BetaCalibrator, fit_beta_calibrator


def default_slice_key(row: Mapping[str, Any]) -> str:
    """The default slice key: `market_pair_type`. Rows without one
    fall into an "UNKNOWN" bucket which shares the global calibrator."""
    return str(row.get("market_pair_type") or "UNKNOWN")


@dataclass(frozen=True)
class SliceConditionedCalibrator:
    """Dispatches calibration by a per-row slice key.

    `per_slice` maps a slice key to its BetaCalibrator; `global_fit`
    is the fallback for slice keys not in the map (or for rows whose
    slice does not have enough data to fit reliably -- see
    `min_slice_pairs`).

    Callers who have a row use `.calibrate_from_row(p, row)`. Callers
    who only have a bare probability get the `global_fit.calibrate(p)`
    via `.calibrate(p)`; that gracefully lets this class be used as a
    drop-in `JointProbabilityCalibrator` even when the call site does
    not carry a row.
    """

    per_slice: dict[str, BetaCalibrator]
    global_fit: BetaCalibrator
    slice_key_fn: Callable[[Mapping[str, Any]], str] = default_slice_key
    min_slice_pairs: int = 100

    def calibrate(self, p: float) -> float:
        return self.global_fit.calibrate(p)

    def calibrate_from_row(self, p: float, row: Mapping[str, Any]) -> float:
        key = self.slice_key_fn(row)
        cal = self.per_slice.get(key)
        if cal is None or cal.n_fitted_pairs < self.min_slice_pairs:
            return self.global_fit.calibrate(p)
        return cal.calibrate(p)

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_fit": {
                "slope": self.global_fit.slope,
                "intercept": self.global_fit.intercept,
                "n_fitted_pairs": self.global_fit.n_fitted_pairs,
            },
            "per_slice": {
                key: {
                    "slope": cal.slope,
                    "intercept": cal.intercept,
                    "n_fitted_pairs": cal.n_fitted_pairs,
                }
                for key, cal in sorted(self.per_slice.items())
            },
            "min_slice_pairs": self.min_slice_pairs,
        }


def fit_slice_conditioned_calibrator(
    rows: Iterable[Mapping[str, Any]],
    *,
    slice_key_fn: Callable[[Mapping[str, Any]], str] = default_slice_key,
    min_slice_pairs: int = 100,
) -> SliceConditionedCalibrator:
    """Fit one BetaCalibrator per unique slice key and one global
    fallback. Slices with fewer than `min_slice_pairs` rows still get
    a fit stored but the dispatcher won't use them at inference --
    they're kept for transparency / future re-consideration.

    Never mutates the input rows.
    """
    rows_list = list(rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows_list:
        groups[slice_key_fn(r)].append(r)
    per_slice: dict[str, BetaCalibrator] = {
        key: fit_beta_calibrator(sub_rows) for key, sub_rows in groups.items()
    }
    global_fit = fit_beta_calibrator(rows_list)
    return SliceConditionedCalibrator(
        per_slice=per_slice,
        global_fit=global_fit,
        slice_key_fn=slice_key_fn,
        min_slice_pairs=min_slice_pairs,
    )
