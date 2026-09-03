"""Tests for the slice-conditioned calibrator.

Two purposes:
    * Unit: dispatcher picks the correct per-slice sub-calibrator,
      falls back to the global fit on unknown slices and slices below
      the min_slice_pairs floor.
    * Real-ledger regression: on the current pair ledger, the per-
      market-pair-type sub-calibrators have DIFFERENT slopes (positive
      for R|R, negative for TB-heavy) -- proving the slice conditioning
      is not a no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence.pair_ledger_calibration import BetaCalibrator
from sports.mlb.parlay_v2.promotion_coherence.slice_conditioned_calibrator import (
    SliceConditionedCalibrator,
    default_slice_key,
    fit_slice_conditioned_calibrator,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER = REPO_ROOT / "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"


def _syn_rows(market: str, mapping: list[tuple[float, int]]) -> list[dict]:
    return [
        {"predicted_joint_probability": p, "both_win": bool(y), "market_pair_type": market}
        for p, y in mapping
    ]


# --- unit ----------------------------------------------------------------

def test_default_slice_key_reads_market_pair_type() -> None:
    assert default_slice_key({"market_pair_type": "R|R"}) == "R|R"
    assert default_slice_key({}) == "UNKNOWN"


def test_dispatch_uses_per_slice_calibrator_when_present() -> None:
    per_slice = {
        "A": BetaCalibrator(slope=0.0, intercept=+10.0, n_fitted_pairs=200),
        "B": BetaCalibrator(slope=0.0, intercept=-10.0, n_fitted_pairs=200),
    }
    global_fit = BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=1000)
    cal = SliceConditionedCalibrator(
        per_slice=per_slice, global_fit=global_fit,
        slice_key_fn=default_slice_key, min_slice_pairs=100,
    )
    # slice "A" -> always ~1.0
    assert cal.calibrate_from_row(0.3, {"market_pair_type": "A"}) > 0.99
    # slice "B" -> always ~0.0
    assert cal.calibrate_from_row(0.3, {"market_pair_type": "B"}) < 0.01


def test_dispatch_falls_back_when_slice_missing() -> None:
    per_slice = {"A": BetaCalibrator(slope=0.0, intercept=+10.0, n_fitted_pairs=200)}
    global_fit = BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=1000)
    cal = SliceConditionedCalibrator(
        per_slice=per_slice, global_fit=global_fit,
        slice_key_fn=default_slice_key, min_slice_pairs=100,
    )
    # Unknown slice -> global identity fit on p=0.3 returns 0.3
    assert cal.calibrate_from_row(0.3, {"market_pair_type": "UNKNOWN_SLICE"}) == pytest.approx(0.3, abs=1e-6)


def test_dispatch_falls_back_when_slice_too_thin() -> None:
    per_slice = {"A": BetaCalibrator(slope=0.0, intercept=+10.0, n_fitted_pairs=20)}
    global_fit = BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=1000)
    cal = SliceConditionedCalibrator(
        per_slice=per_slice, global_fit=global_fit,
        slice_key_fn=default_slice_key, min_slice_pairs=100,
    )
    # Slice A has n_fitted=20 which is below min_slice_pairs=100 -> global fallback
    assert cal.calibrate_from_row(0.3, {"market_pair_type": "A"}) == pytest.approx(0.3, abs=1e-6)


def test_bare_calibrate_uses_global_fit() -> None:
    per_slice = {"A": BetaCalibrator(slope=0.0, intercept=+10.0, n_fitted_pairs=200)}
    global_fit = BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=1000)
    cal = SliceConditionedCalibrator(
        per_slice=per_slice, global_fit=global_fit,
        slice_key_fn=default_slice_key, min_slice_pairs=100,
    )
    # Without a row, only the global fit is used.
    assert cal.calibrate(0.3) == pytest.approx(0.3, abs=1e-6)


def test_fit_produces_a_calibrator_per_unique_slice() -> None:
    rows = (
        _syn_rows("A", [(0.5, 1)] * 100)
        + _syn_rows("B", [(0.5, 0)] * 150)
        + _syn_rows("C", [(0.3, 1), (0.7, 0)] * 60)
    )
    cal = fit_slice_conditioned_calibrator(rows)
    assert set(cal.per_slice.keys()) == {"A", "B", "C"}
    assert cal.global_fit.n_fitted_pairs == 100 + 150 + 120  # 370


def test_as_dict_serializes_per_slice_and_global() -> None:
    rows = (
        _syn_rows("A", [(0.5, 1)] * 100)
        + _syn_rows("B", [(0.5, 0)] * 100)
    )
    d = fit_slice_conditioned_calibrator(rows).as_dict()
    assert "global_fit" in d and "per_slice" in d and "min_slice_pairs" in d
    assert set(d["per_slice"].keys()) == {"A", "B"}


# --- real-ledger regression --------------------------------------------

def _load_real():
    import json
    if not LEDGER.exists():
        return None
    rows = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("settlement_status") == "settled":
                rows.append(r)
    return rows


def test_real_ledger_slice_calibrators_differ_by_market() -> None:
    rows = _load_real()
    if rows is None:
        pytest.skip("real ledger missing")
    cal = fit_slice_conditioned_calibrator(rows)
    # All three real markets should be present with n >= 100
    slopes = {k: v.slope for k, v in cal.per_slice.items() if v.n_fitted_pairs >= 100}
    assert "R|R" in slopes and "R|TB" in slopes and "TB|TB" in slopes
    # Slopes must be genuinely different -- if they were all identical,
    # the slice conditioning would be pointless.
    unique_slopes = sorted(set(round(s, 3) for s in slopes.values()))
    assert len(unique_slopes) == 3, (
        f"per-slice slopes collapsed to {unique_slopes}; slice conditioning "
        f"is no longer providing per-market fits."
    )
    # And the R|R slope should be the CLOSEST to zero (or most positive)
    # -- it is the flat/near-signal slice.
    r_r_slope = slopes["R|R"]
    tb_tb_slope = slopes["TB|TB"]
    assert abs(r_r_slope) < abs(tb_tb_slope), (
        f"R|R slope |{r_r_slope}| no longer closer-to-zero than TB|TB "
        f"|{tb_tb_slope}|; slice ordering has changed."
    )
