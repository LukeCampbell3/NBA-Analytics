"""Tests for the beta calibrator that resolves the pair-ledger
miscalibration.

Structure:

    * Unit: math primitives (logit/sigmoid), the calibrator dataclass,
      IdentityCalibrator, apply_calibrator_to_row.
    * Fit-quality: synthetic 3-slate fixture where the calibrator MUST
      converge to slope 1 / intercept 0 on already-calibrated data,
      and MUST shrink the gap on deliberately-biased data.
    * Real-data regressions on the current pair ledger: in-sample gap
      shrinks by an order of magnitude, leave-one-slate-out (LOSO) OOS
      gap stays under 2 pp on average.
    * Wiring: promotion_confidence_components accepts a calibrator and
      substitutes the calibrated joint into promotion_margin correctly.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from sports.mlb.parlay_v2.promotion_coherence import (
    BetaCalibrator,
    IdentityCalibrator,
    apply_calibrator_to_row,
    decide_coherent_promotion,
    fit_beta_calibrator,
    promotion_confidence_components,
)
from sports.mlb.parlay_v2.promotion_coherence.pair_ledger_calibration import (
    DEFAULT_LEDGER,
    _clip_probability,
    _logit,
    _sigmoid,
    build_report,
    calibration_by_decile,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_PATH = REPO_ROOT / DEFAULT_LEDGER


# --- unit: math primitives ---------------------------------------------

def test_clip_probability_bounds_open_interval() -> None:
    assert 0.0 < _clip_probability(0.0) < 0.5
    assert 0.5 < _clip_probability(1.0) < 1.0
    assert _clip_probability(0.5) == pytest.approx(0.5)


def test_logit_and_sigmoid_are_inverses() -> None:
    for p in [0.01, 0.25, 0.5, 0.75, 0.99]:
        assert _sigmoid(_logit(p)) == pytest.approx(p, abs=1e-6)


def test_sigmoid_is_numerically_stable_at_extremes() -> None:
    assert _sigmoid(1000) == pytest.approx(1.0)
    assert _sigmoid(-1000) == pytest.approx(0.0)


# --- unit: calibrator classes -------------------------------------------

def test_identity_calibrator_returns_input_unchanged() -> None:
    c = IdentityCalibrator()
    for p in [0.05, 0.4, 0.9]:
        assert c.calibrate(p) == pytest.approx(p)


def test_identity_calibrator_clips_extremes() -> None:
    c = IdentityCalibrator()
    assert 0 < c.calibrate(0.0) < 1
    assert 0 < c.calibrate(1.0) < 1


def test_beta_calibrator_identity_when_slope_1_intercept_0() -> None:
    c = BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=100)
    for p in [0.1, 0.3, 0.7]:
        assert c.calibrate(p) == pytest.approx(p, abs=1e-6)


def test_beta_calibrator_shrinks_when_slope_below_1() -> None:
    c = BetaCalibrator(slope=0.5, intercept=0.0, n_fitted_pairs=100)
    # Shrinking toward the 0.5 mid-point.
    assert 0.5 < c.calibrate(0.9) < 0.9
    assert 0.1 < c.calibrate(0.1) < 0.5


def test_apply_calibrator_to_row_returns_none_on_bad_row() -> None:
    c = IdentityCalibrator()
    assert apply_calibrator_to_row({}, c) is None
    assert apply_calibrator_to_row({"predicted_joint_probability": "n/a"}, c) is None


def test_apply_calibrator_to_row_returns_calibrated_value() -> None:
    c = BetaCalibrator(slope=0.5, intercept=-1.0, n_fitted_pairs=100)
    row = {"predicted_joint_probability": 0.4}
    out = apply_calibrator_to_row(row, c)
    assert out is not None
    assert out == pytest.approx(_sigmoid(0.5 * _logit(0.4) - 1.0))


# --- fit-quality on synthetic fixtures -----------------------------------

def _make_synthetic_rows(mapping: list[tuple[float, int]]) -> list[dict]:
    """Turn [(predicted, outcome), ...] into pair-observation-shaped rows."""
    return [
        {"predicted_joint_probability": p, "both_win": bool(y)}
        for p, y in mapping
    ]


def test_calibrator_fit_on_already_calibrated_data_gives_near_identity() -> None:
    """Deterministic fixture where predicted == P(win) exactly. The
    fitted calibrator should be close to slope 1, intercept 0."""
    rng = random.Random(0)
    rows = []
    for _ in range(2000):
        p = rng.uniform(0.1, 0.9)
        # Force actual = 1 with probability p using deterministic seed
        y = 1 if rng.random() < p else 0
        rows.append({"predicted_joint_probability": p, "both_win": bool(y)})
    cal = fit_beta_calibrator(rows)
    # Won't be exactly 1.0 / 0.0 on finite noisy data, but should be
    # close.
    assert cal.slope == pytest.approx(1.0, abs=0.15)
    assert cal.intercept == pytest.approx(0.0, abs=0.15)


def test_calibrator_shrinks_deliberately_over_predicted_probabilities() -> None:
    """Predicted probability is uniformly 0.4 but actual hit rate is
    0.1. The calibrator MUST push the output below the input."""
    rng = random.Random(1)
    rows = []
    for _ in range(2000):
        p = 0.4
        y = 1 if rng.random() < 0.1 else 0
        rows.append({"predicted_joint_probability": p, "both_win": bool(y)})
    cal = fit_beta_calibrator(rows)
    calibrated = cal.calibrate(0.4)
    # Actual base rate ~0.1; calibrated output should be nearby.
    assert calibrated == pytest.approx(0.1, abs=0.03)
    assert calibrated < 0.4


def test_calibrator_on_empty_rows_returns_identity() -> None:
    cal = fit_beta_calibrator([])
    assert cal.slope == 1.0 and cal.intercept == 0.0 and cal.n_fitted_pairs == 0


def test_calibrator_records_n_fitted_pairs_for_provenance() -> None:
    rows = _make_synthetic_rows([(0.5, 1)] * 42)
    cal = fit_beta_calibrator(rows)
    assert cal.n_fitted_pairs == 42


def test_calibration_by_decile_reports_gap_columns() -> None:
    predicted = [0.05, 0.15, 0.25, 0.35]
    calibrated = [0.05, 0.10, 0.10, 0.10]
    actuals = [0, 0, 1, 0]
    rows = calibration_by_decile(predicted, calibrated, actuals)
    # Bucket 0: 1 row, mean_predicted 0.05, actual 0.0 -> gap_raw +0.05
    assert rows[0].n_pairs == 1
    assert rows[0].gap_raw == pytest.approx(0.05)
    assert rows[0].gap_calibrated == pytest.approx(0.05)
    # Bucket 3: 1 row, mean_predicted 0.35, actual 0.0 -> gap_raw +0.35
    assert rows[3].n_pairs == 1
    assert rows[3].gap_raw == pytest.approx(0.35)


# --- real-ledger regressions --------------------------------------------

@pytest.fixture(scope="module")
def real_ledger_report():
    if not LEDGER_PATH.exists():
        pytest.skip(f"real pair ledger missing: {LEDGER_PATH}")
    return build_report(ledger_path=LEDGER_PATH)


def test_real_ledger_beta_calibrator_shrinks_in_sample_gap_by_order_of_magnitude(real_ledger_report) -> None:
    """Whole-of-branch headline test. Raw calibration gap is ~+0.12;
    after fitting the beta calibrator in-sample, mean per-decile gap
    must drop below 2 pp. That's the "resolved" number."""
    raw = real_ledger_report.in_sample_mean_gap_raw
    cal = real_ledger_report.in_sample_mean_gap_calibrated
    assert raw is not None and cal is not None
    assert raw > 0.05, f"raw gap {raw:+.4f} smaller than expected -- ledger may have changed"
    assert abs(cal) < 0.02, (
        f"in-sample calibrated gap {cal:+.4f} did not shrink below 2 pp; "
        f"beta calibrator may have failed to converge"
    )
    # And a meaningful shrinkage: the calibrated |gap| must be at least
    # 5x smaller than the raw |gap|.
    assert abs(cal) * 5 < abs(raw)


def test_leave_one_slate_out_calibration_gap_stays_under_2pp(real_ledger_report) -> None:
    """Honest OOS check. The calibrator must generalize -- fit on 3
    slates, evaluated on 1 held-out slate, mean per-decile calibration
    gap must average below 2 pp across the 4 folds. Anything looser
    would suggest overfitting."""
    raw = real_ledger_report.oos_mean_gap_raw
    cal = real_ledger_report.oos_mean_gap_calibrated
    assert raw is not None and cal is not None
    assert raw > 0.05
    assert abs(cal) < 0.02, (
        f"leave-one-slate-out mean calibrated gap {cal:+.4f} exceeds 2 pp; "
        f"calibrator may be overfitting or a slate's regime shifted"
    )


def test_leave_one_slate_out_holds_up_on_every_fold(real_ledger_report) -> None:
    """No individual held-out slate should have a calibrated |gap| >
    5 pp -- if one does, the calibrator is failing on that particular
    regime and the regression deserves an explicit look."""
    for fold in real_ledger_report.leave_one_slate_out_folds:
        assert fold.mean_gap_calibrated is not None
        assert abs(fold.mean_gap_calibrated) < 0.05, (
            f"held-out slate {fold.held_out_slate}: calibrated gap "
            f"{fold.mean_gap_calibrated:+.4f} exceeds 5 pp"
        )


def test_global_calibrator_slope_indicates_narrow_range_correction(real_ledger_report) -> None:
    """The current fit produces a NEGATIVE slope on this ledger --
    within the narrow predicted-joint range [0.14, 0.29], the raw
    model's confidence has negative correlation with actual outcome.
    That is itself a finding worth pinning: it says the joint model
    is not merely mis-scaled, it is inverted in the range it operates.
    If this ever flips positive, the model has learned real signal in
    the interim and this pin (and the analysis doc) get an update."""
    g = real_ledger_report.global_calibrator
    assert g["slope"] < 0, (
        f"global calibrator slope is {g['slope']:+.4f}; if positive, the model has "
        f"real signal now -- rerun analysis"
    )
    assert g["n_fitted_pairs"] == real_ledger_report.total_settled_rows


# --- wiring: promotion_confidence_components with a calibrator ---------

def _payload_with_joint(joint: float, price: float = 5.0) -> dict:
    return {
        "parlays": {
            "eligible": True, "action": "ACT",
            "public_quality_overlay": {
                "action": "ABSTAIN",
                "joint_probability": joint,
                "combined_decimal_price": price,
                "leg_probabilities": [0.55, 0.55],
            },
        },
    }


def test_promotion_confidence_components_default_uses_raw_joint() -> None:
    comps = promotion_confidence_components(_payload_with_joint(0.30))
    assert comps.calibrated_joint_probability == pytest.approx(0.30)


def test_promotion_confidence_components_applies_calibrator_when_supplied() -> None:
    # Calibrator that halves the log-odds
    cal = BetaCalibrator(slope=0.5, intercept=0.0, n_fitted_pairs=1)
    comps = promotion_confidence_components(_payload_with_joint(0.30), joint_calibrator=cal)
    expected = _sigmoid(0.5 * _logit(0.30))
    assert comps.calibrated_joint_probability == pytest.approx(expected)


def test_promotion_margin_reflects_calibrated_joint_not_raw() -> None:
    """A row that would pass a margin-0.05 floor on raw joint (0.30)
    fails it on a calibrator that shrinks joint below break-even."""
    payload = _payload_with_joint(0.30, price=4.0)  # break-even 0.25
    raw = promotion_confidence_components(payload)
    assert raw.promotion_margin == pytest.approx(0.30 - 0.25)  # +0.05

    aggressive = BetaCalibrator(slope=0.1, intercept=-1.5, n_fitted_pairs=1)
    calibrated = promotion_confidence_components(payload, joint_calibrator=aggressive)
    assert calibrated.promotion_margin < raw.promotion_margin


def test_decide_coherent_promotion_threads_calibrator_through() -> None:
    payload = _payload_with_joint(0.55, price=2.5)  # raw margin: 0.55-0.4=0.15
    # No calibrator -> passes overlay ABSTAIN check but margin fine
    baseline = decide_coherent_promotion(payload)
    # Aggressive calibrator that pulls 0.55 well below 0.40
    aggressive = BetaCalibrator(slope=0.1, intercept=-2.0, n_fitted_pairs=1)
    with_cal = decide_coherent_promotion(payload, joint_calibrator=aggressive)
    assert (with_cal.components.calibrated_joint_probability
            < baseline.components.calibrated_joint_probability)
    # A margin block appears in `with_cal` that isn't in `baseline`
    # (or both are ABSTAIN for other reasons -- the overlay is still
    # ABSTAIN in this fixture).
    assert baseline.action == "ABSTAIN"
    assert with_cal.action == "ABSTAIN"
