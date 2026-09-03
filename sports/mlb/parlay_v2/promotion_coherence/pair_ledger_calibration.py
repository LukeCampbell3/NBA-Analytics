"""Fit and apply a beta calibrator to the pair-observation ledger's
predicted joint probability.

Purpose. The pool-gap investigation on this branch (see
`investigate_pool_gap.py`, and BACKTEST_ANALYSIS.md) established that
the real pair ledger's joint model over-predicts hit rate by ~12 pp
per decile -- the model is miscalibrated, not merely pessimistic.
The promotion-coherence proposal (Item 6) already names beta
calibration as the recommended fix.

What this module does.

    * Fits a beta calibrator in log-odds space:
          logit(p_calibrated) = slope * logit(p_raw) + intercept
      Two parameters, minimized via BFGS against binary cross-entropy
      on (predicted, both_win) pairs. Small, robust, interpretable.

    * Evaluates the calibrator honestly under leave-one-slate-out
      cross-validation. For each of the N slates, the calibrator is
      fitted on the other N-1 and scored on the held-out slate; per-
      decile calibration gap is reported for held-out slates only.
      This is the number that says whether the calibrator would help
      NEW slates, not just the slates it was fit on.

    * Provides `apply_calibrator_to_row(row, calibrator)` and a
      `JointProbabilityCalibrator` protocol so the promotion-coherence
      layer can consume a calibrated joint transparently -- the raw
      pair-observation row is never mutated.

Scope preserved. Read-only. Writes only under this subpackage. No
live-selector import. `PromotionConfidenceComponents` gains an
optional `joint_calibrator` argument in a follow-up commit; this
module by itself changes nothing about live behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"
DEFAULT_REPORT = "sports/mlb/parlay_v2/promotion_coherence/reports/pair_ledger_calibration.json"

# Numerical safety margins for logit / sigmoid so a p == 0 or p == 1
# never becomes ±inf.
_P_EPS = 1e-6


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clip_probability(p: float) -> float:
    if p < _P_EPS:
        return _P_EPS
    if p > 1.0 - _P_EPS:
        return 1.0 - _P_EPS
    return p


def _logit(p: float) -> float:
    q = _clip_probability(p)
    return math.log(q / (1.0 - q))


def _sigmoid(z: float) -> float:
    # numerically stable
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# --- calibrator ---------------------------------------------------------

class JointProbabilityCalibrator(Protocol):
    """Anything a caller can `.calibrate(p)` to get a mapped probability
    back. Beta calibrator conforms; a future isotonic implementation
    will too; a "no-op" identity calibrator also does. Keeping the
    protocol narrow so the promotion-coherence layer never needs to
    reach into calibrator internals."""

    def calibrate(self, p: float) -> float:  # pragma: no cover
        ...


@dataclass(frozen=True)
class BetaCalibrator:
    """logit(p_out) = slope * logit(p_in) + intercept.

    Two parameters, fitted from data. `n_fitted_pairs` is preserved so
    a caller can tell a well-fit calibrator (fitted on hundreds+ of
    pairs) from a nearly-vacuous one (fitted on a handful).
    """

    slope: float
    intercept: float
    n_fitted_pairs: int

    def calibrate(self, p: float) -> float:
        z = self.slope * _logit(p) + self.intercept
        return _sigmoid(z)


class IdentityCalibrator:
    """No-op calibrator -- returns its input unchanged. Useful as a
    default so the coherence-gate call path is uniform whether or not
    the caller supplies a real calibrator."""

    def calibrate(self, p: float) -> float:
        return _clip_probability(p)


def _fit_data(rows: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[int] = []
    for row in rows:
        p = _finite(row.get("predicted_joint_probability"))
        if p is None or not (0.0 < p < 1.0):
            continue
        both_win = row.get("both_win")
        if both_win is None:
            continue
        xs.append(_logit(p))
        ys.append(1 if both_win else 0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def fit_beta_calibrator(rows: Iterable[dict[str, Any]]) -> BetaCalibrator:
    """Fit a beta calibrator by minimizing binary cross-entropy in
    log-odds space. Two parameters. Small (~4KB memory), robust on
    a few hundred pairs, and interpretable -- slope > 1 means the
    calibrator sharpens confidence, < 1 shrinks it; intercept shifts
    the overall base rate.
    """
    xs, ys = _fit_data(rows)
    n = int(xs.shape[0])
    if n == 0:
        # Falling back to the identity is the honest answer -- with
        # zero fittable rows we cannot honestly fit anything.
        return BetaCalibrator(slope=1.0, intercept=0.0, n_fitted_pairs=0)

    def _nll(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        z = a * xs + b
        # log(sigmoid(z)) and log(1 - sigmoid(z)) computed stably as
        # softplus expressions.
        # log sigmoid(z) = -softplus(-z), log(1 - sigmoid(z)) = -softplus(z)
        neg_z = -z
        # np.logaddexp is stable
        log_sig = -np.logaddexp(0.0, neg_z)      # log(sigmoid(z))
        log_1msig = -np.logaddexp(0.0, z)         # log(1 - sigmoid(z))
        return float(-np.sum(ys * log_sig + (1.0 - ys) * log_1msig))

    result = minimize(_nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    slope, intercept = float(result.x[0]), float(result.x[1])
    return BetaCalibrator(slope=slope, intercept=intercept, n_fitted_pairs=n)


def apply_calibrator_to_row(
    row: dict[str, Any], calibrator: JointProbabilityCalibrator,
) -> Optional[float]:
    """Return the calibrated joint probability for a row, or None
    when the row has no valid raw joint probability."""
    p = _finite(row.get("predicted_joint_probability"))
    if p is None:
        return None
    return calibrator.calibrate(p)


# --- honest evaluation --------------------------------------------------

@dataclass
class DecileCalibrationRow:
    decile_index: int
    decile_lower: float
    decile_upper: float
    n_pairs: int
    mean_predicted: Optional[float]
    mean_calibrated: Optional[float]
    mean_actual: Optional[float]
    gap_raw: Optional[float]        # predicted - actual (+ over-predicts)
    gap_calibrated: Optional[float]  # calibrated - actual


def _bin_by_probability(values: list[float]) -> list[int]:
    """Assign each probability to a decile bucket 0..9 with edges
    [0, 0.1, 0.2, ..., 1.0]. Values >= 1 land in the last bucket."""
    return [min(9, max(0, int(v * 10))) for v in values]


def calibration_by_decile(
    predicted: list[float],
    calibrated: list[float],
    actuals: list[int],
) -> list[DecileCalibrationRow]:
    assert len(predicted) == len(calibrated) == len(actuals)
    buckets: dict[int, dict[str, list]] = {
        i: {"pred": [], "cal": [], "act": []} for i in range(10)
    }
    for p, c, y in zip(predicted, calibrated, actuals):
        b = min(9, max(0, int(p * 10)))
        buckets[b]["pred"].append(p)
        buckets[b]["cal"].append(c)
        buckets[b]["act"].append(y)

    out: list[DecileCalibrationRow] = []
    for i in range(10):
        b = buckets[i]
        n = len(b["pred"])
        if n == 0:
            out.append(DecileCalibrationRow(
                decile_index=i, decile_lower=i / 10, decile_upper=(i + 1) / 10,
                n_pairs=0, mean_predicted=None, mean_calibrated=None,
                mean_actual=None, gap_raw=None, gap_calibrated=None,
            ))
            continue
        mp = statistics.fmean(b["pred"])
        mc = statistics.fmean(b["cal"])
        ma = statistics.fmean(b["act"])
        out.append(DecileCalibrationRow(
            decile_index=i, decile_lower=i / 10, decile_upper=(i + 1) / 10,
            n_pairs=n, mean_predicted=mp, mean_calibrated=mc,
            mean_actual=ma, gap_raw=mp - ma, gap_calibrated=mc - ma,
        ))
    return out


@dataclass
class LeaveOneSlateOutFold:
    held_out_slate: str
    train_slate_count: int
    train_pair_count: int
    held_out_pair_count: int
    calibrator: dict[str, Any]  # slope, intercept, n_fitted_pairs
    mean_gap_raw: Optional[float]
    mean_gap_calibrated: Optional[float]
    decile_rows: list[DecileCalibrationRow] = field(default_factory=list)


@dataclass
class CalibrationReport:
    generated_at_utc: str
    ledger_path: str
    total_settled_rows: int
    slates_covered: list[str]
    global_calibrator: dict[str, Any]
    in_sample_mean_gap_raw: Optional[float]
    in_sample_mean_gap_calibrated: Optional[float]
    leave_one_slate_out_folds: list[LeaveOneSlateOutFold]
    oos_mean_gap_raw: Optional[float]
    oos_mean_gap_calibrated: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "ledger_path": self.ledger_path,
            "total_settled_rows": self.total_settled_rows,
            "slates_covered": self.slates_covered,
            "global_calibrator": self.global_calibrator,
            "in_sample_mean_gap_raw": self.in_sample_mean_gap_raw,
            "in_sample_mean_gap_calibrated": self.in_sample_mean_gap_calibrated,
            "leave_one_slate_out_folds": [
                {
                    "held_out_slate": f.held_out_slate,
                    "train_slate_count": f.train_slate_count,
                    "train_pair_count": f.train_pair_count,
                    "held_out_pair_count": f.held_out_pair_count,
                    "calibrator": f.calibrator,
                    "mean_gap_raw": f.mean_gap_raw,
                    "mean_gap_calibrated": f.mean_gap_calibrated,
                    "decile_rows": [
                        {
                            "decile_index": d.decile_index,
                            "decile_lower": d.decile_lower,
                            "decile_upper": d.decile_upper,
                            "n_pairs": d.n_pairs,
                            "mean_predicted": d.mean_predicted,
                            "mean_calibrated": d.mean_calibrated,
                            "mean_actual": d.mean_actual,
                            "gap_raw": d.gap_raw,
                            "gap_calibrated": d.gap_calibrated,
                        }
                        for d in f.decile_rows
                    ],
                }
                for f in self.leave_one_slate_out_folds
            ],
            "oos_mean_gap_raw": self.oos_mean_gap_raw,
            "oos_mean_gap_calibrated": self.oos_mean_gap_calibrated,
        }


def _load_settled_rows(ledger_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not ledger_path.exists():
        return rows
    with open(ledger_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("settlement_status") == "settled":
                rows.append(r)
    return rows


def _mean_decile_gap(decile_rows: list[DecileCalibrationRow], attr: str) -> Optional[float]:
    values = [getattr(d, attr) for d in decile_rows if getattr(d, attr) is not None]
    if not values:
        return None
    return statistics.fmean(values)


def build_report(*, ledger_path: Path) -> CalibrationReport:
    rows = _load_settled_rows(ledger_path)
    slates = sorted({r.get("slate_id") for r in rows if r.get("slate_id")})

    global_calibrator = fit_beta_calibrator(rows)

    # In-sample calibration (fitted on all rows, evaluated on all rows).
    # Reported alongside OOS so an auditor sees the gap between the two
    # honestly.
    in_pred = [
        _finite(r.get("predicted_joint_probability"))
        for r in rows
        if _finite(r.get("predicted_joint_probability")) is not None
    ]
    in_actuals = [1 if r.get("both_win") else 0 for r in rows]
    # Align lengths -- filter rows by validity in one pass:
    in_pairs = [
        (_finite(r.get("predicted_joint_probability")), 1 if r.get("both_win") else 0)
        for r in rows
        if _finite(r.get("predicted_joint_probability")) is not None
    ]
    in_pred = [p for p, _ in in_pairs]
    in_actuals = [y for _, y in in_pairs]
    in_calibrated = [global_calibrator.calibrate(p) for p in in_pred]
    in_deciles = calibration_by_decile(in_pred, in_calibrated, in_actuals)

    # Leave-one-slate-out.
    folds: list[LeaveOneSlateOutFold] = []
    oos_gap_raw_values: list[float] = []
    oos_gap_calibrated_values: list[float] = []
    for held in slates:
        train_rows = [r for r in rows if r.get("slate_id") != held]
        held_rows = [r for r in rows if r.get("slate_id") == held]
        cal = fit_beta_calibrator(train_rows)
        held_pairs = [
            (_finite(r.get("predicted_joint_probability")), 1 if r.get("both_win") else 0)
            for r in held_rows
            if _finite(r.get("predicted_joint_probability")) is not None
        ]
        pred = [p for p, _ in held_pairs]
        actuals = [y for _, y in held_pairs]
        calibrated = [cal.calibrate(p) for p in pred]
        deciles = calibration_by_decile(pred, calibrated, actuals)
        mean_raw = _mean_decile_gap(deciles, "gap_raw")
        mean_cal = _mean_decile_gap(deciles, "gap_calibrated")
        if mean_raw is not None:
            oos_gap_raw_values.append(mean_raw)
        if mean_cal is not None:
            oos_gap_calibrated_values.append(mean_cal)
        folds.append(LeaveOneSlateOutFold(
            held_out_slate=held,
            train_slate_count=len(slates) - 1,
            train_pair_count=len(train_rows),
            held_out_pair_count=len(held_rows),
            calibrator={
                "slope": cal.slope, "intercept": cal.intercept,
                "n_fitted_pairs": cal.n_fitted_pairs,
            },
            mean_gap_raw=mean_raw,
            mean_gap_calibrated=mean_cal,
            decile_rows=deciles,
        ))

    return CalibrationReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        ledger_path=str(ledger_path),
        total_settled_rows=len(rows),
        slates_covered=slates,
        global_calibrator={
            "slope": global_calibrator.slope,
            "intercept": global_calibrator.intercept,
            "n_fitted_pairs": global_calibrator.n_fitted_pairs,
        },
        in_sample_mean_gap_raw=_mean_decile_gap(in_deciles, "gap_raw"),
        in_sample_mean_gap_calibrated=_mean_decile_gap(in_deciles, "gap_calibrated"),
        leave_one_slate_out_folds=folds,
        oos_mean_gap_raw=statistics.fmean(oos_gap_raw_values) if oos_gap_raw_values else None,
        oos_mean_gap_calibrated=statistics.fmean(oos_gap_calibrated_values) if oos_gap_calibrated_values else None,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fit a beta calibrator on the real pair-observation ledger; evaluate under leave-one-slate-out.")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_report(ledger_path=args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")
    print(f"settled rows: {report.total_settled_rows}  slates: {len(report.slates_covered)}")
    g = report.global_calibrator
    print(f"global beta calibrator: slope={g['slope']:+.4f} intercept={g['intercept']:+.4f} n_fitted={g['n_fitted_pairs']}")
    print(f"IN-SAMPLE mean per-decile calibration gap: raw {report.in_sample_mean_gap_raw:+.4f} -> calibrated {report.in_sample_mean_gap_calibrated:+.4f}")
    if report.oos_mean_gap_raw is not None and report.oos_mean_gap_calibrated is not None:
        print(f"LEAVE-ONE-SLATE-OUT mean per-decile calibration gap: raw {report.oos_mean_gap_raw:+.4f} -> calibrated {report.oos_mean_gap_calibrated:+.4f}")
        for fold in report.leave_one_slate_out_folds:
            raw = f"{fold.mean_gap_raw:+.4f}" if fold.mean_gap_raw is not None else " n/a  "
            cal = f"{fold.mean_gap_calibrated:+.4f}" if fold.mean_gap_calibrated is not None else " n/a  "
            print(f"  held out {fold.held_out_slate}: raw {raw} -> calibrated {cal}  "
                  f"(train n={fold.train_pair_count}, held n={fold.held_out_pair_count})")


if __name__ == "__main__":
    _cli()
