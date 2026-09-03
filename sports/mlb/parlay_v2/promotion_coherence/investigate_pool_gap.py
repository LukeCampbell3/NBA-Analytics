"""Investigate the 18-percentage-point hit-rate gap between the real
pair pool and the broader synthetic pool.

Context. `BACKTEST_ANALYSIS.md` documents that on the current data the
real pair-observation ledger has a ~7-8% hit rate while the synthetic
cross-game pair ledger (from settled singles) has ~26%. Two consistent
interpretations were named there and neither was picked:

    (A) The production candidate selector is systematically discarding
        good pairs -- the underlying model has predictive value but a
        narrow, poor-quality subpool is what actually gets scored.
    (B) The independence-assumed synthetic joint is optimistic --
        cross-game singles are close to independent, so a broader pool's
        singles-hit-rate-squared naturally matches its observed hit rate,
        while a proper joint model on more-selected pairs is correctly
        pessimistic about correlated failure.

This module instruments both ledgers to distinguish (A) from (B) --
or, if the honest answer is "both, partly," to say that with concrete
numbers.

Data it produces (all reported side-by-side, real vs synthetic):

    * predicted-joint-probability distribution (min / p25 / p50 / p75 / max)
    * per-leg model-probability distribution (synthetic only; real
      ledger does not yet carry per-leg model probability until the
      pair-ingest v1.1 capture matures)
    * calibration bins: predicted-joint-probability decile -> mean
      actual hit rate. If (A) is right, both ledgers show similar
      calibration and the gap is upstream. If (B) is right, the
      synthetic pool over-predicts joint hit rate systematically.
    * combined-price distribution -- confirms both pools are sampling
      similar price regimes so hit-rate comparisons are apples-to-
      apples on that axis at least.

Nothing here mutates any ledger. Read-only, writes a JSON report under
sports/mlb/parlay_v2/promotion_coherence/reports/pool_gap_report.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REAL_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"
DEFAULT_SYNTHETIC_LEDGER = "sports/mlb/parlay_v2/promotion_coherence/reports/synthetic_cross_game_pair_ledger.jsonl"
DEFAULT_REPORT = "sports/mlb/parlay_v2/promotion_coherence/reports/pool_gap_report.json"

# 10 buckets in [0, 1] -- deciles of predicted joint probability.
_DECILE_EDGES = [i / 10 for i in range(11)]


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bucket_index(value: float, edges: list[float]) -> int:
    """Return decile index [0..9] for `value` in [0, 1]. Values <= 0
    land in bucket 0; values >= 1 land in the last bucket."""
    for i in range(len(edges) - 1):
        if value < edges[i + 1]:
            return i
    return len(edges) - 2


@dataclass
class Distribution:
    n: int
    min: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    max: Optional[float]
    mean: Optional[float]

    @classmethod
    def summarize(cls, values: list[float]) -> "Distribution":
        if not values:
            return cls(n=0, min=None, p25=None, p50=None, p75=None, max=None, mean=None)
        vals = sorted(values)
        def pct(p: float) -> float:
            idx = min(len(vals) - 1, int(round((len(vals) - 1) * p)))
            return vals[idx]
        return cls(
            n=len(vals),
            min=vals[0],
            p25=pct(0.25),
            p50=pct(0.50),
            p75=pct(0.75),
            max=vals[-1],
            mean=statistics.fmean(vals),
        )


@dataclass
class CalibrationBin:
    decile_lower: float
    decile_upper: float
    n_pairs: int
    mean_predicted_joint: Optional[float]
    mean_actual_hit_rate: Optional[float]  # both_win rate
    calibration_gap: Optional[float]  # predicted - actual; +ve means model over-predicts


def _calibration_bins(rows: list[dict[str, Any]]) -> list[CalibrationBin]:
    buckets: dict[int, list[dict[str, Any]]] = {i: [] for i in range(10)}
    for row in rows:
        joint = _finite(row.get("predicted_joint_probability"))
        if joint is None:
            continue
        buckets[_bucket_index(joint, _DECILE_EDGES)].append(row)

    bins: list[CalibrationBin] = []
    for i in range(10):
        rows_i = buckets[i]
        if not rows_i:
            bins.append(CalibrationBin(
                decile_lower=_DECILE_EDGES[i], decile_upper=_DECILE_EDGES[i + 1],
                n_pairs=0, mean_predicted_joint=None,
                mean_actual_hit_rate=None, calibration_gap=None,
            ))
            continue
        predicted = [_finite(r.get("predicted_joint_probability")) for r in rows_i]
        predicted = [p for p in predicted if p is not None]
        wins = sum(1 for r in rows_i if r.get("both_win"))
        mean_pred = statistics.fmean(predicted) if predicted else None
        mean_actual = wins / len(rows_i)
        gap = (mean_pred - mean_actual) if mean_pred is not None else None
        bins.append(CalibrationBin(
            decile_lower=_DECILE_EDGES[i], decile_upper=_DECILE_EDGES[i + 1],
            n_pairs=len(rows_i), mean_predicted_joint=mean_pred,
            mean_actual_hit_rate=mean_actual, calibration_gap=gap,
        ))
    return bins


@dataclass
class PoolReport:
    name: str
    row_count: int
    hit_rate: Optional[float]
    predicted_joint_distribution: Distribution
    combined_price_distribution: Distribution
    per_leg_model_probability_distribution: Distribution
    calibration_bins: list[CalibrationBin] = field(default_factory=list)
    calibration_gap_mean_across_populated_bins: Optional[float] = None


def summarize_pool(
    name: str,
    rows: list[dict[str, Any]],
    *,
    joint_calibrator: Any | None = None,
) -> PoolReport:
    """Summarize a pool. If joint_calibrator is provided, every row's
    predicted_joint_probability is passed through it before summary --
    both distributions and calibration bins use the calibrated value.
    Rows are NOT mutated."""
    def _joint_of(row: dict[str, Any]) -> Optional[float]:
        raw = _finite(row.get("predicted_joint_probability"))
        if raw is None:
            return None
        return float(joint_calibrator.calibrate(raw)) if joint_calibrator is not None else raw

    joints = [_joint_of(r) for r in rows]
    joints = [x for x in joints if x is not None]
    prices = [_finite(r.get("quoted_pair_price")) for r in rows]
    prices = [x for x in prices if x is not None]
    # Per-leg model probabilities are on synthetic rows and on
    # pair-ingest v1.1+ rows. On v1 real rows they are None.
    per_leg = []
    for r in rows:
        for k in ("leg_1_model_probability", "leg_2_model_probability"):
            v = _finite(r.get(k))
            if v is not None:
                per_leg.append(v)

    wins = sum(1 for r in rows if r.get("both_win"))
    hit_rate = (wins / len(rows)) if rows else None

    if joint_calibrator is not None:
        # Re-shape rows with calibrated joint on the fly so calibration
        # bins bucket by the calibrated value the caller cares about.
        bin_rows = [
            dict(r, predicted_joint_probability=_joint_of(r))
            for r in rows
            if _joint_of(r) is not None
        ]
    else:
        bin_rows = rows
    bins = _calibration_bins(bin_rows)
    gaps = [b.calibration_gap for b in bins if b.calibration_gap is not None]
    mean_gap = statistics.fmean(gaps) if gaps else None

    return PoolReport(
        name=name,
        row_count=len(rows),
        hit_rate=hit_rate,
        predicted_joint_distribution=Distribution.summarize(joints),
        combined_price_distribution=Distribution.summarize(prices),
        per_leg_model_probability_distribution=Distribution.summarize(per_leg),
        calibration_bins=bins,
        calibration_gap_mean_across_populated_bins=mean_gap,
    )


# --- report -------------------------------------------------------------

@dataclass
class PoolGapReport:
    generated_at_utc: str
    real_ledger_path: str
    synthetic_ledger_path: str
    real_pool: PoolReport
    synthetic_pool: PoolReport
    hit_rate_gap_percentage_points: Optional[float]
    calibration_gap_delta: Optional[float]
    interpretation_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "real_ledger_path": self.real_ledger_path,
            "synthetic_ledger_path": self.synthetic_ledger_path,
            "real_pool": asdict(self.real_pool),
            "synthetic_pool": asdict(self.synthetic_pool),
            "hit_rate_gap_percentage_points": self.hit_rate_gap_percentage_points,
            "calibration_gap_delta": self.calibration_gap_delta,
            "interpretation_notes": self.interpretation_notes,
        }


def _interpret(real: PoolReport, synth: PoolReport) -> list[str]:
    """Turn the numbers into short, honest headlines a reader can act on.

    Rules of engagement (this module never picks an interpretation):
    * Every string here starts from a comparison already present in
      the report and adds no new inference beyond a labelling of
      which pool is over-, under-, or well-calibrated.
    * `abs(gap) < 0.02` counts as "well calibrated" -- ~2 pp per
      decile is close enough that reweighting will not measurably move
      the promotion decision.
    """
    notes: list[str] = []
    if real.hit_rate is not None and synth.hit_rate is not None:
        gap = synth.hit_rate - real.hit_rate
        notes.append(
            f"Hit-rate gap: synthetic {synth.hit_rate:.3f} - real {real.hit_rate:.3f} = "
            f"{gap:+.3f} ({gap*100:+.1f} pp)."
        )
    if (real.calibration_gap_mean_across_populated_bins is not None
            and synth.calibration_gap_mean_across_populated_bins is not None):
        real_gap = real.calibration_gap_mean_across_populated_bins
        synth_gap = synth.calibration_gap_mean_across_populated_bins
        delta = real_gap - synth_gap
        notes.append(
            f"Mean per-decile calibration gap (predicted - actual): real {real_gap:+.4f}, "
            f"synthetic {synth_gap:+.4f}, difference {delta:+.4f}."
        )
        real_well = abs(real_gap) < 0.02
        synth_well = abs(synth_gap) < 0.02
        if real_well and synth_well:
            notes.append(
                "Both pools well-calibrated (|gap| < 2 pp); the hit-rate gap is a pool-"
                "composition fact, not a model-quality one, and consistent with "
                "interpretation (A) -- upstream selector shapes the pool, not the model."
            )
        elif real_well and not synth_well:
            notes.append(
                "Real pool is well-calibrated; synthetic pool is not. The calibrator (if "
                "one was applied) resolved the real-pool miscalibration; a calibrator fit "
                "on the real pool will not generalize to the broader synthetic pool because "
                "the two pools have different predicted-probability distributions."
            )
        elif synth_well and not real_well:
            notes.append(
                "Synthetic pool is well-calibrated; real pool is not. This is the diagnosis "
                "the branch's `pair_ledger_calibration.py` was built to resolve -- fit a "
                "beta calibrator on the real ledger and re-run with --with-calibrator."
            )
        elif real_gap > synth_gap + 0.02:
            notes.append(
                "The real pool's joint model over-predicts hit rate MORE than the synthetic "
                "pool's does. The real ledger was already the more-conservative pool by "
                "construction, so a larger positive gap here suggests the frozen production "
                "model is miscalibrated in a way the naive-independence baseline is not. "
                "Interpretation (B) is not supported by this data."
            )
        elif synth_gap > real_gap + 0.02:
            notes.append(
                "The synthetic pool's independence-assumed joint over-predicts hit rate "
                "MORE than the real pool's joint does. This supports interpretation (B): "
                "the real joint model is correctly capturing correlation the naive "
                "independence baseline misses."
            )
        else:
            notes.append(
                "Both pools miscalibrated in the same direction and roughly the same amount; "
                "no clean discrimination between interpretations from calibration alone."
            )
    if real.per_leg_model_probability_distribution.n == 0:
        notes.append(
            "Real ledger carries zero per-leg model probabilities today (v1 schema); as "
            "pair-ingest v1.1 capture matures on prospective slates, this comparison "
            "grows a fourth axis (per-leg calibration) that will further discriminate the "
            "interpretations."
        )
    return notes


def build_report(
    *,
    real_ledger: Path,
    synthetic_ledger: Path,
    joint_calibrator: Any | None = None,
) -> PoolGapReport:
    real_rows = _load_jsonl(real_ledger)
    synth_rows = _load_jsonl(synthetic_ledger)
    real_pool = summarize_pool(
        "REAL_PAIR_OBSERVATION_LEDGER", real_rows,
        joint_calibrator=joint_calibrator,
    )
    synth_pool = summarize_pool(
        "SYNTHETIC_CROSS_GAME_PAIR_LEDGER", synth_rows,
        joint_calibrator=joint_calibrator,
    )
    hit_rate_gap = None
    if real_pool.hit_rate is not None and synth_pool.hit_rate is not None:
        hit_rate_gap = 100 * (synth_pool.hit_rate - real_pool.hit_rate)
    cal_delta = None
    if (real_pool.calibration_gap_mean_across_populated_bins is not None
            and synth_pool.calibration_gap_mean_across_populated_bins is not None):
        cal_delta = (
            real_pool.calibration_gap_mean_across_populated_bins
            - synth_pool.calibration_gap_mean_across_populated_bins
        )
    return PoolGapReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        real_ledger_path=str(real_ledger),
        synthetic_ledger_path=str(synthetic_ledger),
        real_pool=real_pool,
        synthetic_pool=synth_pool,
        hit_rate_gap_percentage_points=hit_rate_gap,
        calibration_gap_delta=cal_delta,
        interpretation_notes=_interpret(real_pool, synth_pool),
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Investigate the real-vs-synthetic pair-pool hit-rate gap, before and (optionally) after calibration.")
    parser.add_argument("--real", type=Path, default=REPO_ROOT / DEFAULT_REAL_LEDGER)
    parser.add_argument("--synthetic", type=Path, default=REPO_ROOT / DEFAULT_SYNTHETIC_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    parser.add_argument(
        "--with-calibrator", action="store_true",
        help="Also fit a beta calibrator on the real ledger and rerun the pool comparison after applying it. "
             "Reports BEFORE and AFTER side-by-side so the resolution is visible.",
    )
    args = parser.parse_args()

    before = build_report(real_ledger=args.real, synthetic_ledger=args.synthetic)
    payload = {"before_calibration": before.to_dict()}
    print("=== BEFORE calibration ===")
    print(f"real pool: {before.real_pool.row_count} rows, hit rate {before.real_pool.hit_rate}")
    print(f"synthetic pool: {before.synthetic_pool.row_count} rows, hit rate {before.synthetic_pool.hit_rate}")
    if before.hit_rate_gap_percentage_points is not None:
        print(f"headline gap: {before.hit_rate_gap_percentage_points:+.1f} pp")
    for note in before.interpretation_notes:
        print(f"  - {note}")

    if args.with_calibrator:
        # Local import to keep the module importable without scipy for
        # callers that only need the raw investigation.
        from .pair_ledger_calibration import fit_beta_calibrator
        rows = _load_jsonl(args.real)
        calibrator = fit_beta_calibrator(rows)
        print(f"\n=== fitted BETA CALIBRATOR (in-sample on real ledger) ===")
        print(f"  slope={calibrator.slope:+.4f}  intercept={calibrator.intercept:+.4f}  n_fitted={calibrator.n_fitted_pairs}")
        after = build_report(
            real_ledger=args.real, synthetic_ledger=args.synthetic,
            joint_calibrator=calibrator,
        )
        payload["calibrator"] = {
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
            "n_fitted_pairs": calibrator.n_fitted_pairs,
        }
        payload["after_calibration"] = after.to_dict()
        print("\n=== AFTER calibration ===")
        print(f"real pool: {after.real_pool.row_count} rows, hit rate {after.real_pool.hit_rate}")
        print(f"synthetic pool: {after.synthetic_pool.row_count} rows, hit rate {after.synthetic_pool.hit_rate}")
        if after.hit_rate_gap_percentage_points is not None:
            print(f"headline gap: {after.hit_rate_gap_percentage_points:+.1f} pp")
        for note in after.interpretation_notes:
            print(f"  - {note}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
