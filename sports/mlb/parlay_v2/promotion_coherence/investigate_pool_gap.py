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


def summarize_pool(name: str, rows: list[dict[str, Any]]) -> PoolReport:
    joints = [_finite(r.get("predicted_joint_probability")) for r in rows]
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

    bins = _calibration_bins(rows)
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
    * The comparisons are what they are; every string here starts from
      a comparison already present in the report and adds no new
      inference.
    * If both pools show similar calibration gaps but very different
      hit rates, the imbalance is more likely upstream (interpretation
      A). If the synthetic pool shows a small calibration gap while
      the real pool shows a large one, the joint model is being
      pessimistic (interpretation B). Anything else, say so.
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
        if abs(delta) < 0.02:
            notes.append(
                "Calibration is comparable across pools; the hit-rate gap is more "
                "consistent with interpretation (A) -- upstream selector -- than with "
                "interpretation (B) -- joint-model pessimism."
            )
        elif real_gap > synth_gap + 0.02:
            notes.append(
                "The real pool's joint model over-predicts hit rate MORE than the synthetic "
                "pool's does. This is unusual -- the real ledger was already the more-"
                "conservative pool by construction, so a larger positive gap here suggests "
                "the frozen production model is not being pessimistic (interpretation B is "
                "not supported); the gap likely reflects a different upstream sampling issue."
            )
        elif synth_gap > real_gap + 0.02:
            notes.append(
                "The synthetic pool's independence-assumed joint over-predicts hit rate "
                "MORE than the real pool's calibrated joint does. This supports "
                "interpretation (B): the real joint model is correctly capturing "
                "correlation the naive independence baseline misses. The hit-rate gap is "
                "at least partly a synthetic-optimism artefact, not upstream selection alone."
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
) -> PoolGapReport:
    real_rows = _load_jsonl(real_ledger)
    synth_rows = _load_jsonl(synthetic_ledger)
    real_pool = summarize_pool("REAL_PAIR_OBSERVATION_LEDGER", real_rows)
    synth_pool = summarize_pool("SYNTHETIC_CROSS_GAME_PAIR_LEDGER", synth_rows)
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
    parser = argparse.ArgumentParser(description="Investigate the real-vs-synthetic pair-pool hit-rate gap.")
    parser.add_argument("--real", type=Path, default=REPO_ROOT / DEFAULT_REAL_LEDGER)
    parser.add_argument("--synthetic", type=Path, default=REPO_ROOT / DEFAULT_SYNTHETIC_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_report(real_ledger=args.real, synthetic_ledger=args.synthetic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")
    print(f"\nreal pool: {report.real_pool.row_count} rows, hit rate {report.real_pool.hit_rate}")
    print(f"synthetic pool: {report.synthetic_pool.row_count} rows, hit rate {report.synthetic_pool.hit_rate}")
    print(f"\nheadline gap: {report.hit_rate_gap_percentage_points:+.1f} pp" if report.hit_rate_gap_percentage_points is not None else "no gap comparable")
    for note in report.interpretation_notes:
        print(f"  - {note}")


if __name__ == "__main__":
    _cli()
