"""End-to-end validation of the promotion-coherence shadow stack.

Fully honest, leave-one-slate-out backtest of THREE promotion strategies
on the real pair-observation ledger:

    A. RAW -- no calibration, no slice conditioning. The baseline.
    B. GLOBAL_BETA -- single global BetaCalibrator fit on the training
       slates, applied at inference.
    C. SLICE_CONDITIONED_BETA -- per-market-pair-type BetaCalibrator
       plus a global fallback, fit on the training slates, applied at
       inference. This is the resolution of the negative-slope finding.

For each strategy, evaluated at each promotion-margin floor in a sweep,
we report on the held-out slate:

    * admitted_count
    * hit_rate (both_win / admitted)
    * total realized return per unit
    * mean realized return per unit

Then the numbers roll up across all folds. The strategy is DECLARED
"exceeds threshold" only when a specific promotion floor produces:

    * total realized return per unit STRICTLY POSITIVE across the
      concatenated OOS folds, AND
    * strictly better than the RAW baseline at the same floor, AND
    * admitted at least MIN_ADMITTED_FOR_THRESHOLD = 100 pairs across
      folds (anti-cherry-pick guard).

The output json carries the full sweep for every strategy so an
auditor can see what the numbers were, not just whether the flag
tripped.

Read-only, deterministic, no live-selector import.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Optional

from .backtest_pair_ledger import compute_promotion_margin
from .pair_ledger_calibration import BetaCalibrator, fit_beta_calibrator
from .slice_conditioned_calibrator import (
    SliceConditionedCalibrator,
    fit_slice_conditioned_calibrator,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"
DEFAULT_REPORT = "sports/mlb/parlay_v2/promotion_coherence/reports/system_validation.json"

DEFAULT_FLOORS = tuple(round(x / 100, 3) for x in range(-5, 11, 1))
MIN_ADMITTED_FOR_THRESHOLD = 100


def _load_settled(ledger: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not ledger.exists():
        return rows
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("settlement_status") == "settled":
                rows.append(r)
    return rows


@dataclass
class FloorAggregate:
    floor: float
    admitted: int
    wins: int
    losses: int
    hit_rate: Optional[float]
    total_return_per_unit: float
    mean_return_per_unit: Optional[float]

    @classmethod
    def from_admitted(cls, floor: float, rows: list[dict[str, Any]]) -> "FloorAggregate":
        admitted = len(rows)
        wins = sum(1 for r in rows if r.get("both_win"))
        losses = admitted - wins
        returns = []
        for r in rows:
            v = r.get("actual_pair_return")
            if isinstance(v, (int, float)):
                returns.append(float(v))
        total = sum(returns) if returns else 0.0
        mean = fmean(returns) if returns else None
        hit = (wins / admitted) if admitted else None
        return cls(floor=floor, admitted=admitted, wins=wins, losses=losses,
                   hit_rate=hit, total_return_per_unit=total, mean_return_per_unit=mean)


def _sweep_strategy(
    train_rows: list[dict[str, Any]],
    held_rows: list[dict[str, Any]],
    floors: list[float],
    *,
    joint_calibrator: Any | None,
    apply_same_game_penalty: bool,
) -> list[FloorAggregate]:
    """Evaluate ONE strategy on ONE held-out slate. train_rows is
    only used by the caller to fit the calibrator; this function
    just walks the held-out rows and admits them by margin >= floor.
    """
    out: list[FloorAggregate] = []
    for f in floors:
        admitted: list[dict[str, Any]] = []
        for r in held_rows:
            m = compute_promotion_margin(
                r,
                apply_same_game_penalty=apply_same_game_penalty,
                joint_calibrator=joint_calibrator,
            )
            if m is None:
                continue
            if m >= f:
                admitted.append(r)
        out.append(FloorAggregate.from_admitted(f, admitted))
    return out


@dataclass
class StrategyFold:
    strategy: str
    held_out_slate: str
    train_row_count: int
    held_row_count: int
    calibrator_summary: Optional[dict[str, Any]]
    per_floor: list[FloorAggregate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "held_out_slate": self.held_out_slate,
            "train_row_count": self.train_row_count,
            "held_row_count": self.held_row_count,
            "calibrator_summary": self.calibrator_summary,
            "per_floor": [asdict(x) for x in self.per_floor],
        }


@dataclass
class StrategyAggregate:
    strategy: str
    total_folds: int
    per_floor_aggregate: dict[float, FloorAggregate]  # concatenated across folds

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_folds": self.total_folds,
            "per_floor_aggregate": {
                str(f): asdict(agg) for f, agg in sorted(self.per_floor_aggregate.items())
            },
        }


def _concat_folds(folds: list[StrategyFold]) -> StrategyAggregate:
    if not folds:
        return StrategyAggregate(strategy="", total_folds=0, per_floor_aggregate={})
    strategy = folds[0].strategy
    floors = sorted({fa.floor for f in folds for fa in f.per_floor})
    agg: dict[float, FloorAggregate] = {}
    for floor in floors:
        admitted = 0
        wins = 0
        losses = 0
        total_ret = 0.0
        returns: list[float] = []
        for f in folds:
            for fa in f.per_floor:
                if fa.floor != floor:
                    continue
                admitted += fa.admitted
                wins += fa.wins
                losses += fa.losses
                total_ret += fa.total_return_per_unit
                if fa.mean_return_per_unit is not None and fa.admitted > 0:
                    # reconstruct returns list is not possible here without
                    # more detail; keep mean-of-means as an approximation only.
                    pass
        # We only need aggregate counts + total_return; mean is total/admitted.
        mean = (total_ret / admitted) if admitted else None
        hit = (wins / admitted) if admitted else None
        agg[floor] = FloorAggregate(floor=floor, admitted=admitted, wins=wins,
                                    losses=losses, hit_rate=hit,
                                    total_return_per_unit=total_ret,
                                    mean_return_per_unit=mean)
    return StrategyAggregate(strategy=strategy, total_folds=len(folds),
                             per_floor_aggregate=agg)


@dataclass
class ThresholdVerdict:
    strategy: str
    exceeds_threshold: bool
    winning_floor: Optional[float]
    winning_total_return_per_unit: Optional[float]
    winning_hit_rate: Optional[float]
    winning_admitted: Optional[int]
    baseline_total_return_at_winning_floor: Optional[float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_threshold(
    strategy_name: str,
    strategy_agg: StrategyAggregate,
    raw_agg: StrategyAggregate,
    *,
    min_admitted: int = MIN_ADMITTED_FOR_THRESHOLD,
) -> ThresholdVerdict:
    """A strategy 'exceeds threshold' when at least one floor
    simultaneously:
        * total_return_per_unit > 0
        * strictly beats RAW at the same floor
        * admits >= min_admitted pairs across all folds
    """
    best_floor: Optional[float] = None
    best_return = 0.0
    best_hit = None
    best_admitted = None
    best_baseline = None
    for floor, agg in strategy_agg.per_floor_aggregate.items():
        if agg.admitted < min_admitted:
            continue
        if agg.total_return_per_unit <= 0.0:
            continue
        raw_at_floor = raw_agg.per_floor_aggregate.get(floor)
        raw_ret = raw_at_floor.total_return_per_unit if raw_at_floor else 0.0
        if agg.total_return_per_unit <= raw_ret:
            continue
        if best_floor is None or agg.total_return_per_unit > best_return:
            best_floor = floor
            best_return = agg.total_return_per_unit
            best_hit = agg.hit_rate
            best_admitted = agg.admitted
            best_baseline = raw_ret

    if best_floor is None:
        return ThresholdVerdict(
            strategy=strategy_name, exceeds_threshold=False,
            winning_floor=None, winning_total_return_per_unit=None,
            winning_hit_rate=None, winning_admitted=None,
            baseline_total_return_at_winning_floor=None,
            reason=(f"no floor met all three conditions (positive return, "
                    f">= {min_admitted} admitted, strict improvement over RAW)"),
        )
    return ThresholdVerdict(
        strategy=strategy_name, exceeds_threshold=True,
        winning_floor=best_floor, winning_total_return_per_unit=best_return,
        winning_hit_rate=best_hit, winning_admitted=best_admitted,
        baseline_total_return_at_winning_floor=best_baseline,
        reason=(f"floor {best_floor:+.2f} admits {best_admitted} pairs, "
                f"total_ret {best_return:+.3f} vs RAW {best_baseline:+.3f}"),
    )


@dataclass
class ValidationReport:
    generated_at_utc: str
    ledger_path: str
    total_settled_rows: int
    slates_covered: list[str]
    floors_swept: list[float]
    min_admitted_for_threshold: int
    strategies: dict[str, list[StrategyFold]]
    strategy_aggregates: dict[str, StrategyAggregate]
    threshold_verdicts: dict[str, ThresholdVerdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "ledger_path": self.ledger_path,
            "total_settled_rows": self.total_settled_rows,
            "slates_covered": self.slates_covered,
            "floors_swept": self.floors_swept,
            "min_admitted_for_threshold": self.min_admitted_for_threshold,
            "strategies": {
                name: [f.to_dict() for f in folds]
                for name, folds in self.strategies.items()
            },
            "strategy_aggregates": {
                name: agg.to_dict() for name, agg in self.strategy_aggregates.items()
            },
            "threshold_verdicts": {
                name: v.to_dict() for name, v in self.threshold_verdicts.items()
            },
        }


def run_validation(
    *,
    ledger_path: Path,
    floors: Iterable[float] = DEFAULT_FLOORS,
    min_admitted: int = MIN_ADMITTED_FOR_THRESHOLD,
) -> ValidationReport:
    floors_list = list(floors)
    rows = _load_settled(ledger_path)
    slates = sorted({r.get("slate_id") for r in rows if r.get("slate_id")})

    strategies: dict[str, list[StrategyFold]] = {
        "RAW": [], "GLOBAL_BETA": [], "SLICE_CONDITIONED_BETA": [],
    }

    for held in slates:
        train = [r for r in rows if r.get("slate_id") != held]
        held_rows = [r for r in rows if r.get("slate_id") == held]

        # RAW: no calibrator
        strategies["RAW"].append(StrategyFold(
            strategy="RAW", held_out_slate=held,
            train_row_count=len(train), held_row_count=len(held_rows),
            calibrator_summary=None,
            per_floor=_sweep_strategy(train, held_rows, floors_list,
                                      joint_calibrator=None,
                                      apply_same_game_penalty=False),
        ))

        # GLOBAL_BETA
        global_cal = fit_beta_calibrator(train)
        strategies["GLOBAL_BETA"].append(StrategyFold(
            strategy="GLOBAL_BETA", held_out_slate=held,
            train_row_count=len(train), held_row_count=len(held_rows),
            calibrator_summary={
                "slope": global_cal.slope, "intercept": global_cal.intercept,
                "n_fitted_pairs": global_cal.n_fitted_pairs,
            },
            per_floor=_sweep_strategy(train, held_rows, floors_list,
                                      joint_calibrator=global_cal,
                                      apply_same_game_penalty=False),
        ))

        # SLICE_CONDITIONED_BETA
        slice_cal = fit_slice_conditioned_calibrator(train)
        strategies["SLICE_CONDITIONED_BETA"].append(StrategyFold(
            strategy="SLICE_CONDITIONED_BETA", held_out_slate=held,
            train_row_count=len(train), held_row_count=len(held_rows),
            calibrator_summary=slice_cal.as_dict(),
            per_floor=_sweep_strategy(train, held_rows, floors_list,
                                      joint_calibrator=slice_cal,
                                      apply_same_game_penalty=False),
        ))

    aggregates: dict[str, StrategyAggregate] = {
        name: _concat_folds(folds) for name, folds in strategies.items()
    }
    raw_agg = aggregates["RAW"]
    verdicts: dict[str, ThresholdVerdict] = {}
    for name, agg in aggregates.items():
        verdicts[name] = _evaluate_threshold(name, agg, raw_agg, min_admitted=min_admitted)

    return ValidationReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        ledger_path=str(ledger_path),
        total_settled_rows=len(rows),
        slates_covered=slates,
        floors_swept=floors_list,
        min_admitted_for_threshold=min_admitted,
        strategies=strategies,
        strategy_aggregates=aggregates,
        threshold_verdicts=verdicts,
    )


def _print_strategy(name: str, agg: StrategyAggregate) -> None:
    print(f"\n--- {name} ---")
    print(f"{'floor':>7} {'admitted':>9} {'wins':>5} {'losses':>7} {'hit':>7} {'sum_ret':>9} {'mean_ret':>9}")
    for floor, fa in sorted(agg.per_floor_aggregate.items()):
        hit = f"{fa.hit_rate:.3f}" if fa.hit_rate is not None else "  -  "
        mean = f"{fa.mean_return_per_unit:+.4f}" if fa.mean_return_per_unit is not None else "   -   "
        print(f"{floor:+7.2f} {fa.admitted:>9} {fa.wins:>5} {fa.losses:>7} {hit:>7} "
              f"{fa.total_return_per_unit:+9.3f} {mean:>9}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="End-to-end leave-one-slate-out validation of the promotion-coherence shadow stack.")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    parser.add_argument("--min-admitted", type=int, default=MIN_ADMITTED_FOR_THRESHOLD)
    args = parser.parse_args()

    report = run_validation(ledger_path=args.ledger, min_admitted=args.min_admitted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")
    print(f"\nledger: {report.total_settled_rows} settled rows across {len(report.slates_covered)} slates")
    for name in ["RAW", "GLOBAL_BETA", "SLICE_CONDITIONED_BETA"]:
        _print_strategy(name, report.strategy_aggregates[name])

    print("\n=== THRESHOLD VERDICTS ===")
    for name in ["GLOBAL_BETA", "SLICE_CONDITIONED_BETA"]:
        v = report.threshold_verdicts[name]
        status = "EXCEEDS" if v.exceeds_threshold else "does NOT exceed"
        print(f"  {name}: {status} threshold  -- {v.reason}")


if __name__ == "__main__":
    _cli()
