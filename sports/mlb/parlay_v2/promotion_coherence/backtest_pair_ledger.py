"""Backtest the promotion-margin rule against the real settled pair
observation ledger.

The pair observation ledger at
sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl is
the largest real settled parlay-pair dataset this repo carries: 3,120+
graded pair observations across the currently-frozen slate window, each
row already carrying the exact quantities the promotion-margin rule
needs:

    predicted_joint_probability    (already-calibrated joint)
    quoted_pair_price              (combined decimal price, so
                                    break_even = 1 / price)
    actual_pair_return             (realized unit return, +profit or -1)
    both_win, leg_1_result, leg_2_result

This backtest sweeps promotion-margin floors, reports admitted-count,
hit-rate, and realized return per unit for each floor, and slices by
same_game / cross_game and market_pair_type. It does not force any
threshold to win: it reports every floor's numbers side-by-side with the
"accept every scored pair" baseline, and calls out where the coherent
rule strictly dominates and where it does not.

Scope note. This is a broad backtest of the joint-probability +
price economics the promotion-margin rule operates on. The `parlays.
public_quality_overlay` leg-probability floor is not testable from this
ledger (no per-leg model probability is stored), so this run measures
the margin rule specifically. The overlay-authority test lives in the
Sept 2 regression and the shadow-replay module.

Reads only. Writes only under this subpackage. No live-selector import.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = "sports/mlb/parlay_v2/calibration/reports/pair_observation_ledger.jsonl"
DEFAULT_SYNTHETIC_LEDGER = "sports/mlb/parlay_v2/promotion_coherence/reports/synthetic_cross_game_pair_ledger.jsonl"
DEFAULT_REPORT = "sports/mlb/parlay_v2/promotion_coherence/reports/pair_ledger_backtest.json"

# Sweep in 1 percentage-point steps across a range wide enough to include
# meaningfully-below-zero floors (which admits everything with any real
# priced joint) and clearly-positive floors that would abstain on most
# real pairs. The zero-margin floor is the natural break-even parity
# check.
DEFAULT_FLOOR_SWEEP = tuple(round(x / 100.0, 2) for x in range(-10, 11, 1))


# --- pair -> promotion_margin --------------------------------------------

def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _break_even(price: Any) -> Optional[float]:
    p = _finite(price)
    if p is None or p <= 1.0:
        return None
    return 1.0 / p


def compute_promotion_margin(
    row: dict[str, Any],
    *,
    apply_same_game_penalty: bool = False,
    same_game_profile: Any | None = None,
    joint_calibrator: Any | None = None,
) -> Optional[float]:
    """Promotion margin for one pair-observation row.

    Default is the raw `predicted_joint - break_even` -- deductions
    stay zero unless the caller asks for one. `apply_same_game_penalty`
    subtracts the shared-failure deduction on same-game rows (zero on
    cross-game rows), using `SameGamePenaltyProfile` defaults unless a
    custom `same_game_profile` is passed. Nothing here silently applies
    a penalty; the flag is explicit at every call site.

    `joint_calibrator` (optional) is any object with a `.calibrate(p)`
    method (see JointProbabilityCalibrator). If a
    SliceConditionedCalibrator is passed, its `.calibrate_from_row`
    method is used automatically so the per-row slice picks its own
    sub-calibrator.
    """
    joint = _finite(row.get("predicted_joint_probability"))
    break_even = _break_even(row.get("quoted_pair_price"))
    if joint is None or break_even is None:
        return None
    if joint_calibrator is not None:
        calibrate_from_row = getattr(joint_calibrator, "calibrate_from_row", None)
        if callable(calibrate_from_row):
            joint = float(calibrate_from_row(joint, row))
        else:
            joint = float(joint_calibrator.calibrate(joint))
    deduction = 0.0
    if apply_same_game_penalty:
        # Local import to keep this module standalone when the same-
        # game penalty is not requested.
        from .same_game_penalty import same_game_shared_failure_deduction
        deduction = same_game_shared_failure_deduction(row, profile=same_game_profile)
    return joint - deduction - break_even


# --- backtest -----------------------------------------------------------

@dataclass
class FloorResult:
    floor: float
    admitted_count: int
    total_count: int
    admitted_share: float
    wins: int
    losses: int
    hit_rate: Optional[float]
    total_return_per_unit: float
    mean_return_per_unit: Optional[float]
    return_stdev_per_unit: Optional[float]

    @classmethod
    def build(
        cls, *, floor: float, admitted: list[dict[str, Any]], total_count: int,
    ) -> "FloorResult":
        admitted_count = len(admitted)
        wins = sum(1 for r in admitted if r.get("both_win"))
        losses = admitted_count - wins
        returns = [_finite(r.get("actual_pair_return")) for r in admitted]
        returns = [x for x in returns if x is not None]
        total_return = sum(returns) if returns else 0.0
        mean_return = mean(returns) if returns else None
        stdev = pstdev(returns) if len(returns) >= 2 else None
        hit_rate = (wins / admitted_count) if admitted_count else None
        return cls(
            floor=floor,
            admitted_count=admitted_count,
            total_count=total_count,
            admitted_share=(admitted_count / total_count) if total_count else 0.0,
            wins=wins,
            losses=losses,
            hit_rate=hit_rate,
            total_return_per_unit=total_return,
            mean_return_per_unit=mean_return,
            return_stdev_per_unit=stdev,
        )


def _apply_floor(
    rows: list[dict[str, Any]],
    floor: float,
    *,
    apply_same_game_penalty: bool = False,
    same_game_profile: Any | None = None,
    joint_calibrator: Any | None = None,
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for row in rows:
        margin = compute_promotion_margin(
            row,
            apply_same_game_penalty=apply_same_game_penalty,
            same_game_profile=same_game_profile,
            joint_calibrator=joint_calibrator,
        )
        if margin is None:
            continue
        if margin >= floor:
            admitted.append(row)
    return admitted


def sweep_floors(
    rows: list[dict[str, Any]],
    floors: Iterable[float],
    *,
    apply_same_game_penalty: bool = False,
    same_game_profile: Any | None = None,
    joint_calibrator: Any | None = None,
) -> list[FloorResult]:
    total = len(rows)
    return [
        FloorResult.build(
            floor=f,
            admitted=_apply_floor(
                rows, f,
                apply_same_game_penalty=apply_same_game_penalty,
                same_game_profile=same_game_profile,
                joint_calibrator=joint_calibrator,
            ),
            total_count=total,
        )
        for f in floors
    ]


@dataclass
class SliceReport:
    name: str
    filter_description: str
    row_count: int
    baseline_all_admitted: FloorResult
    floor_sweep: list[FloorResult]
    best_floor_by_total_return: Optional[FloorResult]
    best_floor_by_mean_return: Optional[FloorResult]
    best_floor_by_hit_rate: Optional[FloorResult]
    strict_dominance_over_baseline: Optional[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _best_by(sweep: list[FloorResult], key) -> Optional[FloorResult]:
    filtered = [r for r in sweep if key(r) is not None]
    if not filtered:
        return None
    return max(filtered, key=key)


def _strict_dominance(baseline: FloorResult, sweep: list[FloorResult]) -> Optional[dict[str, Any]]:
    """A floor 'strictly dominates' the baseline if it beats baseline on
    both total return and mean return, without admitting fewer than 20
    pairs (arbitrary anti-overfit floor -- named in the output so it is
    not silent)."""
    MIN_ADMITTED = 20
    candidates = []
    for r in sweep:
        if r.floor == baseline.floor:
            continue
        if r.admitted_count < MIN_ADMITTED:
            continue
        if r.total_return_per_unit is None or baseline.total_return_per_unit is None:
            continue
        if r.mean_return_per_unit is None or baseline.mean_return_per_unit is None:
            continue
        if (
            r.total_return_per_unit > baseline.total_return_per_unit
            and r.mean_return_per_unit > baseline.mean_return_per_unit
        ):
            candidates.append(r)
    if not candidates:
        return None
    best = max(candidates, key=lambda r: (r.total_return_per_unit, r.mean_return_per_unit))
    return {
        "min_admitted_for_dominance": MIN_ADMITTED,
        "dominant_floor": best.floor,
        "dominant_admitted_count": best.admitted_count,
        "dominant_total_return_per_unit": best.total_return_per_unit,
        "dominant_mean_return_per_unit": best.mean_return_per_unit,
        "dominant_hit_rate": best.hit_rate,
        "baseline_total_return_per_unit": baseline.total_return_per_unit,
        "baseline_mean_return_per_unit": baseline.mean_return_per_unit,
        "baseline_hit_rate": baseline.hit_rate,
    }


def build_slice_report(
    *, name: str, filter_description: str,
    rows: list[dict[str, Any]], floors: Iterable[float],
    apply_same_game_penalty: bool = False,
    same_game_profile: Any | None = None,
) -> SliceReport:
    floors = list(floors)
    sweep = sweep_floors(
        rows, floors,
        apply_same_game_penalty=apply_same_game_penalty,
        same_game_profile=same_game_profile,
    )
    # Baseline = the smallest floor in the sweep, which admits every
    # scored pair.
    baseline = min(sweep, key=lambda r: r.floor) if sweep else FloorResult.build(
        floor=float("nan"), admitted=[], total_count=0,
    )
    return SliceReport(
        name=name,
        filter_description=filter_description,
        row_count=len(rows),
        baseline_all_admitted=baseline,
        floor_sweep=sweep,
        best_floor_by_total_return=_best_by(sweep, lambda r: r.total_return_per_unit),
        best_floor_by_mean_return=_best_by(sweep, lambda r: r.mean_return_per_unit),
        best_floor_by_hit_rate=_best_by(sweep, lambda r: r.hit_rate),
        strict_dominance_over_baseline=_strict_dominance(baseline, sweep),
    )


# --- data loading -------------------------------------------------------

def load_ledger(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rows_settled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("settlement_status") == "settled"]


def _slice_by_market(rows: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("market_pair_type") == market]


# --- report -------------------------------------------------------------

@dataclass
class LedgerBacktestReport:
    generated_at_utc: str
    ledger_path: str
    ledger_row_count: int
    settled_row_count: int
    floors_swept: list[float]
    slices: list[SliceReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "ledger_path": self.ledger_path,
            "ledger_row_count": self.ledger_row_count,
            "settled_row_count": self.settled_row_count,
            "floors_swept": self.floors_swept,
            "slices": [s.to_dict() for s in self.slices],
        }


def build_report(
    *,
    ledger_path: Path,
    floors: Iterable[float] = DEFAULT_FLOOR_SWEEP,
) -> LedgerBacktestReport:
    rows = load_ledger(ledger_path)
    settled = _rows_settled(rows)
    floors_list = list(floors)

    same_game_rows = [r for r in settled if r.get("same_game")]
    slices: list[SliceReport] = [
        build_slice_report(name="ALL_SETTLED_PAIRS",
                           filter_description="every settled pair-observation row",
                           rows=settled, floors=floors_list),
        build_slice_report(name="CROSS_GAME_PAIRS",
                           filter_description="settled pairs with same_game=False",
                           rows=[r for r in settled if not r.get("same_game")],
                           floors=floors_list),
        build_slice_report(name="SAME_GAME_PAIRS",
                           filter_description="settled pairs with same_game=True (no shared-failure deduction)",
                           rows=same_game_rows, floors=floors_list),
    ]
    if same_game_rows:
        slices.append(
            build_slice_report(
                name="SAME_GAME_PAIRS_WITH_SHARED_FAILURE_PENALTY",
                filter_description=(
                    "settled pairs with same_game=True, promotion margin adjusted by "
                    "SameGamePenaltyProfile() defaults (base 0.05 + optional 0.03 same-team + "
                    "optional 0.02 total-market, capped at 0.15). Reported side-by-side with "
                    "the raw same-game slice above so the effect of turning the deduction on is visible."
                ),
                rows=same_game_rows, floors=floors_list,
                apply_same_game_penalty=True,
            )
        )
    if settled:
        slices.append(
            build_slice_report(
                name="ALL_SETTLED_PAIRS_WITH_SHARED_FAILURE_PENALTY",
                filter_description=(
                    "every settled row, promotion margin adjusted by SameGamePenaltyProfile() "
                    "on same-game rows only (cross-game rows unchanged). Compare directly with "
                    "ALL_SETTLED_PAIRS above -- the delta at every floor is the impact of adding "
                    "the shared-failure deduction as a first-class penalty."
                ),
                rows=settled, floors=floors_list,
                apply_same_game_penalty=True,
            )
        )
    market_counts: dict[str, int] = {}
    for r in settled:
        market_counts[r.get("market_pair_type") or "UNKNOWN"] = (
            market_counts.get(r.get("market_pair_type") or "UNKNOWN", 0) + 1
        )
    for market, count in sorted(market_counts.items(), key=lambda kv: -kv[1]):
        if count < 100:
            continue
        slices.append(
            build_slice_report(
                name=f"MARKET_{market}",
                filter_description=f"settled pairs with market_pair_type={market}",
                rows=_slice_by_market(settled, market),
                floors=floors_list,
            )
        )

    return LedgerBacktestReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        ledger_path=str(ledger_path),
        ledger_row_count=len(rows),
        settled_row_count=len(settled),
        floors_swept=floors_list,
        slices=slices,
    )


# --- CLI -----------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "     "
    if isinstance(value, float):
        return f"{value:+.4f}" if value != int(value) else f"{value:.0f}"
    return str(value)


def _print_slice(slice_: SliceReport) -> None:
    print(f"\n=== {slice_.name} ({slice_.row_count} rows) ===")
    print(slice_.filter_description)
    print(f"{'floor':>7} {'admitted':>9} {'share':>7} {'wins':>5} {'losses':>7}"
          f" {'hit':>7} {'sum_ret':>9} {'mean_ret':>9} {'stdev':>9}")
    for r in slice_.floor_sweep:
        hit = f"{r.hit_rate:.3f}" if r.hit_rate is not None else "  -  "
        mean_r = f"{r.mean_return_per_unit:+.4f}" if r.mean_return_per_unit is not None else "   -   "
        stdev = f"{r.return_stdev_per_unit:.4f}" if r.return_stdev_per_unit is not None else "   -   "
        share = f"{r.admitted_share:.3f}" if r.admitted_share is not None else "  -  "
        print(f"{r.floor:+7.2f} {r.admitted_count:>9} {share:>7} {r.wins:>5}"
              f" {r.losses:>7} {hit:>7} {r.total_return_per_unit:+9.3f} {mean_r:>9} {stdev:>9}")
    d = slice_.strict_dominance_over_baseline
    if d is None:
        print("  (no floor strictly dominates the accept-all baseline on this slice)")
    else:
        print(f"  strict-dominance floor: {d['dominant_floor']:+.2f}  "
              f"(admits {d['dominant_admitted_count']}, "
              f"total_ret {d['dominant_total_return_per_unit']:+.3f}, "
              f"mean_ret {d['dominant_mean_return_per_unit']:+.4f}, "
              f"hit {d['dominant_hit_rate']:.3f})")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the promotion-margin rule against the real settled pair observation ledger, and optionally against a synthetic cross-game ledger for scale.",
    )
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / DEFAULT_LEDGER)
    parser.add_argument(
        "--synthetic-ledger", type=Path, default=REPO_ROOT / DEFAULT_SYNTHETIC_LEDGER,
        help="Optional synthetic cross-game pair ledger to backtest side-by-side. "
             "Missing file is skipped, not an error -- the synthetic ledger is generated separately.",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Only backtest the real pair-observation ledger, even if the synthetic ledger is present.",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / DEFAULT_REPORT)
    parser.add_argument("--floor-min", type=float, default=-0.10)
    parser.add_argument("--floor-max", type=float, default=+0.10)
    parser.add_argument("--floor-step", type=float, default=0.01)
    args = parser.parse_args()

    floor_min = args.floor_min
    floor_max = args.floor_max
    step = args.floor_step
    floors: list[float] = []
    x = floor_min
    while x <= floor_max + 1e-9:
        floors.append(round(x, 4))
        x += step

    reports: list[tuple[str, LedgerBacktestReport]] = []
    print(f"=== REAL pair-observation ledger ===")
    real_report = build_report(ledger_path=args.ledger, floors=floors)
    reports.append(("real", real_report))
    print(f"ledger rows: {real_report.ledger_row_count}  settled: {real_report.settled_row_count}")
    for s in real_report.slices:
        _print_slice(s)

    if not args.skip_synthetic and args.synthetic_ledger.exists():
        print(f"\n=== SYNTHETIC cross-game pair ledger ===")
        print(f"(from settled singles -- exploratory, not real pair observations)")
        synth_report = build_report(ledger_path=args.synthetic_ledger, floors=floors)
        reports.append(("synthetic", synth_report))
        print(f"ledger rows: {synth_report.ledger_row_count}  settled: {synth_report.settled_row_count}")
        for s in synth_report.slices:
            _print_slice(s)
    else:
        if args.skip_synthetic:
            print("\n(--skip-synthetic set; synthetic ledger not evaluated)")
        else:
            print(f"\n(no synthetic ledger at {args.synthetic_ledger} -- run "
                  f"`python -m sports.mlb.parlay_v2.promotion_coherence.synthesize_pairs` to generate one)")

    out_payload = {
        report_name: r.to_dict() for report_name, r in reports
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    _cli()
