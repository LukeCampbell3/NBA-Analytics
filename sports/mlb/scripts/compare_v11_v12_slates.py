#!/usr/bin/env python3
"""
v12 Phase 2: real, disclosed backtest comparing v11's live top-N/
diversification-capped board (premium_evidence_gated_v11) against
premium_safe_ev_v12_shadow's miss-budget-constrained SafeEV optimizer
(safe_ev_optimizer.py), over the same real archived dates and the same
real v11-eligible candidate pool build_v11_eligible_training_set.py
already builds for the winner-signature model -- reused directly here
(prepare_and_filter_candidates()), never rebuilt or approximated.

Applies the real asymmetric promotion gate from the v12 proposal:
    ROI_v12 > ROI_v11 AND HitRate_v12 >= HitRate_v11 - PROMOTION_HIT_RATE_MARGIN

Run across a real sweep of miss-budget values (not a single guessed
number) so the honest result is "no budget cleared the gate yet" when
that is what the real data shows, rather than a single cherry-picked
number reported as if it were the only one tried.

SHADOW ONLY. This script only ever reports a decision to a report JSON;
it never changes what v11 actually selects or publishes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import select_high_precision_predictions as shp  # noqa: E402
from build_v11_eligible_training_set import (  # noqa: E402
    DEFAULT_DAILY_RUNS_ROOT,
    DEFAULT_PROCESSED_ROOT,
    find_raw_pool_csvs,
    parse_v11_args,
)
from pick_survival_model import american_profit_per_unit, to_float  # noqa: E402
from safe_ev_optimizer import optimize_slate  # noqa: E402
from validate_historical_final_pools import build_actual_lookup, grade_result, normalize_player_key  # noqa: E402

REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_REPORT_JSON = (
    REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "v12_v11_slate_comparison_2026.json"
)
DEFAULT_MISS_BUDGETS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
# A real, disclosed asymmetric-gate margin -- v12 may trail v11's real hit
# rate by at most this much and still be considered, provided its ROI is
# strictly better. Not tuned against this harness's own output.
PROMOTION_HIT_RATE_MARGIN = 0.05


def grade_candidate(candidate: Any, actual_lookup: dict[tuple[str, str, str, str], float]) -> str | None:
    """"win"/"loss", or None for a push or a missing/unsettled real
    actual. Reuses validate_historical_final_pools.py's own real actual-
    lookup and grading -- never a hand-reimplemented copy of it."""
    player_key = normalize_player_key(candidate.player)
    lookup_key = (candidate.run_date.isoformat(), player_key, candidate.target, str(candidate.game_id))
    actual = actual_lookup.get(lookup_key)
    if actual is None:
        return None
    result = grade_result(actual, candidate.market_line, candidate.direction)
    return result if result in {"win", "loss"} else None


def slate_metrics(selected: list[Any], actual_lookup: dict[tuple[str, str, str, str], float]) -> dict[str, Any]:
    graded: list[tuple[str, float]] = []
    for candidate in selected:
        result = grade_candidate(candidate, actual_lookup)
        if result is None:
            continue
        side_price = to_float(getattr(candidate, "selected_side_price", None))
        profit = american_profit_per_unit(side_price) if side_price is not None else None
        if profit is None:
            continue
        graded.append((result, profit if result == "win" else -1.0))
    if not graded:
        return {"picks": len(selected), "settled": 0, "wins": 0, "hit_rate": None, "roi": None}
    wins = sum(1 for result, _ in graded if result == "win")
    profits = [profit for _, profit in graded]
    return {
        "picks": len(selected),
        "settled": len(graded),
        "wins": wins,
        "hit_rate": wins / len(graded),
        "roi": sum(profits) / len(profits),
    }


def gate_pass(roi_v11: float | None, hit_v11: float | None, roi_v12: float | None, hit_v12: float | None, margin: float) -> bool:
    """The real asymmetric promotion rule, isolated so it's testable
    without a full backtest fixture: strictly better ROI, and a hit rate
    that trails v11's by no more than `margin`."""
    return bool(
        roi_v11 is not None
        and roi_v12 is not None
        and hit_v11 is not None
        and hit_v12 is not None
        and roi_v12 > roi_v11
        and hit_v12 >= hit_v11 - margin
    )


def run_comparison(
    *,
    daily_runs_root: Path = DEFAULT_DAILY_RUNS_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
    miss_budgets: tuple[float, ...] = DEFAULT_MISS_BUDGETS,
) -> dict[str, Any]:
    actual_lookup = build_actual_lookup(processed_root)
    pool_csvs = find_raw_pool_csvs(daily_runs_root)

    v11_selected_all: list[Any] = []
    v12_selected_by_budget: dict[float, list[Any]] = {budget: [] for budget in miss_budgets}
    dates_scanned = 0
    errors: dict[str, str] = {}

    for pool_csv in pool_csvs:
        date_label = pool_csv.parent.name
        try:
            args = parse_v11_args(pool_csv)
            eligible, _rejected = shp.prepare_and_filter_candidates(args)
        except Exception as exc:  # a single bad archived date must never abort the whole comparison
            errors[date_label] = f"{type(exc).__name__}: {exc}"
            continue
        dates_scanned += 1
        v11_selected_all.extend(shp.select_top_candidates(eligible, args))
        for budget in miss_budgets:
            outcome = optimize_slate(
                eligible,
                miss_budget=budget,
                max_picks=int(args.top_n),
                max_per_market_bucket=int(args.max_per_market_bucket),
                max_per_team=int(args.max_per_team),
            )
            v12_selected_by_budget[budget].extend(outcome["selected"])

    v11_metrics = slate_metrics(v11_selected_all, actual_lookup)
    roi_v11 = v11_metrics.get("roi")
    hit_v11 = v11_metrics.get("hit_rate")

    by_budget: dict[str, Any] = {}
    eligible_budgets: list[float] = []
    for budget in miss_budgets:
        v12_metrics = slate_metrics(v12_selected_by_budget[budget], actual_lookup)
        passes = gate_pass(roi_v11, hit_v11, v12_metrics.get("roi"), v12_metrics.get("hit_rate"), PROMOTION_HIT_RATE_MARGIN)
        if passes:
            eligible_budgets.append(budget)
        by_budget[str(budget)] = {"metrics": v12_metrics, "promotion_gate_pass": passes}

    return {
        "schema_version": 1,
        "product_version": "premium_safe_ev_v12_shadow",
        "compared_against": "premium_evidence_gated_v13",
        "shadow_only": True,
        "archived_dates_scanned": dates_scanned,
        "dates_with_load_errors": errors,
        "promotion_gate": {
            "rule": "ROI_v12 > ROI_v11 AND HitRate_v12 >= HitRate_v11 - margin",
            "hit_rate_margin": PROMOTION_HIT_RATE_MARGIN,
            "decision": "eligible" if eligible_budgets else "remain_shadow",
            "eligible_miss_budgets": eligible_budgets,
        },
        "v11": v11_metrics,
        "v12_by_miss_budget": by_budget,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--miss-budget", type=float, nargs="+", default=list(DEFAULT_MISS_BUDGETS))
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    report = run_comparison(
        daily_runs_root=args.daily_runs_root,
        processed_root=args.processed_root,
        miss_budgets=tuple(args.miss_budget),
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
