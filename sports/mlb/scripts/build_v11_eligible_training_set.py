#!/usr/bin/env python3
"""
Build the real, v11-eligible historical training set for the winner-
signature model (v12 Phase 1 -- see the SafeEV veto proposal).

For each archived real daily run under sports/mlb/data/predictions/
daily_runs/<date>/, re-runs the exact real candidate build and structural
filter the live selector runs (select_high_precision_predictions.py's own
prepare_and_filter_candidates(), imported and called directly -- never a
hand-reconstructed approximation of its gate logic) against that day's raw
pool CSV, using the live selector's exact real thresholds (kept as
"v11" for continuity -- see V11_SELECTOR_ARGS below). Keeps only rows
that would have let through the structural gates (real market, 1+
confirmed book, 35+ history rows, real hit-probability >= 0.70, etc.)
-- this is deliberately NOT the top-N/diversification-capped board,
since "would this bet have been judged safe" and "did it make today's
board" are different questions; v12's winner-signature model wants the
former population.

Attaches real settled outcomes via validate_historical_final_pools.py's own
build_actual_lookup()/grade_result() (reused, not reimplemented) -- the
same real Data-Proc-MLB actual-value lookup and win/loss/push grading every
other backtest in this repo already trusts.

Known limitation, stated here rather than discovered downstream: this uses
CURRENT calibration/bet-profile-prior state when replaying each archived
date, not a point-in-time reconstruction of what those priors looked like
on that real day -- the same simplification optimize_walk_forward_policy.py
already makes for its own broad_candidate_policy() replay. A push (result
== "push") is excluded from the training target, matching every other real
grading path in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import select_high_precision_predictions as shp  # noqa: E402
from validate_historical_final_pools import (  # noqa: E402
    build_actual_lookup,
    grade_result,
    normalize_player_key,
)

REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "v11_eligible_training_set_2026.csv"
DEFAULT_REPORT_JSON = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "v11_eligible_training_set_report_2026.json"

# The live selector's exact real primary-policy args (sports/site/
# pipeline/run_daily_predictions.py's MLB_PRIMARY_POLICY_ARGS) --
# test_build_v11_eligible_training_set.py asserts this stays byte-
# identical to that list so this never silently trains against a stale
# policy. Kept as "V11_SELECTOR_ARGS" for continuity even though the live
# profile has since moved to v13 (a real book-count-gate relaxation only,
# not a selectivity change -- see run_daily_predictions.py's own comment)
# -- this constant always mirrors whatever the live selector currently
# runs, by construction of the sync test, regardless of its own name.
V11_SELECTOR_ARGS: list[str] = [
    "--top-n", "25",
    "--require-real-market-source",
    "--min-market-books", "1",
    "--min-common-market-books", "1",
    "--min-history-rows", "35",
    "--min-prediction", "0.10",
    "--min-hit-probability", "0.70",
    "--min-graded-hit-rate", "0.70",
    "--max-push-probability", "0.15",
    "--min-abs-edge", "0.10",
    "--min-expected-value", "0.15",
    "--pitcher-k-min-starter-history", "15",
    "--pitcher-k-min-projected-ip", "5.25",
    "--pitcher-k-min-projected-pitches", "75",
    "--pitcher-k-max-days-since-history", "14",
    "--pitcher-k-min-abs-edge", "0.15",
    "--pitcher-k-max-abs-edge", "1.0",
    "--pitcher-k-min-model-hit-probability", "0.50",
    "--pitcher-k-max-model-hit-probability", "0.65",
    "--pitcher-k-min-expected-value", "0.0",
    "--pitcher-k-min-american-price", "-130",
    "--pitcher-k-max-american-price", "130",
    "--max-pitcher-k-picks", "1",
    "--core-min-american-price", "-180",
    "--core-max-american-price", "125",
    "--min-over-picks", "0",
    "--max-over-picks", "25",
    "--max-under-picks", "0",
    "--daily-pick-soft-cap", "25",
    "--post-cap-min-selection-score", "0.50",
    "--max-per-market-bucket", "6",
    "--max-per-team", "6",
    "--min-historical-bet-profile-support", "0",
    "--min-historical-bet-profile-win-rate", "0",
    "--min-historical-market-availability-support", "0",
    "--min-historical-market-availability-rate", "0",
]


def find_raw_pool_csvs(daily_runs_root: Path) -> list[Path]:
    """Every archived date's *raw* pool CSV (daily_prediction_pool_<date>.csv)
    -- explicitly not the *_high_precision_predictions.csv sibling, which is
    already selected by whatever policy was live that day."""
    paths = []
    for date_dir in sorted(p for p in daily_runs_root.iterdir() if p.is_dir()):
        candidate = date_dir / f"daily_prediction_pool_{date_dir.name}.csv"
        if candidate.exists():
            paths.append(candidate)
    return paths


def parse_v11_args(pool_csv: Path) -> argparse.Namespace:
    """Real argparse.Namespace via select_high_precision_predictions.py's
    own parse_args() -- every default it doesn't explicitly set here comes
    from that parser, never hand-guessed."""
    argv = ["select_high_precision_predictions.py", "--pool-csv", str(pool_csv), *V11_SELECTOR_ARGS]
    original_argv = sys.argv
    sys.argv = argv
    try:
        return shp.parse_args()
    finally:
        sys.argv = original_argv


def candidate_to_row(candidate: Any, actual_lookup: dict[tuple[str, str, str, str], float]) -> dict[str, Any] | None:
    player_key = normalize_player_key(candidate.player)
    lookup_key = (candidate.run_date.isoformat(), player_key, candidate.target, str(candidate.game_id))
    actual = actual_lookup.get(lookup_key)
    if actual is None:
        return None
    result = grade_result(actual, candidate.market_line, candidate.direction)
    if result == "push":
        return None

    return {
        "date": candidate.run_date.isoformat(),
        "player": candidate.player,
        "game_id": candidate.game_id,
        "target": candidate.target,
        "direction": candidate.direction,
        "player_type": str(candidate.raw.get("Player_Type", "")).strip().lower(),
        "win": int(result == "win"),
        "model_hit_probability": candidate.calibrated_hit_probability,
        "model_graded_hit_rate": candidate.calibrated_graded_hit_rate,
        "survival_probability": candidate.survival_probability,
        "directional_edge": candidate.edge,
        "abs_edge": candidate.abs_edge,
        "market_implied_probability": candidate.market_implied_probability,
        "market_line_std": candidate.market_line_std,
        "market_books": candidate.market_books,
        "market_common_books": candidate.market_common_books,
        "history_rows": candidate.history_rows,
        "historical_bucket_win_rate": candidate.historical_bucket_win_rate,
        "historical_bucket_support": candidate.historical_bucket_support,
        "historical_bet_profile_win_rate": candidate.historical_bet_profile_win_rate,
        "historical_bet_profile_roi": candidate.historical_bet_profile_roi,
        "historical_bet_profile_support": candidate.historical_bet_profile_support,
        "historical_market_availability_rate": candidate.historical_market_availability_rate,
        "historical_market_availability_support": candidate.historical_market_availability_support,
        "live_confidence_calibration_adjustment": candidate.live_confidence_calibration_adjustment,
        "selected_side_price": candidate.selected_side_price,
        "price_confirmed": candidate.price_confirmed,
        "expected_value_per_unit": candidate.expected_value_per_unit,
        "market_bucket": candidate.market_bucket,
    }


def build_training_set(
    daily_runs_root: Path = DEFAULT_DAILY_RUNS_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_lookup = build_actual_lookup(processed_root)
    pool_csvs = find_raw_pool_csvs(daily_runs_root)

    rows: list[dict[str, Any]] = []
    per_date_counts: dict[str, int] = {}
    reject_totals: Counter = Counter()
    dates_with_zero_settled_eligible: list[str] = []
    errors: dict[str, str] = {}

    for pool_csv in pool_csvs:
        date_label = pool_csv.parent.name
        try:
            args = parse_v11_args(pool_csv)
            eligible, rejected = shp.prepare_and_filter_candidates(args)
        except Exception as exc:  # a single bad archived date must never abort the whole build
            errors[date_label] = f"{type(exc).__name__}: {exc}"
            continue
        reject_totals.update(rejected)

        settled_this_date = 0
        for candidate in eligible:
            row = candidate_to_row(candidate, actual_lookup)
            if row is not None:
                rows.append(row)
                settled_this_date += 1
        per_date_counts[date_label] = settled_this_date
        if settled_this_date == 0:
            dates_with_zero_settled_eligible.append(date_label)

    report = {
        "archived_dates_scanned": len(pool_csvs),
        "dates_with_load_errors": errors,
        "v11_eligible_settled_rows": len(rows),
        "v11_eligible_settled_dates": sum(1 for count in per_date_counts.values() if count > 0),
        "per_date_row_counts": per_date_counts,
        "dates_with_zero_settled_eligible_rows": dates_with_zero_settled_eligible,
        "structural_rejection_totals": dict(reject_totals),
    }
    return rows, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    rows, report = build_training_set(daily_runs_root=args.daily_runs_root, processed_root=args.processed_root)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import pandas as pd

        pd.DataFrame.from_records(rows).to_csv(args.out_csv, index=False)
    else:
        args.out_csv.write_text("", encoding="utf-8")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
