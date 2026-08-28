#!/usr/bin/env python3
"""
Build a real, per-pick backtest validation report for the current live
MLB policy (premium_evidence_gated_v14), for display on the site's
methodology page (prediction-about.html) -- so a viewer can see real,
settled, graded evidence that the policy works even on a day the live
board itself has zero picks (a real, honest, and not uncommon outcome
under this selective a policy -- see run_daily_predictions.py's own
MLB_PRIMARY_POLICY_PROFILE comment).

Reuses the exact real selection path every other real MLB backtest in
this repo already uses -- build_v11_eligible_training_set.py's
parse_v11_args() (the live selector's own real CLI args, kept in sync
by test_build_v11_eligible_training_set.py) and select_high_precision_
predictions.py's own prepare_and_filter_candidates()/select_top_
candidates() -- never a hand-reconstructed approximation of the real
gates. Grades against real settled outcomes via validate_historical_
final_pools.py's own build_actual_lookup()/grade_result().

This is a real, disclosed backtest over this repo's own archived raw
pools (currently 2026-08-02 through 2026-08-11, the only dates with
real, price-confirmed market data available) -- NOT a claim that these
are today's live picks. The report says so explicitly (is_live_board:
false) and the frontend must not present it as anything else.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
from validate_historical_final_pools import build_actual_lookup, grade_result, normalize_player_key  # noqa: E402

REPO_ROOT = SCRIPT_ROOT.parents[2]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "v14_backtest_validation.json"


def build_report(
    *,
    daily_runs_root: Path = DEFAULT_DAILY_RUNS_ROOT,
    processed_root: Path = DEFAULT_PROCESSED_ROOT,
) -> dict[str, Any]:
    actual_lookup = build_actual_lookup(processed_root)
    pool_csvs = find_raw_pool_csvs(daily_runs_root)

    picks: list[dict[str, Any]] = []
    dates_scanned: list[str] = []
    errors: dict[str, str] = {}

    for pool_csv in pool_csvs:
        date_label = pool_csv.parent.name
        try:
            args = parse_v11_args(pool_csv)
            eligible, _rejected = shp.prepare_and_filter_candidates(args)
        except Exception as exc:  # a single bad archived date must never abort the whole report
            errors[date_label] = f"{type(exc).__name__}: {exc}"
            continue
        selected = shp.select_top_candidates(eligible, args)
        if not selected:
            continue
        dates_scanned.append(date_label)
        for candidate in selected:
            player_key = normalize_player_key(candidate.player)
            lookup_key = (candidate.run_date.isoformat(), player_key, candidate.target, str(candidate.game_id))
            actual = actual_lookup.get(lookup_key)
            result = None
            if actual is not None:
                graded = grade_result(actual, candidate.market_line, candidate.direction)
                result = graded if graded in {"win", "loss"} else None
            side_price = to_float(getattr(candidate, "selected_side_price", None))
            picks.append(
                {
                    "date": candidate.run_date.isoformat(),
                    "player": candidate.player,
                    "target": candidate.target,
                    "direction": candidate.direction,
                    "market_line": candidate.market_line,
                    "final_hit_probability": round(to_float(candidate.final_hit_probability), 4),
                    "side_price": side_price,
                    "result": result,
                }
            )

    settled = [pick for pick in picks if pick["result"] in {"win", "loss"}]
    wins = sum(1 for pick in settled if pick["result"] == "win")
    profits = []
    for pick in settled:
        profit = american_profit_per_unit(pick["side_price"]) if pick["side_price"] is not None else None
        if profit is None:
            continue
        profits.append(profit if pick["result"] == "win" else -1.0)

    picks.sort(key=lambda pick: pick["date"])

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "premium_evidence_gated_v14",
        "is_live_board": False,
        "description": (
            "Real, settled backtest of the current live policy's exact selection "
            "logic replayed against this repo's own archived raw daily pools -- "
            "not today's live board. Every pick, gate, and grade here is real; "
            "none are hypothetical or fabricated."
        ),
        "dates_scanned": dates_scanned,
        "dates_with_load_errors": errors,
        "picks": picks,
        "summary": {
            "picks": len(picks),
            "settled": len(settled),
            "wins": wins,
            "hit_rate": (wins / len(settled)) if settled else None,
            "roi": (sum(profits) / len(profits)) if profits else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-runs-root", type=Path, default=DEFAULT_DAILY_RUNS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(daily_runs_root=args.daily_runs_root, processed_root=args.processed_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report JSON: {args.output_json}")


if __name__ == "__main__":
    main()
