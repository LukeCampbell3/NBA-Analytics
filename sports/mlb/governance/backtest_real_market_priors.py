#!/usr/bin/env python3
"""Walk-forward comparison of mixed-line and executable-market MLB priors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.scripts.select_high_precision_predictions import (  # noqa: E402
    HISTORICAL_BET_TARGET_SPECS,
    blend_probability_with_prior,
    estimate_count_hit_probabilities,
    grade_result,
    market_bucket_key,
    target_direction_key,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_OUTPUT = REPO_ROOT / "sports/mlb/data/predictions/backtests/real_market_prior_walk_forward.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-prior-rows", type=int, default=50)
    parser.add_argument("--authorization-probability", type=float, default=0.75)
    return parser.parse_args()


def load_rows(root: Path, season: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    required = {"Date"}
    for columns in HISTORICAL_BET_TARGET_SPECS.values():
        required.update(columns)
    for path in sorted(root.glob(f"*/{season}_processed_processed.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in required)
        except Exception:
            continue
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for target, columns in HISTORICAL_BET_TARGET_SPECS.items():
            actual_col, line_col, gap_col, source_col, books_col, over_col, under_col = columns
            if not set(columns).issubset(frame.columns):
                continue
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            line = pd.to_numeric(frame[line_col], errors="coerce")
            gap = pd.to_numeric(frame[gap_col], errors="coerce")
            books = pd.to_numeric(frame[books_col], errors="coerce").fillna(0)
            over_price = pd.to_numeric(frame[over_col], errors="coerce")
            under_price = pd.to_numeric(frame[under_col], errors="coerce")
            source = frame[source_col].astype(str).str.strip().str.lower()
            mask = dates.notna() & actual.notna() & line.notna() & gap.notna() & gap.ne(0)
            for index in frame.index[mask]:
                direction = "OVER" if gap.at[index] > 0 else "UNDER"
                price = over_price.at[index] if direction == "OVER" else under_price.at[index]
                prediction = max(0.0, float(line.at[index] + gap.at[index]))
                model_probability = estimate_count_hit_probabilities(
                    prediction, float(line.at[index]), direction
                )[2]
                records.append(
                    {
                        "date": dates.at[index],
                        "target": target,
                        "direction": direction,
                        "line": float(line.at[index]),
                        "actual": float(actual.at[index]),
                        "model_probability": float(model_probability),
                        "result": grade_result(float(actual.at[index]), float(line.at[index]), direction),
                        "real_priced": bool(
                            source.at[index] == "real"
                            and books.at[index] > 0
                            and pd.notna(price)
                            and abs(float(price)) >= 100.0
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def empirical_prior(
    counts: dict[str, list[int]], target: str, direction: str, line: float, minimum_rows: int
) -> tuple[float, int]:
    line_values = counts.get(market_bucket_key(target, direction, line), [0, 0])
    if sum(line_values) >= minimum_rows:
        return line_values[0] / sum(line_values), sum(line_values)
    target_values = counts.get(target_direction_key(target, direction), [0, 0])
    if sum(target_values):
        return target_values[0] / sum(target_values), sum(target_values)
    return 0.5, 0


def update_counts(counts: dict[str, list[int]], row: Any) -> None:
    if row.result == "push":
        return
    value = int(row.result == "win")
    for key in (
        market_bucket_key(row.target, row.direction, row.line),
        target_direction_key(row.target, row.direction),
    ):
        counts[key][0] += value
        counts[key][1] += 1 - value


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in rows if row["probability"] >= threshold]
    if not selected:
        return {"selections": 0, "wins": 0, "losses": 0, "hit_rate": None, "mean_probability": None, "brier_score": None}
    wins = sum(int(row["win"]) for row in selected)
    return {
        "selections": len(selected),
        "wins": wins,
        "losses": len(selected) - wins,
        "hit_rate": wins / len(selected),
        "mean_probability": sum(row["probability"] for row in selected) / len(selected),
        "calibration_gap": (sum(row["probability"] for row in selected) / len(selected)) - (wins / len(selected)),
        "brier_score": sum((row["probability"] - row["win"]) ** 2 for row in selected) / len(selected),
    }


def run_backtest(frame: pd.DataFrame, *, minimum_rows: int, threshold: float) -> dict[str, Any]:
    mixed_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    real_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mixed_predictions: list[dict[str, Any]] = []
    real_predictions: list[dict[str, Any]] = []
    dates = sorted(frame["date"].dropna().unique())
    for evaluation_date in dates:
        day = frame.loc[frame["date"] == evaluation_date]
        for row in day.loc[day["real_priced"]].itertuples(index=False):
            if row.result == "push":
                continue
            for counts, output in ((mixed_counts, mixed_predictions), (real_counts, real_predictions)):
                prior, support = empirical_prior(counts, row.target, row.direction, row.line, minimum_rows)
                probability, _ = blend_probability_with_prior(
                    row.model_probability,
                    prior_probability=prior,
                    support=support,
                    max_weight=0.35,
                    strength=400.0,
                )
                output.append({"date": evaluation_date, "probability": probability, "win": row.result == "win"})
        for row in day.itertuples(index=False):
            update_counts(mixed_counts, row)
            if row.real_priced:
                update_counts(real_counts, row)
    split_index = max(1, min(len(dates) - 1, math.floor(len(dates) * 0.75)))
    development_dates = set(dates[:split_index])
    holdout_dates = set(dates[split_index:])
    development_real = [row for row in real_predictions if row["date"] in development_dates]
    minimum_development_selections = max(30, math.ceil(len(development_real) * 0.15))
    threshold_grid = [0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90]
    development_grid = []
    for candidate_threshold in threshold_grid:
        result = summarize(development_real, candidate_threshold)
        development_grid.append({"threshold": candidate_threshold, **result})
    eligible_thresholds = [
        row for row in development_grid if int(row["selections"]) >= minimum_development_selections
    ]
    chosen = (
        max(
            eligible_thresholds,
            key=lambda row: (
                float(row["hit_rate"] or 0.0),
                -float(row["brier_score"] or 1.0),
                int(row["selections"]),
            ),
        )
        if eligible_thresholds
        else {"threshold": threshold}
    )
    chosen_threshold = float(chosen["threshold"])
    holdout_mixed = [row for row in mixed_predictions if row["date"] in holdout_dates]
    holdout_real = [row for row in real_predictions if row["date"] in holdout_dates]
    return {
        "method": "strict_date_ordered_walk_forward",
        "evaluation_universe": "real_price_confirmed_market_rows_only",
        "prior_training": {
            "old": "all_market_rows_including_synthetic",
            "new": "real_price_confirmed_market_rows_only",
        },
        "evaluation_dates": len(dates),
        "minimum_prior_rows": minimum_rows,
        "authorization_probability": threshold,
        "old_mixed_prior": summarize(mixed_predictions, threshold),
        "new_real_market_prior": summarize(real_predictions, threshold),
        "threshold_selection": {
            "method": "first_75_percent_development_last_25_percent_locked_holdout",
            "development_dates": len(development_dates),
            "holdout_dates": len(holdout_dates),
            "minimum_development_selections": minimum_development_selections,
            "development_grid": development_grid,
            "selected_threshold": chosen_threshold,
            "holdout_old_mixed_prior": summarize(holdout_mixed, chosen_threshold),
            "holdout_new_real_market_prior": summarize(holdout_real, chosen_threshold),
        },
    }


def main() -> None:
    args = parse_args()
    frame = load_rows(args.processed_root.resolve(), int(args.season))
    if frame.empty:
        raise SystemExit("No historical rows were available.")
    report = run_backtest(
        frame,
        minimum_rows=int(args.minimum_prior_rows),
        threshold=float(args.authorization_probability),
    )
    report.update(
        {
            "schema_version": 1,
            "season": int(args.season),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_rows": int(len(frame)),
            "real_price_confirmed_rows": int(frame["real_priced"].sum()),
            "limitations": [
                "This isolates probability-prior behavior; it is not a replay of every production gate.",
                "Historical real-market coverage is sparse and results do not certify future return.",
            ],
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
