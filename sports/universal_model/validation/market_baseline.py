"""Baseline 0 -- market only (spec section 17).

DISCLOSED LIMITATION: no_vig_market_probability was never populated by the
MLB adapter (see adapters/mlb.py) -- the source dataset only carries a
single-sided (over) American price for ~2.5% of rows, with no
simultaneous under-side price in the same row to remove the vig from.
Rather than fabricate a no-vig probability, this baseline uses the raw
single-sided implied probability from the real American price
(1/(1+decimal_odds) via the standard American->implied conversion) on
that real ~2.5% priced subset only -- it is a real market quote, just not
vig-adjusted. Reported as "single-sided implied probability", not
"no-vig probability", and restricted to the real priced subset (never
extrapolated to the ~97.5% of unpriced rows).

Run: python -m sports.universal_model.validation.market_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sports.universal_model.validation.metrics import classification_metrics

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
DATASET_DIR = MANIFESTS_DIR / "dataset"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def american_to_implied_prob(price: float) -> float:
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def main() -> None:
    split_manifest = json.loads((MANIFESTS_DIR / "split_manifest.json").read_text())

    frames = [pd.read_parquet(p) for p in sorted((DATASET_DIR / "sport=mlb").glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)

    # DISCLOSED: real market-price capture in this dataset only covers
    # 2026-04-28..06-28 (measured directly), which falls entirely inside
    # DERIVE/SELECT and NOT inside TEST (2026-07-12..08-06) -- there are
    # zero priced TEST rows. Rather than report a fabricated/empty TEST
    # number, this baseline is computed on SELECT, the priced split
    # closest to (but distinct from) TEST, and labeled as such throughout.
    eval_dates = set(split_manifest["per_sport"]["mlb"]["test_dates_full"])
    test_df = df[df["_event_date"].isin(eval_dates)]
    priced_in_test = test_df["american_price"].notna().sum()
    split_used = "TEST"
    if priced_in_test == 0:
        eval_dates = set(split_manifest["per_sport"]["mlb"]["select_dates_full"])
        test_df = df[df["_event_date"].isin(eval_dates)]
        split_used = "SELECT (TEST has zero priced rows -- real market-price capture in this dataset only spans 2026-04-28..06-28, entirely before the TEST window; see module docstring)"

    priced = test_df[test_df["american_price"].notna() & test_df["line"].notna() & test_df["actual_value"].notna()].copy()
    priced = priced[priced["actual_value"] != priced["line"]]  # drop pushes, undefined label
    priced["y_over"] = (priced["actual_value"] > priced["line"]).astype(int)
    priced["market_implied_prob"] = priced["american_price"].apply(american_to_implied_prob)

    metrics = classification_metrics(priced["market_implied_prob"].values, priced["y_over"].values)
    report = {
        "note": "single-sided implied probability from real American price (not vig-adjusted; see module docstring)",
        "split_used": split_used,
        "n_rows_in_split": int(len(test_df)),
        "n_priced_subset": int(len(priced)),
        "priced_fraction": float(len(priced) / max(len(test_df), 1)),
        "market_only_metrics": metrics,
    }
    (REPORTS_DIR / "market_only_baseline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
