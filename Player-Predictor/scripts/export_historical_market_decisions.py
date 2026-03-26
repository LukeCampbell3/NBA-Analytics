#!/usr/bin/env python3
"""
Export historical market-comparison rows into a long-form decisions table.

Input:
- row-level CSV from scripts/backtest_inference_accuracy.py --csv-out

Output:
- one row per player / target / historical prediction
- includes a single `actual` column for easier auditing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PTS", "TRB", "AST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a long-form historical decisions table.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model/analysis/latest_market_comparison_strict_rows.csv"),
        help="Wide row-level CSV from backtest_inference_accuracy.py",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/historical_market_decisions_long.csv"),
        help="Output long-form CSV path",
    )
    return parser.parse_args()


def active_only_mask(df: pd.DataFrame) -> pd.Series:
    minutes = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    return (
        (pd.to_numeric(df.get("did_not_play"), errors="coerce").fillna(0.0) < 0.5)
        & ~(
            (pd.to_numeric(df.get("actual_PTS"), errors="coerce").fillna(0.0) == 0.0)
            & (pd.to_numeric(df.get("actual_TRB"), errors="coerce").fillna(0.0) == 0.0)
            & (pd.to_numeric(df.get("actual_AST"), errors="coerce").fillna(0.0) == 0.0)
            & (minutes <= 0.0)
        )
    )


def build_long_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    active_mask = active_only_mask(df)

    for _, row in df.iterrows():
        for target in TARGETS:
            market_line = row.get(f"market_{target}", np.nan)
            prediction = row.get(f"pred_{target}", np.nan)
            actual = row.get(f"actual_{target}", np.nan)
            if pd.isna(market_line) or pd.isna(prediction) or pd.isna(actual):
                continue

            edge = float(prediction - market_line)
            actual_minus_market = float(actual - market_line)
            if edge > 0:
                direction = "OVER"
            elif edge < 0:
                direction = "UNDER"
            else:
                direction = "PUSH"

            if direction == "OVER":
                result = "win" if actual_minus_market > 0 else ("push" if actual_minus_market == 0 else "loss")
            elif direction == "UNDER":
                result = "win" if actual_minus_market < 0 else ("push" if actual_minus_market == 0 else "loss")
            else:
                result = "push" if actual_minus_market == 0 else "loss"

            records.append(
                {
                    "player": row.get("player"),
                    "target": target,
                    "target_date": row.get("target_date"),
                    "csv": row.get("csv"),
                    "target_index": row.get("target_index"),
                    "prediction": float(prediction),
                    "market_line": float(market_line),
                    "actual": float(actual),
                    "baseline": float(row.get(f"baseline_{target}", np.nan)) if pd.notna(row.get(f"baseline_{target}", np.nan)) else np.nan,
                    "edge": edge,
                    "abs_edge": abs(edge),
                    "actual_minus_market": actual_minus_market,
                    "direction": direction,
                    "result": result,
                    "model_beats_market_error": bool(abs(float(prediction - actual)) < abs(float(market_line - actual))),
                    "belief_uncertainty": float(row.get("belief_uncertainty", np.nan)) if pd.notna(row.get("belief_uncertainty", np.nan)) else np.nan,
                    "feasibility": float(row.get("feasibility", np.nan)) if pd.notna(row.get("feasibility", np.nan)) else np.nan,
                    "uncertainty_sigma": float(row.get(f"{target}_uncertainty_sigma", np.nan)) if pd.notna(row.get(f"{target}_uncertainty_sigma", np.nan)) else np.nan,
                    "spike_probability": float(row.get(f"{target}_spike_probability", np.nan)) if pd.notna(row.get(f"{target}_spike_probability", np.nan)) else np.nan,
                    "fallback_blend": float(row.get("fallback_blend", np.nan)) if pd.notna(row.get("fallback_blend", np.nan)) else np.nan,
                    "fallback_reasons": row.get("fallback_reasons", ""),
                    "schema_repaired": bool(row.get("schema_repaired", False)),
                    "used_default_ids": bool(row.get("used_default_ids", False)),
                    "nan_feature_repaired": bool(row.get("nan_feature_repaired", False)),
                    "active_only_row": bool(active_mask.loc[row.name]),
                }
            )

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        return long_df
    long_df = long_df.sort_values(["target_date", "player", "target"]).reset_index(drop=True)
    return long_df


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    long_df = build_long_rows(df)
    if long_df.empty:
        raise RuntimeError("No long-form rows were produced from the input CSV.")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.csv_out, index=False)

    print("\n" + "=" * 90)
    print("HISTORICAL MARKET DECISIONS EXPORTED")
    print("=" * 90)
    print(f"Input rows:   {len(df)}")
    print(f"Output rows:  {len(long_df)}")
    print(f"CSV:          {args.csv_out}")
    print("\nSample:")
    show_cols = ["player", "target", "target_date", "prediction", "market_line", "actual", "direction", "result", "edge"]
    print(long_df[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
