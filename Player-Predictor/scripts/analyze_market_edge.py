#!/usr/bin/env python3
"""
Analyze model-vs-market directional edge from row-level backtest output.

This script is built for the CSV produced by:
    scripts/backtest_inference_accuracy.py --csv-out ...

It focuses on:
1. directional performance versus the market line
2. bias (pred - market)
3. disagreement edge curves
4. top-quartile and top-decile disagreement strategy backtests
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PTS", "TRB", "AST"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze model edge versus market lines.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model/analysis/latest_market_comparison_strict_rows.csv"),
        help="Row-level CSV from backtest_inference_accuracy.py",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
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


def safe_rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def maybe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def build_direction_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    market_col = f"market_{target}"
    pred_col = f"pred_{target}"
    actual_col = f"actual_{target}"
    covered = df.loc[df[market_col].notna()].copy()
    if covered.empty:
        return covered

    covered["pred_minus_market"] = pd.to_numeric(covered[pred_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
    covered["actual_minus_market"] = pd.to_numeric(covered[actual_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
    covered["abs_disagreement"] = covered["pred_minus_market"].abs()
    belief = pd.to_numeric(covered.get("belief_uncertainty"), errors="coerce").fillna(1.0)
    feas = pd.to_numeric(covered.get("feasibility"), errors="coerce").fillna(0.0)
    covered["edge_confidence"] = covered["abs_disagreement"] * np.clip(1.0 - belief, 0.0, 1.0) * np.clip(feas, 0.0, None)
    covered["model_over"] = covered["pred_minus_market"] > 0
    covered["model_under"] = covered["pred_minus_market"] < 0
    covered["model_push"] = covered["pred_minus_market"] == 0
    covered["actual_over"] = covered["actual_minus_market"] > 0
    covered["actual_under"] = covered["actual_minus_market"] < 0
    covered["actual_push"] = covered["actual_minus_market"] == 0
    covered["directional_correct"] = (
        (covered["model_over"] & covered["actual_over"])
        | (covered["model_under"] & covered["actual_under"])
    )
    covered["directional_called"] = covered["model_over"] | covered["model_under"]
    covered["model_abs_err"] = (pd.to_numeric(covered[pred_col], errors="coerce") - pd.to_numeric(covered[actual_col], errors="coerce")).abs()
    covered["market_abs_err"] = (pd.to_numeric(covered[market_col], errors="coerce") - pd.to_numeric(covered[actual_col], errors="coerce")).abs()
    return covered


def summarize_directional(covered: pd.DataFrame) -> dict:
    over_calls = int(covered["model_over"].sum())
    under_calls = int(covered["model_under"].sum())
    push_calls = int(covered["model_push"].sum())
    over_wins = int((covered["model_over"] & covered["actual_over"]).sum())
    under_wins = int((covered["model_under"] & covered["actual_under"]).sum())
    push_actual_pushes = int((covered["model_push"] & covered["actual_push"]).sum())
    directional_calls = over_calls + under_calls
    directional_wins = over_wins + under_wins
    return {
        "rows": int(len(covered)),
        "avg_pred_minus_market": maybe_float(covered["pred_minus_market"].mean()),
        "median_pred_minus_market": maybe_float(covered["pred_minus_market"].median()),
        "avg_abs_disagreement": maybe_float(covered["abs_disagreement"].mean()),
        "avg_edge_confidence": maybe_float(covered["edge_confidence"].mean()),
        "over_calls": over_calls,
        "over_wins": over_wins,
        "over_losses": over_calls - over_wins,
        "over_win_rate": safe_rate(over_wins, over_calls),
        "under_calls": under_calls,
        "under_wins": under_wins,
        "under_losses": under_calls - under_wins,
        "under_win_rate": safe_rate(under_wins, under_calls),
        "push_calls": push_calls,
        "push_actual_pushes": push_actual_pushes,
        "directional_calls": directional_calls,
        "directional_wins": directional_wins,
        "directional_win_rate": safe_rate(directional_wins, directional_calls),
    }


def add_quantile_bins(values: pd.Series, labels: list[str]) -> pd.Series:
    ranks = values.rank(method="first", pct=True)
    bins = pd.cut(
        ranks,
        bins=np.linspace(0.0, 1.0, len(labels) + 1),
        labels=labels,
        include_lowest=True,
    )
    return bins.astype(str)


def build_bucket_summary(covered: pd.DataFrame) -> list[dict]:
    working = covered.loc[covered["directional_called"]].copy()
    if working.empty:
        return []
    labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    working["disagreement_bucket"] = add_quantile_bins(working["abs_disagreement"], labels)
    results = []
    for label in labels:
        bucket = working.loc[working["disagreement_bucket"] == label]
        if bucket.empty:
            continue
        calls = int(len(bucket))
        wins = int(bucket["directional_correct"].sum())
        results.append(
            {
                "bucket": label,
                "rows": calls,
                "avg_abs_disagreement": maybe_float(bucket["abs_disagreement"].mean()),
                "avg_edge_confidence": maybe_float(bucket["edge_confidence"].mean()),
                "directional_win_rate": safe_rate(wins, calls),
                "model_mae": maybe_float(bucket["model_abs_err"].mean()),
                "market_mae": maybe_float(bucket["market_abs_err"].mean()),
                "improvement_vs_market": maybe_float(bucket["market_abs_err"].mean() - bucket["model_abs_err"].mean()),
            }
        )
    return results


def summarize_strategy_slice(slice_df: pd.DataFrame) -> dict:
    calls = int(slice_df["directional_called"].sum())
    wins = int(slice_df["directional_correct"].sum())
    over_calls = int(slice_df["model_over"].sum())
    under_calls = int(slice_df["model_under"].sum())
    over_wins = int((slice_df["model_over"] & slice_df["actual_over"]).sum())
    under_wins = int((slice_df["model_under"] & slice_df["actual_under"]).sum())
    return {
        "rows": int(len(slice_df)),
        "directional_calls": calls,
        "directional_wins": wins,
        "directional_win_rate": safe_rate(wins, calls),
        "model_mae": maybe_float(slice_df["model_abs_err"].mean()),
        "market_mae": maybe_float(slice_df["market_abs_err"].mean()),
        "improvement_vs_market": maybe_float(slice_df["market_abs_err"].mean() - slice_df["model_abs_err"].mean()),
        "avg_pred_minus_market": maybe_float(slice_df["pred_minus_market"].mean()),
        "avg_abs_disagreement": maybe_float(slice_df["abs_disagreement"].mean()),
        "avg_edge_confidence": maybe_float(slice_df["edge_confidence"].mean()),
        "over_calls": over_calls,
        "over_win_rate": safe_rate(over_wins, over_calls),
        "under_calls": under_calls,
        "under_win_rate": safe_rate(under_wins, under_calls),
    }


def build_strategy_backtests(covered: pd.DataFrame) -> dict:
    working = covered.loc[covered["directional_called"]].copy()
    if working.empty:
        return {}
    working = working.sort_values("abs_disagreement", ascending=False).reset_index(drop=True)
    quartile_n = max(1, int(np.ceil(len(working) * 0.25)))
    decile_n = max(1, int(np.ceil(len(working) * 0.10)))
    confidence_quartile_n = max(1, int(np.ceil(len(working) * 0.25)))
    confidence_decile_n = max(1, int(np.ceil(len(working) * 0.10)))
    working_conf = working.sort_values("edge_confidence", ascending=False).reset_index(drop=True)
    return {
        "all_directional": summarize_strategy_slice(working),
        "top_quartile_disagreement": summarize_strategy_slice(working.head(quartile_n)),
        "top_decile_disagreement": summarize_strategy_slice(working.head(decile_n)),
        "top_quartile_confidence": summarize_strategy_slice(working_conf.head(confidence_quartile_n)),
        "top_decile_confidence": summarize_strategy_slice(working_conf.head(confidence_decile_n)),
    }


def analyze_subset(df: pd.DataFrame) -> dict:
    subset_summary = {"rows": int(len(df)), "targets": {}}
    for target in TARGETS:
        covered = build_direction_frame(df, target)
        if covered.empty:
            continue
        subset_summary["targets"][target] = {
            "directional": summarize_directional(covered),
            "bucket_curve": build_bucket_summary(covered),
            "strategy_backtests": build_strategy_backtests(covered),
        }
    return subset_summary


def print_subset(label: str, summary: dict) -> None:
    print("\n" + "=" * 90)
    print(f"EDGE FILTER REPORT: {label}")
    print("=" * 90)
    print(f"Rows: {summary['rows']}")
    for target in TARGETS:
        payload = summary["targets"].get(target)
        if not payload:
            continue
        directional = payload["directional"]
        print(f"\n{target}:")
        print(
            f"  Bias (pred-market): mean={directional['avg_pred_minus_market']:+.3f} "
            f"median={directional['median_pred_minus_market']:+.3f}"
        )
        print(
            f"  Over wins: {directional['over_wins']}/{directional['over_calls']} "
            f"({(directional['over_win_rate'] or 0.0) * 100:.1f}%)"
        )
        print(
            f"  Under wins: {directional['under_wins']}/{directional['under_calls']} "
            f"({(directional['under_win_rate'] or 0.0) * 100:.1f}%)"
        )
        print(
            f"  Directional win rate: {directional['directional_wins']}/{directional['directional_calls']} "
            f"({(directional['directional_win_rate'] or 0.0) * 100:.1f}%)"
        )

        print("  Disagreement edge curve:")
        for bucket in payload["bucket_curve"]:
            print(
                f"    {bucket['bucket']}: rows={bucket['rows']} "
                f"avg_gap={bucket['avg_abs_disagreement']:.3f} "
                f"win_rate={(bucket['directional_win_rate'] or 0.0) * 100:.1f}% "
                f"model_mae={bucket['model_mae']:.3f} "
                f"market_mae={bucket['market_mae']:.3f}"
            )

        strategies = payload["strategy_backtests"]
        print("  Strategy backtests:")
        for strategy_name in [
            "all_directional",
            "top_quartile_disagreement",
            "top_decile_disagreement",
            "top_quartile_confidence",
            "top_decile_confidence",
        ]:
            strat = strategies.get(strategy_name)
            if not strat:
                continue
            print(
                f"    {strategy_name}: rows={strat['rows']} "
                f"win_rate={(strat['directional_win_rate'] or 0.0) * 100:.1f}% "
                f"model_mae={strat['model_mae']:.3f} "
                f"market_mae={strat['market_mae']:.3f} "
                f"avg_gap={strat['avg_abs_disagreement']:.3f}"
            )


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    active_df = df.loc[active_only_mask(df)].copy()

    report = {
        "source_csv": str(csv_path),
        "all_rows": analyze_subset(df),
        "active_only": analyze_subset(active_df),
    }

    print_subset("ALL ROWS", report["all_rows"])
    print_subset("ACTIVE ONLY", report["active_only"])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.json_out}")


if __name__ == "__main__":
    main()
