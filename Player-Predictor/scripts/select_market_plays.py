#!/usr/bin/env python3
"""
Rank upcoming market plays using historical model-vs-market edge behavior.

Inputs:
- upcoming slate CSV from scripts/build_upcoming_slate.py
- historical row-level CSV from scripts/backtest_inference_accuracy.py --csv-out

The selector:
- computes target-specific disagreement percentiles
- applies conservative per-target thresholds
- maps each candidate to an expected win rate from historical backtests
- ranks filtered plays for decision use
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["PTS", "TRB", "AST"]
TARGET_THRESHOLDS = {
    "PTS": {"consider_pct": 0.75, "strong_pct": 0.90},
    "TRB": {"consider_pct": 0.85, "strong_pct": 0.95},
    "AST": {"consider_pct": 0.85, "strong_pct": 0.95},
}
HEURISTIC_EDGE_SCALES = {
    "PTS": 3.0,
    "TRB": 1.2,
    "AST": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and rank upcoming market plays.")
    parser.add_argument(
        "--slate-csv",
        type=Path,
        default=Path("model/analysis/upcoming_market_slate.csv"),
        help="Upcoming slate CSV from build_upcoming_slate.py",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("model/analysis/latest_market_comparison_strict_rows.csv"),
        help="Historical row-level backtest CSV",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.csv"),
        help="Output ranked plays CSV path",
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


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def build_history_lookup(history_df: pd.DataFrame) -> dict[str, dict]:
    active_history = history_df.loc[active_only_mask(history_df)].copy()
    lookup: dict[str, dict] = {}
    for target in TARGETS:
        market_col = f"market_{target}"
        pred_col = f"pred_{target}"
        actual_col = f"actual_{target}"
        covered = active_history.loc[active_history[market_col].notna()].copy()
        if covered.empty:
            continue
        pred_minus_market = pd.to_numeric(covered[pred_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
        actual_minus_market = pd.to_numeric(covered[actual_col], errors="coerce") - pd.to_numeric(covered[market_col], errors="coerce")
        abs_gap = pred_minus_market.abs()
        called = pred_minus_market != 0
        correct = ((pred_minus_market > 0) & (actual_minus_market > 0)) | ((pred_minus_market < 0) & (actual_minus_market < 0))
        working = pd.DataFrame(
            {
                "pred_minus_market": pred_minus_market,
                "actual_minus_market": actual_minus_market,
                "abs_gap": abs_gap,
                "directional_called": called,
                "directional_correct": correct,
            }
        )
        working = working.loc[working["directional_called"]].copy()
        if working.empty:
            continue

        quartile_cut = float(working["abs_gap"].quantile(0.75))
        decile_cut = float(working["abs_gap"].quantile(0.90))

        def rate(mask: pd.Series) -> float | None:
            subset = working.loc[mask]
            if subset.empty:
                return None
            return float(subset["directional_correct"].mean())

        lookup[target] = {
            "all_rate": float(working["directional_correct"].mean()),
            "quartile_cut": quartile_cut,
            "decile_cut": decile_cut,
            "top_quartile_rate": rate(working["abs_gap"] >= quartile_cut),
            "top_decile_rate": rate(working["abs_gap"] >= decile_cut),
            "gaps_sorted": np.sort(working["abs_gap"].to_numpy(dtype=float)),
        }
    return lookup


def percentile_of_gap(gaps_sorted: np.ndarray, gap: float) -> float:
    if gaps_sorted.size == 0:
        return 0.0
    rank = np.searchsorted(gaps_sorted, gap, side="right")
    return float(rank / gaps_sorted.size)


def classify_play(target: str, percentile: float) -> str:
    thresholds = TARGET_THRESHOLDS[target]
    if percentile >= thresholds["strong_pct"]:
        return "strong"
    if percentile >= thresholds["consider_pct"]:
        return "consider"
    return "pass"


def expected_rate_for(target: str, percentile: float, history_info: dict) -> float:
    thresholds = TARGET_THRESHOLDS[target]
    if percentile >= thresholds["strong_pct"] and history_info.get("top_decile_rate") is not None:
        return float(history_info["top_decile_rate"])
    if percentile >= thresholds["consider_pct"] and history_info.get("top_quartile_rate") is not None:
        return float(history_info["top_quartile_rate"])
    return float(history_info["all_rate"])


def heuristic_percentile_and_rate(target: str, abs_gap: float) -> tuple[float, float]:
    scale = float(HEURISTIC_EDGE_SCALES[target])
    gap_pct = float(np.clip(abs_gap / scale, 0.01, 0.99))
    expected_rate = float(np.clip(0.50 + 0.35 * gap_pct, 0.50, 0.85))
    return gap_pct, expected_rate


def build_play_rows(slate_df: pd.DataFrame, history_lookup: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in slate_df.iterrows():
        belief = safe_float(row.get("belief_uncertainty"), default=1.0)
        feas = max(0.0, safe_float(row.get("feasibility"), default=0.0))
        for target in TARGETS:
            history_info = history_lookup.get(target)
            pred = safe_float(row.get(f"pred_{target}"))
            market = safe_float(row.get(f"market_{target}"))
            if np.isnan(pred) or np.isnan(market):
                continue
            edge = pred - market
            if edge == 0.0:
                direction = "PUSH"
            elif edge > 0.0:
                direction = "OVER"
            else:
                direction = "UNDER"
            abs_gap = abs(edge)
            if history_info is None:
                gap_pct, expected_rate = heuristic_percentile_and_rate(target, abs_gap)
            else:
                gap_pct = percentile_of_gap(history_info["gaps_sorted"], abs_gap)
                expected_rate = expected_rate_for(target, gap_pct, history_info)
            recommendation = classify_play(target, gap_pct)
            confidence_score = abs_gap * np.clip(1.0 - belief, 0.0, 1.0) * feas
            rows.append(
                {
                    "player": row["player"],
                    "market_date": row.get("market_date"),
                    "target": target,
                    "direction": direction,
                    "prediction": pred,
                    "market_line": market,
                    "edge": edge,
                    "abs_edge": abs_gap,
                    "gap_percentile": gap_pct,
                    "recommendation": recommendation,
                    "expected_win_rate": expected_rate,
                    "confidence_score": confidence_score,
                    "belief_uncertainty": belief,
                    "feasibility": feas,
                    "fallback_blend": safe_float(row.get("fallback_blend"), default=0.0),
                    "market_books": safe_float(row.get(f"market_books_{target}"), default=np.nan),
                    "baseline": safe_float(row.get(f"baseline_{target}"), default=np.nan),
                    "baseline_edge": safe_float(row.get(f"baseline_edge_{target}"), default=np.nan),
                    "uncertainty_sigma": safe_float(row.get(f"{target}_uncertainty_sigma"), default=np.nan),
                    "spike_probability": safe_float(row.get(f"{target}_spike_probability"), default=np.nan),
                    "history_rows": int(row.get("history_rows", 0)),
                    "last_history_date": row.get("last_history_date"),
                    "csv": row.get("csv"),
                }
            )
    plays = pd.DataFrame.from_records(rows)
    if plays.empty:
        return plays
    plays = plays.sort_values(
        ["recommendation", "expected_win_rate", "confidence_score", "abs_edge"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return plays


def recommendation_rank(label: str) -> int:
    order = {"strong": 0, "consider": 1, "pass": 2}
    return order.get(label, 3)


def main() -> None:
    args = parse_args()
    slate_path = args.slate_csv.resolve()
    history_path = args.history_csv.resolve()
    if not slate_path.exists():
        raise FileNotFoundError(f"Slate CSV not found: {slate_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"History CSV not found: {history_path}")

    slate_df = pd.read_csv(slate_path)
    history_df = pd.read_csv(history_path)
    history_lookup = build_history_lookup(history_df)
    plays = build_play_rows(slate_df, history_lookup)
    if plays.empty:
        raise RuntimeError("No playable rows were produced from the provided slate/history inputs.")

    plays["recommendation_rank"] = plays["recommendation"].map(recommendation_rank)
    plays = plays.sort_values(
        ["recommendation_rank", "expected_win_rate", "confidence_score", "abs_edge"],
        ascending=[True, False, False, False],
    ).drop(columns=["recommendation_rank"]).reset_index(drop=True)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    plays.to_csv(args.csv_out, index=False)

    summary = {
        "slate_csv": str(slate_path),
        "history_csv": str(history_path),
        "n_plays": int(len(plays)),
        "recommendation_counts": plays["recommendation"].value_counts().to_dict(),
        "top_strong": plays.loc[plays["recommendation"] == "strong"].head(10).to_dict(orient="records"),
        "top_consider": plays.loc[plays["recommendation"] == "consider"].head(10).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("UPCOMING MARKET PLAY SELECTOR")
    print("=" * 90)
    print(f"Slate:  {slate_path}")
    print(f"Rows:   {len(plays)}")
    print(f"Saved:  {args.csv_out}")
    print(f"JSON:   {args.json_out}")
    print("Recommendation counts:")
    for label, count in plays["recommendation"].value_counts().items():
        print(f"  {label}: {count}")

    show_cols = [
        "player",
        "target",
        "direction",
        "prediction",
        "market_line",
        "edge",
        "gap_percentile",
        "expected_win_rate",
        "confidence_score",
        "recommendation",
    ]
    print("\nTop plays:")
    print(plays[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
