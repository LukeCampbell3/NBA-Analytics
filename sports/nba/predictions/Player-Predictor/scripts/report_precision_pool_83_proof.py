#!/usr/bin/env python3
"""Build a statistical proof report for the NBA precision-pool hit-rate target."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report whether replay/live rows statistically support an 83% precision-pool hit rate.")
    parser.add_argument("--rows-csv", type=Path, required=True, help="Row-level replay or live-monitor CSV with resolved outcomes.")
    parser.add_argument("--mode", type=str, default="precision_pool", help="Mode to evaluate when a mode column exists.")
    parser.add_argument("--mode-col", type=str, default="mode")
    parser.add_argument("--run-date-col", type=str, default="run_date")
    parser.add_argument("--result-col", type=str, default="result")
    parser.add_argument("--prob-col", type=str, default="precision_pool_prob")
    parser.add_argument("--target-hit-rate", type=float, default=0.83)
    parser.add_argument("--confidence", type=float, default=0.95, help="One-sided Wilson lower-bound confidence.")
    parser.add_argument("--min-resolved-plays", type=int, default=30)
    parser.add_argument("--min-resolved-days", type=int, default=5)
    parser.add_argument("--short-window-days", type=int, default=7)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("sports/validation/precision_pool_83_proof_report.json"),
        help="Output JSON report.",
    )
    parser.add_argument("--out-daily-csv", type=Path, default=None, help="Optional daily proof CSV.")
    return parser.parse_args()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def wilson_lower_bound(wins: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return float("nan")
    p_hat = float(wins) / float(total)
    z = NormalDist().inv_cdf(float(np.clip(confidence, 0.50, 0.9999)))
    denom = 1.0 + (z * z / total)
    centre = p_hat + (z * z / (2.0 * total))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z * z / (4.0 * total * total)))
    return float(max(0.0, (centre - margin) / denom))


def min_wins_for_wilson_target(total: int, target: float, confidence: float) -> int | None:
    if total <= 0:
        return None
    for wins in range(0, total + 1):
        if wilson_lower_bound(wins, total, confidence) >= float(target):
            return int(wins)
    return None


def _status(hit_rate: float, lower_bound: float, resolved: int, days: int, args: argparse.Namespace) -> str:
    if resolved < int(args.min_resolved_plays) or days < int(args.min_resolved_days):
        return "insufficient_sample"
    if lower_bound >= float(args.target_hit_rate):
        return "proven_at_confidence"
    if hit_rate >= float(args.target_hit_rate):
        return "observed_target_met_not_statistically_proven"
    return "target_not_met"


def _summarize(df: pd.DataFrame, args: argparse.Namespace, label: str) -> dict[str, Any]:
    resolved = df.loc[df["__result__"].isin(["win", "loss"])].copy()
    total = int(len(resolved))
    wins = int((resolved["__result__"] == "win").sum())
    losses = int((resolved["__result__"] == "loss").sum())
    days = int(resolved["__run_date__"].nunique()) if "__run_date__" in resolved.columns else 0
    hit_rate = float(wins / total) if total > 0 else float("nan")
    lower = wilson_lower_bound(wins, total, confidence=float(args.confidence))
    required_wins = min_wins_for_wilson_target(total, target=float(args.target_hit_rate), confidence=float(args.confidence))
    prob_col = "__prob__"
    mean_prob = float(pd.to_numeric(resolved[prob_col], errors="coerce").mean()) if prob_col in resolved.columns and total else float("nan")
    return {
        "label": label,
        "status": _status(hit_rate, lower, total, days, args),
        "target_hit_rate": float(args.target_hit_rate),
        "confidence": float(args.confidence),
        "resolved_plays": total,
        "resolved_days": days,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit_rate,
        "wilson_lower_bound": lower,
        "mean_probability": mean_prob,
        "calibration_gap_pp": float((hit_rate - mean_prob) * 100.0) if total and mean_prob == mean_prob else float("nan"),
        "required_wins_at_current_sample": required_wins,
        "additional_wins_needed_at_current_sample": int(max(0, int(required_wins) - wins)) if required_wins is not None else None,
        "min_resolved_plays": int(args.min_resolved_plays),
        "min_resolved_days": int(args.min_resolved_days),
    }


def _prepare_rows(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Rows CSV not found: {path}")
    df = pd.read_csv(path)
    if args.mode_col in df.columns:
        df = df.loc[df[args.mode_col].astype(str).str.strip().str.lower() == str(args.mode).strip().lower()].copy()
    required = [args.run_date_col, args.result_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise RuntimeError("No rows available after applying mode filter.")

    work = df.copy()
    token_dates = pd.to_datetime(work[args.run_date_col].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    generic_dates = pd.to_datetime(work[args.run_date_col], errors="coerce")
    work["__run_date__"] = token_dates.fillna(generic_dates)
    work["__result__"] = work[args.result_col].astype(str).str.lower().str.strip()
    if args.prob_col in work.columns:
        work["__prob__"] = pd.to_numeric(work[args.prob_col], errors="coerce")
    elif "expected_win_rate" in work.columns:
        work["__prob__"] = pd.to_numeric(work["expected_win_rate"], errors="coerce")
    else:
        work["__prob__"] = np.nan
    return work


def main() -> None:
    args = parse_args()
    rows_csv = args.rows_csv.resolve()
    work = _prepare_rows(rows_csv, args)
    resolved = work.loc[work["__result__"].isin(["win", "loss"])].copy()
    if resolved.empty:
        raise RuntimeError("No resolved win/loss rows available.")

    daily = (
        resolved.assign(__win__=resolved["__result__"].eq("win").astype(int))
        .groupby("__run_date__", dropna=False)
        .agg(resolved=("__win__", "size"), wins=("__win__", "sum"), mean_probability=("__prob__", "mean"))
        .reset_index()
        .sort_values("__run_date__")
    )
    daily["losses"] = daily["resolved"] - daily["wins"]
    daily["hit_rate"] = daily["wins"] / daily["resolved"]
    daily["wilson_lower_bound"] = [
        wilson_lower_bound(int(row.wins), int(row.resolved), confidence=float(args.confidence))
        for row in daily.itertuples(index=False)
    ]
    daily["run_date"] = pd.to_datetime(daily["__run_date__"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = daily.drop(columns=["__run_date__"])

    latest_date = resolved["__run_date__"].max()
    short_cutoff = latest_date - pd.Timedelta(days=int(max(1, args.short_window_days)) - 1) if pd.notna(latest_date) else pd.NaT
    short = resolved.loc[resolved["__run_date__"] >= short_cutoff].copy() if pd.notna(short_cutoff) else resolved.iloc[0:0].copy()
    overall_summary = _summarize(resolved, args, "overall")
    short_summary = _summarize(short, args, f"latest_{int(max(1, args.short_window_days))}_days")

    payload = {
        "proof_type": "historical_or_live_rows_statistical_monitor",
        "rows_csv": str(rows_csv),
        "mode": str(args.mode),
        "target_hit_rate": float(args.target_hit_rate),
        "confidence": float(args.confidence),
        "overall": overall_summary,
        "short_window": short_summary,
        "proof_passed": bool(overall_summary["status"] == "proven_at_confidence" and short_summary["status"] in {"proven_at_confidence", "insufficient_sample"}),
        "interpretation": (
            "83% is statistically proven only when wilson_lower_bound >= target_hit_rate. "
            "A raw hit rate above 83% with a lower bound below 83% is promising, but not proof."
        ),
    }

    out_json = args.out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_daily = args.out_daily_csv.resolve() if args.out_daily_csv else out_json.with_name(f"{out_json.stem}_daily.csv")
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_daily, index=False)

    print(f"Proof JSON: {out_json}")
    print(f"Daily CSV:  {out_daily}")
    print(json.dumps({"overall": overall_summary, "short_window": short_summary, "proof_passed": payload["proof_passed"]}, indent=2))


if __name__ == "__main__":
    main()
