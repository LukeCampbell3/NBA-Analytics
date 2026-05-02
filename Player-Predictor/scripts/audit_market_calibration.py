#!/usr/bin/env python3
"""
Audit calibration of graded market-play outputs.

This script evaluates whether:
1. expected_win_rate tracks realized hit rate
2. final_confidence behaves as a useful monotonic decision score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPECTED_BINS = [0.0, 0.55, 0.58, 0.60, 0.62, 0.65, 1.0]
DEFAULT_CONFIDENCE_BINS = [0.0, 0.04, 0.05, 0.07, 0.09, 0.12, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit calibration from graded market-play CSVs.")
    parser.add_argument("--csvs", nargs="+", type=Path, required=True, help="One or more graded CSV files containing expected_win_rate, final_confidence, and result.")
    parser.add_argument("--label", type=str, default="calibration_audit", help="Label to include in outputs.")
    parser.add_argument("--dedupe-play-signature", action="store_true", help="Drop duplicate rows by play_signature before scoring.")
    parser.add_argument("--json-out", type=Path, required=True, help="Output JSON summary path.")
    parser.add_argument("--bucket-csv-out", type=Path, default=None, help="Optional long-form bucket summary CSV path.")
    return parser.parse_args()


def _result_frame(csv_paths: list[Path], dedupe_play_signature: bool) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined.loc[combined["result"].isin(["win", "loss"])].copy()
    combined["win"] = (combined["result"] == "win").astype(int)
    if dedupe_play_signature and "play_signature" in combined.columns:
        combined = combined.drop_duplicates(subset=["play_signature"]).reset_index(drop=True)
    return combined


def _bucket_summary(df: pd.DataFrame, column: str, bins: list[float], labels: list[str], bucket_type: str) -> tuple[list[dict], pd.DataFrame]:
    working = df.copy()
    working["bucket"] = pd.cut(working[column], bins=bins, labels=labels, include_lowest=True)
    summary = (
        working.groupby("bucket", observed=False)
        .agg(
            rows=("win", "size"),
            avg_value=(column, "mean"),
            realized_win_rate=("win", "mean"),
        )
        .reset_index()
    )
    summary["bucket_type"] = bucket_type
    summary["bucket"] = summary["bucket"].astype(str)
    return summary.to_dict(orient="records"), summary


def _overall_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "avg_expected_win_rate": None,
            "avg_final_confidence": None,
            "brier_score": None,
            "calibration_gap": None,
        }
    brier = float(np.mean((pd.to_numeric(df["expected_win_rate"], errors="coerce").fillna(0.5) - df["win"]) ** 2))
    avg_expected = float(pd.to_numeric(df["expected_win_rate"], errors="coerce").mean())
    realized = float(df["win"].mean())
    return {
        "rows": int(len(df)),
        "wins": int(df["win"].sum()),
        "losses": int((1 - df["win"]).sum()),
        "win_rate": realized,
        "avg_expected_win_rate": avg_expected,
        "avg_final_confidence": float(pd.to_numeric(df["final_confidence"], errors="coerce").mean()) if "final_confidence" in df.columns else None,
        "brier_score": brier,
        "calibration_gap": avg_expected - realized,
    }


def main() -> None:
    args = parse_args()
    df = _result_frame([path.resolve() for path in args.csvs], dedupe_play_signature=bool(args.dedupe_play_signature))

    payload = {
        "label": args.label,
        "dedupe_play_signature": bool(args.dedupe_play_signature),
        "source_csvs": [str(path.resolve()) for path in args.csvs],
        "overall": _overall_summary(df),
        "expected_win_rate_buckets": [],
        "final_confidence_buckets": [],
    }

    bucket_frames: list[pd.DataFrame] = []
    if not df.empty and "expected_win_rate" in df.columns:
        expected_labels = ["<=0.55", "0.55-0.58", "0.58-0.60", "0.60-0.62", "0.62-0.65", ">0.65"]
        expected_payload, expected_frame = _bucket_summary(df, "expected_win_rate", DEFAULT_EXPECTED_BINS, expected_labels, "expected_win_rate")
        payload["expected_win_rate_buckets"] = expected_payload
        bucket_frames.append(expected_frame)
    if not df.empty and "final_confidence" in df.columns:
        confidence_labels = ["<=0.04", "0.04-0.05", "0.05-0.07", "0.07-0.09", "0.09-0.12", ">0.12"]
        confidence_payload, confidence_frame = _bucket_summary(df, "final_confidence", DEFAULT_CONFIDENCE_BINS, confidence_labels, "final_confidence")
        payload["final_confidence_buckets"] = confidence_payload
        bucket_frames.append(confidence_frame)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.bucket_csv_out is not None:
        args.bucket_csv_out.parent.mkdir(parents=True, exist_ok=True)
        bucket_df = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()
        bucket_df.to_csv(args.bucket_csv_out, index=False)

    print("\n" + "=" * 90)
    print("MARKET CALIBRATION AUDIT")
    print("=" * 90)
    print(f"Label:    {args.label}")
    print(f"Rows:     {payload['overall']['rows']}")
    print(f"Win rate: {payload['overall']['win_rate']}")
    print(f"Avg EWR:  {payload['overall']['avg_expected_win_rate']}")
    print(f"Brier:    {payload['overall']['brier_score']}")
    print(f"JSON:     {args.json_out}")
    if args.bucket_csv_out is not None:
        print(f"Buckets:  {args.bucket_csv_out}")


if __name__ == "__main__":
    main()
