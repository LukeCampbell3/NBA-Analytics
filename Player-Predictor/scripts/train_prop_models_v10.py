#!/usr/bin/env python3
"""
Train NBA v10 prop probability artifacts.

v10 starts from v9 training rows, then adds:
  - market-residual probability branch
  - direct line-crossing classifier
  - Brier-optimized probability blender
  - predicted Brier-risk gate feature
  - shrinkage side-prior artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
sys.path.insert(0, str(TRAINING_DIR))

from nba_v10_probability_stack import add_v10_features, fit_v10_probability_stack


def _load_v9_rows(v9_manifest: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads(v9_manifest.read_text(encoding="utf-8"))
    output = Path(manifest["output"])
    if str(output).startswith("/workspace/"):
        output = ROOT.parent / str(output).replace("/workspace/", "", 1)
    if not output.is_absolute():
        output = (ROOT.parent / output).resolve()
    rows_path = output / "data" / "prop_training_rows.csv"
    rows = pd.read_csv(rows_path)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date"]).copy()
    return manifest, rows


def _brier(probs, y) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(probs, y, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=float).clip(0.001, 0.999)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = (p >= bins[idx]) & (p < bins[idx + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _write_side_prior_summary(rows: pd.DataFrame, output_dir: Path) -> dict:
    side_dir = output_dir / "side_priors"
    side_dir.mkdir(parents=True, exist_ok=True)
    featured = add_v10_features(rows)
    summary = {
        "market_side": (
            featured.groupby("market")
            .agg(n=("result_over", "size"), raw_over_rate=("result_over", "mean"), shrunk_over_rate=("market_side_prior_over", "mean"))
            .reset_index()
            .to_dict("records")
        ),
        "market_line": (
            featured.groupby(["market", "line_bucket"])
            .agg(n=("result_over", "size"), raw_over_rate=("result_over", "mean"), shrunk_over_rate=("market_line_over_rate_prior", "mean"))
            .reset_index()
            .to_dict("records")
        ),
        "shrink_k": 300,
    }
    (side_dir / "side_prior_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA v10 prop probability stack")
    parser.add_argument("--v9-manifest", type=Path, default=ROOT / "model" / "props" / "v9" / "manifest.json")
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "model" / "props" / "v10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v9_manifest, rows = _load_v9_rows(args.v9_manifest)
    if args.train_start:
        rows = rows[rows["date"] >= pd.Timestamp(args.train_start)]
    if args.train_end:
        rows = rows[rows["date"] <= pd.Timestamp(args.train_end)]
    if len(rows) < 500:
        raise ValueError("v10 requires at least 500 training rows")

    args.output.mkdir(parents=True, exist_ok=True)
    stack = fit_v10_probability_stack(rows)
    stack.save(args.output / "probability_stack" / "v10_probability_stack.pkl")
    side_summary = _write_side_prior_summary(rows, args.output)

    scored = stack.predict_components(rows)
    diagnostics = {
        "training_rows": int(len(scored)),
        "brier_distribution": _brier(scored["p_over_raw"], scored["result_over"]),
        "brier_v10_raw": _brier(scored["p_v10_raw"], scored["result_over"]),
        "ece_distribution": _ece(scored["p_over_raw"], scored["result_over"]),
        "ece_v10_raw": _ece(scored["p_v10_raw"], scored["result_over"]),
        "p_v10_distribution": scored["p_v10_raw"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "brier_risk_distribution": scored["brier_risk"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
    }
    (args.output / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )

    data_dir = args.output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(data_dir / "prop_training_rows_v10_scored.csv", index=False)

    manifest = {
        "model_version": "prop_engine_v10",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_v9_manifest": str(args.v9_manifest),
        "output": str(args.output),
        "rows": int(len(rows)),
        "players": int(rows["player"].nunique()),
        "date_min": str(rows["date"].min().date()),
        "date_max": str(rows["date"].max().date()),
        "artifacts": {
            "probability_stack": "probability_stack/v10_probability_stack.pkl",
            "side_priors": "side_priors/side_prior_summary.json",
            "training_diagnostics": "training_diagnostics.json",
        },
        "summaries": {
            "diagnostics": diagnostics,
            "side_prior_markets": side_summary["market_side"],
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest["summaries"]["diagnostics"], indent=2, default=str))
    print(f"\nWrote v10 prop artifacts to {args.output}")


if __name__ == "__main__":
    main()
