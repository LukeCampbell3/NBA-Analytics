#!/usr/bin/env python3
"""
Build an upcoming NBA market slate by pairing future market lines with the current
production predictor.

This intentionally does not merge future market rows into historical training data.
Instead it:
- loads a normalized market snapshot (wide format)
- finds each player's processed history
- runs inference on the history only
- writes a slate table with prediction, market, and model-vs-market edge columns
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from structured_stack_inference import StructuredStackInference  # noqa: E402


DATA_DIR = REPO_ROOT / "Data-Proc"
MODEL_DIR = REPO_ROOT / "model"
DEFAULT_MARKET_WIDE = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba" / "latest_player_props_wide.parquet"
TARGETS = ["PTS", "TRB", "AST"]


def normalize_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def resolve_manifest_path(run_id: str | None, latest: bool) -> Path:
    if run_id:
        return MODEL_DIR / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return MODEL_DIR / "latest_structured_lstm_stack.json"
    return MODEL_DIR / "production_structured_lstm_stack.json"


def load_market_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market snapshot not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "Player" not in df.columns:
        raise ValueError("Market snapshot must include a Player column")
    if "Market_Date" not in df.columns:
        raise ValueError("Market snapshot must include a Market_Date column")
    df = df.copy()
    df["Player"] = df["Player"].astype(str).map(normalize_name)
    df["Market_Date"] = pd.to_datetime(df["Market_Date"], errors="coerce")
    return df


def infer_player_csv(player_name: str, season: int) -> Path | None:
    candidate = DATA_DIR / player_name / f"{season}_processed_processed.csv"
    return candidate if candidate.exists() else None


def build_records(
    predictor: StructuredStackInference,
    market_df: pd.DataFrame,
    season: int,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []

    for _, market_row in market_df.iterrows():
        player = str(market_row["Player"])
        csv_path = infer_player_csv(player, season)
        if csv_path is None:
            skipped.append({"player": player, "reason": f"missing processed csv for season {season}"})
            continue

        history_df = pd.read_csv(csv_path)
        if history_df.empty:
            skipped.append({"player": player, "reason": "empty processed csv"})
            continue

        if "Date" in history_df.columns:
            history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
            history_df = history_df.loc[history_df["Date"].notna()].copy()
            history_df = history_df.loc[history_df["Date"] < market_row["Market_Date"]].copy()
        if len(history_df) < predictor.seq_len:
            skipped.append({"player": player, "reason": f"insufficient history rows ({len(history_df)})"})
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            explanation = predictor.predict(history_df, assume_prepared=True)

        latest_row = history_df.iloc[-1]
        record = {
            "player": player,
            "market_date": str(market_row["Market_Date"].date()) if pd.notna(market_row["Market_Date"]) else None,
            "history_rows": int(len(history_df)),
            "last_history_date": str(pd.to_datetime(latest_row["Date"]).date()) if "Date" in latest_row.index and pd.notna(latest_row["Date"]) else None,
            "csv": str(csv_path),
            "belief_uncertainty": float(explanation["latent_environment"].get("belief_uncertainty", 0.0)),
            "feasibility": float(explanation["latent_environment"].get("feasibility", 0.0)),
            "role_shift_risk": float(explanation["latent_environment"].get("role_shift_risk", 0.0)),
            "volatility_regime_risk": float(explanation["latent_environment"].get("volatility_regime_risk", 0.0)),
            "context_pressure_risk": float(explanation["latent_environment"].get("context_pressure_risk", 0.0)),
            "fallback_blend": float(explanation.get("data_quality", {}).get("fallback_blend", 0.0)),
            "fallback_reasons": ",".join(explanation.get("data_quality", {}).get("fallback_reasons", [])),
        }
        for target in TARGETS:
            pred_value = float(explanation["predicted"][target])
            baseline_value = float(explanation["baseline"][target])
            market_value = market_row.get(f"Market_{target}", np.nan)
            market_value = float(market_value) if pd.notna(market_value) else np.nan
            record[f"pred_{target}"] = pred_value
            record[f"baseline_{target}"] = baseline_value
            record[f"market_{target}"] = market_value
            record[f"edge_{target}"] = pred_value - market_value if pd.notna(market_value) else np.nan
            record[f"baseline_edge_{target}"] = baseline_value - market_value if pd.notna(market_value) else np.nan
            record[f"{target}_uncertainty_sigma"] = float(explanation["target_factors"][target].get("uncertainty_sigma", 0.0))
            record[f"{target}_spike_probability"] = float(explanation["target_factors"][target].get("spike_probability", 0.0))
            record[f"market_books_{target}"] = float(market_row.get(f"Market_{target}_books", np.nan)) if pd.notna(market_row.get(f"Market_{target}_books", np.nan)) else np.nan
        records.append(record)

    return records, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an upcoming market slate with model-vs-market edges.")
    parser.add_argument("--season", type=int, required=True, help="Season end year, e.g. 2026 for 2025-26.")
    parser.add_argument("--market-wide-path", type=Path, default=DEFAULT_MARKET_WIDE, help="Normalized wide market snapshot.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.csv", help="Output CSV path.")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "model" / "analysis" / "upcoming_market_slate.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_manifest_path(args.run_id, args.latest)
    predictor = StructuredStackInference(model_dir=str(MODEL_DIR), manifest_path=manifest_path)
    market_df = load_market_wide(args.market_wide_path)
    records, skipped = build_records(predictor, market_df, args.season)

    if not records:
        raise RuntimeError(f"No upcoming slate rows built. Skipped={len(skipped)} sample={skipped[:5]}")

    results_df = pd.DataFrame.from_records(records).sort_values(["market_date", "player"]).reset_index(drop=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.csv_out, index=False)
    payload = {
        "manifest_path": str(manifest_path),
        "run_id": predictor.metadata.get("run_id"),
        "market_snapshot": str(args.market_wide_path),
        "season": args.season,
        "rows": int(len(results_df)),
        "skipped": skipped,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("UPCOMING MARKET SLATE BUILT")
    print("=" * 80)
    print(f"Rows:     {len(results_df)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"CSV:      {args.csv_out}")
    print(f"JSON:     {args.json_out}")
    print("\nSample:")
    sample_cols = [
        "player",
        "market_date",
        "pred_PTS",
        "market_PTS",
        "edge_PTS",
        "pred_TRB",
        "market_TRB",
        "edge_TRB",
        "pred_AST",
        "market_AST",
        "edge_AST",
    ]
    print(results_df[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
