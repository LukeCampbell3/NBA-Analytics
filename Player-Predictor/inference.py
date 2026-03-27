#!/usr/bin/env python3
"""
Structured-stack inference CLI for a single player history file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from market_validation import backtest_history, print_validation_summary, save_validation_outputs, summarize_validation_records
from structured_stack_inference import StructuredStackInference


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


def resolve_manifest_path(model_dir: Path, run_id: str | None, latest: bool, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if run_id:
        return model_dir / "runs" / run_id / "lstm_v7_metadata.json"
    if latest:
        return model_dir / "latest_structured_lstm_stack.json"
    return model_dir / "production_structured_lstm_stack.json"


def resolve_csv_path(csv_path: Path | None, player: str | None, season: int) -> Path:
    if csv_path is not None:
        return csv_path
    if not player:
        raise ValueError("Provide either --csv or --player")
    return REPO_ROOT / "Data-Proc" / normalize_name(player) / f"{int(season)}_processed_processed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured-stack inference for one player history CSV.")
    parser.add_argument("--csv", type=Path, default=None, help="Explicit processed player CSV path.")
    parser.add_argument("--player", type=str, default=None, help="Player name (used with --season) to resolve Data-Proc CSV.")
    parser.add_argument("--season", type=int, default=2026, help="Season end year for --player resolution.")
    parser.add_argument("--rows", type=int, default=30, help="How many recent rows to feed into inference.")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "model", help="Model directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit manifest path.")
    parser.add_argument("--run-id", type=str, default=None, help="Specific immutable run id.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production.")
    parser.add_argument("--debug", action="store_true", help="Include debug payload in output.")
    parser.add_argument("--validate-history", action="store_true", help="Backtest historical rows in this CSV against actual stats and market lines.")
    parser.add_argument("--max-validation-rows", type=int, default=0, help="Optional limit for the most recent validation rows.")
    parser.add_argument("--validation-csv-out", type=Path, default=None, help="Optional row-level validation CSV output.")
    parser.add_argument("--validation-json-out", type=Path, default=None, help="Optional validation summary JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_csv_path(args.csv, args.player, args.season)
    if not csv_path.exists():
        raise FileNotFoundError(f"History CSV not found: {csv_path}")

    model_dir = args.model_dir.resolve()
    manifest_path = resolve_manifest_path(model_dir, args.run_id, args.latest, args.manifest)
    predictor = StructuredStackInference(model_dir=str(model_dir), manifest_path=manifest_path)

    history_df = pd.read_csv(csv_path)
    if history_df.empty:
        raise RuntimeError(f"History CSV is empty: {csv_path}")

    if args.validate_history:
        rows, failures = backtest_history(
            predictor,
            history_df,
            csv_path=csv_path,
            player_name=args.player or (str(history_df["Player"].iloc[0]) if "Player" in history_df.columns else csv_path.parent.name),
            min_history_rows=int(getattr(predictor, "seq_len", 10)),
            max_predictions=args.max_validation_rows if args.max_validation_rows > 0 else None,
        )
        rows_df = pd.DataFrame.from_records(rows)
        summary = summarize_validation_records(rows_df, failures=failures)
        print_validation_summary(summary)
        if not rows_df.empty:
            preview_cols = [
                "date",
                "actual_PTS",
                "pred_PTS",
                "market_PTS",
                "directional_correct_PTS",
                "actual_TRB",
                "pred_TRB",
                "market_TRB",
                "directional_correct_TRB",
                "actual_AST",
                "pred_AST",
                "market_AST",
                "directional_correct_AST",
            ]
            preview_cols = [col for col in preview_cols if col in rows_df.columns]
            print("\nRecent rows:")
            print(rows_df[preview_cols].tail(10).to_string(index=False))
        save_validation_outputs(rows_df, summary, args.validation_csv_out, args.validation_json_out)
        return

    explain = predictor.predict(history_df.tail(int(args.rows)), assume_prepared=True, return_debug=bool(args.debug))

    print("\n" + "=" * 80)
    print("STRUCTURED STACK INFERENCE")
    print("=" * 80)
    print(f"CSV:      {csv_path}")
    print(f"Run id:   {predictor.metadata.get('run_id')}")
    print("\nPredicted:")
    for target, value in explain.get("predicted", {}).items():
        print(f"  {target}: {float(value):.2f}")

    print("\nBaseline:")
    for target, value in explain.get("baseline", {}).items():
        print(f"  {target}: {float(value):.2f}")

    print("\nData quality:")
    quality = explain.get("data_quality", {})
    print(f"  schema_repaired: {quality.get('schema_repaired')}")
    print(f"  used_default_ids: {quality.get('used_default_ids')}")
    print(f"  fallback_blend: {quality.get('fallback_blend')}")
    print(f"  fallback_reasons: {quality.get('fallback_reasons')}")

    print("\nJSON:")
    print(json.dumps(explain, indent=2))


if __name__ == "__main__":
    main()
