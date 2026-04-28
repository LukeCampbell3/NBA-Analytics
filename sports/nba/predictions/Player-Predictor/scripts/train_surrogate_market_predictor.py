#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "inference"))

from surrogate_market_predictor import (  # noqa: E402
    DEFAULT_BUNDLE_PATH,
    DEFAULT_SUMMARY_PATH,
    build_training_frame,
    train_surrogate_models,
    write_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NBA surrogate market predictor bundle.")
    parser.add_argument("--data-proc-root", type=Path, default=REPO_ROOT / "Data-Proc")
    parser.add_argument("--bundle-out", type=Path, default=REPO_ROOT / "model" / DEFAULT_BUNDLE_PATH)
    parser.add_argument("--summary-json-out", type=Path, default=REPO_ROOT / "model" / DEFAULT_SUMMARY_PATH)
    parser.add_argument("--min-history-games", type=int, default=5)
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_df = build_training_frame(args.data_proc_root.resolve(), min_history_games=int(args.min_history_games))
    if training_df.empty:
        raise RuntimeError(f"No training rows were built from {args.data_proc_root}")

    models, numeric_medians, metrics = train_surrogate_models(
        training_df,
        holdout_days=int(args.holdout_days),
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
    )
    write_bundle(
        bundle_path=args.bundle_out.resolve(),
        summary_path=args.summary_json_out.resolve(),
        models=models,
        numeric_medians=numeric_medians,
        target_metrics=metrics["targets"],
        split_summary=metrics["split"],
        min_history_games=int(args.min_history_games),
    )

    print("\n" + "=" * 84)
    print("SURROGATE MARKET PREDICTOR TRAINED")
    print("=" * 84)
    print(f"Training rows: {len(training_df)}")
    print(f"Bundle:        {args.bundle_out.resolve()}")
    print(f"Summary JSON:  {args.summary_json_out.resolve()}")
    for target, payload in metrics["targets"].items():
        print(
            f"{target}: mae={payload['mae']:.4f} "
            f"rmse={payload['rmse']:.4f} "
            f"valid_rows={payload['valid_rows']}"
        )


if __name__ == "__main__":
    main()
