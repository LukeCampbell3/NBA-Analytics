#!/usr/bin/env python3
"""Build a recent settled selector pool from authentic NFL market closes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_backtest import (  # noqa: E402
    evaluate_market_backtest,
    load_market_archive,
)
from sports.nfl.predictions.market_selector import (  # noqa: E402
    build_learning_frames,
    build_prediction_pool,
    score_probabilities,
    summarize_market_rows,
)
from sports.nfl.predictions.pipeline import load_weekly_stats  # noqa: E402


NFL_ROOT = REPO_ROOT / "sports" / "nfl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--stats",
        type=Path,
        default=NFL_ROOT / "data/raw/player_stats_deployment.parquet",
    )
    parser.add_argument(
        "--selector-artifact",
        type=Path,
        default=NFL_ROOT / "model/nfl_market_selector.joblib",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, low_memory=False)
    market_report, market_rows = evaluate_market_backtest(
        predictions, load_market_archive(args.lines), minimum_edge_yards=0.0
    )
    stats = load_weekly_stats(
        args.stats, start_season=2018, end_season=args.season
    )
    selector = joblib.load(args.selector_artifact)
    frames, _, _ = build_learning_frames(
        stats,
        market_rows,
        latent=pd.DataFrame(columns=["player_id", "season", "week"]),
    )
    parts: list[pd.DataFrame] = []
    architecture: dict[str, str] = {}
    promotion: dict[str, str] = {}
    for target, frame in frames.items():
        model_info = selector["models"][target]
        probabilities = model_info["model"].predict_proba(
            frame[model_info["features"]]
        )[:, 1]
        scored = score_probabilities(
            frame,
            probabilities,
            minimum_side_probability=selector["minimum_side_probability"],
            minimum_no_vig_advantage=selector["minimum_no_vig_advantage"],
        )
        scored["target_promotion_status"] = model_info["promotion_status"]
        parts.append(scored)
        architecture[target] = model_info["architecture"]
        promotion[target] = model_info["promotion_status"]
    pool = build_prediction_pool(
        pd.concat(parts, ignore_index=True),
        evaluation_split=f"recent_{args.season}_out_of_time",
        architecture_by_target=architecture,
        promotion_by_target=promotion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.output, index=False)
    passing = pool.loc[pool["target"].eq("passing")]
    manifest = {
        "schema_version": 1,
        "season": args.season,
        "line_source": "sportsgameodds_consensus_close",
        "line_scope": "explicit provider consensus closes; not named-book execution proof",
        "line_rows": int(len(load_market_archive(args.lines))),
        "matched_market_rows": int(len(market_rows)),
        "eligible_selector_rows": int(len(pool)),
        "eligible_passing_rows": int(len(passing)),
        "season_weeks": int(pool[["season", "week"]].drop_duplicates().shape[0]),
        "passing_result": summarize_market_rows(passing),
        "market_join": market_report["overall"],
        "line_file_sha256": sha256(args.lines),
        "prediction_file_sha256": sha256(args.predictions),
        "output_file_sha256": sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
