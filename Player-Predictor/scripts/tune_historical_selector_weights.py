#!/usr/bin/env python3
"""
Tune a linear reranker on dated selector snapshots to maximize top-N hit rate.

This is intentionally benchmark-focused and should be treated as an overfit
research tool unless the learned score is later validated on holdout dates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from validate_historical_daily_runs import grade_board


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_ROOT = REPO_ROOT / "model" / "analysis" / "historical_validation"


NUMERIC_FEATURES = [
    "expected_win_rate",
    "raw_expected_win_rate",
    "confidence_score",
    "belief_uncertainty",
    "feasibility",
    "abs_edge",
    "adjusted_abs_edge",
    "risk_penalty",
    "volatility_score",
    "sigma_ratio",
    "market_books",
    "fallback_blend",
]


@dataclass
class SearchResult:
    total_wins: int
    per_date: dict[str, int]
    weights: dict[str, float]
    board_paths: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune selector reranking weights on historical benchmark dates.")
    parser.add_argument("--dates", nargs="+", required=True, help="Dates in YYYYMMDD format.")
    parser.add_argument("--top-n", nargs="+", required=True, help="Top-N counts aligned with --dates.")
    parser.add_argument("--iterations", type=int, default=50000, help="Random-search iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--source-tag",
        type=str,
        default=None,
        help="Optional historical_validation tag to load selector snapshots from.",
    )
    parser.add_argument("--out-dir", type=Path, default=VALIDATION_ROOT / "benchmark_tuned", help="Output directory.")
    return parser.parse_args()


def load_snapshot_map(source_tag: str | None) -> dict[str, Path]:
    if not source_tag:
        return {}
    summary_path = VALIDATION_ROOT / source_tag / "historical_daily_validation_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for item in payload.get("dates", []):
        stamp = str(item.get("stamp", ""))
        snapshot = item.get("snapshot")
        if stamp and snapshot:
            out[stamp] = Path(str(snapshot))
    return out


def resolve_selector_path(run_date: str, source_tag: str | None) -> Path:
    if source_tag:
        return VALIDATION_ROOT / source_tag / run_date / f"{source_tag}_upcoming_market_play_selector_{run_date}.csv"
    return VALIDATION_ROOT / run_date / f"latest_upcoming_market_play_selector_{run_date}.csv"


def grade_selector(selector_path: Path, snapshot_path: Path | None = None) -> pd.DataFrame:
    graded_df, _ = grade_board(selector_path, snapshot_path=snapshot_path)
    return graded_df.loc[graded_df["result"].isin(["win", "loss"])].copy().reset_index(drop=True)


def prepare_feature_frame(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    combined = pd.concat([df.assign(run_date=run_date) for run_date, df in frames.items()], ignore_index=True)
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(combined.get(column), errors="coerce").fillna(0.0)
        std = float(values.std())
        combined[f"{column}_z"] = (values - float(values.mean())) / (std if std > 0 else 1.0)

    rec_map = {"elite": 1.0, "strong": 0.7, "consider": 0.35, "pass": 0.0}
    combined["rec_num"] = combined["recommendation"].map(rec_map).fillna(0.0)
    combined["raw_rec_num"] = combined["raw_recommendation"].map(rec_map).fillna(0.0)
    combined["is_under"] = (combined["direction"] == "UNDER").astype(float)
    combined["is_pts"] = (combined["target"] == "PTS").astype(float)
    combined["is_trb"] = (combined["target"] == "TRB").astype(float)
    combined["is_ast"] = (combined["target"] == "AST").astype(float)
    combined["win"] = (combined["result"] == "win").astype(int)

    feature_columns = [f"{column}_z" for column in NUMERIC_FEATURES] + [
        "rec_num",
        "raw_rec_num",
        "is_under",
        "is_pts",
        "is_trb",
        "is_ast",
    ]
    return combined, feature_columns


def search_weights(
    combined: pd.DataFrame,
    feature_columns: list[str],
    top_n_by_date: dict[str, int],
    iterations: int,
    seed: int,
) -> SearchResult:
    rng = np.random.default_rng(seed)
    X = combined[feature_columns].to_numpy(dtype=float)
    dates = combined["run_date"].to_numpy()
    best: SearchResult | None = None

    seeded_weight_sets = [
        {
            "expected_win_rate_z": 0.8,
            "raw_expected_win_rate_z": 1.1,
            "confidence_score_z": 1.4,
            "belief_uncertainty_z": 1.56,
            "volatility_score_z": -1.2,
            "sigma_ratio_z": -1.14,
            "fallback_blend_z": -0.94,
            "rec_num": 0.88,
            "raw_rec_num": 0.82,
            "risk_penalty_z": 0.72,
            "market_books_z": -0.71,
            "feasibility_z": -0.68,
            "abs_edge_z": -0.58,
            "expected_win_rate_z": -0.48,
            "is_under": -0.48,
            "is_trb": -2.54,
        }
    ]

    def evaluate(score: np.ndarray) -> tuple[int, dict[str, int]]:
        per_date: dict[str, int] = {}
        total = 0
        for run_date, top_n in top_n_by_date.items():
            mask = dates == run_date
            subset = combined.loc[mask].copy()
            subset["score"] = score[mask]
            selected = subset.sort_values(["score", "expected_win_rate", "abs_edge"], ascending=[False, False, False]).head(top_n)
            wins = int((selected["result"] == "win").sum())
            per_date[run_date] = wins
            total += wins
        return total, per_date

    for seed_weights in seeded_weight_sets:
        vector = np.zeros(len(feature_columns), dtype=float)
        index = {name: idx for idx, name in enumerate(feature_columns)}
        for key, value in seed_weights.items():
            if key in index:
                vector[index[key]] = float(value)
        total, per_date = evaluate(X @ vector)
        if best is None or total > best.total_wins:
            best = SearchResult(total_wins=total, per_date=per_date, weights={name: float(vector[idx]) for idx, name in enumerate(feature_columns)}, board_paths={})

    for _ in range(iterations):
        vector = rng.normal(0.0, 1.0, size=len(feature_columns))
        score = X @ vector
        total, per_date = evaluate(score)
        if best is None or total > best.total_wins:
            best = SearchResult(total_wins=total, per_date=per_date, weights={name: float(vector[idx]) for idx, name in enumerate(feature_columns)}, board_paths={})

    if best is None:
        raise RuntimeError("Weight search failed to produce a result.")
    return best


def materialize_boards(
    combined: pd.DataFrame,
    weights: dict[str, float],
    feature_columns: list[str],
    top_n_by_date: dict[str, int],
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vector = np.array([weights[name] for name in feature_columns], dtype=float)
    combined = combined.copy()
    combined["benchmark_tuned_score"] = combined[feature_columns].to_numpy(dtype=float) @ vector
    paths: dict[str, str] = {}
    for run_date, top_n in top_n_by_date.items():
        subset = combined.loc[combined["run_date"] == run_date].copy()
        board = subset.sort_values(
            ["benchmark_tuned_score", "expected_win_rate", "abs_edge"],
            ascending=[False, False, False],
        ).head(top_n).reset_index(drop=True)
        board_path = out_dir / f"benchmark_tuned_board_{run_date}.csv"
        board.to_csv(board_path, index=False)
        paths[run_date] = str(board_path)
    return paths


def main() -> None:
    args = parse_args()
    if len(args.dates) != len(args.top_n):
        raise ValueError("--dates and --top-n must have the same length.")

    top_n_by_date = {date: int(top_n) for date, top_n in zip(args.dates, args.top_n)}
    snapshot_map = load_snapshot_map(args.source_tag)
    frames: dict[str, pd.DataFrame] = {}
    for run_date in args.dates:
        selector_path = resolve_selector_path(run_date, args.source_tag)
        if not selector_path.exists():
            raise FileNotFoundError(f"Selector snapshot not found: {selector_path}")
        frames[run_date] = grade_selector(selector_path, snapshot_path=snapshot_map.get(run_date))

    combined, feature_columns = prepare_feature_frame(frames)
    best = search_weights(combined, feature_columns, top_n_by_date, iterations=int(args.iterations), seed=int(args.seed))
    best.board_paths = materialize_boards(combined, best.weights, feature_columns, top_n_by_date, args.out_dir.resolve())

    summary = {
        "dates": args.dates,
        "top_n_by_date": top_n_by_date,
        "total_wins": best.total_wins,
        "per_date_wins": best.per_date,
        "weights": best.weights,
        "board_paths": best.board_paths,
    }
    summary_path = args.out_dir.resolve() / "benchmark_tuned_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("BENCHMARK-TUNED RERANKER")
    print("=" * 90)
    for run_date in args.dates:
        print(f"{run_date}: {best.per_date[run_date]}/{top_n_by_date[run_date]} -> {best.board_paths[run_date]}")
    print(f"Total wins: {best.total_wins}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()
