#!/usr/bin/env python3
"""
Sweep rejector score floors and downstream board gates on saved selector snapshots.

This is a fast research tool for finding a practical tradeoff between:
- hit rate
- number of graded decisions
- total board size

It uses a source historical_validation tag that already contains selector CSVs with
accept_reject_score values and grades candidate boards against the corresponding
reconstructed snapshots.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_engine.policy_tuning import build_default_shadow_strategies
from post_process_market_plays import compute_final_board
from validate_historical_daily_runs import VALIDATION_ROOT, grade_board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep selective rejector settings on saved historical selector snapshots.")
    parser.add_argument("--source-tag", type=str, required=True, help="historical_validation tag containing selector snapshots with accept_reject_score.")
    parser.add_argument("--dates", nargs="+", required=True, help="Dates in YYYYMMDD format.")
    parser.add_argument("--base-profile", type=str, default="production_selective_rejector", help="Base policy profile used for fixed board settings.")
    parser.add_argument("--score-floors", nargs="+", type=float, required=True, help="Rejector score floors to evaluate.")
    parser.add_argument("--min-final-confidences", nargs="+", type=float, required=True, help="Final confidence gates to evaluate.")
    parser.add_argument("--max-total-plays", nargs="+", type=int, required=True, help="Overall board caps to evaluate.")
    parser.add_argument("--out-dir", type=Path, default=VALIDATION_ROOT / "selective_rejector_balance_sweep", help="Directory for JSON/CSV outputs.")
    return parser.parse_args()


def _load_snapshot_map(source_tag: str) -> dict[str, Path]:
    summary_path = VALIDATION_ROOT / source_tag / "historical_daily_validation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation summary not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for item in payload.get("dates", []):
        stamp = str(item.get("stamp", ""))
        snapshot = item.get("snapshot")
        if stamp and snapshot:
            out[stamp] = Path(str(snapshot))
    return out


def _selector_path(source_tag: str, stamp: str) -> Path:
    path = VALIDATION_ROOT / source_tag / stamp / f"{source_tag}_upcoming_market_play_selector_{stamp}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selector snapshot not found: {path}")
    return path


def _policy_map() -> dict[str, dict]:
    return {config.name: config.to_dict() for config in build_default_shadow_strategies()}


def _grade_board_for_snapshot(board: pd.DataFrame, snapshot_path: Path) -> tuple[pd.DataFrame, dict]:
    with tempfile.NamedTemporaryFile(prefix="rejector_sweep_", suffix=".csv", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        board.to_csv(temp_path, index=False)
        return grade_board(temp_path, snapshot_path=snapshot_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _evaluate_config(
    *,
    source_tag: str,
    stamps: list[str],
    snapshot_map: dict[str, Path],
    base_policy: dict,
    score_floor: float,
    min_final_confidence: float,
    max_total_plays: int,
) -> dict:
    totals = {
        "rows": 0,
        "graded_rows": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "missing": 0,
        "final_rows": 0,
        "expected_profit_fraction": 0.0,
    }
    per_date: dict[str, dict] = {}

    for stamp in stamps:
        selector = pd.read_csv(_selector_path(source_tag, stamp))
        if "accept_reject_score" not in selector.columns:
            raise ValueError(f"Selector snapshot is missing accept_reject_score: {stamp}")
        selector = selector.loc[
            pd.to_numeric(selector["accept_reject_score"], errors="coerce").fillna(-1.0) >= float(score_floor)
        ].copy()
        board = compute_final_board(
            selector,
            american_odds=base_policy["american_odds"],
            min_ev=base_policy["min_ev"],
            min_final_confidence=min_final_confidence,
            min_recommendation=base_policy["min_recommendation"],
            ranking_mode=base_policy["ranking_mode"],
            max_plays_per_player=base_policy["max_plays_per_player"],
            max_plays_per_target=base_policy["max_plays_per_target"],
            max_total_plays=max_total_plays,
            max_target_plays={
                "PTS": base_policy["max_pts_plays"],
                "TRB": base_policy["max_trb_plays"],
                "AST": base_policy["max_ast_plays"],
            },
            non_pts_min_gap_percentile=base_policy["non_pts_min_gap_percentile"],
            edge_adjust_k=base_policy["edge_adjust_k"],
            min_bet_win_rate=base_policy["min_bet_win_rate"],
            medium_bet_win_rate=base_policy["medium_bet_win_rate"],
            full_bet_win_rate=base_policy["full_bet_win_rate"],
            medium_tier_percentile=base_policy["medium_tier_percentile"],
            strong_tier_percentile=base_policy["strong_tier_percentile"],
            elite_tier_percentile=base_policy["elite_tier_percentile"],
            small_bet_fraction=base_policy["small_bet_fraction"],
            medium_bet_fraction=base_policy["medium_bet_fraction"],
            full_bet_fraction=base_policy["full_bet_fraction"],
            max_bet_fraction=base_policy["max_bet_fraction"],
            max_total_bet_fraction=base_policy["max_total_bet_fraction"],
        )
        graded, summary = _grade_board_for_snapshot(board, snapshot_path=snapshot_map[stamp])
        expected_profit_fraction = (
            float(pd.to_numeric(board.get("expected_profit_fraction"), errors="coerce").fillna(0.0).sum())
            if not board.empty
            else 0.0
        )

        totals["rows"] += int(summary["rows"])
        totals["graded_rows"] += int(summary["graded_rows"])
        totals["wins"] += int(summary["wins"])
        totals["losses"] += int(summary["losses"])
        totals["pushes"] += int(summary["pushes"])
        totals["missing"] += int(len(graded) - int(summary["graded_rows"]))
        totals["final_rows"] += int(len(board))
        totals["expected_profit_fraction"] += expected_profit_fraction

        per_date[stamp] = {
            "final_rows": int(len(board)),
            "graded_rows": int(summary["graded_rows"]),
            "wins": int(summary["wins"]),
            "losses": int(summary["losses"]),
            "pushes": int(summary["pushes"]),
            "missing": int(len(graded) - int(summary["graded_rows"])),
            "win_rate": summary["win_rate"],
            "expected_profit_fraction": expected_profit_fraction,
        }

    decisions = int(totals["wins"] + totals["losses"])
    win_rate = (float(totals["wins"]) / decisions) if decisions else None
    avg_expected_profit_fraction = (float(totals["expected_profit_fraction"]) / totals["final_rows"]) if totals["final_rows"] else None
    score = ((win_rate or 0.0) - 0.52) * decisions

    return {
        "score_floor": float(score_floor),
        "min_final_confidence": float(min_final_confidence),
        "max_total_plays": int(max_total_plays),
        "wins": int(totals["wins"]),
        "losses": int(totals["losses"]),
        "pushes": int(totals["pushes"]),
        "missing": int(totals["missing"]),
        "decisions": decisions,
        "graded_rows": int(totals["graded_rows"]),
        "final_rows": int(totals["final_rows"]),
        "win_rate": win_rate,
        "expected_profit_fraction": float(totals["expected_profit_fraction"]),
        "avg_expected_profit_fraction": avg_expected_profit_fraction,
        "score": float(score),
        "per_date": per_date,
    }


def _pareto_frontier(results: list[dict]) -> list[dict]:
    ordered = sorted(
        results,
        key=lambda item: (
            -(item.get("decisions") or 0),
            -(item.get("win_rate") or 0.0),
            -(item.get("expected_profit_fraction") or 0.0),
        ),
    )
    frontier: list[dict] = []
    best_win_rate = -1.0
    for item in ordered:
        win_rate = float(item.get("win_rate") or 0.0)
        if win_rate > best_win_rate:
            frontier.append(item)
            best_win_rate = win_rate
    return sorted(frontier, key=lambda item: ((item.get("decisions") or 0), (item.get("win_rate") or 0.0)))


def main() -> None:
    args = parse_args()
    policy_map = _policy_map()
    if args.base_profile not in policy_map:
        raise ValueError(f"Unknown base profile: {args.base_profile}")
    base_policy = policy_map[args.base_profile]
    snapshot_map = _load_snapshot_map(args.source_tag)
    stamps = [str(value) for value in args.dates]

    results: list[dict] = []
    for score_floor in args.score_floors:
        for min_final_confidence in args.min_final_confidences:
            for max_total_plays in args.max_total_plays:
                results.append(
                    _evaluate_config(
                        source_tag=args.source_tag,
                        stamps=stamps,
                        snapshot_map=snapshot_map,
                        base_policy=base_policy,
                        score_floor=float(score_floor),
                        min_final_confidence=float(min_final_confidence),
                        max_total_plays=int(max_total_plays),
                    )
                )

    results = sorted(
        results,
        key=lambda item: (
            -(item.get("score") or 0.0),
            -(item.get("win_rate") or 0.0),
            -(item.get("decisions") or 0),
            -(item.get("expected_profit_fraction") or 0.0),
        ),
    )
    frontier = _pareto_frontier(results)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "sweep_summary.json"
    results_csv = out_dir / "sweep_results.csv"
    frontier_json = out_dir / "sweep_pareto_frontier.json"

    pd.DataFrame(
        [
            {
                key: value
                for key, value in item.items()
                if key != "per_date"
            }
            for item in results
        ]
    ).to_csv(results_csv, index=False)
    results_json.write_text(
        json.dumps(
            {
                "source_tag": args.source_tag,
                "base_profile": args.base_profile,
                "dates": stamps,
                "top_result": results[0] if results else None,
                "pareto_frontier": frontier,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    frontier_json.write_text(json.dumps(frontier, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("SELECTIVE REJECTOR BALANCE SWEEP")
    print("=" * 90)
    if results:
        best = results[0]
        print(
            "Best: "
            f"floor={best['score_floor']:.4f}, "
            f"min_conf={best['min_final_confidence']:.3f}, "
            f"max_total={best['max_total_plays']}, "
            f"wins={best['wins']}, losses={best['losses']}, "
            f"decisions={best['decisions']}, win_rate={best['win_rate']}"
        )
    print(f"CSV:      {results_csv}")
    print(f"JSON:     {results_json}")
    print(f"Frontier: {frontier_json}")


if __name__ == "__main__":
    main()
