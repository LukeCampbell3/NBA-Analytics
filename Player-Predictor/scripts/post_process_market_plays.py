#!/usr/bin/env python3
"""
Post-process ranked market plays into a final actionable board.

This layer is intentionally separate from selection so we can:
- compute EV from expected win rate
- de-correlate by player
- filter to positive-EV / minimum-quality plays
- produce a tighter, final board for execution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process selected market plays into a final board.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model/analysis/upcoming_market_play_selector.csv"),
        help="Selector CSV from select_market_plays.py",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.csv"),
        help="Output CSV path for the final board.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("model/analysis/final_market_plays.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--american-odds",
        type=int,
        default=-110,
        help="Assumed book odds for EV calculation when actual odds are unavailable.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.0,
        help="Minimum expected value required to keep a play.",
    )
    parser.add_argument(
        "--min-final-confidence",
        type=float,
        default=0.03,
        help="Minimum final confidence required to keep a play.",
    )
    parser.add_argument(
        "--min-recommendation",
        type=str,
        default="consider",
        choices=["consider", "strong", "elite"],
        help="Lowest selector recommendation allowed into the final board.",
    )
    parser.add_argument(
        "--max-plays-per-player",
        type=int,
        default=1,
        help="Maximum number of plays to keep per player after ranking.",
    )
    parser.add_argument(
        "--max-plays-per-target",
        type=int,
        default=0,
        help="Maximum number of final plays to keep per target when target-specific caps are not supplied.",
    )
    parser.add_argument(
        "--max-pts-plays",
        type=int,
        default=6,
        help="Maximum final PTS plays to keep.",
    )
    parser.add_argument(
        "--max-trb-plays",
        type=int,
        default=4,
        help="Maximum final TRB plays to keep.",
    )
    parser.add_argument(
        "--max-ast-plays",
        type=int,
        default=2,
        help="Maximum final AST plays to keep.",
    )
    parser.add_argument(
        "--max-total-plays",
        type=int,
        default=12,
        help="Maximum number of final plays to keep overall.",
    )
    parser.add_argument(
        "--non-pts-min-gap-percentile",
        type=float,
        default=0.90,
        help="Minimum disagreement percentile required for TRB/AST plays.",
    )
    parser.add_argument(
        "--edge-adjust-k",
        type=float,
        default=0.30,
        help="Weight for edge-adjusted EV ranking.",
    )
    return parser.parse_args()


def american_profit_per_unit(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def recommendation_rank(label: str) -> int:
    order = {"elite": 0, "strong": 1, "consider": 2, "pass": 3}
    return order.get(str(label), 3)


def minimum_recommendation_rank(label: str) -> int:
    return {"elite": 0, "strong": 1, "consider": 2}[label]


def compute_final_board(
    plays: pd.DataFrame,
    american_odds: int = -110,
    min_ev: float = 0.0,
    min_final_confidence: float = 0.02,
    min_recommendation: str = "consider",
    max_plays_per_player: int = 1,
    max_plays_per_target: int = 8,
    max_total_plays: int = 20,
    max_target_plays: dict[str, int] | None = None,
    non_pts_min_gap_percentile: float = 0.90,
    edge_adjust_k: float = 0.30,
) -> pd.DataFrame:
    out = plays.copy()
    if out.empty:
        return out

    payout = american_profit_per_unit(american_odds)
    out["expected_win_rate"] = pd.to_numeric(out["expected_win_rate"], errors="coerce")
    out["gap_percentile"] = pd.to_numeric(out["gap_percentile"], errors="coerce").fillna(0.0)
    out["belief_uncertainty"] = pd.to_numeric(out.get("belief_uncertainty"), errors="coerce").fillna(1.0)
    out["feasibility"] = pd.to_numeric(out.get("feasibility"), errors="coerce").fillna(0.0)
    out["abs_edge"] = pd.to_numeric(out.get("abs_edge"), errors="coerce").fillna(0.0)

    out["ev"] = out["expected_win_rate"] * payout - (1.0 - out["expected_win_rate"])
    out["final_confidence"] = out["gap_percentile"] * np.clip(1.0 - out["belief_uncertainty"], 0.0, 1.0) * np.clip(out["feasibility"], 0.0, None)
    out["recommendation_rank"] = out["recommendation"].map(recommendation_rank)
    edge_baseline = out.groupby("target")["abs_edge"].transform(lambda s: s.median() if len(s) else 1.0).replace(0.0, 1.0)
    out["edge_scale"] = (out["abs_edge"] / edge_baseline).clip(lower=0.50, upper=2.50)
    out["ev_adjusted"] = out["ev"] * (1.0 + float(edge_adjust_k) * (out["edge_scale"] - 1.0))

    out = out.loc[out["recommendation_rank"] <= minimum_recommendation_rank(min_recommendation)].copy()
    out = out.loc[out["ev"] >= float(min_ev)].copy()
    out = out.loc[out["final_confidence"] >= float(min_final_confidence)].copy()
    out = out.loc[(out["target"] == "PTS") | (out["gap_percentile"] >= float(non_pts_min_gap_percentile))].copy()
    if out.empty:
        return out

    out = out.sort_values(
        ["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
        ascending=[False, False, False, False],
    )
    out = out.groupby("player", as_index=False, sort=False).head(max_plays_per_player).copy()
    if max_target_plays:
        parts = []
        for target, cap in max_target_plays.items():
            parts.append(out.loc[out["target"] == target].head(int(cap)))
        out = pd.concat(parts, ignore_index=True) if parts else out.iloc[0:0].copy()
    elif max_plays_per_target > 0:
        out = out.groupby("target", as_index=False, sort=False).head(max_plays_per_target).copy()
    if max_total_plays > 0:
        out = out.head(max_total_plays).copy()
    out = out.sort_values(
        ["ev_adjusted", "expected_win_rate", "final_confidence", "abs_edge"],
        ascending=[False, False, False, False],
    ).drop(columns=["recommendation_rank"])
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Selector CSV not found: {csv_path}")

    plays = pd.read_csv(csv_path)
    final_board = compute_final_board(
        plays,
        american_odds=args.american_odds,
        min_ev=args.min_ev,
        min_final_confidence=args.min_final_confidence,
        min_recommendation=args.min_recommendation,
        max_plays_per_player=args.max_plays_per_player,
        max_plays_per_target=args.max_plays_per_target,
        max_total_plays=args.max_total_plays,
        max_target_plays={"PTS": args.max_pts_plays, "TRB": args.max_trb_plays, "AST": args.max_ast_plays},
        non_pts_min_gap_percentile=args.non_pts_min_gap_percentile,
        edge_adjust_k=args.edge_adjust_k,
    )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    final_board.to_csv(args.csv_out, index=False)

    payload = {
        "source_csv": str(csv_path),
        "rows_in": int(len(plays)),
        "rows_out": int(len(final_board)),
        "american_odds": int(args.american_odds),
        "min_ev": float(args.min_ev),
        "min_final_confidence": float(args.min_final_confidence),
        "min_recommendation": args.min_recommendation,
        "max_plays_per_player": int(args.max_plays_per_player),
        "max_plays_per_target": int(args.max_plays_per_target),
        "max_pts_plays": int(args.max_pts_plays),
        "max_trb_plays": int(args.max_trb_plays),
        "max_ast_plays": int(args.max_ast_plays),
        "max_total_plays": int(args.max_total_plays),
        "non_pts_min_gap_percentile": float(args.non_pts_min_gap_percentile),
        "edge_adjust_k": float(args.edge_adjust_k),
        "top_plays": final_board.head(20).to_dict(orient="records"),
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("FINAL MARKET PLAY BOARD")
    print("=" * 90)
    print(f"Input rows:   {len(plays)}")
    print(f"Output rows:  {len(final_board)}")
    print(f"CSV:          {args.csv_out}")
    print(f"JSON:         {args.json_out}")
    if not final_board.empty:
        show_cols = [
            "player",
            "target",
            "direction",
            "prediction",
            "market_line",
            "abs_edge",
            "expected_win_rate",
            "ev",
            "ev_adjusted",
            "final_confidence",
            "recommendation",
        ]
        print("\nTop final plays:")
        print(final_board[show_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
