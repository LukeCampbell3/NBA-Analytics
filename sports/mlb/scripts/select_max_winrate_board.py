#!/usr/bin/env python3
"""
Maximum Win-Rate Board Selector

Ultra-strict policy focused on short-term win rate above all else.
Only publishes picks from historically proven >85% win rate buckets
where the model also projects >90% raw probability.

Target: pool size >= 5, miss rate <= 1 per board.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
CALIBRATION_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"
STANDARD_MARKET_LINES = {"H": 0.5, "TB": 1.5, "R": 0.5, "HR": 0.5}


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)

# Proven high-win-rate buckets from historical calibration data
# Only include buckets with >85% historical win rate AND >500 graded samples
ELITE_BUCKETS = {
    # bucket_key: (historical_win_rate, graded_samples)
    "TB|UNDER|3.5": (0.950, 580),
    "TB|UNDER|2.5": (0.934, 2452),
    "TB|UNDER|1.5": (0.866, 6359),
    "TB|UNDER|1.0": (0.891, 119),  # smaller sample but very high
    "K|UNDER|2.5": (0.899, 8138),
    "R|UNDER|1.5": (0.920, 678),
    "H|UNDER|2.5": (0.927, 55),  # small sample
}

# Broader high-win-rate buckets (>79% with >2000 samples)
STRONG_BUCKETS = {
    **ELITE_BUCKETS,
    "TB|UNDER|5.5": (0.843, 16022),  # target-direction level, line-specific data sparse
    "TB|UNDER|6.5": (0.843, 16022),  # same
    "TB|UNDER|4.5": (0.843, 16022),  # same
    "TB|UNDER|8.5": (0.843, 16022),  # same
    "R|UNDER|0.5": (0.786, 16809),
    "R|UNDER|2.5": (0.792, 17500),  # target-direction level
    "H|UNDER|1.5": (0.790, 4078),
    "HR|UNDER|0.5": (0.830, 5000),  # HR unders — most hitters don't HR
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _valid_american_price(value: Any) -> float | None:
    price = _safe_float(value, default=float("nan"))
    if not math.isfinite(price) or abs(price) < 100.0 or abs(price - round(price)) > 1e-6:
        return None
    return price


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    total = 0.0
    for i in range(k + 1):
        if lam == 0.0:
            total += 1.0 if i == 0 else 0.0
        else:
            log_p = (-lam) + (i * math.log(lam)) - math.lgamma(i + 1)
            total += math.exp(log_p)
    return min(1.0, total)


def compute_under_probability(prediction: float, line: float) -> float:
    """Probability of going UNDER the line based on Poisson model."""
    lam = max(0.001, prediction)
    floor_line = math.floor(line)
    is_integer = abs(line - round(line)) < 1e-9
    if is_integer:
        # P(X < line) = CDF(line-1)
        return poisson_cdf(int(round(line)) - 1, lam)
    else:
        # P(X <= floor(line)) = CDF(floor)
        return poisson_cdf(int(floor_line), lam)


def compute_over_probability(prediction: float, line: float) -> float:
    """Probability of going OVER the line based on Poisson model."""
    lam = max(0.001, prediction)
    floor_line = math.floor(line)
    is_integer = abs(line - round(line)) < 1e-9
    if is_integer:
        return 1.0 - poisson_cdf(int(round(line)), lam)
    else:
        return 1.0 - poisson_cdf(int(floor_line), lam)


def market_bucket_key(target: str, direction: str, line: float) -> str:
    return f"{target}|{direction}|{line:.1f}"


def is_standard_bettable_line(target: str, line: float) -> bool:
    if target in STANDARD_MARKET_LINES:
        return abs(float(line) - STANDARD_MARKET_LINES[target]) < 1e-9
    if target == "K":
        doubled = float(line) * 2.0
        return abs(doubled - round(doubled)) < 1e-9 and int(round(doubled)) % 2 == 1
    return False


@dataclass
class MaxWRCandidate:
    """Candidate ranked purely for win rate."""
    player: str
    team: str
    target: str
    direction: str
    market_line: float
    prediction: float
    edge: float
    model_probability: float  # raw Poisson probability
    bucket_key: str
    bucket_win_rate: float
    bucket_samples: int
    market_source: str
    market_books: int
    market_common_books: int
    market_common_book_keys: str
    market_line_std: float
    market_over_price: float | None
    market_under_price: float | None
    selected_side_price: float
    selected_sportsbook_key: str
    selected_sportsbook: str
    history_rows: int
    days_since_history: int
    # Composite confidence = blend of model prob and bucket historical rate
    composite_confidence: float
    game_id: str
    opponent: str
    raw: dict


def select_max_winrate_board(
    pool_csv: Path,
    *,
    min_model_probability: float = 0.88,
    min_bucket_win_rate: float = 0.83,
    min_bucket_samples: int = 500,
    min_history_rows: int = 35,
    min_market_books: int = 5,
    min_common_market_books: int = 2,
    max_days_since_history: int = 3,
    require_real_market: bool = True,
    min_edge: float = 0.50,
    max_board_size: int = 7,
    max_per_team: int = 2,
    max_per_game: int = 2,
) -> list[MaxWRCandidate]:
    """Select picks maximizing win rate from the raw prediction pool."""

    df = pd.read_csv(pool_csv)
    candidates: list[MaxWRCandidate] = []

    for _, row in df.iterrows():
        target = str(row.get("Target", "")).strip().upper()
        if target not in {"TB", "R", "K", "H", "HR"}:
            continue

        market_source = str(row.get("Market_Source", "")).strip().lower()
        if require_real_market and market_source != "real":
            continue

        market_books = _safe_int(row.get("Market_Books"))
        if market_books < min_market_books:
            continue
        market_common_books = _safe_int(row.get("Market_Common_Books"))
        if market_common_books < min_common_market_books:
            continue

        history_rows = _safe_int(row.get("History_Rows"))
        if history_rows < min_history_rows:
            continue

        days_since = _safe_int(row.get("Days_Since_History"), default=99)
        # If Days_Since_History not in pool, compute from Last_History_Date
        if days_since == 99:
            last_hist = str(row.get("Last_History_Date", "")).strip()[:10]
            run_date_str = str(row.get("Prediction_Run_Date", "") or row.get("Game_Date", "")).strip()[:10]
            if last_hist and run_date_str:
                try:
                    from datetime import date as _date
                    lh = _date.fromisoformat(last_hist)
                    rd = _date.fromisoformat(run_date_str)
                    days_since = (rd - lh).days
                except (ValueError, TypeError):
                    pass
        if days_since > max_days_since_history:
            continue

        prediction = _safe_float(row.get("Prediction"))
        market_line = _safe_float(row.get("Market_Line"))
        edge = _safe_float(row.get("Edge"))
        if not is_standard_bettable_line(target, market_line):
            continue

        # Determine direction from edge
        if edge > 0:
            direction = "OVER"
            model_prob = compute_over_probability(prediction, market_line)
        elif edge < 0:
            direction = "UNDER"
            model_prob = compute_under_probability(prediction, market_line)
        else:
            continue

        market_over_price = _valid_american_price(row.get("Market_Over_Price"))
        market_under_price = _valid_american_price(row.get("Market_Under_Price"))
        selected_side_price = market_over_price if direction == "OVER" else market_under_price
        selected_sportsbook_key = str(
            row.get("Market_Over_Book_Key" if direction == "OVER" else "Market_Under_Book_Key", "")
        ).strip().lower()
        selected_sportsbook = str(
            row.get("Market_Over_Book" if direction == "OVER" else "Market_Under_Book", "")
        ).strip()
        if selected_side_price is None or not selected_sportsbook_key or not selected_sportsbook:
            continue

        if abs(edge) < min_edge:
            continue

        if model_prob < min_model_probability:
            continue

        # Check bucket
        bucket_key = market_bucket_key(target, direction, market_line)

        # Look up in STRONG_BUCKETS (use target-direction fallback for high lines)
        bucket_info = STRONG_BUCKETS.get(bucket_key)
        if bucket_info is None:
            # For high lines (e.g. TB|UNDER|5.5), use the target-direction level
            td_key = f"{target}|{direction}"
            # TB|UNDER has 84.3% across all lines with 16022 samples
            td_fallbacks = {
                "TB|UNDER": (0.843, 16022),
                "R|UNDER": (0.792, 17500),
                "K|UNDER": (0.844, 9583),
                "H|UNDER": (0.641, 12974),
                "HR|UNDER": (0.830, 5000),
            }
            bucket_info = td_fallbacks.get(td_key)

        if bucket_info is None:
            continue

        bucket_win_rate, bucket_samples = bucket_info
        if bucket_win_rate < min_bucket_win_rate:
            continue
        if bucket_samples < min_bucket_samples:
            continue

        # Composite confidence: weight model more when it's very high
        # Model prob is "how likely THIS specific player goes under given their projection"
        # Bucket rate is "how often does this bucket win historically"
        # Use the minimum of the two as a conservative estimate
        composite = min(model_prob, bucket_win_rate)
        # But when both are very high, blend upward slightly
        if model_prob > 0.95 and bucket_win_rate > 0.85:
            composite = 0.60 * bucket_win_rate + 0.40 * model_prob

        candidates.append(MaxWRCandidate(
            player=str(row.get("Player", "")).strip(),
            team=str(row.get("Team", "")).strip(),
            target=target,
            direction=direction,
            market_line=market_line,
            prediction=prediction,
            edge=edge,
            model_probability=model_prob,
            bucket_key=bucket_key,
            bucket_win_rate=bucket_win_rate,
            bucket_samples=bucket_samples,
            market_source=market_source,
            market_books=market_books,
            market_common_books=market_common_books,
            market_common_book_keys=str(row.get("Market_Common_Book_Keys", "")).strip().lower(),
            market_line_std=_safe_float(row.get("Market_Line_Std")),
            market_over_price=market_over_price,
            market_under_price=market_under_price,
            selected_side_price=selected_side_price,
            selected_sportsbook_key=selected_sportsbook_key,
            selected_sportsbook=selected_sportsbook,
            history_rows=history_rows,
            days_since_history=days_since,
            composite_confidence=composite,
            game_id=str(row.get("Game_ID", "")).strip(),
            opponent=str(row.get("Opponent", "")).strip(),
            raw=dict(row),
        ))

    # Sort by composite confidence descending, then by bucket win rate, then model prob
    candidates.sort(
        key=lambda c: (c.composite_confidence, c.bucket_win_rate, c.model_probability, c.history_rows),
        reverse=True,
    )

    # Apply board limits
    from collections import Counter
    board: list[MaxWRCandidate] = []
    team_counts: Counter = Counter()
    game_counts: Counter = Counter()
    player_seen: set[str] = set()

    for c in candidates:
        if c.player in player_seen:
            continue
        if team_counts[c.team] >= max_per_team:
            continue
        if game_counts[c.game_id] >= max_per_game:
            continue
        if len(board) >= max_board_size:
            break

        board.append(c)
        player_seen.add(c.player)
        team_counts[c.team] += 1
        game_counts[c.game_id] += 1

    return board


def write_exporter_csv(path: Path, board: list[MaxWRCandidate]) -> None:
    """Write a CSV in the format expected by export_web_prediction_payload.py."""
    import csv as csv_mod

    fieldnames = [
        "Rank", "Prediction_Run_Date", "Game_Date", "Commence_Time_UTC",
        "Game_ID", "Game_Status_Code", "Player", "Player_ID", "Player_Type",
        "Team", "Opponent", "Is_Home", "Target", "Direction", "Prediction",
        "Market_Line", "Market_Source", "Original_Direction", "Direction_Flip_Applied",
        "Market_Books", "Market_Book_Keys", "Market_Common_Books", "Market_Common_Book_Keys",
        "Market_Line_Std", "Market_Over_Price", "Market_Under_Price",
        "Selected_Side_Price", "Selected_Sportsbook_Key", "Selected_Sportsbook",
        "Edge", "Abs_Edge", "History_Rows", "Last_History_Date", "Days_Since_History",
        "Model_Selected", "Model_Members", "Model_Val_MAE", "Model_Val_RMSE",
        "Model_Hit_Probability", "Estimated_Hit_Probability", "Estimated_Push_Probability",
        "Model_Graded_Hit_Rate", "Estimated_Graded_Hit_Rate",
        "Historical_Bucket_Key", "Historical_Prior_Source",
        "Historical_Bucket_Win_Rate", "Historical_Bucket_Support",
        "Historical_Prior_Weight", "Market_Implied_Probability",
        "Expected_Value_Per_Unit", "Price_Confirmed",
        "Historical_Bet_Profile_Key", "Historical_Bet_Profile_Source",
        "Historical_Bet_Profile_Win_Rate", "Historical_Bet_Profile_Support",
        "Historical_Bet_Profile_ROI", "Historical_Bet_Profile_Prior_Weight",
        "Historical_Market_Availability_Key", "Historical_Market_Availability_Source",
        "Historical_Market_Availability_Rate", "Historical_Market_Availability_Support",
        "Historical_Market_Avg_Books", "Edge_Over_MAE",
        "Precision_Score", "Selection_Score", "Confidence_Tier", "Market_Bucket",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, c in enumerate(board, 1):
            raw = c.raw
            writer.writerow({
                "Rank": idx,
                "Prediction_Run_Date": raw.get("Prediction_Run_Date", ""),
                "Game_Date": raw.get("Game_Date", ""),
                "Commence_Time_UTC": raw.get("Commence_Time_UTC", ""),
                "Game_ID": c.game_id,
                "Game_Status_Code": raw.get("Game_Status_Code", ""),
                "Player": c.player,
                "Player_ID": raw.get("Player_ID", ""),
                "Player_Type": raw.get("Player_Type", ""),
                "Team": c.team,
                "Opponent": c.opponent,
                "Is_Home": raw.get("Is_Home", ""),
                "Target": c.target,
                "Direction": c.direction,
                "Prediction": f"{c.prediction:.6f}",
                "Market_Line": f"{c.market_line:.6f}",
                "Market_Source": c.market_source,
                "Original_Direction": c.direction,
                "Direction_Flip_Applied": 0,
                "Market_Books": c.market_books,
                "Market_Book_Keys": raw.get("Market_Book_Keys", ""),
                "Market_Common_Books": c.market_common_books,
                "Market_Common_Book_Keys": c.market_common_book_keys,
                "Market_Line_Std": f"{c.market_line_std:.6f}",
                "Market_Over_Price": "" if c.market_over_price is None else f"{c.market_over_price:.6f}",
                "Market_Under_Price": "" if c.market_under_price is None else f"{c.market_under_price:.6f}",
                "Selected_Side_Price": f"{c.selected_side_price:.6f}",
                "Selected_Sportsbook_Key": c.selected_sportsbook_key,
                "Selected_Sportsbook": c.selected_sportsbook,
                "Edge": f"{c.edge:.6f}",
                "Abs_Edge": f"{abs(c.edge):.6f}",
                "History_Rows": c.history_rows,
                "Last_History_Date": raw.get("Last_History_Date", ""),
                "Days_Since_History": c.days_since_history,
                "Model_Selected": raw.get("Model_Selected", ""),
                "Model_Members": raw.get("Model_Members", ""),
                "Model_Val_MAE": raw.get("Model_Val_MAE", ""),
                "Model_Val_RMSE": raw.get("Model_Val_RMSE", ""),
                "Model_Hit_Probability": f"{c.model_probability:.6f}",
                "Estimated_Hit_Probability": f"{c.composite_confidence:.6f}",
                "Estimated_Push_Probability": "0.000000",
                "Model_Graded_Hit_Rate": f"{c.model_probability:.6f}",
                "Estimated_Graded_Hit_Rate": f"{c.bucket_win_rate:.6f}",
                "Historical_Bucket_Key": c.bucket_key,
                "Historical_Prior_Source": "line_bucket",
                "Historical_Bucket_Win_Rate": f"{c.bucket_win_rate:.6f}",
                "Historical_Bucket_Support": c.bucket_samples,
                "Historical_Prior_Weight": "0.350000",
                "Market_Implied_Probability": "",
                "Expected_Value_Per_Unit": "",
                "Price_Confirmed": 1,
                "Historical_Bet_Profile_Key": "",
                "Historical_Bet_Profile_Source": "",
                "Historical_Bet_Profile_Win_Rate": f"{c.bucket_win_rate:.6f}",
                "Historical_Bet_Profile_Support": c.bucket_samples,
                "Historical_Bet_Profile_ROI": "",
                "Historical_Bet_Profile_Prior_Weight": "0.000000",
                "Historical_Market_Availability_Key": "",
                "Historical_Market_Availability_Source": "",
                "Historical_Market_Availability_Rate": "0.500000",
                "Historical_Market_Availability_Support": 0,
                "Historical_Market_Avg_Books": f"{c.market_books:.6f}",
                "Edge_Over_MAE": f"{abs(c.edge) / 0.8:.6f}",
                "Precision_Score": f"{c.composite_confidence:.6f}",
                "Selection_Score": f"{c.composite_confidence:.6f}",
                "Confidence_Tier": "elite" if c.composite_confidence >= 0.94 else "strong",
                "Market_Bucket": c.bucket_key,
            })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Maximum win-rate MLB board selector.")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV compatible with the web exporter.")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON with detailed board stats.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON compatible with pipeline.")
    parser.add_argument("--min-model-prob", type=float, default=0.92)
    parser.add_argument("--min-bucket-win-rate", type=float, default=0.86)
    parser.add_argument("--min-bucket-samples", type=int, default=500)
    parser.add_argument("--min-history-rows", type=int, default=40)
    parser.add_argument("--min-market-books", type=int, default=5)
    parser.add_argument("--min-common-market-books", type=int, default=2)
    parser.add_argument("--max-days-since-history", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=0.35)
    parser.add_argument("--max-board-size", type=int, default=7)
    args = parser.parse_args()

    board = select_max_winrate_board(
        args.pool_csv,
        min_model_probability=args.min_model_prob,
        min_bucket_win_rate=args.min_bucket_win_rate,
        min_bucket_samples=args.min_bucket_samples,
        min_history_rows=args.min_history_rows,
        min_market_books=args.min_market_books,
        min_common_market_books=args.min_common_market_books,
        max_days_since_history=args.max_days_since_history,
        min_edge=args.min_edge,
        max_board_size=args.max_board_size,
    )

    print("=" * 70)
    print(f"MAX WIN-RATE BOARD — {len(board)} picks")
    print("=" * 70)
    for i, c in enumerate(board, 1):
        print(f"\n{i}. {c.player} ({c.team}) vs {c.opponent}")
        print(f"   {c.target} {c.direction} {c.market_line}")
        print(f"   Prediction: {c.prediction:.3f}  Edge: {c.edge:.3f}")
        print(f"   Model Prob: {c.model_probability:.1%}  Bucket WR: {c.bucket_win_rate:.1%} ({c.bucket_samples} samples)")
        print(f"   Composite Confidence: {c.composite_confidence:.1%}")
        print(f"   Market: {c.market_source} ({c.market_books} books, std={c.market_line_std:.3f})")
        print(f"   History: {c.history_rows} rows, {c.days_since_history}d stale")

    avg_conf = sum(c.composite_confidence for c in board) / len(board) if board else 0
    avg_bucket = sum(c.bucket_win_rate for c in board) / len(board) if board else 0
    avg_model = sum(c.model_probability for c in board) / len(board) if board else 0

    if board:
        print(f"\n{'='*70}")
        print(f"BOARD STATS:")
        print(f"  Picks: {len(board)}")
        print(f"  Avg Composite Confidence: {avg_conf:.1%}")
        print(f"  Avg Bucket Win Rate: {avg_bucket:.1%}")
        print(f"  Avg Model Probability: {avg_model:.1%}")
        print(f"  Expected misses (1 - avg_bucket_wr): {(1-avg_bucket)*len(board):.2f}")
        print(f"  Target: <= 1 miss per {len(board)}-pick board")

    # Write exporter-compatible CSV
    if args.out_csv:
        write_exporter_csv(args.out_csv, board)
        print(f"\n  Output CSV: {args.out_csv}")

    # Write summary JSON (compatible with pipeline)
    if args.summary_json:
        summary = {
            "pool_csv": report_path(args.pool_csv),
            "out_csv": report_path(args.out_csv) if args.out_csv else "",
            "rows_supported": 0,
            "rows_after_filters": len(board),
            "rows_selected": len(board),
            "selection": {
                "policy": "max_winrate_v1",
                "min_model_prob": args.min_model_prob,
                "min_bucket_win_rate": args.min_bucket_win_rate,
                "min_bucket_samples": args.min_bucket_samples,
                "min_history_rows": args.min_history_rows,
                "min_market_books": args.min_market_books,
                "min_common_market_books": args.min_common_market_books,
                "max_days_since_history": args.max_days_since_history,
                "min_edge": args.min_edge,
                "max_board_size": args.max_board_size,
            },
            "avg_abs_edge": round(sum(abs(c.edge) for c in board) / len(board), 4) if board else 0,
            "avg_hit_probability": round(avg_conf, 4),
            "avg_graded_hit_rate": round(avg_bucket, 4),
            "avg_precision_score": round(avg_conf, 4),
            "avg_historical_bucket_win_rate": round(avg_bucket, 4),
            "by_target": {},
            "by_direction": {},
            "publication_strategy": "max_winrate_v1",
        }
        from collections import Counter
        summary["by_target"] = dict(Counter(c.target for c in board))
        summary["by_direction"] = dict(Counter(c.direction for c in board))
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Summary JSON: {args.summary_json}")

    if args.out_json:
        output = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pool_csv": report_path(args.pool_csv),
            "board_size": len(board),
            "avg_composite_confidence": round(avg_conf, 4) if board else 0,
            "avg_bucket_win_rate": round(avg_bucket, 4) if board else 0,
            "avg_model_probability": round(avg_model, 4) if board else 0,
            "expected_misses": round((1-avg_bucket)*len(board), 3) if board else 0,
            "picks": [
                {
                    "rank": i,
                    "player": c.player,
                    "team": c.team,
                    "opponent": c.opponent,
                    "target": c.target,
                    "direction": c.direction,
                    "market_line": c.market_line,
                    "prediction": round(c.prediction, 4),
                    "edge": round(c.edge, 4),
                    "model_probability": round(c.model_probability, 4),
                    "bucket_key": c.bucket_key,
                    "bucket_win_rate": round(c.bucket_win_rate, 4),
                    "bucket_samples": c.bucket_samples,
                    "composite_confidence": round(c.composite_confidence, 4),
                    "market_books": c.market_books,
                    "market_common_books": c.market_common_books,
                    "selected_sportsbook_key": c.selected_sportsbook_key,
                    "selected_sportsbook": c.selected_sportsbook,
                    "history_rows": c.history_rows,
                }
                for i, c in enumerate(board, 1)
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"  Output JSON: {args.out_json}")


if __name__ == "__main__":
    main()
