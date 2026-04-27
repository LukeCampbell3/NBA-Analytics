#!/usr/bin/env python3
"""
Build a smaller, higher-precision MLB prediction pool from a raw daily pool CSV.

The current MLB sample pool is heavy on candidate volume, but the practical need is
often a much tighter board with better odds of landing. This selector improves that
by:

1. Keeping only modeled rows by default (baseline rows are excluded).
2. Restricting to count-style MLB targets where a Poisson approximation is usable.
3. Estimating directional hit probability from the model mean and market line.
4. Penalizing stale history, low sample support, low edge, and high push exposure.
5. Enforcing concentration limits by player, game, and team.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


SUPPORTED_COUNT_TARGETS = {"H", "TB", "R", "K"}
FINAL_STATUS_CODES = {"F", "C", "D", "X"}
UPCOMING_STATUS_CODES = {"", "P", "S", "NS"}


@dataclass
class Candidate:
    raw: dict[str, str]
    player: str
    player_id: str
    team: str
    game_id: str
    target: str
    direction: str
    prediction: float
    market_line: float
    market_source: str
    edge: float
    abs_edge: float
    history_rows: int
    model_selected: str
    model_val_mae: float
    model_val_rmse: float
    run_date: date
    last_history_date: date | None
    days_since_history: int | None
    game_status_code: str
    hit_probability: float
    push_probability: float
    graded_hit_rate: float
    edge_over_mae: float
    history_score: float
    recency_score: float
    precision_score: float
    confidence_tier: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a tighter, higher-precision MLB prediction board.")
    parser.add_argument("--pool-csv", type=Path, required=True, help="Raw MLB prediction pool CSV.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV for the selected board.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--top-n", type=int, default=15, help="Maximum number of plays to keep.")
    parser.add_argument("--min-abs-edge", type=float, default=0.45, help="Minimum absolute edge required.")
    parser.add_argument("--min-history-rows", type=int, default=11, help="Minimum history rows required.")
    parser.add_argument("--min-hit-probability", type=float, default=0.58, help="Minimum raw win probability.")
    parser.add_argument("--min-graded-hit-rate", type=float, default=0.60, help="Minimum win rate on graded outcomes.")
    parser.add_argument("--max-push-probability", type=float, default=0.24, help="Maximum push probability.")
    parser.add_argument("--max-days-since-history", type=int, default=4, help="Maximum staleness of last history row.")
    parser.add_argument("--max-per-player", type=int, default=1, help="Maximum selected rows per player.")
    parser.add_argument("--max-per-game", type=int, default=2, help="Maximum selected rows per game.")
    parser.add_argument("--max-per-team", type=int, default=3, help="Maximum selected rows per team.")
    parser.add_argument(
        "--allow-baseline",
        action="store_true",
        help="Allow baseline rows. Default behavior keeps only non-baseline modeled rows.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=sorted(SUPPORTED_COUNT_TARGETS),
        help="Optional target whitelist. Defaults to supported count targets.",
    )
    parser.add_argument(
        "--require-real-market-source",
        action="store_true",
        help="Keep only rows backed by real sportsbook market lines.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Pool CSV not found: {path}")


def default_output_paths(pool_csv: Path) -> tuple[Path, Path]:
    stem = pool_csv.stem
    return (
        pool_csv.with_name(f"{stem}_high_precision_predictions.csv"),
        pool_csv.with_name(f"{stem}_high_precision_predictions_summary.json"),
    )


def parse_date(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def to_float(value: str, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_int(value: str, default: int = 0) -> int:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return default
    return out


def is_upcoming_status(status_code: str, detail: str) -> bool:
    code = str(status_code or "").strip().upper()
    detail_text = str(detail or "").strip().lower()
    if code in FINAL_STATUS_CODES:
        return False
    if "final" in detail_text or "completed" in detail_text:
        return False
    return code in UPCOMING_STATUS_CODES or not code


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    lam = max(0.0, float(lam))
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    log_p = (-lam) + (k * math.log(lam)) - math.lgamma(k + 1)
    return math.exp(log_p)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return min(1.0, sum(poisson_pmf(i, lam) for i in range(k + 1)))


def infer_direction(edge: float) -> str | None:
    if edge > 0:
        return "OVER"
    if edge < 0:
        return "UNDER"
    return None


def estimate_count_hit_probabilities(prediction: float, market_line: float, direction: str) -> tuple[float, float, float]:
    lam = max(0.0, prediction)
    rounded = round(market_line)
    is_integer_line = abs(market_line - rounded) < 1e-9

    if is_integer_line:
        push_probability = poisson_pmf(int(rounded), lam)
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(rounded), lam)
        else:
            hit_probability = poisson_cdf(int(rounded) - 1, lam)
    else:
        floor_line = math.floor(market_line)
        push_probability = 0.0
        if direction == "OVER":
            hit_probability = 1.0 - poisson_cdf(int(floor_line), lam)
        else:
            hit_probability = poisson_cdf(int(floor_line), lam)

    settle_probability = max(1e-9, 1.0 - push_probability)
    graded_hit_rate = hit_probability / settle_probability
    return (
        max(0.0, min(1.0, hit_probability)),
        max(0.0, min(1.0, push_probability)),
        max(0.0, min(1.0, graded_hit_rate)),
    )


def confidence_tier(score: float) -> str:
    if score >= 1.0:
        return "elite"
    if score >= 0.88:
        return "strong"
    if score >= 0.76:
        return "consider"
    return "pass"


def build_candidate(row: dict[str, str]) -> Candidate | None:
    target = str(row.get("Target", "")).strip().upper()
    if target not in SUPPORTED_COUNT_TARGETS:
        return None

    edge = to_float(row.get("Edge"))
    direction = infer_direction(edge)
    if direction is None:
        return None

    prediction = max(0.0, to_float(row.get("Prediction")))
    market_line = max(0.0, to_float(row.get("Market_Line")))
    market_source = str(row.get("Market_Source", "")).strip().lower() or "synthetic"
    history_rows = to_int(row.get("History_Rows"))
    model_val_mae = max(0.05, to_float(row.get("Model_Val_MAE"), default=0.0))
    model_val_rmse = max(model_val_mae, to_float(row.get("Model_Val_RMSE"), default=model_val_mae))
    run_date = parse_date(row.get("Prediction_Run_Date")) or parse_date(row.get("Game_Date"))
    if run_date is None:
        return None

    last_history_date = parse_date(row.get("Last_History_Date"))
    days_since_history = (run_date - last_history_date).days if last_history_date is not None else None
    hit_probability, push_probability, graded_hit_rate = estimate_count_hit_probabilities(prediction, market_line, direction)

    history_score = min(history_rows / 18.0, 1.0)
    recency_score = 0.0 if days_since_history is None else max(0.0, 1.0 - (days_since_history / 7.0))
    edge_over_mae = abs(edge) / max(model_val_mae, 0.1)

    reliability_core = (
        0.56 * hit_probability +
        0.24 * graded_hit_rate +
        0.12 * history_score +
        0.08 * recency_score
    )
    precision_score = reliability_core * (1.0 + 0.18 * min(edge_over_mae, 3.0)) * (1.0 - 0.55 * push_probability)

    return Candidate(
        raw=row,
        player=str(row.get("Player", "")).strip(),
        player_id=str(row.get("Player_ID", "")).strip(),
        team=str(row.get("Team", "")).strip(),
        game_id=str(row.get("Game_ID", "")).strip(),
        target=target,
        direction=direction,
        prediction=prediction,
        market_line=market_line,
        market_source=market_source,
        edge=edge,
        abs_edge=abs(edge),
        history_rows=history_rows,
        model_selected=str(row.get("Model_Selected", "")).strip(),
        model_val_mae=model_val_mae,
        model_val_rmse=model_val_rmse,
        run_date=run_date,
        last_history_date=last_history_date,
        days_since_history=days_since_history,
        game_status_code=str(row.get("Game_Status_Code", "")).strip().upper(),
        hit_probability=hit_probability,
        push_probability=push_probability,
        graded_hit_rate=graded_hit_rate,
        edge_over_mae=edge_over_mae,
        history_score=history_score,
        recency_score=recency_score,
        precision_score=precision_score,
        confidence_tier=confidence_tier(precision_score),
    )


def load_candidates(pool_csv: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with open(pool_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = build_candidate(row)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def filter_candidates(candidates: Iterable[Candidate], args: argparse.Namespace) -> tuple[list[Candidate], Counter]:
    allowed_targets = {str(value).strip().upper() for value in args.targets}
    rejected = Counter()
    kept: list[Candidate] = []

    for candidate in candidates:
        row = candidate.raw
        if candidate.target not in allowed_targets:
            rejected["unsupported_target"] += 1
            continue
        if not args.allow_baseline and candidate.model_selected.lower() == "baseline":
            rejected["baseline_model"] += 1
            continue
        if not is_upcoming_status(row.get("Game_Status_Code", ""), row.get("Game_Status_Detail", "")):
            rejected["non_upcoming_status"] += 1
            continue
        if candidate.abs_edge < float(args.min_abs_edge):
            rejected["edge_too_small"] += 1
            continue
        if candidate.history_rows < int(args.min_history_rows):
            rejected["history_too_short"] += 1
            continue
        if args.require_real_market_source and candidate.market_source != "real":
            rejected["synthetic_market_source"] += 1
            continue
        if candidate.market_source != "real" and candidate.target not in {"H", "TB", "R", "K"}:
            rejected["non_core_synthetic_market"] += 1
            continue
        if candidate.market_source != "real" and candidate.direction == "UNDER":
            rejected["synthetic_under_not_actionable"] += 1
            continue
        if candidate.hit_probability < float(args.min_hit_probability):
            rejected["hit_probability_too_low"] += 1
            continue
        if candidate.graded_hit_rate < float(args.min_graded_hit_rate):
            rejected["graded_hit_rate_too_low"] += 1
            continue
        if candidate.push_probability > float(args.max_push_probability):
            rejected["push_probability_too_high"] += 1
            continue
        if candidate.days_since_history is None or candidate.days_since_history > int(args.max_days_since_history):
            rejected["history_too_stale"] += 1
            continue
        kept.append(candidate)

    return kept, rejected


def select_top_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            row.precision_score,
            1.0 if row.market_source == "real" else 0.0,
            row.hit_probability,
            row.graded_hit_rate,
            row.abs_edge,
            row.history_rows,
        ),
        reverse=True,
    )

    selected: list[Candidate] = []
    by_player: Counter[str] = Counter()
    by_game: Counter[str] = Counter()
    by_team: Counter[str] = Counter()

    for candidate in ordered:
        if by_player[candidate.player_id or candidate.player] >= int(args.max_per_player):
            continue
        if by_game[candidate.game_id] >= int(args.max_per_game):
            continue
        if by_team[candidate.team] >= int(args.max_per_team):
            continue

        selected.append(candidate)
        by_player[candidate.player_id or candidate.player] += 1
        by_game[candidate.game_id] += 1
        by_team[candidate.team] += 1

        if len(selected) >= int(args.top_n):
            break

    return selected


def write_selected_csv(path: Path, selected: list[Candidate]) -> None:
    fieldnames = [
        "Rank",
        "Prediction_Run_Date",
        "Game_Date",
        "Commence_Time_UTC",
        "Game_ID",
        "Game_Status_Code",
        "Player",
        "Player_ID",
        "Player_Type",
        "Team",
        "Opponent",
        "Is_Home",
        "Target",
        "Direction",
        "Prediction",
        "Market_Line",
        "Market_Source",
        "Edge",
        "Abs_Edge",
        "History_Rows",
        "Last_History_Date",
        "Days_Since_History",
        "Model_Selected",
        "Model_Members",
        "Model_Val_MAE",
        "Model_Val_RMSE",
        "Estimated_Hit_Probability",
        "Estimated_Push_Probability",
        "Estimated_Graded_Hit_Rate",
        "Edge_Over_MAE",
        "Precision_Score",
        "Confidence_Tier",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, candidate in enumerate(selected, start=1):
            writer.writerow(
                {
                    "Rank": idx,
                    "Prediction_Run_Date": candidate.raw.get("Prediction_Run_Date", ""),
                    "Game_Date": candidate.raw.get("Game_Date", ""),
                    "Commence_Time_UTC": candidate.raw.get("Commence_Time_UTC", ""),
                    "Game_ID": candidate.game_id,
                    "Game_Status_Code": candidate.game_status_code,
                    "Player": candidate.player,
                    "Player_ID": candidate.player_id,
                    "Player_Type": candidate.raw.get("Player_Type", ""),
                    "Team": candidate.team,
                    "Opponent": candidate.raw.get("Opponent", ""),
                    "Is_Home": candidate.raw.get("Is_Home", ""),
                    "Target": candidate.target,
                    "Direction": candidate.direction,
                    "Prediction": f"{candidate.prediction:.6f}",
                    "Market_Line": f"{candidate.market_line:.6f}",
                    "Market_Source": candidate.market_source,
                    "Edge": f"{candidate.edge:.6f}",
                    "Abs_Edge": f"{candidate.abs_edge:.6f}",
                    "History_Rows": candidate.history_rows,
                    "Last_History_Date": candidate.last_history_date.isoformat() if candidate.last_history_date else "",
                    "Days_Since_History": "" if candidate.days_since_history is None else candidate.days_since_history,
                    "Model_Selected": candidate.model_selected,
                    "Model_Members": candidate.raw.get("Model_Members", ""),
                    "Model_Val_MAE": f"{candidate.model_val_mae:.6f}",
                    "Model_Val_RMSE": f"{candidate.model_val_rmse:.6f}",
                    "Estimated_Hit_Probability": f"{candidate.hit_probability:.6f}",
                    "Estimated_Push_Probability": f"{candidate.push_probability:.6f}",
                    "Estimated_Graded_Hit_Rate": f"{candidate.graded_hit_rate:.6f}",
                    "Edge_Over_MAE": f"{candidate.edge_over_mae:.6f}",
                    "Precision_Score": f"{candidate.precision_score:.6f}",
                    "Confidence_Tier": candidate.confidence_tier,
                }
            )


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    pool_csv: Path,
    total_candidates: int,
    eligible_candidates: list[Candidate],
    selected: list[Candidate],
    rejected: Counter,
) -> None:
    by_target = Counter(candidate.target for candidate in selected)
    by_direction = Counter(candidate.direction for candidate in selected)
    by_team = Counter(candidate.team for candidate in selected)
    summary = {
        "pool_csv": str(pool_csv.resolve()),
        "out_csv": str((args.out_csv or default_output_paths(pool_csv)[0]).resolve()),
        "rows_supported": total_candidates,
        "rows_after_filters": len(eligible_candidates),
        "rows_selected": len(selected),
        "selection": {
            "top_n": int(args.top_n),
            "min_abs_edge": float(args.min_abs_edge),
            "min_history_rows": int(args.min_history_rows),
            "min_hit_probability": float(args.min_hit_probability),
            "min_graded_hit_rate": float(args.min_graded_hit_rate),
            "max_push_probability": float(args.max_push_probability),
            "max_days_since_history": int(args.max_days_since_history),
            "max_per_player": int(args.max_per_player),
            "max_per_game": int(args.max_per_game),
            "max_per_team": int(args.max_per_team),
            "allow_baseline": bool(args.allow_baseline),
            "require_real_market_source": bool(args.require_real_market_source),
            "targets": [str(value).strip().upper() for value in args.targets],
        },
        "filter_rejections": dict(rejected),
        "avg_abs_edge": round(sum(candidate.abs_edge for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_hit_probability": round(sum(candidate.hit_probability for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_graded_hit_rate": round(sum(candidate.graded_hit_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_precision_score": round(sum(candidate.precision_score for candidate in selected) / len(selected), 6) if selected else 0.0,
        "by_target": dict(by_target),
        "by_direction": dict(by_direction),
        "by_team": dict(by_team),
        "selected_preview": [
            {
                "rank": idx,
                "player": candidate.player,
                "team": candidate.team,
                "target": candidate.target,
                "direction": candidate.direction,
                "market_line": candidate.market_line,
                "market_source": candidate.market_source,
                "prediction": round(candidate.prediction, 4),
                "estimated_hit_probability": round(candidate.hit_probability, 4),
                "estimated_graded_hit_rate": round(candidate.graded_hit_rate, 4),
                "precision_score": round(candidate.precision_score, 4),
            }
            for idx, candidate in enumerate(selected[:10], start=1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_file(args.pool_csv)

    default_csv, default_summary = default_output_paths(args.pool_csv)
    if args.out_csv is None:
        args.out_csv = default_csv
    if args.summary_json is None:
        args.summary_json = default_summary

    candidates = load_candidates(args.pool_csv)
    eligible, rejected = filter_candidates(candidates, args)
    selected = select_top_candidates(eligible, args)

    write_selected_csv(args.out_csv, selected)
    write_summary_json(args.summary_json, args, args.pool_csv, len(candidates), eligible, selected, rejected)

    print("\n" + "=" * 88)
    print("MLB HIGH-PRECISION SELECTOR")
    print("=" * 88)
    print(f"Pool CSV:           {args.pool_csv}")
    print(f"Supported rows:     {len(candidates)}")
    print(f"Rows after filters: {len(eligible)}")
    print(f"Rows selected:      {len(selected)}")
    print(f"Output CSV:         {args.out_csv}")
    print(f"Summary JSON:       {args.summary_json}")
    if selected:
        print("\nTop selections:")
        for idx, candidate in enumerate(selected[:10], start=1):
            print(
                f"{idx:>2}. {candidate.player} {candidate.target} {candidate.direction} "
                f"(line {candidate.market_line:.1f}, pred {candidate.prediction:.3f}, "
                f"hit {candidate.hit_probability:.1%}, graded {candidate.graded_hit_rate:.1%}, "
                f"score {candidate.precision_score:.3f})"
            )


if __name__ == "__main__":
    main()
