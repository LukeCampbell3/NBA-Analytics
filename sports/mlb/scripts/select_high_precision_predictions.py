#!/usr/bin/env python3
"""
Build a smaller, higher-precision MLB prediction pool from a raw daily pool CSV.

This selector now leans harder toward raw win probability and board stability by:

1. Keeping only modeled rows by default (baseline rows are excluded).
2. Restricting to count-style MLB targets where a Poisson approximation is usable.
3. Calibrating model-implied hit rates with empirical target/direction/line buckets.
4. Penalizing stale history, low sample support, low edge quality, and push exposure.
5. Preventing one exact market bucket (for example `H OVER 0.5`) from dominating the board.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_HISTORY_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_CALIBRATION_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"

SUPPORTED_COUNT_TARGETS = {"H", "TB", "R", "K"}
FINAL_STATUS_CODES = {"F", "C", "D", "X"}
UPCOMING_STATUS_CODES = {"", "P", "S", "NS"}
HISTORICAL_TARGET_SPECS: dict[str, tuple[str, str, str]] = {
    "H": ("H", "Market_H", "H_market_gap"),
    "TB": ("TB", "Market_TB", "TB_market_gap"),
    "R": ("R", "Market_R", "R_market_gap"),
    "K": ("K", "Market_K", "K_market_gap"),
}


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
    model_hit_probability: float
    push_probability: float
    model_graded_hit_rate: float
    historical_bucket_key: str
    historical_prior_source: str
    historical_bucket_win_rate: float
    historical_bucket_support: int
    historical_prior_weight: float
    calibrated_hit_probability: float
    calibrated_graded_hit_rate: float
    edge_over_mae: float
    history_score: float
    recency_score: float
    bucket_support_score: float
    precision_score: float
    selection_score: float
    confidence_tier: str
    market_bucket: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a tighter, higher-precision MLB prediction board.")
    parser.add_argument("--pool-csv", type=Path, required=True, help="Raw MLB prediction pool CSV.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV for the selected board.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--top-n", type=int, default=10, help="Maximum number of plays to keep.")
    parser.add_argument("--min-abs-edge", type=float, default=0.45, help="Minimum absolute edge required.")
    parser.add_argument("--min-history-rows", type=int, default=11, help="Minimum history rows required.")
    parser.add_argument("--min-hit-probability", type=float, default=0.58, help="Minimum calibrated win probability.")
    parser.add_argument("--min-graded-hit-rate", type=float, default=0.60, help="Minimum calibrated win rate on graded outcomes.")
    parser.add_argument("--max-push-probability", type=float, default=0.24, help="Maximum push probability.")
    parser.add_argument("--max-days-since-history", type=int, default=4, help="Maximum staleness of last history row.")
    parser.add_argument("--max-per-player", type=int, default=1, help="Maximum selected rows per player.")
    parser.add_argument("--max-per-game", type=int, default=2, help="Maximum selected rows per game.")
    parser.add_argument("--max-per-team", type=int, default=3, help="Maximum selected rows per team.")
    parser.add_argument(
        "--max-per-market-bucket",
        type=int,
        default=4,
        help="Maximum selected rows from one exact target/direction/line market bucket.",
    )
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
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Processed MLB history root used to build empirical bucket priors.",
    )
    parser.add_argument(
        "--history-season",
        type=int,
        default=None,
        help="Season year used for empirical bucket priors. Defaults from pool run date.",
    )
    parser.add_argument(
        "--history-cache-json",
        type=Path,
        default=None,
        help="Optional cache JSON for empirical target/direction/line priors.",
    )
    parser.add_argument(
        "--refresh-history-cache",
        action="store_true",
        help="Recompute historical bucket priors even if the cache JSON exists.",
    )
    parser.add_argument(
        "--min-history-bucket-rows",
        type=int,
        default=50,
        help="Minimum graded rows required before using a line-specific historical bucket prior.",
    )
    parser.add_argument(
        "--max-history-prior-weight",
        type=float,
        default=0.35,
        help="Maximum weight given to empirical bucket priors when calibrating hit rates.",
    )
    parser.add_argument(
        "--history-prior-strength",
        type=float,
        default=400.0,
        help="Larger values make model probabilities dominate longer before historical priors take over.",
    )
    parser.add_argument(
        "--disable-historical-calibration",
        action="store_true",
        help="Disable empirical target/direction/line calibration and use model-only probabilities.",
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


def infer_history_season(pool_csv: Path, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    digits = "".join(char for char in pool_csv.stem if char.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return int(datetime.now(timezone.utc).year)


def default_history_cache_path(season: int) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"historical_bucket_priors_{int(season)}.json"


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


def format_market_line(line: float) -> str:
    return f"{float(line):.1f}"


def target_direction_key(target: str, direction: str) -> str:
    return f"{str(target).strip().upper()}|{str(direction).strip().upper()}"


def market_bucket_key(target: str, direction: str, market_line: float) -> str:
    return f"{target_direction_key(target, direction)}|{format_market_line(market_line)}"


def _empty_bucket_stats() -> dict[str, float | int]:
    return {"rows": 0, "graded_rows": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": 0.5, "push_rate": 0.0}


def _finalize_bucket_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    rows = int(stats.get("rows", 0))
    wins = int(stats.get("wins", 0))
    losses = int(stats.get("losses", 0))
    pushes = int(stats.get("pushes", 0))
    graded_rows = wins + losses
    win_rate = (wins / graded_rows) if graded_rows else 0.5
    push_rate = (pushes / rows) if rows else 0.0
    return {
        "rows": rows,
        "graded_rows": graded_rows,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(max(0.0, min(1.0, win_rate))),
        "push_rate": float(max(0.0, min(1.0, push_rate))),
    }


def _update_bucket(stats: dict[str, float | int], *, wins: int, losses: int, pushes: int) -> None:
    stats["rows"] = int(stats.get("rows", 0)) + int(wins + losses + pushes)
    stats["wins"] = int(stats.get("wins", 0)) + int(wins)
    stats["losses"] = int(stats.get("losses", 0)) + int(losses)
    stats["pushes"] = int(stats.get("pushes", 0)) + int(pushes)


def build_historical_bucket_priors(history_dir: Path, season: int) -> dict:
    target_direction_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    line_bucket_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    files = sorted(history_dir.glob(f"*/{int(season)}_processed_processed.csv"))

    required_columns = {"Date"}
    for actual_col, market_col, gap_col in HISTORICAL_TARGET_SPECS.values():
        required_columns.update({actual_col, market_col, gap_col})

    for path in files:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in required_columns)
        except Exception:
            continue
        if frame.empty:
            continue

        for target, (actual_col, market_col, gap_col) in HISTORICAL_TARGET_SPECS.items():
            if actual_col not in frame.columns or market_col not in frame.columns or gap_col not in frame.columns:
                continue

            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            market_line = pd.to_numeric(frame[market_col], errors="coerce")
            gap = pd.to_numeric(frame[gap_col], errors="coerce")
            mask = actual.notna() & market_line.notna() & gap.notna() & gap.ne(0)
            if not bool(mask.any()):
                continue

            sub = pd.DataFrame(
                {
                    "actual": actual.loc[mask],
                    "market_line": market_line.loc[mask],
                    "gap": gap.loc[mask],
                }
            )
            sub["direction"] = sub["gap"].gt(0).map({True: "OVER", False: "UNDER"})
            sub["win"] = (
                (sub["direction"].eq("OVER") & sub["actual"].gt(sub["market_line"]))
                | (sub["direction"].eq("UNDER") & sub["actual"].lt(sub["market_line"]))
            )
            sub["push"] = sub["actual"].eq(sub["market_line"])
            sub["loss"] = ~(sub["win"] | sub["push"])

            for direction, part in sub.groupby("direction"):
                td_key = target_direction_key(target, str(direction))
                _update_bucket(
                    target_direction_counts[td_key],
                    wins=int(part["win"].sum()),
                    losses=int(part["loss"].sum()),
                    pushes=int(part["push"].sum()),
                )

                for line_value, line_part in part.groupby("market_line"):
                    bucket_key = market_bucket_key(target, str(direction), float(line_value))
                    _update_bucket(
                        line_bucket_counts[bucket_key],
                        wins=int(line_part["win"].sum()),
                        losses=int(line_part["loss"].sum()),
                        pushes=int(line_part["push"].sum()),
                    )

    return {
        "season": int(season),
        "history_dir": str(history_dir.resolve()),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_count": int(len(files)),
        "target_direction": {key: _finalize_bucket_stats(value) for key, value in sorted(target_direction_counts.items())},
        "line_buckets": {key: _finalize_bucket_stats(value) for key, value in sorted(line_bucket_counts.items())},
    }


def load_or_build_historical_bucket_priors(
    *,
    history_dir: Path,
    season: int,
    cache_json: Path | None,
    refresh: bool,
) -> dict:
    if cache_json is not None and cache_json.exists() and not refresh:
        try:
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
            if int(payload.get("season", season)) == int(season):
                return payload
        except Exception:
            pass

    payload = build_historical_bucket_priors(history_dir, season)
    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def lookup_historical_bucket_prior(
    calibration: dict | None,
    *,
    target: str,
    direction: str,
    market_line: float,
    min_line_rows: int,
) -> tuple[str, float, int, str]:
    if not isinstance(calibration, dict):
        return "fallback", 0.5, 0, "fallback"

    line_key = market_bucket_key(target, direction, market_line)
    line_bucket = calibration.get("line_buckets", {}).get(line_key, {})
    line_rows = int(line_bucket.get("graded_rows", 0) or 0)
    if line_rows >= int(max(0, min_line_rows)):
        return line_key, float(line_bucket.get("win_rate", 0.5) or 0.5), line_rows, "line_bucket"

    td_key = target_direction_key(target, direction)
    td_bucket = calibration.get("target_direction", {}).get(td_key, {})
    td_rows = int(td_bucket.get("graded_rows", 0) or 0)
    if td_rows > 0:
        return td_key, float(td_bucket.get("win_rate", 0.5) or 0.5), td_rows, "target_direction"

    return "fallback", 0.5, 0, "fallback"


def blend_probability_with_prior(
    model_probability: float,
    *,
    prior_probability: float,
    support: int,
    max_weight: float,
    strength: float,
) -> tuple[float, float]:
    support_value = max(0.0, float(support))
    max_weight = float(max(0.0, min(1.0, max_weight)))
    strength = max(1.0, float(strength))
    if support_value <= 0.0 or max_weight <= 0.0:
        return float(max(0.0, min(1.0, model_probability))), 0.0

    weight = min(max_weight, support_value / (support_value + strength))
    blended = ((1.0 - weight) * float(model_probability)) + (weight * float(prior_probability))
    return float(max(0.0, min(1.0, blended))), float(weight)


def build_candidate(
    row: dict[str, str],
    *,
    calibration: dict | None,
    min_history_bucket_rows: int,
    max_history_prior_weight: float,
    history_prior_strength: float,
) -> Candidate | None:
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
    model_hit_probability, push_probability, model_graded_hit_rate = estimate_count_hit_probabilities(prediction, market_line, direction)

    historical_bucket_key, historical_bucket_win_rate, historical_bucket_support, historical_prior_source = lookup_historical_bucket_prior(
        calibration,
        target=target,
        direction=direction,
        market_line=market_line,
        min_line_rows=min_history_bucket_rows,
    )
    calibrated_hit_probability, historical_prior_weight = blend_probability_with_prior(
        model_hit_probability,
        prior_probability=historical_bucket_win_rate,
        support=historical_bucket_support,
        max_weight=max_history_prior_weight,
        strength=history_prior_strength,
    )
    calibrated_graded_hit_rate, _ = blend_probability_with_prior(
        model_graded_hit_rate,
        prior_probability=historical_bucket_win_rate,
        support=historical_bucket_support,
        max_weight=max_history_prior_weight,
        strength=history_prior_strength,
    )

    history_score = min(history_rows / 18.0, 1.0)
    recency_score = 0.0 if days_since_history is None else max(0.0, 1.0 - (days_since_history / 7.0))
    edge_over_mae = abs(edge) / max(model_val_mae, 0.1)
    bucket_support_score = min(max(float(historical_bucket_support), 0.0) / 500.0, 1.0)

    reliability_core = (
        0.54 * calibrated_hit_probability
        + 0.16 * calibrated_graded_hit_rate
        + 0.10 * history_score
        + 0.08 * recency_score
        + 0.07 * bucket_support_score
        + 0.05 * max(0.0, historical_bucket_win_rate - 0.50)
    )
    selection_score = reliability_core * (1.0 + 0.08 * min(edge_over_mae, 3.0)) * (1.0 - 0.55 * push_probability)
    selection_score = max(0.0, float(selection_score))

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
        model_hit_probability=model_hit_probability,
        push_probability=push_probability,
        model_graded_hit_rate=model_graded_hit_rate,
        historical_bucket_key=historical_bucket_key,
        historical_prior_source=historical_prior_source,
        historical_bucket_win_rate=historical_bucket_win_rate,
        historical_bucket_support=historical_bucket_support,
        historical_prior_weight=historical_prior_weight,
        calibrated_hit_probability=calibrated_hit_probability,
        calibrated_graded_hit_rate=calibrated_graded_hit_rate,
        edge_over_mae=edge_over_mae,
        history_score=history_score,
        recency_score=recency_score,
        bucket_support_score=bucket_support_score,
        precision_score=selection_score,
        selection_score=selection_score,
        confidence_tier=confidence_tier(selection_score),
        market_bucket=market_bucket_key(target, direction, market_line),
    )


def load_candidates(
    pool_csv: Path,
    *,
    calibration: dict | None,
    min_history_bucket_rows: int,
    max_history_prior_weight: float,
    history_prior_strength: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    with open(pool_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = build_candidate(
                row,
                calibration=calibration,
                min_history_bucket_rows=min_history_bucket_rows,
                max_history_prior_weight=max_history_prior_weight,
                history_prior_strength=history_prior_strength,
            )
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
        if candidate.calibrated_hit_probability < float(args.min_hit_probability):
            rejected["hit_probability_too_low"] += 1
            continue
        if candidate.calibrated_graded_hit_rate < float(args.min_graded_hit_rate):
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
            row.selection_score,
            row.historical_bucket_win_rate,
            1.0 if row.market_source == "real" else 0.0,
            row.calibrated_hit_probability,
            row.calibrated_graded_hit_rate,
            row.abs_edge,
            row.history_rows,
        ),
        reverse=True,
    )

    selected: list[Candidate] = []
    by_player: Counter[str] = Counter()
    by_game: Counter[str] = Counter()
    by_team: Counter[str] = Counter()
    by_market_bucket: Counter[str] = Counter()

    for candidate in ordered:
        if by_player[candidate.player_id or candidate.player] >= int(args.max_per_player):
            continue
        if by_game[candidate.game_id] >= int(args.max_per_game):
            continue
        if by_team[candidate.team] >= int(args.max_per_team):
            continue
        if int(args.max_per_market_bucket) > 0 and by_market_bucket[candidate.market_bucket] >= int(args.max_per_market_bucket):
            continue

        selected.append(candidate)
        by_player[candidate.player_id or candidate.player] += 1
        by_game[candidate.game_id] += 1
        by_team[candidate.team] += 1
        by_market_bucket[candidate.market_bucket] += 1

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
        "Model_Hit_Probability",
        "Estimated_Hit_Probability",
        "Estimated_Push_Probability",
        "Model_Graded_Hit_Rate",
        "Estimated_Graded_Hit_Rate",
        "Historical_Bucket_Key",
        "Historical_Prior_Source",
        "Historical_Bucket_Win_Rate",
        "Historical_Bucket_Support",
        "Historical_Prior_Weight",
        "Edge_Over_MAE",
        "Precision_Score",
        "Selection_Score",
        "Confidence_Tier",
        "Market_Bucket",
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
                    "Model_Hit_Probability": f"{candidate.model_hit_probability:.6f}",
                    "Estimated_Hit_Probability": f"{candidate.calibrated_hit_probability:.6f}",
                    "Estimated_Push_Probability": f"{candidate.push_probability:.6f}",
                    "Model_Graded_Hit_Rate": f"{candidate.model_graded_hit_rate:.6f}",
                    "Estimated_Graded_Hit_Rate": f"{candidate.calibrated_graded_hit_rate:.6f}",
                    "Historical_Bucket_Key": candidate.historical_bucket_key,
                    "Historical_Prior_Source": candidate.historical_prior_source,
                    "Historical_Bucket_Win_Rate": f"{candidate.historical_bucket_win_rate:.6f}",
                    "Historical_Bucket_Support": candidate.historical_bucket_support,
                    "Historical_Prior_Weight": f"{candidate.historical_prior_weight:.6f}",
                    "Edge_Over_MAE": f"{candidate.edge_over_mae:.6f}",
                    "Precision_Score": f"{candidate.precision_score:.6f}",
                    "Selection_Score": f"{candidate.selection_score:.6f}",
                    "Confidence_Tier": candidate.confidence_tier,
                    "Market_Bucket": candidate.market_bucket,
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
    calibration: dict | None,
) -> None:
    by_target = Counter(candidate.target for candidate in selected)
    by_direction = Counter(candidate.direction for candidate in selected)
    by_team = Counter(candidate.team for candidate in selected)
    by_market_bucket = Counter(candidate.market_bucket for candidate in selected)
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
            "max_per_market_bucket": int(args.max_per_market_bucket),
            "allow_baseline": bool(args.allow_baseline),
            "require_real_market_source": bool(args.require_real_market_source),
            "targets": [str(value).strip().upper() for value in args.targets],
            "history_season": int(args.history_season),
            "min_history_bucket_rows": int(args.min_history_bucket_rows),
            "max_history_prior_weight": float(args.max_history_prior_weight),
            "history_prior_strength": float(args.history_prior_strength),
            "historical_calibration_enabled": not bool(args.disable_historical_calibration),
        },
        "historical_calibration": {
            "cache_json": str(args.history_cache_json.resolve()) if args.history_cache_json else "",
            "history_dir": str(args.history_dir.resolve()),
            "season": int(args.history_season),
            "source_file_count": int((calibration or {}).get("source_file_count", 0)),
            "updated_at_utc": (calibration or {}).get("updated_at_utc"),
        },
        "filter_rejections": dict(rejected),
        "avg_abs_edge": round(sum(candidate.abs_edge for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_model_hit_probability": round(sum(candidate.model_hit_probability for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_hit_probability": round(sum(candidate.calibrated_hit_probability for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_model_graded_hit_rate": round(sum(candidate.model_graded_hit_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_graded_hit_rate": round(sum(candidate.calibrated_graded_hit_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_precision_score": round(sum(candidate.precision_score for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_historical_bucket_win_rate": round(sum(candidate.historical_bucket_win_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_historical_prior_weight": round(sum(candidate.historical_prior_weight for candidate in selected) / len(selected), 6) if selected else 0.0,
        "by_target": dict(by_target),
        "by_direction": dict(by_direction),
        "by_team": dict(by_team),
        "by_market_bucket": dict(by_market_bucket),
        "selected_preview": [
            {
                "rank": idx,
                "player": candidate.player,
                "team": candidate.team,
                "target": candidate.target,
                "direction": candidate.direction,
                "market_line": candidate.market_line,
                "market_bucket": candidate.market_bucket,
                "market_source": candidate.market_source,
                "prediction": round(candidate.prediction, 4),
                "model_hit_probability": round(candidate.model_hit_probability, 4),
                "estimated_hit_probability": round(candidate.calibrated_hit_probability, 4),
                "historical_bucket_win_rate": round(candidate.historical_bucket_win_rate, 4),
                "historical_bucket_support": int(candidate.historical_bucket_support),
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

    args.history_season = infer_history_season(args.pool_csv, args.history_season)
    if args.history_cache_json is None:
        args.history_cache_json = default_history_cache_path(args.history_season)

    default_csv, default_summary = default_output_paths(args.pool_csv)
    if args.out_csv is None:
        args.out_csv = default_csv
    if args.summary_json is None:
        args.summary_json = default_summary

    calibration = None
    if not args.disable_historical_calibration:
        calibration = load_or_build_historical_bucket_priors(
            history_dir=args.history_dir.resolve(),
            season=int(args.history_season),
            cache_json=args.history_cache_json.resolve() if args.history_cache_json else None,
            refresh=bool(args.refresh_history_cache),
        )

    candidates = load_candidates(
        args.pool_csv,
        calibration=calibration,
        min_history_bucket_rows=int(args.min_history_bucket_rows),
        max_history_prior_weight=float(args.max_history_prior_weight),
        history_prior_strength=float(args.history_prior_strength),
    )
    eligible, rejected = filter_candidates(candidates, args)
    selected = select_top_candidates(eligible, args)

    write_selected_csv(args.out_csv, selected)
    write_summary_json(args.summary_json, args, args.pool_csv, len(candidates), eligible, selected, rejected, calibration)

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
                f"model {candidate.model_hit_probability:.1%}, calibrated {candidate.calibrated_hit_probability:.1%}, "
                f"bucket {candidate.historical_bucket_win_rate:.1%} x {candidate.historical_bucket_support}, "
                f"score {candidate.selection_score:.3f})"
            )


if __name__ == "__main__":
    main()
