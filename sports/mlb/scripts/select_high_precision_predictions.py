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
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.mlb.decision_engine.matchup_network import NETWORK_VERSION  # noqa: E402

try:
    from .live_board_confidence import apply_live_board_calibration
except ImportError:
    from live_board_confidence import apply_live_board_calibration

try:
    from .pick_survival_model import apply_pick_survival_model
except ImportError:
    from pick_survival_model import apply_pick_survival_model


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_HISTORY_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_CALIBRATION_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"

SUPPORTED_COUNT_TARGETS = {"H", "TB", "R", "HR", "RBI", "K", "ER"}
STANDARD_MARKET_LINES = {"H": 0.5, "TB": 1.5, "R": 0.5, "HR": 0.5, "RBI": 0.5}
MAX_CALIBRATED_PROBABILITY = 0.80
FINAL_STATUS_CODES = {"F", "C", "D", "X"}
UPCOMING_STATUS_CODES = {"", "P", "S", "NS"}
RECENT_FORM_LOOKBACK_DAYS = 14
RECENT_FORM_MIN_LINE_ROWS = 10
RECENT_FORM_MAX_WEIGHT = 0.22
RECENT_FORM_STRENGTH = 45.0
CORE_SELECTION_PROFILE = "core_market_v1"
OPTIMIZED_OVER_SELECTION_PROFILE = "r_tb_over_moderate_edge_v1"
OPTIMIZED_OVER_PROFILE_STATUS = "probation"
PITCHER_K_OVER_SELECTION_PROFILE = "pitcher_k_over_workload_v1"
PITCHER_K_OVER_PROFILE_STATUS = "probation"
HISTORICAL_TARGET_SPECS: dict[str, tuple[str, str, str]] = {
    "H": ("H", "Market_H", "H_market_gap"),
    "TB": ("TB", "Market_TB", "TB_market_gap"),
    "R": ("R", "Market_R", "R_market_gap"),
    "HR": ("HR", "Market_HR", "HR_market_gap"),
    "RBI": ("RBI", "Market_RBI", "RBI_market_gap"),
    "K": ("K", "Market_K", "K_market_gap"),
    "ER": ("ER", "Market_ER", "ER_market_gap"),
}
HISTORICAL_BET_TARGET_SPECS: dict[str, tuple[str, str, str, str, str, str, str]] = {
    "H": ("H", "Market_H", "H_market_gap", "Market_Source_H", "Market_H_books", "Market_H_over_price", "Market_H_under_price"),
    "TB": ("TB", "Market_TB", "TB_market_gap", "Market_Source_TB", "Market_TB_books", "Market_TB_over_price", "Market_TB_under_price"),
    "R": ("R", "Market_R", "R_market_gap", "Market_Source_R", "Market_R_books", "Market_R_over_price", "Market_R_under_price"),
    "HR": ("HR", "Market_HR", "HR_market_gap", "Market_Source_HR", "Market_HR_books", "Market_HR_over_price", "Market_HR_under_price"),
    "RBI": ("RBI", "Market_RBI", "RBI_market_gap", "Market_Source_RBI", "Market_RBI_books", "Market_RBI_over_price", "Market_RBI_under_price"),
    "K": ("K", "Market_K", "K_market_gap", "Market_Source_K", "Market_K_books", "Market_K_over_price", "Market_K_under_price"),
    "ER": ("ER", "Market_ER", "ER_market_gap", "Market_Source_ER", "Market_ER_books", "Market_ER_over_price", "Market_ER_under_price"),
}


def report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass
class Candidate:
    raw: dict[str, str]
    player: str
    player_id: str
    team: str
    game_id: str
    target: str
    direction: str
    original_direction: str
    direction_flip_applied: bool
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
    live_confidence_calibration_key: str
    live_confidence_calibration_support: int
    live_confidence_calibration_adjustment: float
    market_books: int
    market_book_keys: str
    market_common_books: int
    market_common_book_keys: str
    market_line_std: float
    market_over_price: float | None
    market_under_price: float | None
    selected_side_price: float | None
    opposite_side_price: float | None
    selected_sportsbook_key: str
    selected_sportsbook: str
    price_confirmed: bool
    market_implied_probability: float | None
    expected_value_per_unit: float | None
    historical_bet_profile_key: str
    historical_bet_profile_source: str
    historical_bet_profile_win_rate: float
    historical_bet_profile_support: int
    historical_bet_profile_roi: float | None
    historical_bet_profile_prior_weight: float
    historical_market_availability_key: str
    historical_market_availability_source: str
    historical_market_availability_rate: float
    historical_market_availability_support: int
    historical_market_avg_books: float
    edge_over_mae: float
    history_score: float
    recency_score: float
    bucket_support_score: float
    precision_score: float
    selection_score: float
    confidence_tier: str
    market_bucket: str
    survival_probability: float | None = None
    survival_expected_value: float | None = None
    survival_model_status: str = "disabled"
    survival_model_support: int = 0
    survival_rank_active: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a tighter, higher-precision MLB prediction board.")
    parser.add_argument("--pool-csv", type=Path, required=True, help="Raw MLB prediction pool CSV.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV for the selected board.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--top-n", type=int, default=6, help="Maximum number of plays to keep.")
    parser.add_argument(
        "--daily-pick-soft-cap",
        type=int,
        default=0,
        help="Normal daily board size. Picks above this count must clear the post-cap score floor; 0 disables.",
    )
    parser.add_argument(
        "--post-cap-min-selection-score",
        type=float,
        default=0.0,
        help="Minimum selection score required after the daily soft cap has been reached.",
    )
    parser.add_argument("--min-abs-edge", type=float, default=0.45, help="Minimum absolute edge required.")
    parser.add_argument("--min-history-rows", type=int, default=11, help="Minimum history rows required.")
    parser.add_argument(
        "--min-prediction",
        type=float,
        default=0.0,
        help="Minimum model projection required; use a positive threshold to reject DNP or role-risk rows.",
    )
    parser.add_argument("--min-hit-probability", type=float, default=0.58, help="Minimum calibrated win probability.")
    parser.add_argument("--min-graded-hit-rate", type=float, default=0.68, help="Minimum calibrated win rate on graded outcomes.")
    parser.add_argument(
        "--optimized-over-targets",
        nargs="*",
        default=[],
        help="Targets that use the separately validated OVER selection profile.",
    )
    parser.add_argument("--over-min-abs-edge", type=float, default=None, help="Optional minimum edge for optimized OVER targets.")
    parser.add_argument("--over-max-abs-edge", type=float, default=None, help="Optional maximum edge for optimized OVER targets.")
    parser.add_argument(
        "--over-min-model-hit-probability",
        type=float,
        default=None,
        help="Optional minimum unblended model probability for optimized OVER targets.",
    )
    parser.add_argument(
        "--over-max-model-hit-probability",
        type=float,
        default=None,
        help="Optional maximum unblended model probability for optimized OVER targets.",
    )
    parser.add_argument(
        "--over-min-expected-value",
        type=float,
        default=None,
        help="Optional minimum expected profit per unit for optimized OVER targets.",
    )
    parser.add_argument(
        "--over-max-american-price",
        type=float,
        default=None,
        help="Optional longest acceptable American price for optimized OVER targets.",
    )
    parser.add_argument(
        "--core-max-american-price",
        type=float,
        default=None,
        help="Optional longest acceptable American price for non-optimized core selections.",
    )
    parser.add_argument(
        "--core-min-american-price",
        type=float,
        default=None,
        help="Optional most heavily juiced acceptable American price for non-optimized core selections.",
    )
    parser.add_argument(
        "--over-min-history-rows",
        type=int,
        default=None,
        help="Optional minimum player-history depth for the probationary optimized OVER profile.",
    )
    parser.add_argument(
        "--enable-pitcher-k-over-profile",
        action="store_true",
        help="Enable the workload-gated, probable-starter K OVER profile.",
    )
    parser.add_argument("--pitcher-k-min-starter-history", type=int, default=15)
    parser.add_argument("--pitcher-k-min-projected-ip", type=float, default=5.25)
    parser.add_argument("--pitcher-k-min-projected-pitches", type=float, default=75.0)
    parser.add_argument("--pitcher-k-max-days-since-history", type=int, default=14)
    parser.add_argument("--pitcher-k-min-abs-edge", type=float, default=0.15)
    parser.add_argument("--pitcher-k-max-abs-edge", type=float, default=1.0)
    parser.add_argument("--pitcher-k-min-model-hit-probability", type=float, default=0.50)
    parser.add_argument("--pitcher-k-max-model-hit-probability", type=float, default=0.65)
    parser.add_argument("--pitcher-k-min-expected-value", type=float, default=0.0)
    parser.add_argument("--pitcher-k-min-american-price", type=float, default=-130.0)
    parser.add_argument("--pitcher-k-max-american-price", type=float, default=130.0)
    parser.add_argument(
        "--max-pitcher-k-picks",
        type=int,
        default=1,
        help="Maximum workload-profile pitcher strikeout picks on the board.",
    )
    parser.add_argument(
        "--min-over-picks",
        type=int,
        default=0,
        help="Reserve up to this many board positions for eligible OVER picks.",
    )
    parser.add_argument(
        "--max-over-picks",
        type=int,
        default=0,
        help="Maximum OVER picks on the board. Set 0 to disable the cap.",
    )
    parser.add_argument(
        "--max-under-picks",
        type=int,
        default=0,
        help="Maximum UNDER picks on the board. Set 0 to disable the cap.",
    )
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
        "--optimized-over-max-per-market-bucket",
        type=int,
        default=None,
        help="Optional market-bucket cap applied only to validated optimized OVER targets.",
    )
    parser.add_argument(
        "--min-market-books",
        type=int,
        default=2,
        help="Optional minimum contributing books required for real-market rows. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-market-line-std",
        type=float,
        default=0.0,
        help="Optional maximum line standard deviation allowed for real-market rows. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-expected-value",
        type=float,
        default=-1.0,
        help="Optional minimum expected profit per unit for priced real-market rows. Set below -0.99 to disable.",
    )
    parser.add_argument(
        "--min-common-market-books",
        type=int,
        default=1,
        help="Minimum major-book coverage at the exact selected line. Set 0 to disable.",
    )
    parser.add_argument(
        "--allow-unpriced-side",
        action="store_true",
        help="Allow real-market rows without a valid selected-side American price.",
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
        "--history-before-date",
        type=str,
        default=None,
        help="Leakage-safe cutoff: use only history strictly before this YYYY-MM-DD date.",
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
    parser.add_argument(
        "--bet-profile-cache-json",
        type=Path,
        default=None,
        help="Optional cache JSON for settled real-market MLB bet-profile priors.",
    )
    parser.add_argument(
        "--refresh-bet-profile-cache",
        action="store_true",
        help="Recompute historical bet-profile priors even if the cache JSON exists.",
    )
    parser.add_argument(
        "--min-bet-profile-rows",
        type=int,
        default=12,
        help="Minimum settled priced rows required before using a line-specific MLB bet-profile prior.",
    )
    parser.add_argument(
        "--max-bet-profile-prior-weight",
        type=float,
        default=0.25,
        help="Maximum weight given to settled real-market bet-profile priors when refining graded hit rates.",
    )
    parser.add_argument(
        "--bet-profile-prior-strength",
        type=float,
        default=80.0,
        help="Larger values make model calibration dominate longer before settled bet-profile priors take over.",
    )
    parser.add_argument(
        "--min-market-availability-rows",
        type=int,
        default=12,
        help="Minimum real-market rows required before using a line-specific side-price availability prior.",
    )
    parser.add_argument(
        "--disable-historical-bet-profiles",
        action="store_true",
        help="Disable settled real-market bet-profile and placeability priors.",
    )
    parser.add_argument(
        "--live-confidence-cache-json",
        type=Path,
        default=None,
        help="Current-profile calibration learned from settled, price-confirmed published boards.",
    )
    parser.add_argument(
        "--disable-live-confidence-calibration",
        action="store_true",
        help="Disable current-profile target/direction confidence corrections.",
    )
    parser.add_argument(
        "--pick-survival-cache-json",
        type=Path,
        default=None,
        help="Shadow-only pick-survival model; annotations never change selection ordering or eligibility.",
    )
    parser.add_argument(
        "--disable-pick-survival-shadow",
        action="store_true",
        help="Disable shadow pick-survival annotations.",
    )
    parser.add_argument(
        "--allow-synthetic-unders",
        action="store_true",
        help="Allow synthetic under positions. Default behavior keeps synthetic fallback boards over-only.",
    )
    parser.add_argument(
        "--prefer-confident-side",
        action="store_true",
        help="For synthetic or unpriced rows, compare both sides and keep the historically stronger direction.",
    )
    parser.add_argument(
        "--min-historical-bet-profile-support",
        type=int,
        default=0,
        help="Optional minimum settled historical bet-profile support required.",
    )
    parser.add_argument(
        "--min-historical-bet-profile-win-rate",
        type=float,
        default=0.0,
        help="Optional minimum settled historical bet-profile win rate required when support is present.",
    )
    parser.add_argument(
        "--min-historical-market-availability-support",
        type=int,
        default=0,
        help="Optional minimum historical market-availability support required.",
    )
    parser.add_argument(
        "--min-historical-market-availability-rate",
        type=float,
        default=0.0,
        help="Optional minimum historical rate at which this market side was actually priced.",
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


def infer_pool_run_date(pool_csv: Path) -> date | None:
    digits = "".join(char for char in pool_csv.stem if char.isdigit())
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return None


def default_history_cache_path(season: int) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"historical_bucket_priors_{int(season)}.json"


def default_bet_profile_cache_path(season: int) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"historical_bet_profile_priors_{int(season)}.json"


def default_live_confidence_cache_path(season: int) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"live_board_confidence_calibration_{int(season)}.json"


def default_pick_survival_cache_path(season: int) -> Path:
    return DEFAULT_CALIBRATION_ROOT / f"pick_survival_model_{int(season)}.json"


def load_live_confidence_calibration(path: Path | None, run_date: date | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cutoff = parse_date(payload.get("history_before_date"))
    if run_date is not None and cutoff is not None and cutoff > run_date:
        return None
    return payload


def load_pick_survival_model(path: Path | None, run_date: date | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cutoff = parse_date(payload.get("history_before_date"))
    if run_date is not None and cutoff is not None and cutoff > run_date:
        return None
    return payload


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


def is_standard_bettable_line(target: str, market_line: float) -> bool:
    target_key = str(target).strip().upper()
    if target_key in STANDARD_MARKET_LINES:
        return abs(float(market_line) - STANDARD_MARKET_LINES[target_key]) < 1e-9
    if target_key in {"K", "ER"}:
        doubled = float(market_line) * 2.0
        return abs(doubled - round(doubled)) < 1e-9 and int(round(doubled)) % 2 == 1
    return False


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def target_direction_key(target: str, direction: str) -> str:
    return f"{str(target).strip().upper()}|{str(direction).strip().upper()}"


def market_bucket_key(target: str, direction: str, market_line: float) -> str:
    return f"{target_direction_key(target, direction)}|{format_market_line(market_line)}"


def probability_bucket(probability: float) -> str:
    bounded = clamp01(probability)
    lower = math.floor(bounded * 20.0) / 20.0
    upper = min(1.0, lower + 0.05)
    return f"{lower:.2f}-{upper:.2f}"


def target_probability_key(target: str, direction: str, graded_probability: float) -> str:
    return f"{target_direction_key(target, direction)}|{probability_bucket(graded_probability)}"


def line_probability_key(target: str, direction: str, market_line: float, graded_probability: float) -> str:
    return f"{market_bucket_key(target, direction, market_line)}|{probability_bucket(graded_probability)}"


def american_implied_probability(price: float | None) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not math.isfinite(value) or abs(value) < 100.0 or abs(value - round(value)) > 1e-6:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def american_profit_per_unit(price: float | None) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not math.isfinite(value) or abs(value) < 100.0:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


def no_vig_side_probability(
    side_price: float | None,
    opposite_price: float | None,
) -> float | None:
    side_implied = american_implied_probability(side_price)
    if side_implied is None:
        return None
    opposite_implied = american_implied_probability(opposite_price)
    if opposite_implied is None:
        return side_implied
    total = side_implied + opposite_implied
    if total <= 1e-9:
        return side_implied
    return side_implied / total


def expected_profit_per_unit(probability: float, price: float | None) -> float | None:
    profit_if_win = american_profit_per_unit(price)
    if profit_if_win is None:
        return None
    probability = clamp01(probability)
    return (probability * profit_if_win) - ((1.0 - probability) * 1.0)


def directional_model_gap(prediction: float, market_line: float, direction: str) -> float:
    if str(direction).strip().upper() == "UNDER":
        return float(market_line) - float(prediction)
    return float(prediction) - float(market_line)


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


def infer_recent_history_cutoff(
    history_dir: Path,
    season: int,
    lookback_days: int,
    history_before_date: date | None = None,
) -> date | None:
    files = sorted(history_dir.glob(f"*/{int(season)}_processed_processed.csv"))
    max_game_date: date | None = None
    for path in files:
        try:
            frame = pd.read_csv(path, usecols=["Date"])
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.date.dropna()
        if history_before_date is not None:
            dates = dates.loc[dates < history_before_date]
        if dates.empty:
            continue
        candidate = max(dates)
        if max_game_date is None or candidate > max_game_date:
            max_game_date = candidate
    if max_game_date is None:
        return None
    return max_game_date - timedelta(days=max(1, int(lookback_days) - 1))


def build_historical_bucket_priors(
    history_dir: Path,
    season: int,
    history_before_date: date | None = None,
) -> dict:
    target_direction_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    line_bucket_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    recent_target_direction_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    recent_line_bucket_counts: dict[str, dict[str, float | int]] = defaultdict(_empty_bucket_stats)
    files = sorted(history_dir.glob(f"*/{int(season)}_processed_processed.csv"))
    recent_cutoff = infer_recent_history_cutoff(
        history_dir,
        season,
        RECENT_FORM_LOOKBACK_DAYS,
        history_before_date,
    )

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
        frame = frame.copy()
        frame["_game_date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.date
        if history_before_date is not None:
            frame = frame.loc[frame["_game_date"] < history_before_date].copy()
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
                    "game_date": frame.loc[mask, "_game_date"],
                }
            )
            sub["direction"] = sub["gap"].gt(0).map({True: "OVER", False: "UNDER"})
            sub["win"] = (
                (sub["direction"].eq("OVER") & sub["actual"].gt(sub["market_line"]))
                | (sub["direction"].eq("UNDER") & sub["actual"].lt(sub["market_line"]))
            )
            sub["push"] = sub["actual"].eq(sub["market_line"])
            sub["loss"] = ~(sub["win"] | sub["push"])
            if recent_cutoff is not None:
                sub["is_recent"] = pd.to_datetime(sub["game_date"], errors="coerce").dt.date.ge(recent_cutoff)
            else:
                sub["is_recent"] = False

            for direction, part in sub.groupby("direction"):
                td_key = target_direction_key(target, str(direction))
                _update_bucket(
                    target_direction_counts[td_key],
                    wins=int(part["win"].sum()),
                    losses=int(part["loss"].sum()),
                    pushes=int(part["push"].sum()),
                )
                recent_part = part.loc[part["is_recent"]].copy()
                if not recent_part.empty:
                    _update_bucket(
                        recent_target_direction_counts[td_key],
                        wins=int(recent_part["win"].sum()),
                        losses=int(recent_part["loss"].sum()),
                        pushes=int(recent_part["push"].sum()),
                    )

                for line_value, line_part in part.groupby("market_line"):
                    bucket_key = market_bucket_key(target, str(direction), float(line_value))
                    _update_bucket(
                        line_bucket_counts[bucket_key],
                        wins=int(line_part["win"].sum()),
                        losses=int(line_part["loss"].sum()),
                        pushes=int(line_part["push"].sum()),
                    )
                    recent_line_part = line_part.loc[line_part["is_recent"]].copy()
                    if not recent_line_part.empty:
                        _update_bucket(
                            recent_line_bucket_counts[bucket_key],
                            wins=int(recent_line_part["win"].sum()),
                            losses=int(recent_line_part["loss"].sum()),
                            pushes=int(recent_line_part["push"].sum()),
                        )

    return {
        "season": int(season),
        "history_dir": report_path(history_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_count": int(len(files)),
        "history_before_date": history_before_date.isoformat() if history_before_date is not None else "",
        "recent_form_lookback_days": int(RECENT_FORM_LOOKBACK_DAYS),
        "recent_form_cutoff_date": recent_cutoff.isoformat() if recent_cutoff is not None else "",
        "target_direction": {key: _finalize_bucket_stats(value) for key, value in sorted(target_direction_counts.items())},
        "line_buckets": {key: _finalize_bucket_stats(value) for key, value in sorted(line_bucket_counts.items())},
        "recent_target_direction": {
            key: _finalize_bucket_stats(value) for key, value in sorted(recent_target_direction_counts.items())
        },
        "recent_line_buckets": {
            key: _finalize_bucket_stats(value) for key, value in sorted(recent_line_bucket_counts.items())
        },
    }


def load_or_build_historical_bucket_priors(
    *,
    history_dir: Path,
    season: int,
    cache_json: Path | None,
    refresh: bool,
    history_before_date: date | None = None,
) -> dict:
    if cache_json is not None and cache_json.exists() and not refresh:
        try:
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
            expected_cutoff = history_before_date.isoformat() if history_before_date is not None else ""
            if (
                int(payload.get("season", season)) == int(season)
                and str(payload.get("history_before_date", "")) == expected_cutoff
            ):
                return payload
        except Exception:
            pass

    payload = build_historical_bucket_priors(history_dir, season, history_before_date)
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


def lookup_recent_bucket_prior(
    calibration: dict | None,
    *,
    target: str,
    direction: str,
    market_line: float,
    min_line_rows: int = RECENT_FORM_MIN_LINE_ROWS,
) -> tuple[str, float, int, str]:
    if not isinstance(calibration, dict):
        return "fallback", 0.5, 0, "fallback"

    line_key = market_bucket_key(target, direction, market_line)
    line_bucket = calibration.get("recent_line_buckets", {}).get(line_key, {})
    line_rows = int(line_bucket.get("graded_rows", 0) or 0)
    if line_rows >= int(max(0, min_line_rows)):
        return line_key, float(line_bucket.get("win_rate", 0.5) or 0.5), line_rows, "recent_line_bucket"

    td_key = target_direction_key(target, direction)
    td_bucket = calibration.get("recent_target_direction", {}).get(td_key, {})
    td_rows = int(td_bucket.get("graded_rows", 0) or 0)
    if td_rows > 0:
        return td_key, float(td_bucket.get("win_rate", 0.5) or 0.5), td_rows, "recent_target_direction"

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


def _empty_availability_stats() -> dict[str, float | int]:
    return {"rows": 0, "side_price_rows": 0, "books_sum": 0.0, "availability_rate": 0.0, "avg_books": 0.0}


def _finalize_availability_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    rows = int(stats.get("rows", 0))
    side_price_rows = int(stats.get("side_price_rows", 0))
    books_sum = float(stats.get("books_sum", 0.0))
    availability_rate = (side_price_rows / rows) if rows else 0.0
    avg_books = (books_sum / rows) if rows else 0.0
    return {
        "rows": rows,
        "side_price_rows": side_price_rows,
        "availability_rate": float(clamp01(availability_rate)),
        "avg_books": float(max(0.0, avg_books)),
    }


def _update_availability_bucket(stats: dict[str, float | int], *, books: int, side_price_confirmed: bool) -> None:
    stats["rows"] = int(stats.get("rows", 0)) + 1
    stats["side_price_rows"] = int(stats.get("side_price_rows", 0)) + int(bool(side_price_confirmed))
    stats["books_sum"] = float(stats.get("books_sum", 0.0)) + float(max(0, int(books)))


def _empty_bet_profile_stats() -> dict[str, float | int]:
    return {
        "rows": 0,
        "graded_rows": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "units_sum": 0.0,
        "win_rate": 0.5,
        "roi_per_bet": 0.0,
    }


def _finalize_bet_profile_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    rows = int(stats.get("rows", 0))
    wins = int(stats.get("wins", 0))
    losses = int(stats.get("losses", 0))
    pushes = int(stats.get("pushes", 0))
    graded_rows = wins + losses
    units_sum = float(stats.get("units_sum", 0.0))
    win_rate = (wins / graded_rows) if graded_rows else 0.5
    roi_per_bet = (units_sum / rows) if rows else 0.0
    return {
        "rows": rows,
        "graded_rows": graded_rows,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "units_sum": units_sum,
        "win_rate": float(clamp01(win_rate)),
        "roi_per_bet": float(roi_per_bet),
    }


def _update_bet_profile_bucket(stats: dict[str, float | int], *, result: str, units: float | None) -> None:
    stats["rows"] = int(stats.get("rows", 0)) + 1
    if result == "win":
        stats["wins"] = int(stats.get("wins", 0)) + 1
    elif result == "loss":
        stats["losses"] = int(stats.get("losses", 0)) + 1
    elif result == "push":
        stats["pushes"] = int(stats.get("pushes", 0)) + 1
    if units is not None:
        stats["units_sum"] = float(stats.get("units_sum", 0.0)) + float(units)


def grade_result(actual: float, market_line: float, direction: str) -> str:
    if direction == "OVER":
        if actual > market_line:
            return "win"
        if actual == market_line:
            return "push"
        return "loss"
    if actual < market_line:
        return "win"
    if actual == market_line:
        return "push"
    return "loss"


def settled_units(result: str, side_price: float | None) -> float | None:
    profit_if_win = american_profit_per_unit(side_price)
    if profit_if_win is None:
        return None
    if result == "win":
        return float(profit_if_win)
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    return None


def build_historical_bet_profile_priors(
    history_dir: Path,
    season: int,
    history_before_date: date | None = None,
) -> dict:
    availability_target_direction: dict[str, dict[str, float | int]] = defaultdict(_empty_availability_stats)
    availability_line_buckets: dict[str, dict[str, float | int]] = defaultdict(_empty_availability_stats)
    bet_profiles_target_probability: dict[str, dict[str, float | int]] = defaultdict(_empty_bet_profile_stats)
    bet_profiles_line_probability: dict[str, dict[str, float | int]] = defaultdict(_empty_bet_profile_stats)
    files = sorted(history_dir.glob(f"*/{int(season)}_processed_processed.csv"))

    required_columns = {"Date"}
    for columns in HISTORICAL_BET_TARGET_SPECS.values():
        required_columns.update(columns)

    for path in files:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in required_columns)
        except Exception:
            continue
        if frame.empty:
            continue
        if history_before_date is not None:
            game_dates = pd.to_datetime(frame["Date"], errors="coerce").dt.date
            frame = frame.loc[game_dates < history_before_date].copy()
        if frame.empty:
            continue

        for target, (actual_col, market_col, gap_col, source_col, books_col, over_col, under_col) in HISTORICAL_BET_TARGET_SPECS.items():
            present_cols = {actual_col, market_col, gap_col, source_col, books_col, over_col, under_col}
            if not present_cols.issubset(frame.columns):
                continue

            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            market_line = pd.to_numeric(frame[market_col], errors="coerce")
            gap = pd.to_numeric(frame[gap_col], errors="coerce")
            books = pd.to_numeric(frame[books_col], errors="coerce").fillna(0)
            over_price = pd.to_numeric(frame[over_col], errors="coerce")
            under_price = pd.to_numeric(frame[under_col], errors="coerce")
            market_source = frame[source_col].astype(str).str.strip().str.lower()

            mask = actual.notna() & market_line.notna() & gap.notna() & gap.ne(0)
            if not bool(mask.any()):
                continue

            sub = pd.DataFrame(
                {
                    "actual": actual.loc[mask],
                    "market_line": market_line.loc[mask],
                    "gap": gap.loc[mask],
                    "books": books.loc[mask],
                    "over_price": over_price.loc[mask],
                    "under_price": under_price.loc[mask],
                    "market_source": market_source.loc[mask],
                }
            )
            sub["direction"] = sub["gap"].gt(0).map({True: "OVER", False: "UNDER"})
            sub["prediction"] = (sub["market_line"] + sub["gap"]).clip(lower=0.0)
            sub["model_graded_hit_rate"] = sub.apply(
                lambda row: estimate_count_hit_probabilities(
                    float(row["prediction"]),
                    float(row["market_line"]),
                    str(row["direction"]),
                )[2],
                axis=1,
            )

            for _, row in sub.iterrows():
                direction = str(row["direction"])
                td_key = target_direction_key(target, direction)
                line_key = market_bucket_key(target, direction, float(row["market_line"]))
                books_value = int(max(0.0, float(row["books"])))
                is_real = str(row["market_source"]) == "real" and books_value > 0
                if not is_real:
                    continue

                selected_side_price = float(row["over_price"]) if direction == "OVER" and pd.notna(row["over_price"]) else None
                if direction == "UNDER" and pd.notna(row["under_price"]):
                    selected_side_price = float(row["under_price"])
                side_price_confirmed = american_implied_probability(selected_side_price) is not None

                _update_availability_bucket(
                    availability_target_direction[td_key],
                    books=books_value,
                    side_price_confirmed=side_price_confirmed,
                )
                _update_availability_bucket(
                    availability_line_buckets[line_key],
                    books=books_value,
                    side_price_confirmed=side_price_confirmed,
                )

                if not side_price_confirmed:
                    continue

                profile_td_key = target_probability_key(target, direction, float(row["model_graded_hit_rate"]))
                profile_line_key = line_probability_key(
                    target,
                    direction,
                    float(row["market_line"]),
                    float(row["model_graded_hit_rate"]),
                )
                result = grade_result(float(row["actual"]), float(row["market_line"]), direction)
                units = settled_units(result, selected_side_price)
                _update_bet_profile_bucket(bet_profiles_target_probability[profile_td_key], result=result, units=units)
                _update_bet_profile_bucket(bet_profiles_line_probability[profile_line_key], result=result, units=units)

    return {
        "season": int(season),
        "history_dir": report_path(history_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_count": int(len(files)),
        "history_before_date": history_before_date.isoformat() if history_before_date is not None else "",
        "availability_target_direction": {
            key: _finalize_availability_stats(value)
            for key, value in sorted(availability_target_direction.items())
        },
        "availability_line_buckets": {
            key: _finalize_availability_stats(value)
            for key, value in sorted(availability_line_buckets.items())
        },
        "bet_profiles_target_probability": {
            key: _finalize_bet_profile_stats(value)
            for key, value in sorted(bet_profiles_target_probability.items())
        },
        "bet_profiles_line_probability": {
            key: _finalize_bet_profile_stats(value)
            for key, value in sorted(bet_profiles_line_probability.items())
        },
    }


def load_or_build_historical_bet_profile_priors(
    *,
    history_dir: Path,
    season: int,
    cache_json: Path | None,
    refresh: bool,
    history_before_date: date | None = None,
) -> dict:
    if cache_json is not None and cache_json.exists() and not refresh:
        try:
            payload = json.loads(cache_json.read_text(encoding="utf-8"))
            expected_cutoff = history_before_date.isoformat() if history_before_date is not None else ""
            if (
                int(payload.get("season", season)) == int(season)
                and str(payload.get("history_before_date", "")) == expected_cutoff
            ):
                return payload
        except Exception:
            pass

    payload = build_historical_bet_profile_priors(history_dir, season, history_before_date)
    if cache_json is not None:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def lookup_historical_bet_profile_prior(
    priors: dict | None,
    *,
    target: str,
    direction: str,
    market_line: float,
    graded_hit_rate: float,
    min_line_rows: int,
) -> tuple[str, float, int, str, float | None]:
    if not isinstance(priors, dict):
        return "fallback", 0.5, 0, "fallback", None

    line_key = line_probability_key(target, direction, market_line, graded_hit_rate)
    line_bucket = priors.get("bet_profiles_line_probability", {}).get(line_key, {})
    line_rows = int(line_bucket.get("rows", 0) or 0)
    if line_rows >= int(max(0, min_line_rows)):
        return (
            line_key,
            float(line_bucket.get("win_rate", 0.5) or 0.5),
            line_rows,
            "line_probability",
            float(line_bucket.get("roi_per_bet", 0.0) or 0.0),
        )

    td_key = target_probability_key(target, direction, graded_hit_rate)
    td_bucket = priors.get("bet_profiles_target_probability", {}).get(td_key, {})
    td_rows = int(td_bucket.get("rows", 0) or 0)
    if td_rows > 0:
        return (
            td_key,
            float(td_bucket.get("win_rate", 0.5) or 0.5),
            td_rows,
            "target_probability",
            float(td_bucket.get("roi_per_bet", 0.0) or 0.0),
        )

    return "fallback", 0.5, 0, "fallback", None


def lookup_historical_market_availability_prior(
    priors: dict | None,
    *,
    target: str,
    direction: str,
    market_line: float,
    min_line_rows: int,
) -> tuple[str, float, int, str, float]:
    if not isinstance(priors, dict):
        return "fallback", 0.0, 0, "fallback", 0.0

    line_key = market_bucket_key(target, direction, market_line)
    line_bucket = priors.get("availability_line_buckets", {}).get(line_key, {})
    line_rows = int(line_bucket.get("rows", 0) or 0)
    if line_rows >= int(max(0, min_line_rows)):
        return (
            line_key,
            float(line_bucket.get("availability_rate", 0.0) or 0.0),
            line_rows,
            "line_bucket",
            float(line_bucket.get("avg_books", 0.0) or 0.0),
        )

    td_key = target_direction_key(target, direction)
    td_bucket = priors.get("availability_target_direction", {}).get(td_key, {})
    td_rows = int(td_bucket.get("rows", 0) or 0)
    if td_rows > 0:
        return (
            td_key,
            float(td_bucket.get("availability_rate", 0.0) or 0.0),
            td_rows,
            "target_direction",
            float(td_bucket.get("avg_books", 0.0) or 0.0),
        )

    return "fallback", 0.0, 0, "fallback", 0.0


def build_candidate_for_direction(
    row: dict[str, str],
    *,
    direction: str,
    calibration: dict | None,
    bet_profile_priors: dict | None = None,
    live_confidence_calibration: dict | None = None,
    min_history_bucket_rows: int,
    max_history_prior_weight: float,
    history_prior_strength: float,
    min_bet_profile_rows: int = 12,
    max_bet_profile_prior_weight: float = 0.25,
    bet_profile_prior_strength: float = 80.0,
    min_market_availability_rows: int = 12,
) -> Candidate | None:
    target = str(row.get("Target", "")).strip().upper()
    if target not in SUPPORTED_COUNT_TARGETS:
        return None

    edge = to_float(row.get("Edge"))
    direction = str(direction or "").strip().upper()
    if direction not in {"OVER", "UNDER"}:
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
    long_run_hit_probability, historical_prior_weight = blend_probability_with_prior(
        model_hit_probability,
        prior_probability=historical_bucket_win_rate,
        support=historical_bucket_support,
        max_weight=max_history_prior_weight,
        strength=history_prior_strength,
    )
    long_run_graded_hit_rate, _ = blend_probability_with_prior(
        model_graded_hit_rate,
        prior_probability=historical_bucket_win_rate,
        support=historical_bucket_support,
        max_weight=max_history_prior_weight,
        strength=history_prior_strength,
    )
    _, recent_bucket_win_rate, recent_bucket_support, recent_prior_source = lookup_recent_bucket_prior(
        calibration,
        target=target,
        direction=direction,
        market_line=market_line,
        min_line_rows=RECENT_FORM_MIN_LINE_ROWS,
    )
    calibrated_hit_probability, recent_prior_weight = blend_probability_with_prior(
        long_run_hit_probability,
        prior_probability=recent_bucket_win_rate,
        support=recent_bucket_support,
        max_weight=RECENT_FORM_MAX_WEIGHT,
        strength=RECENT_FORM_STRENGTH,
    )
    calibrated_graded_hit_rate, _ = blend_probability_with_prior(
        long_run_graded_hit_rate,
        prior_probability=recent_bucket_win_rate,
        support=recent_bucket_support,
        max_weight=RECENT_FORM_MAX_WEIGHT,
        strength=RECENT_FORM_STRENGTH,
    )

    market_books = max(0, to_int(row.get("Market_Books")))
    market_book_keys = str(row.get("Market_Book_Keys", "")).strip().lower()
    market_common_books = max(0, to_int(row.get("Market_Common_Books")))
    market_common_book_keys = str(row.get("Market_Common_Book_Keys", "")).strip().lower()
    market_line_std = max(0.0, to_float(row.get("Market_Line_Std"), default=0.0))
    market_over_price = to_float(row.get("Market_Over_Price"), default=float("nan"))
    if not math.isfinite(market_over_price):
        market_over_price = None
    market_under_price = to_float(row.get("Market_Under_Price"), default=float("nan"))
    if not math.isfinite(market_under_price):
        market_under_price = None
    selected_side_price = market_over_price if direction == "OVER" else market_under_price
    opposite_side_price = market_under_price if direction == "OVER" else market_over_price
    selected_sportsbook_key = str(
        row.get("Market_Over_Book_Key" if direction == "OVER" else "Market_Under_Book_Key", "")
    ).strip().lower()
    selected_sportsbook = str(
        row.get("Market_Over_Book" if direction == "OVER" else "Market_Under_Book", "")
    ).strip()
    price_confirmed = bool(
        american_implied_probability(selected_side_price) is not None
        and selected_sportsbook_key
        and selected_sportsbook
    )
    market_implied_probability = no_vig_side_probability(selected_side_price, opposite_side_price)

    (
        historical_bet_profile_key,
        historical_bet_profile_win_rate,
        historical_bet_profile_support,
        historical_bet_profile_source,
        historical_bet_profile_roi,
    ) = lookup_historical_bet_profile_prior(
        bet_profile_priors,
        target=target,
        direction=direction,
        market_line=market_line,
        graded_hit_rate=calibrated_graded_hit_rate,
        min_line_rows=min_bet_profile_rows,
    )
    (
        validated_graded_hit_rate,
        historical_bet_profile_prior_weight,
    ) = blend_probability_with_prior(
        calibrated_graded_hit_rate,
        prior_probability=historical_bet_profile_win_rate,
        support=historical_bet_profile_support,
        max_weight=max_bet_profile_prior_weight,
        strength=bet_profile_prior_strength,
    )
    (
        validated_graded_hit_rate,
        live_confidence_calibration_key,
        live_confidence_calibration_support,
        live_confidence_calibration_adjustment,
    ) = apply_live_board_calibration(
        validated_graded_hit_rate,
        live_confidence_calibration,
        target=target,
        direction=direction,
    )
    validated_graded_hit_rate = min(MAX_CALIBRATED_PROBABILITY, validated_graded_hit_rate)
    calibrated_hit_probability = min(
        MAX_CALIBRATED_PROBABILITY,
        validated_graded_hit_rate * max(0.0, 1.0 - push_probability),
    )
    (
        historical_market_availability_key,
        historical_market_availability_rate,
        historical_market_availability_support,
        historical_market_availability_source,
        historical_market_avg_books,
    ) = lookup_historical_market_availability_prior(
        bet_profile_priors,
        target=target,
        direction=direction,
        market_line=market_line,
        min_line_rows=min_market_availability_rows,
    )

    expected_value_per_unit = expected_profit_per_unit(validated_graded_hit_rate, selected_side_price)

    history_score = min(history_rows / 18.0, 1.0)
    recency_score = 0.0 if days_since_history is None else max(0.0, 1.0 - (days_since_history / 7.0))
    edge_over_mae = max(0.0, directional_model_gap(prediction, market_line, direction)) / max(model_val_mae, 0.1)
    bucket_support_score = min(max(float(historical_bucket_support), 0.0) / 500.0, 1.0)
    bet_profile_support_score = min(max(float(historical_bet_profile_support), 0.0) / 120.0, 1.0)
    recent_support_score = min(max(float(recent_bucket_support), 0.0) / 60.0, 1.0)
    recent_form_lift = 0.0
    if recent_prior_source != "fallback":
        recent_form_lift = max(-0.08, min(0.08, recent_bucket_win_rate - historical_bucket_win_rate))

    reliability_core = (
        0.49 * calibrated_hit_probability
        + 0.17 * validated_graded_hit_rate
        + 0.10 * history_score
        + 0.08 * recency_score
        + 0.07 * bucket_support_score
        + 0.04 * bet_profile_support_score
        + 0.03 * recent_support_score
        + 0.03 * max(0.0, historical_bucket_win_rate - 0.50)
        + 0.02 * max(0.0, historical_bet_profile_win_rate - 0.50)
        + 0.05 * recent_form_lift
    )
    selection_score = reliability_core * (1.0 + 0.08 * min(edge_over_mae, 3.0)) * (1.0 - 0.55 * push_probability)
    if recent_prior_source != "fallback":
        selection_score *= 1.0 + (0.30 * recent_prior_weight * recent_form_lift)
    if market_source == "real":
        books_score = clamp01(market_books / 5.0)
        common_books_score = clamp01(market_common_books / 3.0)
        consensus_score = clamp01(1.0 - (market_line_std / 1.5)) if market_books > 1 else 0.5 if market_books == 1 else 0.0
        ev_bonus = clamp01((((expected_value_per_unit or 0.0) + 0.05) / 0.20)) if expected_value_per_unit is not None else 0.0
        availability_bonus = clamp01(historical_market_availability_rate)
        roi_bonus = (
            clamp01(((float(historical_bet_profile_roi) + 0.08) / 0.24))
            if historical_bet_profile_roi is not None
            else 0.0
        )
        price_confirmed_bonus = 1.0 if price_confirmed else 0.0
        selection_score *= (
            0.88
            + (0.03 * books_score)
            + (0.03 * consensus_score)
            + (0.04 * ev_bonus)
            + (0.04 * availability_bonus)
            + (0.03 * roi_bonus)
            + (0.03 * price_confirmed_bonus)
            + (0.04 * common_books_score)
        )
        if not price_confirmed:
            selection_score *= 0.88 + (0.12 * availability_bonus)
    selection_score = max(0.0, float(selection_score))

    return Candidate(
        raw=row,
        player=str(row.get("Player", "")).strip(),
        player_id=str(row.get("Player_ID", "")).strip(),
        team=str(row.get("Team", "")).strip(),
        game_id=str(row.get("Game_ID", "")).strip(),
        target=target,
        direction=direction,
        original_direction=infer_direction(edge) or direction,
        direction_flip_applied=bool((infer_direction(edge) or direction) != direction),
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
        calibrated_graded_hit_rate=validated_graded_hit_rate,
        live_confidence_calibration_key=live_confidence_calibration_key,
        live_confidence_calibration_support=live_confidence_calibration_support,
        live_confidence_calibration_adjustment=live_confidence_calibration_adjustment,
        market_books=market_books,
        market_book_keys=market_book_keys,
        market_common_books=market_common_books,
        market_common_book_keys=market_common_book_keys,
        market_line_std=market_line_std,
        market_over_price=market_over_price,
        market_under_price=market_under_price,
        selected_side_price=selected_side_price,
        opposite_side_price=opposite_side_price,
        selected_sportsbook_key=selected_sportsbook_key,
        selected_sportsbook=selected_sportsbook,
        price_confirmed=price_confirmed,
        market_implied_probability=market_implied_probability,
        expected_value_per_unit=expected_value_per_unit,
        historical_bet_profile_key=historical_bet_profile_key,
        historical_bet_profile_source=historical_bet_profile_source,
        historical_bet_profile_win_rate=historical_bet_profile_win_rate,
        historical_bet_profile_support=historical_bet_profile_support,
        historical_bet_profile_roi=historical_bet_profile_roi,
        historical_bet_profile_prior_weight=historical_bet_profile_prior_weight,
        historical_market_availability_key=historical_market_availability_key,
        historical_market_availability_source=historical_market_availability_source,
        historical_market_availability_rate=historical_market_availability_rate,
        historical_market_availability_support=historical_market_availability_support,
        historical_market_avg_books=historical_market_avg_books,
        edge_over_mae=edge_over_mae,
        history_score=history_score,
        recency_score=recency_score,
        bucket_support_score=bucket_support_score,
        precision_score=selection_score,
        selection_score=selection_score,
        confidence_tier=confidence_tier(selection_score),
        market_bucket=market_bucket_key(target, direction, market_line),
    )


def build_candidate(
    row: dict[str, str],
    *,
    calibration: dict | None,
    bet_profile_priors: dict | None = None,
    live_confidence_calibration: dict | None = None,
    min_history_bucket_rows: int,
    max_history_prior_weight: float,
    history_prior_strength: float,
    min_bet_profile_rows: int = 12,
    max_bet_profile_prior_weight: float = 0.25,
    bet_profile_prior_strength: float = 80.0,
    min_market_availability_rows: int = 12,
    prefer_confident_side: bool = False,
) -> Candidate | None:
    edge = to_float(row.get("Edge"))
    primary_direction = infer_direction(edge)
    if primary_direction is None:
        return None

    primary = build_candidate_for_direction(
        row,
        direction=primary_direction,
        calibration=calibration,
        bet_profile_priors=bet_profile_priors,
        live_confidence_calibration=live_confidence_calibration,
        min_history_bucket_rows=min_history_bucket_rows,
        max_history_prior_weight=max_history_prior_weight,
        history_prior_strength=history_prior_strength,
        min_bet_profile_rows=min_bet_profile_rows,
        max_bet_profile_prior_weight=max_bet_profile_prior_weight,
        bet_profile_prior_strength=bet_profile_prior_strength,
        min_market_availability_rows=min_market_availability_rows,
    )
    if primary is None or not prefer_confident_side:
        return primary

    if primary.market_source == "real" and primary.price_confirmed:
        return primary

    alternate_direction = "UNDER" if primary_direction == "OVER" else "OVER"
    alternate = build_candidate_for_direction(
        row,
        direction=alternate_direction,
        calibration=calibration,
        bet_profile_priors=bet_profile_priors,
        live_confidence_calibration=live_confidence_calibration,
        min_history_bucket_rows=min_history_bucket_rows,
        max_history_prior_weight=max_history_prior_weight,
        history_prior_strength=history_prior_strength,
        min_bet_profile_rows=min_bet_profile_rows,
        max_bet_profile_prior_weight=max_bet_profile_prior_weight,
        bet_profile_prior_strength=bet_profile_prior_strength,
        min_market_availability_rows=min_market_availability_rows,
    )
    if alternate is None:
        return primary

    if directional_model_gap(alternate.prediction, alternate.market_line, alternate.direction) < -0.15:
        return primary

    primary_rank = (
        primary.calibrated_graded_hit_rate,
        primary.calibrated_hit_probability,
        primary.selection_score,
        primary.historical_bet_profile_win_rate,
        primary.historical_market_availability_rate,
    )
    alternate_rank = (
        alternate.calibrated_graded_hit_rate,
        alternate.calibrated_hit_probability,
        alternate.selection_score,
        alternate.historical_bet_profile_win_rate,
        alternate.historical_market_availability_rate,
    )
    if alternate_rank <= primary_rank:
        return primary

    graded_edge = alternate.calibrated_graded_hit_rate - primary.calibrated_graded_hit_rate
    score_edge = alternate.selection_score - primary.selection_score
    if graded_edge < 0.015 and score_edge < 0.03:
        return primary
    return alternate


def load_candidates(
    pool_csv: Path,
    *,
    calibration: dict | None,
    bet_profile_priors: dict | None = None,
    live_confidence_calibration: dict | None = None,
    min_history_bucket_rows: int,
    max_history_prior_weight: float,
    history_prior_strength: float,
    min_bet_profile_rows: int = 12,
    max_bet_profile_prior_weight: float = 0.25,
    bet_profile_prior_strength: float = 80.0,
    min_market_availability_rows: int = 12,
    prefer_confident_side: bool = False,
    pick_survival_model: dict | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    with open(pool_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = build_candidate(
                row,
                calibration=calibration,
                bet_profile_priors=bet_profile_priors,
                live_confidence_calibration=live_confidence_calibration,
                min_history_bucket_rows=min_history_bucket_rows,
                max_history_prior_weight=max_history_prior_weight,
                history_prior_strength=history_prior_strength,
                min_bet_profile_rows=min_bet_profile_rows,
                max_bet_profile_prior_weight=max_bet_profile_prior_weight,
                bet_profile_prior_strength=bet_profile_prior_strength,
                min_market_availability_rows=min_market_availability_rows,
                prefer_confident_side=prefer_confident_side,
            )
            if candidate is not None:
                (
                    candidate.survival_probability,
                    candidate.survival_expected_value,
                    candidate.survival_model_status,
                    candidate.survival_model_support,
                    candidate.survival_rank_active,
                ) = apply_pick_survival_model(candidate, pick_survival_model)
                candidates.append(candidate)
    return candidates


def filter_candidates(candidates: Iterable[Candidate], args: argparse.Namespace) -> tuple[list[Candidate], Counter]:
    allowed_targets = {str(value).strip().upper() for value in args.targets}
    optimized_over_targets = {
        str(value).strip().upper() for value in getattr(args, "optimized_over_targets", []) if str(value).strip()
    }
    rejected = Counter()
    kept: list[Candidate] = []

    for candidate in candidates:
        row = candidate.raw
        use_optimized_over_profile = candidate.direction == "OVER" and candidate.target in optimized_over_targets
        use_pitcher_k_over_profile = bool(getattr(args, "enable_pitcher_k_over_profile", False)) and bool(
            candidate.direction == "OVER"
            and candidate.target == "K"
            and str(row.get("Player_Type", "")).strip().lower() == "pitcher"
        )
        if candidate.target not in allowed_targets:
            rejected["unsupported_target"] += 1
            continue
        if candidate.market_source == "real" and not is_standard_bettable_line(candidate.target, candidate.market_line):
            rejected["nonstandard_market_line"] += 1
            continue
        if not args.allow_baseline and candidate.model_selected.lower() == "baseline":
            rejected["baseline_model"] += 1
            continue
        if not is_upcoming_status(row.get("Game_Status_Code", ""), row.get("Game_Status_Detail", "")):
            rejected["non_upcoming_status"] += 1
            continue
        min_abs_edge = float(args.min_abs_edge)
        if use_optimized_over_profile and getattr(args, "over_min_abs_edge", None) is not None:
            min_abs_edge = float(args.over_min_abs_edge)
        if use_pitcher_k_over_profile:
            min_abs_edge = float(args.pitcher_k_min_abs_edge)
        if candidate.abs_edge < min_abs_edge:
            rejected["edge_too_small"] += 1
            continue
        if (
            use_optimized_over_profile
            and getattr(args, "over_max_abs_edge", None) is not None
            and candidate.abs_edge > float(args.over_max_abs_edge)
        ):
            rejected["optimized_over_edge_too_large"] += 1
            continue
        if use_pitcher_k_over_profile and candidate.abs_edge > float(args.pitcher_k_max_abs_edge):
            rejected["pitcher_k_edge_too_large"] += 1
            continue
        min_history_rows = (
            int(args.pitcher_k_min_starter_history)
            if use_pitcher_k_over_profile
            else int(args.min_history_rows)
        )
        if candidate.history_rows < min_history_rows:
            rejected["history_too_short"] += 1
            continue
        if (
            use_optimized_over_profile
            and getattr(args, "over_min_history_rows", None) is not None
            and candidate.history_rows < int(args.over_min_history_rows)
        ):
            rejected["optimized_over_history_too_short"] += 1
            continue
        if use_pitcher_k_over_profile:
            if str(row.get("Starter_Confirmed", "")).strip().lower() not in {"1", "true", "yes"}:
                rejected["pitcher_starter_unconfirmed"] += 1
                continue
            if int(to_float(row.get("Starter_History_Rows"), 0.0) or 0) < int(args.pitcher_k_min_starter_history):
                rejected["pitcher_starter_history_too_short"] += 1
                continue
            if float(to_float(row.get("Projected_IP"), 0.0) or 0.0) < float(args.pitcher_k_min_projected_ip):
                rejected["pitcher_projected_ip_too_low"] += 1
                continue
            if float(to_float(row.get("Projected_Pitches"), 0.0) or 0.0) < float(args.pitcher_k_min_projected_pitches):
                rejected["pitcher_projected_pitches_too_low"] += 1
                continue
        if candidate.prediction < float(args.min_prediction):
            rejected["prediction_too_low"] += 1
            continue
        if args.require_real_market_source and candidate.market_source != "real":
            rejected["synthetic_market_source"] += 1
            continue
        if candidate.market_source != "real" and candidate.target not in {"H", "TB", "R", "K"}:
            rejected["non_core_synthetic_market"] += 1
            continue
        if candidate.market_source != "real" and candidate.direction == "UNDER" and not bool(args.allow_synthetic_unders):
            rejected["synthetic_under_not_actionable"] += 1
            continue
        if int(args.min_market_books) > 0 and candidate.market_source != "real":
            rejected["synthetic_market_source"] += 1
            continue
        if candidate.market_source == "real" and int(args.min_market_books) > 0 and candidate.market_books < int(args.min_market_books):
            rejected["market_books_too_low"] += 1
            continue
        if (
            candidate.market_source == "real"
            and int(getattr(args, "min_common_market_books", 0)) > 0
            and candidate.market_common_books < int(args.min_common_market_books)
        ):
            rejected["common_market_books_too_low"] += 1
            continue
        if candidate.market_source == "real" and not candidate.price_confirmed and not bool(getattr(args, "allow_unpriced_side", False)):
            rejected["side_price_unconfirmed"] += 1
            continue
        if (
            use_optimized_over_profile
            and getattr(args, "over_max_american_price", None) is not None
            and candidate.selected_side_price is not None
            and candidate.selected_side_price > float(args.over_max_american_price)
        ):
            rejected["optimized_over_price_too_long"] += 1
            continue
        if (
            not use_optimized_over_profile
            and not use_pitcher_k_over_profile
            and getattr(args, "core_max_american_price", None) is not None
            and candidate.selected_side_price is not None
            and candidate.selected_side_price > float(args.core_max_american_price)
        ):
            rejected["core_price_too_long"] += 1
            continue
        if (
            not use_optimized_over_profile
            and not use_pitcher_k_over_profile
            and getattr(args, "core_min_american_price", None) is not None
            and candidate.selected_side_price is not None
            and candidate.selected_side_price < float(args.core_min_american_price)
        ):
            rejected["core_price_too_heavily_juiced"] += 1
            continue
        if use_pitcher_k_over_profile and candidate.selected_side_price is not None:
            if candidate.selected_side_price < float(args.pitcher_k_min_american_price):
                rejected["pitcher_k_price_too_heavily_juiced"] += 1
                continue
            if candidate.selected_side_price > float(args.pitcher_k_max_american_price):
                rejected["pitcher_k_price_too_long"] += 1
                continue
        if (
            candidate.market_source == "real"
            and float(args.max_market_line_std) > 0.0
            and candidate.market_books > 1
            and candidate.market_line_std > float(args.max_market_line_std)
        ):
            rejected["market_line_too_volatile"] += 1
            continue
        hit_probability = candidate.calibrated_hit_probability
        min_hit_probability = float(args.min_hit_probability)
        if use_optimized_over_profile:
            hit_probability = candidate.model_hit_probability
            if getattr(args, "over_min_model_hit_probability", None) is not None:
                min_hit_probability = float(args.over_min_model_hit_probability)
        elif use_pitcher_k_over_profile:
            hit_probability = candidate.model_hit_probability
            min_hit_probability = float(args.pitcher_k_min_model_hit_probability)
        if hit_probability < min_hit_probability:
            rejected["hit_probability_too_low"] += 1
            continue
        if (
            use_optimized_over_profile
            and getattr(args, "over_max_model_hit_probability", None) is not None
            and hit_probability > float(args.over_max_model_hit_probability)
        ):
            rejected["optimized_over_hit_probability_too_high"] += 1
            continue
        if use_pitcher_k_over_profile and hit_probability > float(args.pitcher_k_max_model_hit_probability):
            rejected["pitcher_k_hit_probability_too_high"] += 1
            continue
        graded_hit_rate = (
            candidate.model_graded_hit_rate
            if use_optimized_over_profile or use_pitcher_k_over_profile
            else candidate.calibrated_graded_hit_rate
        )
        min_graded_hit_rate = (
            float(getattr(args, "over_min_model_hit_probability"))
            if use_optimized_over_profile and getattr(args, "over_min_model_hit_probability", None) is not None
            else float(args.pitcher_k_min_model_hit_probability)
            if use_pitcher_k_over_profile
            else float(args.min_graded_hit_rate)
        )
        if graded_hit_rate < min_graded_hit_rate:
            rejected["graded_hit_rate_too_low"] += 1
            continue
        if (
            use_optimized_over_profile
            and getattr(args, "over_max_model_hit_probability", None) is not None
            and graded_hit_rate > float(args.over_max_model_hit_probability)
        ):
            rejected["optimized_over_graded_hit_rate_too_high"] += 1
            continue
        if use_pitcher_k_over_profile and graded_hit_rate > float(args.pitcher_k_max_model_hit_probability):
            rejected["pitcher_k_graded_hit_rate_too_high"] += 1
            continue
        min_expected_value = float(args.min_expected_value)
        if use_optimized_over_profile and getattr(args, "over_min_expected_value", None) is not None:
            min_expected_value = float(args.over_min_expected_value)
        elif use_pitcher_k_over_profile:
            min_expected_value = float(args.pitcher_k_min_expected_value)
        if (
            candidate.market_source == "real"
            and min_expected_value > -0.99
            and candidate.expected_value_per_unit is not None
            and candidate.expected_value_per_unit < min_expected_value
        ):
            rejected["expected_value_too_low"] += 1
            continue
        if candidate.push_probability > float(args.max_push_probability):
            rejected["push_probability_too_high"] += 1
            continue
        max_days_since_history = (
            int(args.pitcher_k_max_days_since_history)
            if use_pitcher_k_over_profile
            else int(args.max_days_since_history)
        )
        if candidate.days_since_history is None or candidate.days_since_history > max_days_since_history:
            rejected["history_too_stale"] += 1
            continue
        if (
            int(args.min_historical_bet_profile_support) > 0
            and candidate.historical_bet_profile_support < int(args.min_historical_bet_profile_support)
        ):
            rejected["historical_bet_profile_support_too_low"] += 1
            continue
        if (
            float(args.min_historical_bet_profile_win_rate) > 0.0
            and candidate.historical_bet_profile_support > 0
            and candidate.historical_bet_profile_win_rate < float(args.min_historical_bet_profile_win_rate)
        ):
            rejected["historical_bet_profile_win_rate_too_low"] += 1
            continue
        if (
            int(args.min_historical_market_availability_support) > 0
            and candidate.historical_market_availability_support < int(args.min_historical_market_availability_support)
        ):
            rejected["historical_market_availability_support_too_low"] += 1
            continue
        if candidate.historical_market_availability_rate < float(args.min_historical_market_availability_rate):
            rejected["historical_market_availability_rate_too_low"] += 1
            continue
        kept.append(candidate)

    return kept, rejected


def select_top_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    optimized_over_targets = {
        str(value).strip().upper()
        for value in getattr(args, "optimized_over_targets", [])
        if str(value).strip()
    }
    ordered = sorted(
        candidates,
        key=lambda row: (
            round(row.selection_score, 2) if row.survival_rank_active else row.selection_score,
            (
                row.survival_probability
                if row.survival_rank_active and row.survival_probability is not None
                else row.selection_score
            ),
            row.selection_score,
            (row.expected_value_per_unit if row.expected_value_per_unit is not None else -999.0),
            1.0 if row.price_confirmed else 0.0,
            row.market_books,
            row.market_common_books,
            row.historical_market_availability_rate,
            row.historical_bet_profile_win_rate,
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
    by_direction: Counter[str] = Counter()
    by_selection_profile: Counter[str] = Counter()
    selected_prop_keys: set[tuple[str, ...]] = set()

    def try_add(candidate: Candidate) -> bool:
        if len(selected) >= int(args.top_n):
            return False
        daily_pick_soft_cap = max(0, int(getattr(args, "daily_pick_soft_cap", 0)))
        post_cap_min_selection_score = max(
            0.0,
            float(getattr(args, "post_cap_min_selection_score", 0.0)),
        )
        if (
            daily_pick_soft_cap > 0
            and len(selected) >= daily_pick_soft_cap
            and candidate.selection_score < post_cap_min_selection_score
        ):
            return False
        prop_key = (
            str(candidate.player_id or candidate.player).strip().lower(),
            str(candidate.team).strip().upper(),
            str(candidate.raw.get("Opponent", "")).strip().upper(),
            str(candidate.raw.get("Game_Date", "")).strip(),
            str(candidate.target).strip().upper(),
            str(candidate.direction).strip().upper(),
            f"{float(candidate.market_line):.3f}",
        )
        if prop_key in selected_prop_keys:
            return False
        if by_player[candidate.player_id or candidate.player] >= int(args.max_per_player):
            return False
        if by_game[candidate.game_id] >= int(args.max_per_game):
            return False
        if by_team[candidate.team] >= int(args.max_per_team):
            return False
        market_bucket_cap = int(args.max_per_market_bucket)
        if (
            candidate.direction == "OVER"
            and candidate.target in optimized_over_targets
            and getattr(args, "optimized_over_max_per_market_bucket", None) is not None
        ):
            market_bucket_cap = int(args.optimized_over_max_per_market_bucket)
        if market_bucket_cap > 0 and by_market_bucket[candidate.market_bucket] >= market_bucket_cap:
            return False
        max_over_picks = max(0, int(getattr(args, "max_over_picks", 0)))
        if candidate.direction == "OVER" and max_over_picks > 0 and by_direction["OVER"] >= max_over_picks:
            return False
        max_under_picks = max(0, int(getattr(args, "max_under_picks", 0)))
        if candidate.direction == "UNDER" and max_under_picks > 0 and by_direction["UNDER"] >= max_under_picks:
            return False
        profile = candidate_selection_profile(candidate, args)
        max_pitcher_k_picks = max(0, int(getattr(args, "max_pitcher_k_picks", 0)))
        if (
            profile == PITCHER_K_OVER_SELECTION_PROFILE
            and max_pitcher_k_picks > 0
            and by_selection_profile[profile] >= max_pitcher_k_picks
        ):
            return False

        selected.append(candidate)
        selected_prop_keys.add(prop_key)
        by_player[candidate.player_id or candidate.player] += 1
        by_game[candidate.game_id] += 1
        by_team[candidate.team] += 1
        by_market_bucket[candidate.market_bucket] += 1
        by_direction[candidate.direction] += 1
        by_selection_profile[profile] += 1
        return True

    min_over_picks = min(max(0, int(getattr(args, "min_over_picks", 0))), int(args.top_n))
    if min_over_picks > 0:
        for candidate in ordered:
            if candidate.direction != "OVER":
                continue
            try_add(candidate)
            if by_direction["OVER"] >= min_over_picks or len(selected) >= int(args.top_n):
                break

    for candidate in ordered:
        try_add(candidate)

        if len(selected) >= int(args.top_n):
            break

    return selected


def candidate_selection_profile(candidate: Candidate, args: argparse.Namespace) -> str:
    if (
        bool(getattr(args, "enable_pitcher_k_over_profile", False))
        and candidate.direction == "OVER"
        and candidate.target == "K"
        and str(candidate.raw.get("Player_Type", "")).strip().lower() == "pitcher"
    ):
        return PITCHER_K_OVER_SELECTION_PROFILE
    optimized_targets = {
        str(value).strip().upper() for value in getattr(args, "optimized_over_targets", []) if str(value).strip()
    }
    if candidate.direction == "OVER" and candidate.target in optimized_targets:
        return OPTIMIZED_OVER_SELECTION_PROFILE
    return CORE_SELECTION_PROFILE


def write_selected_csv(path: Path, selected: list[Candidate], args: argparse.Namespace) -> None:
    fieldnames = [
        "Rank",
        "Selection_Profile",
        "Prediction_Run_Date",
        "Game_Date",
        "Commence_Time_UTC",
        "Game_ID",
        "Game_Status_Code",
        "Player",
        "Player_ID",
        "Player_Type",
        "Opposing_Pitcher_ID",
        "Opposing_Pitcher",
        "Matchup_Network_Version",
        "Batter_Profile_Strength",
        "Pitcher_Profile_Vulnerability",
        "Pitcher_Profile_Uncertainty",
        "Batter_Vs_Starter_Games",
        "Batter_Vs_Starter_Lift",
        "Archetype_Neighbor_Games",
        "Archetype_Neighbor_Effective_Support",
        "Archetype_Neighbor_Lift",
        "Matchup_Network_Score",
        "Matchup_Network_Confidence",
        "Matchup_Network_Adjustment",
        "Starter_Confirmed",
        "Starter_History_Rows",
        "Projected_IP",
        "Projected_Pitches",
        "Team",
        "Opponent",
        "Is_Home",
        "Target",
        "Direction",
        "Prediction",
        "Market_Line",
        "Market_Source",
        "Original_Direction",
        "Direction_Flip_Applied",
        "Market_Books",
        "Market_Book_Keys",
        "Market_Common_Books",
        "Market_Common_Book_Keys",
        "Market_Line_Std",
        "Market_Over_Price",
        "Market_Under_Price",
        "Selected_Side_Price",
        "Opposite_Side_Price",
        "Selected_Sportsbook_Key",
        "Selected_Sportsbook",
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
        "Live_Confidence_Calibration_Key",
        "Live_Confidence_Calibration_Support",
        "Live_Confidence_Calibration_Adjustment",
        "Pick_Survival_Probability_Shadow",
        "Pick_Survival_EV_Shadow",
        "Pick_Survival_Model_Status",
        "Pick_Survival_Model_Support",
        "Pick_Survival_Rank_Active",
        "Market_Implied_Probability",
        "Expected_Value_Per_Unit",
        "Price_Confirmed",
        "Historical_Bet_Profile_Key",
        "Historical_Bet_Profile_Source",
        "Historical_Bet_Profile_Win_Rate",
        "Historical_Bet_Profile_Support",
        "Historical_Bet_Profile_ROI",
        "Historical_Bet_Profile_Prior_Weight",
        "Historical_Market_Availability_Key",
        "Historical_Market_Availability_Source",
        "Historical_Market_Availability_Rate",
        "Historical_Market_Availability_Support",
        "Historical_Market_Avg_Books",
        "Edge_Over_MAE",
        "Precision_Score",
        "Selection_Score",
        "Confidence_Tier",
        "Market_Bucket",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for idx, candidate in enumerate(selected, start=1):
            writer.writerow(
                {
                    "Rank": idx,
                    "Selection_Profile": candidate_selection_profile(candidate, args),
                    "Prediction_Run_Date": candidate.raw.get("Prediction_Run_Date", ""),
                    "Game_Date": candidate.raw.get("Game_Date", ""),
                    "Commence_Time_UTC": candidate.raw.get("Commence_Time_UTC", ""),
                    "Game_ID": candidate.game_id,
                    "Game_Status_Code": candidate.game_status_code,
                    "Player": candidate.player,
                    "Player_ID": candidate.player_id,
                    "Player_Type": candidate.raw.get("Player_Type", ""),
                    "Opposing_Pitcher_ID": candidate.raw.get("Opposing_Pitcher_ID", ""),
                    "Opposing_Pitcher": candidate.raw.get("Opposing_Pitcher", ""),
                    "Matchup_Network_Version": candidate.raw.get("Matchup_Network_Version", ""),
                    "Batter_Profile_Strength": candidate.raw.get("Batter_Profile_Strength", ""),
                    "Pitcher_Profile_Vulnerability": candidate.raw.get("Pitcher_Profile_Vulnerability", ""),
                    "Pitcher_Profile_Uncertainty": candidate.raw.get("Pitcher_Profile_Uncertainty", ""),
                    "Batter_Vs_Starter_Games": candidate.raw.get("Batter_Vs_Starter_Games", ""),
                    "Batter_Vs_Starter_Lift": candidate.raw.get("Batter_Vs_Starter_Lift", ""),
                    "Archetype_Neighbor_Games": candidate.raw.get("Archetype_Neighbor_Games", ""),
                    "Archetype_Neighbor_Effective_Support": candidate.raw.get("Archetype_Neighbor_Effective_Support", ""),
                    "Archetype_Neighbor_Lift": candidate.raw.get("Archetype_Neighbor_Lift", ""),
                    "Matchup_Network_Score": candidate.raw.get("Matchup_Network_Score", ""),
                    "Matchup_Network_Confidence": candidate.raw.get("Matchup_Network_Confidence", ""),
                    "Matchup_Network_Adjustment": candidate.raw.get("Matchup_Network_Adjustment", ""),
                    "Starter_Confirmed": candidate.raw.get("Starter_Confirmed", ""),
                    "Starter_History_Rows": candidate.raw.get("Starter_History_Rows", ""),
                    "Projected_IP": candidate.raw.get("Projected_IP", ""),
                    "Projected_Pitches": candidate.raw.get("Projected_Pitches", ""),
                    "Team": candidate.team,
                    "Opponent": candidate.raw.get("Opponent", ""),
                    "Is_Home": candidate.raw.get("Is_Home", ""),
                    "Target": candidate.target,
                    "Direction": candidate.direction,
                    "Prediction": f"{candidate.prediction:.6f}",
                    "Market_Line": f"{candidate.market_line:.6f}",
                    "Market_Source": candidate.market_source,
                    "Original_Direction": candidate.original_direction,
                    "Direction_Flip_Applied": int(candidate.direction_flip_applied),
                    "Market_Books": candidate.market_books,
                    "Market_Book_Keys": candidate.market_book_keys,
                    "Market_Common_Books": candidate.market_common_books,
                    "Market_Common_Book_Keys": candidate.market_common_book_keys,
                    "Market_Line_Std": f"{candidate.market_line_std:.6f}",
                    "Market_Over_Price": "" if candidate.market_over_price is None else f"{candidate.market_over_price:.6f}",
                    "Market_Under_Price": "" if candidate.market_under_price is None else f"{candidate.market_under_price:.6f}",
                    "Selected_Side_Price": "" if candidate.selected_side_price is None else f"{candidate.selected_side_price:.6f}",
                    "Opposite_Side_Price": "" if candidate.opposite_side_price is None else f"{candidate.opposite_side_price:.6f}",
                    "Selected_Sportsbook_Key": candidate.selected_sportsbook_key,
                    "Selected_Sportsbook": candidate.selected_sportsbook,
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
                    "Live_Confidence_Calibration_Key": candidate.live_confidence_calibration_key,
                    "Live_Confidence_Calibration_Support": candidate.live_confidence_calibration_support,
                    "Live_Confidence_Calibration_Adjustment": f"{candidate.live_confidence_calibration_adjustment:.6f}",
                    "Pick_Survival_Probability_Shadow": "" if candidate.survival_probability is None else f"{candidate.survival_probability:.6f}",
                    "Pick_Survival_EV_Shadow": "" if candidate.survival_expected_value is None else f"{candidate.survival_expected_value:.6f}",
                    "Pick_Survival_Model_Status": candidate.survival_model_status,
                    "Pick_Survival_Model_Support": candidate.survival_model_support,
                    "Pick_Survival_Rank_Active": int(candidate.survival_rank_active),
                    "Market_Implied_Probability": "" if candidate.market_implied_probability is None else f"{candidate.market_implied_probability:.6f}",
                    "Expected_Value_Per_Unit": "" if candidate.expected_value_per_unit is None else f"{candidate.expected_value_per_unit:.6f}",
                    "Price_Confirmed": int(candidate.price_confirmed),
                    "Historical_Bet_Profile_Key": candidate.historical_bet_profile_key,
                    "Historical_Bet_Profile_Source": candidate.historical_bet_profile_source,
                    "Historical_Bet_Profile_Win_Rate": f"{candidate.historical_bet_profile_win_rate:.6f}",
                    "Historical_Bet_Profile_Support": candidate.historical_bet_profile_support,
                    "Historical_Bet_Profile_ROI": "" if candidate.historical_bet_profile_roi is None else f"{candidate.historical_bet_profile_roi:.6f}",
                    "Historical_Bet_Profile_Prior_Weight": f"{candidate.historical_bet_profile_prior_weight:.6f}",
                    "Historical_Market_Availability_Key": candidate.historical_market_availability_key,
                    "Historical_Market_Availability_Source": candidate.historical_market_availability_source,
                    "Historical_Market_Availability_Rate": f"{candidate.historical_market_availability_rate:.6f}",
                    "Historical_Market_Availability_Support": candidate.historical_market_availability_support,
                    "Historical_Market_Avg_Books": f"{candidate.historical_market_avg_books:.6f}",
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
    bet_profile_priors: dict | None,
    live_confidence_calibration: dict | None,
    pick_survival_model: dict | None,
) -> None:
    by_target = Counter(candidate.target for candidate in selected)
    by_direction = Counter(candidate.direction for candidate in selected)
    by_team = Counter(candidate.team for candidate in selected)
    by_market_bucket = Counter(candidate.market_bucket for candidate in selected)
    summary = {
        "pool_csv": report_path(pool_csv),
        "out_csv": report_path(args.out_csv or default_output_paths(pool_csv)[0]),
        "rows_supported": total_candidates,
        "rows_after_filters": len(eligible_candidates),
        "rows_selected": len(selected),
        "selection": {
            "matchup_network_enabled": True,
            "matchup_network_version": NETWORK_VERSION,
            "top_n": int(args.top_n),
            "daily_pick_soft_cap": int(getattr(args, "daily_pick_soft_cap", 0)),
            "post_cap_min_selection_score": float(
                getattr(args, "post_cap_min_selection_score", 0.0)
            ),
            "min_abs_edge": float(args.min_abs_edge),
            "min_history_rows": int(args.min_history_rows),
            "min_prediction": float(args.min_prediction),
            "min_hit_probability": float(args.min_hit_probability),
            "min_graded_hit_rate": float(args.min_graded_hit_rate),
            "optimized_over_profile": OPTIMIZED_OVER_SELECTION_PROFILE,
            "optimized_over_profile_status": OPTIMIZED_OVER_PROFILE_STATUS,
            "pitcher_k_over_profile": PITCHER_K_OVER_SELECTION_PROFILE,
            "pitcher_k_over_profile_status": PITCHER_K_OVER_PROFILE_STATUS,
            "history_before_date": args.history_before_date or "",
            "optimized_over_targets": [str(value).strip().upper() for value in args.optimized_over_targets],
            "over_min_abs_edge": args.over_min_abs_edge,
            "over_max_abs_edge": args.over_max_abs_edge,
            "over_min_model_hit_probability": args.over_min_model_hit_probability,
            "over_max_model_hit_probability": args.over_max_model_hit_probability,
            "over_min_expected_value": args.over_min_expected_value,
            "over_min_history_rows": args.over_min_history_rows,
            "pitcher_k_over_profile_enabled": bool(args.enable_pitcher_k_over_profile),
            "pitcher_k_min_starter_history": int(args.pitcher_k_min_starter_history),
            "pitcher_k_min_projected_ip": float(args.pitcher_k_min_projected_ip),
            "pitcher_k_min_projected_pitches": float(args.pitcher_k_min_projected_pitches),
            "pitcher_k_max_days_since_history": int(args.pitcher_k_max_days_since_history),
            "pitcher_k_min_abs_edge": float(args.pitcher_k_min_abs_edge),
            "pitcher_k_max_abs_edge": float(args.pitcher_k_max_abs_edge),
            "pitcher_k_min_model_hit_probability": float(args.pitcher_k_min_model_hit_probability),
            "pitcher_k_max_model_hit_probability": float(args.pitcher_k_max_model_hit_probability),
            "pitcher_k_min_expected_value": float(args.pitcher_k_min_expected_value),
            "pitcher_k_min_american_price": float(args.pitcher_k_min_american_price),
            "pitcher_k_max_american_price": float(args.pitcher_k_max_american_price),
            "max_pitcher_k_picks": int(args.max_pitcher_k_picks),
            "core_min_american_price": args.core_min_american_price,
            "core_max_american_price": args.core_max_american_price,
            "over_max_american_price": args.over_max_american_price,
            "min_over_picks": int(args.min_over_picks),
            "max_over_picks": int(args.max_over_picks),
            "max_under_picks": int(args.max_under_picks),
            "max_push_probability": float(args.max_push_probability),
            "max_days_since_history": int(args.max_days_since_history),
            "max_per_player": int(args.max_per_player),
            "max_per_game": int(args.max_per_game),
            "max_per_team": int(args.max_per_team),
            "max_per_market_bucket": int(args.max_per_market_bucket),
            "optimized_over_max_per_market_bucket": args.optimized_over_max_per_market_bucket,
            "min_market_books": int(args.min_market_books),
            "min_common_market_books": int(args.min_common_market_books),
            "max_market_line_std": float(args.max_market_line_std),
            "min_expected_value": float(args.min_expected_value),
            "allow_unpriced_side": bool(args.allow_unpriced_side),
            "allow_baseline": bool(args.allow_baseline),
            "require_real_market_source": bool(args.require_real_market_source),
            "targets": [str(value).strip().upper() for value in args.targets],
            "history_season": int(args.history_season),
            "min_history_bucket_rows": int(args.min_history_bucket_rows),
            "max_history_prior_weight": float(args.max_history_prior_weight),
            "history_prior_strength": float(args.history_prior_strength),
            "historical_calibration_enabled": not bool(args.disable_historical_calibration),
            "min_bet_profile_rows": int(args.min_bet_profile_rows),
            "max_bet_profile_prior_weight": float(args.max_bet_profile_prior_weight),
            "bet_profile_prior_strength": float(args.bet_profile_prior_strength),
            "min_market_availability_rows": int(args.min_market_availability_rows),
            "historical_bet_profiles_enabled": not bool(args.disable_historical_bet_profiles),
            "live_confidence_calibration_enabled": not bool(args.disable_live_confidence_calibration),
            "pick_survival_shadow_enabled": not bool(args.disable_pick_survival_shadow),
            "allow_synthetic_unders": bool(args.allow_synthetic_unders),
            "prefer_confident_side": bool(args.prefer_confident_side),
            "min_historical_bet_profile_support": int(args.min_historical_bet_profile_support),
            "min_historical_bet_profile_win_rate": float(args.min_historical_bet_profile_win_rate),
            "min_historical_market_availability_support": int(args.min_historical_market_availability_support),
            "min_historical_market_availability_rate": float(args.min_historical_market_availability_rate),
        },
        "historical_calibration": {
            "cache_json": report_path(args.history_cache_json) if args.history_cache_json else "",
            "history_dir": report_path(args.history_dir),
            "season": int(args.history_season),
            "source_file_count": int((calibration or {}).get("source_file_count", 0)),
            "updated_at_utc": (calibration or {}).get("updated_at_utc"),
        },
        "historical_bet_profiles": {
            "cache_json": report_path(args.bet_profile_cache_json) if args.bet_profile_cache_json else "",
            "history_dir": report_path(args.history_dir),
            "season": int(args.history_season),
            "source_file_count": int((bet_profile_priors or {}).get("source_file_count", 0)),
            "updated_at_utc": (bet_profile_priors or {}).get("updated_at_utc"),
        },
        "live_confidence_calibration": {
            "cache_json": report_path(args.live_confidence_cache_json) if args.live_confidence_cache_json else "",
            "history_before_date": (live_confidence_calibration or {}).get("history_before_date"),
            "graded_rows": int((live_confidence_calibration or {}).get("graded_rows", 0)),
            "brier_score_before": (live_confidence_calibration or {}).get("brier_score_before"),
            "brier_score_after": (live_confidence_calibration or {}).get("brier_score_after"),
            "walk_forward_validation": (live_confidence_calibration or {}).get("walk_forward_validation", {}),
        },
        "pick_survival_shadow": {
            "cache_json": report_path(args.pick_survival_cache_json) if args.pick_survival_cache_json else "",
            "model_version": (pick_survival_model or {}).get("model_version"),
            "status": (pick_survival_model or {}).get("status", "disabled"),
            "shadow_only": bool((pick_survival_model or {}).get("shadow_only", True)),
            "history_before_date": (pick_survival_model or {}).get("history_before_date"),
            "training_rows": int((pick_survival_model or {}).get("training_rows", 0)),
            "training_dates": int((pick_survival_model or {}).get("training_dates", 0)),
            "expanding_oof_validation": (pick_survival_model or {}).get("expanding_oof_validation", {}),
            "rolling_origin_validation": (pick_survival_model or {}).get("rolling_origin_validation", {}),
            "holdout": (pick_survival_model or {}).get("holdout", {}),
            "promotion_gate": (pick_survival_model or {}).get("promotion_gate", {}),
            "deployment_gate": (pick_survival_model or {}).get("deployment_gate", {}),
            "affects_selection": (pick_survival_model or {}).get("deployment_gate", {}).get("authority")
            == "rank_tiebreaker",
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
        "avg_historical_bet_profile_win_rate": round(sum(candidate.historical_bet_profile_win_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_historical_bet_profile_prior_weight": round(sum(candidate.historical_bet_profile_prior_weight for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_live_confidence_calibration_adjustment": round(
            sum(candidate.live_confidence_calibration_adjustment for candidate in selected) / len(selected), 6
        ) if selected else 0.0,
        "avg_historical_market_availability_rate": round(sum(candidate.historical_market_availability_rate for candidate in selected) / len(selected), 6) if selected else 0.0,
        "avg_market_books": round(sum(candidate.market_books for candidate in selected) / len(selected), 6) if selected else 0.0,
        "price_confirmed_count": int(sum(candidate.price_confirmed for candidate in selected)),
        "avg_expected_value_per_unit": round(
            sum(candidate.expected_value_per_unit for candidate in selected if candidate.expected_value_per_unit is not None)
            / max(1, sum(1 for candidate in selected if candidate.expected_value_per_unit is not None)),
            6,
        ) if selected else 0.0,
        "by_target": dict(by_target),
        "by_direction": dict(by_direction),
        "by_team": dict(by_team),
        "by_market_bucket": dict(by_market_bucket),
        "selected_preview": [
            {
                "rank": idx,
                "selection_profile": candidate_selection_profile(candidate, args),
                "player": candidate.player,
                "team": candidate.team,
                "target": candidate.target,
                "direction": candidate.direction,
                "original_direction": candidate.original_direction,
                "direction_flip_applied": bool(candidate.direction_flip_applied),
                "market_line": candidate.market_line,
                "market_bucket": candidate.market_bucket,
                "market_source": candidate.market_source,
                "opposing_pitcher": str(candidate.raw.get("Opposing_Pitcher", "")),
                "batter_profile_strength": to_float(candidate.raw.get("Batter_Profile_Strength"), 0.0),
                "pitcher_profile_vulnerability": to_float(candidate.raw.get("Pitcher_Profile_Vulnerability"), 0.0),
                "pitcher_profile_uncertainty": to_float(candidate.raw.get("Pitcher_Profile_Uncertainty"), 0.0),
                "batter_vs_starter_games": int(to_float(candidate.raw.get("Batter_Vs_Starter_Games"), 0.0)),
                "archetype_neighbor_games": int(to_float(candidate.raw.get("Archetype_Neighbor_Games"), 0.0)),
                "archetype_neighbor_effective_support": to_float(candidate.raw.get("Archetype_Neighbor_Effective_Support"), 0.0),
                "archetype_neighbor_lift": to_float(candidate.raw.get("Archetype_Neighbor_Lift"), 0.0),
                "matchup_network_score": to_float(candidate.raw.get("Matchup_Network_Score"), 0.0),
                "matchup_network_confidence": to_float(candidate.raw.get("Matchup_Network_Confidence"), 0.0),
                "matchup_network_adjustment": to_float(candidate.raw.get("Matchup_Network_Adjustment"), 0.0),
                "starter_confirmed": str(candidate.raw.get("Starter_Confirmed", "")).strip().lower() in {"1", "true", "yes"},
                "starter_history_rows": int(to_float(candidate.raw.get("Starter_History_Rows"), 0.0)),
                "projected_ip": to_float(candidate.raw.get("Projected_IP"), 0.0),
                "projected_pitches": to_float(candidate.raw.get("Projected_Pitches"), 0.0),
                "prediction": round(candidate.prediction, 4),
                "model_hit_probability": round(candidate.model_hit_probability, 4),
                "estimated_hit_probability": round(candidate.calibrated_hit_probability, 4),
                "historical_bucket_win_rate": round(candidate.historical_bucket_win_rate, 4),
                "historical_bucket_support": int(candidate.historical_bucket_support),
                "market_books": int(candidate.market_books),
                "market_common_books": int(candidate.market_common_books),
                "selected_sportsbook_key": candidate.selected_sportsbook_key,
                "selected_sportsbook": candidate.selected_sportsbook,
                "price_confirmed": bool(candidate.price_confirmed),
                "historical_bet_profile_win_rate": round(candidate.historical_bet_profile_win_rate, 4),
                "historical_bet_profile_support": int(candidate.historical_bet_profile_support),
                "live_confidence_calibration_key": candidate.live_confidence_calibration_key,
                "live_confidence_calibration_support": int(candidate.live_confidence_calibration_support),
                "live_confidence_calibration_adjustment": round(candidate.live_confidence_calibration_adjustment, 4),
                "pick_survival_probability_shadow": None if candidate.survival_probability is None else round(candidate.survival_probability, 4),
                "pick_survival_ev_shadow": None if candidate.survival_expected_value is None else round(candidate.survival_expected_value, 4),
                "pick_survival_model_status": candidate.survival_model_status,
                "pick_survival_model_support": int(candidate.survival_model_support),
                "pick_survival_rank_active": bool(candidate.survival_rank_active),
                "historical_market_availability_rate": round(candidate.historical_market_availability_rate, 4),
                "expected_value_per_unit": None if candidate.expected_value_per_unit is None else round(candidate.expected_value_per_unit, 4),
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
    history_before_date = parse_date(args.history_before_date) if args.history_before_date else None
    if args.history_before_date and history_before_date is None:
        raise ValueError("--history-before-date must be a valid YYYY-MM-DD date")

    args.history_season = infer_history_season(args.pool_csv, args.history_season)
    if args.history_cache_json is None and history_before_date is None:
        args.history_cache_json = default_history_cache_path(args.history_season)
    if args.bet_profile_cache_json is None and history_before_date is None:
        args.bet_profile_cache_json = default_bet_profile_cache_path(args.history_season)
    if args.live_confidence_cache_json is None and history_before_date is None:
        args.live_confidence_cache_json = default_live_confidence_cache_path(args.history_season)
    if args.pick_survival_cache_json is None and history_before_date is None:
        args.pick_survival_cache_json = default_pick_survival_cache_path(args.history_season)

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
            history_before_date=history_before_date,
        )
    bet_profile_priors = None
    if not args.disable_historical_bet_profiles:
        bet_profile_priors = load_or_build_historical_bet_profile_priors(
            history_dir=args.history_dir.resolve(),
            season=int(args.history_season),
            cache_json=args.bet_profile_cache_json.resolve() if args.bet_profile_cache_json else None,
            refresh=bool(args.refresh_bet_profile_cache),
            history_before_date=history_before_date,
        )
    live_confidence_calibration = None
    if not args.disable_live_confidence_calibration:
        pool_run_date = infer_pool_run_date(args.pool_csv)
        live_confidence_calibration = load_live_confidence_calibration(
            args.live_confidence_cache_json.resolve() if args.live_confidence_cache_json else None,
            pool_run_date,
        )
    pick_survival_model = None
    if not args.disable_pick_survival_shadow:
        pool_run_date = infer_pool_run_date(args.pool_csv)
        pick_survival_model = load_pick_survival_model(
            args.pick_survival_cache_json.resolve() if args.pick_survival_cache_json else None,
            pool_run_date,
        )

    candidates = load_candidates(
        args.pool_csv,
        calibration=calibration,
        bet_profile_priors=bet_profile_priors,
        live_confidence_calibration=live_confidence_calibration,
        min_history_bucket_rows=int(args.min_history_bucket_rows),
        max_history_prior_weight=float(args.max_history_prior_weight),
        history_prior_strength=float(args.history_prior_strength),
        min_bet_profile_rows=int(args.min_bet_profile_rows),
        max_bet_profile_prior_weight=float(args.max_bet_profile_prior_weight),
        bet_profile_prior_strength=float(args.bet_profile_prior_strength),
        min_market_availability_rows=int(args.min_market_availability_rows),
        prefer_confident_side=bool(args.prefer_confident_side),
        pick_survival_model=pick_survival_model,
    )
    eligible, rejected = filter_candidates(candidates, args)
    selected = select_top_candidates(eligible, args)

    write_selected_csv(args.out_csv, selected, args)
    write_summary_json(
        args.summary_json,
        args,
        args.pool_csv,
        len(candidates),
        eligible,
        selected,
        rejected,
        calibration,
        bet_profile_priors,
        live_confidence_calibration,
        pick_survival_model,
    )

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
