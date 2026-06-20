#!/usr/bin/env python3
"""Leakage-aware walk-forward backtest for the MLB prediction selector."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.scripts.select_high_precision_predictions import (
    build_candidate,
    estimate_count_hit_probabilities,
    filter_candidates,
    infer_direction,
    line_probability_key,
    market_bucket_key,
    probability_bucket,
    select_top_candidates,
    target_direction_key,
    target_probability_key,
)


DEFAULT_UNIVERSE = SPORT_ROOT / "data" / "predictions" / "calibration" / "historical_pool_universe_2026.csv"
DEFAULT_OUTPUT_ROOT = SPORT_ROOT / "data" / "predictions" / "backtests"
DEFAULT_ARCHIVED_VALIDATION = REPO_ROOT / "sports" / "validation" / "mlb_historical_final_pool_validation.json"
DEFAULT_EXTERNAL_AUDIT = DEFAULT_OUTPUT_ROOT / "mlb_20260617_external_audit.json"
DEFAULT_RAW_POOL_AUDIT = DEFAULT_OUTPUT_ROOT / "mlb_20260619_raw_pool_partial_audit.json"
SUPPORTED_TARGETS = ["H", "K", "R", "TB"]


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    args: SimpleNamespace
    prefer_confident_side: bool = False
    require_confirmed_price: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-training-days", type=int, default=14)
    parser.add_argument("--short-window-days", type=int, default=7)
    parser.add_argument("--archived-validation-json", type=Path, default=DEFAULT_ARCHIVED_VALIDATION)
    parser.add_argument("--external-audit-json", type=Path, default=DEFAULT_EXTERNAL_AUDIT)
    parser.add_argument("--raw-pool-audit-json", type=Path, default=DEFAULT_RAW_POOL_AUDIT)
    return parser.parse_args()


def selector_args(**overrides: Any) -> SimpleNamespace:
    values = {
        "top_n": 50,
        "min_abs_edge": 0.10,
        "min_history_rows": 11,
        "min_prediction": 0.0,
        "min_hit_probability": 0.53,
        "min_graded_hit_rate": 0.53,
        "max_push_probability": 0.18,
        "max_days_since_history": 90,
        "max_per_player": 2,
        "max_per_game": 4,
        "max_per_team": 5,
        "max_per_market_bucket": 4,
        "min_market_books": 1,
        "max_market_line_std": 0.0,
        "min_expected_value": 0.0,
        "allow_baseline": False,
        "require_real_market_source": False,
        "targets": SUPPORTED_TARGETS,
        "min_history_bucket_rows": 50,
        "max_history_prior_weight": 0.35,
        "history_prior_strength": 400.0,
        "min_bet_profile_rows": 12,
        "max_bet_profile_prior_weight": 0.25,
        "bet_profile_prior_strength": 80.0,
        "min_market_availability_rows": 12,
        "allow_synthetic_unders": False,
        "min_historical_bet_profile_support": 0,
        "min_historical_bet_profile_win_rate": 0.0,
        "min_historical_market_availability_support": 0,
        "min_historical_market_availability_rate": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def policies() -> list[Policy]:
    return [
        Policy(
            name="production_action_board",
            description="Current publish policy with market-depth, history, role, and concentration gates.",
            args=selector_args(
                top_n=10,
                min_abs_edge=0.60,
                min_history_rows=30,
                min_prediction=0.10,
                min_hit_probability=0.60,
                min_graded_hit_rate=0.72,
                max_push_probability=0.12,
                max_days_since_history=4,
                max_per_player=1,
                max_per_game=2,
                max_per_team=3,
                max_per_market_bucket=2,
                min_market_books=5,
                require_real_market_source=True,
                min_historical_bet_profile_support=12,
                min_historical_bet_profile_win_rate=0.55,
                min_historical_market_availability_support=20,
                min_historical_market_availability_rate=0.45,
            ),
            require_confirmed_price=True,
        ),
        Policy(
            name="published_real_market",
            description="Current published thresholds on rows backed by at least one sportsbook.",
            args=selector_args(),
            require_confirmed_price=True,
        ),
        Policy(
            name="directional_model_replay",
            description="Long-window model replay on stored synthetic or real lines; not an ROI test.",
            args=selector_args(
                min_market_books=0,
                min_expected_value=-1.0,
                allow_synthetic_unders=True,
            ),
        ),
        Policy(
            name="guardrailed_short_board",
            description="Six-play research board with tighter probability, recency, and concentration limits.",
            args=selector_args(
                top_n=6,
                min_abs_edge=0.45,
                min_hit_probability=0.58,
                min_graded_hit_rate=0.68,
                max_push_probability=0.18,
                max_days_since_history=4,
                max_per_player=1,
                max_per_game=2,
                max_per_team=3,
                max_per_market_bucket=2,
                min_market_books=0,
                min_expected_value=-1.0,
                allow_synthetic_unders=True,
            ),
        ),
    ]


def grade(actual: float, line: float, direction: str) -> str:
    if direction == "OVER":
        return "win" if actual > line else "push" if actual == line else "loss"
    return "win" if actual < line else "push" if actual == line else "loss"


def american_profit(price: float | None) -> float | None:
    if price is None or not math.isfinite(price) or abs(price) < 100.0:
        return None
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def result_units(result: str, price: float | None = None) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -1.0
    return american_profit(price) if american_profit(price) is not None else 100.0 / 110.0


def empty_result_stats() -> dict[str, float | int]:
    return {"rows": 0, "graded_rows": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": 0.5, "push_rate": 0.0}


def finalize_result_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    wins = int(stats["wins"])
    losses = int(stats["losses"])
    rows = int(stats["rows"])
    graded = wins + losses
    return {
        **stats,
        "graded_rows": graded,
        "win_rate": wins / graded if graded else 0.5,
        "push_rate": int(stats["pushes"]) / rows if rows else 0.0,
    }


def update_result_stats(stats: dict[str, float | int], result: str) -> None:
    stats["rows"] = int(stats["rows"]) + 1
    key = {"win": "wins", "loss": "losses", "push": "pushes"}[result]
    stats[key] = int(stats[key]) + 1


def prior_payload(history: pd.DataFrame, evaluation_date: pd.Timestamp) -> tuple[dict[str, Any], dict[str, Any]]:
    target_direction: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    line_buckets: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    recent_target_direction: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    recent_line_buckets: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    availability_td: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"rows": 0, "side_price_rows": 0, "books_sum": 0.0}
    )
    availability_line: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"rows": 0, "side_price_rows": 0, "books_sum": 0.0}
    )
    profile_td: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    profile_line: dict[str, dict[str, float | int]] = defaultdict(empty_result_stats)
    recent_cutoff = evaluation_date - timedelta(days=14)

    for row in history.to_dict(orient="records"):
        direction = infer_direction(float(row["Edge"]))
        if direction is None:
            continue
        target = str(row["Target"])
        line = float(row["Market_Line"])
        actual = float(row["Actual"])
        outcome = grade(actual, line, direction)
        td_key = target_direction_key(target, direction)
        line_key = market_bucket_key(target, direction, line)
        update_result_stats(target_direction[td_key], outcome)
        update_result_stats(line_buckets[line_key], outcome)
        if pd.Timestamp(row["_date"]) >= recent_cutoff:
            update_result_stats(recent_target_direction[td_key], outcome)
            update_result_stats(recent_line_buckets[line_key], outcome)

        books = int(max(0, float(row.get("Market_Books", 0) or 0)))
        if str(row.get("Market_Source", "")).lower() != "real" or books <= 0:
            continue
        side_price = row.get("Market_Over_Price") if direction == "OVER" else row.get("Market_Under_Price")
        side_price = float(side_price) if pd.notna(side_price) else None
        confirmed = side_price is not None and math.isfinite(side_price) and abs(side_price) > 1e-9
        for bucket in (availability_td[td_key], availability_line[line_key]):
            bucket["rows"] = int(bucket["rows"]) + 1
            bucket["side_price_rows"] = int(bucket["side_price_rows"]) + int(confirmed)
            bucket["books_sum"] = float(bucket["books_sum"]) + books
        if not confirmed:
            continue
        prediction = float(row["Prediction"])
        model_graded = estimate_count_hit_probabilities(prediction, line, direction)[2]
        profile_keys = (
            target_probability_key(target, direction, model_graded),
            line_probability_key(target, direction, line, model_graded),
        )
        for bucket, key in zip((profile_td, profile_line), profile_keys):
            update_result_stats(bucket[key], outcome)
            bucket[key]["units_sum"] = float(bucket[key].get("units_sum", 0.0)) + result_units(outcome, side_price)

    def finalized(values: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
        return {key: finalize_result_stats(value) for key, value in values.items()}

    def finalized_availability(values: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
        output = {}
        for key, value in values.items():
            rows = int(value["rows"])
            output[key] = {
                **value,
                "availability_rate": int(value["side_price_rows"]) / rows if rows else 0.0,
                "avg_books": float(value["books_sum"]) / rows if rows else 0.0,
            }
        return output

    def finalized_profiles(values: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
        output = finalized(values)
        for value in output.values():
            rows = int(value["rows"])
            value["roi_per_bet"] = float(value.get("units_sum", 0.0)) / rows if rows else 0.0
        return output

    calibration = {
        "target_direction": finalized(target_direction),
        "line_buckets": finalized(line_buckets),
        "recent_target_direction": finalized(recent_target_direction),
        "recent_line_buckets": finalized(recent_line_buckets),
    }
    bet_profiles = {
        "availability_target_direction": finalized_availability(availability_td),
        "availability_line_buckets": finalized_availability(availability_line),
        "bet_profiles_target_probability": finalized_profiles(profile_td),
        "bet_profiles_line_probability": finalized_profiles(profile_line),
    }
    return calibration, bet_profiles


def history_before(universe: pd.DataFrame, evaluation_date: pd.Timestamp) -> pd.DataFrame:
    return universe.loc[universe["_date"] < evaluation_date].copy()


def evaluate_policy(
    universe: pd.DataFrame,
    evaluation_dates: list[pd.Timestamp],
    policy: Policy,
    prior_cache: dict[pd.Timestamp, tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    selected_rows: list[dict[str, Any]] = []
    cache = prior_cache if prior_cache is not None else {}
    for evaluation_date in evaluation_dates:
        day = universe.loc[universe["_date"] == evaluation_date]
        if evaluation_date not in cache:
            history = history_before(universe, evaluation_date)
            cache[evaluation_date] = prior_payload(history, evaluation_date)
        calibration, bet_profiles = cache[evaluation_date]
        candidates = []
        for row in day.to_dict(orient="records"):
            candidate = build_candidate(
                row,
                calibration=calibration,
                bet_profile_priors=bet_profiles,
                min_history_bucket_rows=policy.args.min_history_bucket_rows,
                max_history_prior_weight=policy.args.max_history_prior_weight,
                history_prior_strength=policy.args.history_prior_strength,
                min_bet_profile_rows=policy.args.min_bet_profile_rows,
                max_bet_profile_prior_weight=policy.args.max_bet_profile_prior_weight,
                bet_profile_prior_strength=policy.args.bet_profile_prior_strength,
                min_market_availability_rows=policy.args.min_market_availability_rows,
                prefer_confident_side=policy.prefer_confident_side,
            )
            if candidate is not None:
                candidates.append(candidate)
        eligible, _ = filter_candidates(candidates, policy.args)
        if policy.require_confirmed_price:
            eligible = [candidate for candidate in eligible if candidate.price_confirmed]
        for candidate in select_top_candidates(eligible, policy.args):
            actual = float(candidate.raw["Actual"])
            outcome = grade(actual, candidate.market_line, candidate.direction)
            selected_rows.append(
                {
                    "policy": policy.name,
                    "date": evaluation_date.strftime("%Y-%m-%d"),
                    "player": candidate.player,
                    "player_id": candidate.player_id,
                    "team": candidate.team,
                    "opponent": str(candidate.raw.get("Opponent", "")),
                    "game_id": candidate.game_id,
                    "target": candidate.target,
                    "direction": candidate.direction,
                    "line": candidate.market_line,
                    "actual": actual,
                    "result": outcome,
                    "model_hit_probability": candidate.model_hit_probability,
                    "hit_probability": candidate.calibrated_hit_probability,
                    "probability": candidate.calibrated_graded_hit_rate,
                    "push_probability": candidate.push_probability,
                    "abs_edge": candidate.abs_edge,
                    "history_rows": candidate.history_rows,
                    "days_since_history": candidate.days_since_history,
                    "selection_score": candidate.selection_score,
                    "market_bucket": candidate.market_bucket,
                    "price_confirmed": candidate.price_confirmed,
                    "expected_value_per_unit": candidate.expected_value_per_unit,
                    "historical_bet_profile_win_rate": candidate.historical_bet_profile_win_rate,
                    "historical_bet_profile_support": candidate.historical_bet_profile_support,
                    "historical_market_availability_rate": candidate.historical_market_availability_rate,
                    "historical_market_availability_support": candidate.historical_market_availability_support,
                    "market_source": candidate.market_source,
                    "books": candidate.market_books,
                    "side_price": candidate.selected_side_price,
                    "units": result_units(outcome, candidate.selected_side_price),
                }
            )
    return pd.DataFrame(selected_rows)


def wilson_interval(wins: int, losses: int, z: float = 1.96) -> tuple[float | None, float | None]:
    n = wins + losses
    if n == 0:
        return None, None
    p = wins / n
    denominator = 1.0 + (z * z / n)
    center = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denominator
    return center - margin, center + margin


def longest_streak(results: Iterable[str], target: str) -> int:
    longest = current = 0
    for result in results:
        current = current + 1 if result == target else 0
        longest = max(longest, current)
    return longest


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"plays": 0, "graded": 0, "wins": 0, "losses": 0, "pushes": 0, "hit_rate": None}
    graded = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    wins = int((graded["result"] == "win").sum())
    losses = int((graded["result"] == "loss").sum())
    low, high = wilson_interval(wins, losses)
    units = frame["units"].astype(float)
    equity = units.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    probabilities = graded["probability"].astype(float)
    actuals = graded["result"].eq("win").astype(float)
    return {
        "plays": int(len(frame)),
        "graded": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "pushes": int((frame["result"] == "push").sum()),
        "hit_rate": wins / len(graded) if len(graded) else None,
        "hit_rate_wilson_95_low": low,
        "hit_rate_wilson_95_high": high,
        "avg_model_probability": float(probabilities.mean()) if len(graded) else None,
        "calibration_gap": float(probabilities.mean() - actuals.mean()) if len(graded) else None,
        "brier_score": float(((probabilities - actuals) ** 2).mean()) if len(graded) else None,
        "net_units": float(units.sum()),
        "roi_per_play": float(units.sum() / len(frame)),
        "max_drawdown_units": float(drawdown.min()) if len(drawdown) else 0.0,
        "longest_win_streak": longest_streak(frame["result"], "win"),
        "longest_loss_streak": longest_streak(frame["result"], "loss"),
        "dates_with_picks": int(frame["date"].nunique()),
        "real_market_plays": int(frame["market_source"].eq("real").sum()),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MLB Prediction Method Backtest",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Method",
        "",
        "Every evaluation date uses only rows from earlier dates to construct historical and recent-form priors. "
        "Candidate direction is re-graded from the selected side. Flat-stake units use recorded odds when available "
        "and -110 only as a research proxy otherwise.",
        "",
        "## Results",
        "",
        "| Policy | Window | W-L-P | Hit rate | 95% interval | Model estimate | Net units* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["policies"].items():
        for window in ("long_window", "short_window"):
            stats = result[window]
            interval = "n/a"
            if stats.get("hit_rate_wilson_95_low") is not None:
                interval = f"{stats['hit_rate_wilson_95_low']:.1%}-{stats['hit_rate_wilson_95_high']:.1%}"
            hit_rate = "n/a" if stats.get("hit_rate") is None else f"{stats['hit_rate']:.1%}"
            estimate = "n/a" if stats.get("avg_model_probability") is None else f"{stats['avg_model_probability']:.1%}"
            units = "n/a" if stats.get("net_units") is None else f"{stats['net_units']:+.2f}"
            lines.append(
                f"| {name} | {window.replace('_', ' ')} | {stats['wins']}-{stats['losses']}-{stats['pushes']} "
                f"| {hit_rate} | {interval} | {estimate} | {units} |"
            )
    lines.extend(
        [
            "",
            "*Recorded prices are used when present; unpriced research rows use a -110 proxy and are not executable ROI.*",
            "",
            "## Observed Boards",
            "",
            f"- Archived boards: {report['observed_evidence']['archived_boards']['wins']}-"
            f"{report['observed_evidence']['archived_boards']['losses']} "
            f"({report['observed_evidence']['archived_boards']['hit_rate']:.1%}) on "
            f"{report['observed_evidence']['archived_boards']['graded']} graded picks.",
            f"- June 17 after deduplication: {report['observed_evidence']['june_17_deduplicated']['wins']}-"
            f"{report['observed_evidence']['june_17_deduplicated']['losses']} "
            f"({report['observed_evidence']['june_17_deduplicated']['hit_rate']:.1%}) on "
            f"{report['observed_evidence']['june_17_deduplicated']['graded']} graded picks.",
            f"- June 19 raw top-edge partial audit: {report['observed_evidence']['june_19_raw_pool_partial']['wins']}-"
            f"{report['observed_evidence']['june_19_raw_pool_partial']['losses']} on "
            f"{report['observed_evidence']['june_19_raw_pool_partial']['graded']} completed-game rows; "
            "this is a raw-pool diagnostic, not a finalized-board result.",
            f"- Combined observed direction: {report['observed_evidence']['combined']['wins']}-"
            f"{report['observed_evidence']['combined']['losses']} "
            f"({report['observed_evidence']['combined']['hit_rate']:.1%}); 95% interval "
            f"{report['observed_evidence']['combined']['hit_rate_wilson_95_low']:.1%}-"
            f"{report['observed_evidence']['combined']['hit_rate_wilson_95_high']:.1%}.",
            "",
            "## Calibration Audit",
            "",
            f"The raw 90-100% estimate bucket realized "
            f"{report['calibration_audit']['top_bucket']['actual_hit_rate']:.1%} on "
            f"{report['calibration_audit']['top_bucket']['graded']} outcomes, versus an average estimate of "
            f"{report['calibration_audit']['top_bucket']['average_estimate']:.1%}.",
            "",
            f"**Verdict: {report['promotion_verdict'].upper()}**",
            "",
            "## Interpretation",
            "",
            *[f"- {item}" for item in report["interpretation"]],
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def raw_calibration_audit(universe: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for row in universe.to_dict(orient="records"):
        direction = infer_direction(float(row["Edge"]))
        if direction is None:
            continue
        outcome = grade(float(row["Actual"]), float(row["Market_Line"]), direction)
        if outcome == "push":
            continue
        probability = estimate_count_hit_probabilities(
            float(row["Prediction"]), float(row["Market_Line"]), direction
        )[2]
        rows.append({"probability": probability, "win": outcome == "win"})
    frame = pd.DataFrame(rows)
    top = frame.loc[frame["probability"] >= 0.90]
    return {
        "all_candidates": {
            "graded": int(len(frame)),
            "actual_hit_rate": float(frame["win"].mean()),
            "average_estimate": float(frame["probability"].mean()),
        },
        "top_bucket": {
            "range": "0.90-1.00",
            "graded": int(len(top)),
            "actual_hit_rate": float(top["win"].mean()),
            "average_estimate": float(top["probability"].mean()),
            "calibration_gap": float(top["probability"].mean() - top["win"].mean()),
        },
    }


def observed_evidence(archived_path: Path, audit_path: Path, raw_pool_audit_path: Path) -> dict[str, Any]:
    archived_payload = json.loads(archived_path.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_pool_audit = json.loads(raw_pool_audit_path.read_text(encoding="utf-8"))
    archived_overall = archived_payload["overall"]
    archived_graded = int(archived_overall["graded_play_count"])
    archived_wins = int(round(float(archived_overall["hit_rate"]) * archived_graded))
    archived = {
        "source": str(archived_path.resolve()),
        "graded": archived_graded,
        "wins": archived_wins,
        "losses": archived_graded - archived_wins,
        "hit_rate": float(archived_overall["hit_rate"]),
        "price_confirmed_count": int(archived_overall["price_confirmed_count"]),
        "priced_hit_rate": archived_overall["priced_hit_rate"],
        "priced_roi": archived_overall["priced_roi"],
    }
    june = dict(audit_payload["post_duplicate_suppression"])
    june["source"] = str(audit_path.resolve())
    wins = archived["wins"] + int(june["wins"])
    losses = archived["losses"] + int(june["losses"])
    low, high = wilson_interval(wins, losses)
    return {
        "archived_boards": archived,
        "june_17_deduplicated": june,
        "june_19_raw_pool_partial": {
            **raw_pool_audit["completed_top_edge_sample"],
            "source": str(raw_pool_audit_path.resolve()),
        },
        "combined": {
            "wins": wins,
            "losses": losses,
            "graded": wins + losses,
            "hit_rate": wins / (wins + losses),
            "hit_rate_wilson_95_low": low,
            "hit_rate_wilson_95_high": high,
        },
    }


def main() -> None:
    args = parse_args()
    universe = pd.read_csv(args.universe_csv.resolve())
    universe["_date"] = pd.to_datetime(universe["Game_Date"], errors="coerce").dt.normalize()
    universe = universe.loc[universe["_date"].notna() & universe["Target"].isin(SUPPORTED_TARGETS)].copy()
    for column in ["Prediction", "Market_Line", "Edge", "Actual", "Market_Books", "Market_Over_Price", "Market_Under_Price"]:
        universe[column] = pd.to_numeric(universe[column], errors="coerce")
    universe = universe.dropna(subset=["Prediction", "Market_Line", "Edge", "Actual"])
    dates = sorted(pd.Timestamp(value) for value in universe["_date"].unique())
    priced_dates = sorted(
        pd.Timestamp(value)
        for value in universe.loc[universe["Market_Books"].fillna(0).gt(0), "_date"].unique()
    )
    evaluation_dates = dates[max(0, int(args.min_training_days)) :]
    short_dates = set(evaluation_dates[-max(1, int(args.short_window_days)) :])

    frames = []
    report_policies: dict[str, Any] = {}
    prior_cache: dict[pd.Timestamp, tuple[dict[str, Any], dict[str, Any]]] = {}
    for policy in policies():
        frame = evaluate_policy(universe, evaluation_dates, policy, prior_cache)
        frames.append(frame)
        short_frame = frame.loc[pd.to_datetime(frame["date"]).isin(short_dates)] if not frame.empty else frame
        report_policies[policy.name] = {
            "description": policy.description,
            "config": {**vars(policy.args), "require_confirmed_price": policy.require_confirmed_price},
            "long_window": summarize(frame),
            "short_window": summarize(short_frame),
            "by_target": {str(key): summarize(part) for key, part in frame.groupby("target")} if not frame.empty else {},
            "by_direction": {str(key): summarize(part) for key, part in frame.groupby("direction")} if not frame.empty else {},
        }

    production = report_policies["production_action_board"]["long_window"]
    published = report_policies["published_real_market"]["long_window"]
    proxy = report_policies["directional_model_replay"]["long_window"]
    guardrail = report_policies["guardrailed_short_board"]["long_window"]
    interpretation = [
        f"The production action policy contains {production['graded']} graded historical picks across "
        f"{production.get('dates_with_picks', 0)} dates; it remains shadow-only until the priced sample is materially larger.",
        f"The placeable real-market sample contains {published['graded']} graded picks across "
        f"{published.get('dates_with_picks', 0)} dates; it is too small to support a long-term claim.",
        f"The long-window directional replay hit {proxy.get('hit_rate'):.1%} on {proxy['graded']} graded picks; "
        "this measures stored-line directional accuracy, not realizable betting profit.",
        f"The tighter six-play board hit {guardrail.get('hit_rate'):.1%} on {guardrail['graded']} graded picks "
        f"with a 95% interval of {guardrail.get('hit_rate_wilson_95_low'):.1%}-"
        f"{guardrail.get('hit_rate_wilson_95_high'):.1%}.",
    ]
    limitations = [
        f"The historical universe ends on {dates[-1].strftime('%Y-%m-%d')}; real sportsbook rows cover "
        f"{priced_dates[0].strftime('%Y-%m-%d')} through {priced_dates[-1].strftime('%Y-%m-%d')} across "
        f"{len(priced_dates)} dates." if priced_dates else "The historical universe contains no real sportsbook rows.",
        "Synthetic-line rows test model ranking and grading logic but cannot establish executable ROI or closing-line value.",
        "The replay covers H, K, R, and TB; the current published board also includes HR and ER, which lack this backtest universe.",
        "Lineup confirmation, roster validation, duplicate suppression, and stale-data withholding reduce publishing risk but do not create predictive edge.",
        "No backtest can guarantee short-term or long-term wins; promotion requires prospective, timestamped shadow results.",
    ]
    report = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "universe_csv": str(args.universe_csv.resolve()),
        "universe_rows": int(len(universe)),
        "universe_start": dates[0].strftime("%Y-%m-%d"),
        "universe_end": dates[-1].strftime("%Y-%m-%d"),
        "evaluation_start": evaluation_dates[0].strftime("%Y-%m-%d"),
        "evaluation_end": evaluation_dates[-1].strftime("%Y-%m-%d"),
        "evaluation_dates": len(evaluation_dates),
        "short_window_days": int(args.short_window_days),
        "leakage_controls": {
            "date_bounded_priors": True,
            "strict_prior_date_comparison": "history_date < evaluation_date",
            "selected_direction_regraded": True,
        },
        "policies": report_policies,
        "calibration_audit": raw_calibration_audit(universe),
        "observed_evidence": observed_evidence(
            args.archived_validation_json.resolve(),
            args.external_audit_json.resolve(),
            args.raw_pool_audit_json.resolve(),
        ),
        "promotion_verdict": "shadow_only_not_validated",
        "interpretation": interpretation,
        "limitations": limitations,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_root / "mlb_walk_forward_backtest_rows.csv"
    json_path = args.output_root / "mlb_walk_forward_backtest.json"
    md_path = args.output_root / "mlb_walk_forward_backtest.md"
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(rows_path, index=False)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    print(f"\nRows: {rows_path}\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()
