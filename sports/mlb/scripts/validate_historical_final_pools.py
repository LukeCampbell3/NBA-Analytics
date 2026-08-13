#!/usr/bin/env python3
"""
Validate archived MLB final pools against settled historical results.

This script grades each archived high-precision MLB selection using the current
processed MLB files, then reports:

1. overall hit rate across archived final-pool picks
2. hit rate / ROI on bets that were both real-market and side-price confirmed
3. per-date summaries so we can inspect board quality over time
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .live_board_confidence import iter_main_board_paths
except ImportError:
    from live_board_confidence import iter_main_board_paths


REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_REPORT_JSON = REPO_ROOT / "sports" / "validation" / "mlb_historical_final_pool_validation.json"
DEFAULT_MARKET_HISTORY_REPORT = (
    REPO_ROOT
    / "sports"
    / "mlb"
    / "data"
    / "raw"
    / "market_odds"
    / "mlb"
    / "historical_recovered"
    / "history_player_props_long.json"
)
TARGET_TO_ACTUAL_COL = {
    "H": "H",
    "TB": "TB",
    "R": "R",
    "HR": "HR",
    "RBI": "RBI",
    "K": "K",
    "ER": "ER",
}
PROFILE_PROMOTION_MIN_GRADED_PLAYS = 50
PROFILE_PROMOTION_BREAK_EVEN_RATE = 0.5238
OPTIMIZED_OVER_SELECTION_PROFILE = "r_tb_over_moderate_edge_v1"
PREMIUM_PRODUCTION_PROFILE = "premium_evidence_gated_v7"
DEFAULT_OVER_MIN_HISTORY_ROWS = 55
DEFAULT_OVER_HOLDOUT_START_DATE = "2026-06-01"
DEFAULT_CORE_MAX_AMERICAN_PRICE = -200
DEFAULT_DAILY_PICK_SOFT_CAP = 3
DEFAULT_POST_CAP_MIN_SELECTION_SCORE = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate archived MLB final pools against settled historical results.")
    parser.add_argument("--daily-runs-root", type=Path, default=DAILY_RUNS_ROOT, help="Root directory containing archived MLB daily runs.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT, help="Root directory containing processed MLB player files.")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON, help="Destination JSON report path.")
    parser.add_argument(
        "--market-history-report",
        type=Path,
        default=DEFAULT_MARKET_HISTORY_REPORT,
        help="Optional exact-line recovery report used to document the validation source.",
    )
    return parser.parse_args()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def normalize_player_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def is_valid_american_price(value: Any) -> bool:
    price = to_float(value)
    return bool(
        price is not None
        and (price <= -100.0 or price >= 100.0)
        and abs(price - round(price)) <= 1e-6
    )


def wilson_interval(wins: int, losses: int, z: float = 1.96) -> tuple[float | None, float | None]:
    total = int(wins) + int(losses)
    if total <= 0:
        return None, None
    probability = float(wins) / total
    denominator = 1.0 + (z * z / total)
    center = (probability + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt((probability * (1.0 - probability) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


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


def american_profit_per_unit(price: float | None) -> float | None:
    if price is None:
        return None
    value = float(price)
    if not math.isfinite(value) or abs(value) < 1e-9:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


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


def build_actual_lookup(processed_root: Path) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    usecols = ["Date", "Player", "Game_ID", *TARGET_TO_ACTUAL_COL.values()]
    for path in processed_root.glob("*/2026_processed_processed.csv"):
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in usecols)
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns or "Player" not in frame.columns or "Game_ID" not in frame.columns:
            continue

        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["Player_Key"] = frame["Player"].astype(str).map(normalize_player_key)
        frame["Game_ID"] = frame["Game_ID"].astype(str)

        for target, actual_col in TARGET_TO_ACTUAL_COL.items():
            if actual_col not in frame.columns:
                continue
            actual = pd.to_numeric(frame[actual_col], errors="coerce")
            mask = frame["Date"].notna() & frame["Player_Key"].ne("") & frame["Game_ID"].ne("") & actual.notna()
            if not bool(mask.any()):
                continue
            part = frame.loc[mask, ["Date", "Player_Key", "Game_ID"]].copy()
            part["Actual"] = actual.loc[mask].astype(float)
            for _, row in part.iterrows():
                lookup[(str(row["Date"]), str(row["Player_Key"]), target, str(row["Game_ID"]))] = float(row["Actual"])
    return lookup


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "play_count": 0,
            "date_count": 0,
            "graded_play_count": 0,
            "hit_rate": None,
            "hit_rate_wilson_95_low": None,
            "hit_rate_wilson_95_high": None,
            "line_placeable_count": 0,
            "price_confirmed_count": 0,
            "priced_play_count": 0,
            "priced_graded_count": 0,
            "priced_hit_rate": None,
            "priced_roi": None,
            "avg_units_per_priced_pool": None,
            "avg_estimated_graded_hit_rate": None,
            "calibration_gap": None,
            "brier_score": None,
        }

    graded = frame.loc[frame["result"].isin(["win", "loss"])].copy()
    priced = frame.loc[frame["units"].notna()].copy()
    priced_graded = priced.loc[priced["result"].isin(["win", "loss"])].copy()
    wins = int((graded["result"] == "win").sum())
    losses = int((graded["result"] == "loss").sum())
    wilson_low, wilson_high = wilson_interval(wins, losses)
    pool_units = priced.groupby("run_date", dropna=False)["units"].sum() if not priced.empty else pd.Series(dtype="float64")
    probabilities = (
        pd.to_numeric(graded["estimated_graded_hit_rate"], errors="coerce")
        if "estimated_graded_hit_rate" in graded.columns
        else pd.Series(index=graded.index, dtype="float64")
    )
    valid_probability = probabilities.notna()
    probability_actuals = graded.loc[valid_probability, "result"].eq("win").astype(float)
    probabilities = probabilities.loc[valid_probability].astype(float)
    return {
        "play_count": int(len(frame)),
        "date_count": int(frame["run_date"].nunique()),
        "graded_play_count": int(len(graded)),
        "hit_rate": float((graded["result"] == "win").mean()) if not graded.empty else None,
        "hit_rate_wilson_95_low": wilson_low,
        "hit_rate_wilson_95_high": wilson_high,
        "line_placeable_count": int(frame["line_placeable"].sum()),
        "price_confirmed_count": int(frame["price_confirmed"].sum()),
        "priced_play_count": int(len(priced)),
        "priced_graded_count": int(len(priced_graded)),
        "priced_hit_rate": float((priced_graded["result"] == "win").mean()) if not priced_graded.empty else None,
        "priced_roi": float(priced["units"].mean()) if not priced.empty else None,
        "avg_units_per_priced_pool": float(pool_units.mean()) if not pool_units.empty else None,
        "avg_estimated_graded_hit_rate": float(probabilities.mean()) if not probabilities.empty else None,
        "calibration_gap": float(probabilities.mean() - probability_actuals.mean()) if not probabilities.empty else None,
        "brier_score": float(((probabilities - probability_actuals) ** 2).mean()) if not probabilities.empty else None,
    }


def assess_profile_promotion(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    graded_plays = int(summary.get("graded_play_count", 0) or 0)
    wilson_low = to_float(summary.get("hit_rate_wilson_95_low"))
    priced_roi = to_float(summary.get("priced_roi"))
    play_count = int(summary.get("play_count", 0) or 0)
    price_confirmed_count = int(summary.get("price_confirmed_count", 0) or 0)
    if graded_plays < PROFILE_PROMOTION_MIN_GRADED_PLAYS:
        reasons.append(f"fewer than {PROFILE_PROMOTION_MIN_GRADED_PLAYS} graded plays")
    if wilson_low is None or wilson_low <= PROFILE_PROMOTION_BREAK_EVEN_RATE:
        reasons.append("95% hit-rate lower bound does not clear the -110 break-even rate")
    if priced_roi is None or priced_roi <= 0.0:
        reasons.append("confirmed-price ROI is not positive")
    if price_confirmed_count != play_count:
        reasons.append("not every selected row has a confirmed named-book price")
    return {
        "status": "eligible_for_review" if not reasons else "probation",
        "eligible_for_review": not reasons,
        "minimum_graded_plays": PROFILE_PROMOTION_MIN_GRADED_PLAYS,
        "minimum_wilson_95_low": PROFILE_PROMOTION_BREAK_EVEN_RATE,
        "reasons": reasons,
    }


def summarize_over_maturity_route(
    rows: list[dict[str, Any]],
    *,
    min_history_rows: int = DEFAULT_OVER_MIN_HISTORY_ROWS,
    holdout_start_date: str = DEFAULT_OVER_HOLDOUT_START_DATE,
    core_max_american_price: float = DEFAULT_CORE_MAX_AMERICAN_PRICE,
) -> dict[str, Any]:
    optimized_over = [
        row for row in rows if row.get("selection_profile") == OPTIMIZED_OVER_SELECTION_PROFILE
    ]
    mature_over = [
        row for row in optimized_over if int(row.get("history_rows", 0) or 0) >= int(min_history_rows)
    ]
    routed = [
        row
        for row in rows
        if row.get("selection_profile") != OPTIMIZED_OVER_SELECTION_PROFILE
        or int(row.get("history_rows", 0) or 0) >= int(min_history_rows)
    ]
    premium_routed = [
        row
        for row in rows
        if (
            row.get("selection_profile") == OPTIMIZED_OVER_SELECTION_PROFILE
            and int(row.get("history_rows", 0) or 0) >= int(min_history_rows)
        )
        or (
            row.get("selection_profile") != OPTIMIZED_OVER_SELECTION_PROFILE
            and (to_float(row.get("selected_side_price")) or 0.0) <= float(core_max_american_price)
        )
    ]
    calibration = [row for row in optimized_over if str(row.get("run_date", "")) < holdout_start_date]
    holdout = [row for row in mature_over if str(row.get("run_date", "")) >= holdout_start_date]
    return {
        "optimized_over_profile": OPTIMIZED_OVER_SELECTION_PROFILE,
        "minimum_history_rows": int(min_history_rows),
        "core_max_american_price": float(core_max_american_price),
        "holdout_start_date": holdout_start_date,
        "all_optimized_over": summarize_rows(optimized_over),
        "calibration_period_optimized_over": summarize_rows(calibration),
        "holdout_mature_optimized_over": summarize_rows(holdout),
        "combined_maturity_gated_policy": summarize_rows(routed),
        "premium_price_defended_policy": summarize_rows(premium_routed),
    }


def summarize_daily_volume_route(
    rows: list[dict[str, Any]],
    *,
    soft_cap: int = DEFAULT_DAILY_PICK_SOFT_CAP,
    post_cap_min_selection_score: float = DEFAULT_POST_CAP_MIN_SELECTION_SCORE,
) -> dict[str, Any]:
    cap = max(0, int(soft_cap))
    score_floor = max(0.0, float(post_cap_min_selection_score))

    def is_routed(row: dict[str, Any]) -> bool:
        return bool(
            cap <= 0
            or int(row.get("rank", 0) or 0) <= cap
            or float(row.get("selection_score", 0.0) or 0.0) >= score_floor
        )

    routed = [row for row in rows if is_routed(row)]
    removed = [row for row in rows if not is_routed(row)]

    ranks = sorted({int(row.get("rank", 0) or 0) for row in rows if int(row.get("rank", 0) or 0) > 0})
    by_rank = {
        str(rank): summarize_rows([row for row in rows if int(row.get("rank", 0) or 0) == rank])
        for rank in ranks
    }
    cumulative_by_rank = {
        str(rank): summarize_rows(
            [row for row in rows if 0 < int(row.get("rank", 0) or 0) <= rank]
        )
        for rank in ranks
    }
    recommended_soft_cap = None
    if ranks:
        recommended_soft_cap = max(
            ranks,
            key=lambda rank: (
                float(cumulative_by_rank[str(rank)].get("hit_rate_wilson_95_low") or -1.0),
                float(cumulative_by_rank[str(rank)].get("priced_roi") or -999.0),
                int(cumulative_by_rank[str(rank)].get("graded_play_count") or 0),
            ),
        )

    dates = sorted({str(row.get("run_date", "")) for row in rows if str(row.get("run_date", ""))})
    baseline_counts = {
        run_date: sum(str(row.get("run_date", "")) == run_date for row in rows)
        for run_date in dates
    }
    routed_counts = {
        run_date: sum(str(row.get("run_date", "")) == run_date for row in routed)
        for run_date in dates
    }
    return {
        "soft_cap": cap,
        "post_cap_min_selection_score": score_floor,
        "baseline": summarize_rows(rows),
        "adaptive_policy": summarize_rows(routed),
        "removed_tail": summarize_rows(removed),
        "baseline_daily_pick_counts": baseline_counts,
        "adaptive_daily_pick_counts": routed_counts,
        "by_rank": by_rank,
        "cumulative_by_rank": cumulative_by_rank,
        "cap_optimization": {
            "objective": "maximize the 95% Wilson hit-rate lower bound, then confirmed-price ROI and graded sample",
            "recommended_soft_cap": recommended_soft_cap,
            "configured_soft_cap": cap,
            "configured_cap_matches_recommendation": recommended_soft_cap == cap,
        },
    }


def main() -> None:
    args = parse_args()
    actual_lookup = build_actual_lookup(args.processed_root.resolve())
    rows: list[dict[str, Any]] = []
    by_date: list[dict[str, Any]] = []

    selected_paths = iter_main_board_paths(args.daily_runs_root)
    for path in selected_paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Game_Date" not in frame.columns:
            continue

        date_rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            run_date = str(row.get("Game_Date", ""))[:10]
            player_key = normalize_player_key(row.get("Player_ID") or row.get("Player"))
            game_id = str(row.get("Game_ID", "") or "")
            target = str(row.get("Target", "")).strip().upper()
            direction = str(row.get("Direction", "")).strip().upper()
            market_line = to_float(row.get("Market_Line"))
            if not run_date or not player_key or not game_id or target not in TARGET_TO_ACTUAL_COL or market_line is None:
                continue

            actual = actual_lookup.get((run_date, player_key, target, game_id))
            result = ""
            if actual is not None:
                result = grade_result(float(actual), float(market_line), direction)

            market_source = str(row.get("Market_Source", "")).strip().lower()
            books = int(to_float(row.get("Market_Books")) or 0)
            selected_side_price = to_float(row.get("Selected_Side_Price"))
            if selected_side_price is None:
                selected_side_price = to_float(row.get("Market_Over_Price")) if direction == "OVER" else to_float(row.get("Market_Under_Price"))
            line_placeable = bool(market_source == "real" and books > 0)
            price_confirmed = bool(
                is_valid_american_price(selected_side_price)
                and str(row.get("Selected_Sportsbook_Key", "")).strip()
                and str(row.get("Selected_Sportsbook", "")).strip()
            )
            units = settled_units(result, selected_side_price) if result and price_confirmed else None
            estimated_graded_hit_rate = to_float(row.get("Estimated_Graded_Hit_Rate"))

            record = {
                "run_date": run_date,
                "rank": int(to_float(row.get("Rank")) or 0),
                "player": str(row.get("Player", "")),
                "target": target,
                "direction": direction,
                "selection_profile": str(row.get("Selection_Profile", "")).strip() or "unlabeled",
                "history_rows": int(to_float(row.get("History_Rows")) or 0),
                "selection_score": float(to_float(row.get("Selection_Score")) or 0.0),
                "selected_side_price": selected_side_price,
                "market_line": float(market_line),
                "actual": None if actual is None else float(actual),
                "result": result,
                "line_placeable": line_placeable,
                "price_confirmed": price_confirmed,
                "units": units,
                "estimated_graded_hit_rate": estimated_graded_hit_rate,
                "source_file": str(path),
            }
            rows.append(record)
            date_rows.append(record)

        date_summary = summarize_rows(date_rows)
        date_summary["run_date"] = path.parent.name
        by_date.append(date_summary)

    rows_frame = pd.DataFrame(rows)
    by_selection_profile = {
        str(profile): summarize_rows(part.to_dict(orient="records"))
        for profile, part in rows_frame.groupby("selection_profile", dropna=False)
    } if not rows_frame.empty else {}
    profile_promotion_assessments = {
        profile: assess_profile_promotion(summary)
        for profile, summary in by_selection_profile.items()
    }
    market_history: dict[str, Any] = {}
    if args.market_history_report and args.market_history_report.exists():
        try:
            recovered = json.loads(args.market_history_report.read_text(encoding="utf-8"))
            market_history = {
                key: recovered.get(key)
                for key in [
                    "source",
                    "row_count",
                    "capture_count",
                    "event_date_count",
                    "first_event_date",
                    "last_event_date",
                    "bookmaker_count",
                ]
            }
            market_history["report"] = portable_path(args.market_history_report)
        except (OSError, json.JSONDecodeError):
            market_history = {}
    report = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "validation_method": "walk-forward selection with actual settled outcomes and exact named-book prices",
        "production_candidate_profile": PREMIUM_PRODUCTION_PROFILE,
        "synthetic_events_used": False,
        "daily_runs_root": portable_path(args.daily_runs_root),
        "processed_root": portable_path(args.processed_root),
        "market_history": market_history,
        "source_file_count": len(selected_paths),
        "overall": summarize_rows(rows),
        "by_selection_profile": by_selection_profile,
        "profile_promotion_assessments": profile_promotion_assessments,
        "over_maturity_route": summarize_over_maturity_route(rows),
        "daily_volume_route": summarize_daily_volume_route(rows),
        "by_date": by_date,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    overall = report["overall"]
    print("MLB HISTORICAL FINAL POOL VALIDATION")
    print(f"Play count:              {overall['play_count']}")
    print(f"Graded play count:       {overall['graded_play_count']}")
    print(f"Overall hit rate:        {overall['hit_rate']}")
    print(f"Line placeable count:    {overall['line_placeable_count']}")
    print(f"Price confirmed count:   {overall['price_confirmed_count']}")
    print(f"Priced play count:       {overall['priced_play_count']}")
    print(f"Priced hit rate:         {overall['priced_hit_rate']}")
    print(f"Priced ROI:              {overall['priced_roi']}")
    print(f"Avg units / priced pool: {overall['avg_units_per_priced_pool']}")
    print(f"Report JSON:             {args.report_json}")


if __name__ == "__main__":
    main()
