#!/usr/bin/env python3
"""Tune MLB selector thresholds with chronological train/validation/holdout splits."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.scripts.backtest_prediction_method import (
    Policy,
    evaluate_policy,
    selector_args,
    wilson_interval,
)
from sports.mlb.scripts.select_high_precision_predictions import SUPPORTED_COUNT_TARGETS


DEFAULT_UNIVERSE = SPORT_ROOT / "data" / "predictions" / "calibration" / "historical_pool_universe_2026.csv"
DEFAULT_OUTPUT_ROOT = SPORT_ROOT / "data" / "predictions" / "optimization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-cache", type=Path, default=None)
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--min-training-days", type=int, default=14)
    parser.add_argument("--validation-days", type=int, default=14)
    parser.add_argument("--holdout-days", type=int, default=14)
    return parser.parse_args()


def load_universe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_date"] = pd.to_datetime(frame["Game_Date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["_date"].notna() & frame["Target"].isin(SUPPORTED_COUNT_TARGETS)].copy()
    for column in ["Prediction", "Market_Line", "Edge", "Actual", "Market_Books", "Market_Over_Price", "Market_Under_Price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Prediction", "Market_Line", "Edge", "Actual"])


def broad_candidate_policy() -> Policy:
    return Policy(
        name="walk_forward_candidate_ledger",
        description="All modeled, fresh count-prop candidates before board-size optimization.",
        args=selector_args(
            top_n=100000,
            min_abs_edge=0.0,
            min_history_rows=10,
            min_prediction=0.0,
            min_hit_probability=0.0,
            min_graded_hit_rate=0.0,
            max_push_probability=1.0,
            max_days_since_history=4,
            max_per_player=100,
            max_per_game=1000,
            max_per_team=1000,
            max_per_market_bucket=100000,
            min_market_books=0,
            min_expected_value=-1.0,
            allow_synthetic_unders=True,
        ),
    )


def build_candidate_ledger(
    universe: pd.DataFrame,
    evaluation_dates: list[pd.Timestamp],
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    metadata_path = cache_path.with_suffix(".meta.json")
    expected = {
        "universe_rows": int(len(universe)),
        "universe_end": max(evaluation_dates).strftime("%Y-%m-%d"),
        "evaluation_dates": len(evaluation_dates),
    }
    if cache_path.exists() and metadata_path.exists() and not refresh:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == expected:
                return pd.read_csv(cache_path)
        except (OSError, json.JSONDecodeError):
            pass

    ledger = evaluate_policy(universe, evaluation_dates, broad_candidate_policy())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(cache_path, index=False)
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return ledger


def policy_grid() -> list[dict[str, Any]]:
    configs = []
    values = itertools.product(
        [3, 4, 6, 8, 10],
        [0.60, 0.68, 0.74],
        [0.72, 0.80, 0.86],
        [0.12, 0.18],
        [0.40, 0.65],
        [1, 2],
    )
    for index, (top_n, min_hit, min_graded, max_push, min_edge, max_bucket) in enumerate(values, start=1):
        configs.append(
            {
                "name": f"candidate_{index:03d}",
                "top_n": top_n,
                "min_hit_probability": min_hit,
                "min_graded_hit_rate": min_graded,
                "max_push_probability": max_push,
                "min_abs_edge": min_edge,
                "max_per_market_bucket": max_bucket,
                "max_per_player": 1,
                "max_per_game": 2,
                "max_per_team": 3,
                "min_historical_bet_profile_support": 12,
                "min_historical_bet_profile_win_rate": 0.55,
                "min_historical_market_availability_support": 20,
                "min_historical_market_availability_rate": 0.45,
            }
        )
    return configs


def benchmark_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "production_current",
            "top_n": 10,
            "min_hit_probability": 0.58,
            "min_graded_hit_rate": 0.68,
            "max_push_probability": 0.24,
            "min_abs_edge": 0.45,
            "max_per_market_bucket": 4,
            "max_per_player": 1,
            "max_per_game": 2,
            "max_per_team": 3,
            "min_historical_bet_profile_support": 0,
            "min_historical_bet_profile_win_rate": 0.0,
            "min_historical_market_availability_support": 0,
            "min_historical_market_availability_rate": 0.0,
        },
        {
            "name": "guardrailed_six",
            "top_n": 6,
            "min_hit_probability": 0.58,
            "min_graded_hit_rate": 0.68,
            "max_push_probability": 0.18,
            "min_abs_edge": 0.45,
            "max_per_market_bucket": 2,
            "max_per_player": 1,
            "max_per_game": 2,
            "max_per_team": 3,
            "min_historical_bet_profile_support": 0,
            "min_historical_bet_profile_win_rate": 0.0,
            "min_historical_market_availability_support": 0,
            "min_historical_market_availability_rate": 0.0,
        },
    ]


def select_config(ledger: pd.DataFrame, dates: set[str], config: dict[str, Any]) -> pd.DataFrame:
    """Real bug fix (found while investigating an implausible 96-100%
    holdout hit rate): this never filtered by market_source, and
    broad_candidate_policy()'s own ledger is ~97.5% synthetic (non-real-
    market) rows -- a config "optimized" against that ledger is mostly
    measuring the model beating its own estimated lines, not a real,
    executable market price. The real live selector ALWAYS runs with
    --require-real-market-source (see MLB_PRIMARY_POLICY_ARGS in
    run_daily_predictions.py); this tool exists to tune thresholds for
    that real selector, so it must be evaluated on the same real-only
    population, unconditionally -- never a configurable option that
    could be silently left off again."""
    selected_indices: list[int] = []
    relevant = ledger.loc[ledger["date"].isin(dates) & ledger["market_source"].eq("real")].copy()
    relevant = relevant.loc[
        (relevant["hit_probability"] >= float(config["min_hit_probability"]))
        & (relevant["probability"] >= float(config["min_graded_hit_rate"]))
        & (relevant["push_probability"] <= float(config["max_push_probability"]))
        & (relevant["abs_edge"] >= float(config["min_abs_edge"]))
        & (
            relevant["historical_bet_profile_support"]
            >= int(config.get("min_historical_bet_profile_support", 0))
        )
        & (
            relevant["historical_bet_profile_win_rate"]
            >= float(config.get("min_historical_bet_profile_win_rate", 0.0))
        )
        & (
            relevant["historical_market_availability_support"]
            >= int(config.get("min_historical_market_availability_support", 0))
        )
        & (
            relevant["historical_market_availability_rate"]
            >= float(config.get("min_historical_market_availability_rate", 0.0))
        )
    ]
    relevant = relevant.sort_values(["date", "selection_score"], ascending=[True, False])

    for _, part in relevant.groupby("date", sort=True):
        by_player: Counter[str] = Counter()
        by_game: Counter[str] = Counter()
        by_team: Counter[str] = Counter()
        by_bucket: Counter[str] = Counter()
        kept = 0
        for row in part.itertuples():
            player_key = str(row.player_id or row.player).strip().lower()
            game_key = str(row.game_id).strip()
            team_key = str(row.team).strip().upper()
            bucket_key = str(row.market_bucket).strip()
            if by_player[player_key] >= int(config["max_per_player"]):
                continue
            if by_game[game_key] >= int(config["max_per_game"]):
                continue
            if by_team[team_key] >= int(config["max_per_team"]):
                continue
            if by_bucket[bucket_key] >= int(config["max_per_market_bucket"]):
                continue
            selected_indices.append(int(row.Index))
            by_player[player_key] += 1
            by_game[game_key] += 1
            by_team[team_key] += 1
            by_bucket[bucket_key] += 1
            kept += 1
            if kept >= int(config["top_n"]):
                break
    return ledger.loc[selected_indices].sort_values(["date", "selection_score"], ascending=[True, False]).copy()


def max_drawdown(units: pd.Series) -> float:
    if units.empty:
        return 0.0
    equity = units.cumsum()
    peaks = equity.cummax().clip(lower=0.0)
    return float((equity - peaks).min())


def score_rows(rows: pd.DataFrame, date_count: int) -> dict[str, Any]:
    if rows.empty:
        return {
            "plays": 0,
            "graded": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": None,
            "hit_rate_wilson_95_low": None,
            "hit_rate_wilson_95_high": None,
            "proxy_net_units": 0.0,
            "proxy_roi": None,
            "proxy_profit_per_day": 0.0,
            "max_drawdown_units": 0.0,
            "avg_model_probability": None,
            "calibration_gap": None,
            "priced_plays": 0,
            "priced_net_units": 0.0,
            "priced_roi": None,
        }
    graded = rows.loc[rows["result"].isin(["win", "loss"])].copy()
    wins = int(graded["result"].eq("win").sum())
    losses = int(graded["result"].eq("loss").sum())
    low, high = wilson_interval(wins, losses)
    proxy_units = rows["result"].map({"win": 100.0 / 110.0, "loss": -1.0, "push": 0.0}).fillna(0.0)
    priced = rows.loc[rows["price_confirmed"].astype(bool)].copy()
    priced_units = priced["units"].astype(float)
    hit_rate = wins / (wins + losses) if wins + losses else None
    avg_probability = float(graded["probability"].mean()) if not graded.empty else None
    return {
        "plays": int(len(rows)),
        "graded": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "pushes": int(rows["result"].eq("push").sum()),
        "hit_rate": hit_rate,
        "hit_rate_wilson_95_low": low,
        "hit_rate_wilson_95_high": high,
        "proxy_net_units": float(proxy_units.sum()),
        "proxy_roi": float(proxy_units.sum() / len(rows)),
        "proxy_profit_per_day": float(proxy_units.sum() / max(1, date_count)),
        "max_drawdown_units": max_drawdown(proxy_units),
        "avg_model_probability": avg_probability,
        "calibration_gap": float(avg_probability - hit_rate) if avg_probability is not None and hit_rate is not None else None,
        "priced_plays": int(len(priced)),
        "priced_net_units": float(priced_units.sum()) if not priced.empty else 0.0,
        "priced_roi": float(priced_units.mean()) if not priced.empty else None,
    }


def objective(train: dict[str, Any], validation: dict[str, Any], train_days: int, validation_days: int) -> float | None:
    if train["graded"] < max(60, train_days * 2) or validation["graded"] < max(20, validation_days * 2):
        return None
    train_low = float(train["hit_rate_wilson_95_low"] or 0.0)
    validation_low = float(validation["hit_rate_wilson_95_low"] or 0.0)
    calibration_penalty = min(0.25, abs(float(validation["calibration_gap"] or 0.0)))
    profit_score = min(2.0, max(-1.0, float(validation["proxy_profit_per_day"]))) / 2.0
    drawdown_penalty = min(0.20, abs(min(0.0, float(validation["max_drawdown_units"]))) / 50.0)
    return (
        (0.38 * validation_low)
        + (0.20 * float(validation["hit_rate"] or 0.0))
        + (0.17 * train_low)
        + (0.15 * profit_score)
        - (0.07 * calibration_penalty)
        - (0.03 * drawdown_penalty)
    )


def compact_stats(stats: dict[str, Any]) -> str:
    if stats["hit_rate"] is None:
        return "n/a"
    return (
        f"{stats['wins']}-{stats['losses']}-{stats['pushes']} "
        f"({stats['hit_rate']:.1%}), {stats['proxy_net_units']:+.2f}u proxy"
    )


def markdown_report(report: dict[str, Any]) -> str:
    chosen = report["chosen"]
    lines = [
        "# MLB Walk-Forward Policy Optimization",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Split",
        "",
        f"- Training: {report['splits']['train']['start']} through {report['splits']['train']['end']}",
        f"- Validation: {report['splits']['validation']['start']} through {report['splits']['validation']['end']}",
        f"- Untouched holdout: {report['splits']['holdout']['start']} through {report['splits']['holdout']['end']}",
        "",
        "## Selected Policy",
        "",
        f"- Config: `{json.dumps(chosen['config'], sort_keys=True)}`",
        f"- Training: {compact_stats(chosen['train'])}",
        f"- Validation: {compact_stats(chosen['validation'])}",
        f"- Holdout: {compact_stats(chosen['holdout'])}",
        f"- Recent seven days: {compact_stats(chosen['recent_7'])}",
        "",
        "## Holdout Comparison",
        "",
        "| Policy | W-L-P | Hit rate | 95% low | Proxy units | Drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, payload in report["holdout_comparison"].items():
        stats = payload["stats"]
        lines.append(
            f"| {name} | {stats['wins']}-{stats['losses']}-{stats['pushes']} | "
            f"{stats['hit_rate']:.1%} | {stats['hit_rate_wilson_95_low']:.1%} | "
            f"{stats['proxy_net_units']:+.2f} | {stats['max_drawdown_units']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"**Promotion verdict: {report['promotion']['verdict']}**",
            "",
            *[f"- {reason}" for reason in report["promotion"]["reasons"]],
            "",
            "## Guardrails",
            "",
            "- Configuration selection never reads holdout outcomes.",
            "- Invalid American prices are excluded from price-confirmed ROI.",
            "- Proxy profit assumes flat -110 stakes and is not executable ROI.",
            "- Production promotion requires prospective price-confirmed holdout volume.",
            "",
        ]
    )
    return "\n".join(lines)


def split_dates(evaluation_dates: list[str], validation_days: int, holdout_days: int) -> tuple[list[str], list[str], list[str]]:
    if len(evaluation_dates) <= validation_days + holdout_days + 7:
        raise ValueError("Not enough evaluation dates for train/validation/holdout optimization.")
    holdout = evaluation_dates[-holdout_days:]
    validation = evaluation_dates[-(holdout_days + validation_days) : -holdout_days]
    train = evaluation_dates[: -(holdout_days + validation_days)]
    return train, validation, holdout


def split_metadata(dates: list[str]) -> dict[str, Any]:
    return {"start": dates[0], "end": dates[-1], "dates": len(dates)}


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_cache = args.candidate_cache.resolve() if args.candidate_cache else output_root / "mlb_walk_forward_candidate_ledger.csv"
    universe = load_universe(args.universe_csv.resolve())
    all_dates = sorted(pd.Timestamp(value) for value in universe["_date"].unique())
    evaluation_dates = all_dates[int(args.min_training_days) :]
    ledger = build_candidate_ledger(
        universe,
        evaluation_dates,
        candidate_cache,
        bool(args.refresh_candidates),
    )
    date_values = sorted(str(value) for value in ledger["date"].unique())
    train_dates, validation_dates, holdout_dates = split_dates(
        date_values,
        int(args.validation_days),
        int(args.holdout_days),
    )
    train_set, validation_set, holdout_set = set(train_dates), set(validation_dates), set(holdout_dates)

    scored = []
    for config in policy_grid():
        train_rows = select_config(ledger, train_set, config)
        validation_rows = select_config(ledger, validation_set, config)
        train_stats = score_rows(train_rows, len(train_dates))
        validation_stats = score_rows(validation_rows, len(validation_dates))
        score = objective(train_stats, validation_stats, len(train_dates), len(validation_dates))
        if score is not None:
            scored.append({"config": config, "objective": score, "train": train_stats, "validation": validation_stats})
    if not scored:
        raise RuntimeError("No policy met minimum training and validation volume.")
    scored.sort(key=lambda row: (row["objective"], row["validation"]["proxy_net_units"]), reverse=True)
    chosen = scored[0]
    chosen_holdout_rows = select_config(ledger, holdout_set, chosen["config"])
    chosen_holdout = score_rows(chosen_holdout_rows, len(holdout_dates))
    recent_dates = set(holdout_dates[-7:])
    chosen_recent = score_rows(select_config(ledger, recent_dates, chosen["config"]), len(recent_dates))
    chosen = {**chosen, "holdout": chosen_holdout, "recent_7": chosen_recent}

    comparison: dict[str, Any] = {}
    for config in [*benchmark_configs(), chosen["config"]]:
        name = "optimized_candidate" if config is chosen["config"] else str(config["name"])
        rows = select_config(ledger, holdout_set, config)
        comparison[name] = {"config": config, "stats": score_rows(rows, len(holdout_dates))}

    holdout_gate = chosen_holdout
    promotion_reasons = []
    if holdout_gate["graded"] < len(holdout_dates) * 2:
        promotion_reasons.append("holdout volume is below two graded plays per day")
    if (holdout_gate["hit_rate_wilson_95_low"] or 0.0) <= 0.5238:
        promotion_reasons.append("holdout 95% lower bound does not clear -110 break-even")
    if holdout_gate["proxy_net_units"] <= 0.0:
        promotion_reasons.append("holdout proxy profit is not positive")
    if holdout_gate["priced_plays"] < 30:
        promotion_reasons.append("fewer than 30 valid price-confirmed holdout plays")
    promotion = {
        "verdict": "promote" if not promotion_reasons else "shadow_only",
        "reasons": promotion_reasons or ["all statistical and price-confirmed gates passed"],
    }

    report = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "universe": {
            "path": str(args.universe_csv.resolve()),
            "rows": int(len(universe)),
            "start": min(all_dates).strftime("%Y-%m-%d"),
            "end": max(all_dates).strftime("%Y-%m-%d"),
        },
        "candidate_ledger_rows": int(len(ledger)),
        "configs_scored": len(scored),
        "splits": {
            "train": split_metadata(train_dates),
            "validation": split_metadata(validation_dates),
            "holdout": split_metadata(holdout_dates),
        },
        "chosen": chosen,
        "holdout_comparison": comparison,
        "promotion": promotion,
        "leaderboard": scored[:20],
    }
    report_json = output_root / "mlb_walk_forward_policy_optimization.json"
    report_md = output_root / "mlb_walk_forward_policy_optimization.md"
    leaderboard_csv = output_root / "mlb_walk_forward_policy_leaderboard.csv"
    selected_rows_csv = output_root / "mlb_walk_forward_optimized_holdout_rows.csv"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md.write_text(markdown_report(report), encoding="utf-8")
    pd.DataFrame(
        [
            {
                **row["config"],
                "objective": row["objective"],
                "train_hit_rate": row["train"]["hit_rate"],
                "validation_hit_rate": row["validation"]["hit_rate"],
                "validation_proxy_units": row["validation"]["proxy_net_units"],
                "validation_wilson_low": row["validation"]["hit_rate_wilson_95_low"],
            }
            for row in scored
        ]
    ).to_csv(leaderboard_csv, index=False)
    chosen_holdout_rows.to_csv(selected_rows_csv, index=False)
    print(markdown_report(report))
    print(f"\nJSON: {report_json}\nMarkdown: {report_md}\nLeaderboard: {leaderboard_csv}")


if __name__ == "__main__":
    main()
