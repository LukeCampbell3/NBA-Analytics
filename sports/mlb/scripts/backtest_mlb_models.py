#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.io_utils import ensure_dir, write_json
from pipeline.training import (
    ROLE_TARGETS,
    RollingBaselineRegressor,
    _candidate_models,
    _feature_columns,
    _fit_target_models,
    _load_processed_frame,
    _time_split,
)


@dataclass
class BacktestConfig:
    processed_dir: Path
    output_rows: Path
    output_summary: Path
    season: int = 2026
    min_train_rows: int = 100
    min_train_dates: int = 2
    min_edge: float = 0.0
    include_synthetic_market: bool = False
    val_fraction: float = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward MLB backtest against historical outcomes and market lines.")
    parser.add_argument("--season", type=int, default=2026, help="Season year.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Processed MLB directory.",
    )
    parser.add_argument(
        "--output-rows",
        type=Path,
        default=ROOT / "models" / "analysis" / "backtest_rows.csv",
        help="Row-level backtest output CSV.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=ROOT / "models" / "analysis" / "backtest_summary.json",
        help="Summary JSON output.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=100,
        help="Minimum prior rows required before scoring a date.",
    )
    parser.add_argument(
        "--min-train-dates",
        type=int,
        default=2,
        help="Minimum prior unique dates required before scoring a date.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum absolute prediction-vs-line edge required to place a graded bet.",
    )
    parser.add_argument(
        "--include-synthetic-market",
        action="store_true",
        help="Also grade rows whose market lines are synthetic placeholders.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction used during model selection on each walk-forward training window.",
    )
    return parser.parse_args()


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_baseline_model(train_df: pd.DataFrame, feature_cols: list[str], target: str) -> RollingBaselineRegressor:
    baseline_col = f"{target}_rolling_avg"
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0)
    baseline_value = float(y_train.mean()) if len(y_train) else 0.0
    if baseline_col in train_df.columns:
        baseline_mean = pd.to_numeric(train_df[baseline_col], errors="coerce").dropna()
        if not baseline_mean.empty:
            baseline_value = float(baseline_mean.mean())
    return RollingBaselineRegressor(
        feature_col=baseline_col if baseline_col in feature_cols else None,
        fallback_value=baseline_value,
    )


def _refit_selected_bundle(
    train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    X_train = _clean_numeric_frame(train_df[feature_cols].copy())
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    member_names = [str(name) for name in selected.get("members", [])]
    weights = [float(weight) for weight in selected.get("weights", [])]
    if not member_names:
        member_names = [str(selected.get("name") or "baseline")]
        weights = [1.0]

    models: list[Any] = []
    candidates = _candidate_models()
    for member in member_names:
        if member == "baseline":
            model = _build_baseline_model(train_df, feature_cols=feature_cols, target=target)
        else:
            if member not in candidates:
                raise KeyError(f"Unknown selected model member: {member}")
            model = candidates[member]
            model.fit(X_train, y_train)
        models.append(model)
    return {
        "name": str(selected.get("name") or member_names[0]),
        "members": member_names,
        "weights": weights if weights else [1.0] * len(models),
        "models": models,
    }


def _predict_bundle(bundle: dict[str, Any], score_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    preds = np.zeros(len(score_df), dtype=np.float64)
    X_score = _clean_numeric_frame(score_df[feature_cols].copy())
    for weight, model in zip(bundle["weights"], bundle["models"]):
        preds += float(weight) * np.asarray(model.predict(X_score), dtype=np.float64)
    return preds


def _american_profit_multiplier(price: float | None) -> float | None:
    if price is None or pd.isna(price):
        return None
    price = float(price)
    if price == 0.0:
        return None
    if price > 0.0:
        return price / 100.0
    return 100.0 / abs(price)


def _grade_bet(
    *,
    prediction: float,
    actual: float,
    market_line: float | None,
    over_price: float | None,
    under_price: float | None,
    market_source: str,
    min_edge: float,
    include_synthetic_market: bool,
) -> dict[str, Any]:
    out = {
        "bet_eligible": False,
        "bet_direction": None,
        "bet_price": None,
        "bet_result": None,
        "bet_units": None,
        "bet_edge": None,
    }
    if market_line is None or pd.isna(market_line):
        return out
    if market_source != "real" and not include_synthetic_market:
        return out

    edge = float(prediction) - float(market_line)
    out["bet_edge"] = edge
    if abs(edge) <= float(min_edge):
        return out

    direction = "OVER" if edge > 0 else "UNDER"
    price = over_price if direction == "OVER" else under_price
    profit_multiplier = _american_profit_multiplier(price)
    if profit_multiplier is None:
        return out

    out["bet_eligible"] = True
    out["bet_direction"] = direction
    out["bet_price"] = float(price)

    if float(actual) == float(market_line):
        out["bet_result"] = "push"
        out["bet_units"] = 0.0
        return out

    won = float(actual) > float(market_line) if direction == "OVER" else float(actual) < float(market_line)
    out["bet_result"] = "win" if won else "loss"
    out["bet_units"] = float(profit_multiplier if won else -1.0)
    return out


def _summarize_rows(rows_df: pd.DataFrame, *, include_synthetic_market: bool) -> dict[str, Any]:
    if rows_df.empty:
        return {
            "rows": 0,
            "prediction_dates": 0,
            "date_min": None,
            "date_max": None,
            "mae": None,
            "rmse": None,
            "market_rows_any": 0,
            "market_rows_real": 0,
            "market_rows_scoped": 0,
            "graded_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "roi_per_bet": None,
            "net_units": 0.0,
            "line_mae_any": None,
            "line_mae_scoped": None,
            "market_scope": "real+synthetic" if include_synthetic_market else "real_only",
        }

    actual = pd.to_numeric(rows_df["actual"], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(rows_df["prediction"], errors="coerce").fillna(0.0)
    market_line = pd.to_numeric(rows_df["market_line"], errors="coerce")
    market_source = rows_df["market_source"].astype(str)
    bet_rows = rows_df.loc[rows_df["bet_eligible"].fillna(False)].copy()
    wins = int((bet_rows["bet_result"] == "win").sum())
    losses = int((bet_rows["bet_result"] == "loss").sum())
    pushes = int((bet_rows["bet_result"] == "push").sum())
    decisions = wins + losses
    graded_bets = wins + losses + pushes
    net_units = float(pd.to_numeric(bet_rows["bet_units"], errors="coerce").fillna(0.0).sum()) if not bet_rows.empty else 0.0
    line_mask = market_line.notna()
    scoped_line_mask = line_mask if include_synthetic_market else (line_mask & (market_source == "real"))
    return {
        "rows": int(len(rows_df)),
        "prediction_dates": int(pd.to_datetime(rows_df["date"], errors="coerce").dt.normalize().nunique()),
        "date_min": str(pd.to_datetime(rows_df["date"], errors="coerce").min().date()),
        "date_max": str(pd.to_datetime(rows_df["date"], errors="coerce").max().date()),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "market_rows_any": int(line_mask.sum()),
        "market_rows_real": int((rows_df["market_source"].astype(str) == "real").sum()),
        "market_rows_scoped": int(scoped_line_mask.sum()),
        "graded_bets": int(graded_bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": (wins / decisions) if decisions else None,
        "roi_per_bet": (net_units / decisions) if decisions else None,
        "net_units": net_units,
        "line_mae_any": float(mean_absolute_error(actual[line_mask], market_line[line_mask])) if line_mask.any() else None,
        "line_mae_scoped": (
            float(mean_absolute_error(actual[scoped_line_mask], market_line[scoped_line_mask]))
            if scoped_line_mask.any()
            else None
        ),
        "market_scope": "real+synthetic" if include_synthetic_market else "real_only",
    }


def _walk_forward_role(df: pd.DataFrame, role: str, config: BacktestConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = ROLE_TARGETS[role]
    feature_cols = _feature_columns(df, targets=targets, role=role)
    all_dates = sorted(df["Date"].dropna().dt.normalize().unique().tolist())
    records: list[dict[str, Any]] = []
    skipped_dates: list[dict[str, Any]] = []
    selected_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for prediction_date in all_dates:
        history_df = df.loc[df["Date"] < prediction_date].copy()
        score_df = df.loc[df["Date"].dt.normalize() == prediction_date].copy()
        history_dates = int(history_df["Date"].dt.normalize().nunique()) if not history_df.empty else 0

        if score_df.empty:
            continue
        if len(history_df) < int(config.min_train_rows):
            skipped_dates.append(
                {
                    "date": str(pd.Timestamp(prediction_date).date()),
                    "reason": f"history_rows_lt_{int(config.min_train_rows)}",
                    "history_rows": int(len(history_df)),
                    "history_dates": history_dates,
                }
            )
            continue
        if history_dates < int(config.min_train_dates):
            skipped_dates.append(
                {
                    "date": str(pd.Timestamp(prediction_date).date()),
                    "reason": f"history_dates_lt_{int(config.min_train_dates)}",
                    "history_rows": int(len(history_df)),
                    "history_dates": history_dates,
                }
            )
            continue

        train_df, val_df = _time_split(history_df, val_fraction=float(config.val_fraction))
        if train_df.empty or val_df.empty:
            skipped_dates.append(
                {
                    "date": str(pd.Timestamp(prediction_date).date()),
                    "reason": "train_val_split_failed",
                    "history_rows": int(len(history_df)),
                    "history_dates": history_dates,
                }
            )
            continue

        for target in targets:
            target_variance = float(pd.to_numeric(history_df[target], errors="coerce").fillna(0.0).var())
            if math.isclose(target_variance, 0.0, abs_tol=1e-12):
                skipped_dates.append(
                    {
                        "date": str(pd.Timestamp(prediction_date).date()),
                        "reason": f"{target}_zero_variance",
                        "history_rows": int(len(history_df)),
                        "history_dates": history_dates,
                    }
                )
                continue

            payload = _fit_target_models(train_df, val_df, feature_cols=feature_cols, target=target)
            selected = payload["selected"]
            selected_counter[target][str(selected.get("name", "unknown"))] += 1
            bundle = _refit_selected_bundle(
                history_df,
                feature_cols=feature_cols,
                target=target,
                selected=selected,
            )
            pred = _predict_bundle(bundle, score_df, feature_cols=feature_cols)

            market_col = f"Market_{target}"
            source_col = f"Market_Source_{target}"
            over_col = f"Market_{target}_over_price"
            under_col = f"Market_{target}_under_price"
            books_col = f"Market_{target}_books"
            std_col = f"Market_{target}_line_std"

            for idx, row in score_df.reset_index(drop=True).iterrows():
                actual = pd.to_numeric(pd.Series([row.get(target)]), errors="coerce").iloc[0]
                market_line = pd.to_numeric(pd.Series([row.get(market_col)]), errors="coerce").iloc[0]
                over_price = pd.to_numeric(pd.Series([row.get(over_col)]), errors="coerce").iloc[0]
                under_price = pd.to_numeric(pd.Series([row.get(under_col)]), errors="coerce").iloc[0]
                market_source = str(row.get(source_col, "missing") or "missing")
                grade = _grade_bet(
                    prediction=float(pred[idx]),
                    actual=float(actual) if pd.notna(actual) else 0.0,
                    market_line=float(market_line) if pd.notna(market_line) else None,
                    over_price=float(over_price) if pd.notna(over_price) else None,
                    under_price=float(under_price) if pd.notna(under_price) else None,
                    market_source=market_source,
                    min_edge=float(config.min_edge),
                    include_synthetic_market=bool(config.include_synthetic_market),
                )
                records.append(
                    {
                        "date": str(pd.Timestamp(prediction_date).date()),
                        "role": role,
                        "target": target,
                        "player": row.get("Player"),
                        "team": row.get("Team"),
                        "opponent": row.get("Opponent"),
                        "game_id": row.get("Game_ID"),
                        "history_rows": int(len(history_df)),
                        "history_dates": history_dates,
                        "selection_train_rows": int(len(train_df)),
                        "selection_val_rows": int(len(val_df)),
                        "selected_model": str(selected.get("name")),
                        "selected_members": "|".join(str(item) for item in bundle["members"]),
                        "prediction": float(pred[idx]),
                        "actual": float(actual) if pd.notna(actual) else None,
                        "abs_error": abs(float(pred[idx]) - float(actual)) if pd.notna(actual) else None,
                        "squared_error": (float(pred[idx]) - float(actual)) ** 2 if pd.notna(actual) else None,
                        "market_line": float(market_line) if pd.notna(market_line) else None,
                        "market_source": market_source,
                        "market_books": pd.to_numeric(pd.Series([row.get(books_col)]), errors="coerce").iloc[0],
                        "market_line_std": pd.to_numeric(pd.Series([row.get(std_col)]), errors="coerce").iloc[0],
                        "market_over_price": float(over_price) if pd.notna(over_price) else None,
                        "market_under_price": float(under_price) if pd.notna(under_price) else None,
                        **grade,
                    }
                )

    rows_df = pd.DataFrame.from_records(records)
    summary = {
        "role": role,
        "rows_available": int(len(df)),
        "unique_dates_available": int(df["Date"].dt.normalize().nunique()),
        "prediction_dates_scored": int(rows_df["date"].nunique()) if not rows_df.empty else 0,
        "selected_model_counts": {target: dict(counter) for target, counter in selected_counter.items()},
        "skipped_dates": skipped_dates,
        "targets": {},
    }

    for target in targets:
        target_rows = rows_df.loc[rows_df["target"] == target].copy() if not rows_df.empty else pd.DataFrame()
        summary["targets"][target] = _summarize_rows(
            target_rows,
            include_synthetic_market=bool(config.include_synthetic_market),
        )
    summary["overall"] = _summarize_rows(rows_df, include_synthetic_market=bool(config.include_synthetic_market))
    return rows_df, summary


def run_backtest(config: BacktestConfig) -> dict[str, Any]:
    processed_dir = config.processed_dir.resolve()
    all_rows: list[pd.DataFrame] = []
    role_summaries: dict[str, Any] = {}
    failures: dict[str, str] = {}

    for role in ["hitter", "pitcher"]:
        try:
            df = _load_processed_frame(processed_dir, season=config.season, role=role)
            if df.empty:
                raise RuntimeError(f"No processed rows found for role={role}")
            rows_df, summary = _walk_forward_role(df, role=role, config=config)
            role_summaries[role] = summary
            all_rows.append(rows_df)
        except Exception as exc:
            failures[role] = str(exc)

    non_empty_rows = [frame for frame in all_rows if not frame.empty]
    combined = pd.concat(non_empty_rows, ignore_index=True) if non_empty_rows else pd.DataFrame()
    ensure_dir(config.output_rows.resolve().parent)
    combined.to_csv(config.output_rows.resolve(), index=False)

    overall = _summarize_rows(combined, include_synthetic_market=bool(config.include_synthetic_market))
    payload = {
        "season": int(config.season),
        "processed_dir": str(processed_dir),
        "rows_csv": str(config.output_rows.resolve()),
        "config": {
            "min_train_rows": int(config.min_train_rows),
            "min_train_dates": int(config.min_train_dates),
            "min_edge": float(config.min_edge),
            "include_synthetic_market": bool(config.include_synthetic_market),
            "val_fraction": float(config.val_fraction),
        },
        "overall": overall,
        "roles": role_summaries,
        "failures": failures,
        "notes": [
            "Bet grading uses only rows with Market_* values present.",
            "By default, only rows with Market_Source_* == real are graded as betting-line validation.",
            "Synthetic market rows can be included explicitly for debugging, but they are not real sportsbook validation.",
        ],
    }
    write_json(config.output_summary.resolve(), payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_backtest(
        BacktestConfig(
            processed_dir=args.processed_dir,
            output_rows=args.output_rows,
            output_summary=args.output_summary,
            season=int(args.season),
            min_train_rows=int(args.min_train_rows),
            min_train_dates=int(args.min_train_dates),
            min_edge=float(args.min_edge),
            include_synthetic_market=bool(args.include_synthetic_market),
            val_fraction=float(args.val_fraction),
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
