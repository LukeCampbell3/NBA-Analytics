#!/usr/bin/env python3
"""Train and apply a leakage-safe shadow model for MLB pick survival."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_OUTPUT_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"
MODEL_VERSION = "mlb_pick_survival_logit_v2"
# v12 Phase 1 (SafeEV veto): a second real model over a DIFFERENT, narrower
# population -- rows that would have survived v11's own structural gates
# (see build_v11_eligible_training_set.py) -- with a richer feature set
# than pick_survival_model's own (this model's population already carries
# real historical-bucket/bet-profile/market-availability enrichment that
# build_historical_candidates() above doesn't compute). Never promotes a
# bet on its own: see apply_winner_signature_model()'s safe_probability =
# min(v11_probability, winner_signature_probability) -- negative authority
# only, matching the proposal's own stated rule.
WINNER_SIGNATURE_MODEL_VERSION = "mlb_winner_signature_logit_v1"
WINNER_SIGNATURE_NUMERIC_FEATURES = (
    "directional_edge",
    "abs_edge",
    "model_hit_probability",
    "market_implied_probability",
    "market_line_std",
    "market_books",
    "market_common_books",
    "history_rows",
    "historical_bucket_win_rate",
    "historical_bucket_support",
    "historical_bet_profile_win_rate",
    "historical_bet_profile_roi",
    "historical_bet_profile_support",
    "historical_market_availability_rate",
    "historical_market_availability_support",
    "live_confidence_calibration_adjustment",
    # Real "disagreement between independent probability estimators" --
    # computed from quantities this repo already produces (the calibrated
    # model probability, the real no-vig market-implied probability, and
    # pick_survival_model's own shadow output), not a new estimator.
    "model_market_disagreement",
    "model_survival_disagreement",
)
TARGETS = ("H", "TB", "R", "HR", "RBI", "K", "ER")
STANDARD_MARKET_LINES = {"H": 0.5, "TB": 1.5, "R": 0.5, "HR": 0.5, "RBI": 0.5}
NUMERIC_FEATURES = (
    "directional_edge",
    "model_hit_probability",
    "history_rows",
    "market_books",
    "market_line_std",
    "market_implied_probability",
)
CATEGORICAL_FEATURES = tuple(
    [f"target={target}" for target in TARGETS]
    + ["direction=OVER", "direction=UNDER", "player_type=hitter", "player_type=pitcher"]
)
MIN_TRAIN_ROWS = 180
MIN_TRAIN_DATES = 8
MIN_SEGMENT_ROWS = 10


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def to_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def valid_american_price(value: Any, *, min_price: float, max_price: float) -> bool:
    price = to_float(value)
    return bool(
        price is not None
        and abs(price) >= 100.0
        and abs(price - round(price)) <= 1e-6
        and min_price <= price <= max_price
    )


def american_implied_probability(price: float | None) -> float | None:
    if price is None or abs(price) < 100.0:
        return None
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def american_profit_per_unit(price: float | None) -> float | None:
    if price is None or abs(price) < 100.0:
        return None
    return price / 100.0 if price > 0 else 100.0 / abs(price)


def no_vig_probability(side_price: float | None, other_price: float | None) -> float | None:
    side = american_implied_probability(side_price)
    other = american_implied_probability(other_price)
    if side is None:
        return None
    if other is None or side + other <= 0:
        return side
    return side / (side + other)


def is_standard_line(target: str, line: float) -> bool:
    if target in STANDARD_MARKET_LINES:
        return abs(line - STANDARD_MARKET_LINES[target]) <= 1e-6
    return target in {"K", "ER"} and line >= 0.5 and abs((line * 2.0) - round(line * 2.0)) <= 1e-6


def poisson_probabilities(projection: float, line: float, direction: str) -> tuple[float, float, float]:
    lam = max(0.0, float(projection))
    floor_line = int(math.floor(line))
    probabilities: list[float] = []
    term = math.exp(-lam)
    probabilities.append(term)
    for value in range(1, max(0, floor_line) + 1):
        term *= lam / value
        probabilities.append(term)
    probability_at_or_below = min(1.0, sum(probabilities))
    is_integer_line = abs(line - round(line)) <= 1e-9
    push = probabilities[int(round(line))] if is_integer_line and int(round(line)) < len(probabilities) else 0.0
    if direction == "OVER":
        hit = max(0.0, 1.0 - probability_at_or_below)
    else:
        hit = max(0.0, probability_at_or_below - push) if is_integer_line else probability_at_or_below
    graded = hit / max(1e-9, 1.0 - push)
    return min(1.0, hit), min(1.0, push), min(1.0, graded)


def grade(actual: float, line: float, direction: str) -> str:
    if direction == "OVER":
        return "win" if actual > line else "push" if actual == line else "loss"
    return "win" if actual < line else "push" if actual == line else "loss"


def _feature_row(
    *,
    target: str,
    direction: str,
    player_type: str,
    projection: float,
    market_line: float,
    history_rows: int,
    market_books: int,
    market_line_std: float,
    side_price: float,
    other_price: float | None,
) -> dict[str, float | str]:
    hit, push, graded = poisson_probabilities(projection, market_line, direction)
    directional_edge = projection - market_line if direction == "OVER" else market_line - projection
    return {
        "target": target,
        "direction": direction,
        "player_type": player_type,
        "projection": projection,
        "market_line": market_line,
        "directional_edge": directional_edge,
        "edge_ratio": directional_edge / max(0.5, market_line),
        "model_hit_probability": hit,
        "model_graded_hit_rate": graded,
        "push_probability": push,
        "history_rows": float(max(0, history_rows)),
        "market_books": float(max(0, market_books)),
        "market_line_std": max(0.0, market_line_std),
        "market_implied_probability": no_vig_probability(side_price, other_price) or 0.5,
        "profit_per_unit": american_profit_per_unit(side_price) or 0.0,
    }


def build_historical_candidates(
    processed_root: Path,
    season: int,
    before_date: date,
    *,
    min_market_books: int = 2,
    min_price: float = -300.0,
    max_price: float = 250.0,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(processed_root.glob(f"*/{int(season)}_processed_processed.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame:
            continue
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame[frame["Date"].dt.date < before_date].sort_values("Date")
        if frame.empty:
            continue
        for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
            player_type = str(row.get("Player_Type", "")).strip().lower()
            for target in TARGETS:
                actual = to_float(row.get(target))
                line = to_float(row.get(f"Market_{target}"))
                gap = to_float(row.get(f"{target}_market_gap"))
                source = str(row.get(f"Market_Source_{target}", "")).strip().lower()
                books = int(to_float(row.get(f"Market_{target}_books")) or 0)
                if actual is None or line is None or gap is None or source != "real" or books < min_market_books:
                    continue
                if not is_standard_line(target, line):
                    continue
                projection = max(0.0, line + gap)
                over_price = to_float(row.get(f"Market_{target}_over_price"))
                under_price = to_float(row.get(f"Market_{target}_under_price"))
                for direction, side_price, other_price in (
                    ("OVER", over_price, under_price),
                    ("UNDER", under_price, over_price),
                ):
                    if not valid_american_price(side_price, min_price=min_price, max_price=max_price):
                        continue
                    result = grade(actual, line, direction)
                    if result == "push":
                        continue
                    features = _feature_row(
                        target=target,
                        direction=direction,
                        player_type=player_type,
                        projection=projection,
                        market_line=line,
                        history_rows=row_number - 1,
                        market_books=books,
                        market_line_std=to_float(row.get(f"Market_{target}_line_std")) or 0.0,
                        side_price=float(side_price),
                        other_price=other_price,
                    )
                    records.append(
                        {
                            **features,
                            "date": row["Date"].date().isoformat(),
                            "player": str(row.get("Player", path.parent.name)),
                            "game_id": str(row.get("Game_ID", "")).removesuffix(".0"),
                            "side_price": float(side_price),
                            "win": int(result == "win"),
                        }
                    )
    output = pd.DataFrame.from_records(records)
    if output.empty:
        return output
    return output.drop_duplicates(["date", "player", "game_id", "target", "direction"], keep="last").sort_values(
        ["date", "player", "target"]
    ).reset_index(drop=True)


def _design_matrix(
    rows: pd.DataFrame,
    *,
    means: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    numeric = rows.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    calculated_means = means or {name: float(numeric[name].median()) for name in numeric_features}
    numeric = numeric.fillna(calculated_means)
    calculated_scales = scales or {
        name: max(float(numeric[name].std(ddof=0)), 1e-6) for name in numeric_features
    }
    columns = [((numeric[name] - calculated_means[name]) / calculated_scales[name]).to_numpy() for name in numeric_features]
    for key in categorical_features:
        field, expected = key.split("=", 1)
        columns.append(rows[field].astype(str).str.lower().eq(expected.lower()).astype(float).to_numpy())
    return np.column_stack(columns), calculated_means, calculated_scales


def _fit(
    rows: pd.DataFrame,
    c_value: float,
    *,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> tuple[LogisticRegression, dict[str, float], dict[str, float]]:
    matrix, means, scales = _design_matrix(rows, numeric_features=numeric_features, categorical_features=categorical_features)
    model = LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs")
    model.fit(matrix, rows["win"].astype(int).to_numpy())
    return model, means, scales


def _predict(
    model: LogisticRegression,
    rows: pd.DataFrame,
    means: dict[str, float],
    scales: dict[str, float],
    *,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> np.ndarray:
    matrix, _, _ = _design_matrix(rows, means=means, scales=scales, numeric_features=numeric_features, categorical_features=categorical_features)
    return model.predict_proba(matrix)[:, 1]


def _rank_score(probability: float, price: float, roi_weight: float) -> float:
    profit = american_profit_per_unit(price) or 0.0
    expected_roi = probability * profit - (1.0 - probability)
    return probability + roi_weight * max(-1.0, min(2.0, expected_roi))


def _wilson_interval(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + (z * z / rows)
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt((probability * (1.0 - probability) / rows) + (z * z / (4.0 * rows * rows))) / denominator
    return center - margin, center + margin


def _day_block_roi_interval(selected: pd.DataFrame, *, samples: int = 2000) -> tuple[float | None, float | None]:
    if selected.empty:
        return None, None
    day_profits: list[list[float]] = []
    for _, day in selected.groupby("date", sort=True):
        day_profits.append(
            [
                (american_profit_per_unit(float(price)) or 0.0) if int(win) else -1.0
                for price, win in zip(day["side_price"], day["win"])
            ]
        )
    if not day_profits:
        return None, None
    rng = np.random.default_rng(20260806)
    estimates = []
    for _ in range(samples):
        sampled = [day_profits[index] for index in rng.integers(0, len(day_profits), len(day_profits))]
        estimates.append(float(np.mean([profit for day in sampled for profit in day])))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_ranked(rows: pd.DataFrame, probabilities: Iterable[float], *, top_k: int, roi_weight: float) -> dict[str, Any]:
    evaluated = rows.copy()
    evaluated["probability"] = np.asarray(list(probabilities), dtype=float)
    evaluated["rank_score"] = [
        _rank_score(float(probability), float(price), roi_weight)
        for probability, price in zip(evaluated["probability"], evaluated["side_price"])
    ]
    selected_parts: list[pd.DataFrame] = []
    parlay_returns: list[float] = []
    diversified_parlay_returns: list[float] = []
    for _, day in evaluated.groupby("date", sort=True):
        day = day.sort_values(["rank_score", "probability"], ascending=False)
        day = day.drop_duplicates("player", keep="first").head(top_k)
        if day.empty:
            continue
        selected_parts.append(day)
        if len(day) == top_k:
            decimal_odds = [1.0 + (american_profit_per_unit(float(price)) or 0.0) for price in day["side_price"]]
            parlay_returns.append(float(np.prod(decimal_odds) - 1.0) if bool(day["win"].all()) else -1.0)
        diversified = evaluated.loc[evaluated["date"].eq(day["date"].iloc[0])].sort_values(
            ["rank_score", "probability"], ascending=False
        )
        diversified = diversified.drop_duplicates("player", keep="first").drop_duplicates("game_id", keep="first").head(top_k)
        if len(diversified) == top_k:
            decimal_odds = [
                1.0 + (american_profit_per_unit(float(price)) or 0.0) for price in diversified["side_price"]
            ]
            diversified_parlay_returns.append(
                float(np.prod(decimal_odds) - 1.0) if bool(diversified["win"].all()) else -1.0
            )
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else evaluated.iloc[0:0]
    if selected.empty:
        return {"rows": 0, "days": 0, "wins": 0, "win_rate": None, "roi": None, "parlay_roi": None}
    profits = [
        (american_profit_per_unit(float(price)) or 0.0) if int(win) else -1.0
        for price, win in zip(selected["side_price"], selected["win"])
    ]
    win_rate_low, win_rate_high = _wilson_interval(int(selected["win"].sum()), len(selected))
    roi_low, roi_high = _day_block_roi_interval(selected)
    return {
        "rows": int(len(selected)),
        "days": int(selected["date"].nunique()),
        "wins": int(selected["win"].sum()),
        "win_rate": float(selected["win"].mean()),
        "win_rate_wilson_95_low": win_rate_low,
        "win_rate_wilson_95_high": win_rate_high,
        "roi": float(np.mean(profits)),
        "roi_day_bootstrap_95_low": roi_low,
        "roi_day_bootstrap_95_high": roi_high,
        "parlay_days": len(parlay_returns),
        "parlay_roi": float(np.mean(parlay_returns)) if parlay_returns else None,
        "diversified_parlay_constraint": "maximum_one_leg_per_game",
        "diversified_parlay_days": len(diversified_parlay_returns),
        "diversified_parlay_roi": float(np.mean(diversified_parlay_returns)) if diversified_parlay_returns else None,
    }


def _probability_metrics(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    actual = rows["win"].astype(float).to_numpy()
    return {
        "rows": int(len(rows)),
        "dates": int(rows["date"].nunique()),
        "wins": int(actual.sum()),
        "win_rate": float(actual.mean()),
        "mean_probability": float(probabilities.mean()),
        "brier_score": float(np.mean((probabilities - actual) ** 2)),
        "log_loss": float(-np.mean(actual * np.log(np.clip(probabilities, 1e-6, 1.0)) + (1.0 - actual) * np.log(np.clip(1.0 - probabilities, 1e-6, 1.0)))),
    }


def train_survival_model(
    rows: pd.DataFrame,
    *,
    top_k: int = 3,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES,
    model_version: str = MODEL_VERSION,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_train_dates: int = MIN_TRAIN_DATES,
) -> dict[str, Any]:
    """Real expanding-window walk-forward logistic training + rolling-
    origin validation + fixed holdout + asymmetric promotion/deployment
    gates. Feature set, model identity, and minimum-support bar are all
    parameterized (defaulting to this module's own pick_survival_model
    contract) so a second real model -- the v12 winner-signature model,
    train_winner_signature_model() below -- gets the exact same rigor
    through the same tested statistical core, never a hand-duplicated
    approximation of it."""
    dates = sorted(rows["date"].unique())
    if len(rows) < min_train_rows or len(dates) < min_train_dates:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "insufficient_support",
            "shadow_only": True,
            "training_rows": int(len(rows)),
            "training_dates": len(dates),
        }
    holdout_date_count = min(3, max(3, len(dates) // 4))
    development_dates = dates[:-holdout_date_count]
    holdout_dates = dates[-holdout_date_count:]
    initial_training_dates = max(4, len(development_dates) // 2)
    holdout = rows[rows["date"].isin(holdout_dates)]
    if len(development_dates) <= initial_training_dates or holdout.empty:
        return {
            "schema_version": 1,
            "model_version": model_version,
            "status": "insufficient_class_support",
            "shadow_only": True,
            "training_rows": int(len(rows)),
            "training_dates": len(dates),
        }
    trials: list[dict[str, Any]] = []
    oof_by_c: dict[float, pd.DataFrame] = {}
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3):
        fold_rows: list[pd.DataFrame] = []
        for fold_index in range(initial_training_dates, len(development_dates)):
            fold_train_dates = development_dates[:fold_index]
            fold_date = development_dates[fold_index]
            fold_train = rows[rows["date"].isin(fold_train_dates)]
            fold_validation = rows[rows["date"].eq(fold_date)]
            if fold_train["win"].nunique() < 2 or fold_validation.empty:
                continue
            model, means, scales = _fit(fold_train, c_value, numeric_features=numeric_features, categorical_features=categorical_features)
            predicted = fold_validation.copy()
            predicted["survival_probability"] = _predict(model, fold_validation, means, scales, numeric_features=numeric_features, categorical_features=categorical_features)
            fold_rows.append(predicted)
        if not fold_rows:
            continue
        oof_rows = pd.concat(fold_rows, ignore_index=True)
        oof_by_c[c_value] = oof_rows
        probability_metrics = _probability_metrics(
            oof_rows, oof_rows["survival_probability"].astype(float).to_numpy()
        )
        for roi_weight in (0.0, 0.025, 0.05, 0.10, 0.20):
            ranked = evaluate_ranked(
                oof_rows,
                oof_rows["survival_probability"].astype(float).to_numpy(),
                top_k=top_k,
                roi_weight=roi_weight,
            )
            win_rate = float(ranked.get("win_rate") or 0.0)
            roi = float(ranked.get("roi") or -1.0)
            score = win_rate + 0.15 * max(-0.5, min(0.5, roi)) - 0.05 * float(
                probability_metrics["brier_score"]
            )
            trials.append(
                {
                    "c": c_value,
                    "roi_weight": roi_weight,
                    "objective": score,
                    "brier_score": probability_metrics["brier_score"],
                    "top_k": ranked,
                }
            )
    if not trials:
        return {
            "schema_version": 1,
            "model_version": MODEL_VERSION,
            "status": "insufficient_walk_forward_support",
            "shadow_only": True,
            "training_rows": int(len(rows)),
            "training_dates": len(dates),
        }
    best = max(trials, key=lambda trial: (trial["objective"], -trial["brier_score"], -trial["c"]))
    selected_oof = oof_by_c[float(best["c"])]
    oof_probabilities = selected_oof["survival_probability"].astype(float).to_numpy()
    oof_baseline_probabilities = selected_oof["model_hit_probability"].astype(float).to_numpy()
    development = rows[rows["date"].isin(development_dates)]
    holdout_model, holdout_means, holdout_scales = _fit(development, float(best["c"]), numeric_features=numeric_features, categorical_features=categorical_features)
    holdout_probabilities = _predict(holdout_model, holdout, holdout_means, holdout_scales, numeric_features=numeric_features, categorical_features=categorical_features)
    baseline_probabilities = holdout["model_hit_probability"].astype(float).to_numpy()
    validation_survival = evaluate_ranked(
        selected_oof, oof_probabilities, top_k=top_k, roi_weight=float(best["roi_weight"])
    )
    validation_baseline = evaluate_ranked(
        selected_oof, oof_baseline_probabilities, top_k=top_k, roi_weight=0.0
    )
    holdout_survival = evaluate_ranked(
        holdout, holdout_probabilities, top_k=top_k, roi_weight=float(best["roi_weight"])
    )
    holdout_baseline = evaluate_ranked(holdout, baseline_probabilities, top_k=top_k, roi_weight=0.0)
    rolling_parts: list[pd.DataFrame] = []
    for fold_index in range(initial_training_dates, len(dates)):
        fold_train = rows[rows["date"].isin(dates[:fold_index])]
        fold_test = rows[rows["date"].eq(dates[fold_index])]
        if fold_train["win"].nunique() < 2 or fold_test.empty:
            continue
        rolling_model, rolling_means, rolling_scales = _fit(fold_train, float(best["c"]), numeric_features=numeric_features, categorical_features=categorical_features)
        predicted = fold_test.copy()
        predicted["survival_probability"] = _predict(
            rolling_model, fold_test, rolling_means, rolling_scales, numeric_features=numeric_features, categorical_features=categorical_features
        )
        rolling_parts.append(predicted)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    rolling_probabilities = rolling["survival_probability"].astype(float).to_numpy()
    rolling_baseline_probabilities = rolling["model_hit_probability"].astype(float).to_numpy()
    rolling_survival = evaluate_ranked(
        rolling, rolling_probabilities, top_k=top_k, roi_weight=float(best["roi_weight"])
    )
    rolling_baseline = evaluate_ranked(
        rolling, rolling_baseline_probabilities, top_k=top_k, roi_weight=0.0
    )
    promotion_checks = {
        "minimum_30_holdout_plays": int(holdout_survival.get("rows", 0)) >= 30,
        "minimum_10_holdout_dates": int(holdout_survival.get("days", 0)) >= 10,
        "win_rate_not_below_baseline": float(holdout_survival.get("win_rate") or 0.0)
        >= float(holdout_baseline.get("win_rate") or 0.0),
        "roi_not_below_baseline": float(holdout_survival.get("roi") or -1.0)
        >= float(holdout_baseline.get("roi") or -1.0),
        "brier_not_above_baseline": float(_probability_metrics(holdout, holdout_probabilities)["brier_score"])
        <= float(_probability_metrics(holdout, baseline_probabilities)["brier_score"]),
    }
    deployment_checks = {
        "minimum_18_rolling_picks": int(rolling_survival.get("rows", 0)) >= 18,
        "minimum_7_rolling_dates": int(rolling_survival.get("days", 0)) >= 7,
        "rolling_win_rate_not_below_baseline": float(rolling_survival.get("win_rate") or 0.0)
        >= float(rolling_baseline.get("win_rate") or 0.0),
        "rolling_roi_not_below_baseline": float(rolling_survival.get("roi") or -1.0)
        >= float(rolling_baseline.get("roi") or -1.0),
        "rolling_roi_lower_bound_positive": float(
            rolling_survival.get("roi_day_bootstrap_95_low") or -1.0
        ) > 0.0,
        "fixed_holdout_win_rate_not_below_baseline": float(holdout_survival.get("win_rate") or 0.0)
        >= float(holdout_baseline.get("win_rate") or 0.0),
        "fixed_holdout_roi_not_below_baseline": float(holdout_survival.get("roi") or -1.0)
        >= float(holdout_baseline.get("roi") or -1.0),
    }
    final_model, means, scales = _fit(rows, float(best["c"]), numeric_features=numeric_features, categorical_features=categorical_features)
    feature_names = [*numeric_features, *categorical_features]
    segment_support = {
        f"{target}|{direction}": int(len(segment))
        for (target, direction), segment in rows.groupby(["target", "direction"])
    }
    return {
        "schema_version": 1,
        "model_version": model_version,
        "status": "shadow",
        "shadow_only": True,
        "objective": "expanding_window_daily_top_k_win_rate_with_bounded_roi_and_brier_tiebreak",
        "calibration_method": "regularized_logistic_direct_probability",
        "candidate_sides": "both_confirmed_sides",
        "research_basis": {
            "time_ordered_validation": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
            "out_of_sample_calibration": "https://scikit-learn.org/stable/modules/calibration.html",
            "beta_calibration_considered": "https://proceedings.mlr.press/v54/kull17a.html",
            "selective_abstention": "https://proceedings.mlr.press/v130/gangrade21a.html",
        },
        "priced_rows_only": True,
        "real_market_rows_only": True,
        "strict_history_cutoff": True,
        "training_rows": int(len(rows)),
        "training_dates": len(dates),
        "training_start_date": dates[0],
        "training_end_date": dates[-1],
        "minimum_segment_rows": MIN_SEGMENT_ROWS,
        "segment_support": segment_support,
        "split": {
            "initial_training_end_date": development_dates[initial_training_dates - 1],
            "expanding_validation_start_date": development_dates[initial_training_dates],
            "expanding_validation_end_date": development_dates[-1],
            "expanding_validation_dates": len(development_dates) - initial_training_dates,
            "holdout_start_date": holdout_dates[0],
            "holdout_end_date": dates[-1],
        },
        "selected_hyperparameters": {"c": best["c"], "roi_weight": best["roi_weight"], "top_k": top_k},
        "expanding_oof_validation": {
            "probability_metrics": _probability_metrics(selected_oof, oof_probabilities),
            "survival_top_k": validation_survival,
            "baseline_top_k": validation_baseline,
        },
        "rolling_origin_validation": {
            "probability_metrics": _probability_metrics(rolling, rolling_probabilities),
            "baseline_probability_metrics": _probability_metrics(rolling, rolling_baseline_probabilities),
            "survival_top_k": rolling_survival,
            "baseline_top_k": rolling_baseline,
        },
        "validation_trials": [
            {
                "c": trial["c"],
                "roi_weight": trial["roi_weight"],
                "objective": trial["objective"],
                "brier_score": trial["brier_score"],
                "top_k_win_rate": trial["top_k"]["win_rate"],
                "top_k_roi": trial["top_k"]["roi"],
            }
            for trial in trials
        ],
        "holdout": {
            "probability_metrics": _probability_metrics(holdout, holdout_probabilities),
            "baseline_probability_metrics": _probability_metrics(holdout, baseline_probabilities),
            "survival_top_k": holdout_survival,
            "baseline_top_k": holdout_baseline,
        },
        "promotion_gate": {
            "decision": "eligible" if all(promotion_checks.values()) else "remain_shadow",
            "checks": promotion_checks,
        },
        "deployment_gate": {
            "authority": "shadow_only",
            "research_recommendation": "rank_tiebreaker" if all(deployment_checks.values()) else "remain_shadow",
            "probability_authority": "shadow_only",
            "affects_eligibility": False,
            "selection_score_band": 0.01,
            "checks": deployment_checks,
        },
        "feature_contract": {
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "means": means,
            "scales": scales,
            "coefficients": {name: float(value) for name, value in zip(feature_names, final_model.coef_[0])},
            "intercept": float(final_model.intercept_[0]),
        },
    }


def candidate_features(candidate: Any) -> dict[str, float | str]:
    raw = getattr(candidate, "raw", {}) or {}
    side_price = to_float(getattr(candidate, "selected_side_price", None)) or 0.0
    other_price = to_float(getattr(candidate, "opposite_side_price", None))
    return _feature_row(
        target=str(getattr(candidate, "target", "")).upper(),
        direction=str(getattr(candidate, "direction", "")).upper(),
        player_type=str(raw.get("Player_Type", "")).lower(),
        projection=float(getattr(candidate, "prediction", 0.0)),
        market_line=float(getattr(candidate, "market_line", 0.0)),
        history_rows=int(getattr(candidate, "history_rows", 0)),
        market_books=int(getattr(candidate, "market_books", 0)),
        market_line_std=float(getattr(candidate, "market_line_std", 0.0)),
        side_price=side_price,
        other_price=other_price,
    )


def apply_pick_survival_model(candidate: Any, payload: dict[str, Any] | None) -> tuple[float | None, float | None, str, int, bool]:
    if not isinstance(payload, dict) or payload.get("status") not in {"shadow", "active"}:
        return None, None, "disabled", 0, False
    run_date = getattr(candidate, "run_date", None)
    training_end = payload.get("training_end_date")
    if isinstance(run_date, date) and training_end and training_end >= run_date.isoformat():
        return None, None, "cutoff_violation", int(payload.get("training_rows", 0) or 0), False
    segment_key = f"{str(getattr(candidate, 'target', '')).upper()}|{str(getattr(candidate, 'direction', '')).upper()}"
    segment_support = int(payload.get("segment_support", {}).get(segment_key, 0) or 0)
    if segment_support < int(payload.get("minimum_segment_rows", MIN_SEGMENT_ROWS) or MIN_SEGMENT_ROWS):
        return None, None, "insufficient_segment_support", segment_support, False
    contract = payload.get("feature_contract", {})
    features = candidate_features(candidate)
    logit = float(contract.get("intercept", 0.0))
    for name in contract.get("numeric_features", []):
        mean = float(contract.get("means", {}).get(name, 0.0))
        scale = max(float(contract.get("scales", {}).get(name, 1.0)), 1e-6)
        value = float(features.get(name, mean))
        logit += ((value - mean) / scale) * float(contract.get("coefficients", {}).get(name, 0.0))
    for key in contract.get("categorical_features", []):
        field, expected = key.split("=", 1)
        active = float(str(features.get(field, "")).lower() == expected.lower())
        logit += active * float(contract.get("coefficients", {}).get(key, 0.0))
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
    profit = american_profit_per_unit(to_float(getattr(candidate, "selected_side_price", None)))
    expected_roi = None if profit is None else probability * profit - (1.0 - probability)
    rank_active = bool(
        payload.get("status") == "active"
        and not bool(payload.get("shadow_only", True))
        and payload.get("deployment_gate", {}).get("authority") == "rank_tiebreaker"
    )
    return probability, expected_roi, MODEL_VERSION, segment_support, bool(rank_active)


def _with_disagreement_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    model_probability = pd.to_numeric(rows.get("model_hit_probability"), errors="coerce")
    market_probability = pd.to_numeric(rows.get("market_implied_probability"), errors="coerce")
    survival_probability = pd.to_numeric(rows.get("survival_probability"), errors="coerce")
    rows["model_market_disagreement"] = (model_probability - market_probability).abs()
    rows["model_survival_disagreement"] = (model_probability - survival_probability).abs()
    return rows


def train_winner_signature_model(rows: pd.DataFrame, *, top_k: int = 3) -> dict[str, Any]:
    """v12 Phase 1: trains P(win | v11-eligible) -- rows is expected to
    already be the v11-eligible, real-settled population (see
    build_v11_eligible_training_set.py), not the broader population
    pick_survival_model's own build_historical_candidates() produces.
    Reuses train_survival_model()'s exact statistical core (expanding-OOF
    + rolling-origin + fixed holdout + asymmetric promotion/deployment
    gates) with this model's own feature set and identity."""
    enriched = _with_disagreement_features(rows)
    return train_survival_model(
        enriched,
        top_k=top_k,
        numeric_features=WINNER_SIGNATURE_NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        model_version=WINNER_SIGNATURE_MODEL_VERSION,
    )


def candidate_winner_signature_features(candidate: Any) -> dict[str, float | str]:
    raw = getattr(candidate, "raw", {}) or {}
    model_probability = to_float(getattr(candidate, "calibrated_hit_probability", None))
    market_probability = to_float(getattr(candidate, "market_implied_probability", None))
    survival_probability = to_float(getattr(candidate, "survival_probability", None))
    return {
        "target": str(getattr(candidate, "target", "")).upper(),
        "direction": str(getattr(candidate, "direction", "")).upper(),
        "player_type": str(raw.get("Player_Type", "")).lower(),
        "directional_edge": to_float(getattr(candidate, "edge", None)) or 0.0,
        "abs_edge": to_float(getattr(candidate, "abs_edge", None)) or 0.0,
        "model_hit_probability": model_probability if model_probability is not None else 0.0,
        "market_implied_probability": market_probability if market_probability is not None else 0.0,
        "market_line_std": to_float(getattr(candidate, "market_line_std", None)) or 0.0,
        "market_books": float(getattr(candidate, "market_books", 0) or 0),
        "market_common_books": float(getattr(candidate, "market_common_books", 0) or 0),
        "history_rows": float(getattr(candidate, "history_rows", 0) or 0),
        "historical_bucket_win_rate": to_float(getattr(candidate, "historical_bucket_win_rate", None)) or 0.0,
        "historical_bucket_support": float(getattr(candidate, "historical_bucket_support", 0) or 0),
        "historical_bet_profile_win_rate": to_float(getattr(candidate, "historical_bet_profile_win_rate", None)) or 0.0,
        "historical_bet_profile_roi": to_float(getattr(candidate, "historical_bet_profile_roi", None)) or 0.0,
        "historical_bet_profile_support": float(getattr(candidate, "historical_bet_profile_support", 0) or 0),
        "historical_market_availability_rate": to_float(getattr(candidate, "historical_market_availability_rate", None)) or 0.0,
        "historical_market_availability_support": float(getattr(candidate, "historical_market_availability_support", 0) or 0),
        "live_confidence_calibration_adjustment": to_float(getattr(candidate, "live_confidence_calibration_adjustment", None)) or 0.0,
        "model_market_disagreement": abs((model_probability or 0.0) - (market_probability or 0.0)),
        "model_survival_disagreement": abs((model_probability or 0.0) - survival_probability) if survival_probability is not None else 0.0,
    }


def apply_winner_signature_model(
    candidate: Any, payload: dict[str, Any] | None
) -> tuple[float | None, float | None, float | None, float | None, bool, str, int]:
    """Real, negative-authority-only SafeEV computation. Returns
    (winner_signature_probability, safe_probability, safe_expected_value,
    safe_probability_edge, safe_ev_veto, status, support). safe_probability
    is ALWAYS min(v11's own calibrated_hit_probability, this model's
    output) when both are real -- this model can flag a v11-eligible pick
    as more fragile than its own probability implies, it can never raise
    a pick's probability. safe_ev_veto is a visible flag only in this
    phase, never a filter -- v11's actual selection is unaffected."""
    v11_probability = to_float(getattr(candidate, "calibrated_hit_probability", None))
    if not isinstance(payload, dict) or payload.get("status") not in {"shadow", "active"}:
        return None, v11_probability, None, None, False, "disabled", 0
    run_date = getattr(candidate, "run_date", None)
    training_end = payload.get("training_end_date")
    if isinstance(run_date, date) and training_end and training_end >= run_date.isoformat():
        return None, v11_probability, None, None, False, "cutoff_violation", int(payload.get("training_rows", 0) or 0)
    segment_key = f"{str(getattr(candidate, 'target', '')).upper()}|{str(getattr(candidate, 'direction', '')).upper()}"
    segment_support = int(payload.get("segment_support", {}).get(segment_key, 0) or 0)
    if segment_support < int(payload.get("minimum_segment_rows", MIN_SEGMENT_ROWS) or MIN_SEGMENT_ROWS):
        return None, v11_probability, None, None, False, "insufficient_segment_support", segment_support

    contract = payload.get("feature_contract", {})
    features = candidate_winner_signature_features(candidate)
    logit = float(contract.get("intercept", 0.0))
    for name in contract.get("numeric_features", []):
        mean = float(contract.get("means", {}).get(name, 0.0))
        scale = max(float(contract.get("scales", {}).get(name, 1.0)), 1e-6)
        value = float(features.get(name, mean))
        logit += ((value - mean) / scale) * float(contract.get("coefficients", {}).get(name, 0.0))
    for key in contract.get("categorical_features", []):
        field, expected = key.split("=", 1)
        active = float(str(features.get(field, "")).lower() == expected.lower())
        logit += active * float(contract.get("coefficients", {}).get(key, 0.0))
    winner_signature_probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    if v11_probability is None:
        safe_probability = None
    else:
        safe_probability = min(v11_probability, winner_signature_probability)

    decimal_price = None
    side_price = to_float(getattr(candidate, "selected_side_price", None))
    if side_price is not None:
        profit = american_profit_per_unit(side_price)
        decimal_price = None if profit is None else 1.0 + profit
    safe_expected_value = None
    safe_probability_edge = None
    if safe_probability is not None and decimal_price is not None:
        safe_expected_value = safe_probability * decimal_price - 1.0
        safe_probability_edge = safe_probability - (1.0 / decimal_price)

    veto_margin = 0.02  # a real, disclosed threshold -- not tuned against this phase's own thin holdout
    safe_ev_veto = bool(
        v11_probability is not None and safe_probability is not None and safe_probability < v11_probability - veto_margin
    )
    return (
        winner_signature_probability,
        safe_probability,
        safe_expected_value,
        safe_probability_edge,
        safe_ev_veto,
        "active" if bool(payload.get("status") == "active") else "shadow",
        segment_support,
    )


# Real isotonic recalibration of hit probability against real settled
# outcomes -- see hit_probability_calibration.py for the training side
# (harvest_calibration_rows/train_hit_probability_calibration), which
# imports select_high_precision_predictions.py to build its training
# rows. The apply/load side lives here instead, matching this module's
# existing role as the dependency-free half of every shadow/veto model
# select_high_precision_predictions.py applies -- putting it there
# instead would create an import cycle (that module already imports
# FROM this one).
def apply_hit_probability_calibration(
    model_hit_probability: float | None, payload: dict[str, Any] | None
) -> tuple[float | None, str, int]:
    """Real, negative-authority-only application: returns
    (historically_calibrated_hit_probability, status, training_rows).
    Callers combine this with the existing calibrated_hit_probability via
    min(...) -- this can only ever pull a candidate's effective
    probability DOWN toward what real settled outcomes actually support
    (found by direct investigation: a real n=6,084 sample showed a "70%"
    candidate wins about 63% of the time), never raise it. Returns
    (None, "disabled", 0) if the model isn't active/shadow, has no
    breakpoints, or the input probability is missing."""
    if not isinstance(payload, dict) or payload.get("status") not in {"shadow", "active"}:
        return None, "disabled", 0
    breakpoints = payload.get("breakpoints") or []
    if not breakpoints or model_hit_probability is None:
        return None, "disabled", 0
    xs = [float(point[0]) for point in breakpoints]
    ys = [float(point[1]) for point in breakpoints]
    calibrated = float(np.interp(float(model_hit_probability), xs, ys))
    calibrated = max(0.0, min(1.0, calibrated))
    return calibrated, str(payload.get("status")), int(payload.get("training_rows", 0) or 0)


def load_hit_probability_calibration(path: Path | None, run_date: date | None) -> dict | None:
    """Same real load/point-in-time-safety contract as load_pick_
    survival_model(): a model trained using a given date's own outcome
    can never be applied when replaying that same date."""
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    training_end = payload.get("training_end_date")
    if isinstance(run_date, date) and training_end and str(training_end) >= run_date.isoformat():
        return None
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--before-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-market-books", type=int, default=2)
    parser.add_argument("--min-price", type=float, default=-300.0)
    parser.add_argument("--max-price", type=float, default=250.0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or DEFAULT_OUTPUT_ROOT / f"pick_survival_model_{args.season}.json"
    rows = build_historical_candidates(
        args.processed_root.resolve(),
        int(args.season),
        args.before_date,
        min_market_books=int(args.min_market_books),
        min_price=float(args.min_price),
        max_price=float(args.max_price),
    )
    payload = train_survival_model(rows, top_k=max(1, int(args.top_k)))
    payload.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "history_before_date": args.before_date.isoformat(),
            "processed_root": portable_path(args.processed_root),
            "market_guardrails": {
                "minimum_books": int(args.min_market_books),
                "minimum_american_price": float(args.min_price),
                "maximum_american_price": float(args.max_price),
                "standard_lines_only": True,
            },
        }
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Pick survival model: {output_json}")
    print(f"Status: {payload['status']}; rows={payload['training_rows']}; dates={payload['training_dates']}")
    if payload.get("holdout"):
        baseline = payload["holdout"]["baseline_top_k"]
        survival = payload["holdout"]["survival_top_k"]
        print(f"Holdout top-k win rate: {baseline['win_rate']} -> {survival['win_rate']}")
        print(f"Holdout top-k ROI: {baseline['roi']} -> {survival['roi']}")


if __name__ == "__main__":
    main()
