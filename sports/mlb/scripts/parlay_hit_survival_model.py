#!/usr/bin/env python3
"""Role-aware, chronologically validated survival model for MLB hit parlay legs."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


MODEL_VERSION = "mlb_parlay_hit_survival_role_hgb_v1"
EVIDENCE_LABEL = "SYNTHETIC_LINE_HIT_RATE_DIAGNOSTIC_NO_ROI_CLAIM"
FEATURES = (
    "projection",
    "baseline",
    "last_hits",
    "batting_order",
    "history_rows",
    "is_home",
)
MIN_HISTORY_ROWS = 35
MIN_TRAINING_DATES = 45
MIN_TRAINING_ROWS = 3000
FROZEN_TRAINING_CUTOFF_EXCLUSIVE = date(2026, 7, 30)


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def normalize_player(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _beta_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.column_stack((np.log(clipped), -np.log1p(-clipped)))


def _metrics(actual: pd.Series, probability: np.ndarray) -> dict[str, float | int | None]:
    if len(actual) == 0:
        return {"rows": 0, "wins": 0, "hit_rate": None, "mean_probability": None, "brier_score": None, "log_loss": None}
    labels = actual.astype(int).to_numpy()
    predictions = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    return {
        "rows": int(len(labels)),
        "wins": int(labels.sum()),
        "hit_rate": float(labels.mean()),
        "mean_probability": float(predictions.mean()),
        "brier_score": float(brier_score_loss(labels, predictions)),
        "log_loss": float(log_loss(labels, predictions, labels=[0, 1])),
    }


def _wilson_interval(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + (z * z / rows)
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt((probability * (1.0 - probability) / rows) + (z * z / (4.0 * rows * rows))) / denominator
    return center - margin, center + margin


def _daily_top_two_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int | None]:
    if frame.empty:
        return {"legs": 0, "slates": 0, "leg_hit_rate": None, "ticket_hit_rate": None}
    ranked = frame.copy()
    ranked["_probability"] = probability
    selected: list[pd.Series] = []
    for _, slate in ranked.groupby("date", sort=True):
        games: set[str] = set()
        for _, row in slate.sort_values(["_probability", "projection"], ascending=False, kind="stable").iterrows():
            game_id = str(row.get("game_id") or "")
            if game_id and game_id in games:
                continue
            selected.append(row)
            if game_id:
                games.add(game_id)
            if len(games) >= 2 or (not game_id and len(selected) >= 2):
                break
    if not selected:
        return {"legs": 0, "slates": 0, "leg_hit_rate": None, "ticket_hit_rate": None}
    output = pd.DataFrame(selected)
    by_date = output.groupby("date", sort=True)["win"]
    complete = by_date.size().eq(2)
    ticket_results = by_date.min().loc[complete]
    leg_wins = int(output["win"].sum())
    ticket_wins = int(ticket_results.sum())
    leg_low, leg_high = _wilson_interval(leg_wins, len(output))
    ticket_low, ticket_high = _wilson_interval(ticket_wins, len(ticket_results))
    return {
        "legs": int(len(output)),
        "leg_wins": leg_wins,
        "slates": int(len(ticket_results)),
        "ticket_wins": ticket_wins,
        "leg_hit_rate": float(output["win"].mean()),
        "leg_hit_rate_wilson_95_low": leg_low,
        "leg_hit_rate_wilson_95_high": leg_high,
        "ticket_hit_rate": float(ticket_results.mean()) if len(ticket_results) else None,
        "ticket_hit_rate_wilson_95_low": ticket_low,
        "ticket_hit_rate_wilson_95_high": ticket_high,
        "mean_probability": float(output["_probability"].mean()),
    }


def build_training_rows(processed_root: Path, *, before_date: date) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    records: list[pd.DataFrame] = []
    latest_context: dict[str, dict[str, float]] = {}
    required = {
        "Date",
        "Game_ID",
        "H",
        "Market_H",
        "H_market_gap",
        "H_rolling_avg",
        "Batting_Order",
        "Is_Home",
        "Did_Not_Play",
    }
    for path in sorted(processed_root.glob("*/20*_processed_processed.csv")):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["_date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[frame["_date"].dt.date < before_date].sort_values("_date", kind="stable")
        if frame.empty:
            continue
        frame["_history_rows"] = np.arange(len(frame), dtype=int)
        frame["_baseline"] = pd.to_numeric(frame["H_rolling_avg"], errors="coerce").shift(1)
        frame["_last_hits"] = pd.to_numeric(frame["H"], errors="coerce").shift(1)
        line = pd.to_numeric(frame["Market_H"], errors="coerce")
        projection = line + pd.to_numeric(frame["H_market_gap"], errors="coerce")
        did_not_play = pd.to_numeric(frame["Did_Not_Play"], errors="coerce").fillna(0)
        player_name = str(frame.iloc[-1].get("Player") or path.parent.name)
        latest = frame.iloc[-1]
        latest_context[normalize_player(player_name)] = {
            "last_hits": _number(latest.get("H"), 0.0),
            "recent_batting_order": _number(latest.get("Batting_Order"), 6.0),
        }
        eligible = pd.DataFrame(
            {
                "date": frame["_date"].dt.date.astype(str),
                "game_id": frame["Game_ID"].astype(str),
                "projection": projection,
                "baseline": frame["_baseline"],
                "last_hits": frame["_last_hits"],
                "batting_order": pd.to_numeric(frame["Batting_Order"], errors="coerce"),
                "history_rows": frame["_history_rows"].astype(float),
                "is_home": pd.to_numeric(frame["Is_Home"], errors="coerce"),
                "win": (pd.to_numeric(frame["H"], errors="coerce") > 0.5).astype(int),
                "line": line,
                "did_not_play": did_not_play,
            }
        )
        eligible = eligible.loc[
            eligible["line"].sub(0.5).abs().le(1e-9)
            & eligible["history_rows"].ge(MIN_HISTORY_ROWS)
            & eligible["did_not_play"].eq(0)
        ]
        records.append(eligible)
    if not records:
        return pd.DataFrame(columns=["date", "game_id", *FEATURES, "win"]), latest_context
    output = pd.concat(records, ignore_index=True)
    output = output.dropna(subset=["date", *FEATURES, "win"]).sort_values(["date", "game_id"], kind="stable")
    return output.reset_index(drop=True), latest_context


@dataclass
class HitSurvivalBundle:
    model: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    latest_context: dict[str, dict[str, float]]
    report: dict[str, Any]

    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        row = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)
        raw = float(self.model.predict_proba(row)[0, 1])
        calibrated = float(self.calibrator.predict_proba(_beta_features(np.asarray([raw])))[0, 1])
        return raw, calibrated


def fit_hit_survival_model(processed_root: Path, *, before_date: date) -> HitSurvivalBundle | None:
    rows, latest_context = build_training_rows(processed_root, before_date=before_date)
    effective_cutoff = min(before_date, FROZEN_TRAINING_CUTOFF_EXCLUSIVE)
    rows = rows.loc[rows["date"] < effective_cutoff.isoformat()].copy()
    dates = sorted(rows["date"].unique()) if not rows.empty else []
    if len(rows) < MIN_TRAINING_ROWS or len(dates) < MIN_TRAINING_DATES:
        return None
    development_end = max(1, int(len(dates) * 0.70))
    calibration_end = max(development_end + 1, int(len(dates) * 0.85))
    development_dates = set(dates[:development_end])
    calibration_dates = set(dates[development_end:calibration_end])
    holdout_dates = set(dates[calibration_end:])
    development = rows.loc[rows["date"].isin(development_dates)]
    calibration = rows.loc[rows["date"].isin(calibration_dates)]
    holdout = rows.loc[rows["date"].isin(holdout_dates)]
    if development.empty or calibration.empty or holdout.empty:
        return None

    model = HistGradientBoostingClassifier(
        max_leaf_nodes=7,
        max_iter=100,
        learning_rate=0.05,
        l2_regularization=10.0,
        min_samples_leaf=100,
        random_state=20260813,
    )
    model.fit(development.loc[:, FEATURES], development["win"].astype(int))
    calibration_raw = model.predict_proba(calibration.loc[:, FEATURES])[:, 1]
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, solver="lbfgs")
    calibrator.fit(_beta_features(calibration_raw), calibration["win"].astype(int))
    calibration_probability = calibrator.predict_proba(_beta_features(calibration_raw))[:, 1]
    holdout_raw = model.predict_proba(holdout.loc[:, FEATURES])[:, 1]
    holdout_probability = calibrator.predict_proba(_beta_features(holdout_raw))[:, 1]
    baseline_probability = 1.0 - np.exp(-holdout["projection"].clip(lower=0.0).to_numpy())
    report = {
        "model_version": MODEL_VERSION,
        "status": "development_shadow",
        "evidence_label": EVIDENCE_LABEL,
        "claim_scope": "hit-rate ranking and confidence diagnostic only; executable prices are required for ROI",
        "features": list(FEATURES),
        "training_rows": int(len(rows)),
        "training_dates": int(len(dates)),
        "frozen_training_cutoff_exclusive": effective_cutoff.isoformat(),
        "partitions": {
            "development": {"start": min(development_dates), "end": max(development_dates)},
            "calibration": {"start": min(calibration_dates), "end": max(calibration_dates)},
            "locked_recent_holdout": {"start": min(holdout_dates), "end": max(holdout_dates)},
        },
        "calibration": _metrics(calibration["win"], calibration_probability),
        "locked_recent_holdout": {
            **_metrics(holdout["win"], holdout_probability),
            "daily_top_two": _daily_top_two_metrics(holdout, holdout_probability),
            "projection_baseline_daily_top_two": _daily_top_two_metrics(holdout, baseline_probability),
        },
    }
    return HitSurvivalBundle(model=model, calibrator=calibrator, latest_context=latest_context, report=report)


def candidate_features(
    candidate: Any,
    bundle: HitSurvivalBundle,
    *,
    confirmed_batting_order: float | None = None,
) -> tuple[dict[str, float], str]:
    raw = getattr(candidate, "raw", {}) or {}
    context = bundle.latest_context.get(normalize_player(getattr(candidate, "player", "")), {})
    batting_order = confirmed_batting_order
    batting_order_source = "confirmed_lineup"
    if batting_order is None or not math.isfinite(float(batting_order)):
        batting_order = _number(context.get("recent_batting_order"), 6.0)
        batting_order_source = "prior_start_proxy"
    return (
        {
            "projection": float(getattr(candidate, "prediction", 0.0)),
            "baseline": _number(raw.get("Baseline"), float(getattr(candidate, "prediction", 0.0))),
            "last_hits": _number(context.get("last_hits"), 0.0),
            "batting_order": max(1.0, min(9.0, float(batting_order))),
            "history_rows": float(getattr(candidate, "history_rows", 0)),
            "is_home": _number(raw.get("Is_Home"), 0.0),
        },
        batting_order_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--before-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = fit_hit_survival_model(args.processed_root.resolve(), before_date=args.before_date)
    if bundle is None:
        raise SystemExit("Insufficient history to fit the hit-survival model")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(bundle.report, indent=2) + "\n", encoding="utf-8")
    holdout = bundle.report["locked_recent_holdout"]["daily_top_two"]
    baseline = bundle.report["locked_recent_holdout"]["projection_baseline_daily_top_two"]
    print(
        f"Holdout top-two legs: {holdout['leg_hit_rate']:.3f} vs {baseline['leg_hit_rate']:.3f}; "
        f"tickets: {holdout['ticket_hit_rate']:.3f} vs {baseline['ticket_hit_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
