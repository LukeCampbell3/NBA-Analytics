"""Chronological Formula 1 race-outcome model."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import log
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "grid_position",
    "grid_known",
    "driver_starts",
    "driver_average_finish",
    "driver_recent_finish",
    "driver_win_rate",
    "driver_podium_rate",
    "driver_dnf_rate",
    "constructor_average_finish",
    "constructor_recent_finish",
    "constructor_win_rate",
    "circuit_average_finish",
    "season_points_share",
    "standing_position",
)


@dataclass
class Form:
    starts: int = 0
    finish_sum: float = 0.0
    wins: int = 0
    podiums: int = 0
    dnfs: int = 0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def update(self, finish: int, dnf: bool) -> None:
        self.starts += 1
        self.finish_sum += finish
        self.wins += int(finish == 1)
        self.podiums += int(finish <= 3)
        self.dnfs += int(dnf)
        self.recent.append(float(finish))


class FeatureState:
    def __init__(self) -> None:
        self.drivers: defaultdict[str, Form] = defaultdict(Form)
        self.constructors: defaultdict[str, Form] = defaultdict(Form)
        self.circuits: defaultdict[tuple[str, str], Form] = defaultdict(Form)
        self.season_points: defaultdict[tuple[int, str], float] = defaultdict(float)
        self.season_starts: defaultdict[tuple[int, str], int] = defaultdict(int)

    @staticmethod
    def _mean(form: Form, default: float = 11.5) -> float:
        return form.finish_sum / form.starts if form.starts else default

    @staticmethod
    def _recent(form: Form, default: float = 11.5) -> float:
        return sum(form.recent) / len(form.recent) if form.recent else default

    def features(self, entry: dict[str, Any], event: dict[str, Any]) -> list[float]:
        driver = self.drivers[entry["driver_id"]]
        constructor = self.constructors[entry["constructor_id"]]
        circuit = self.circuits[(entry["driver_id"], event.get("circuit_id", ""))]
        grid = int(entry.get("grid") or 0)
        season = int(event["season"])
        season_points = self.season_points[(season, entry["driver_id"])]
        max_points = max(
            [value for (year, _), value in self.season_points.items() if year == season] or [0.0]
        )
        starts = self.season_starts[(season, entry["driver_id"])]
        standing = int(entry.get("standing_position") or (1 + sum(
            self.season_points[(season, other)] > season_points
            for (year, other) in self.season_points
            if year == season
        )))
        return [
            float(grid if grid > 0 else 11.5) / 22.0,
            float(grid > 0),
            min(driver.starts, 100) / 100.0,
            self._mean(driver) / 22.0,
            self._recent(driver) / 22.0,
            driver.wins / driver.starts if driver.starts else 0.0,
            driver.podiums / driver.starts if driver.starts else 0.0,
            driver.dnfs / driver.starts if driver.starts else 0.20,
            self._mean(constructor) / 22.0,
            self._recent(constructor) / 22.0,
            constructor.wins / constructor.starts if constructor.starts else 0.0,
            self._mean(circuit) / 22.0,
            season_points / max_points if max_points > 0 else 0.0,
            float(standing) / 22.0 if starts or entry.get("standing_position") else 0.5,
        ]

    def update_race(self, race: dict[str, Any]) -> None:
        season = int(race["season"])
        for row in race["results"]:
            finish = int(row["finish"])
            dnf = bool(row.get("dnf"))
            self.drivers[row["driver_id"]].update(finish, dnf)
            self.constructors[row["constructor_id"]].update(finish, dnf)
            self.circuits[(row["driver_id"], race.get("circuit_id", ""))].update(finish, dnf)
            self.season_points[(season, row["driver_id"])] += float(row.get("points") or 0.0)
            self.season_starts[(season, row["driver_id"])] += 1


def build_training_rows(history: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], FeatureState]:
    state = FeatureState()
    features: list[list[float]] = []
    labels = {"win": [], "podium": [], "top6": []}
    race_ids: list[str] = []
    for race in history:
        race_id = f"{race['season']}-{race['round']}"
        for result in race["results"]:
            entry = {**result, "standing_position": None}
            features.append(state.features(entry, race))
            finish = int(result["finish"])
            labels["win"].append(int(finish == 1))
            labels["podium"].append(int(finish <= 3))
            labels["top6"].append(int(finish <= 6))
            race_ids.append(race_id)
        state.update_race(race)
    return (
        np.asarray(features, dtype=float),
        {key: np.asarray(values, dtype=int) for key, values in labels.items()},
        race_ids,
        state,
    )


def _fit_classifier(x: np.ndarray, y: np.ndarray) -> Any:
    if len(np.unique(y)) < 2:
        raise ValueError("F1 training history must contain both positive and negative outcomes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, max_iter=2000, class_weight=None, random_state=17),
    )
    model.fit(x, y)
    return model


def _normalize_to_total(values: np.ndarray, total: float) -> np.ndarray:
    values = np.clip(values, 1e-9, 1.0)
    if not len(values):
        return values
    result = np.zeros_like(values, dtype=float)
    remaining = np.ones(len(values), dtype=bool)
    remaining_total = min(float(total), float(len(values)))
    weights = values.astype(float).copy()
    while remaining.any() and remaining_total > 1e-12:
        active_weights = weights[remaining]
        weight_sum = float(active_weights.sum())
        allocation = (
            active_weights * (remaining_total / weight_sum)
            if weight_sum > 0
            else np.full(int(remaining.sum()), remaining_total / int(remaining.sum()))
        )
        capped = allocation >= 1.0
        active_indices = np.flatnonzero(remaining)
        if not capped.any():
            result[active_indices] = allocation
            remaining_total = 0.0
            break
        capped_indices = active_indices[capped]
        result[capped_indices] = 1.0
        remaining[capped_indices] = False
        remaining_total -= float(len(capped_indices))
    return result


def train_and_evaluate(history: list[dict[str, Any]]) -> tuple[dict[str, Any], FeatureState, dict[str, Any]]:
    x, labels, race_ids, final_state = build_training_rows(history)
    if len(x) < 200:
        raise ValueError(f"At least 200 historical driver-race rows are required; received {len(x)}")
    unique_races = list(dict.fromkeys(race_ids))
    split_index = max(1, int(len(unique_races) * 0.8))
    train_races = set(unique_races[:split_index])
    test_races = set(unique_races[split_index:])
    train_mask = np.asarray([race_id in train_races for race_id in race_ids])
    test_mask = np.asarray([race_id in test_races for race_id in race_ids])
    if not test_mask.any():
        test_mask = ~train_mask

    backtest_models = {target: _fit_classifier(x[train_mask], y[train_mask]) for target, y in labels.items()}
    backtest_probabilities = {
        target: model.predict_proba(x[test_mask])[:, 1] for target, model in backtest_models.items()
    }
    test_ids = np.asarray(race_ids)[test_mask]
    test_win = labels["win"][test_mask]
    winner_logs: list[float] = []
    top_pick_hits = 0
    for race_id in dict.fromkeys(test_ids.tolist()):
        mask = test_ids == race_id
        normalized = _normalize_to_total(backtest_probabilities["win"][mask], 1.0)
        actual = test_win[mask]
        actual_index = int(np.argmax(actual))
        winner_logs.append(-log(max(float(normalized[actual_index]), 1e-9)))
        top_pick_hits += int(int(np.argmax(normalized)) == actual_index)

    metrics = {
        "holdout_races": len(set(test_ids.tolist())),
        "holdout_rows": int(test_mask.sum()),
        "winner_top_pick_accuracy": top_pick_hits / max(1, len(set(test_ids.tolist()))),
        "winner_log_loss": float(np.mean(winner_logs)) if winner_logs else None,
        "winner_brier": float(brier_score_loss(test_win, backtest_probabilities["win"])),
        "podium_brier": float(brier_score_loss(labels["podium"][test_mask], backtest_probabilities["podium"])),
        "top6_brier": float(brier_score_loss(labels["top6"][test_mask], backtest_probabilities["top6"])),
        "evaluation": "chronological final-20-percent race holdout",
    }
    final_models = {target: _fit_classifier(x, y) for target, y in labels.items()}
    metadata = {
        "name": "F1 chronological form logistic ensemble v1",
        "feature_names": list(FEATURE_NAMES),
        "training_rows": len(x),
        "training_races": len(unique_races),
        "trained_through": history[-1]["date"] if history else None,
        "backtest": metrics,
    }
    return final_models, final_state, metadata


def predict_event(
    models: dict[str, Any], state: FeatureState, event: dict[str, Any], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not entries:
        return []
    x = np.asarray([state.features(entry, event) for entry in entries], dtype=float)
    raw = {target: model.predict_proba(x)[:, 1] for target, model in models.items()}
    probabilities = {
        "win": _normalize_to_total(raw["win"], 1.0),
        "podium": np.minimum(_normalize_to_total(raw["podium"], 3.0), 1.0),
        "top6": np.minimum(_normalize_to_total(raw["top6"], 6.0), 1.0),
    }
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        rows.append(
            {
                "driver_id": entry["driver_id"],
                "driver": entry["driver"],
                "driver_number": entry.get("driver_number", ""),
                "constructor": entry.get("constructor", ""),
                "grid_position": int(entry.get("grid") or 0) or None,
                "standing_position": entry.get("standing_position"),
                "win_probability": float(probabilities["win"][index]),
                "podium_probability": float(probabilities["podium"][index]),
                "top6_probability": float(probabilities["top6"][index]),
            }
        )
    rows.sort(key=lambda row: row["win_probability"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["model_rank"] = rank
    return rows
