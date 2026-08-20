from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .protocol import ALLOCATION_PATH_PROTOCOL, AllocationPathProtocol


ENDPOINT_FEATURES = [
    "close_share",
    "close_team_total",
    "close_hhi",
    "close_entropy",
]
PATH_FEATURES = ENDPOINT_FEATURES + [
    "delta_share",
    "player_total_variation",
    "player_path_efficiency",
    "direction_reversals",
    "allocation_displacement_l1",
    "allocation_path_length_l1",
    "allocation_path_efficiency",
    "delta_hhi",
    "delta_entropy",
]
FROZEN_CHECKPOINTS = (20, 30, 50)


@dataclass(frozen=True)
class ConfirmationResult:
    player_predictions: pd.DataFrame
    event_evaluations: pd.DataFrame
    report: dict[str, Any]


def _model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def _simplex_projection(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(len(clipped), 1.0 / len(clipped), dtype=float)
    return clipped / total


def _paired_bootstrap(
    values: np.ndarray,
    *,
    alpha: float,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    one_sided_lcb = float(np.quantile(means, alpha))
    two_sided_low = float(np.quantile(means, alpha / 2.0))
    two_sided_high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return one_sided_lcb, two_sided_low, two_sided_high


def _one_sided_sign_flip_pvalue(
    values: np.ndarray,
    *,
    practical_delta: float,
    samples: int,
    seed: int,
) -> float:
    centered = np.asarray(values, dtype=float) - practical_delta
    if len(centered) == 0:
        return np.nan
    observed = float(centered.mean())
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceedances = 0
    remaining = int(samples)
    chunk_size = 5_000
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(chunk, len(centered)))
        permuted = (signs * centered).mean(axis=1)
        exceedances += int(np.sum(permuted >= observed))
        remaining -= chunk
    return float((exceedances + 1) / (samples + 1))


def evaluate_improvement_sequence(
    improvements: np.ndarray | pd.Series,
    *,
    protocol: AllocationPathProtocol = ALLOCATION_PATH_PROTOCOL,
    seed_offset: int = 0,
) -> dict[str, Any]:
    values = np.asarray(improvements, dtype=float)
    values = values[np.isfinite(values)]
    lcb, ci_low, ci_high = _paired_bootstrap(
        values,
        alpha=protocol.checkpoint_alpha,
        samples=protocol.bootstrap_samples,
        seed=protocol.random_seed + seed_offset,
    )
    p_value = _one_sided_sign_flip_pvalue(
        values,
        practical_delta=protocol.practical_mae_improvement,
        samples=protocol.sign_flip_samples,
        seed=protocol.random_seed + 1 + seed_offset,
    )
    passed = bool(
        len(values) >= protocol.minimum_confirmation_events
        and lcb > protocol.practical_mae_improvement
        and p_value < protocol.checkpoint_alpha
    )
    return {
        "events": int(len(values)),
        "mean_mae_improvement": float(values.mean()) if len(values) else None,
        "practical_mae_improvement": protocol.practical_mae_improvement,
        "familywise_alpha": protocol.one_sided_alpha,
        "checkpoint_alpha": protocol.checkpoint_alpha,
        "one_sided_checkpoint_lcb": None if np.isnan(lcb) else float(lcb),
        "checkpoint_confidence_level": 1.0 - protocol.checkpoint_alpha,
        "paired_bootstrap_interval": [
            None if np.isnan(ci_low) else float(ci_low),
            None if np.isnan(ci_high) else float(ci_high),
        ],
        "one_sided_sign_flip_p": None if np.isnan(p_value) else float(p_value),
        "passed": passed,
    }


def _validate_settled_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "unit_id",
        "event_id",
        "event_date",
        "team",
        "player",
        "realized_share",
        "open_share",
        *ENDPOINT_FEATURES,
        *PATH_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"settled path features are missing required columns: {missing}")
    clean = frame.copy()
    clean["event_date"] = pd.to_datetime(clean["event_date"], errors="raise").dt.normalize()
    numeric = sorted({"realized_share", "open_share", *ENDPOINT_FEATURES, *PATH_FEATURES})
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    invalid = ~np.isfinite(clean[numeric]).all(axis=1)
    if bool(invalid.any()):
        raise ValueError("settled path features contain non-finite model values")
    return clean.sort_values(["event_date", "unit_id", "player"], kind="mergesort")


def chronological_confirmation(
    settled_player_features: pd.DataFrame,
    *,
    protocol: AllocationPathProtocol = ALLOCATION_PATH_PROTOCOL,
) -> ConfirmationResult:
    """Compare endpoint and endpoint-plus-path models on identical chronological folds."""

    if settled_player_features.empty:
        report = {
            "representation_version": protocol.version,
            "status": "INSUFFICIENT_REAL_PATH_EVENTS",
            "reason": "no valid settled event-team-market units",
            "settled_units": 0,
            "settled_game_events": 0,
            "oof_game_events": 0,
            "statistical_unit": protocol.statistical_unit,
            "decision_checkpoint": None,
            "path_authorized": False,
        }
        return ConfirmationResult(pd.DataFrame(), pd.DataFrame(), report)

    frame = _validate_settled_features(settled_player_features)
    player_prediction_rows: list[pd.DataFrame] = []
    unit_evaluation_rows: list[dict[str, Any]] = []

    for event_date in sorted(frame["event_date"].unique()):
        train = frame.loc[frame["event_date"] < event_date]
        today = frame.loc[frame["event_date"] == event_date]
        train_events = int(train["event_id"].nunique())
        if train_events < protocol.minimum_train_events:
            continue

        endpoint_model = _model()
        path_model = _model()
        endpoint_model.fit(train[ENDPOINT_FEATURES], train["realized_share"])
        path_model.fit(train[PATH_FEATURES], train["realized_share"])

        raw_endpoint = endpoint_model.predict(today[ENDPOINT_FEATURES])
        raw_path = path_model.predict(today[PATH_FEATURES])
        scored_today = today.copy()
        scored_today["endpoint_prediction_raw"] = raw_endpoint
        scored_today["path_prediction_raw"] = raw_path
        scored_parts: list[pd.DataFrame] = []
        for unit_id, unit in scored_today.groupby("unit_id", sort=True):
            unit = unit.copy()
            unit["endpoint_prediction"] = _simplex_projection(unit["endpoint_prediction_raw"].to_numpy())
            unit["path_prediction"] = _simplex_projection(unit["path_prediction_raw"].to_numpy())
            endpoint_mae = float(np.abs(unit["endpoint_prediction"] - unit["realized_share"]).mean())
            path_mae = float(np.abs(unit["path_prediction"] - unit["realized_share"]).mean())
            unit_evaluation_rows.append(
                {
                    "unit_id": unit_id,
                    "event_id": unit["event_id"].iloc[0],
                    "event_date": pd.Timestamp(event_date),
                    "team": unit["team"].iloc[0],
                    "players": int(len(unit)),
                    "training_events": train_events,
                    "endpoint_mae": endpoint_mae,
                    "path_mae": path_mae,
                    "mae_improvement": endpoint_mae - path_mae,
                }
            )
            scored_parts.append(unit)
        if scored_parts:
            player_prediction_rows.append(pd.concat(scored_parts, ignore_index=True))

    predictions = (
        pd.concat(player_prediction_rows, ignore_index=True)
        if player_prediction_rows
        else frame.iloc[0:0].copy()
    )
    unit_evaluations = pd.DataFrame(unit_evaluation_rows)
    if unit_evaluations.empty:
        evaluations = pd.DataFrame()
    else:
        evaluations = (
            unit_evaluations.groupby(["event_id", "event_date"], as_index=False, sort=True)
            .agg(
                teams=("team", "nunique"),
                players=("players", "sum"),
                training_events=("training_events", "min"),
                endpoint_mae=("endpoint_mae", "mean"),
                path_mae=("path_mae", "mean"),
            )
        )
        evaluations["mae_improvement"] = evaluations["endpoint_mae"] - evaluations["path_mae"]
    settled_units = int(frame["unit_id"].nunique())
    settled_events = int(frame["event_id"].nunique())
    oof_events = int(len(evaluations))

    open_mae = float(np.abs(frame["open_share"] - frame["realized_share"]).mean())
    close_mae = float(np.abs(frame["close_share"] - frame["realized_share"]).mean())
    closer = np.abs(frame["close_share"] - frame["realized_share"]) < np.abs(
        frame["open_share"] - frame["realized_share"]
    )

    checkpoints: dict[str, dict[str, Any]] = {}
    for checkpoint in FROZEN_CHECKPOINTS:
        if oof_events >= checkpoint:
            checkpoints[str(checkpoint)] = evaluate_improvement_sequence(
                evaluations["mae_improvement"].iloc[:checkpoint],
                protocol=protocol,
                seed_offset=checkpoint,
            )

    available_checkpoints = [value for value in FROZEN_CHECKPOINTS if oof_events >= value]
    decision_checkpoint = max(available_checkpoints) if available_checkpoints else None
    if decision_checkpoint is None:
        status = "INSUFFICIENT_REAL_PATH_EVENTS"
        reason = (
            f"requires {protocol.minimum_confirmation_events} held-out game events; observed {oof_events}"
        )
        path_authorized = False
    else:
        decision = checkpoints[str(decision_checkpoint)]
        path_authorized = bool(decision["passed"])
        status = (
            "PATH_INCREMENTAL_VALUE_SUPPORTED"
            if path_authorized
            else "PATH_INCREMENTAL_VALUE_NOT_SUPPORTED"
        )
        reason = (
            "practical-effect LCB and sign-flip gates passed"
            if path_authorized
            else "frozen practical-effect confirmation gate did not pass"
        )

    report = {
        "representation_version": protocol.version,
        "status": status,
        "reason": reason,
        "path_authorized": path_authorized,
        "settled_units": settled_units,
        "settled_game_events": settled_events,
        "oof_game_events": oof_events,
        "statistical_unit": protocol.statistical_unit,
        "decision_checkpoint": decision_checkpoint,
        "t1": {
            "open_mae": open_mae,
            "close_mae": close_mae,
            "open_minus_close_mae": open_mae - close_mae,
        },
        "t2": {
            "player_coordinates": int(len(frame)),
            "fraction_close_closer_than_open": float(closer.mean()),
        },
        "t3": {
            "endpoint_features": ENDPOINT_FEATURES,
            "path_features": PATH_FEATURES,
            "model": protocol.endpoint_model,
            "practical_mae_improvement": protocol.practical_mae_improvement,
            "familywise_alpha": protocol.one_sided_alpha,
            "per_checkpoint_alpha": protocol.checkpoint_alpha,
            "frozen_checkpoints": list(FROZEN_CHECKPOINTS),
            "checkpoint_results": checkpoints,
        },
    }
    return ConfirmationResult(predictions, evaluations, report)
