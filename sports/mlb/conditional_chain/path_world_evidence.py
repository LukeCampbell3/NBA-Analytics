from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .outcome_worlds import (
    WorldDistribution,
    WorldPath,
    apply_joint_world_evidence_path,
    update_world_distribution,
)
from .protocol import ALLOCATION_PATH_PROTOCOL


CHECKPOINT_MINUTES = tuple(abs(value) for value in ALLOCATION_PATH_PROTOCOL.checkpoints_minutes)
CHECKPOINT_LABELS = tuple(f"T-{value}" for value in CHECKPOINT_MINUTES)
SHARE_COLUMNS = tuple(f"share_m{value}" for value in CHECKPOINT_MINUTES)
LINE_COLUMNS = tuple(f"line_m{value}" for value in CHECKPOINT_MINUTES)
MINIMUM_EVIDENCE_TRAINING_ROWS = 20
PROBABILITY_EPSILON = 1e-4

ENDPOINT_MODEL_FEATURES = (
    "robust_score",
    "share_current",
    "line_current",
)
PATH_MODEL_FEATURES = ENDPOINT_MODEL_FEATURES + (
    "delta_share",
    "delta_line",
    "share_total_variation",
    "line_total_variation",
    "share_path_efficiency",
    "line_path_efficiency",
    "share_direction_reversals",
    "line_direction_reversals",
    "last_share_step",
    "last_line_step",
)


@dataclass(frozen=True)
class EvidenceModel:
    model: Pipeline | None
    features: tuple[str, ...]
    training_rows: int
    positive_rows: int
    history_end_exclusive: pd.Timestamp | None

    @property
    def fitted(self) -> bool:
        return self.model is not None


@dataclass(frozen=True)
class PathEvidenceBundle:
    endpoint: EvidenceModel
    checkpoints: tuple[EvidenceModel, ...]
    mode: str
    training_rows: int
    history_end_exclusive: pd.Timestamp | None


@dataclass(frozen=True)
class CandidateEvidencePath:
    world_path: WorldPath
    candidate_posteriors: np.ndarray
    cumulative_world_log_evidence: np.ndarray
    incremental_world_log_evidence: np.ndarray
    diagnostics: pd.DataFrame


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "event_date" not in result:
        raise ValueError("frame is missing event_date")
    result["event_date"] = pd.to_datetime(result["event_date"], errors="raise").dt.normalize()
    return result


def _path_join_keys(candidates: pd.DataFrame, paths: pd.DataFrame) -> list[str]:
    # Unlike sports/nba/conditional_chain (a single player_points market), an
    # MLB player can have multiple prop markets live on the same event, so
    # "market" must be part of the join key or two different props for the
    # same player/event would collide onto one path.
    if "event_id" in candidates.columns and "event_id" in paths.columns:
        candidate_event = candidates["event_id"].notna().all()
        path_event = paths["event_id"].notna().all()
        if bool(candidate_event and path_event):
            return ["event_id", "player", "market"]
    return ["event_date", "player", "market"]


def merge_candidates_with_paths(
    candidates: pd.DataFrame,
    path_features: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Join candidates to pre-outcome allocation paths without using settlement fields."""

    candidate_required = {"event_date", "player", "market", "robust_score"}
    path_required = {"event_date", "player", "market", *SHARE_COLUMNS, *LINE_COLUMNS}
    missing_candidates = sorted(candidate_required - set(candidates.columns))
    missing_paths = sorted(path_required - set(path_features.columns))
    if missing_candidates:
        raise ValueError(f"candidates are missing columns: {missing_candidates}")
    if missing_paths:
        raise ValueError(f"path features are missing columns: {missing_paths}")

    candidate_frame = _normalize_dates(candidates)
    path_frame = _normalize_dates(path_features)
    candidate_frame["player"] = candidate_frame["player"].astype(str)
    path_frame["player"] = path_frame["player"].astype(str)
    candidate_frame["market"] = candidate_frame["market"].astype(str)
    path_frame["market"] = path_frame["market"].astype(str)
    keys = _path_join_keys(candidate_frame, path_frame)

    pre_outcome_columns = list(dict.fromkeys([
        *keys,
        "event_date",
        "player",
        "market",
        *SHARE_COLUMNS,
        *LINE_COLUMNS,
    ]))
    safe_paths = path_frame[pre_outcome_columns].copy()
    duplicate = safe_paths.duplicated(keys, keep=False)
    if bool(duplicate.any()):
        duplicated = safe_paths.loc[duplicate, keys].drop_duplicates().head(5)
        raise ValueError(
            "path features are not unique on pre-outcome join keys: "
            f"{duplicated.to_dict(orient='records')}"
        )

    merged = candidate_frame.merge(
        safe_paths,
        on=keys,
        how="left" if require_complete else "inner",
        validate="many_to_one",
        suffixes=("", "_path"),
    )
    if "event_date_path" in merged:
        merged = merged.drop(columns=["event_date_path"])
    if "player_path" in merged:
        merged = merged.drop(columns=["player_path"])

    numeric = ["robust_score", *SHARE_COLUMNS, *LINE_COLUMNS]
    for column in numeric:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    complete = np.isfinite(merged[numeric]).all(axis=1)
    if require_complete and not bool(complete.all()):
        missing_rows = merged.loc[~complete, [column for column in keys if column in merged]].head(5)
        raise ValueError(
            "candidate rows are missing complete frozen path coordinates: "
            f"{missing_rows.to_dict(orient='records')}"
        )
    return merged.loc[complete].copy().reset_index(drop=True)


def _direction_reversals(values: np.ndarray) -> int:
    signs = np.sign(np.diff(values))
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def _path_efficiency(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    total_variation = float(np.abs(np.diff(values)).sum())
    if total_variation <= 0.0:
        return 0.0
    return float(abs(values[-1] - values[0]) / total_variation)


def _inverted_sequence(values: np.ndarray) -> np.ndarray:
    """Reflect interior deviations around the endpoint-preserving straight path."""

    if len(values) <= 2:
        return values.copy()
    checkpoints = np.asarray(CHECKPOINT_MINUTES, dtype=float)
    progress = (checkpoints[0] - checkpoints) / (checkpoints[0] - checkpoints[-1])
    baseline = values[0] + progress * (values[-1] - values[0])
    reflected = 2.0 * baseline - values
    reflected[0] = values[0]
    reflected[-1] = values[-1]
    return reflected


def transform_path_arrays(
    shares: np.ndarray,
    lines: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    shares = np.asarray(shares, dtype=float).copy()
    lines = np.asarray(lines, dtype=float).copy()
    if shares.shape != lines.shape or shares.ndim != 2 or shares.shape[1] != len(CHECKPOINT_MINUTES):
        raise ValueError("path arrays must be rows by frozen checkpoints")
    if mode == "real":
        pass
    elif mode == "shuffled":
        # Preserve T-240 and T-5 endpoints while destroying chronological interior order.
        permutation = np.asarray([0, 3, 2, 1, 4], dtype=int)
        shares = shares[:, permutation]
        lines = lines[:, permutation]
    elif mode == "inverted":
        shares = np.vstack([_inverted_sequence(row) for row in shares])
        lines = np.vstack([_inverted_sequence(row) for row in lines])
        shares = np.clip(shares, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        lines = np.clip(lines, PROBABILITY_EPSILON, None)
    else:
        raise ValueError("mode must be one of: real, shuffled, inverted")
    return shares, lines


def prefix_features(
    merged: pd.DataFrame,
    checkpoint_index: int,
    *,
    mode: str = "real",
) -> pd.DataFrame:
    if checkpoint_index < 0 or checkpoint_index >= len(CHECKPOINT_MINUTES):
        raise ValueError("checkpoint_index is outside the frozen path")
    shares, lines = transform_path_arrays(
        merged.loc[:, SHARE_COLUMNS].to_numpy(dtype=float),
        merged.loc[:, LINE_COLUMNS].to_numpy(dtype=float),
        mode=mode,
    )
    shares = shares[:, : checkpoint_index + 1]
    lines = lines[:, : checkpoint_index + 1]
    rows: list[dict[str, float]] = []
    for row_index in range(len(merged)):
        share_values = shares[row_index]
        line_values = lines[row_index]
        share_tv = float(np.abs(np.diff(share_values)).sum()) if len(share_values) > 1 else 0.0
        line_tv = float(np.abs(np.diff(line_values)).sum()) if len(line_values) > 1 else 0.0
        rows.append(
            {
                "robust_score": float(merged["robust_score"].iloc[row_index]),
                "share_current": float(share_values[-1]),
                "line_current": float(line_values[-1]),
                "delta_share": float(share_values[-1] - share_values[0]),
                "delta_line": float(line_values[-1] - line_values[0]),
                "share_total_variation": share_tv,
                "line_total_variation": line_tv,
                "share_path_efficiency": _path_efficiency(share_values),
                "line_path_efficiency": _path_efficiency(line_values),
                "share_direction_reversals": float(_direction_reversals(share_values)),
                "line_direction_reversals": float(_direction_reversals(line_values)),
                "last_share_step": float(share_values[-1] - share_values[-2]) if len(share_values) > 1 else 0.0,
                "last_line_step": float(line_values[-1] - line_values[-2]) if len(line_values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows, index=merged.index)


def endpoint_features(merged: pd.DataFrame) -> pd.DataFrame:
    return prefix_features(merged, len(CHECKPOINT_MINUTES) - 1, mode="real").loc[
        :, ENDPOINT_MODEL_FEATURES
    ]


def _fit_model(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    feature_names: tuple[str, ...],
    history_end_exclusive: pd.Timestamp | None,
    minimum_rows: int,
) -> EvidenceModel:
    clean = features.loc[:, feature_names].copy()
    for column in feature_names:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    labels = pd.to_numeric(labels, errors="coerce")
    valid = np.isfinite(clean.to_numpy(dtype=float)).all(axis=1) & labels.isin([0, 1])
    clean = clean.loc[valid]
    target = labels.loc[valid].astype(int)
    positive_rows = int(target.sum())
    if len(clean) < minimum_rows or target.nunique() < 2:
        return EvidenceModel(
            model=None,
            features=feature_names,
            training_rows=int(len(clean)),
            positive_rows=positive_rows,
            history_end_exclusive=history_end_exclusive,
        )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=500,
                    random_state=20260820,
                ),
            ),
        ]
    )
    model.fit(clean, target)
    return EvidenceModel(
        model=model,
        features=feature_names,
        training_rows=int(len(clean)),
        positive_rows=positive_rows,
        history_end_exclusive=history_end_exclusive,
    )


def fit_path_evidence_bundle(
    history_candidates: pd.DataFrame,
    history_paths: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
    mode: str = "real",
    minimum_rows: int = MINIMUM_EVIDENCE_TRAINING_ROWS,
) -> PathEvidenceBundle:
    cutoff = pd.Timestamp(as_of_date).normalize()
    candidates = _normalize_dates(history_candidates)
    paths = _normalize_dates(history_paths)
    candidates = candidates.loc[candidates["event_date"] < cutoff].copy()
    paths = paths.loc[paths["event_date"] < cutoff].copy()
    if "leg_result" not in candidates:
        raise ValueError("history candidates are missing leg_result")
    candidates["binary_leg_win"] = pd.to_numeric(candidates["leg_result"], errors="coerce").eq(1.0).astype(int)
    merged = merge_candidates_with_paths(candidates, paths, require_complete=False)
    history_end = cutoff

    endpoint = _fit_model(
        endpoint_features(merged) if len(merged) else pd.DataFrame(columns=ENDPOINT_MODEL_FEATURES),
        merged["binary_leg_win"] if len(merged) else pd.Series(dtype=float),
        feature_names=ENDPOINT_MODEL_FEATURES,
        history_end_exclusive=history_end,
        minimum_rows=minimum_rows,
    )
    checkpoint_models: list[EvidenceModel] = []
    for checkpoint_index in range(len(CHECKPOINT_MINUTES)):
        features = (
            prefix_features(merged, checkpoint_index, mode=mode)
            if len(merged)
            else pd.DataFrame(columns=PATH_MODEL_FEATURES)
        )
        checkpoint_models.append(
            _fit_model(
                features,
                merged["binary_leg_win"] if len(merged) else pd.Series(dtype=float),
                feature_names=PATH_MODEL_FEATURES,
                history_end_exclusive=history_end,
                minimum_rows=minimum_rows,
            )
        )
    return PathEvidenceBundle(
        endpoint=endpoint,
        checkpoints=tuple(checkpoint_models),
        mode=mode,
        training_rows=int(len(merged)),
        history_end_exclusive=history_end,
    )


def _predict(
    fitted: EvidenceModel,
    features: pd.DataFrame,
    fallback_probabilities: Iterable[float],
) -> np.ndarray:
    fallback = np.asarray(list(fallback_probabilities), dtype=float)
    if len(fallback) != len(features):
        raise ValueError("fallback probabilities must align with feature rows")
    if fitted.model is None:
        return np.clip(fallback, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    values = fitted.model.predict_proba(features.loc[:, fitted.features])[:, 1]
    return np.clip(values, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def endpoint_posteriors(
    bundle: PathEvidenceBundle,
    current_candidates_with_paths: pd.DataFrame,
    fallback_probabilities: Iterable[float],
) -> np.ndarray:
    return _predict(bundle.endpoint, endpoint_features(current_candidates_with_paths), fallback_probabilities)


def path_posteriors(
    bundle: PathEvidenceBundle,
    current_candidates_with_paths: pd.DataFrame,
    fallback_probabilities: Iterable[float],
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for checkpoint_index, fitted in enumerate(bundle.checkpoints):
        features = prefix_features(
            current_candidates_with_paths,
            checkpoint_index,
            mode=bundle.mode,
        )
        probabilities.append(_predict(fitted, features, fallback_probabilities))
    return np.vstack(probabilities)


def candidate_posteriors_to_world_log_evidence(
    prior: WorldDistribution,
    posterior_probabilities: Iterable[float],
) -> np.ndarray:
    """Map calibrated candidate posteriors to likelihood evidence over joint worlds.

    This is deliberately candidate-conditioned evidence. It is not a claim of a
    learned non-factorized interaction model. The update acts on the exact joint
    world distribution rather than reranking candidates.
    """

    posterior = np.asarray(list(posterior_probabilities), dtype=float)
    if posterior.shape != (prior.candidate_count,):
        raise ValueError("posterior probabilities must align with candidate order")
    if not np.isfinite(posterior).all():
        raise ValueError("posterior probabilities must be finite")
    base = np.clip(prior.marginals, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    posterior = np.clip(posterior, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    log_win = np.log(posterior / base)
    log_loss = np.log((1.0 - posterior) / (1.0 - base))
    return prior.outcomes @ log_win + (1 - prior.outcomes) @ log_loss


def build_candidate_evidence_path(
    prior: WorldDistribution,
    candidate_posteriors: np.ndarray,
    *,
    checkpoint_labels: Iterable[str] = CHECKPOINT_LABELS,
) -> CandidateEvidencePath:
    posterior = np.asarray(candidate_posteriors, dtype=float)
    if posterior.ndim != 2 or posterior.shape[1] != prior.candidate_count:
        raise ValueError("candidate_posteriors must be checkpoints by candidates")
    labels = tuple(str(value) for value in checkpoint_labels)
    if len(labels) != len(posterior):
        raise ValueError("checkpoint labels must align with candidate posteriors")

    cumulative = np.vstack(
        [candidate_posteriors_to_world_log_evidence(prior, row) for row in posterior]
    )
    incremental = cumulative.copy()
    if len(incremental) > 1:
        incremental[1:] = cumulative[1:] - cumulative[:-1]
    world_path = apply_joint_world_evidence_path(
        prior,
        incremental,
        checkpoint_labels=labels,
    )
    diagnostic_rows: list[dict[str, Any]] = []
    for checkpoint_index, label in enumerate(labels):
        for candidate_index, candidate_id in enumerate(prior.candidate_ids):
            diagnostic_rows.append(
                {
                    "checkpoint": label,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "prior_probability": float(prior.marginals[candidate_index]),
                    "posterior_probability": float(posterior[checkpoint_index, candidate_index]),
                    "model_delta_probability": float(
                        posterior[checkpoint_index, candidate_index] - prior.marginals[candidate_index]
                    ),
                }
            )
    return CandidateEvidencePath(
        world_path=world_path,
        candidate_posteriors=posterior,
        cumulative_world_log_evidence=cumulative,
        incremental_world_log_evidence=incremental,
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def direct_final_distribution(
    prior: WorldDistribution,
    candidate_posteriors: Iterable[float],
) -> WorldDistribution:
    """Reference one-step update used to verify incremental path evidence."""

    evidence = candidate_posteriors_to_world_log_evidence(prior, candidate_posteriors)
    return update_world_distribution(prior, evidence)
