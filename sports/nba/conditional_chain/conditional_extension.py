from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXTENSION_FEATURES = [
    "candidate_robust_score",
    "candidate_selected_probability",
    "candidate_path_support",
    "prefix_min_robust_score",
    "prefix_mean_path_support",
    "prefix_length",
    "allocation_coherence",
]


@dataclass
class ConditionalExtensionModel:
    minimum_training_rows: int = 30
    _pipeline: Pipeline | None = None
    training_rows: int = 0

    @property
    def fitted(self) -> bool:
        return self._pipeline is not None

    def fit(self, extension_rows: pd.DataFrame) -> "ConditionalExtensionModel":
        required = {*EXTENSION_FEATURES, "extension_hit", "prefix_survived"}
        missing = sorted(required - set(extension_rows.columns))
        if missing:
            raise ValueError(f"extension training rows are missing columns: {missing}")
        training = extension_rows.loc[extension_rows["prefix_survived"].eq(1)].copy()
        training["extension_hit"] = pd.to_numeric(training["extension_hit"], errors="coerce")
        training = training.dropna(subset=[*EXTENSION_FEATURES, "extension_hit"])
        if len(training) < self.minimum_training_rows:
            raise ValueError(
                f"requires {self.minimum_training_rows} prefix-surviving rows; observed {len(training)}"
            )
        if training["extension_hit"].nunique() < 2:
            raise ValueError("extension training requires both hit and miss outcomes")
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(C=1.0, max_iter=5_000, random_state=20260820),
                ),
            ]
        )
        self._pipeline.fit(training[EXTENSION_FEATURES], training["extension_hit"].astype(int))
        self.training_rows = int(len(training))
        return self

    def predict_survival(self, rows: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("conditional extension model is not fitted")
        missing = sorted(set(EXTENSION_FEATURES) - set(rows.columns))
        if missing:
            raise ValueError(f"extension scoring rows are missing columns: {missing}")
        return self._pipeline.predict_proba(rows[EXTENSION_FEATURES])[:, 1]


def extension_feature_row(prefix: pd.DataFrame, candidate: pd.Series) -> dict[str, float]:
    candidate_requirement = 1.0 if str(candidate["side"]).upper() == "OVER" else -1.0
    coherence: list[float] = []
    for _, prior in prefix.iterrows():
        if str(prior["team"]) != str(candidate["team"]):
            continue
        if str(prior["market"]) != str(candidate["market"]):
            continue
        prior_requirement = 1.0 if str(prior["side"]).upper() == "OVER" else -1.0
        observed_relation = np.sign(float(prior["delta_share"]) * float(candidate["delta_share"]))
        required_relation = prior_requirement * candidate_requirement
        coherence.append(float(observed_relation * required_relation))
    return {
        "candidate_robust_score": float(candidate["robust_score"]),
        "candidate_selected_probability": float(candidate["selected_probability"]),
        "candidate_path_support": float(candidate["path_support"]),
        "prefix_min_robust_score": float(prefix["robust_score"].min()),
        "prefix_mean_path_support": float(prefix["path_support"].mean()),
        "prefix_length": float(len(prefix)),
        "allocation_coherence": float(np.mean(coherence)) if coherence else 0.0,
    }


def build_extension_training_ledger(selected_decisions: pd.DataFrame) -> pd.DataFrame:
    """Create conditional rows from one historically selected chain per slate."""

    required = {
        "slate_date",
        "decision_id",
        "leg_order",
        "hit",
        "event_id",
        "team",
        "player",
        "market",
        "side",
        "robust_score",
        "selected_probability",
        "delta_share",
        "path_support",
    }
    missing = sorted(required - set(selected_decisions.columns))
    if missing:
        raise ValueError(f"selected decisions are missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for (slate_date, decision_id), decision in selected_decisions.groupby(
        ["slate_date", "decision_id"], sort=True
    ):
        decision = decision.sort_values("leg_order", kind="mergesort").reset_index(drop=True)
        if len(decision) != 4 or decision["leg_order"].tolist() != [1, 2, 3, 4]:
            raise ValueError(f"decision {decision_id} must contain ordered legs 1..4")
        for index in range(1, 4):
            prefix = decision.iloc[:index]
            candidate = decision.iloc[index]
            prefix_survived = bool(pd.to_numeric(prefix["hit"], errors="coerce").eq(1.0).all())
            row = extension_feature_row(prefix, candidate)
            row.update(
                {
                    "slate_date": slate_date,
                    "decision_id": decision_id,
                    "candidate_leg_order": index + 1,
                    "prefix_survived": int(prefix_survived),
                    "extension_hit": int(float(candidate["hit"]) == 1.0),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
